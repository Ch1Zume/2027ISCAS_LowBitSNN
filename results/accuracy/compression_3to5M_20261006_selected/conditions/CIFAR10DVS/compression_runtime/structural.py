"""Dense structural pruning with TRAIN covariance reconstruction and rule preservation.

Both physical EI banks count towards the budget. No pruning masks, sparse
parameter accounting, signed physical weights, or extra inference model are used.
"""

import torch
from torch import nn


def paired_convolutions(model):
    return [(name, module) for name, module in model.backbone.named_modules()
            if hasattr(module, "excitatory") and hasattr(module, "inhibitory")]


def parameter_counts(model):
    groups = {"backbone": model.backbone, "hidden": model.hidden_layers,
              "classifier": model.classifier}
    result = {name: sum(p.numel() for p in module.parameters()) for name, module in groups.items()}
    result["total"] = sum(p.numel() for p in model.parameters())
    result["trainable_by_bp"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert result["total"] == sum(result[name] for name in groups)
    return result


@torch.no_grad()
def assert_physical_weights(model):
    names = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)) or (
                hasattr(module, "thresholds") and hasattr(module, "weight")):
            if module.weight.dtype != torch.float32 or not torch.isfinite(module.weight).all():
                raise AssertionError(f"Nonfinite or non-FP32 physical weight: {name}")
            if module.weight.min() < 0:
                raise AssertionError(f"Negative physical weight: {name}")
            names.append(name)
    return names


@torch.no_grad()
def select_channels(covariance, count, importance=None):
    """Weighted pivoted Cholesky chooses active, nonredundant channel features."""
    size = covariance.shape[0]
    if not 0 < count <= size or covariance.shape != (size, size):
        raise ValueError("Invalid channel covariance or target width")
    if count == size:
        return torch.arange(size, device=covariance.device)
    if importance is None:
        importance = torch.ones(size, device=covariance.device)
    weight = (importance.clamp_min(0) / importance.mean().clamp_min(1e-12) + 0.1).sqrt()
    gram = covariance * weight[:, None] * weight[None, :]
    residual = gram.diag().clone().clamp_min(0)
    factor = torch.zeros(size, count, device=gram.device, dtype=gram.dtype)
    chosen = torch.zeros(size, device=gram.device, dtype=torch.bool)
    indices = []
    for index in range(count):
        pivot = residual.masked_fill(chosen, -torch.inf).argmax()
        column = gram[:, pivot].clone()
        if index:
            column -= factor[:, :index] @ factor[pivot, :index]
        column /= residual[pivot].clamp_min(1e-12).sqrt()
        factor[:, index] = column
        residual = (residual - column.square()).clamp_min(0)
        chosen[pivot] = True
        indices.append(pivot)
    return torch.stack(indices).sort().values


@torch.no_grad()
def reconstruction_map(covariance, kept, ridge=1e-3):
    """Fit all teacher channels from retained ones without changing feature count."""
    if len(kept) == covariance.shape[0]:
        return torch.eye(len(kept), device=covariance.device, dtype=covariance.dtype)
    gram = covariance[kept][:, kept]
    regularizer = ridge * gram.diag().mean().clamp_min(1e-8)
    gram = gram + torch.eye(len(kept), device=gram.device) * regularizer
    return torch.linalg.solve(gram, covariance[kept])


@torch.no_grad()
def hidden_indices(source, target, scores):
    """Keep complete hybrid competition groups and their BP/STDP assignment."""
    device = scores.device
    if source.rule != target.rule:
        raise ValueError("Compression must preserve hidden learning rules")
    if source.rule != "hybrid":
        return scores.argsort(descending=True, stable=True)[:target.out_features].sort().values
    group_size = source.competition_size
    if target.competition_size != group_size:
        raise ValueError("Hybrid competition size must be preserved")
    group_scores = scores.reshape(-1, group_size).sum(1)
    source_mask = source.stdp_neuron_mask.reshape(-1, group_size)
    target_mask = target.stdp_neuron_mask.reshape(-1, group_size)
    remaining = torch.ones(len(source_mask), device=device, dtype=torch.bool)
    selected = []
    for mask in target_mask:
        compatible = (source_mask == mask).all(1) & remaining
        if not compatible.any():
            raise ValueError("Insufficient source groups with the required BP/STDP pattern")
        group = group_scores.masked_fill(~compatible, -torch.inf).argmax()
        remaining[group] = False
        selected.append(group * group_size + torch.arange(group_size, device=device))
    indices = torch.cat(selected)
    if not torch.equal(source.stdp_neuron_mask[indices], target.stdp_neuron_mask):
        raise AssertionError("Pruning changed BP/STDP assignments")
    return indices


class TrainingStatistics:
    """Read-only activation statistics; the caller supplies only TRAIN batches."""
    def __init__(self, model, max_points=2048):
        self.model, self.max_points = model, max_points
        self.covariances, self.hidden_outputs = {}, {}
        self.handles = []
        for name, module in paired_convolutions(model):
            self.handles.append(module.register_forward_pre_hook(self._input_hook(name)))
        self.handles.append(model.pool.register_forward_hook(self._pool_hook))
        for name, module in model.hidden_layers.items():
            self.handles.append(module.register_forward_hook(self._hidden_hook(name)))

    @torch.no_grad()
    def _accumulate(self, name, features):
        features = features.detach().movedim(2, -1).reshape(-1, features.shape[2])
        stride = max(1, (len(features) + self.max_points - 1) // self.max_points)
        features = features[::stride][:self.max_points].float()
        gram = features.T @ features / max(len(features), 1)
        if name not in self.covariances:
            self.covariances[name] = [gram, 1]
        else:
            self.covariances[name][0].add_(gram)
            self.covariances[name][1] += 1

    def _input_hook(self, name):
        def hook(module, inputs):
            self._accumulate(name, inputs[0])
        return hook

    def _pool_hook(self, module, inputs, output):
        self._accumulate("pool", output)

    def _hidden_hook(self, name):
        def hook(module, inputs, output):
            self.hidden_outputs.setdefault(name, []).append(output.detach().mean(0).cpu())
        return hook

    def finish(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return ({name: value / count for name, (value, count) in self.covariances.items()},
                {name: torch.cat(rows) for name, rows in self.hidden_outputs.items()})


@torch.no_grad()
def prune_from_teacher(teacher, student, covariances, activities, labels):
    source = paired_convolutions(teacher)
    target = paired_convolutions(student)
    if len(source) != len(target):
        raise ValueError("Only width compression of the same convolution topology is supported")
    device = next(teacher.parameters()).device
    kept_channels = []
    for index, ((name, module), (_, compact)) in enumerate(zip(source, target)):
        if index + 1 < len(source):
            next_name, next_module = source[index + 1]
            covariance = covariances[next_name]
            importance = (next_module.excitatory.weight - next_module.inhibitory.weight).abs().mean((0, 2, 3))
        else:
            covariance = covariances["pool"]
            hidden = next(iter(teacher.hidden_layers.values()))
            importance = hidden.weight.reshape(hidden.out_features, module.excitatory.out_channels, -1).mean((0, 2))
        kept_channels.append(select_channels(covariance, compact.excitatory.out_channels, importance))

    source_state, target_state = teacher.state_dict(), student.state_dict()
    for name, value in target_state.items():
        if name in source_state and source_state[name].shape == value.shape:
            value.copy_(source_state[name])
    for index, ((name, module), (compact_name, compact)) in enumerate(zip(source, target)):
        if name != compact_name:
            raise ValueError("Source and student block paths differ")
        kept = kept_channels[index]
        inputs = (torch.arange(module.excitatory.in_channels, device=device) if index == 0
                  else kept_channels[index - 1])
        mapping = reconstruction_map(covariances[name], inputs)
        effective = (module.excitatory.weight - module.inhibitory.weight)[kept]
        reconstructed = torch.einsum("oihw,ki->okhw", effective, mapping)
        target_state[f"backbone.{name}.excitatory.weight"].copy_(reconstructed.clamp_min(0))
        target_state[f"backbone.{name}.inhibitory.weight"].copy_((-reconstructed).clamp_min(0))
        # The sequential block immediately following a paired convolution is BN.
        normalization = str(int(name) + 1)
        for suffix in ("weight", "bias", "running_mean", "running_var"):
            key = f"backbone.{normalization}.{suffix}"
            target_state[key].copy_(source_state[key][kept])

    # Rank hidden neurons by class-dependent activity and downstream discrimination.
    downstream = teacher.classifier.weight - teacher.classifier.weight.mean(0, keepdim=True)
    selections = {}
    hidden_sources = list(teacher.hidden_layers.items())
    for name, hidden in reversed(hidden_sources):
        activity = activities[name].to(device)
        label_device = labels.to(device)
        class_rates = torch.stack([activity[label_device == label].mean(0)
                                   for label in label_device.unique()])
        separation = class_rates.std(0, unbiased=False) + activity.std(0, unbiased=False) * 0.1
        scores = separation * (downstream.square().mean(0).sqrt() + 1e-8)
        selections[name] = hidden_indices(hidden, student.hidden_layers[name], scores)
        downstream = downstream @ hidden.weight

    spatial = next(iter(teacher.hidden_layers.values())).in_features // teacher.cfg["custom_channels"][-1]
    incoming = kept_channels[-1]
    incoming = (incoming[:, None] * spatial + torch.arange(spatial, device=device)).flatten()
    for name, hidden in hidden_sources:
        kept = selections[name]
        compact = student.hidden_layers[name]
        weights = hidden.weight[kept][:, incoming].clone()
        # Restore the original positive input mass after removing presynaptic neurons.
        scales = hidden.weight[kept].sum(1) / weights.sum(1).clamp_min(1e-12)
        weights.mul_(scales[:, None]).clamp_(0, hidden.weight_max)
        target_state[f"hidden_layers.{name}.weight"].copy_(weights)
        for suffix in ("bias", "thresholds.offset", "thresholds.rate"):
            key = f"hidden_layers.{name}.{suffix}"
            target_state[key].copy_(source_state[key][kept])
        target_state[f"hidden_layers.{name}.row_sum_target"].copy_(weights.sum(1))
        incoming = kept
    target_state["classifier.weight"].copy_(teacher.classifier.weight[:, incoming]
                                             * teacher.classifier.in_features / student.classifier.in_features)
    target_state["classifier.bias"].copy_(teacher.classifier.bias)
    student.load_state_dict(target_state, strict=True)
    student.project_nonnegative()
    assert_physical_weights(student)
    return {"method": "dense_structured_channel_neuron_pruning_train_covariance_reconstruction",
            "channel_indices": [values.cpu().tolist() for values in kept_channels],
            "hidden_indices": {name: values.cpu().tolist() for name, values in selections.items()},
            "hybrid_rules_preserved": True,
            "hidden_transfer": "preserve selected rows and their positive input mass",
            "convolution_transfer": "TRAIN ridge feature reconstruction; canonical nonnegative EI banks"}
