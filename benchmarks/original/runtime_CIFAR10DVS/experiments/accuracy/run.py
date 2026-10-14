"""Run float learning once, then compare identical test inputs at FP/2/4 bits."""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from snn.inference_quant import quantized_hidden_weights
from snn.wta import WinnerOptimizerGuard
from .config import DATASETS, PROJECT_ROOT, load_config, epoch_budget, dataset_spec
from .data import PROBE_SAMPLING, build_datasets, build_loaders, build_probe_loader, prepare_batch, augment_training_frames
from .models import AccuracySNN, SOURCE_URLS
from .distillation import OwnTeacherBundle
from .monitoring import TrainingAborted, TrainingMonitor, monitoring_config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for checks or a working GPU for full training")
    return torch.device(name)


def rng_state():
    return (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    torch.set_rng_state(state[0].cpu())
    if state[1]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state[1]])


def make_lr_scheduler(optimizer, cfg, epochs):
    if cfg.get("lr_schedule", "cosine") == "plateau":
        settings = cfg.get("lr_plateau", {})
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=settings.get("factor", 0.5),
            patience=settings.get("patience", 5), min_lr=settings.get("min_lr", 1e-5),
            threshold=settings.get("threshold", cfg["monitoring"]["min_accuracy_improvement"]),
            threshold_mode="abs")
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


def step_lr_scheduler(scheduler, validation_accuracy):
    before = [group["lr"] for group in scheduler.optimizer.param_groups]
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(validation_accuracy)
    else:
        scheduler.step()
    after = [group["lr"] for group in scheduler.optimizer.param_groups]
    return before, after


def validate_resume_budget(checkpoint, cfg, epochs, allow_extension=False):
    """Only metric-driven plateau schedules can extend an existing budget."""
    if checkpoint is None:
        if allow_extension:
            raise ValueError("extend_plateau_budget requires an own-run resume checkpoint")
        return None
    original = checkpoint["metadata"].get("training_epochs", epochs)
    if allow_extension:
        if (epochs <= original or cfg.get("lr_schedule", "cosine") != "plateau"
                or checkpoint["metadata"]["config"].get("lr_schedule", "cosine") != "plateau"
                or checkpoint["scheduler"].get("mode") != "max"
                or "T_max" in checkpoint["scheduler"]):
            raise ValueError("Only an increasing plateau budget may be extended; cosine schedules must keep their original budget")
        return {"original_epoch_budget": original, "extended_epoch_budget": epochs,
                "policy": "retain metric-driven LR state and all own-run training state"}
    if original != epochs:
        raise ValueError("Resume with the original total epoch budget to preserve the LR schedule")
    return checkpoint["metadata"].get("schedule_budget_extension")


def validate_own_custom_phase(checkpoint, args, cfg):
    """Allow an explicit regularization phase of this dataset's own custom run.

    Event coding, split, and synaptic constraints remain identical. An explicit
    first-hidden STDP-to-hybrid phase can retain the trained weights while
    assigning disjoint neuron populations to BP and STDP.
    """
    meta = checkpoint["metadata"]
    origin = meta.get("initialization", {})
    if (args.model != "custom" or meta["model"] != "custom"
            or not (origin.get("kind") == "from_scratch"
                    or origin.get("kind") == "own_run_adaptation" and origin.get("from_scratch_origin") is True)):
        raise ValueError("Own initialization requires a custom checkpoint originating from scratch")
    previous = meta["config"]
    for name in set(previous.get("dataset_profiles", {})) | set(cfg.get("dataset_profiles", {})):
        old = {k: v for k, v in previous.get("dataset_profiles", {}).get(name, {}).items() if k not in {"lr", "epochs"}}
        new = {k: v for k, v in cfg.get("dataset_profiles", {}).get(name, {}).items() if k not in {"lr", "epochs"}}
        if old != new:
            native_change = (cfg.get("own_native_resolution_adaptation", False)
                             and name == args.dataset == "CIFAR10DVS"
                             and old.get("resolution") == 64 and new.get("resolution") == 128
                             and {k:v for k,v in old.items() if k != "resolution"}
                                 == {k:v for k,v in new.items() if k != "resolution"})
            activity_keys = {"event_transform", "event_mass_reference", "event_gain_limits"}
            activity_change = (cfg.get("own_event_activity_adaptation", False)
                               and name == args.dataset == "DVSGesture"
                               and old.get("event_transform") == "log1p"
                               and new.get("event_transform") == "mass_normalized_log1p"
                               and {k:v for k,v in old.items() if k not in activity_keys}
                                   == {k:v for k,v in new.items() if k not in activity_keys})
            if not (native_change or activity_change):
                raise ValueError("Own phase must preserve input coding, resolution, time steps, and pooling")
    if cfg.get("own_native_resolution_adaptation", False) and (
            previous.get("dataset_profiles", {}).get(args.dataset, {}).get("resolution") != 64
            or cfg.get("dataset_profiles", {}).get(args.dataset, {}).get("resolution") != 128):
        raise ValueError("Own native phase only permits CIFAR64-to128 resolution adaptation")
    if cfg.get("own_event_activity_adaptation", False) and (
            args.dataset != "DVSGesture" or previous.get("dataset_profiles", {}).get(args.dataset, {}).get("event_transform") != "log1p"):
        raise ValueError("Own activity adaptation only permits Gesture log1p to normalized log1p")
    mutable = {"out_dir", "training_augmentation", "hidden_dropout", "lr", "model_learning_rates",
               "optimizer", "weight_decay", "training_epochs", "lr_schedule", "lr_plateau",
               "bn_refit_batches", "monitoring", "dataset_profiles",
               "hidden_competition_size", "hidden_winner_tiebreak", "hidden_winner_scope", "own_phase_restore_rng",
               "bp_row_normalize_layers", "own_distillation", "own_native_resolution_adaptation", "classifier_projection", "own_event_activity_adaptation", "temporal_training_loss"}
    if cfg.get("own_first_hidden_hybrid_adaptation", False):
        if previous.get("learning_rules") != ["stdp", "bp"] or cfg["learning_rules"] != ["hybrid", "bp"]:
            raise ValueError("Own learning adaptation only permits first STDP to hybrid; second stays BP")
        # Masks are derived from the new rules and are not checkpoint buffers.
        # We keep weights, row-sum targets, and threshold state exactly intact.
        mutable |= {"learning_rules", "hybrid_stdp_period", "own_first_hidden_hybrid_adaptation"}
    normalized = copy.deepcopy(meta)
    if cfg.get("own_native_resolution_adaptation", False):
        normalized["dataset"]["resolution"] = 128
    temporal_inflation = previous.get("backbone_variant") == "balanced_ei" and cfg.get("backbone_variant") == "balanced_temporal_ei"
    if temporal_inflation:
        # This is an explicit architecture phase, never an ordinary resume.
        mutable = mutable | {"backbone_variant", "custom_temporal_layers"}
    if cfg.get("own_time_bn_adaptation", False):
        if (previous.get("custom_batch_norm", "shared") != "shared"
                or cfg.get("custom_batch_norm") != "time"
                or previous.get("backbone_variant") != "balanced_ei"):
            raise ValueError("Own time-BN phase only permits shared-to-time spatial EI normalization")
        mutable |= {"custom_batch_norm", "own_time_bn_adaptation"}
    for key in mutable:
        if key in cfg:
            normalized["config"][key] = copy.deepcopy(cfg[key])
        else:
            normalized["config"].pop(key, None)
    identity_args = copy.copy(args)
    identity_args.resume = args.own_init_checkpoint
    check_run_identity(normalized, identity_args, cfg)
    for key in ("hidden_initial_row_sum", "data_dir", "init", "weight_bits"):
        if previous.get(key) != cfg.get(key):
            raise ValueError(f"Own phase must preserve {key}")
    return {key: {"previous": previous.get(key), "new": cfg.get(key)}
            for key in sorted(mutable - {"out_dir"}) if previous.get(key) != cfg.get(key)}


def initialize_own_custom_phase(model, checkpoint, args, cfg):
    changes = validate_own_custom_phase(checkpoint, args, cfg)
    state = dict(checkpoint["model"])
    expanded_bn = []
    if cfg.get("own_time_bn_adaptation", False):
        from .nonnegative_backbones import TemporalBatchNorm2d
        for name, module in model.backbone.named_modules():
            if isinstance(module, TemporalBatchNorm2d):
                prefix = f"backbone.{name}."
                for key in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                    value = state.pop(prefix + key)
                    for t in range(module.steps):
                        state[f"{prefix}banks.{t}.{key}"] = value.clone()
                expanded_bn.append({"layer": f"backbone.{name}", "steps": module.steps,
                                    "initial_affine_and_moments": "exact clones of own shared BN"})
    inflated = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)) or hasattr(module, "eligible_rows"):
            weight = state.get(f"{name}.weight")
            if weight is None or weight.dtype != torch.float32 or not torch.isfinite(weight).all() or (weight < 0).any():
                raise ValueError("Own checkpoint must contain finite FP32 nonnegative synapses")
            if isinstance(module, torch.nn.Conv3d) and weight.ndim == 4:
                if checkpoint["metadata"]["config"].get("backbone_variant") != "balanced_ei":
                    raise ValueError("Temporal inflation requires own spatial EI predecessor")
                expected = module.weight.shape[:2] + module.weight.shape[3:]
                if tuple(weight.shape) != tuple(expected):
                    raise ValueError("Temporal inflation must preserve convolution geometry")
                expanded = torch.zeros_like(module.weight, device="cpu")
                expanded[:, :, -1].copy_(weight)  # Current-frame tap; history initially zero.
                state[f"{name}.weight"] = expanded
                inflated.append({"weight": f"{name}.weight", "previous_shape": list(weight.shape),
                                 "new_shape": list(expanded.shape), "current_frame_tap": 2,
                                 "history_taps_initially_zero": True})
    model.load_state_dict(state)
    model.initialization = {"kind": "own_run_adaptation", "from_scratch_origin": True,
                            "source_checkpoint": str(Path(args.own_init_checkpoint).resolve()),
                            "source_epoch": checkpoint["epoch"], "changes": changes,
                            "optimizer_and_guard": "fresh explicitly registered training phase",
                            "thresholds_and_weights": "retained from own checkpoint"}
    if inflated:
        model.initialization["temporal_inflation"] = inflated
        model.initialization["thresholds_and_weights"] = "own parameters retained; spatial kernels placed at current-frame taps; history kernels zero"
    if expanded_bn:
        model.initialization["time_normalization_expansion"] = expanded_bn
    if cfg.get("own_first_hidden_hybrid_adaptation", False):
        first = model.hidden_layers["hidden1"]
        model.initialization["learning_rule_adaptation"] = {
            "layer": "hidden1", "previous": "stdp", "new": "hybrid",
            "bp_neurons": int(first.bp_neuron_mask.sum()),
            "stdp_neurons": int(first.stdp_neuron_mask.sum()),
            "stdp_assignment": f"indices divisible by {first.hybrid_stdp_period}",
            "weights_thresholds_row_targets": "retained exactly; no recalibration",
            "wta_eligibility": "required for both BP and STDP writes",
        }
    if cfg.get("own_native_resolution_adaptation", False):
        model.initialization["input_resolution_adaptation"] = {
            "previous": 64, "new": 128, "temporal_coding_and_pooling": "retained",
            "parameters_thresholds_row_targets": "retained exactly",
            "channel_and_linear_width_budget": "unchanged",
            "spatially_unfolded_convolution_neurons_multiplier": 4,
            "normalization": "TRAIN-only BN refit before initial probe and every VALID"}
    if cfg.get("own_event_activity_adaptation", False):
        profile = cfg["dataset_profiles"][args.dataset]
        model.initialization["input_activity_adaptation"] = {
            "kind": "per_presentation_gain_before_log1p", "label_independent": True,
            "fixed_TRAIN_reference": profile["event_mass_reference"],
            "gain_limits": profile["event_gain_limits"],
            "parameters_thresholds_row_targets": "retained exactly",
            "normalization": "TRAIN-only BN refit before initial probe and every VALID"}
    model.backbone_calibration = checkpoint["metadata"].get("backbone_calibration")
    model.hidden_calibration = checkpoint["metadata"].get("hidden_calibration")
    if cfg.get("own_phase_restore_rng", False):
        if "rng" not in checkpoint or "loader_rng" not in checkpoint:
            raise ValueError("Own phase RNG alignment requires checkpoint RNG and loader state")
        restore_rng(checkpoint["rng"])
        model.initialization["phase_randomness"] = "retained own predecessor RNG and TRAIN loader state"
    return changes


def supervised_training_forward(model, frames, labels):
    """One model presentation; temporal supervision never adds WTA cycles.

    Per-step CE matches the official TET classification term (lambda=0),
    with this repository's time-first layout. Inference always averages logits.
    """
    if getattr(model, "cfg", {}).get("temporal_training_loss", "mean_ce") == "per_step_ce":
        sequence = model.forward_sequence(frames)
        if sequence.ndim != 3 or sequence.shape[0] < 1 or sequence.shape[1] != labels.numel():
            raise ValueError("Temporal logits must have shape [T, batch, classes]")
        logits = sequence.mean(0)
        loss = F.cross_entropy(sequence.flatten(0, 1), labels.repeat(sequence.shape[0]))
        return logits, loss, {"hard_ce_loss": F.cross_entropy(logits, labels).item(),
                              "temporal_ce_loss": loss.item()}
    logits = model(frames)
    return logits, F.cross_entropy(logits, labels), {}


def train_epoch(model, loader, optimizer, dataset, device, steps=None,
                monitor=None, epoch=0, on_probe=None, distiller=None):
    model.train()
    correct = total = 0
    loss_sum = 0.0
    component_sums = {}
    started = time.monotonic()
    progress_every = int(os.environ.get("SNN_PROGRESS_EVERY", "20"))
    for batch_index, (data, labels) in enumerate(loader, 1):
        if monitor:
            monitor.begin_batch(batch_index)
        model.reset_membranes()
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        frames = prepare_batch(data, dataset, device, steps, getattr(model, "cfg", None))
        frames = augment_training_frames(frames, dataset, getattr(model, "cfg", None))
        labels = labels.to(device)
        targets = distiller.probabilities(frames) if distiller else None
        if distiller:
            logits = model(frames)
            loss, components = distiller.loss(logits, labels, targets)
        else:
            logits, loss, components = supervised_training_forward(model, frames, labels)
        for key in ("hard_ce_loss", "soft_kl_scaled_loss", "temporal_ce_loss"):
            if key in components:
                component_sums[key] = component_sums.get(key, 0.) + components[key] * labels.numel()
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise TrainingAborted("nonfinite_training_loss", {"epoch": epoch, "batch": batch_index, "loss": loss_value})
        loss.backward()
        diagnostics = monitor.diagnostics(model) if monitor and monitor.capture else {}
        try:
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.bp_parameters(), 1.0, error_if_nonfinite=True)
        except RuntimeError as error:
            if "non-finite" not in str(error):
                raise
            raise TrainingAborted("nonfinite_training_gradient", {"epoch": epoch, "batch": batch_index,
                                                                   "loss": loss_value}) from error
        with WinnerOptimizerGuard(model.hidden_layers.values(), optimizer):
            optimizer.step()
        for hidden in model.hidden_layers.values():
            hidden.apply_stdp()
        model.project_nonnegative()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.numel()
        loss_sum += loss_value * labels.numel()
        elapsed = time.monotonic() - started
        progress = {"epoch": epoch, "batch": batch_index, "batches": len(loader), "samples": total,
                    "loss": loss_value, "mean_loss": loss_sum/total, "train_top1_accuracy": correct/total,
                    "gradient_norm": gradient_norm.item(), "lr": optimizer.param_groups[0]["lr"],
                    "elapsed_seconds": elapsed, "samples_per_second": total/max(elapsed, 1e-9),
                    "estimated_epoch_remaining_seconds": elapsed/batch_index*(len(loader)-batch_index)}
        progress.update(components)
        progress["learning_rates"] = {group.get("name", str(index)): group["lr"]
                                       for index, group in enumerate(optimizer.param_groups)}
        if monitor and monitor.capture:
            monitor.log_batch(**progress, **diagnostics)
        elif not monitor and progress_every > 0 and (batch_index == 1 or batch_index % progress_every == 0):
            print(json.dumps({"event": "train_progress", **progress}), flush=True)
        if monitor and on_probe and monitor.should_probe():
            on_probe(epoch, batch_index)
    model.reset_membranes()
    return {"correct": correct, "samples": total, "top1_accuracy": correct/max(total, 1),
            "loss": loss_sum/max(total, 1), "elapsed_seconds": time.monotonic()-started,
            **{key: value/max(total, 1) for key, value in component_sums.items()},
            **({"loss_kind": "hard_ce_and_own_teacher_kl"} if distiller else
               {"loss_kind": "per_step_ce"} if getattr(model, "cfg", {}).get("temporal_training_loss", "mean_ce") == "per_step_ce" else {})}


@torch.no_grad()
def evaluate(model, loader, dataset, device, steps=None, max_samples=None, progress=None):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    started = last_log = time.monotonic()
    predictions = labels_histogram = logit_min = logit_max = confusion = None
    for data, labels in loader:
        if max_samples is not None:
            remaining = max_samples-total
            if remaining <= 0:
                break
            data, labels = data[:remaining], labels[:remaining]
        model.reset_membranes()
        frames = prepare_batch(data, dataset, device, steps, getattr(model, "cfg", None))
        logits = model(frames)
        if not torch.isfinite(logits).all():
            raise TrainingAborted("nonfinite_evaluation_logits", {"samples": total})
        labels = labels.to(device)
        predicted = logits.argmax(1)
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        counts = torch.bincount(predicted, minlength=logits.shape[1])
        label_counts = torch.bincount(labels, minlength=logits.shape[1])
        matrix = torch.bincount(labels*logits.shape[1]+predicted, minlength=logits.shape[1]**2).reshape(logits.shape[1], logits.shape[1])
        if predictions is None:
            predictions, labels_histogram = counts, label_counts
            confusion = matrix
            logit_min, logit_max = logits.amin(0), logits.amax(0)
        else:
            predictions += counts
            labels_histogram += label_counts
            confusion += matrix
            logit_min = torch.minimum(logit_min, logits.amin(0))
            logit_max = torch.maximum(logit_max, logits.amax(0))
        correct += (predicted == labels).sum().item()
        total += labels.numel()
        if progress and time.monotonic()-last_log >= 30:
            progress(samples=total, loss=loss_sum/total, top1_accuracy=correct/total,
                     elapsed_seconds=time.monotonic()-started)
            last_log = time.monotonic()
    model.reset_membranes()
    if total == 0:
        raise ValueError("Empty evaluation dataset")
    return {"correct": correct, "samples": total, "top1_accuracy": correct/total, "loss": loss_sum/total,
            "prediction_histogram": predictions.tolist(), "label_histogram": labels_histogram.tolist(),
            "dominant_prediction_fraction": predictions.max().item()/total,
            "majority_class_baseline": labels_histogram.max().item()/total,
            "logit_range_max": (logit_max-logit_min).max().item(),
            "confusion_matrix": confusion.tolist(),
            "per_class_accuracy": [confusion[i, i].item()/n if n else None for i, n in enumerate(labels_histogram.tolist())]}


def validation_probe(model, loader, dataset, device, steps, samples, seed):
    """Evaluate the fixed probe/Poisson stream; preserve training RNG and mode."""
    saved_rng = rng_state()
    training_modes = [(module, module.training) for module in model.modules()]
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return evaluate(model, loader, dataset, device, steps, max_samples=samples)
    finally:
        restore_rng(saved_rng)
        for module, training in training_modes:
            module.training = training
        model.reset_membranes()


def evaluate_precisions(model, loader, dataset, device, steps=None):
    model.eval()
    initial_rng = rng_state()
    results = {"float32": evaluate(model, loader, dataset, device, steps)}
    for bits in (2, 4):
        restore_rng(initial_rng)
        with quantized_hidden_weights(model, bits) as quantization:
            results[f"uint{bits}"] = evaluate(model, loader, dataset, device, steps)
            results[f"uint{bits}"]["quantized_layers"] = [item["layer"] for item in quantization]
    return results


@torch.no_grad()
def calibrate_backbone_normalization(model, loader, dataset, device, steps=None, batches=8):
    """Initialize BN population moments from TRAIN only before hidden calibration."""
    saved_rng = rng_state()
    generator = getattr(loader, "generator", None)
    loader_rng = generator.get_state() if generator is not None else None
    modes = [(module, module.training) for module in model.modules()]
    norms = [module for module in model.backbone.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    momenta = [module.momentum for module in norms]
    count = samples = 0
    try:
        model.backbone.train()
        for module in norms:
            module.reset_running_stats()
            module.momentum = None
        for data, _ in loader:
            model.reset_membranes()
            frames = prepare_batch(data, dataset, device, steps, getattr(model, "cfg", None))
            model.backbone(frames)
            samples += data.shape[0]
            count += 1
            if count >= batches:
                break
        return {"kind": "training_only_BN_moments", "batches": count, "samples": samples}
    finally:
        for module, momentum in zip(norms, momenta):
            module.momentum = momentum
        restore_rng(saved_rng)
        if loader_rng is not None:
            generator.set_state(loader_rng)
        for module, training in modes:
            module.training = training
        model.reset_membranes()


@torch.no_grad()
def calibrate_hidden(model, loader, dataset, device, steps=None):
    """Set hidden current scales from one TRAIN batch without consuming RNG."""
    saved_rng = rng_state()
    generator = getattr(loader, "generator", None)
    loader_rng = generator.get_state() if generator is not None else None
    modes = [(module, module.training) for module in model.modules()]
    summary = {}
    try:
        model.eval()
        data, _ = next(iter(loader))
        frames = prepare_batch(data, dataset, device, steps, getattr(model, "cfg", None))
        model.reset_membranes()
        features = model.pool(model.backbone(frames))
        if features.ndim == 5:
            features = features.flatten(2)
        for name, hidden in model.hidden_layers.items():
            current = F.linear(features, hidden.weight, hidden.bias).mean().item()
            target = 0.6*hidden.thresholds.baseline
            gain = target/current if current > 1e-12 else 1.0
            hidden.weight.mul_(gain).clamp_(0, hidden.weight_max)
            hidden.reset_row_sum_target()
            features = hidden(features)
            summary[name] = {"mean_current_before": current, "target_mean_current": target,
                             "weight_gain": gain, "calibration_spike_rate": features.mean().item(),
                             "training_samples": data.shape[0]}
        model.project_nonnegative()
        return summary
    finally:
        restore_rng(saved_rng)
        if loader_rng is not None:
            generator.set_state(loader_rng)
        for module, training in modes:
            module.training = training
        model.reset_membranes()


def metadata(model, cfg, mode, split, limit=None, steps=None):
    return {"model": model.family, "dataset": asdict(dataset_spec(model.dataset, cfg)),
            "input_preprocessing": {"native_source": asdict(DATASETS[model.dataset]),
                                    "runtime_profile": cfg.get("dataset_profiles", {}).get(model.dataset, {}),
                                    "temporal_rebin": "adjacent bin sum; counts conserved before log1p",
                                    "spatial_rebin": "nonoverlapping pixel sum; counts conserved before log1p"},
            "event_activity_normalization": {
                "profile": cfg.get("dataset_profiles", {}).get(model.dataset, {}),
                "stage": "after count-preserving rebin, before log1p",
                "label_independent": True, "post_normalization_counts_conserved": False}
                if cfg.get("dataset_profiles", {}).get(model.dataset, {}).get("event_transform") == "mass_normalized_log1p" else None,
            "homeostasis": mode, "seed": cfg["seed"], "training_precision": "float32",
            "quantization_scope": "nonnegative hidden Linear weights only; per-output-row max scale, zero point 0",
            "convolution_learning": "BP, float32, nonnegative projection, no WTA",
            "classification_head": "BP, float32, nonnegative projection, no WTA, no quantization",
            "classifier_projection": cfg.get("classifier_projection", "clamp"),
            "training_objective": cfg.get("temporal_training_loss", "mean_ce"),
            "validation_readout": "mean temporal logits; hard-label cross entropy",
            "hidden_learning": cfg["learning_rules"], "threshold_state_at_eval": "frozen",
            "hidden_winner_tiebreak": cfg.get("hidden_winner_tiebreak", "index"),
            "hidden_winner_scope": cfg.get("hidden_winner_scope", "presentation"),
            "hidden_neuron_rules": {name: {"bp_neurons": int(h.bp_neuron_mask.sum()),
                                           "stdp_neurons": int(h.stdp_neuron_mask.sum()),
                                           "assignment": f"indices divisible by {h.hybrid_stdp_period} use STDP" if h.rule == "hybrid" else h.rule}
                                    for name, h in model.hidden_layers.items()},
            "source": model.source, "initialization": model.initialization,
            "own_distillation": getattr(model, "distillation_provenance", None),
            "backbone_calibration": getattr(model, "backbone_calibration", None),
            "training_augmentation": cfg.get("training_augmentation", {}).get(model.dataset, {}),
            "lr_schedule": {"kind": cfg.get("lr_schedule", "cosine"),
                            "plateau": cfg.get("lr_plateau", {}),
                            "metric": "complete_validation_top1" if cfg.get("lr_schedule", "cosine") == "plateau" else None},
            "bn_refit_policy": {"training_samples_only": True, "batches": cfg.get("bn_refit_batches", 0),
                                "stages": ["validation_probe", "full_validation"]},
            "hidden_calibration": getattr(model, "hidden_calibration", None),
            "adaptations": model.adaptations, "size": model.describe_size(),
            "split_manifest": str(Path(split).resolve()) if split else None,
            "config": cfg, "sample_limit": limit, "step_override": steps,
            "evidence_scope": "functional_check" if limit or steps else "full_dataset",
            "paired_poisson_inputs": True}


def complete_test_evaluation(model, loaders, args, cfg, run_meta, out, monitor, stop):
    """Evaluate the best completed-epoch checkpoint and preserve stop outcome."""
    best_state = torch.load(out / "best.pth", map_location=next(model.parameters()).device, weights_only=True)
    model.load_state_dict(best_state["model"])
    if getattr(args, "validation_only", False):
        result = {"metadata": run_meta, "selected_epoch": best_state["epoch"],
                  "best_validation_accuracy": best_state["best_validation_accuracy"],
                  "training_stop": stop, "test_evaluated": False}
        (out / "validation.json").write_text(json.dumps(result, indent=2)+"\n")
        monitor.emit("validation_qualification_completed", result=str((out / "validation.json").resolve()),
                     best_validation_accuracy=best_state["best_validation_accuracy"], test_evaluated=False)
        return
    seed_everything(cfg["seed"]+1000)
    monitor.emit("test_evaluation_started", selected_epoch=best_state["epoch"])
    scores = evaluate_precisions(model, loaders[2], args.dataset, next(model.parameters()).device, args.steps)
    outcome = "class_collapse" if stop["reason"] == "persistent_single_class_predictions_without_validation_improvement" else "normal_completion"
    result = {"metadata": run_meta, "selected_epoch": best_state["epoch"], "training_stop": stop,
              "training_outcome": outcome, "best_validation_accuracy": best_state["best_validation_accuracy"], "test": scores}
    (out / "accuracy.json").write_text(json.dumps(result, indent=2)+"\n")
    monitor.emit("training_completed", result=str((out / "accuracy.json").resolve()), test=scores,
                 training_stop=stop, training_outcome=outcome)


def run_training(args, cfg):
    if "cpu_threads" in cfg:
        torch.set_num_threads(cfg["cpu_threads"])
    seed_everything(cfg["seed"])
    device = choose_device(args.device or cfg.get("device", "auto"))
    # Adam's non-capturable scalar step belongs on CPU. Loading the entire
    # checkpoint onto CUDA would also move those counters and RNG byte tensors.
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    own_path = getattr(args, "own_init_checkpoint", None)
    if cfg.get("own_time_bn_adaptation", False) and not (own_path or args.resume):
        raise ValueError("Own time-BN adaptation requires own initialization or continuous resume")
    if cfg.get("own_time_bn_adaptation", False) and cfg.get("bn_refit_batches", 0) < 1:
        raise ValueError("Own time-BN phase requires TRAIN-only BN refit")
    if cfg.get("own_first_hidden_hybrid_adaptation", False) and not (own_path or args.resume):
        raise ValueError("Own learning adaptation requires own initialization or its continuous resume")
    if cfg.get("own_distillation") is not None and not (own_path or args.resume):
        raise ValueError("Own distillation requires own initialization or continuous resume")
    if cfg.get("own_native_resolution_adaptation", False) and not (own_path or args.resume):
        raise ValueError("Own native resolution adaptation requires own initialization or continuous resume")
    if cfg.get("own_native_resolution_adaptation", False) and cfg.get("bn_refit_batches", 0) < 1:
        raise ValueError("Own native phase requires TRAIN-only BN refit")
    if cfg.get("own_event_activity_adaptation", False) and not (own_path or args.resume):
        raise ValueError("Own activity adaptation requires own initialization or continuous resume")
    if cfg.get("own_event_activity_adaptation", False) and cfg.get("bn_refit_batches", 0) < 1:
        raise ValueError("Own activity adaptation requires TRAIN-only BN refit")
    if own_path and args.model != "custom":
        raise ValueError("Own initialization requires a custom checkpoint originating from scratch")
    if own_path and (args.resume or args.init_checkpoint):
        raise ValueError("Own initialization cannot be combined with resume or external initialization")
    own_checkpoint = torch.load(own_path, map_location="cpu", weights_only=True) if own_path else None
    if checkpoint is not None:
        args.batch_size = args.batch_size or checkpoint["metadata"].get("batch_size", DATASETS[args.dataset].batch_size)
    loaders = build_loaders(args.dataset, cfg, download=args.download,
                            batch_size=args.batch_size, limit=args.limit)
    model = AccuracySNN(args.model, args.dataset, cfg, args.homeostasis).to(device)
    if args.model == "custom":
        model.initialization = {"kind": "from_scratch"}
        if args.init_checkpoint:
            raise ValueError("Custom training starts from scratch; use --resume to continue its own run")
    elif not args.resume:
        if args.init_checkpoint or cfg.get("init") in {"imagenet", "dataset_snn"}:
            model.initialize_official(args.init_checkpoint)
        else:
            raise ValueError("Official models require pretrained initialization")
    if own_checkpoint is not None:
        initialize_own_custom_phase(model, own_checkpoint, args, cfg)
        if cfg.get("own_phase_restore_rng", False):
            loaders[0].generator.set_state(own_checkpoint["loader_rng"].cpu())
        checksum = hashlib.sha256()
        with Path(own_path).open("rb") as source_file:
            for block in iter(lambda: source_file.read(8 * 1024 ** 2), b""):
                checksum.update(block)
        model.initialization["source_checkpoint_sha256"] = checksum.hexdigest()
        del own_checkpoint
    if not args.resume and not own_path and cfg.get("backbone_variant") in {"balanced_ei", "balanced_residual_ei", "balanced_temporal_ei"}:
        model.backbone_calibration = calibrate_backbone_normalization(model, loaders[0], args.dataset, device, args.steps)
    if not args.resume and not own_path and cfg.get("calibrate_hidden", False):
        model.hidden_calibration = calibrate_hidden(model, loaders[0], args.dataset, device, args.steps)
    learning_rate = cfg.get("dataset_profiles", {}).get(args.dataset, {}).get("lr", cfg.get("model_learning_rates", {}).get(args.model, cfg["lr"]))
    if cfg.get("optimizer", "adam") == "sgd":
        optimizer = torch.optim.SGD(model.bp_parameters(), lr=learning_rate, momentum=0.9, weight_decay=cfg["weight_decay"])
    else:
        optimizer = torch.optim.Adam(model.bp_parameters(), lr=learning_rate, weight_decay=cfg["weight_decay"])
    epochs = args.epochs or (checkpoint["metadata"].get("training_epochs") if checkpoint else None) or epoch_budget(cfg, args.model, args.dataset)
    budget_extension = validate_resume_budget(checkpoint, cfg, epochs, getattr(args, "extend_plateau_budget", False))
    qualification_window = getattr(args, "qualification_epochs", None)
    if qualification_window is not None:
        if (not getattr(args, "validation_only", False) or isinstance(qualification_window, bool)
                or not isinstance(qualification_window, int) or not 1 <= qualification_window <= epochs):
            raise ValueError("qualification_epochs requires validation_only and an integer within the total epoch budget")
    scheduler = make_lr_scheduler(optimizer, cfg, epochs)
    out = Path(cfg["out_dir"]) / f"{args.model}_{args.dataset}_{args.homeostasis}_seed{cfg['seed']}"
    if args.limit or args.steps:
        out = out / "functional_check"
    out.mkdir(parents=True, exist_ok=True)
    start, best, history = 0, -1.0, []
    if checkpoint is not None:
        check_run_identity(checkpoint["metadata"], args, cfg)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        model.initialization = checkpoint["metadata"]["initialization"]
        model.hidden_calibration = checkpoint["metadata"].get("hidden_calibration")
        model.backbone_calibration = checkpoint["metadata"].get("backbone_calibration")
        start, best = checkpoint["epoch"]+1, checkpoint["best_validation_accuracy"]
        history = checkpoint["history"]
        restore_rng(checkpoint["rng"])
        loaders[0].generator.set_state(checkpoint["loader_rng"].cpu())
        best_epoch = max(history, key=lambda entry: entry["validation"]["top1_accuracy"])["epoch"]
        if checkpoint["epoch"] == best_epoch:
            selected = checkpoint
        else:
            original_best = Path(args.resume).parent / "best.pth"
            if not original_best.is_file():
                raise FileNotFoundError("Resume needs the historical best.pth beside the source checkpoint")
            selected = torch.load(original_best, map_location="cpu", weights_only=True)
            check_run_identity(selected["metadata"], args, cfg)
            if selected["epoch"] != best_epoch:
                raise ValueError("The source best.pth does not match the resumed run's validation history")
        torch.save(selected, out / "best.pth")
    distiller = None
    if cfg.get("own_distillation") is not None:
        if args.limit or args.steps:
            raise ValueError("Own distillation requires full dataset input protocol")
        distiller = OwnTeacherBundle(cfg, args.dataset, args.homeostasis, loaders[3], device)
        model.distillation_provenance = distiller.provenance
    run_meta = metadata(model, cfg, args.homeostasis, loaders[3], args.limit, args.steps)
    if getattr(args, "validation_only", False):
        run_meta["evidence_scope"] = "validation_qualification"
    run_meta["training_epochs"] = epochs
    if budget_extension is not None:
        run_meta["schedule_budget_extension"] = budget_extension
    run_meta["batch_size"] = loaders[0].batch_size
    probe_loader, probe_manifest = build_probe_loader(
        loaders[1], cfg["monitoring"]["probe_samples"], cfg["seed"]+2000)
    run_meta["validation_probe"] = probe_manifest
    (out / "validation_probe.json").write_text(json.dumps(probe_manifest, indent=2)+"\n")
    monitor = TrainingMonitor(out, cfg["monitoring"])
    if checkpoint and "monitoring_state" in checkpoint:
        monitor.guard.state = checkpoint["monitoring_state"]
    else:
        monitor.guard.state["global_step"] = start*len(loaders[0])
    monitor.attach(model)
    epoch = start
    stop = {"reason": "epoch_budget_completed"}

    def probe(epoch_index, batch_index, initial=False):
        if cfg.get("bn_refit_batches", 0):
            stats = calibrate_backbone_normalization(model, loaders[0], args.dataset, device, args.steps,
                                                    batches=cfg["bn_refit_batches"])
            monitor.emit("bn_refit", stage="validation_probe", epoch=epoch_index, **stats)
        monitor.emit("validation_probe_started", epoch=epoch_index, batch=batch_index, initial=initial)
        scores = validation_probe(model, probe_loader, args.dataset, device, args.steps,
                                  cfg["monitoring"]["probe_samples"], cfg["seed"]+2000)
        monitor.emit("validation_probe", epoch=epoch_index, batch=batch_index, initial=initial, **scores)
        monitor.guard.check_validation(scores, initial=initial)

    try:
        monitor.emit("training_started", model=args.model, dataset=args.dataset, start_epoch=start,
                     epoch_budget=epochs, monitoring=cfg["monitoring"], initialization=model.initialization)
        end = epochs
        if qualification_window is not None:
            end = min(epochs, start + qualification_window)
            stop = {"reason": "qualification_window_completed", "epoch": end-1,
                    "total_epoch_budget": epochs, "qualification_epochs": qualification_window}
        if monitor.guard.plateau_reached(start):
            end = start
            stop = {"reason": "validation_accuracy_plateau", "epoch": start-1,
                    "stale_epochs": monitor.guard.state["stale_epochs"]}
            monitor.emit("early_stopping", **stop)
        else:
            probe(start, 0, initial=True)
        for epoch in range(start, end):
            train = train_epoch(model, loaders[0], optimizer, args.dataset, device, args.steps,
                                monitor=monitor, epoch=epoch, on_probe=probe, distiller=distiller)
            if cfg.get("bn_refit_batches", 0):
                stats = calibrate_backbone_normalization(model, loaders[0], args.dataset, device, args.steps,
                                                        batches=cfg["bn_refit_batches"])
                monitor.emit("bn_refit", stage="full_validation", epoch=epoch, **stats)
            monitor.emit("validation_started", epoch=epoch)
            val = evaluate(model, loaders[1], args.dataset, device, args.steps,
                           progress=lambda **stats: monitor.emit("validation_progress", epoch=epoch, **stats))
            previous_lrs, current_lrs = step_lr_scheduler(scheduler, val["top1_accuracy"])
            if cfg.get("lr_schedule", "cosine") == "plateau" and previous_lrs != current_lrs:
                monitor.emit("learning_rate_reduced", epoch=epoch, previous_lrs=previous_lrs,
                             learning_rates=current_lrs, complete_validation_accuracy=val["top1_accuracy"])
            improved = val["top1_accuracy"] > best
            best = max(best, val["top1_accuracy"])
            entry = {"epoch": epoch, "train": train, "validation": val}
            history.append(entry)
            monitor.emit("epoch_completed", **entry)
            early_stop = monitor.guard.epoch_should_stop(val["top1_accuracy"], epoch+1)
            # Apply guards before publishing a successful training result.
            monitor.guard.check_validation(val, stage="epoch")
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "epoch": epoch,
                     "best_validation_accuracy": best, "history": history,
                     "metadata": run_meta, "rng": rng_state(), "loader_rng": loaders[0].generator.get_state(),
                     "monitoring_state": monitor.guard.state}
            torch.save(state, out / "latest.pth")
            if improved:
                torch.save(state, out / "best.pth")
            if early_stop:
                stop = {"reason": "validation_accuracy_plateau", "epoch": epoch,
                        "stale_epochs": monitor.guard.state["stale_epochs"]}
                monitor.emit("early_stopping", **stop)
                break
        if not (out / "best.pth").exists():
            raise RuntimeError("No best checkpoint is available; resume in the original output directory or run at least one epoch")
        complete_test_evaluation(model, loaders, args, cfg, run_meta, out, monitor, stop)
    except TrainingAborted as error:
        diagnostic = out / "diagnostic.pth"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "metadata": run_meta, "partial_epoch": epoch, "history": history,
                    "monitoring_state": monitor.guard.state,
                    "reason": error.reason, "resume_supported": False}, diagnostic)
        stopped = monitor.emit("training_aborted", status="aborted", reason=error.reason,
                               metrics=error.metrics, diagnostic_checkpoint=str(diagnostic.resolve()),
                               final_test_evaluated=False)
        (out / "stopped.json").write_text(json.dumps(stopped, indent=2)+"\n")
        if (cfg.get("evaluate_on_collapse", False) and
                error.reason == "persistent_single_class_predictions_without_validation_improvement" and
                (out / "best.pth").exists()):
            stop = {"reason": error.reason, "partial_epoch": epoch, "global_step": monitor.guard.state["global_step"],
                    "diagnostic": str((out / "stopped.json").resolve())}
            complete_test_evaluation(model, loaders, args, cfg, run_meta, out, monitor, stop)
            return
        raise SystemExit(2) from error
    finally:
        monitor.close()


def check_run_identity(meta, args, cfg):
    if meta["config"].get("custom_batch_norm", "shared") != cfg.get("custom_batch_norm", "shared"):
        raise ValueError("Checkpoint protocol mismatch for custom_batch_norm")
    if meta["config"].get("own_time_bn_adaptation", False) != cfg.get("own_time_bn_adaptation", False):
        raise ValueError("Checkpoint protocol mismatch for own_time_bn_adaptation")
    if (meta.get("final_refit") and (getattr(args, "resume", None) or getattr(args, "own_init_checkpoint", None))
            and not getattr(args, "official_train_refit", False)):
        raise ValueError("A model fitted on TRAIN plus VAL cannot return to held-out-validation selection")
    if getattr(args, "resume", None) and meta.get("validation_probe", {}).get("sampling") != PROBE_SAMPLING:
        raise ValueError("Validation probe sampling changed; start a new campaign instead of resuming old guard state")
    expected = (args.model, args.dataset, args.homeostasis, cfg["hidden_widths"], cfg["learning_rules"], cfg["seed"])
    actual = (meta["model"], meta["dataset"]["name"], meta["homeostasis"],
              meta["config"]["hidden_widths"], meta["config"]["learning_rules"], meta["seed"])
    if actual != expected:
        raise ValueError(f"Checkpoint protocol mismatch: expected {expected}, got {actual}")
    if meta["config"].get("hidden_winner_scope", "presentation") != cfg.get("hidden_winner_scope", "presentation"):
        raise ValueError("Checkpoint protocol mismatch for hidden_winner_scope")
    for key in ("threshold", "weight_max", "stdp_lr", "custom_channels", "resnet_adapter",
                "validation_fraction", "cifar_train_fraction", "lr", "weight_decay",
                "vgg_pool_size", "model_learning_rates", "training_epochs",
                "hidden_surrogate", "hidden_initial_row_sum", "backbone_variant", "pretrained_checkpoints", "batch_sizes",
                "calibrate_hidden", "hidden_fan_in", "optimizer", "evaluate_on_collapse",
                "dataset_profiles", "custom_pool_size", "hidden_dropout", "cpu_threads",
                "hidden_competition_size", "bp_row_normalize", "bp_row_normalize_layers", "gesture_validation_subjects", "training_augmentation", "bn_refit_batches", "hybrid_stdp_period",
                "lr_schedule", "lr_plateau", "hidden_winner_tiebreak", "custom_residual_blocks", "custom_residual_connection", "custom_temporal_layers", "own_phase_restore_rng", "own_distillation"):
        if meta["config"].get(key) != cfg.get(key):
            raise ValueError(f"Checkpoint protocol mismatch for {key}")
    if meta["config"].get("own_first_hidden_hybrid_adaptation", False) != cfg.get("own_first_hidden_hybrid_adaptation", False):
        raise ValueError("Checkpoint protocol mismatch for own_first_hidden_hybrid_adaptation")
    if meta["config"].get("own_native_resolution_adaptation", False) != cfg.get("own_native_resolution_adaptation", False):
        raise ValueError("Checkpoint protocol mismatch for own_native_resolution_adaptation")
    if meta["dataset"]["resolution"] != dataset_spec(args.dataset, cfg).resolution:
        raise ValueError("Checkpoint input resolution differs from current dataset protocol")
    if meta["config"].get("classifier_projection", "clamp") != cfg.get("classifier_projection", "clamp"):
        raise ValueError("Checkpoint protocol mismatch for classifier_projection")
    if meta["config"].get("temporal_training_loss", "mean_ce") != cfg.get("temporal_training_loss", "mean_ce"):
        raise ValueError("Checkpoint protocol mismatch for temporal_training_loss")
    if meta["config"].get("own_event_activity_adaptation", False) != cfg.get("own_event_activity_adaptation", False):
        raise ValueError("Checkpoint protocol mismatch for own_event_activity_adaptation")
    if "monitoring" in meta["config"] and monitoring_config(meta["config"]["monitoring"]) != cfg["monitoring"]:
        raise ValueError("Keep monitoring settings for reproducible resume/eval")
    if (meta["sample_limit"], meta["step_override"]) != (args.limit, args.steps):
        raise ValueError("Keep the checkpoint's --limit and --steps settings for resume/eval")
    if meta["dataset"]["steps"] != dataset_spec(args.dataset, cfg).steps:
        raise ValueError("Checkpoint time steps differ from the current dataset protocol")
    expected_batch = args.batch_size or cfg.get("batch_sizes", {}).get(args.dataset, DATASETS[args.dataset].batch_size)
    if "batch_size" in meta and meta["batch_size"] != expected_batch:
        raise ValueError("Keep the checkpoint's batch size for paired evaluation and resume")


def matrix(cfg):
    rows = []
    for model, dataset, mode, bits in product(cfg["models"], cfg["datasets"],
                                            cfg["homeostasis_modes"], cfg["weight_bits"]):
        rows.append({"model": model, "dataset": dataset, "homeostasis": mode, "bits": bits,
                     "steps": dataset_spec(dataset, cfg).steps, "resolution": dataset_spec(dataset, cfg).resolution,
                     "epochs": epoch_budget(cfg, model, dataset),
                     "initialization": "from_scratch" if model == "custom" else
                                       ("author_pretrained_nonnegative_SNN" if cfg.get("backbone_variant") == "positive_ei" else
                                        "official_pretrained_transfer_or_checkpoint")})
    return {"float_training_runs": len(rows)//len(cfg["weight_bits"]),
            "quantized_accuracy_cases": len(rows), "cases": rows,
            "official_initialization_urls": SOURCE_URLS if cfg.get("init") == "imagenet" else {},
            "pretrained_checkpoints": cfg.get("pretrained_checkpoints", {})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/accuracy_matrix.yaml"))
    parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("matrix")
    prepare = subs.add_parser("prepare")
    prepare.add_argument("--dataset", choices=DATASETS, required=True)
    for name in ("train", "eval"):
        sub = subs.add_parser(name)
        sub.add_argument("--model", choices=["vgg11_bn", "resnet18", "custom"], required=True)
        sub.add_argument("--dataset", choices=DATASETS, required=True)
        sub.add_argument("--homeostasis", choices=["off", "exponential", "rate"], default="exponential")
        sub.add_argument("--device")
        sub.add_argument("--batch-size", type=int)
        sub.add_argument("--limit", type=int, help="Functional checks only; not a full-dataset result")
        sub.add_argument("--steps", type=int, help="MNIST functional checks only")
        sub.add_argument("--download", action="store_true")
        if name == "train":
            sub.add_argument("--epochs", type=int)
            sub.add_argument("--init-checkpoint")
            sub.add_argument("--resume")
            sub.add_argument("--own-init-checkpoint",
                             help="New regularization phase from this dataset's own from-scratch custom checkpoint; preserve architecture/rules/coding/split/weights/thresholds")
            sub.add_argument("--extend-plateau-budget", action="store_true",
                             help="Explicitly extend an own-run metric-driven plateau budget; retain its complete training state")
            sub.add_argument("--validation-only", action="store_true", help="Learning qualification; do not evaluate test data")
            sub.add_argument("--qualification-epochs", type=int,
                             help="Validation-only prefix length; preserve --epochs total schedule for exact resume")
        else:
            sub.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command in {"train", "eval"}:
        for key in ("limit", "steps", "batch_size", "epochs", "qualification_epochs"):
            value = getattr(args, key, None)
            if value is not None and value <= 0:
                parser.error(f"--{key.replace('_', '-')} must be positive")
        if args.steps is not None and args.dataset != "MNIST":
            parser.error("Event time steps are fixed by the dataset cache; --steps is for MNIST checks only")
    torch.set_num_threads(args.threads)
    if args.command == "matrix":
        print(json.dumps(matrix(cfg), indent=2))
    elif args.command == "prepare":
        datasets = build_datasets(args.dataset, cfg, download=True)
        print(json.dumps({"dataset": args.dataset, "counts": [len(ds) for ds in datasets[:3]],
                          "split": str(datasets[3].resolve())}))
    elif args.command == "train":
        run_training(args, cfg)
    else:
        seed_everything(cfg["seed"]+1000)
        device = choose_device(args.device or cfg["device"])
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        args.batch_size = args.batch_size or checkpoint["metadata"].get("batch_size", DATASETS[args.dataset].batch_size)
        check_run_identity(checkpoint["metadata"], args, cfg)
        model = AccuracySNN(args.model, args.dataset, cfg, args.homeostasis).to(device)
        model.load_state_dict(checkpoint["model"])
        loaders = build_loaders(args.dataset, cfg, args.download, args.batch_size, args.limit)
        print(json.dumps(evaluate_precisions(model, loaders[2], args.dataset, device, args.steps), indent=2))


if __name__ == "__main__":
    main()
