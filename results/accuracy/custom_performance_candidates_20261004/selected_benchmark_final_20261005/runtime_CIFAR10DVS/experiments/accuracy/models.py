"""Pinned official backbones; all adaptations and initialization sources are reported."""

import hashlib
import inspect
from pathlib import Path

import torch
from torch import nn
from spikingjelly.activation_based import functional, layer, neuron, surrogate
from spikingjelly.activation_based.model import spiking_resnet, spiking_vgg

from snn.wta import WTAHidden
from .config import DATASETS, PROJECT_ROOT, dataset_spec


SOURCE_URLS = {"vgg11_bn": spiking_vgg.model_urls["vgg11_bn"],
               "resnet18": spiking_resnet.model_urls["resnet18"]}


def source_manifest(family):
    module = spiking_vgg if family == "vgg11_bn" else spiking_resnet
    path = Path(inspect.getfile(module))
    return {"package": "spikingjelly", "version": "0.0.0.0.14",
            "module": module.__name__, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "documentation": f"https://spikingjelly.readthedocs.io/zh-cn/0.0.0.0.14/_modules/{module.__name__.replace('.', '/')}.html"}


class AccuracySNN(nn.Module):
    def __init__(self, family, dataset, cfg, mode):
        super().__init__()
        spec = dataset_spec(dataset, cfg)
        self.cfg = cfg
        self.family, self.dataset = family, dataset
        kwargs = dict(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True, backend="torch")
        self.adaptations = ["all Conv/Linear synaptic weights are projected nonnegative"]
        positive_ei = cfg.get("backbone_variant") == "positive_ei"
        if cfg.get("backbone_variant") in {"balanced_ei", "balanced_residual_ei", "balanced_temporal_ei"}:
            if family != "custom":
                raise ValueError("balanced_ei is our custom topology and has no official checkpoint")
            from .nonnegative_backbones import CustomBalancedBackbone, CustomBalancedResidualBackbone
            residual = cfg["backbone_variant"] == "balanced_residual_ei"
            self.backbone = (CustomBalancedResidualBackbone(spec.channels, cfg["custom_channels"], cfg.get("custom_residual_blocks", 2), cfg.get("custom_residual_connection", "current"))
                             if residual else CustomBalancedBackbone(spec.channels, cfg["custom_channels"],
                                cfg.get("custom_temporal_layers", 2) if cfg["backbone_variant"] == "balanced_temporal_ei" else 0,
                                spec.steps if cfg.get("custom_batch_norm", "shared") == "time" else None))
            pool_size = cfg.get("dataset_profiles", {}).get(dataset, {}).get("pool_size", cfg.get("custom_pool_size", 2))
            self.pool = layer.AdaptiveAvgPool2d(pool_size)
            features = cfg["custom_channels"][-1] * pool_size**2
            self.source = {"kind": "project_custom", "initialization": "from_scratch",
                           "topology": "paired nonnegative EI convolution, BatchNorm, PLIF, spatial readout"}
            if cfg.get("custom_batch_norm", "shared") == "time":
                self.source["normalization_reference"] = "https://github.com/Intelligent-Computing-Lab-Panda/BNTT-Batch-Normalization-Through-Time"
                self.adaptations.append("CNN BN has independent per-time-step moments/gamma/beta; learned beta retained unlike the reference; hidden WTA unchanged")
            if cfg["backbone_variant"] == "balanced_temporal_ei":
                self.source["topology"] = "causal three-tap temporal/spatial EI convolution, BatchNorm, PLIF, spatial readout"
                self.adaptations.append(f"first {cfg.get('custom_temporal_layers', 2)} stages use nonnegative E/I Conv3d taps at t-2,t-1,t; no future input")
            if residual:
                self.source["topology"] = "fresh four-stage residual EI convolution, current shortcut before PLIF, binary spike outputs"
                self.adaptations.append("residual shortcuts use paired nonnegative 1x1 projections when shape changes")
                if cfg.get("custom_residual_connection") == "spike_add":
                    self.source["topology"] = "fresh custom residual EI convolution, SEW-style spike addition after PLIF, spike-count spatial readout"
                    self.source["architecture_reference"] = "https://arxiv.org/abs/2102.04159"
                    self.adaptations.append("spike-add shortcuts bypass surrogate derivatives; each PLIF emits binary spikes, branch sums carry spike counts")
            self.adaptations.extend(["separate nonnegative excitatory and inhibitory convolution banks; explicit current subtraction",
                                     "BatchNorm stabilizes current without divisive inhibition",
                                     f"spatial readout retains {pool_size}x{pool_size} positions before hidden layers"])
        elif positive_ei:
            from .nonnegative_backbones import AuthorBackbone, CustomPositiveBackbone
            if family == "custom":
                self.backbone = CustomPositiveBackbone(spec.channels, cfg["custom_channels"], spec.steps, cfg["seed"])
                features = cfg["custom_channels"][-1]
                self.source = {"kind": "project_custom", "initialization": "from_scratch",
                               "topology": "seven-stage CNN with native nonnegative EI synapses"}
                self.adaptations.append("local inhibition prevents positive convolutions from becoming uniform intensity detectors")
            else:
                specification = cfg.get("pretrained_checkpoints", {}).get(family, {}).get(dataset)
                if specification is None:
                    raise ValueError(f"No verified nonnegative SNN checkpoint configured for {family}/{dataset}")
                self.backbone = AuthorBackbone(family, specification, spec.channels, spec.steps, spec.resolution)
                features = 512
                self.source = self.backbone.source
                self.adaptations.extend(self.backbone.adaptations)
            self.pool = nn.Identity()
        elif family == "vgg11_bn":
            self.backbone = spiking_vgg.SpikingVGG.make_layers(
                spiking_vgg.cfgs["A"], batch_norm=True, neuron=neuron.LIFNode, **kwargs)
            pool_size = cfg.get("vgg_pool_size", 7)
            self.pool = layer.AdaptiveAvgPool2d((pool_size, pool_size))
            features = 512*pool_size*pool_size
            if pool_size == 1:
                self.adaptations.append("VGG global average pooling replaces 7x7 expansion; hidden widths unchanged")
            # Native 28x28 MNIST otherwise becomes zero-sized at the fifth pool.
            if spec.resolution < 32:
                [m for m in self.backbone.modules() if isinstance(m, nn.MaxPool2d)][-1].ceil_mode = True
                self.adaptations.append("last VGG max pool uses ceil_mode for native 28x28 MNIST")
            self.source = source_manifest(family)
        elif family == "resnet18":
            self.backbone = spiking_resnet.spiking_resnet18(pretrained=False, spiking_neuron=neuron.LIFNode, **kwargs)
            self.backbone.fc = nn.Identity()
            self.pool = nn.Identity()
            features = 512
            self.source = source_manifest(family)
        elif family == "custom":
            modules, channels = [], spec.channels
            for index, width in enumerate(cfg["custom_channels"]):
                modules.extend([layer.Conv2d(channels, width, 3, padding=1, bias=False),
                                layer.BatchNorm2d(width), neuron.LIFNode(**kwargs)])
                if index in {0, 1, 3, 5}:
                    modules.append(layer.MaxPool2d(2, ceil_mode=True))
                channels = width
            self.backbone = nn.Sequential(*modules)
            self.pool = layer.AdaptiveAvgPool2d(1)
            features = channels
            self.source = {"kind": "project_custom", "initialization": "from_scratch"}
        else:
            raise ValueError(f"Unknown model: {family}")

        if family != "custom" and not positive_ei:
            self._replace_input_conv(spec.channels)
            self.adaptations.append(f"input convolution accepts {spec.channels} native input channels")
        widths = cfg["hidden_widths"]
        if family == "resnet18" and not cfg.get("resnet_adapter", True):
            widths = []
        self.hidden_layers = nn.ModuleDict()
        for index, width in enumerate(widths):
            self.hidden_layers[f"hidden{index+1}"] = WTAHidden(
                features, width, rule=cfg["learning_rules"][index], homeostasis=mode,
                weight_max=cfg["weight_max"], stdp_lr=cfg["stdp_lr"],
                surrogate_type=cfg.get("hidden_surrogate", "rectangle"),
                initial_row_sum=cfg.get("hidden_initial_row_sum", 4.0),
                competition_size=cfg.get("hidden_competition_size"),
                hybrid_stdp_period=cfg.get("hybrid_stdp_period", 4),
                winner_tiebreak=cfg.get("hidden_winner_tiebreak", "index"),
                winner_scope=cfg.get("hidden_winner_scope", "presentation"),
                initial_fan_in=cfg.get("hidden_fan_in"), **cfg["threshold"])
            features = width
        self.classifier = nn.Linear(features, spec.classes)
        self.dropout = nn.Dropout(0.5) if family == "vgg11_bn" and not positive_ei else nn.Dropout(cfg.get("hidden_dropout", 0))
        if cfg.get("backbone_variant") in {"balanced_ei", "balanced_residual_ei", "balanced_temporal_ei"}:
            with torch.no_grad():
                nn.init.uniform_(self.classifier.weight, 0, 0.002)
                self.classifier.bias.zero_()
        if family == "resnet18" and widths:
            self.adaptations.append(f"added hidden FC adapter with widths {widths} before the dataset head")
        self.adaptations.append(f"dataset classifier: {features}->{spec.classes}, BP, float32, no WTA")
        if cfg.get("hidden_winner_scope", "presentation") == "cycle":
            self.adaptations.append("hidden FC groups select one winner each cycle; BP updates the union of winning neurons, STDP uses actual winner events; adaptive thresholds persist")
        else:
            self.adaptations.append("hidden FC neurons use first-winner learning gates and adaptive thresholds")
        functional.set_step_mode(self.backbone, "m")
        if not isinstance(self.pool, nn.Identity):
            functional.set_step_mode(self.pool, "m")
        self.initialization = {"kind": "random_for_functional_checks"}
        self.project_nonnegative()

    def _replace_input_conv(self, channels):
        if self.family == "vgg11_bn":
            original = self.backbone[0]
            self.backbone[0] = layer.Conv2d(channels, original.out_channels, original.kernel_size,
                                          stride=original.stride, padding=original.padding, bias=original.bias is not None)
        else:
            original = self.backbone.conv1
            self.backbone.conv1 = layer.Conv2d(channels, original.out_channels, original.kernel_size,
                                             stride=original.stride, padding=original.padding, bias=False)

    def forward_sequence(self, x):
        x = self.pool(self.backbone(x))
        if x.ndim == 5:
            x = x.flatten(2)
        for hidden in self.hidden_layers.values():
            x = self.dropout(hidden(x))
        return self.classifier(x)

    def forward(self, x):
        return self.forward_sequence(x).mean(0)

    def reset_membranes(self):
        functional.reset_net(self.backbone)

    @torch.no_grad()
    def project_nonnegative(self):
        for module_name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                if module_name == "classifier" and self.cfg.get("classifier_projection", "clamp") == "common_offset":
                    # Each feature column shares the same subtraction across
                    # classes. All physical weights become nonnegative while
                    # every relative class logit is retained.
                    module.weight.sub_(module.weight.amin(dim=0, keepdim=True))
                else:
                    module.weight.clamp_(min=0)
            elif isinstance(module, WTAHidden):
                module.weight.clamp_(0, module.weight_max)
                normalize = (self.cfg.get("bp_row_normalize", False)
                             or module_name.removeprefix("hidden_layers.") in self.cfg.get("bp_row_normalize_layers", []))
                if normalize and module.rule != "stdp" and module.eligible_rows is not None:
                    rows = module.eligible_rows & module.bp_neuron_mask
                    weights = module.weight[rows]
                    weights *= (module.row_sum_target[rows] / weights.sum(1).clamp_min(1e-12)).unsqueeze(1)
                    module.weight[rows] = weights.clamp(0, module.weight_max)
            elif hasattr(module, "_clamp_parameters"):
                module._clamp_parameters()

    def bp_parameters(self):
        excluded = {id(p) for h in self.hidden_layers.values() if h.rule == "stdp" for p in (h.weight, h.bias)}
        return [p for p in self.parameters() if id(p) not in excluded]

    @torch.no_grad()
    def initialize_official(self, checkpoint=None):
        if self.family == "custom":
            raise ValueError("The custom network is trained from scratch")
        if self.source.get("kind") == "author_pretrained_nonnegative_SNN":
            if checkpoint is not None and str(Path(checkpoint).resolve()) != self.source["checkpoint"]:
                raise ValueError("Use the configured, verified author checkpoint for this backbone")
            self.initialization = {"kind": "author_pretrained_nonnegative_SNN",
                                   "path": self.source["checkpoint"], "sha256": self.source["sha256"],
                                   "source_dataset": self.source["source_dataset"]}
            return
        if checkpoint is None:
            url = SOURCE_URLS[self.family]
            cache = PROJECT_ROOT / "checkpoints" / "sources"
            state = torch.hub.load_state_dict_from_url(url, model_dir=str(cache), map_location="cpu",
                                                      check_hash=True, weights_only=True)
            checkpoint = cache / url.rsplit("/", 1)[-1]
            kind = "official_ImageNet_ANN_transfer_not_dataset_SNN_checkpoint"
        else:
            checkpoint = Path(checkpoint)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            state = state.get("net", state.get("model", state.get("state_dict", state)))
            kind = "external_checkpoint"
        state = {key.removeprefix("module."): value for key, value in state.items()}
        prefix = "features." if self.family == "vgg11_bn" else ""
        backbone = {key.removeprefix(prefix): value for key, value in state.items()
                    if key.startswith(prefix) and not key.startswith("fc.")}
        if self.family == "resnet18":
            backbone = {key: value for key, value in backbone.items() if not key.startswith("fc.")}
        first = "0.weight" if self.family == "vgg11_bn" else "conv1.weight"
        incoming = backbone[first]
        channels = DATASETS[self.dataset].channels
        if incoming.shape[1] != channels:
            backbone[first] = incoming.mean(1, keepdim=True).repeat(1, channels, 1, 1) * incoming.shape[1]/channels
        self.backbone.load_state_dict(backbone, strict=True)
        if self.family == "vgg11_bn":
            for index, hidden in enumerate(self.hidden_layers.values()):
                key = f"classifier.{index*3}.weight"
                if key in state and state[key].shape == hidden.weight.shape:
                    hidden.weight.copy_(state[key])
                    hidden.bias.copy_(state[f"classifier.{index*3}.bias"])
        self.project_nonnegative()
        for hidden in self.hidden_layers.values():
            hidden.reset_row_sum_target()
        digest = hashlib.sha256()
        with open(checkpoint, "rb") as file:
            for chunk in iter(lambda: file.read(8*1024*1024), b""):
                digest.update(chunk)
        self.initialization = {"kind": kind, "path": str(checkpoint.resolve()), "sha256": digest.hexdigest()}

    @torch.no_grad()
    def describe_size(self):
        original_training = self.training
        self.eval()
        sizes, handles = [], []
        def record(name):
            def hook(module, inputs, output):
                dimensions = output.shape[2:]
                sizes.append({"layer": name, "width": dimensions[0],
                              "spatial_neurons": int(torch.tensor(dimensions).prod().item())})
            return hook
        for name, module in self.named_modules():
            if isinstance(module, (neuron.BaseNode, WTAHidden)):
                handles.append(module.register_forward_hook(record(name)))
        spec = dataset_spec(self.dataset, self.cfg)
        param = next(self.parameters())
        try:
            self.reset_membranes()
            time = spec.steps if (hasattr(self.backbone, "features") or self.cfg.get("custom_batch_norm", "shared") == "time") else 1
            self(torch.zeros(time, 1, spec.channels, spec.resolution, spec.resolution, device=param.device))
        finally:
            for handle in handles:
                handle.remove()
            self.reset_membranes()
            self.train(original_training)
        return {"counting_rule": "sum convolution output channels and hidden neuron widths; classifier included separately",
                "spiking_width_sum": sum(item["width"] for item in sizes),
                "convolution_channel_sum": sum(module.out_channels for module in self.modules() if isinstance(module, (nn.Conv2d, nn.Conv3d))),
                "width_sum": sum(module.out_channels for module in self.modules() if isinstance(module, (nn.Conv2d, nn.Conv3d))) +
                             sum(module.out_features for module in self.hidden_layers.values()),
                "total_width_sum": sum(module.out_channels for module in self.modules() if isinstance(module, (nn.Conv2d, nn.Conv3d))) +
                                   sum(module.out_features for module in self.hidden_layers.values())+spec.classes,
                "spatial_neurons": sum(item["spatial_neurons"] for item in sizes),
                "parameters": sum(p.numel() for p in self.parameters()), "layers": sizes}
