from dataclasses import asdict, dataclass, replace
from pathlib import Path

import yaml

from .monitoring import monitoring_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    classes: int
    channels: int
    resolution: int
    steps: int
    batch_size: int
    epochs: int
    coding: str


# MNIST uses the existing our_snn.py T=4 default after the design review.
# Event datasets retain their prepared T=16 caches and native resolution.
DATASETS = {
    "MNIST": DatasetSpec("MNIST", 10, 1, 28, 4, 64, 30, "poisson"),
    "NMNIST": DatasetSpec("NMNIST", 10, 2, 34, 16, 8, 64, "event_count_frames"),
    "CIFAR10DVS": DatasetSpec("CIFAR10DVS", 10, 2, 128, 16, 8, 64, "event_count_frames"),
    "DVSGesture": DatasetSpec("DVSGesture", 11, 2, 128, 16, 8, 64, "event_count_frames"),
}


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    allowed = {"models", "datasets", "homeostasis_modes", "weight_bits", "hidden_widths",
               "learning_rules", "seed", "lr", "weight_decay", "stdp_lr", "weight_max",
               "validation_fraction", "cifar_train_fraction", "data_dir", "out_dir",
               "init", "threshold", "device", "num_workers", "custom_channels", "resnet_adapter",
               "monitoring", "training_epochs", "model_learning_rates", "vgg_pool_size",
               "pretrained_checkpoints", "hidden_surrogate", "hidden_initial_row_sum",
               "backbone_variant", "batch_sizes", "calibrate_hidden", "hidden_fan_in", "optimizer",
               "evaluate_on_collapse", "dataset_profiles", "custom_pool_size", "hidden_dropout", "cpu_threads",
               "hidden_competition_size", "hybrid_stdp_period", "hybrid_stdp_granularity",
               "hidden_winner_tiebreak", "hidden_winner_scope", "stdp_mode", "stdp_input_encoding",
               "stdp_event_amplitude", "stdp_coefficient_lsb", "stdp_update_rule", "stdp_record_stats",
               "a_plus", "a_minus", "trace_decay"}
    if set(cfg)-allowed:
        raise ValueError(f"Unknown config keys: {sorted(set(cfg)-allowed)}")
    if not set(cfg.get("datasets", [])) <= DATASETS.keys():
        raise ValueError("Unsupported dataset")
    if not set(cfg.get("models", [])) <= {"vgg11_bn", "resnet18", "custom"}:
        raise ValueError("Unsupported model")
    if not set(cfg.get("homeostasis_modes", [])) <= {"off", "exponential", "rate"}:
        raise ValueError("Unsupported homeostasis mode")
    if cfg.get("weight_bits", [2, 4]) != [2, 4]:
        raise ValueError("This experiment requires both 2-bit and 4-bit inference")
    widths, rules = cfg.get("hidden_widths", []), cfg.get("learning_rules", [])
    if not widths or len(widths) != len(rules) or any(w <= 0 for w in widths):
        raise ValueError("Every hidden width needs a learning rule")
    if not set(rules) <= {"bp", "stdp", "hybrid"}:
        raise ValueError("Hidden learning rules must be bp, stdp or hybrid")
    if "resnet18" in cfg.get("models", []) and not cfg.get("resnet_adapter", True):
        raise ValueError("The agreed ResNet experiment requires the hidden FC adapter")
    cfg["monitoring"] = monitoring_config(cfg.get("monitoring"))
    for key in ("training_epochs", "model_learning_rates"):
        values = cfg.get(key, {})
        if not isinstance(values, dict) or set(values)-{"vgg11_bn", "resnet18", "custom"}:
            raise ValueError(f"Invalid {key} mapping")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0 for v in values.values()):
            raise ValueError(f"{key} values must be positive")
        if key == "training_epochs" and any(not isinstance(v, int) for v in values.values()):
            raise ValueError("training_epochs values must be integers")
    if cfg.get("vgg_pool_size", 7) not in {1, 7}:
        raise ValueError("vgg_pool_size must be 1 or 7")
    if cfg.get("hidden_surrogate", "rectangle") not in {"rectangle", "atan"}:
        raise ValueError("Unknown hidden surrogate")
    if cfg.get("hidden_initial_row_sum", 4.0) <= 0:
        raise ValueError("hidden_initial_row_sum must be positive")
    if cfg.get("optimizer", "adam") not in {"adam", "sgd"}:
        raise ValueError("Unknown optimizer")
    if not isinstance(cfg.get("evaluate_on_collapse", False), bool):
        raise ValueError("evaluate_on_collapse must be boolean")
    if cfg.get("evaluate_on_collapse") and cfg["monitoring"]["collapse_warmup_epochs"] < 1:
        raise ValueError("evaluate_on_collapse requires at least one complete validation epoch")
    if cfg.get("hidden_fan_in") is not None and (not isinstance(cfg["hidden_fan_in"], int) or cfg["hidden_fan_in"] <= 0):
        raise ValueError("hidden_fan_in must be a positive integer")
    if cfg.get("backbone_variant", "spikingjelly") not in {"spikingjelly", "positive_ei", "balanced_ei"}:
        raise ValueError("Unknown backbone variant")
    profiles = cfg.get("dataset_profiles", {})
    if not isinstance(profiles, dict) or set(profiles)-DATASETS.keys():
        raise ValueError("Invalid dataset_profiles mapping")
    for name, profile in profiles.items():
        if not isinstance(profile, dict) or set(profile)-{"steps", "resolution", "event_transform", "pool_size", "epochs", "lr"}:
            raise ValueError(f"Invalid dataset profile for {name}")
        spec = dataset_spec(name, cfg)
        if isinstance(spec.steps, bool) or not isinstance(spec.steps, int) or spec.steps <= 0:
            raise ValueError("Profile time steps must be positive integers")
        if name != "MNIST" and DATASETS[name].steps % spec.steps:
            raise ValueError("Event time steps must exactly divide the native cache length")
        if isinstance(spec.resolution, bool) or not isinstance(spec.resolution, int) or spec.resolution <= 0:
            raise ValueError("Profile resolution must be a positive integer")
        if name == "MNIST" and spec.resolution != DATASETS[name].resolution:
            raise ValueError("MNIST retains its native spatial resolution")
        if DATASETS[name].resolution % spec.resolution:
            raise ValueError("Event resolution must divide the native spatial resolution")
        if profile.get("event_transform", "none") not in {"none", "log1p"}:
            raise ValueError("Unknown event transform")
        if profile.get("pool_size", 1) not in {1, 2, 4}:
            raise ValueError("Profile pool_size must be 1, 2 or 4")
        if "epochs" in profile and (isinstance(profile["epochs"], bool) or not isinstance(profile["epochs"], int) or profile["epochs"] <= 0):
            raise ValueError("Profile epochs must be a positive integer")
        if "lr" in profile and (isinstance(profile["lr"], bool) or not isinstance(profile["lr"], (int, float)) or profile["lr"] <= 0):
            raise ValueError("Profile learning rate must be positive")
    if cfg.get("custom_pool_size", 1) not in {1, 2, 4}:
        raise ValueError("custom_pool_size must be 1, 2 or 4")
    if not 0 <= cfg.get("hidden_dropout", 0) < 1:
        raise ValueError("hidden_dropout must be in [0, 1)")
    threads = cfg.get("cpu_threads", 4)
    if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
        raise ValueError("cpu_threads must be a positive integer")
    batches = cfg.get("batch_sizes", {})
    if not isinstance(batches, dict) or set(batches)-DATASETS.keys() or any(
            isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in batches.values()):
        raise ValueError("Invalid batch_sizes mapping")
    for key, default in (("data_dir", "datasets"), ("out_dir", "results/accuracy")):
        location = Path(cfg.get(key, default)).expanduser()
        cfg[key] = str((PROJECT_ROOT / location if not location.is_absolute() else location).resolve())
    return cfg


def dataset_manifest(spec):
    return asdict(spec)


def dataset_spec(name, cfg=None):
    """Runtime shape; source caches and partition indices remain native."""
    profile = (cfg or {}).get("dataset_profiles", {}).get(name, {})
    return replace(DATASETS[name], **{k: profile[k] for k in ("steps", "resolution") if k in profile})


def epoch_budget(cfg, family, dataset=None):
    default = cfg.get("training_epochs", {}).get(family, 30 if family == "custom" else 10)
    return cfg.get("dataset_profiles", {}).get(dataset, {}).get("epochs", default)
