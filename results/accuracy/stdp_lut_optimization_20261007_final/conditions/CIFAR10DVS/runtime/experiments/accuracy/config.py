from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path

import yaml

from .monitoring import monitoring_config
from .distillation import validate_distillation_config


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
               "hidden_competition_size", "bp_row_normalize", "gesture_validation_subjects", "training_augmentation", "bn_refit_batches", "hybrid_stdp_period",
               "lr_schedule", "lr_plateau", "hidden_winner_tiebreak", "custom_residual_blocks", "custom_residual_connection"}
    allowed.update({"hidden_winner_scope", "custom_temporal_layers", "own_phase_restore_rng",
                    "own_first_hidden_hybrid_adaptation", "bp_row_normalize_layers", "own_distillation", "own_native_resolution_adaptation", "classifier_projection", "own_event_activity_adaptation", "temporal_training_loss"})
    allowed.update({"custom_batch_norm", "own_time_bn_adaptation"})
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
    norm = cfg.get("custom_batch_norm", "shared")
    if not isinstance(norm, str) or norm not in {"shared", "time"}:
        raise ValueError("custom_batch_norm must be shared or time")
    time_adaptation = cfg.get("own_time_bn_adaptation", False)
    if not isinstance(time_adaptation, bool):
        raise ValueError("own_time_bn_adaptation must be boolean")
    if norm == "time" and (cfg.get("models") != ["custom"] or cfg.get("backbone_variant") != "balanced_ei"):
        raise ValueError("Time-specific BN currently requires custom spatial balanced_ei")
    if time_adaptation and (norm != "time" or cfg.get("bn_refit_batches", 0) < 1):
        raise ValueError("Own time-BN adaptation requires time mode and TRAIN BN refit")
    temporal_loss = cfg.get("temporal_training_loss", "mean_ce")
    if not isinstance(temporal_loss, str) or temporal_loss not in {"mean_ce", "per_step_ce"}:
        raise ValueError("temporal_training_loss must be mean_ce or per_step_ce")
    if temporal_loss == "per_step_ce" and (cfg.get("models") != ["custom"] or cfg.get("own_distillation") is not None):
        raise ValueError("Per-step CE requires a custom model without own distillation")
    validate_distillation_config(cfg.get("own_distillation"))
    if cfg.get("own_distillation") is not None and cfg.get("models") != ["custom"]:
        raise ValueError("Own distillation requires a custom student")
    native = cfg.get("own_native_resolution_adaptation", False)
    if not isinstance(native, bool):
        raise ValueError("own_native_resolution_adaptation must be boolean")
    if native and (cfg.get("models") != ["custom"] or cfg.get("datasets") != ["CIFAR10DVS"]
                   or cfg.get("backbone_variant") != "balanced_ei"
                   or dataset_spec("CIFAR10DVS", cfg).resolution != 128):
        raise ValueError("Own native resolution adaptation requires custom CIFAR10DVS balanced_ei at128")
    if cfg.get("classifier_projection", "clamp") not in {"clamp", "common_offset"}:
        raise ValueError("classifier_projection must be clamp or common_offset")
    if cfg.get("classifier_projection", "clamp") == "common_offset" and cfg.get("models") != ["custom"]:
        raise ValueError("Common-offset classifier projection is a custom-network experiment")
    activity = cfg.get("own_event_activity_adaptation", False)
    if not isinstance(activity, bool):
        raise ValueError("own_event_activity_adaptation must be boolean")
    if activity and (cfg.get("models") != ["custom"] or cfg.get("datasets") != ["DVSGesture"]
                     or cfg.get("dataset_profiles", {}).get("DVSGesture", {}).get("event_transform") != "mass_normalized_log1p"):
        raise ValueError("Own event activity adaptation requires custom Gesture normalized event profile")
    widths, rules = cfg.get("hidden_widths", []), cfg.get("learning_rules", [])
    if not widths or len(widths) != len(rules) or any(w <= 0 for w in widths):
        raise ValueError("Every hidden width needs a learning rule")
    if not set(rules) <= {"bp", "stdp", "hybrid"}:
        raise ValueError("Hidden learning rules must be bp, stdp or hybrid")
    group = cfg.get("hidden_competition_size")
    if cfg.get("hidden_winner_tiebreak", "index") not in {"index", "margin"}:
        raise ValueError("hidden_winner_tiebreak must be index or margin")
    if cfg.get("hidden_winner_scope", "presentation") not in {"presentation", "cycle"}:
        raise ValueError("hidden_winner_scope must be presentation or cycle")
    if not isinstance(cfg.get("own_phase_restore_rng", False), bool):
        raise ValueError("own_phase_restore_rng must be boolean")
    if not isinstance(cfg.get("own_first_hidden_hybrid_adaptation", False), bool):
        raise ValueError("own_first_hidden_hybrid_adaptation must be boolean")
    if cfg.get("own_first_hidden_hybrid_adaptation", False) and (
            cfg.get("models") != ["custom"] or rules != ["hybrid", "bp"]):
        raise ValueError("Own first-hidden hybrid adaptation requires custom and hybrid/BP rules")
    if group is not None and (isinstance(group, bool) or not isinstance(group, int) or group <= 0 or any(w % group for w in widths)):
        raise ValueError("hidden_competition_size must divide all hidden widths")
    period = cfg.get("hybrid_stdp_period", 4)
    if (isinstance(period, bool) or not isinstance(period, int) or period < 2
            or ("hybrid" in rules and any((group or width) % period for width in widths))):
        raise ValueError("hybrid_stdp_period must be >=2 and divide hybrid competition groups")
    if not isinstance(cfg.get("bp_row_normalize", False), bool):
        raise ValueError("bp_row_normalize must be boolean")
    normalized_layers = cfg.get("bp_row_normalize_layers", [])
    hidden_names = {f"hidden{index+1}" for index in range(len(widths))}
    if (not isinstance(normalized_layers, list) or any(not isinstance(name, str) for name in normalized_layers)
            or len(set(normalized_layers)) != len(normalized_layers) or not set(normalized_layers) <= hidden_names):
        raise ValueError("bp_row_normalize_layers must list unique existing hidden layer names")
    refit = cfg.get("bn_refit_batches", 0)
    if isinstance(refit, bool) or not isinstance(refit, int) or refit < 0:
        raise ValueError("bn_refit_batches must be a nonnegative integer")
    subjects = cfg.get("gesture_validation_subjects", 0)
    if isinstance(subjects, bool) or not isinstance(subjects, int) or not 0 <= subjects < 23:
        raise ValueError("gesture_validation_subjects must be in [0, 22]")
    augmentation = cfg.get("training_augmentation", {})
    if not isinstance(augmentation, dict) or set(augmentation)-DATASETS.keys():
        raise ValueError("Invalid training_augmentation mapping")
    for name, settings in augmentation.items():
        if not isinstance(settings, dict) or set(settings)-{"translate", "scale_min", "scale_max", "rotation", "horizontal_flip"}:
            raise ValueError("Unknown training augmentation")
        if not 0 <= settings.get("translate", 0) < 0.5 or not 0 <= settings.get("horizontal_flip", 0) <= 1:
            raise ValueError("Invalid augmentation probability or translation")
        if not 0 < settings.get("scale_min", 1) <= settings.get("scale_max", 1) or not 0 <= settings.get("rotation", 0) <= 30:
            raise ValueError("Invalid augmentation scale or rotation")
        if name == "DVSGesture" and settings.get("horizontal_flip", 0):
            raise ValueError("Gesture reflection can change gesture labels; do not enable horizontal flips")
    if "resnet18" in cfg.get("models", []) and not cfg.get("resnet_adapter", True):
        raise ValueError("The agreed ResNet experiment requires the hidden FC adapter")
    cfg["monitoring"] = monitoring_config(cfg.get("monitoring"))
    if cfg.get("lr_schedule", "cosine") not in {"cosine", "plateau"}:
        raise ValueError("lr_schedule must be cosine or plateau")
    plateau = cfg.get("lr_plateau", {})
    if not isinstance(plateau, dict) or set(plateau)-{"factor", "patience", "min_lr", "threshold"}:
        raise ValueError("Invalid lr_plateau mapping")
    for key, value in plateau.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"lr_plateau.{key} must be finite")
        if key == "factor" and not 0 < value < 1:
            raise ValueError("lr_plateau.factor must be in (0, 1)")
        if key == "patience" and (not isinstance(value, int) or value < 1):
            raise ValueError("lr_plateau.patience must be a positive integer")
        if key == "min_lr" and value < 0:
            raise ValueError("lr_plateau.min_lr must be nonnegative")
        if key == "threshold" and not 0 <= value < 1:
            raise ValueError("lr_plateau.threshold must be an accuracy fraction in [0, 1)")
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
    if cfg.get("backbone_variant", "spikingjelly") not in {"spikingjelly", "positive_ei", "balanced_ei", "balanced_residual_ei", "balanced_temporal_ei"}:
        raise ValueError("Unknown backbone variant")
    if cfg.get("backbone_variant") == "balanced_temporal_ei":
        depth = cfg.get("custom_temporal_layers", 2)
        channels = cfg.get("custom_channels", [])
        if (cfg.get("models") != ["custom"] or isinstance(depth, bool) or not isinstance(depth, int)
                or not 1 <= depth <= len(channels)
                or any(isinstance(width, bool) or not isinstance(width, int) or width <= 0 for width in channels)):
            raise ValueError("Custom temporal EI needs custom-only and1..CNNdepth temporal layers")
    if cfg.get("backbone_variant") == "balanced_residual_ei":
        if cfg.get("custom_residual_connection", "current") not in {"current", "spike_add"}:
            raise ValueError("Unknown custom residual connection")
        channels = cfg.get("custom_channels", [])
        blocks = cfg.get("custom_residual_blocks", 2)
        if (cfg.get("models") != ["custom"] or len(channels) != 4
                or any(isinstance(width, bool) or not isinstance(width, int) or width <= 0 for width in channels)
                or isinstance(blocks, bool) or not isinstance(blocks, int) or not 1 <= blocks <= 4):
            raise ValueError("Custom residual EI needs custom-only, four positive widths, and1..4 blocks/stage")
    profiles = cfg.get("dataset_profiles", {})
    if not isinstance(profiles, dict) or set(profiles)-DATASETS.keys():
        raise ValueError("Invalid dataset_profiles mapping")
    for name, profile in profiles.items():
        if not isinstance(profile, dict) or set(profile)-{"steps", "resolution", "event_transform", "pool_size", "epochs", "lr", "event_mass_reference", "event_gain_limits"}:
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
        if profile.get("event_transform", "none") not in {"none", "log1p", "mass_normalized_log1p"}:
            raise ValueError("Unknown event transform")
        if profile.get("event_transform") == "mass_normalized_log1p":
            reference, limits = profile.get("event_mass_reference"), profile.get("event_gain_limits")
            if name == "MNIST" or isinstance(reference, bool) or not isinstance(reference, (int, float)) or not math.isfinite(reference) or reference <= 0:
                raise ValueError("Normalized events require positive finite event_mass_reference and an event dataset")
            if (not isinstance(limits, list) or len(limits) != 2
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in limits)
                    or not 0 < limits[0] <= 1 <= limits[1]):
                raise ValueError("event_gain_limits must be finite positive limits containing unity")
        elif {"event_mass_reference", "event_gain_limits"} & profile.keys():
            raise ValueError("Activity parameters require mass_normalized_log1p")
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
