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
    allowed.update({"custom_batch_norm", "own_time_bn_adaptation", "own_spatial_pool_adaptation"})
    allowed.add("custom_temporal_implementation")
    allowed.add("custom_readout_pool")
    allowed.add("label_smoothing")
    allowed.add("own_average_adaptation")
    allowed.add("freeze_backbone_bn")
    allowed.add("freeze_backbone_parameters")
    allowed.add("classifier_only_phase")
    allowed.add("temporal_readout_start")
    allowed.add("split_seed")
    allowed.add("training_mixup")
    allowed.add("gesture_semantic_reflection")
    allowed.add("custom_pool_after")
    allowed.update({"custom_normalization", "custom_group_norm_groups"})
    allowed.update({"temporal_classifier", "own_temporal_classifier_adaptation"})
    allowed.add("temporal_classifier_initialization")
    allowed.add("hybrid_stdp_granularity")
    time_initialization = cfg.get("temporal_classifier_initialization", "independent")
    if not isinstance(time_initialization, str) or time_initialization not in {"independent", "shared"}:
        raise ValueError("temporal_classifier_initialization must be independent or shared")
    if time_initialization == "shared" and not cfg.get("temporal_classifier", False):
        raise ValueError("Shared temporal_classifier_initialization requires temporal_classifier=true")
    temporal_classifier = cfg.get("temporal_classifier", False)
    temporal_head_adaptation = cfg.get("own_temporal_classifier_adaptation", False)
    if not isinstance(temporal_classifier, bool) or not isinstance(temporal_head_adaptation, bool):
        raise ValueError("temporal_classifier and own_temporal_classifier_adaptation must be boolean")
    if temporal_classifier and (cfg.get("models") != ["custom"]
            or cfg.get("temporal_training_loss", "mean_ce") not in {"mean_ce", "per_step_ce"}
            or cfg.get("own_distillation") is not None):
        raise ValueError("Temporal classifier requires custom mean-CE or per-step CE supervision without distillation")
    if temporal_head_adaptation and not temporal_classifier:
        raise ValueError("Own temporal classifier adaptation requires temporal_classifier=true")
    normalization = cfg.get("custom_normalization", "batch")
    if not isinstance(normalization, str) or normalization not in {"batch", "group"}:
        raise ValueError("custom_normalization must be batch or group")
    if normalization == "group":
        groups = cfg.get("custom_group_norm_groups", 32)
        widths = cfg.get("custom_channels", [])
        if (isinstance(groups, bool) or not isinstance(groups, int) or groups < 1
                or not widths or any(isinstance(w, bool) or not isinstance(w, int) or w < 1 or w % groups for w in widths)):
            raise ValueError("custom_group_norm_groups must be positive and divide every convolution width")
        if (cfg.get("models") != ["custom"] or cfg.get("backbone_variant") != "balanced_ei"
                or cfg.get("custom_batch_norm", "shared") != "shared"
                or cfg.get("bn_refit_batches", 0) != 0 or cfg.get("freeze_backbone_bn", False)
                or cfg.get("own_time_bn_adaptation", False)):
            raise ValueError("GroupNorm requires custom spatial balanced_ei, no time BN, frozen BN or BN refit")
    elif "custom_group_norm_groups" in cfg:
        raise ValueError("custom_group_norm_groups requires custom_normalization=group")
    split_seed = cfg.get("split_seed", cfg.get("seed", 42))
    if isinstance(split_seed, bool) or not isinstance(split_seed, int) or not 0 <= split_seed < 2**63:
        raise ValueError("split_seed must be a nonnegative integer smaller than 2**63")
    readout_start = cfg.get("temporal_readout_start", 0)
    if isinstance(readout_start, bool) or not isinstance(readout_start, int) or readout_start < 0:
        raise ValueError("temporal_readout_start must be a nonnegative integer")
    if readout_start and (cfg.get("models") != ["custom"] or any(
            readout_start >= dataset_spec(name, cfg).steps for name in cfg.get("datasets", []))):
        raise ValueError("temporal_readout_start requires custom and a nonempty temporal readout")
    if set(cfg)-allowed:
        raise ValueError(f"Unknown config keys: {sorted(set(cfg)-allowed)}")
    temporal_implementation = cfg.get("custom_temporal_implementation", "conv3d")
    if not isinstance(temporal_implementation, str) or temporal_implementation not in {"conv3d", "lagged_conv2d"}:
        raise ValueError("custom_temporal_implementation must be conv3d or lagged_conv2d")
    if temporal_implementation == "lagged_conv2d" and cfg.get("backbone_variant") != "balanced_temporal_ei":
        raise ValueError("Lagged Conv2D implementation requires causal temporal EI")
    readout_pool = cfg.get("custom_readout_pool", "avg")
    if not isinstance(readout_pool, str) or readout_pool not in {"avg", "max"}:
        raise ValueError("custom_readout_pool must be avg or max")
    if readout_pool == "max" and (cfg.get("models") != ["custom"] or cfg.get("backbone_variant") not in {"balanced_ei", "balanced_temporal_ei", "balanced_residual_ei"}):
        raise ValueError("Max readout requires our balanced EI custom topology")
    if cfg.get("own_spatial_pool_adaptation", False) and readout_pool != "avg":
        raise ValueError("Nested own spatial expansion requires average readout")
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
    smoothing = cfg.get("label_smoothing", 0.0)
    if (isinstance(smoothing, bool) or not isinstance(smoothing, (int, float))
            or not math.isfinite(smoothing) or not 0 <= smoothing < 1):
        raise ValueError("label_smoothing must be finite and in [0, 1)")
    if smoothing and (cfg.get("models") != ["custom"] or cfg.get("own_distillation") is not None):
        raise ValueError("Label smoothing requires custom supervision without own distillation")
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
    spatial = cfg.get("own_spatial_pool_adaptation", False)
    if not isinstance(spatial, bool):
        raise ValueError("own_spatial_pool_adaptation must be boolean")
    if spatial and (cfg.get("models") != ["custom"] or cfg.get("datasets") != ["DVSGesture"]
                    or cfg.get("backbone_variant") != "balanced_ei"
                    or cfg.get("dataset_profiles", {}).get("DVSGesture", {}).get("pool_size") != 4):
        raise ValueError("Own spatial pool adaptation requires custom Gesture spatial EI pool4")
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
    average_adaptation = cfg.get("own_average_adaptation", False)
    if not isinstance(average_adaptation, bool):
        raise ValueError("own_average_adaptation must be boolean")
    if average_adaptation and (cfg.get("models") != ["custom"] or cfg.get("own_phase_restore_rng", False)):
        raise ValueError("Own averaged initialization requires custom and a fresh seeded RNG policy")
    frozen_bn = cfg.get("freeze_backbone_bn", False)
    if not isinstance(frozen_bn, bool):
        raise ValueError("freeze_backbone_bn must be boolean")
    if frozen_bn and (cfg.get("models") != ["custom"] or cfg.get("bn_refit_batches", 0) != 0):
        raise ValueError("Frozen BN requires custom with bn_refit_batches=0")
    frozen_parameters = cfg.get("freeze_backbone_parameters", False)
    if not isinstance(frozen_parameters, bool):
        raise ValueError("freeze_backbone_parameters must be boolean")
    if frozen_parameters and (cfg.get("models") != ["custom"] or not frozen_bn):
        raise ValueError("Frozen backbone parameters require custom and freeze_backbone_bn=true")
    classifier_only = cfg.get("classifier_only_phase", False)
    if not isinstance(classifier_only, bool):
        raise ValueError("classifier_only_phase must be boolean")
    if classifier_only and (not frozen_parameters or cfg.get("bp_row_normalize", False)
                            or cfg.get("bp_row_normalize_layers")):
        raise ValueError("Classifier-only phase requires a frozen backbone and no hidden row projection")
    if not isinstance(cfg.get("own_first_hidden_hybrid_adaptation", False), bool):
        raise ValueError("own_first_hidden_hybrid_adaptation must be boolean")
    if cfg.get("own_first_hidden_hybrid_adaptation", False) and (
            cfg.get("models") != ["custom"] or rules != ["hybrid", "bp"]):
        raise ValueError("Own first-hidden hybrid adaptation requires custom and hybrid/BP rules")
    if group is not None and (isinstance(group, bool) or not isinstance(group, int) or group <= 0 or any(w % group for w in widths)):
        raise ValueError("hidden_competition_size must divide all hidden widths")
    period = cfg.get("hybrid_stdp_period", 4)
    granularity = cfg.get("hybrid_stdp_granularity", "neuron")
    if not isinstance(granularity, str) or granularity not in {"neuron", "group"}:
        raise ValueError("hybrid_stdp_granularity must be neuron or group")
    if (isinstance(period, bool) or not isinstance(period, int) or period < 2
            or any(rule == "hybrid" and (
                (group or width) % period if granularity == "neuron"
                else width % ((group or width) * period))
                for width, rule in zip(widths, rules))):
        raise ValueError("hybrid_stdp_period must be >=2 and divide the selected neuron or group population")
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
    pool_after = cfg.get("custom_pool_after")
    if pool_after is not None:
        if (not isinstance(pool_after, list) or any(type(i) is not int for i in pool_after)
                or pool_after != sorted(set(pool_after))
                or any(i < 0 or i >= len(cfg["custom_channels"])-1 for i in pool_after)
                or cfg.get("models") != ["custom"] or cfg.get("backbone_variant") != "balanced_ei"):
            raise ValueError("custom_pool_after must specify sorted unique nonfinal balanced custom convolution indices")
    reflection = cfg.get("gesture_semantic_reflection", {})
    if not isinstance(reflection, dict) or set(reflection)-{"probability", "class_permutation"}:
        raise ValueError("Invalid gesture_semantic_reflection mapping")
    if reflection:
        if cfg.get("models") != ["custom"] or cfg.get("datasets") != ["DVSGesture"] or cfg.get("own_distillation") is not None:
            raise ValueError("gesture_semantic_reflection requires custom supervised Gesture")
        probability = reflection.get("probability")
        permutation = reflection.get("class_permutation")
        if isinstance(probability, bool) or not isinstance(probability, (float, int)) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid gesture_semantic_reflection probability")
        expected = [0, 3, 2, 1, 7, 6, 5, 4, 8, 9, 10]
        if permutation != expected or any(type(x) is not int for x in permutation):
            raise ValueError("gesture_semantic_reflection requires audited DatasetFolder class permutation")
    mixup = cfg.get("training_mixup", {})
    if not isinstance(mixup, dict) or set(mixup)-DATASETS.keys():
        raise ValueError("Invalid training_mixup mapping")
    if mixup and (cfg.get("models") != ["custom"] or cfg.get("own_distillation") is not None):
        raise ValueError("training_mixup requires custom supervised training without distillation")
    for name, settings in mixup.items():
        if not isinstance(settings, dict) or set(settings)-{"alpha", "probability"} or "alpha" not in settings:
            raise ValueError("Invalid training_mixup settings")
        for key, value in settings.items():
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError("training_mixup values must be finite numbers")
        if settings["alpha"] <= 0 or not 0 <= settings.get("probability", 1) <= 1:
            raise ValueError("Invalid training_mixup alpha or probability")
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
