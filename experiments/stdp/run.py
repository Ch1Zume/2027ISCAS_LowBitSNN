"""Bounded compressed-model STDP comparison, selected only on complete VALID.

The default condition starts from the final 3M--5M checkpoint. The default experiment
uses two blocks of 16 TRAIN batches and self-distillation from that compressed
checkpoint. All original BP/STDP roles and the full forward sequence are kept.
No TEST sample is read until every condition's checkpoint is fixed by VALID.
"""
import argparse
import ast
import copy
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(os.environ.get("SNN_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
BUNDLE = ROOT / "benchmarks/compressed"
DEFAULT_CONDITIONS = ("legacy_native", "legacy_events", "lut_events")
CONDITIONS = DEFAULT_CONDITIONS + ("lut_native_trace",)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def snapshot_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def adapt_models_source(code):
    """Bridge frozen pre-LUT model constructors without changing their defaults."""
    if "from snn.stdp_lut import stdp_options" not in code:
        code = code.replace("from snn.wta import WTAHidden", "from snn.wta import WTAHidden\nfrom snn.stdp_lut import stdp_options")
    needle = 'weight_max=cfg["weight_max"], stdp_lr=cfg["stdp_lr"],'
    assert code.count(needle) == 1
    if "**stdp_options(cfg)" not in code:
        code = code.replace(needle, needle + "\n                **stdp_options(cfg),")
    if "hybrid_stdp_granularity=" not in code:
        code = code.replace(needle, needle + '\n                hybrid_stdp_granularity=cfg.get("hybrid_stdp_granularity", "neuron"),')
    return code


def compression_rate(dataset, provenance, frozen_run):
    if "backbone_learning_rate" in provenance:
        return provenance["backbone_learning_rate"], "provenance.backbone_learning_rate"
    # The first compression runner predates that provenance field. Its actual
    # optimizer uses the RATES literal in the SHA-bound frozen runner, not cfg.lr.
    for node in ast.parse(frozen_run.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "RATES" for t in node.targets):
            return ast.literal_eval(node.value)[dataset], str(frozen_run) + ":RATES"
    raise ValueError("No auditable actual compression learning rate")


def spatial_student_transfer(model, state):
    """Retain selected trained filters; initialize fine spatial weights from coarse ones."""
    import torch
    assert set(model.hidden_layers) == {"hidden1"}
    source_weight = state["hidden_layers.hidden1.weight"]
    assert tuple(source_weight.shape) == (768, 2048)
    assert tuple(model.hidden_layers['hidden1'].weight.shape) == (256, 7168)
    centered = state['classifier.weight'] - state['classifier.weight'].mean(0, keepdim=True)
    importance = (centered @ source_weight).reshape(10, 512, 4).square().mean((0, 2))
    channels = importance.topk(448).indices.sort().values
    scores = centered.square().mean(0).sqrt() * source_weight.norm(dim=1)
    groups = scores.reshape(48, 16).sum(1).topk(16).indices.sort().values
    rows = (groups[:, None]*16 + torch.arange(16, device=groups.device)).flatten()
    target = model.state_dict()
    for key, value in target.items():
        if key in state and value.shape == state[key].shape:
            value.copy_(state[key])
    convolutions = [(name, module) for name,module in model.backbone.named_modules()
                    if hasattr(module, 'excitatory') and hasattr(module, 'inhibitory')]
    name, module = convolutions[-1]
    for bank in ('excitatory', 'inhibitory'):
        key = f'backbone.{name}.{bank}.weight'
        target[key].copy_(state[key][channels])
    normalization = str(int(name)+1)
    for field in ('weight', 'bias', 'running_mean', 'running_var'):
        key = f'backbone.{normalization}.{field}'
        target[key].copy_(state[key][channels])
    coarse = source_weight[rows].reshape(256, 512, 2, 2)[:, channels]
    expanded = coarse.repeat_interleave(2, -2).repeat_interleave(2, -1)/4
    weights = expanded.flatten(1)
    scales = source_weight[rows].sum(1) / weights.sum(1).clamp_min(1e-12)
    weights = (weights * scales[:, None]).clamp(0, model.hidden_layers['hidden1'].weight_max)
    target['hidden_layers.hidden1.weight'].copy_(weights)
    target['hidden_layers.hidden1.row_sum_target'].copy_(weights.sum(1))
    for field in ('bias', 'thresholds.offset', 'thresholds.rate'):
        key = f'hidden_layers.hidden1.{field}'
        target[key].copy_(state[key][rows])
    target['classifier.weight'].copy_(state['classifier.weight'][:, rows]*3)
    model.load_state_dict(target, strict=True)
    model.project_nonnegative()
    return dict(method='trained channel/group selection by downstream discrimination, 2x2 weights expanded to 4x4, positive row mass restored',
                channel_indices=channels.cpu().tolist(), hidden_indices=rows.cpu().tolist(),
                classifier_scale=3, source='immutable compressed checkpoint; no VALID/TEST statistics')


def fit_nonnegative_classifier(head, features, labels, teacher_targets, emit=None):
    """TRAIN-only ridge initialization and float BP, with a softmax-invariant gauge."""
    import torch
    from torch.nn import functional as F
    features, labels, teacher_targets = features.detach(), labels.detach(), teacher_targets.detach()
    mean = features.mean(0)
    centered = features-mean
    target = F.one_hot(labels, head.out_features).to(features)*5
    target_mean = target.mean(0)
    covariance = centered.T @ centered / len(features)
    covariance += torch.eye(head.in_features,device=features.device)*0.001
    coefficients = torch.linalg.solve(covariance, centered.T @ (target-target_mean)/len(features)).T
    with torch.no_grad():
        head.bias.copy_(target_mean-coefficients @ mean)
        head.weight.copy_(coefficients)
        # Adding the same feature coefficient to all classes shifts all logits
        # equally, preserving their differences while every physical weight is positive.
        head.weight.add_((0.25-head.weight.min(0).values).clamp_min(0).unsqueeze(0))
    optimizer = torch.optim.Adam(head.parameters(),lr=0.01)
    history = []
    for step in range(1,201):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features)
        ce = F.cross_entropy(logits,labels)
        kd = F.kl_div(F.log_softmax(logits/2,dim=1),F.softmax(teacher_targets/2,dim=1),reduction='batchmean')*4
        loss = 0.5*ce+0.5*kd
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all() for p in head.parameters())
        optimizer.step()
        with torch.no_grad():
            head.weight.add_((0.25-head.weight.min(0).values).clamp_min(0).unsqueeze(0))
        assert head.weight.dtype == torch.float32 and head.weight.min() >= 0
        if step == 1 or step % 20 == 0:
            record = dict(step=step,loss=float(loss),ce=float(ce),kd=float(kd),
                          accuracy=float((logits.argmax(1)==labels).float().mean()))
            history.append(record)
            if emit is not None:
                emit('TRAIN_head_fit_progress',**record)
    return dict(partition='TRAIN',samples=len(features),steps=200,ridge=0.001,lr=0.01,
                targets='0.5 CE + 0.5 original TRAIN-only teacher KD, temperature 2',
                nonnegative_gauge_floor=0.25,history=history,hidden_and_backbone_frozen=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", choices=("MNIST", "NMNIST", "CIFAR10DVS", "DVSGesture"))
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=DEFAULT_CONDITIONS)
    parser.add_argument("--datasets", nargs="+", choices=("MNIST", "NMNIST", "CIFAR10DVS", "DVSGesture"))
    parser.add_argument("--stdp-lr", type=float, default=0.001)
    parser.add_argument("--bp-rate-scale", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, help="Train complete TRAIN epochs instead of short blocks")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--skip-test", action="store_true", help="Freeze VALID choices without reading TEST")
    parser.add_argument("--freeze-bn", action="store_true", help="Keep pretrained backbone BatchNorm statistics")
    parser.add_argument("--freeze-backbone", action="store_true", help="Only hidden/head learning; preserve convolution parameters")
    parser.add_argument("--from-scratch", action="store_true", help="Initialize the same compact model with new random weights")
    parser.add_argument("--teacher-source", choices=("compressed", "original_cached"), default="compressed")
    parser.add_argument("--stdp-granularity", choices=("neuron", "group"))
    parser.add_argument("--initial-row-sum", type=float)
    parser.add_argument("--augment-spatial", action="store_true", help="CIFAR TRAIN-only horizontal flips and shifts")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--freeze-bp-hidden", action="store_true")
    parser.add_argument("--steps", type=int, help="Explicit temporal condition; event steps must divide prepared cache length")
    parser.add_argument("--calibrate-bn", action="store_true", help="Reestimate BN statistics on full native TRAIN before VALID; keep weights and homeostasis frozen")
    parser.add_argument("--spatial-student", action="store_true", help="CIFAR-only 4.679M preset: last channel 448, readout 4x4, hidden 256")
    parser.add_argument("--fit-classifier", action="store_true", help="Fit physical nonnegative classifier on frozen full TRAIN features first")
    parser.add_argument("--minimum-initial-valid", type=float, default=0, help="Skip full-network recovery when fitted frozen features miss this VALID floor")
    parser.add_argument('--initial-checkpoint',type=Path,help='Warm-start a compatible trained student; optimizer starts fresh')
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--batches-per-block", type=int, default=16)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Keep completed conditions; archive an incomplete attempt")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if not 1 <= args.blocks <= 4 or not 1 <= args.batches_per_block <= 64:
        parser.error("Fixed screen budget must be 1..4 blocks of 1..64 batches")
    if args.epochs is not None and not 1 <= args.epochs <= 60:
        parser.error("Full-training budget must be 1..60 epochs")
    if not 0 <= args.stdp_lr <= 1 or not 0 < args.bp_rate_scale <= 2 or args.patience < 1:
        parser.error("Invalid learning rate or early-stop patience")
    if args.steps is not None and args.steps < 1:
        parser.error("Time steps must be positive")
    if not 0 <= args.minimum_initial_valid <= 1 or (args.fit_classifier and args.teacher_source != 'original_cached'):
        parser.error("Head fitting requires original TRAIN cache; initial VALID floor must lie in [0,1]")
    if args.initial_checkpoint is not None:
        args.initial_checkpoint=args.initial_checkpoint.resolve()
        if args.from_scratch or not args.initial_checkpoint.is_file():
            parser.error('Initial checkpoint must exist and cannot be combined with from-scratch')
    args.out = args.out.resolve()
    manifest = json.loads((BUNDLE / "training.json").read_text())
    if args.datasets:
        manifest["entries"] = [e for e in manifest["entries"] if e["dataset"] in args.datasets]
    if args.spatial_student and (args.dataset != 'CIFAR10DVS' if args.dataset else [e['dataset'] for e in manifest['entries']] != ['CIFAR10DVS']):
        parser.error("Spatial-student preset is registered only for CIFAR10DVS")
    if not args.dataset:
        if args.condition or args.test_only:
            parser.error("condition/test-only requires dataset")
        assert (args.out / "dependency_audit.json").exists(), "Run audit.py before training"
        assert all((args.out / f"baseline_replay_{e['dataset']}.json").exists() for e in manifest["entries"])
        runtime = args.out / "runtime"
        runtime.mkdir(exist_ok=args.resume)
        for entry in manifest["entries"]:
            target = runtime / entry["dataset"]
            if args.resume and target.exists():
                continue
            shutil.copytree(BUNDLE / entry["source_root"], target,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            (target / "experiments" / "__init__.py").write_text("")
            for name in ("wta.py", "stdp_lut.py"):
                shutil.copy2(ROOT / "snn" / name, target / "snn" / name)
            models = target / "experiments/accuracy/models.py"
            models.write_text(adapt_models_source(models.read_text()))
        previous_protocol = json.loads((args.out / "protocol.json").read_text()) if args.resume else {}
        if args.resume:
            assert args.blocks == previous_protocol["blocks"] and args.batches_per_block == previous_protocol["batches_per_block"]
            assert args.epochs == previous_protocol.get("epochs")
            assert args.stdp_lr == previous_protocol["stdp_lr"]
            assert args.bp_rate_scale == previous_protocol.get("bp_rate_scale", 0.1)
            assert list(args.conditions) == previous_protocol["conditions"]
            assert args.freeze_bn == previous_protocol.get("freeze_bn", False)
            assert args.freeze_backbone == previous_protocol.get("freeze_backbone", False)
            assert args.from_scratch == previous_protocol.get("from_scratch", False)
            assert args.teacher_source == previous_protocol.get("teacher_source", "compressed")
            assert args.stdp_granularity == previous_protocol.get("stdp_granularity")
            assert args.initial_row_sum == previous_protocol.get("initial_row_sum")
            assert args.augment_spatial == previous_protocol.get("augment_spatial", False)
            assert args.workers == previous_protocol.get("workers", 0)
            assert args.freeze_bp_hidden == previous_protocol.get("freeze_bp_hidden", False)
            assert args.steps == previous_protocol.get("steps_override")
            assert args.calibrate_bn == previous_protocol.get("calibrate_bn", False)
            assert args.spatial_student == previous_protocol.get("spatial_student", False)
            assert args.fit_classifier == previous_protocol.get('fit_classifier', False)
            assert args.minimum_initial_valid == previous_protocol.get('minimum_initial_valid',0)
            assert (str(args.initial_checkpoint) if args.initial_checkpoint else None) == previous_protocol.get('initial_checkpoint')
            for name, digest in previous_protocol["source_sha256"].items():
                assert sha(args.out / name) == digest
        driver_history = previous_protocol.get("driver_history", [])
        if args.resume and sha(args.out / "run_frozen.py") != sha(Path(__file__)):
            old_driver = args.out / f"run_frozen_attempt{len(driver_history)+1}.py"
            shutil.copy2(args.out / "run_frozen.py", old_driver)
            driver_history.append({"path": old_driver.name, "sha256": sha(old_driver),
                "reason": "Support historical compression RATES provenance; preserve completed VALID conditions"})
        shutil.copy2(Path(__file__), args.out / "run_frozen.py")
        protocol = {"baseline_bundle": str(BUNDLE), "baseline_manifest_sha256": sha(BUNDLE / "training.json"),
            "datasets": [e["dataset"] for e in manifest["entries"]], "conditions": args.conditions,
            "seed": 42, "blocks": args.blocks, "batches_per_block": args.batches_per_block,
            "epochs": args.epochs, "patience": args.patience, "bp_rate_scale": args.bp_rate_scale,
            "freeze_bn": args.freeze_bn, "freeze_backbone": args.freeze_backbone,
            "from_scratch": args.from_scratch, "teacher_source": args.teacher_source,
            "stdp_granularity": args.stdp_granularity, "initial_row_sum": args.initial_row_sum,
            "augment_spatial": args.augment_spatial,
            "workers": args.workers, "freeze_bp_hidden": args.freeze_bp_hidden,
            "steps_override": args.steps,
            "calibrate_bn": args.calibrate_bn,
            "spatial_student": args.spatial_student,
            "fit_classifier": args.fit_classifier, "minimum_initial_valid": args.minimum_initial_valid,
            'initial_checkpoint':str(args.initial_checkpoint) if args.initial_checkpoint else None,
            'initial_checkpoint_sha256':sha(args.initial_checkpoint) if args.initial_checkpoint else None,
            "validation": "complete VALID at initialization and after each fixed TRAIN block; accuracy then loss",
            "selection": "select each mode independently by VALID; evaluate all fixed states, never rank TEST",
            "test": "complete TEST FP32/uint2/uint4 only after all VALID selections",
            "objective": f"0.5 CE + 0.5 KL*T^2, T=2; teacher source={args.teacher_source}, TRAIN only",
            "initialization": "random compact model" if args.from_scratch else "structured spatial transfer from compressed checkpoint" if args.spatial_student else "final compressed checkpoint; Adam state starts fresh, not training continuation",
            "bp_lr": f"{args.bp_rate_scale} * actual compression recovery backbone rate; hidden BP 0.5x, head 5x, no weight decay",
            "stdp_lr": args.stdp_lr, "stdp_update_rule": "soft_bounds_rownorm", "input_event_amplitude": 1.0,
            "input_encoding": "native forward for legacy_native/lut_native_trace; lut_native_trace samples pooled activity only for learning, using independent trace RNG",
            "rng": "global CPU/CUDA RNG and independent per-layer trace RNG saved in checkpoints; learning trace sampling does not alter native forward RNG",
            "precision": "FP32 train forward/backward, CuDNN TF32 permitted as in compression; hidden Linear only uint2/uint4 at inference",
            "preserved": "physical size, native input coding/resolution, WTA tiebreak/first-winner, exponential homeostasis, full forward after winner; temporal and role overrides are explicit",
            "early_stop": f"two consecutive complete VALID collapses or {args.patience} rounds without improvement; full epoch mode waits at least min(5,budget) rounds",
            "source_sha256": {str(p.relative_to(args.out)): sha(p) for p in runtime.rglob("*.py")},
            "driver_sha256": sha(args.out / "run_frozen.py"), "driver_history": driver_history, "device": args.device}
        write(args.out / "protocol.json", protocol)
        for entry in manifest["entries"]:
            for condition in args.conditions:
                condition_dir = args.out / entry["dataset"] / condition
                if args.resume and (condition_dir / "validation.json").exists():
                    completed = json.loads((condition_dir / "validation.json").read_text())
                    assert sha(condition_dir / "best.pth") == completed["checkpoint_sha256"]
                    continue
                if args.resume and condition_dir.exists():
                    attempt = args.out / "attempts" / (entry["dataset"]+"_"+condition+"_"+str(time.time_ns()))
                    attempt.parent.mkdir(exist_ok=True)
                    condition_dir.rename(attempt)
                command = [sys.executable, str(args.out / "run_frozen.py"), "--out", str(args.out),
                           "--dataset", entry["dataset"], "--condition", condition, "--device", args.device,
                           "--blocks", str(args.blocks), "--batches-per-block", str(args.batches_per_block)]
                command += ["--stdp-lr", str(args.stdp_lr), "--bp-rate-scale", str(args.bp_rate_scale),
                            "--patience", str(args.patience)]
                if args.epochs is not None:
                    command += ["--epochs", str(args.epochs)]
                if args.freeze_bn:
                    command.append("--freeze-bn")
                if args.freeze_backbone:
                    command.append("--freeze-backbone")
                if args.from_scratch:
                    command.append("--from-scratch")
                command += ["--teacher-source", args.teacher_source]
                if args.stdp_granularity:
                    command += ["--stdp-granularity", args.stdp_granularity]
                if args.initial_row_sum is not None:
                    command += ["--initial-row-sum", str(args.initial_row_sum)]
                if args.augment_spatial:
                    command.append("--augment-spatial")
                command += ["--workers", str(args.workers)]
                if args.freeze_bp_hidden:
                    command.append("--freeze-bp-hidden")
                if args.steps is not None:
                    command += ["--steps", str(args.steps)]
                if args.calibrate_bn:
                    command.append("--calibrate-bn")
                if args.spatial_student:
                    command.append("--spatial-student")
                if args.fit_classifier:
                    command.append('--fit-classifier')
                command += ['--minimum-initial-valid',str(args.minimum_initial_valid)]
                if args.initial_checkpoint:
                    command+=['--initial-checkpoint',str(args.initial_checkpoint)]
                write(args.out / "status.json", {"at": datetime.now(timezone.utc).isoformat(),
                    "phase": "TRAIN_VALID", "dataset": entry["dataset"], "condition": condition})
                subprocess.run(command, check=True)
        selections = {e["dataset"]: {c: json.loads((args.out / e["dataset"] / c / "validation.json").read_text())
                       for c in args.conditions} for e in manifest["entries"]}
        write(args.out / "selections_before_TEST.json", selections)
        if args.skip_test:
            write(args.out / "status.json", {"at": datetime.now(timezone.utc).isoformat(),
                  "phase": "VALID_complete_TEST_deferred", "conditions_completed": sum(len(v) for v in selections.values())})
            return
        for entry in manifest["entries"]:
            for condition in args.conditions:
                if args.resume and (args.out / entry["dataset"] / condition / "accuracy.json").exists():
                    continue
                write(args.out / "status.json", {"at": datetime.now(timezone.utc).isoformat(),
                    "phase": "FULL_TEST", "dataset": entry["dataset"], "condition": condition})
                subprocess.run([sys.executable, str(args.out / "run_frozen.py"), "--out", str(args.out),
                    "--dataset", entry["dataset"], "--condition", condition, "--device", args.device,
                    "--test-only"], check=True)
        results = {e["dataset"]: {c: json.loads((args.out / e["dataset"] / c / "accuracy.json").read_text())
                   for c in args.conditions} for e in manifest["entries"]}
        assert all(sha(ROOT / e["checkpoint"]) == e["checkpoint_sha256"] for e in manifest["entries"])
        write(args.out / "results.json", results)
        write(args.out / "status.json", {"at": datetime.now(timezone.utc).isoformat(), "phase": "completed",
              "conditions_completed": len(results)*len(args.conditions),
              "full_test_modes_completed": len(results)*len(args.conditions)*3, "original_bundle_unchanged": True})
        return
    if not args.condition:
        parser.error("dataset requires condition")
    import torch
    from torch.nn import functional as F
    entry = next(e for e in manifest["entries"] if e["dataset"] == args.dataset)
    source = args.out / "runtime" / args.dataset
    protocol = json.loads((args.out / "protocol.json").read_text())
    for name, digest in protocol["source_sha256"].items():
        if name.startswith(f"runtime/{args.dataset}/"):
            assert sha(args.out / name) == digest
    sys.path.insert(0, str(source))
    from experiments.accuracy.models import AccuracySNN
    from experiments.accuracy.data import build_loaders, prepare_batch
    from experiments.accuracy.config import DATASETS
    from experiments.accuracy.run import evaluate, evaluate_precisions, seed_everything, rng_state, restore_rng
    from snn.wta import WinnerOptimizerGuard
    from snn.stdp_lut import stdp_options
    for name in ("experiments.accuracy.models", "experiments.accuracy.data", "experiments.accuracy.run", "snn.wta", "snn.stdp_lut"):
        assert Path(importlib.import_module(name).__file__).resolve().is_relative_to(source)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use explicit --device cpu for a separately recorded condition")
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    out = args.out / args.dataset / args.condition
    original_cfg = json.loads((ROOT / entry["config"]).read_text())
    original_cfg["data_dir"] = os.environ.get("SNN_DATA_DIR", str(ROOT / "datasets"))
    cfg = copy.deepcopy(original_cfg)
    cfg.update(stdp_mode="lut" if args.condition.startswith("lut_") else "legacy_continuous",
               stdp_input_encoding=("bernoulli_trace" if args.condition == "lut_native_trace" else
                                    "native" if args.condition == "legacy_native" else
                                    "binary" if args.dataset == "MNIST" else "bernoulli"),
               stdp_lr=args.stdp_lr, stdp_trace_seed=cfg["seed"]+7000,
               stdp_update_rule="soft_bounds_rownorm", stdp_record_stats=True,
               stdp_event_amplitude=1.0, stdp_coefficient_lsb=0.001)
    cfg.update(freeze_backbone_bn=args.freeze_bn, freeze_backbone_parameters=args.freeze_backbone)
    cfg["stdp_training_spatial_augmentation"] = args.augment_spatial
    cfg["stdp_calibrate_bn_on_train"] = args.calibrate_bn
    cfg['stdp_spatial_student'] = args.spatial_student
    cfg['stdp_fit_classifier_on_train'] = args.fit_classifier
    if args.spatial_student:
        cfg.update(custom_channels=[32, 64, 128, 256, 448], hidden_widths=[256])
        cfg['dataset_profiles'][args.dataset]['pool_size'] = 4
    if args.steps is not None:
        if args.dataset != "MNIST" and DATASETS[args.dataset].steps % args.steps:
            parser.error("Event time steps must divide the existing cache length")
        cfg.setdefault("dataset_profiles", {}).setdefault(args.dataset, {})["steps"] = args.steps
    if args.augment_spatial and args.dataset != "CIFAR10DVS":
        parser.error("Spatial augmentation is registered only for CIFAR10DVS")
    if args.stdp_granularity:
        cfg["hybrid_stdp_granularity"] = args.stdp_granularity
    if args.initial_row_sum is not None:
        if args.initial_row_sum <= 0:
            parser.error("Initial hidden row sum must be positive")
        cfg["hidden_initial_row_sum"] = args.initial_row_sum
    seed_everything(cfg["seed"])
    if args.test_only:
        assert (args.out / "selections_before_TEST.json").exists()
        cfg = json.loads((out / "config.json").read_text())
    else:
        out.mkdir(parents=True, exist_ok=False)
        write(out / "config.json", cfg)
    # Execution workers differ from the historical field but do not change samples.
    loader_cfg = copy.deepcopy(cfg)
    loader_cfg["num_workers"] = args.workers
    loaders = build_loaders(args.dataset, loader_cfg)
    if args.epochs is not None:
        args.blocks, args.batches_per_block = args.epochs, len(loaders[0])
    assert sha(loaders[3]) == entry["split_sha256"]
    model = AccuracySNN("custom", args.dataset, cfg, "exponential").to(device)
    if args.stdp_granularity:
        assert all(h.hybrid_stdp_granularity == args.stdp_granularity for h in model.hidden_layers.values()), "Model ignored requested STDP role granularity"
    parameter_total = sum(p.numel() for p in model.parameters())
    if cfg.get('stdp_spatial_student', False):
        assert parameter_total == 4679375
    else:
        assert parameter_total == entry["parameters"]["total"]
    assert 3_000_000 <= parameter_total <= 5_000_000
    expected = {"validation": entry["validation"]}
    started = time.monotonic()

    def emit(event, **values):
        record = {"at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic()-started,
                  "dataset": args.dataset, "condition": args.condition, "event": event, **values}
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        write(out / "status.json", record)
        print(json.dumps(record, allow_nan=False), flush=True)

    if args.test_only:
        selection = json.loads((out / "validation.json").read_text())
        assert sha(out / "best.pth") == selection["checkpoint_sha256"]
        saved = torch.load(out / "best.pth", map_location=device, weights_only=True)
        model.load_state_dict(saved["model"], strict=True)
        before = snapshot_state(model)
        seed_everything(entry["test_rng_seed"])
        scores = evaluate_precisions(model, loaders[2], args.dataset, device)
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()), "Inference changed training state"
        assert all(s["samples"] == len(loaders[2].dataset) for s in scores.values())
        assert sha(out / "best.pth") == selection["checkpoint_sha256"]
        record = {"dataset": args.dataset, "condition": args.condition, "parameters": parameter_total,
                  "selection": selection, "test": scores, "test_rng_seed": entry["test_rng_seed"],
                  "inference_state_unchanged": True, "original_checkpoint_sha256": entry["checkpoint_sha256"],
                  "evidence": "single seed, complete TEST; initialization and actual training budget are recorded in provenance.json",
                  "training_provenance": json.loads((out / "provenance.json").read_text())}
        write(out / "accuracy.json", record)
        emit("full_test_completed", correct={k: v["correct"] for k, v in scores.items()},
             samples=len(loaders[2].dataset), selected_block=selection["selected_block"])
        return
    saved = torch.load(ROOT / entry["checkpoint"], map_location=device, weights_only=True)
    structural_transfer = None
    inherited_lut_training=False
    if not args.from_scratch:
        if args.initial_checkpoint:
            assert sha(args.initial_checkpoint)==protocol['initial_checkpoint_sha256']
            initial=torch.load(args.initial_checkpoint,map_location=device,weights_only=True)
            model.load_state_dict(initial['model'],strict=True)
            for name,hidden in model.hidden_layers.items():
                hidden.restore_trace_rng(initial.get('trace_rng',{}).get(name))
            inherited_lut_training=initial.get('block',0)>0 and any(v is not None for v in initial.get('trace_rng',{}).values())
            del initial
        elif args.spatial_student:
            structural_transfer = spatial_student_transfer(model, saved['model'])
        else:
            model.load_state_dict(saved["model"], strict=True)
        if args.stdp_granularity:
            # Reassigned formerly BP rows must retain their learned scale when
            # they become normalized STDP rows; no forward weights are changed.
            for hidden in model.hidden_layers.values():
                hidden.reset_row_sum_target()
    calibration = None
    if args.calibrate_bn:
        # Native TRAIN only: no dropout, WTA/homeostasis changes, synaptic
        # updates, teacher targets or VALID/TEST samples. The private loader
        # leaves the training stream untouched. Cumulative moments avoid
        # replacing old statistics with the last class/minibatch only.
        from torch.utils.data import DataLoader
        modules = {name: module for name,module in model.backbone.named_modules()
                   if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)}
        assert modules, "No backbone BatchNorm to calibrate"
        before, global_rng = snapshot_state(model), rng_state()
        momentum = {name:module.momentum for name,module in modules.items()}
        model.eval()
        calibration_loader = DataLoader(loaders[0].dataset, batch_size=loaders[0].batch_size,
            shuffle=True, num_workers=args.workers,
            generator=torch.Generator().manual_seed(cfg["seed"]+6000))
        for module in modules.values():
            module.reset_running_stats()
            module.momentum = None
            module.train()
        calibration_samples = 0
        try:
            with torch.no_grad():
                for data, labels in calibration_loader:
                    model.reset_membranes()
                    model(prepare_batch(data, args.dataset, device, cfg=cfg))
                    calibration_samples += len(labels)
        finally:
            for name,module in modules.items():
                module.momentum = momentum[name]
            model.eval()
            model.reset_membranes()
            restore_rng(global_rng)
        changed = {f"backbone.{name}.{field}" for name in modules
                   for field in ("running_mean", "running_var", "num_batches_tracked")}
        assert all(torch.equal(value, model.state_dict()[key]) for key,value in before.items() if key not in changed), "BN calibration changed protected state"
        assert calibration_samples == len(loaders[0].dataset)
        calibration = dict(samples=calibration_samples, batches=len(calibration_loader), seed=cfg["seed"]+6000,
                           statistic="cumulative mean of batch moments", partition="TRAIN", augmentation=False,
                           weights_and_homeostasis_unchanged=True, bn_layers=list(modules))
        emit("TRAIN_only_BN_calibration_complete", calibration=calibration)
    teacher, teacher_cache = None, None
    teacher_evidence = {"source": args.teacher_source}
    if args.teacher_source == "compressed":
        teacher = AccuracySNN("custom", args.dataset, original_cfg, "exponential").to(device)
        teacher.load_state_dict(saved["model"], strict=True)
        teacher.eval().requires_grad_(False)
        teacher_evidence.update(checkpoint=str(ROOT / entry["checkpoint"]), checkpoint_sha256=entry["checkpoint_sha256"])
    else:
        old_manifest = json.loads((ROOT / "configs/benchmarks.json").read_text())
        old_entry = next(e for e in old_manifest["entries"]
                         if e["bundle"] == "original" and e["dataset"] == args.dataset)
        old_entry["checkpoint"] = str(ROOT / old_entry["checkpoint"])
        assert sha(old_entry["checkpoint"]) == old_entry["checkpoint_sha256"]
        cache_path = Path(os.environ.get("SNN_TEACHER_CACHE_DIR", str(ROOT / "results/new_runs/teacher_cache"))) / args.dataset / "teacher_train_logits.pth"
        teacher_cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        assert teacher_cache["teacher_sha256"] == old_entry["checkpoint_sha256"]
        assert teacher_cache["split_sha256"] == entry["split_sha256"]
        assert len(teacher_cache["labels"]) == len(loaders[0].dataset)
        teacher_cache["logits"] = teacher_cache["logits"].to(device)
        from torch.utils.data import Dataset, DataLoader
        class IndexedFrames(Dataset):
            def __len__(self):
                return len(loaders[0].dataset)
            def __getitem__(self, index):
                data, label = loaders[0].dataset[index]
                return data, label, index
        # Preserve sample order and use explicit indices for the immutable TRAIN cache.
        training = DataLoader(IndexedFrames(), batch_size=loaders[0].batch_size, shuffle=True,
                              num_workers=args.workers, generator=loaders[0].generator,
                              pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
        teacher_evidence.update(checkpoint=old_entry["checkpoint"], checkpoint_sha256=old_entry["checkpoint_sha256"],
                                cache=str(cache_path), cache_sha256=sha(cache_path), partition="TRAIN")
    head_bootstrap = None
    if args.fit_classifier:
        before, global_rng = snapshot_state(model), rng_state()
        model.eval()
        features = torch.empty(len(training.dataset),model.classifier.in_features,device=device)
        labels_all = torch.empty(len(training.dataset),dtype=torch.long,device=device)
        current = {}
        def capture(module, inputs):
            current['features'] = inputs[0].detach().mean(0)
        handle = model.classifier.register_forward_pre_hook(capture)
        feature_loader = DataLoader(training.dataset,batch_size=training.batch_size,shuffle=False,
            num_workers=args.workers,generator=torch.Generator().manual_seed(cfg['seed']+6500))
        try:
            with torch.no_grad():
                for data,labels,indices in feature_loader:
                    assert torch.equal(teacher_cache['labels'][indices],labels)
                    model.reset_membranes()
                    model(prepare_batch(data,args.dataset,device,cfg=cfg))
                    features[indices.to(device)] = current['features']
                    labels_all[indices.to(device)] = labels.to(device)
        finally:
            handle.remove()
            model.reset_membranes()
            restore_rng(global_rng)
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()), 'Feature extraction changed model state'
        head_bootstrap = fit_nonnegative_classifier(model.classifier,features,labels_all,teacher_cache['logits'],emit)
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items() if not k.startswith('classifier.')), 'Head fitting changed protected state'
        del features, labels_all
    if args.freeze_backbone:
        model.backbone.requires_grad_(False)
    if args.freeze_bp_hidden:
        for hidden_layer in model.hidden_layers.values():
            hidden_layer.weight.requires_grad_(False)
            hidden_layer.bias.requires_grad_(False)
    del saved
    compression_provenance = entry["optimizer"]
    recovery_rate, rate_source = compression_rate(args.dataset, compression_provenance,
        BUNDLE / f"conditions/{args.dataset}/compression_runtime/run.py")
    rate = recovery_rate * args.bp_rate_scale
    backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    hidden = [p for h in model.hidden_layers.values() if h.rule != "stdp" for p in (h.weight, h.bias) if p.requires_grad]
    optimizer = torch.optim.Adam([{"params": backbone, "lr": rate}, {"params": hidden, "lr": rate*0.5},
                                  {"params": model.classifier.parameters(), "lr": rate*5}], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.blocks, eta_min=1e-6) if args.epochs else None
    source_hashes = {str(p.relative_to(source)): sha(p) for p in source.rglob("*.py")}
    write(out / "provenance.json", {"original_checkpoint": str(ROOT / entry["checkpoint"]),
        "original_checkpoint_sha256": entry["checkpoint_sha256"], "baseline_parameters": entry["parameters"],
        "parameters": {"total": parameter_total} if args.spatial_student else entry['parameters'],
        "source_root": str(source), "source_sha256": source_hashes, "config_sha256": sha(out / "config.json"),
        "split_sha256": entry["split_sha256"], "bp_learning_rate": rate, "stdp_options": stdp_options(cfg),
        "compression_rate_source": rate_source, "driver_sha256": sha(Path(__file__)),
        "lut": next(h.lut_coefficients.tolist() for h in model.hidden_layers.values()) if cfg["stdp_mode"] == "lut" else None,
        "device": args.device, "torch_version": str(torch.__version__), "seed": cfg["seed"],
        "training_budget": {"blocks": args.blocks, "batches_per_block": args.batches_per_block,
                            "complete_train_epochs": args.epochs},
        "execution_num_workers": args.workers, "original_config_recovery_fields_are_not_current_training_budget": True})
    provenance = json.loads((out / "provenance.json").read_text())
    provenance.update(teacher=teacher_evidence, initialization="random" if args.from_scratch else "structured_spatial_transfer" if args.spatial_student else "compressed_checkpoint",
                      freeze_bn=args.freeze_bn, freeze_backbone=args.freeze_backbone,
                      stdp_granularity=cfg.get("hybrid_stdp_granularity", "neuron"),
                      hidden_initial_row_sum=cfg.get("hidden_initial_row_sum"), freeze_bp_hidden=args.freeze_bp_hidden)
    provenance["bn_calibration"] = calibration
    provenance['structural_transfer'] = structural_transfer
    provenance['head_bootstrap'] = head_bootstrap
    provenance['initial_checkpoint'] = str(args.initial_checkpoint) if args.initial_checkpoint else None
    provenance['initial_checkpoint_sha256'] = sha(args.initial_checkpoint) if args.initial_checkpoint else None
    provenance['inherited_lut_training'] = inherited_lut_training
    if args.initial_checkpoint:
        provenance['initialization']='trained_student_warm_start_fresh_optimizer'
    provenance["actual_hidden_roles"] = {name: {"granularity": h.hybrid_stdp_granularity,
        "stdp_neurons": int(h.stdp_neuron_mask.sum()), "bp_neurons": int(h.bp_neuron_mask.sum()),
        "competition_size": h.competition_size, "stdp_neuron_indices": h.stdp_neuron_mask.nonzero().flatten().tolist()}
        for name,h in model.hidden_layers.items()}
    write(out / "provenance.json", provenance)

    def validation():
        rng = rng_state()
        before = snapshot_state(model)
        try:
            seed_everything(cfg["seed"] + 2000)
            score = evaluate(model, loaders[1], args.dataset, device)
        finally:
            restore_rng(rng)
        assert score["samples"] == len(loaders[1].dataset)
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
        return score

    best = validation()
    best_block = 0
    same_time_condition = cfg.get("dataset_profiles", {}).get(args.dataset, {}).get("steps", DATASETS[args.dataset].steps) == original_cfg.get("dataset_profiles", {}).get(args.dataset, {}).get("steps", DATASETS[args.dataset].steps)
    if args.condition in ("legacy_native", "lut_native_trace") and not args.from_scratch and same_time_condition and not args.calibrate_bn and not args.spatial_student and not args.fit_classifier and not args.initial_checkpoint:
        assert best == expected["validation"], "New legacy runtime differs from frozen complete VALID"
    def save_best(block, score):
        model.reset_membranes()
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "block": block,
            "validation": score, "rng": rng_state(), "train_loader_rng": loaders[0].generator.get_state(),
            "trace_rng": {name: h.trace_rng_state() for name, h in model.hidden_layers.items()}}, out / "best.pth")
    save_best(0, best)
    emit("initial_complete_validation", validation=best)
    history = [{"block": 0, "validation": best}]
    totals = {name: {"winner_events": 0, "nonzero_weight_writes": 0, "raw_delta_nonzero": 0,
                    "raw_delta_abs_max": 0.0, "actual_delta_abs_max": 0.0,
                    "winner_row_histogram": [0]*h.out_features, "winner_trace_histogram": [0]*32}
              for name, h in model.hidden_layers.items()}
    no_improvement = collapse = 0
    stop = "fixed_budget_completed"
    batches = samples = 0
    # Explicit reseed makes paired legacy-events/LUT initialization evaluation
    # independent of their training stream. Loader uses its own seeded generator.
    seed_everything(cfg["seed"] + 3000)
    training = loaders[0] if teacher_cache is None else training
    train_iter = None
    blocks_to_train = args.blocks
    if best['top1_accuracy'] < args.minimum_initial_valid:
        blocks_to_train, stop = 0, 'insufficient_frozen_feature_VALID_after_head_fit'
    for block in range(1, blocks_to_train+1):
        if train_iter is None or args.epochs is not None:
            train_iter = iter(training)
        model.train()
        if args.freeze_bn or args.freeze_backbone:
            for module in model.backbone.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
        loss_sum = correct = count = 0
        for offset in range(args.batches_per_block):
            try:
                item = next(train_iter)
            except StopIteration:
                train_iter = iter(training)
                item = next(train_iter)
            data, labels = item[:2]
            frames = prepare_batch(data, args.dataset, device, cfg=cfg)
            if args.augment_spatial:
                # Identical transforms across all time bins and polarities.
                flip = torch.rand((len(labels), 1, 1, 1), device=device) < 0.5
                frames = torch.where(flip.unsqueeze(0), frames.flip(-1), frames)
                dy, dx = torch.randint(-4, 5, (2,)).tolist()
                height, width = frames.shape[-2:]
                frames = F.pad(frames, (4, 4, 4, 4))[..., 4+dy:4+dy+height, 4+dx:4+dx+width]
            labels = labels.to(device)
            if teacher is not None:
                teacher.reset_membranes()
                with torch.no_grad():
                    target = teacher(frames)
            else:
                indices = item[2]
                assert torch.equal(teacher_cache["labels"][indices], labels.cpu()), "Teacher cache/sample mismatch"
                target = teacher_cache["logits"][indices.to(device)]
            model.reset_membranes()
            optimizer.zero_grad(set_to_none=True)
            logits = model(frames)
            ce = F.cross_entropy(logits, labels)
            kd = F.kl_div(F.log_softmax(logits/2, dim=1), F.softmax(target/2, dim=1), reduction="batchmean")*4
            loss = 0.5*ce + 0.5*kd
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite bounded-training loss")
            loss.backward()
            gradients = [p for p in model.parameters() if p.grad is not None]
            assert gradients and all(p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all() for p in gradients)
            norm = torch.nn.utils.clip_grad_norm_(model.bp_parameters(), 1.0)
            before = {name: h.weight.detach().clone() for name, h in model.hidden_layers.items()}
            winners = {}
            for name, h in model.hidden_layers.items():
                if h.learning_context is not None:
                    post = h.learning_context[1]
                    hist = post.sum((0, 1)).long().tolist()
                    totals[name]["winner_row_histogram"] = [x+y for x, y in zip(totals[name]["winner_row_histogram"], hist)]
                winners[name] = {"eligible_rows": int(h.eligible_rows.sum()),
                                 "eligible_stdp_rows": int((h.eligible_rows & h.stdp_neuron_mask).sum())}
            with WinnerOptimizerGuard(model.hidden_layers.values(), optimizer):
                optimizer.step()
            for name, h in model.hidden_layers.items():
                protected = ~(h.eligible_rows & h.bp_neuron_mask)
                assert torch.equal(h.weight[protected], before[name][protected]), "BP changed protected rows"
                before_stdp = h.weight.detach().clone()
                h.apply_stdp()
                protected = ~(h.eligible_rows & h.stdp_neuron_mask)
                assert torch.equal(h.weight[protected], before_stdp[protected]), "STDP changed protected rows"
                if h.stdp_statistics is not None:
                    s = h.stdp_statistics
                    for k in ("winner_events", "nonzero_weight_writes", "raw_delta_nonzero"):
                        totals[name][k] += s[k]
                    for k in ("raw_delta_abs_max", "actual_delta_abs_max"):
                        totals[name][k] = max(totals[name][k], s.get(k, 0.0))
                    totals[name]["winner_trace_histogram"] = [x+y for x,y in zip(totals[name]["winner_trace_histogram"], s.get("winner_trace_histogram", [0]*32))]
            model.project_nonnegative()
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) or hasattr(module, "stdp_mode"):
                    assert module.weight.dtype == torch.float32 and torch.isfinite(module.weight).all() and module.weight.min() >= 0, name
            batches += 1
            samples += len(labels)
            count += len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            loss_sum += float(loss)*len(labels)
            if offset == 0 or (offset+1) % 4 == 0:
                emit("train_progress", block=block, batch=batches, samples=samples,
                    loss=loss_sum/count, accuracy=correct/count, ce=float(ce), kd=float(kd),
                    gradient_norm=float(norm), winners=winners,
                    stdp={name: h.stdp_statistics for name,h in model.hidden_layers.items()})
        score = validation()
        history.append({"block": block, "validation": score})
        improved = (score["top1_accuracy"], -score["loss"]) > (best["top1_accuracy"], -best["loss"])
        if improved:
            best, best_block = score, block
            save_best(block, score)
            no_improvement = 0
        else:
            no_improvement += 1
        collapsed = score["dominant_prediction_fraction"] >= 0.99 and score["top1_accuracy"] <= score["majority_class_baseline"] + 0.02
        collapse = collapse+1 if collapsed else 0
        emit("block_complete_validation", block=block, validation=score, improved=improved, best_block=best_block)
        if scheduler:
            scheduler.step()
        model.reset_membranes()
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "block": block,
            "validation": score, "rng": rng_state(), "train_loader_rng": loaders[0].generator.get_state(),
            "trace_rng": {name: h.trace_rng_state() for name,h in model.hidden_layers.items()},
            "scheduler": scheduler.state_dict() if scheduler else None}, out / "last.pth")
        if collapse >= 2:
            stop = "sustained_validation_collapse"
            break
        if no_improvement >= args.patience and block >= (min(5, args.blocks) if args.epochs else 1):
            stop = "bounded_validation_no_improvement"
            break
    selected = {"dataset": args.dataset, "condition": args.condition, "selection": "complete VALID accuracy then loss",
        "selected_block": best_block, "validation": best, "history": history, "test_evaluated": False,
        "checkpoint_sha256": sha(out / "best.pth"), "batches_trained": batches, "samples_trained": samples,
        "stop_reason": stop, "training_statistics": totals, "all_bp_stdp_protection_assertions_passed": True}
    write(out / "validation.json", selected)
    emit("validation_selection_frozen", selected_block=best_block, validation=best, batches=batches,
         checkpoint_sha256=selected["checkpoint_sha256"], stop_reason=stop,
         writes={name: value["nonzero_weight_writes"] for name,value in totals.items()})


if __name__ == "__main__":
    main()
