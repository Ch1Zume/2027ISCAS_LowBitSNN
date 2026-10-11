"""Prune a frozen SNN, recover it by distillation, and test a validation-selected state.

Run this file directly: each dataset imports its own pinned benchmark runtime.
The old benchmark files and teacher checkpoint are never edited.
"""

import argparse
import copy
import hashlib
import importlib
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from structural import (TrainingStatistics, assert_physical_weights, parameter_counts,
                        prune_from_teacher)


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "results/accuracy/custom_performance_candidates_20261004/selected_benchmark_final_20261005"
WIDTHS = {"MNIST": [512, 512], "NMNIST": [512, 512],
          "CIFAR10DVS": [768], "DVSGesture": [1024]}
EPOCHS = {"MNIST": 12, "NMNIST": 12, "CIFAR10DVS": 30, "DVSGesture": 30}
RATES = {"MNIST": 0.0002, "NMNIST": 0.0002,
         "CIFAR10DVS": 0.0003, "DVSGesture": 0.0002}
FRONT_CHANNELS = {"MNIST": [64, 128, 192, 256, 384], "NMNIST": [64, 128, 192, 256, 384],
                  "CIFAR10DVS": [64, 128, 192, 256, 384], "DVSGesture": [128, 256, 256, 256, 256]}
FRONT_WIDTHS = {"MNIST": [768, 512], "NMNIST": [768, 512],
                "CIFAR10DVS": [1024], "DVSGesture": [512]}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


class IndexedDataset(Dataset):
    def __init__(self, original):
        self.original = original

    def __len__(self):
        return len(self.original)

    def __getitem__(self, index):
        data, label = self.original[index]
        return data, label, index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=WIDTHS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--calibration-samples", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--head-steps", type=int, default=400)
    parser.add_argument("--profile", choices=["half", "front_preserved"])
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--student-init", type=Path)
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 0:
        parser.error("epochs must be nonnegative")
    if args.calibration_samples <= 0 or args.head_steps < 0:
        parser.error("invalid calibration or head-fit budget")
    if args.evaluate_only and not args.resume:
        parser.error("independent evaluation requires --resume")
    if args.learning_rate is not None and args.learning_rate <= 0:
        parser.error("learning rate must be positive")
    if args.out.exists() and not args.resume:
        parser.error("output exists; use --resume or a new output path")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((BENCHMARK / "benchmark_manifest.json").read_text())
    entry = next(row for row in manifest["entries"] if row["dataset"] == args.dataset)
    source = Path(entry["source_root"])
    if sha(entry["checkpoint"]) != entry["checkpoint_sha256"]:
        raise AssertionError("Frozen teacher checkpoint digest differs")
    for name, digest in entry["source_sha256"].items():
        if sha(source / name) != digest:
            raise AssertionError(f"Frozen teacher source differs: {name}")
    sys.path.insert(0, str(source))
    from experiments.accuracy.models import AccuracySNN
    from experiments.accuracy.data import build_loaders, build_probe_loader, prepare_batch
    from experiments.accuracy.run import (calibrate_backbone_normalization, evaluate,
                                          evaluate_precisions, rng_state, restore_rng, seed_everything)
    from experiments.accuracy.monitoring import TrainingAborted, TrainingMonitor, monitoring_config
    from snn.wta import WinnerOptimizerGuard

    imported = {}
    for name in ("experiments.accuracy.models", "experiments.accuracy.data",
                 "experiments.accuracy.run", "snn.wta", "snn.homeostasis", "snn.inference_quant"):
        location = Path(importlib.import_module(name).__file__).resolve()
        if not location.is_relative_to(source):
            raise AssertionError(f"Imported unpinned runtime: {name}: {location}")
        imported[name] = str(location)
    if not torch.cuda.is_available():
        raise RuntimeError("Compression recovery requires the available CUDA host")
    device = torch.device("cuda")
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    teacher_cfg = json.loads(Path(entry["config_snapshot"]).read_text())
    seed_everything(teacher_cfg["seed"])
    previous_provenance = (json.loads((args.out / "provenance.json").read_text()) if args.resume else {})
    profile = args.profile or previous_provenance.get("compression_profile", "half")
    learning_rate = args.learning_rate or previous_provenance.get("backbone_learning_rate", RATES[args.dataset])
    cfg = copy.deepcopy(teacher_cfg)
    cfg["custom_channels"] = (FRONT_CHANNELS[args.dataset] if profile == "front_preserved"
                              else [width // 2 for width in teacher_cfg["custom_channels"]])
    cfg["hidden_widths"] = FRONT_WIDTHS[args.dataset] if profile == "front_preserved" else WIDTHS[args.dataset]
    cfg["hidden_dropout"] = min(0.2, cfg.get("hidden_dropout", 0))
    cfg["out_dir"] = str(args.out.resolve())
    cfg["num_workers"] = 4
    cfg["freeze_backbone_parameters"] = False
    cfg["classifier_only_phase"] = False
    cfg["freeze_backbone_bn"] = False
    cfg["augmentation"] = {"enabled": False}
    if args.resume:
        cfg["dataset_profiles"][args.dataset] = previous_provenance["input_profile"]
    if args.steps is not None:
        native_steps = 4 if args.dataset == "MNIST" else 16
        if args.steps <= 0 or native_steps % args.steps:
            parser.error("steps must be a positive divisor of the native cache length")
        cfg["dataset_profiles"][args.dataset]["steps"] = args.steps
    epochs = args.epochs if args.epochs is not None else previous_provenance.get("epochs_budget", EPOCHS[args.dataset])
    monitor = TrainingMonitor(args.out, monitoring_config({
        "log_every_batches": 20, "log_every_seconds": 30.0,
        "probe_every_batches": 100, "warmup_batches": 40,
        "collapse_warmup_epochs": 1, "early_stopping_min_epochs": 5,
        "early_stopping_patience": 6, "min_accuracy_improvement": 0.001}))

    def emit(event, **values):
        record = monitor.emit(event, dataset=args.dataset, **values)
        write_json(args.out / "status.json", record)

    if args.resume:
        stored_cfg = json.loads((args.out / "config.json").read_text())
        if stored_cfg != cfg:
            raise AssertionError("Resume configuration differs from the frozen compression profile")
    else:
        write_json(args.out / "config.json", cfg)
    teacher = AccuracySNN("custom", args.dataset, teacher_cfg, "exponential").to(device)
    saved = torch.load(entry["checkpoint"], map_location=device, weights_only=True)
    teacher.load_state_dict(saved["model"], strict=True)
    del saved
    teacher_counts = parameter_counts(teacher)
    teacher.eval().requires_grad_(False)
    student = AccuracySNN("custom", args.dataset, cfg, "exponential").to(device)
    counts = parameter_counts(student)
    if not 3_000_000 <= counts["total"] <= 5_000_000:
        raise AssertionError(f"Dense physical parameter budget failed: {counts}")
    loaders = build_loaders(args.dataset, cfg)
    if sha(loaders[3]) != entry["split_manifest_sha256"]:
        raise AssertionError("Compression changed the frozen train/validation/test split")
    calibration, calibration_manifest = build_probe_loader(loaders[0], args.calibration_samples, cfg["seed"] + 211)
    probe, probe_manifest = build_probe_loader(loaders[1], 256, cfg["seed"] + 212)
    write_json(args.out / "calibration_split.json", calibration_manifest | {"source_partition": "TRAIN"})
    write_json(args.out / "validation_probe.json", probe_manifest)
    provenance = {"dataset": args.dataset, "teacher": entry["checkpoint"],
                  "teacher_sha256": entry["checkpoint_sha256"],
                  "teacher_parameters": teacher_counts, "student_parameters": counts,
                  "compression_ratio": teacher_counts["total"] / counts["total"],
                  "source_root": str(source), "runtime_imports": imported,
                  "source_sha256": entry["source_sha256"],
                  "compression_source_sha256": {name: sha(Path(__file__).parent / name)
                                                for name in ("run.py", "structural.py")},
                  "split_manifest": str(loaders[3]), "split_sha256": sha(loaders[3]),
                  "input_profile": cfg["dataset_profiles"][args.dataset],
                  "parameter_accounting": "all dense parameters including both physical E/I banks and classifier",
                  "protocol": "FP32 learning; nonnegative physical weights; original WTA/BP/STDP; hidden-only inference quantization",
                  "selection": "complete validation accuracy, then validation loss; TEST used only after selection",
                  "epochs_budget": epochs, "temperature": 2.0, "distillation_fraction": 0.5,
                  "teacher_logit_cache": "one fixed TRAIN encoding per sample; no validation/test targets",
                  "training_augmentation": "disabled; native coding and stochastic MNIST Poisson retained",
                  "compression_profile": profile, "backbone_learning_rate": learning_rate,
                  "initialization": "structural compression of trained teacher, not from scratch"}
    if args.student_init is not None:
        provenance["compressed_warm_start"] = {"checkpoint": str(args.student_init.resolve()),
                                                "sha256": sha(args.student_init)}
    if not args.resume:
        snapshot = args.out / "compression_runtime"
        snapshot.mkdir(exist_ok=False)
        for name in ("run.py", "structural.py"):
            shutil.copy2(Path(__file__).parent / name, snapshot / name)
        write_json(args.out / "provenance.json", provenance)
    else:
        provenance = previous_provenance
    emit("compression_started", teacher_parameters=teacher_counts["total"],
         student_parameters=counts["total"], epochs_budget=epochs,
         channels=cfg["custom_channels"], hidden_widths=cfg["hidden_widths"])

    def validation(model, loader=loaders[1]):
        state = rng_state()
        try:
            seed_everything(cfg["seed"] + 2000)
            return evaluate(model, loader, args.dataset, device)
        finally:
            restore_rng(state)

    def head_inputs(model):
        rows = []
        start = cfg.get("temporal_readout_start", 0)
        if hasattr(model.classifier, "features_per_step"):
            raise ValueError("Temporal heads need an explicitly adapted compression fit")
        handle = model.classifier.register_forward_pre_hook(
            lambda module, inputs: rows.append(inputs[0][start:].detach().mean(0)))
        model.eval()
        seed_everything(cfg["seed"] + 213)
        try:
            for data, _ in calibration:
                model.reset_membranes()
                with torch.no_grad():
                    model(prepare_batch(data, args.dataset, device, cfg=cfg))
            return torch.cat(rows)
        finally:
            handle.remove()
            model.reset_membranes()

    def fit_head(targets, labels):
        features = head_inputs(student)
        optimizer = torch.optim.Adam(student.classifier.parameters(), lr=0.01)
        student.eval()
        for _ in range(args.head_steps):
            optimizer.zero_grad(set_to_none=True)
            logits = F.linear(features, student.classifier.weight, student.classifier.bias)
            loss = 0.5 * F.cross_entropy(logits, labels) + 2.0 * F.kl_div(
                F.log_softmax(logits / 2, 1), F.softmax(targets / 2, 1), reduction="batchmean")
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite head recovery loss")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                if cfg.get("classifier_projection", "clamp") == "common_offset":
                    student.classifier.weight.sub_(student.classifier.weight.amin(0, keepdim=True))
                else:
                    student.classifier.weight.clamp_(min=0)
        emit("head_recovered_train_only", samples=len(features), steps=args.head_steps, loss=float(loss.detach()) if args.head_steps else None)

    best_key, best_epoch, start_epoch = (-1.0, -float("inf")), 0, 1
    if args.resume:
        checkpoint = torch.load(args.out / ("best.pth" if args.evaluate_only else "last.pth"), map_location=device, weights_only=True)
        student.load_state_dict(checkpoint["model"], strict=True)
        best_key = tuple(checkpoint["best_key"])
        best_epoch, start_epoch = checkpoint["best_epoch"], checkpoint["epoch"] + 1
    else:
        seed_everything(cfg["seed"] + 213)
        statistics = TrainingStatistics(teacher)
        targets, labels = [], []
        with torch.no_grad():
            for data, batch_labels in calibration:
                teacher.reset_membranes()
                targets.append(teacher(prepare_batch(data, args.dataset, device, cfg=teacher_cfg)))
                labels.append(batch_labels.to(device))
        covariances, activities = statistics.finish()
        targets, labels = torch.cat(targets), torch.cat(labels)
        teacher.reset_membranes()
        if args.student_init is None:
            pruning = prune_from_teacher(teacher, student, covariances, activities, labels)
        else:
            incoming = torch.load(args.student_init, map_location=device, weights_only=True)
            for key in ("custom_channels", "hidden_widths", "learning_rules"):
                if incoming["config"][key] != cfg[key]:
                    raise AssertionError(f"Compressed warm start changed the architecture or learning roles: {key}")
            student.load_state_dict(incoming["model"], strict=True)
            pruning = {"method": "warm_start_from_validation_selected_compressed_checkpoint",
                       "checkpoint": str(args.student_init.resolve()), "sha256": sha(args.student_init),
                       "source_steps": incoming["config"]["dataset_profiles"][args.dataset]["steps"],
                       "target_steps": cfg["dataset_profiles"][args.dataset]["steps"],
                       "hybrid_rules_preserved": True, "all_parameters_reused": True}
            del incoming
        write_json(args.out / "pruning.json", pruning)
        del covariances, activities, statistics
        fit_head(targets, labels)
        initial = validation(student)
        best_key = (initial["top1_accuracy"], -initial["loss"])
        save_checkpoint(args.out / "best.pth", {"model": student.state_dict(), "epoch": 0,
            "best_key": best_key, "best_epoch": 0, "config": cfg, "validation": initial})
        emit("initial_compression_validation", **initial)
        calibration_result = calibrate_backbone_normalization(student, calibration, args.dataset, device, batches=len(calibration))
        fit_head(targets, labels)
        recalibrated = validation(student)
        key = (recalibrated["top1_accuracy"], -recalibrated["loss"])
        emit("bn_recalibrated_validation", calibration=calibration_result, **recalibrated)
        if key > best_key:
            best_key = key
            save_checkpoint(args.out / "best.pth", {"model": student.state_dict(), "epoch": 0,
                "best_key": best_key, "best_epoch": 0, "config": cfg, "validation": recalibrated})
        else:
            student.load_state_dict(torch.load(args.out / "best.pth", map_location=device, weights_only=True)["model"])
        del targets, labels
    assert_physical_weights(student)

    groups = [dict(params=[p for p in student.backbone.parameters() if p.requires_grad], name="backbone", lr=learning_rate),
              dict(params=[p for p in student.hidden_layers.parameters() if p.requires_grad], name="hidden", lr=learning_rate * 0.5),
              dict(params=list(student.classifier.parameters()), name="classifier", lr=learning_rate * 5)]
    optimizer = torch.optim.Adam(groups, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(epochs, 1), eta_min=1e-6)
    if args.resume and not args.evaluate_only:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng(checkpoint["rng"])
        monitor.guard.state = checkpoint["monitor_state"]
    stop_reason = "epoch_budget_completed"
    if not args.evaluate_only and epochs >= start_epoch:
        cache_path = args.out / "teacher_train_logits.pth"
        if cache_path.exists():
            cache = torch.load(cache_path, map_location="cpu", weights_only=True)
            if cache["teacher_sha256"] != entry["checkpoint_sha256"] or cache["split_sha256"] != sha(loaders[3]):
                raise AssertionError("Teacher cache is from another checkpoint or data split")
        else:
            seed_everything(cfg["seed"] + 214)
            teacher_loader = DataLoader(loaders[0].dataset, batch_size=loaders[0].batch_size,
                                        shuffle=False, num_workers=4)
            rows, label_rows, last_log = [], [], time.monotonic()
            with torch.no_grad():
                for index, (data, labels) in enumerate(teacher_loader, 1):
                    teacher.reset_membranes()
                    rows.append(teacher(prepare_batch(data, args.dataset, device, cfg=teacher_cfg)).cpu())
                    label_rows.append(labels)
                    if time.monotonic() - last_log > 30 or index == 1:
                        emit("teacher_train_cache_progress", batches=index, batches_total=len(teacher_loader))
                        last_log = time.monotonic()
            cache = {"logits": torch.cat(rows), "labels": torch.cat(label_rows),
                     "teacher_sha256": entry["checkpoint_sha256"], "split_sha256": sha(loaders[3]),
                     "source_partition": "TRAIN", "encoding_seed": cfg["seed"] + 214}
            save_checkpoint(cache_path, cache)
        cached_logits, cached_labels = cache["logits"].to(device), cache["labels"]
        del cache, teacher
        torch.cuda.empty_cache()
        generator = torch.Generator().manual_seed(cfg["seed"] + 215)
        if args.resume:
            generator.set_state(checkpoint["loader_rng"].cpu())
        training = DataLoader(IndexedDataset(loaders[0].dataset), batch_size=loaders[0].batch_size,
                              shuffle=True, num_workers=4, generator=generator, persistent_workers=True)
        monitor.attach(student)
        seed_everything(cfg["seed"] + 216) if not args.resume else None
        try:
            for epoch in range(start_epoch, epochs + 1):
                student.train()
                started = time.monotonic()
                totals = dict(loss=0.0, ce=0.0, kd=0.0, correct=0, samples=0)
                for batch, (data, labels, indices) in enumerate(training, 1):
                    monitor.begin_batch(batch)
                    if not torch.equal(cached_labels[indices], labels):
                        raise AssertionError("Teacher targets and sample labels are misaligned")
                    student.reset_membranes()
                    optimizer.zero_grad(set_to_none=True)
                    student.zero_grad(set_to_none=True)
                    frames = prepare_batch(data, args.dataset, device, cfg=cfg)
                    labels = labels.to(device)
                    logits = student(frames)
                    ce = F.cross_entropy(logits, labels)
                    kd = F.kl_div(F.log_softmax(logits / 2, 1),
                                  F.softmax(cached_logits[indices.to(device)] / 2, 1), reduction="batchmean") * 4
                    loss = 0.5 * ce + 0.5 * kd
                    if not torch.isfinite(loss):
                        raise TrainingAborted("nonfinite_compression_loss", {"epoch": epoch, "batch": batch})
                    loss.backward()
                    gradient = torch.nn.utils.clip_grad_norm_(student.bp_parameters(), 1, error_if_nonfinite=True)
                    diagnostics = monitor.diagnostics(student) if monitor.capture else {}
                    with WinnerOptimizerGuard(student.hidden_layers.values(), optimizer):
                        optimizer.step()
                    for hidden in student.hidden_layers.values():
                        hidden.apply_stdp()
                    student.project_nonnegative()
                    count = labels.numel()
                    totals["samples"] += count
                    totals["correct"] += int((logits.argmax(1) == labels).sum())
                    for name, value in (("loss", loss), ("ce", ce), ("kd", kd)):
                        totals[name] += float(value.detach()) * count
                    if monitor.capture:
                        elapsed = time.monotonic() - started
                        monitor.log_batch(dataset=args.dataset, epoch=epoch, batch=batch, batches=len(training),
                            loss=float(loss.detach()), cross_entropy=float(ce.detach()), distillation=float(kd.detach()),
                            mean_loss=totals["loss"] / totals["samples"],
                            train_top1_accuracy=totals["correct"] / totals["samples"],
                            gradient_norm=float(gradient), lr=optimizer.param_groups[0]["lr"],
                            elapsed_seconds=elapsed, estimated_epoch_remaining_seconds=elapsed / batch * (len(training) - batch),
                            **diagnostics)
                    if monitor.should_probe():
                        observed = validation(student, probe)
                        monitor.guard.check_validation(observed, stage="probe")
                        emit("validation_probe", epoch=epoch, batch=batch, **observed)
                        student.train()
                observed = validation(student)
                monitor.guard.check_validation(observed, stage="epoch")
                key = (observed["top1_accuracy"], -observed["loss"])
                improved = key > best_key
                if improved:
                    best_key, best_epoch = key, epoch
                    save_checkpoint(args.out / "best.pth", {"model": student.state_dict(), "epoch": epoch,
                        "best_key": best_key, "best_epoch": best_epoch, "config": cfg, "validation": observed})
                should_stop = monitor.guard.epoch_should_stop(observed["top1_accuracy"], epoch)
                scheduler.step()
                assert_physical_weights(student)
                save_checkpoint(args.out / "last.pth", {"model": student.state_dict(), "epoch": epoch,
                    "best_key": best_key, "best_epoch": best_epoch, "config": cfg,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "rng": rng_state(), "loader_rng": generator.get_state(), "monitor_state": monitor.guard.state})
                emit("epoch_completed", epoch=epoch, validation=observed, best_epoch=best_epoch,
                     training={key: value / totals["samples"] for key, value in totals.items() if key not in {"correct", "samples"}},
                     elapsed_seconds=time.monotonic() - started, improved=improved)
                if should_stop:
                    stop_reason = "validation_plateau_early_stop"
                    break
        except TrainingAborted as error:
            stop_reason = error.reason
            emit("training_early_aborted", reason=error.reason, metrics=error.metrics)
    frozen = torch.load(args.out / "best.pth", map_location=device, weights_only=True)
    student.load_state_dict(frozen["model"], strict=True)
    student.eval()
    emit("selected_for_test", epoch=frozen["best_epoch"], validation=frozen["validation"], stop_reason=stop_reason)
    seed_everything(entry["test_rng_seed"])
    before = {name: value.clone() for name, value in student.state_dict().items()}
    scores = evaluate_precisions(student, loaders[2], args.dataset, device)
    if any(not torch.equal(value, before[name]) for name, value in student.state_dict().items()):
        raise AssertionError("Inference changed model weights or adaptation buffers")
    assert_physical_weights(student)
    if sha(entry["checkpoint"]) != entry["checkpoint_sha256"]:
        raise AssertionError("Teacher checkpoint changed")
    result = {"dataset": args.dataset, "parameters": counts, "teacher_parameters": teacher_counts,
              "compression_ratio": provenance["compression_ratio"], "best_epoch": frozen["best_epoch"],
              "validation": frozen["validation"], "test": scores, "stop_reason": stop_reason,
              "checkpoint": str((args.out / "best.pth").resolve()), "checkpoint_sha256": sha(args.out / "best.pth"),
              "test_samples": len(loaders[2].dataset), "inference_state_unchanged": True,
              "teacher_checkpoint_unchanged": True, "at": datetime.now(timezone.utc).isoformat()}
    result_path = args.out / "accuracy.json"
    if args.evaluate_only and result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous["test"] != scores or previous["checkpoint_sha256"] != result["checkpoint_sha256"]:
            raise AssertionError("Independent replay differs from original full-test result")
        write_json(args.out / "replay.json", result | {"exact_replay": True})
    else:
        write_json(result_path, result)
    emit("compression_completed", parameters=counts["total"], best_epoch=frozen["best_epoch"],
         test_accuracy={name: value["top1_accuracy"] for name, value in scores.items()})
    monitor.close()


if __name__ == "__main__":
    main()
