"""Fixed final fit on official TRAIN, after complete-VAL checkpoint selection."""
import argparse
import copy
import hashlib
import json
import math
from argparse import Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from .config import load_config
from .data import build_loaders
from .models import AccuracySNN
from .monitoring import TrainingMonitor
from .run import (calibrate_backbone_normalization, evaluate_precisions, initialize_own_custom_phase,
                  make_lr_scheduler, metadata, restore_rng, rng_state, seed_everything, train_epoch,
                  validate_resume_budget)

FULL_COUNTS = {"MNIST": (60000, 10000), "NMNIST": (60000, 10000),
               "CIFAR10DVS": (9000, 1000), "DVSGesture": (1176, 288)}


def full_train_loader(loaders, cfg, expected_counts):
    """Merge only the two disjoint official TRAIN subsets; seal TEST indices."""
    train, validation, test = (loader.dataset for loader in loaders[:3])
    if not isinstance(train, Subset) or not isinstance(validation, Subset) or train.dataset is not validation.dataset:
        raise ValueError("Final refit needs TRAIN and VAL subsets of the same official training source")
    first, second = set(train.indices), set(validation.indices)
    if first & second or len(first) != len(train) or len(second) != len(validation):
        raise ValueError("TRAIN and VAL indices overlap or contain duplicates")
    indices = sorted(first | second)
    if isinstance(test, Subset) and test.dataset is train.dataset:
        if set(indices) & set(test.indices):
            raise ValueError("Final TRAIN overlaps TEST")
    elif len(indices) != len(train.dataset):
        raise ValueError("Final refit must cover the complete separate official training source")
    if hasattr(train.dataset, "samples") and hasattr(test, "samples"):
        paths = {str(Path(path).resolve()) for path, _ in train.dataset.samples}
        if paths & {str(Path(path).resolve()) for path, _ in test.samples}:
            raise ValueError("Final TRAIN and TEST sample paths overlap")
    if (len(indices), len(test)) != tuple(expected_counts):
        raise ValueError("Final official training/test counts differ from the registered benchmark")
    generator = torch.Generator().manual_seed(cfg["seed"])
    merged = DataLoader(Subset(train.dataset, indices), batch_size=loaders[0].batch_size,
                        shuffle=True, drop_last=False, num_workers=cfg.get("num_workers", 0), generator=generator)
    return merged, {"former_train_indices": sorted(first), "former_validation_indices": sorted(second),
                    "final_train_indices": indices, "test_indices": list(test.indices) if isinstance(test, Subset) else None,
                    "final_train_samples": len(indices), "test_samples": len(test),
                    "validation_after_merge": "none; endpoint fixed before TRAIN union; no post-merge model selection"}


def run_refit(case):
    case = Path(case).resolve()
    protocol = json.loads((case / "protocol.json").read_text())
    source = Path(protocol["source_root"])
    assert all(hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
               for name, digest in protocol["source_sha256"].items())
    assert hashlib.sha256((case / "config.yaml").read_bytes()).hexdigest() == protocol["config_sha256"]
    cfg = load_config(case / "config.yaml")
    dataset = protocol["dataset"]
    scope = protocol.get("evidence_scope", "full_dataset")
    counts = FULL_COUNTS[dataset] if scope == "full_dataset" else protocol["functional_counts"]
    torch.set_num_threads(cfg["cpu_threads"])
    seed_everything(cfg["seed"])
    device = torch.device(protocol.get("device", "cuda"))
    out = Path(cfg["out_dir"]) / f"custom_{dataset}_exponential_seed{cfg['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "accuracy.json").exists(), "Do not overwrite a completed final test"
    parent = Path(protocol["parent_checkpoint"])
    digest = hashlib.sha256()
    with parent.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 ** 2), b""):
            digest.update(block)
    assert digest.hexdigest() == protocol["parent_checkpoint_sha256"]
    saved = torch.load(out / "latest.pth" if (out / "latest.pth").exists() else parent,
                       map_location="cpu", weights_only=True)
    loaders = build_loaders(dataset, cfg, batch_size=saved["metadata"]["batch_size"])
    train, union = full_train_loader(loaders, cfg, counts)
    (case / "final_train_manifest.json").write_text(json.dumps(union, indent=2) + "\n")
    total = protocol["original_total_epoch_budget"]
    validate_resume_budget(saved, cfg, total)
    args = Namespace(model="custom", dataset=dataset, homeostasis="exponential", resume=None,
                     own_init_checkpoint=str(parent), batch_size=train.batch_size,
                     official_train_refit=True,
                     limit=saved["metadata"]["sample_limit"], steps=saved["metadata"]["step_override"])
    model = AccuracySNN("custom", dataset, cfg, "exponential").to(device)
    initialize_own_custom_phase(model, saved, args, cfg)
    if saved["metadata"].get("final_refit"):
        model.initialization = copy.deepcopy(saved["metadata"]["initialization"])
    model.initialization["optimizer_and_guard"] = "retain selected own-run optimizer/RNG; freeze validation selectors before official TRAIN union"
    optimizer = torch.optim.Adam(model.bp_parameters(), lr=cfg["dataset_profiles"][dataset]["lr"],
                                 weight_decay=cfg["weight_decay"])
    optimizer.load_state_dict(saved["optimizer"])
    assert all(state["step"].device.type == "cpu" for state in optimizer.state.values())
    scheduler = make_lr_scheduler(optimizer, cfg, total)
    scheduler.load_state_dict(saved["scheduler"])
    start, end = saved["epoch"] + 1, protocol["fixed_end_epoch"]
    if not 0 < protocol["refit_epochs"] <= 20 or end != protocol["parent_epoch"] + protocol["refit_epochs"]:
        raise ValueError("Final-refit endpoint must be fixed from the selected validation predecessor")
    if start > end + 1:
        raise ValueError("Saved refit passed its frozen endpoint")
    run_meta = metadata(model, cfg, "exponential", loaders[3], args.limit, args.steps)
    run_meta.update(training_epochs=total, batch_size=train.batch_size, evidence_scope=scope,
                    validation_probe=saved["metadata"]["validation_probe"], final_refit=union,
                    selection_validation=protocol["selection_validation"],
                    checkpoint_selection="fixed endpoint registered before official TRAIN union and TEST",
                    refit_lr_policy="continue saved cosine schedule; retain saved LR when scheduler requires a validation metric")
    monitor = TrainingMonitor(out, cfg["monitoring"])
    monitor.guard.state = copy.deepcopy(saved["monitoring_state"])
    monitor.attach(model)
    restore_rng(saved["rng"])
    train.generator.set_state(saved["loader_rng"].cpu())
    history = saved["history"]

    def training_only_bn(epoch, batch, initial=False):
        if cfg.get("bn_refit_batches", 0):
            stats = calibrate_backbone_normalization(model, train, dataset, device, args.steps,
                                                    cfg["bn_refit_batches"])
            monitor.emit("bn_refit", epoch=epoch, stage="official_train_refit_only", **stats)

    try:
        monitor.emit("training_started", model="custom", dataset=dataset, start_epoch=start,
                     fixed_end_epoch=end, final_train_samples=len(train.dataset),
                     validation_after_merge=False, test_until_endpoint=False)
        for epoch in range(start, end + 1):
            score = train_epoch(model, train, optimizer, dataset, device, args.steps,
                                monitor=monitor, epoch=epoch, on_probe=training_only_bn)
            training_only_bn(epoch, 0)
            if cfg.get("lr_schedule", "cosine") == "cosine":
                scheduler.step()
            entry = {"epoch": epoch, "train": score, "validation": None,
                     "training_split": "complete_official_train_after_validation_selection"}
            history.append(entry)
            monitor.emit("epoch_completed", **entry)
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "epoch": epoch, "history": history,
                     "best_validation_accuracy": protocol["selection_validation"]["top1_accuracy"],
                     "metadata": run_meta, "monitoring_state": monitor.guard.state,
                     "rng": rng_state(), "loader_rng": train.generator.get_state()}
            torch.save(state, out / "latest.pth")
        # The endpoint is chosen before this phase, never by merged validation or TEST.
        torch.save(torch.load(out / "latest.pth", map_location="cpu", weights_only=True), out / "best.pth")
        model.eval()
        seed_everything(cfg["seed"] + 1000)
        monitor.emit("test_evaluation_started", selected_epoch=end)
        scores = evaluate_precisions(model, loaders[2], dataset, device, args.steps)
        result = {"metadata": run_meta, "selected_epoch": end,
                  "best_validation_accuracy": protocol["selection_validation"]["top1_accuracy"],
                  "validation_accuracy_scope": "predecessor before final official TRAIN union",
                  "training_stop": {"reason": "frozen_final_refit_endpoint_completed", "epoch": end},
                  "test": scores}
        (out / "accuracy.json").write_text(json.dumps(result, indent=2) + "\n")
        monitor.emit("training_completed", result=str(out / "accuracy.json"), test=scores)
        return result
    finally:
        monitor.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    run_refit(parser.parse_args().case)
