"""Independently replay a selected compressed STDP condition with frozen code."""
import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--dataset", required=True, choices=("MNIST", "NMNIST", "CIFAR10DVS", "DVSGesture"))
    p.add_argument("--condition", default="lut_events", choices=("legacy_native", "legacy_events", "lut_events", "lut_native_trace"))
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    if a.out.exists():
        p.error("Independent replay output must not exist")
    b = a.bundle.resolve()
    condition = b / a.dataset / a.condition
    protocol = json.loads((b / "protocol.json").read_text())
    provenance = json.loads((condition / "provenance.json").read_text())
    expected = json.loads((condition / "accuracy.json").read_text())
    assert sha(condition / "best.pth") == expected["selection"]["checkpoint_sha256"]
    assert sha(condition / "config.json") == provenance["config_sha256"]
    for name, digest in protocol["source_sha256"].items():
        assert sha(b / name) == digest, name
    source = b / "runtime" / a.dataset
    sys.path.insert(0, str(source))
    import torch
    from experiments.accuracy.models import AccuracySNN
    from experiments.accuracy.data import build_loaders
    from experiments.accuracy.run import evaluate_precisions, seed_everything
    for name in ("experiments.accuracy.models", "experiments.accuracy.data", "experiments.accuracy.run", "snn.wta", "snn.stdp_lut"):
        assert Path(importlib.import_module(name).__file__).resolve().is_relative_to(source)
    device = torch.device(provenance["device"])
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = json.loads((condition / "config.json").read_text())
    seed_everything(cfg["seed"])
    model = AccuracySNN("custom", a.dataset, cfg, "exponential").to(device)
    saved = torch.load(condition / "best.pth", map_location=device, weights_only=True)
    model.load_state_dict(saved["model"], strict=True)
    assert sum(p.numel() for p in model.parameters()) == expected["parameters"]
    loader_cfg = dict(cfg, num_workers=0)
    loaders = build_loaders(a.dataset, loader_cfg)
    assert sha(loaders[3]) == provenance["split_sha256"]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    seed_everything(expected["test_rng_seed"])
    scores = evaluate_precisions(model, loaders[2], a.dataset, device)
    assert scores == expected["test"], "Full TEST replay differs"
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()), "Inference changed state"
    result = {"dataset": a.dataset, "condition": a.condition, "exact_full_test_replay": True,
              "parameters": expected["parameters"], "checkpoint_sha256": sha(condition / "best.pth"),
              "runtime_imports_verified": True, "inference_state_unchanged": True,
              "scores": scores, "new_training_steps": 0}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"dataset": a.dataset, "condition": a.condition, "exact_full_test_replay": True,
                      "correct": {k: v["correct"] for k, v in scores.items()}}), flush=True)


if __name__ == "__main__":
    main()
