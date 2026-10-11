"""Replay a published compression bundle using only its frozen student/runtime."""

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["MNIST", "NMNIST", "CIFAR10DVS", "DVSGesture"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "compression_manifest.json").read_text())
    entry = next(item for item in manifest["entries"] if item["dataset"] == args.dataset)
    if args.out.exists():
        raise AssertionError("Preserve prior replay evidence; use a fresh output path")
    for name, expected in entry["bundle_sha256"].items():
        if sha(root / name) != expected:
            raise AssertionError(f"Published file changed: {name}")
    source = (root / entry["source_root"]).resolve()
    sys.path.insert(0, str(source))
    import torch
    from experiments.accuracy.models import AccuracySNN
    from experiments.accuracy.data import build_loaders
    from experiments.accuracy.run import evaluate_precisions, seed_everything

    for name in entry["runtime_imports"]:
        if not Path(importlib.import_module(name).__file__).resolve().is_relative_to(source):
            raise AssertionError(f"Runtime escaped the frozen bundle: {name}")
    if not torch.cuda.is_available():
        raise RuntimeError("Exact published replay requires the CUDA host")
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = json.loads((root / entry["config"]).read_text())
    seed_everything(cfg["seed"])
    loaders = build_loaders(args.dataset, cfg)
    if sha(loaders[3]) != entry["split_sha256"]:
        raise AssertionError("TRAIN/validation/TEST split changed")
    saved = torch.load(root / entry["checkpoint"], map_location="cpu", weights_only=True)
    model = AccuracySNN("custom", args.dataset, cfg, "exponential").cuda()
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    counts = sum(parameter.numel() for parameter in model.parameters())
    if counts != entry["parameters"]["total"] or not 3_000_000 <= counts <= 5_000_000:
        raise AssertionError("Physical dense parameter budget failed")
    before = {name: value.clone() for name, value in model.state_dict().items()}
    seed_everything(entry["test_rng_seed"])
    test = evaluate_precisions(model, loaders[2], args.dataset, torch.device("cuda"))
    expected = json.loads((root / entry["accuracy"]).read_text())["test"]
    if test != expected:
        raise AssertionError("Complete FP32/uint2/uint4 result dictionaries differ")
    if any(not torch.equal(value, before[name]) for name, value in model.state_dict().items()):
        raise AssertionError("Inference changed a weight or adaptation buffer")
    if sha(root / entry["checkpoint"]) != entry["checkpoint_sha256"]:
        raise AssertionError("Published checkpoint changed during inference")
    output = dict(dataset=args.dataset, exact_replay=True, parameters=counts, test=test,
                  checkpoint_sha256=entry["checkpoint_sha256"], model_state_unchanged=True,
                  runtime_imports_verified=True, test_rng_seed=entry["test_rng_seed"], new_training_steps=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(dict(dataset=args.dataset, exact_replay=True, parameters=counts,
                          correct={name: value["correct"] for name, value in test.items()})), flush=True)


if __name__ == "__main__":
    main()
