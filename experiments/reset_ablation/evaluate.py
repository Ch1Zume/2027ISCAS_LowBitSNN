"""Evaluate initial and VALID-selected states only after both branches are frozen."""
import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from common import configure_torch, entry, runtime, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", choices=("preserve", "reset_all"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets")
    args = parser.parse_args()
    out = args.out.resolve()
    selections = json.loads((out / "selections_before_TEST.json").read_text())
    for variant, selected in selections.items():
        assert sha(out / variant / "NMNIST/lut_native_trace/best.pth") == selected["checkpoint_sha256"]
    base = out / args.variant
    for name, value in json.loads((base / "protocol.json").read_text())["source_sha256"].items():
        assert sha(base / name) == value
    modules = runtime(dict(runtime=str(base / "runtime/NMNIST")))
    torch = configure_torch()
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    condition = base / "NMNIST/lut_native_trace"
    cfg = json.loads((condition / "config.json").read_text())
    cfg.update(num_workers=args.workers, data_dir=str(args.data_dir.resolve()))
    loaders = modules["data"].build_loaders("NMNIST", cfg)
    reference = entry("lut_final", "NMNIST")
    assert sha(loaders[3]) == reference["split_sha256"]
    assert [len(loader.dataset) for loader in loaders[:3]] == [53994, 6006, 10000]
    model = modules["models"].AccuracySNN("custom", "NMNIST", cfg, "exponential").to(device).eval()
    quantized = importlib.import_module("snn.inference_quant").quantized_hidden_weights
    from contextlib import nullcontext
    results = {}
    for state, checkpoint in (("initial", condition / "initial.pth"), ("selected", condition / "best.pth")):
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(saved["model"], strict=True)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        results[state] = {}
        for bits in (None, 2, 4):
            precision = "float32" if bits is None else f"uint{bits}"
            modules["run"].seed_everything(reference["test_rng_seed"])
            context = nullcontext() if bits is None else quantized(model, bits)
            with context:
                score = modules["run"].evaluate(model, loaders[2], "NMNIST", device,
                    progress=lambda **x: print(json.dumps(dict(event="TEST_progress", state=state, precision=precision, **x)), flush=True))
            assert score["samples"] == 10000
            assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
            results[state][precision] = score
            if args.variant == "preserve" and state == "initial":
                assert score["correct"] == reference["test"][precision]["correct"], "Frozen baseline did not replay"
            print(json.dumps(dict(event="TEST_complete", state=state, precision=precision, **score)), flush=True)
    results.update(inference_state_unchanged=True, samples_per_mode=10000,
                   baseline_replay_matched=args.variant == "preserve",
                   selected_checkpoint_sha256=sha(condition / "best.pth"))
    (base / "four_state_TEST.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
