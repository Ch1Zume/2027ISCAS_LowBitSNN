"""Frozen full-TEST activity campaign: N-MNIST first, then other task conditions."""
import argparse
import importlib
import json
import os
import sys
import time
import traceback
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from spikingjelly.activation_based.neuron import BaseNode

from activity import ActivityCollector, ObservedLoader
from test_data import FrozenTestFrames, write_json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from common import entry, runtime, load_file, configure_torch, sha, verify


def stamp():
    return datetime.now(timezone.utc).astimezone(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["NMNIST", "MNIST", "DVSGesture", "CIFAR10DVS"])
    parser.add_argument("--precisions", nargs="+", default=["uint4", "uint2", "float32"])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--wait-data-seconds", type=int, default=7200)
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    completed = []
    def status(phase, **values):
        record = dict(at=stamp(), pid=os.getpid(), phase=phase, completed=completed,
                      elapsed_seconds=time.monotonic()-started, **values)
        write_json(a.out / "status.json", record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    status("verify_frozen_payload")
    verified = verify("lut_final")
    configure_torch()
    assert torch.cuda.is_available(), "CUDA required for full TEST"
    device = torch.device("cuda")
    contract = dict(scope="frozen lut_final complete TEST; inference only; no updates or recalibration",
         datasets=a.datasets, precisions=a.precisions, primary="NMNIST/hidden_layers.hidden2",
         counting="layer/sample/view/step; all outputs retained; winners are learning gates only",
         SOP="one binary presynaptic spike times one addressed synapse; include zero weights; exclude padding",
         nonbinary_inputs="nonzero activities and addressed terms only; physical SOP unset until encoding is defined",
         reset="software spike reset and logical boundary initializations; hardware RST counts unset",
         comparison="full TEST accuracy and full score checked against frozen delivery; instrumentation smoke exact",
         conditional_clock_hz=50_000_000, clock_mapping="one complete logical step per clock is conditional",
         power="not measured; voltage/charge mapping, physical reset, gating and circuit energy pending",
         baseline_inputs="N-MNIST hidden2 binary; no assumption about raw sensor frames being binary",
         source_files={name:sha(Path(__file__).parent/name) for name in ("activity.py", "run_activity.py", "test_data.py")},
         frozen_payload_verified=verified)
    write_json(a.out / "contract.json", contract)
    for dataset in a.datasets:
        data_start = time.monotonic()
        while not (a.data_dir / dataset / "TEST_manifest.json").exists():
            if time.monotonic()-data_start > a.wait_data_seconds:
                raise RuntimeError("TEST data not ready: " + dataset)
            status("waiting_for_TEST_data", dataset=dataset)
            time.sleep(30)
        e = entry("lut_final", dataset)
        modules = runtime(e)
        run = modules["run"]
        cfg = json.loads((ROOT / e["config"]).read_text())
        cfg.update(data_dir=str(a.data_dir.resolve()), num_workers=a.workers)
        run.seed_everything(cfg["seed"])
        base = modules["models"].AccuracySNN("custom", dataset, cfg, "exponential").to(device)
        ckpt = torch.load(ROOT / e["checkpoint"], map_location=device, weights_only=True)
        base.load_state_dict(ckpt["model"], strict=True)
        assert sum(p.numel() for p in base.parameters()) == e["parameters"]
        assert sha(ROOT / e["checkpoint"]) == e["checkpoint_sha256"]
        base.eval()
        model = base
        if e["views"] == 2:
            model = load_file("_activity_views", ROOT / e["inference_views"]).InferenceViews(base, 2).eval()
        original = {k:v.clone() for k,v in model.state_dict().items()}
        test = FrozenTestFrames(dataset, a.data_dir)
        # Original batch sizes and DataLoader RNG behavior are preserved.
        batch_size = cfg.get("batch_sizes", {}).get(dataset, modules["data"].DATASETS[dataset].batch_size)
        loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=a.workers)
        quant = importlib.import_module("snn.inference_quant")
        if "quantizer_settings" in e:
            ptq = load_file("_activity_ptq", ROOT / e["ptq"])
            moments = torch.load(ROOT / e["TRAIN_moments"], map_location="cpu", weights_only=True)
            settings = torch.load(ROOT / e["quantizer_settings"], map_location="cpu", weights_only=True)
        for precision in a.precisions:
            dest = a.out / dataset / precision
            summary_file = dest / "summary.json"
            if summary_file.exists():
                old = json.loads(summary_file.read_text())
                assert old["validation"]["counts_match"] and old["checkpoint_sha256"] == e["checkpoint_sha256"]
                assert old["instrumentation_sha256"] == contract["source_files"]
                completed.append(f"{dataset}/{precision}")
                continue
            dest.mkdir(parents=True, exist_ok=True)
            bits = int(precision[-1]) if precision.startswith("uint") else None
            originals = {name:layer.weight.clone() for name,layer in base.hidden_layers.items()}
            if bits and "quantizer_settings" in e:
                option = settings[str(bits)]
                context = ptq.calibrated_weights(model, bits, option["ratios"], option["bias"], moments)
            else:
                context = quant.quantized_hidden_weights(model, bits) if bits else nullcontext()
            condition_start = time.monotonic()
            with context:
                exported = {}
                for name, layer in base.hidden_layers.items():
                    for key,value in (("weight_effective", layer.weight), ("bias", layer.bias),
                                      ("threshold", layer.thresholds.threshold)):
                        exported[name+"__"+key] = value.detach().cpu().numpy()
                    if bits:
                        if "quantizer_settings" in e:
                            _, codes, scales = ptq.quantized_values(originals[name], bits, option["ratios"][name])
                        else:
                            _, codes, scales = quant.quantize_nonnegative(originals[name], bits)
                        assert torch.equal(codes.to(scales.dtype)*scales, layer.weight)
                        exported[name+"__weight_codes"] = codes.detach().cpu().numpy()
                        exported[name+"__weight_scales"] = scales.detach().cpu().numpy()
                np.savez_compressed(dest / "hidden_weights_and_thresholds.npz", **exported)
                # Exact first-batch gate compares outputs, RNG, and state with/without hooks.
                raw_data, labels = next(iter(loader))
                run.seed_everything(e["test_rng_seed"])
                frames = modules["data"].prepare_batch(raw_data, dataset, device, cfg=cfg)
                saved_rng = run.rng_state()
                with torch.no_grad():
                    model.reset_membranes()
                    baseline = model(frames).clone()
                    baseline_rng = run.rng_state()
                    model.reset_membranes()
                    run.restore_rng(saved_rng)
                    smoke = ActivityCollector(base, dest / "smoke", len(labels), e["views"], e["steps_per_view"],
                                              modules["wta"].WTAHidden, BaseNode)
                    try:
                        smoke.begin(labels)
                        instrumented = model(frames)
                        smoke.end()
                        exact_logits = torch.equal(baseline, instrumented)
                        assert exact_logits, "Activity hooks changed logits"
                        instrumented_rng = run.rng_state()
                        assert torch.equal(baseline_rng[0], instrumented_rng[0])
                        assert all(torch.equal(x,y) for x,y in zip(baseline_rng[1], instrumented_rng[1]))
                    finally:
                        smoke.close()
                        model.reset_membranes()
                write_json(dest / "instrumentation_audit.json", dict(first_batch_samples=len(labels),
                     exact_logits=exact_logits, RNG_unchanged=True, shadow_spikes_match=True,
                     scope="integration gate; not a full TEST quality result"))
                collector = ActivityCollector(base, dest, len(test), e["views"], e["steps_per_view"],
                                              modules["wta"].WTAHidden, BaseNode)
                last = [0.0]
                def progress(n):
                    now = time.monotonic()
                    if now-last[0] >= 15 or n == len(test):
                        elapsed = now-condition_start
                        status("full_TEST_activity", dataset=dataset, precision=precision,
                               samples=n, total_samples=len(test), condition_elapsed_seconds=elapsed,
                               ETA_seconds=elapsed*(len(test)-n)/n if n else None,
                               gpu=torch.cuda.get_device_name(), gpu_memory_allocated_bytes=torch.cuda.memory_allocated())
                        last[0] = now
                run.seed_everything(e["test_rng_seed"])
                status("full_TEST_activity", dataset=dataset, precision=precision, samples=0, total_samples=len(test))
                try:
                    score = run.evaluate(model, ObservedLoader(loader, collector, progress), dataset, device)
                    if bits and "quantized_layers" in e["test"][precision]:
                        score["quantized_layers"] = list(base.hidden_layers)
                    layers = collector.finalize()
                finally:
                    collector.close()
                    model.reset_membranes()
            unchanged = all(torch.equal(v,model.state_dict()[k]) for k,v in original.items())
            assert unchanged, "Frozen inference state changed"
            expected = e["test"][precision]
            counts_match = score["correct"] == expected["correct"] and score["samples"] == expected["samples"]
            exact = score == expected
            result = dict(dataset=dataset, precision=precision, samples=len(test), views=e["views"],
                 steps_per_view=e["steps_per_view"], score=score, expected=expected,
                 validation=dict(counts_match=counts_match, exact_full_score_replay=exact,
                                 inference_state_unchanged=unchanged, exact_first_batch_instrumentation=True,
                                 all_hidden_shadow_spikes_match=True),
                 checkpoint_sha256=e["checkpoint_sha256"], split_sha256=e["split_sha256"],
                 TEST_manifest_sha256=sha(a.data_dir / dataset / "TEST_manifest.json"),
                 instrumentation_sha256=contract["source_files"],
                 evidence="complete TEST algorithmic activity; physical power/reset/Hz not measured",
                 runtime_seconds=time.monotonic()-condition_start,
                 environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                                  GPU=torch.cuda.get_device_name(), batch_size=batch_size),
                 layers=layers)
            write_json(summary_file, result)
            assert counts_match, "Full TEST correct/sample counts differ from frozen model; inspect summary"
            completed.append(f"{dataset}/{precision}")
            status("condition_completed", dataset=dataset, precision=precision,
                   accuracy=score["top1_accuracy"], exact_full_score_replay=exact, counts_match=counts_match)
        del model, base, original, ckpt
        torch.cuda.empty_cache()
    status("completed", conditions=len(completed), hardware_power_measured=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--out" in sys.argv:
            status_path = Path(sys.argv[sys.argv.index("--out")+1])/"status.json"
            state = json.loads(status_path.read_text()) if status_path.exists() else {}
            state.update(at=stamp(), phase="failed", error_type=type(error).__name__, error=str(error))
            write_json(status_path, state)
        traceback.print_exc()
        sys.exit(1)
