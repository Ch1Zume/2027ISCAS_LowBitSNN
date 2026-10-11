"""Audit the completed compression dependency using its immutable runtime."""
import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "results/accuracy/compression_3to5M_20261006_selected"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dataset")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((BUNDLE / "compression_manifest.json").read_text())
    if not a.dataset:
        handoff = ROOT / "results/workflows/stdp_after_compression_20261006"
        completion = json.loads((handoff / "dependency_completion.json").read_text())
        turn = "01a1114b-f759-7731-a4cb-ef6b6366e82f"
        assert completion["payload"]["type"] == "task_complete"
        assert completion["payload"]["turn_id"] == turn
        rollout = Path("/home/zhangyufan/.codex/sessions/2026/10/06/rollout-2026-10-06T20-54-14-01a11147-6651-7722-8879-0f8f6f7be80a.jsonl")
        lifecycle = []
        for line in rollout.open():
            x = json.loads(line)
            y = x.get("payload", {})
            if x.get("type") == "event_msg" and y.get("type") in {"task_started", "task_complete", "turn_aborted", "task_failed"}:
                lifecycle.append({"at": x.get("timestamp"), "type": y["type"], "turn_id": y.get("turn_id")})
        assert lifecycle[-1]["type"] == "task_complete" and lifecycle[-1]["turn_id"] == turn
        comparisons = json.loads((BUNDLE / "candidate_comparison.json").read_text())
        records = []
        for e in manifest["entries"]:
            bad = [name for name, digest in e["bundle_sha256"].items() if sha(BUNDLE / name) != digest]
            assert not bad, bad
            candidates = comparisons[e["dataset"]]
            ranked = sorted(candidates, key=lambda r: (-r["validation"]["top1_accuracy"], r["validation"]["loss"]))
            assert ranked[0]["selected"] and ranked[0]["condition"] == e["selected_condition"]
            records.append({"dataset": e["dataset"], "parameters": e["parameters"],
                            "checkpoint_sha256": e["checkpoint_sha256"], "verified_files": len(e["bundle_sha256"]),
                            "input_profile": e["input_profile"], "selection_valid_only": True})
        record = {"read_thread_available": False, "rollout": str(rollout), "lifecycle": lifecycle,
                  "compression_manifest_sha256": sha(BUNDLE / "compression_manifest.json"),
                  "host_gpu_processes": subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"], text=True),
                  "entries": records}
        (a.out / "dependency_audit.json").write_text(json.dumps(record, indent=2) + "\n")
        for e in manifest["entries"]:
            subprocess.run([sys.executable, str(Path(__file__)), "--dataset", e["dataset"], "--out", str(a.out)], check=True)
        return
    import torch
    e = next(e for e in manifest["entries"] if e["dataset"] == a.dataset)
    sys.path.insert(0, str(BUNDLE / e["source_root"]))
    from experiments.accuracy.models import AccuracySNN
    from experiments.accuracy.data import build_loaders, prepare_batch
    from experiments.accuracy.run import seed_everything
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = json.loads((BUNDLE / e["config"]).read_text())
    cfg["num_workers"] = 0
    seed_everything(cfg["seed"])
    model = AccuracySNN("custom", a.dataset, cfg, "exponential").cuda().eval()
    saved = torch.load(BUNDLE / e["checkpoint"], map_location="cuda", weights_only=True)
    model.load_state_dict(saved["model"], strict=True)
    counts = sum(x.numel() for x in model.parameters())
    assert counts == e["parameters"]["total"] and 3_000_000 <= counts <= 5_000_000
    inputs = {}
    handles = []
    for name, h in model.hidden_layers.items():
        def capture(module, xs, name=name):
            x = xs[0].detach()
            inputs[name] = {"shape": list(x.shape), "minimum": x.min().item(), "maximum": x.max().item(),
                            "mean": x.mean().item(), "nonzero_fraction": (x > 0).float().mean().item(),
                            "fractional_fraction": ((x != 0) & (x != 1)).float().mean().item()}
        handles.append(h.register_forward_pre_hook(capture))
    loaders = build_loaders(a.dataset, cfg, batch_size=4)
    assert sha(loaders[3]) == e["split_sha256"]
    data, _ = next(iter(loaders[0]))
    with torch.no_grad():
        model(prepare_batch(data, a.dataset, torch.device("cuda"), cfg=cfg))
    for h in handles:
        h.remove()
    roles = {name: {"rule": h.rule, "competition_size": getattr(h, "competition_size", h.out_features),
                    "stdp_rows": int(h.stdp_neuron_mask.sum()) if hasattr(h, "stdp_neuron_mask") else h.out_features if h.rule == "stdp" else 0,
                    "winner_scope": getattr(h, "winner_scope", "presentation"),
                    "winner_tiebreak": getattr(h, "winner_tiebreak", "index"),
                    "a_plus": h.a_plus, "a_minus": h.a_minus, "trace_decay": h.trace_decay, "stdp_lr": h.stdp_lr}
             for name, h in model.hidden_layers.items()}
    record = {"dataset": a.dataset, "actual_parameters": counts, "inputs_train_partition_eval_mode": inputs,
              "roles": roles, "checkpoint_sha256": sha(BUNDLE / e["checkpoint"]),
              "runtime": str(Path(importlib.import_module("snn.wta").__file__).resolve()),
              "source_unchanged": True}
    (a.out / f"source_audit_{a.dataset}.json").write_text(json.dumps(record, indent=2) + "\n")
    subprocess.run([sys.executable, str(BUNDLE / "reproduce.py"), "--dataset", a.dataset,
                    "--out", str(a.out / f"baseline_replay_{a.dataset}.json")], check=True)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
