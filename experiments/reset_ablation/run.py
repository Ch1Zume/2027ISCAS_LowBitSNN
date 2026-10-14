"""Matched N-MNIST recovery from the final checkpoint, with deferred full TEST.

Use the established LUT recovery driver and original TRAIN-only teacher. The
experimental runtime differs by one reset operation; frozen deliveries are read-only.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from common import entry, sha, verify
from experiments.reset_ablation.behavior import reset_source

FINAL = ROOT / "benchmarks/lut_final/NMNIST"
VARIANTS = ("preserve", "reset_all")


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def stage(out, variant, checkpoint_hash):
    base = out / variant
    if (base / "protocol.json").exists():
        return base
    runtime = base / "runtime/NMNIST"
    shutil.copytree(FINAL, runtime,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (runtime / "experiments/__init__.py").write_text("")
    source = runtime / "snn/wta.py"
    if variant == "reset_all":
        source.write_text(reset_source(source.read_text()))
    # N-MNIST has exactly one 512-neuron competition group in each hidden layer.
    driver = (ROOT / "experiments/stdp/run.py").read_text()
    anchor = 'if offset == 0 or (offset+1) % 4 == 0:'
    assert driver.count(anchor) == 1
    driver = driver.replace(anchor, 'if offset == 0 or (offset+1) % 100 == 0:')
    # Save the first evaluated state for a later, fully deferred zero-training TEST.
    anchor = '    save_best(0, best)\n'
    assert driver.count(anchor) == 1
    driver = driver.replace(anchor, anchor + '    shutil.copy2(out / "best.pth", out / "initial.pth")\n')
    # Record each completed epoch's training score, without changing any updates.
    driver = driver.replace('history.append({"block": block, "validation": score})',
                            'history.append({"block": block, "validation": score, '
                            '"train": {"accuracy": correct/count, "loss": loss_sum/count, "samples": count}})')
    (base / "run_frozen.py").write_text(driver)
    write(base / "protocol.json", dict(variant=variant, initial_checkpoint_sha256=checkpoint_hash,
        source_sha256={str(p.relative_to(base)): sha(p) for p in runtime.rglob("*.py")},
        driver_sha256=sha(base / "run_frozen.py"),
        driver_changes=["progress interval 100", "save initial checkpoint", "persist epoch TRAIN metrics"],
        behavior="unchanged" if variant == "preserve" else
        "clear whole sample-specific hidden layer after its first winner; preserve emitted spikes and winner eligibility"))
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    assert 1 <= args.epochs <= 60
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    started = time.monotonic()
    initial = ROOT / entry("lut_final", "NMNIST")["checkpoint"]
    checkpoint_hash = sha(initial)
    e = entry("lut_final", "NMNIST")
    assert checkpoint_hash == e["checkpoint_sha256"]
    contract = dict(dataset="NMNIST", seed=42, train_seed=3042, epochs_per_variant=args.epochs,
        initialization="same final FP32 model and trace RNG; new Adam optimizer in both variants",
        initial_checkpoint=str(initial), initial_checkpoint_sha256=checkpoint_hash,
        split_sha256=e["split_sha256"], steps=4, resolution=34, batch_size=32,
        train_samples=53994, validation_samples=6006, test_samples=10000,
        learning_rates=dict(backbone=2e-5, hidden_bp=1e-5, classifier=1e-4, stdp_lr=0.001),
        scheduler="cosine over six epochs (or explicit equal budget); eta_min=1e-6",
        loss="0.5 CE + 0.5 original teacher KL, temperature=2; TRAIN-only cached targets",
        changed="after the first winner in each sample/layer, reset every membrane in that layer to zero",
        reset_timing="after retaining the current timestep spikes, before the next timestep",
        reset_applies_to="hidden1 and hidden2; train and inference",
        winner_scope="one first winner per sample/layer/presentation; no extra learning rounds",
        unchanged=["current-step simultaneous spikes", "nonnegative synapses", "BP/STDP roles",
                   "homeostasis rule", "input encoding", "dropout", "classification readout", "PTQ"],
        gradients="Boolean reset trigger is detached; gradient through cleared membrane is zero",
        selection="complete VALID accuracy then loss; epoch zero eligible; TEST deferred until both selected",
        limitation="one seed, equal-budget warm-start recovery; not a from-scratch comparison")
    if args.resume:
        assert json.loads((out / "contract.json").read_text()) == contract
    else:
        write(out / "contract.json", contract)
        snapshot = out / "experiment_source"
        shutil.copytree(Path(__file__).parent, snapshot, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(ROOT / "test/test_reset_ablation.py", snapshot / "test_reset_ablation.py")
    env = dict(os.environ, SNN_PROJECT_ROOT=str(ROOT), SNN_DATA_DIR=str(args.data_dir.resolve()),
               SNN_TEACHER_CACHE_DIR=str(out / "teacher_cache"), TMPDIR="/tmp",
               MPLCONFIGDIR="/tmp/snn-reset-mpl", PYTHONUNBUFFERED="1")
    current = {}

    def status(phase, **values):
        current.update(at=now(), phase=phase, elapsed_seconds=time.monotonic()-started, **values)
        write(out / "status.json", current)

    def child(phase, command, log_name):
        status(phase, command=command, latest=None)
        with (out / log_name).open("a") as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            status(phase, pid=proc.pid)
            for line in proc.stdout:
                log.write(line); log.flush()
                try:
                    record = json.loads(line)
                except ValueError:
                    if line.strip():
                        status(phase, message=line.strip()[-800:])
                    continue
                # Full source records remain in the child log; root status is compact.
                slim = {k: v for k, v in record.items() if k not in ("stdp", "winners", "training_provenance")}
                status(phase, latest=slim)
                print(json.dumps(dict(phase=phase, **slim)), flush=True)
            result = proc.wait()
            if result:
                raise subprocess.CalledProcessError(result, command)

    try:
        status("VERIFY_INPUTS")
        verified = verify()
        write(out / "input_audit.json", dict(verified_payload_files=verified, initial_sha256=checkpoint_hash))
        child("PREPARE_TRAIN", [sys.executable, str(ROOT / "experiments/reset_ablation/prepare_nmnist.py"),
            "--data-dir", str(args.data_dir.resolve())], "prepare.log")
        cache = out / "teacher_cache/NMNIST/teacher_train_logits.pth"
        if not cache.exists():
            child("TRAIN_TEACHER_CACHE", [sys.executable, str(ROOT / "scripts/teacher_cache.py"),
                "--dataset", "NMNIST", "--data-dir", str(args.data_dir.resolve()), "--out", str(cache)], "teacher.log")
        selections = {}
        for variant in VARIANTS:
            base = stage(out, variant, checkpoint_hash)
            condition = base / "NMNIST/lut_native_trace"
            if not (condition / "validation.json").exists():
                if condition.exists():
                    # Restart an incomplete branch explicitly; never overwrite its evidence.
                    archive = base / ("incomplete_attempt_" + str(time.time_ns()))
                    condition.rename(archive)
                child("TRAIN_VALID_" + variant, [sys.executable, str(base / "run_frozen.py"),
                    "--out", str(base), "--dataset", "NMNIST", "--condition", "lut_native_trace",
                    "--epochs", str(args.epochs), "--patience", str(args.epochs+1),
                    "--workers", str(args.workers), "--teacher-source", "original_cached",
                    "--initial-checkpoint", str(initial), "--stdp-lr", "0.001", "--bp-rate-scale", "0.1"],
                    variant + ".log")
            selection = json.loads((condition / "validation.json").read_text())
            assert sha(condition / "best.pth") == selection["checkpoint_sha256"]
            assert selection["batches_trained"] == args.epochs * 1688, "Unequal/incomplete training budget"
            assert selection["samples_trained"] == args.epochs * 53994
            provenance = json.loads((condition / "provenance.json").read_text())
            assert provenance["bp_learning_rate"] == 2e-5
            assert all(h["competition_size"] == 512 for h in provenance["actual_hidden_roles"].values())
            selections[variant] = selection
        write(out / "selections_before_TEST.json", selections)
        for variant in VARIANTS:
            write(out / variant / "selections_before_TEST.json", {"NMNIST": {"lut_native_trace": selections[variant]}})
            if not (out / variant / "four_state_TEST.json").exists():
                child("FULL_TEST_" + variant, [sys.executable, str(ROOT / "experiments/reset_ablation/evaluate.py"),
                    "--out", str(out), "--variant", variant, "--workers", str(args.workers),
                    "--data-dir", str(args.data_dir.resolve())], variant + "_test.log")
        results = {v: json.loads((out / v / "four_state_TEST.json").read_text()) for v in VARIANTS}
        gaps = {precision: 100*(results["reset_all"]["selected"][precision]["top1_accuracy"] -
                               results["preserve"]["selected"][precision]["top1_accuracy"])
                for precision in ("float32", "uint2", "uint4")}
        assert verify() == verified and sha(initial) == checkpoint_hash
        write(out / "results.json", dict(contract=contract, results=results, selections=selections,
            reset_minus_preserve_pp=gaps, frozen_payload_unchanged=True, complete=True))
        with (out / "comparison.csv").open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["behavior", "training_state", "selected_epoch", "FP32_percent", "uint2_percent", "uint4_percent", "TEST_samples"])
            for v in VARIANTS:
                for state in ("initial", "selected"):
                    writer.writerow([v, state, 0 if state == "initial" else selections[v]["selected_block"],
                        *[100*results[v][state][p]["top1_accuracy"] for p in ("float32", "uint2", "uint4")], 10000])
        child("REPORT", [sys.executable, str(ROOT / "experiments/reset_ablation/report.py"), "--out", str(out)], "report.log")
        status("completed", reset_minus_preserve_pp=gaps, pid=None, full_test_modes_completed=12)
    except BaseException as error:
        status("failed", error=repr(error), failed_phase=current.get("phase"))
        raise


if __name__ == "__main__":
    main()
