"""Run the accuracy matrix on one GPU with durable logs and a frozen config.

Dataset preparation writes datasets.json in the campaign directory. A dataset
is eligible only when its status is ready, after successful full preparation.
Invoke this module again to continue a stopped campaign; the training entry
point restores its latest checkpoint. Failed jobs remain recorded for review.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import yaml

from .config import PROJECT_ROOT, load_config


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)
    if path.name == "status.json":
        summary = path.parent / "accuracy_summary.csv"
        temporary = summary.with_suffix(".csv.tmp")
        columns = ["dataset", "model", "homeostasis", "status", "test_samples",
                   "float32_top1_percent", "uint2_top1_percent", "uint4_top1_percent", "result",
                   "training_outcome", "training_stop_reason"]
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for job in value["jobs"]:
                row = {key: job.get(key, "") for key in columns[:4]}
                scores = job.get("test", {}) if job["status"] == "completed" else {}
                row["test_samples"] = scores.get("float32", {}).get("samples", "")
                for precision in ("float32", "uint2", "uint4"):
                    score = scores.get(precision, {}).get("top1_accuracy")
                    row[f"{precision}_top1_percent"] = f"{100*score:.4f}" if score is not None else ""
                row["result"] = job.get("result", "")
                row["training_outcome"] = job.get("training_outcome", "")
                row["training_stop_reason"] = job.get("training_stop", {}).get("reason", "")
                writer.writerow(row)
        temporary.replace(summary)


def source_hashes():
    paths = list((PROJECT_ROOT / "experiments/accuracy").glob("*.py"))
    paths += list((PROJECT_ROOT / "experiments/accuracy/vendor").rglob("*.py"))
    paths += list((PROJECT_ROOT / "experiments/accuracy/vendor").rglob("provenance.json"))
    paths += [PROJECT_ROOT / "snn" / name for name in
              ("homeostasis.py", "wta.py", "inference_quant.py", "neuron.py")]
    return {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def run(directory, config_path):
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "runner.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("This campaign already has an active runner")
    frozen_config = directory / "config.yaml"
    if not frozen_config.exists():
        config = load_config(config_path)
        config["out_dir"] = str(directory / "runs")
        frozen_config.write_text(yaml.safe_dump(config, sort_keys=False))
    config = load_config(frozen_config)
    config_sha256 = hashlib.sha256(frozen_config.read_bytes()).hexdigest()
    hashes = source_hashes()
    status_path = directory / "status.json"
    if status_path.exists():
        state = json.loads(status_path.read_text())
        if state.get("status") in {"paused", "stopped_for_review"}:
            raise SystemExit("Campaign is paused or stopped for review; it cannot automatically continue queued jobs")
        if state["config_sha256"] != config_sha256 or state["source_sha256"] != hashes:
            raise SystemExit("Campaign config or source changed; start a new campaign directory")
        for job in state["jobs"]:
            if job["status"] == "running":
                job["status"] = "queued"
    else:
        state = {"created_at": timestamp(), "config_sha256": config_sha256,
                 "source_sha256": hashes, "config": str(frozen_config),
                 "jobs": [{"model": model, "dataset": dataset, "homeostasis": mode,
                           "status": "queued"}
                          for dataset, model, mode in product(config["datasets"], config["models"],
                                                              config["homeostasis_modes"])]}
    state.update(runner_pid=os.getpid(), started_at=timestamp(), status="running")
    logs = directory / "logs"
    logs.mkdir(exist_ok=True)
    write_json(status_path, state)
    while True:
        readiness = directory / "datasets.json"
        datasets = json.loads(readiness.read_text()) if readiness.exists() else {}
        queued = [job for job in state["jobs"] if job["status"] == "queued"]
        if not queued:
            state["status"] = "completed" if all(job["status"] == "completed" for job in state["jobs"]) else "finished_with_failures"
            state["finished_at"] = timestamp()
            write_json(status_path, state)
            return
        available = [job for job in queued if datasets.get(job["dataset"], {}).get("status") == "ready"]
        if not available:
            state["status"] = "waiting_for_data"
            state["updated_at"] = timestamp()
            state["datasets"] = datasets
            write_json(status_path, state)
            time.sleep(20)
            continue
        job = available[0]
        job_name = f"{job['model']}_{job['dataset']}_{job['homeostasis']}_seed{config['seed']}"
        output = Path(config["out_dir"]) / job_name
        result = output / "accuracy.json"
        if result.exists():
            evidence = json.loads(result.read_text())
            if evidence["metadata"]["evidence_scope"] != "full_dataset":
                raise RuntimeError("Functional-check results cannot complete a campaign job")
            job.update(status="completed", result=str(result), test=evidence["test"],
                       training_outcome=evidence.get("training_outcome", "normal_completion"),
                       training_stop=evidence.get("training_stop", {}))
            write_json(status_path, state)
            continue
        command = [sys.executable, "-B", "-u", "-m", "experiments.accuracy.run",
                   "--config", str(frozen_config), "train", "--model", job["model"],
                   "--dataset", job["dataset"], "--homeostasis", job["homeostasis"], "--device", "cuda"]
        latest = output / "latest.pth"
        if latest.exists():
            command += ["--resume", str(latest)]
        log_path = logs / (job_name + ".log")
        environment = dict(os.environ, MPLBACKEND="Agg")
        with log_path.open("a", buffering=1) as log:
            log.write(json.dumps({"event": "job_started", "at": timestamp(), "command": command}) + "\n")
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT)
            job.update(status="running", pid=process.pid, started_at=timestamp(),
                       log=str(log_path), command=command)
            state.update(status="running", datasets=datasets, active_job=job_name)
            write_json(status_path, state)
            returncode = process.wait()
        job.update(returncode=returncode, finished_at=timestamp())
        if returncode == 0 and result.exists():
            evidence = json.loads(result.read_text())
            job.update(status="completed", result=str(result), test=evidence["test"],
                       training_outcome=evidence.get("training_outcome", "normal_completion"),
                       training_stop=evidence.get("training_stop", {}))
        else:
            job["status"] = "failed"
            stopped = output / "stopped.json"
            if returncode == 2 and stopped.exists():
                evidence = json.loads(stopped.read_text())
                job.update(status="aborted", reason=evidence["reason"], diagnostic=str(stopped))
                state.update(status="stopped_for_review", reason=evidence["reason"], finished_at=timestamp())
                state.pop("active_job", None)
                state.pop("runner_pid", None)
                write_json(status_path, state)
                return
            state.update(status="stopped_for_review", reason="training_process_failed", finished_at=timestamp())
            state.pop("active_job", None)
            state.pop("runner_pid", None)
            write_json(status_path, state)
            return
        state.pop("active_job", None)
        write_json(status_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/accuracy_matrix.yaml"))
    args = parser.parse_args()
    run(Path(args.directory).resolve(), args.config)


if __name__ == "__main__":
    main()
