"""Run the four compression trials sequentially on one GPU and persist status."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["CIFAR10DVS", "DVSGesture", "MNIST", "NMNIST"],
                        choices=["CIFAR10DVS", "DVSGesture", "MNIST", "NMNIST"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.resume:
        parser.error("output directory exists; choose a new path or --resume")
    args.out.mkdir(parents=True, exist_ok=True)
    status_path = args.out / "campaign.json"
    state = json.loads(status_path.read_text()) if status_path.exists() else {
        "objective": "compress each trained SNN to 3M–5M total dense physical parameters",
        "datasets": args.datasets, "trials": {}, "started": datetime.now(timezone.utc).isoformat()}

    def update():
        state["updated"] = datetime.now(timezone.utc).isoformat()
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(status_path)

    for dataset in args.datasets:
        destination = args.out / dataset
        if (destination / "accuracy.json").exists():
            state["trials"][dataset] = {"status": "completed", "result": str(destination / "accuracy.json")}
            continue
        command = [sys.executable, str(Path(__file__).with_name("run.py")),
                   "--dataset", dataset, "--out", str(destination)]
        if (destination / "last.pth").exists():
            command.append("--resume")
        elif destination.exists():
            state["trials"][dataset] = {"status": "incomplete_initialization", "directory": str(destination)}
            update()
            continue
        process = subprocess.Popen(command)
        state["current_dataset"] = dataset
        state["trials"][dataset] = {"status": "running", "pid": process.pid, "command": command}
        update()
        code = process.wait()
        state["trials"][dataset].update(status="completed" if code == 0 else "failed", exit_code=code)
        update()
    state["current_dataset"] = None
    state["status"] = "completed" if all(row["status"] == "completed" for row in state["trials"].values()) else "incomplete"
    update()
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
