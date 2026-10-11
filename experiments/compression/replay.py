"""Independently replay a compressed checkpoint using its exact compression source."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", type=Path, required=True)
    args = parser.parse_args()
    condition = args.condition.resolve()
    record = json.loads((condition / "provenance.json").read_text())
    candidates = [condition / "compression_runtime", condition.parent / "compression_source_v1"]
    source = next((directory for directory in candidates if all(
        (directory / name).is_file() and hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
        for name, digest in record["compression_source_sha256"].items())), None)
    if source is None:
        raise AssertionError("No exact compression source snapshot matches the recorded hashes")
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("frozen_compression_run", source / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = Path(__file__).resolve().parents[2]
    module.BENCHMARK = module.ROOT / "results/accuracy/custom_performance_candidates_20261004/selected_benchmark_final_20261005"
    sys.argv = [str(source / "run.py"), "--dataset", record["dataset"], "--out", str(condition),
                "--resume", "--evaluate-only"]
    module.main()


if __name__ == "__main__":
    main()
