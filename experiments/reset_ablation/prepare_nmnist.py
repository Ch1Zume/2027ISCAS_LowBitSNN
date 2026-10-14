"""Restore missing TRAIN frames; preserve the existing TEST cache and frozen order."""
import argparse
import hashlib
import json
import sys
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets")
    args = parser.parse_args()
    import numpy as np
    from spikingjelly.datasets import integrate_events_by_fixed_frames_number
    from spikingjelly.datasets.n_mnist import NMNIST

    base = args.data_dir / "NMNIST"
    resource = next(r for r in json.loads((ROOT / "configs/dataset_sources.json").read_text())["NMNIST"]
                    if r["name"] == "Train.zip")
    sys.path.insert(0, str(ROOT / 'experiments/power'))
    from test_data import download
    archive = download(resource, base / 'download')
    assert digest(archive, "sha256") == resource["sha256"]
    order = json.loads((ROOT / "configs/event_file_order.json").read_text())["NMNIST"]
    expected = [f for f in order if f.startswith("train/")]
    assert len(expected) == 60000
    frames = base / "frames_number_16_split_by_number"
    with zipfile.ZipFile(archive) as z:
        members = {Path(n).stem: n for n in z.namelist() if n.lower().endswith(".bin")}
        assert len(members) == len(expected)
        def convert(name):
            dest = frames / name
            if dest.exists():
                return
            member = members[Path(name).stem]
            assert Path(member).parent.name == Path(name).parent.name
            # Use the pinned official reader, including its timestamp semantics.
            with tempfile.NamedTemporaryFile(suffix=".bin") as raw:
                raw.write(z.read(member))
                raw.flush()
                events = NMNIST.load_origin_data(raw.name)
            array = integrate_events_by_fixed_frames_number(events, "number", 16, 34, 34)
            assert int(array.sum()) == len(events["t"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            temporary = dest.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, frames=array)
            temporary.replace(dest)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i, _ in enumerate(pool.map(convert, expected), 1):
                if i == 1 or i % 2000 == 0:
                    print(json.dumps(dict(event="restore_TRAIN", samples=i, total=len(expected))), flush=True)
    actual = [str(p.relative_to(frames)) for p in sorted(frames.rglob("*.npz"))]
    assert actual == order, "Restored names/order differ from the frozen population"
    record = dict(train_population=60000, test_population=10000,
                  train_archive_sha256=resource["sha256"], native_frames=16,
                  split_by="number", reader="spikingjelly.datasets.n_mnist.NMNIST.load_origin_data",
                  frozen_file_order_matched=True, existing_TEST_untouched=True)
    (base / "TRAIN_restore_20261014.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
