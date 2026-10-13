"""Restore only frozen TEST samples with the original SpikingJelly adapters.

No TRAIN data, fitting, calibration, or split selection is performed here.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "migration"))
from common import entry, sha


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def split_contract(name):
    e = entry("lut_final", name)
    splits = list((ROOT / "migration/dataset_splits" / name).glob("*.json"))
    matches = [p for p in splits if sha(p) == e["split_sha256"]]
    assert len(matches) == 1, "Frozen split missing or ambiguous"
    split = json.loads(matches[0].read_text())
    if name == "MNIST":
        files = [f"TEST/{i}" for i in split["test_indices"]]
    else:
        order = json.loads((ROOT / "migration/event_file_order.json").read_text())[name]
        population = order if name == "CIFAR10DVS" else [f for f in order if f.startswith("test/")]
        files = [population[i] for i in split["test_indices"]]
    assert len(files) == e["test"]["float32"]["samples"]
    return e, matches[0], split, files


def digest(path, algorithm="md5"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def download(resource, directory):
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / resource["name"]
    partial = dest.with_suffix(dest.suffix + ".part")
    if not dest.exists():
        if not partial.exists() or digest(partial) != resource["md5"]:
            subprocess.run(["curl", "-fL", "--retry", "4", "--connect-timeout", "30",
                            "--continue-at", "-", "--output", str(partial), resource["url"]], check=True)
        assert digest(partial) == resource["md5"], f"Archive MD5 mismatch: {partial}"
        partial.replace(dest)
    assert digest(dest) == resource["md5"], f"Archive MD5 mismatch: {dest}"
    if resource.get("sha256"):
        assert sha(dest) == resource["sha256"]
    return dest


def frame_root(data_dir, name):
    return Path(data_dir) / name / "frames_number_16_split_by_number"


def save_frames(events, dest, size):
    from spikingjelly.datasets import integrate_events_by_fixed_frames_number
    frames = integrate_events_by_fixed_frames_number(events, "number", 16, size, size)
    assert int(frames.sum()) == len(events["t"]), "Event integration lost events"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, frames=frames)
    t = events["t"]
    return dict(events=len(t), first_timestamp_us=int(t[0]) if len(t) else None,
                last_timestamp_us=int(t[-1]) if len(t) else None,
                sensor_duration_us=int(t[-1]) - int(t[0]) if len(t) else None)


def prepare(name, data_dir, workers=4):
    data_dir = Path(data_dir).resolve()
    e, split_path, split, files = split_contract(name)
    base = data_dir / name
    base.mkdir(parents=True, exist_ok=True)
    manifest_path = base / "TEST_manifest.json"
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        assert m["split_sha256"] == e["split_sha256"] and m["sample_files"] == files
        if name != "MNIST":
            assert all((frame_root(data_dir, name) / f).is_file() for f in files)
        return m
    sources = json.loads((ROOT / "migration/dataset_sources.json").read_text())
    source_evidence, raw = [], {}
    if name == "MNIST":
        from torchvision.datasets import MNIST
        from torchvision.transforms import ToTensor
        ds = MNIST(str(base), train=False, download=True, transform=ToTensor())
        assert len(ds) == len(files) and ds.class_to_idx == split["class_to_idx"]
        source_evidence = [{"path": str(p.relative_to(data_dir)), "sha256": sha(p)}
                           for p in sorted((base / "MNIST/raw").glob("*")) if p.is_file()]
    elif name in ("NMNIST", "CIFAR10DVS"):
        from spikingjelly.datasets.n_mnist import NMNIST
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
        resources = [r for r in sources[name] if name != "NMNIST" or r["name"] == "Test.zip"]
        # Archives are independent; use at most four concurrent transfers.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            archives = list(pool.map(lambda r: download(r, base / "download"), resources))
        for archive in archives:
            source_evidence.append(dict(path=str(archive.relative_to(data_dir)), md5=digest(archive),
                                        sha256=sha(archive), whole_archive_verified=True))
            with zipfile.ZipFile(archive) as z:
                members = {Path(f).stem: f for f in z.namelist()
                           if f.lower().endswith(".bin" if name == "NMNIST" else ".aedat")}
                selected = [f for f in files if Path(f).stem in members]
                # N-MNIST basenames are globally unique; CIFAR basenames contain class names.
                assert selected, f"No frozen TEST samples in {archive}"
                raw_dir = base / "TEST_raw" / archive.stem
                raw_dir.mkdir(parents=True, exist_ok=True)
                for f in selected:
                    member = members[Path(f).stem]
                    out = raw_dir / Path(member).name
                    if not out.exists():
                        out.write_bytes(z.read(member))
                def convert(f):
                    origin = raw_dir / Path(members[Path(f).stem]).name
                    events = (NMNIST if name == "NMNIST" else CIFAR10DVS).load_origin_data(str(origin))
                    metadata = save_frames(events, frame_root(data_dir, name) / f, 34 if name == "NMNIST" else 128)
                    return f, metadata
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    raw.update(pool.map(convert, selected))
    else:
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
        archive = download(sources[name][0], base / "download")
        source_evidence.append(dict(path=str(archive.relative_to(data_dir)), md5=digest(archive),
                                    sha256=sha(archive), whole_archive_verified=True))
        raw_dir = base / "TEST_raw"
        raw_dir.mkdir(exist_ok=True)
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            test_list = next(m for m in members if Path(m.name).name == "trials_to_test.txt")
            names = [f.strip() for f in tar.extractfile(test_list).read().decode().splitlines() if f.strip()]
            required = {Path(f).name for f in names}
            required.update(Path(f).stem + "_labels.csv" for f in names)
            for member in members:
                if member.isfile() and Path(member.name).name in required:
                    dest = raw_dir / Path(member.name).name
                    if not dest.exists():
                        with tar.extractfile(member) as src, dest.open("wb") as out:
                            import shutil
                            shutil.copyfileobj(src, out)
        def convert_recording(recording):
            origin = raw_dir / Path(recording).name
            events = DVS128Gesture.load_origin_data(str(origin))
            labels = np.loadtxt(raw_dir / (origin.stem + "_labels.csv"), dtype=np.uint32, delimiter=",", skiprows=1)
            seen, output = [0] * 11, {}
            for label, start, end in labels:
                label = int(label) - 1
                filename = f"test/{label}/{origin.stem}_{seen[label]}.npz"
                seen[label] += 1
                mask = (events["t"] >= start) & (events["t"] < end)
                ev = {k: v[mask] for k, v in events.items()}
                output[filename] = save_frames(ev, frame_root(data_dir, name) / filename, 128)
            return output
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for item in pool.map(convert_recording, names):
                raw.update(item)
    hashes = {}
    if name != "MNIST":
        assert set(raw) == set(files), "Restored TEST filenames differ from frozen selection"
        hashes = {f: sha(frame_root(data_dir, name) / f) for f in files}
        with (base / "TEST_raw_event_metadata.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=["sample_file", "events", "first_timestamp_us",
                                                       "last_timestamp_us", "sensor_duration_us"])
            writer.writeheader()
            for f in files:
                writer.writerow(dict(sample_file=f, **raw[f]))
    m = dict(dataset=name, scope="frozen full TEST only; no TRAIN restoration or recalibration",
             samples=len(files), sample_files=files, split_sha256=sha(split_path),
             class_to_idx=split["class_to_idx"], native_frames=16 if name != "MNIST" else None,
             frame_array_format="original float64 event-count arrays; lossless compressed npz",
             spikingjelly_version="0.0.0.0.14", source_evidence=source_evidence, frame_sha256=hashes,
             TEST_restore_driver_sha256=sha(Path(__file__)))
    write_json(manifest_path, m)
    return m


class FrozenTestFrames(Dataset):
    def __init__(self, name, data_dir, verify_frames=True):
        self.name, self.data_dir = name, Path(data_dir)
        self.e, self.split_path, split, self.files = split_contract(name)
        self.manifest = json.loads((self.data_dir / name / "TEST_manifest.json").read_text())
        assert self.manifest["split_sha256"] == self.e["split_sha256"]
        assert self.manifest["sample_files"] == self.files
        self.class_to_idx = split["class_to_idx"]
        if name == "MNIST":
            from torchvision.datasets import MNIST
            from torchvision.transforms import ToTensor
            self.mnist = MNIST(str(self.data_dir / name), train=False, download=False, transform=ToTensor())
        elif verify_frames:
            for f, h in self.manifest["frame_sha256"].items():
                assert sha(frame_root(self.data_dir, name) / f) == h, f"TEST frame changed: {f}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if self.name == "MNIST":
            return self.mnist[index]
        f = self.files[index]
        with np.load(frame_root(self.data_dir, self.name) / f, allow_pickle=False) as array:
            frames = array["frames"].astype(np.float32)
        label = self.class_to_idx[Path(f).parent.name]
        return torch.from_numpy(frames), label


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=("NMNIST", "MNIST", "DVSGesture", "CIFAR10DVS"))
    p.add_argument("--data-dir", type=Path, default=ROOT / "datasets")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    m = prepare(a.dataset, a.data_dir, a.workers)
    print(json.dumps(dict(dataset=a.dataset, samples=m["samples"], split_sha256=m["split_sha256"])), flush=True)
