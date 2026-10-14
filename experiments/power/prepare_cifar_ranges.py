"""Read frozen CIFAR TEST ZIP members by HTTP range, with source identity and CRC checks.

Whole-archive MD5 is not recomputed for this path. The server ETag must match the
recorded official archive MD5; ZIP CRC, sample SHA256 and frame SHA256 are checked.
"""
import argparse
import csv
import io
import json
import struct
import subprocess
import threading
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from test_data import ROOT, split_contract, frame_root, save_frames, write_json
from common import sha

LOCAL = threading.local()


def session():
    if not hasattr(LOCAL, "http"):
        LOCAL.http = requests.Session()
        retry = Retry(total=6, connect=6, read=6, status=6, backoff_factor=1,
                      status_forcelist=(429,500,502,503,504), allowed_methods=("GET","HEAD"))
        LOCAL.http.mount("https://", HTTPAdapter(max_retries=retry))
    return LOCAL.http


class RemoteArchive(io.RawIOBase):
    def __init__(self, resource, cache):
        self.resource, self.cache = resource, cache
        cache.mkdir(parents=True, exist_ok=True)
        head = session().head(resource["url"], allow_redirects=True, timeout=60)
        head.raise_for_status()
        self.size = int(head.headers["Content-Length"])
        self.etag = head.headers["ETag"].strip('"')
        self.version = head.headers.get("x-amz-version-id")
        assert self.etag == resource["md5"], "Official source identity differs"
        self.pos = 0

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos+offset if whence == 1 else self.size+offset
        return self.pos

    def read(self, size=-1):
        size = self.size-self.pos if size < 0 else min(size, self.size-self.pos)
        if size <= 0:
            return b""
        result = self.range(self.pos, self.pos+size-1, self.cache/f"metadata_{self.pos}_{size}")
        self.pos += size
        return result

    def range(self, lo, hi, dest):
        if not dest.exists() or dest.stat().st_size != hi-lo+1:
            for attempt in range(5):
                try:
                    r = session().get(self.resource["url"], headers={"Range":f"bytes={lo}-{hi}"}, timeout=(30,120))
                    r.raise_for_status()
                    assert r.status_code == 206 and r.headers["Content-Range"].startswith(f"bytes {lo}-{hi}/")
                    assert r.headers["ETag"].strip('"') == self.etag
                    assert len(r.content) == hi-lo+1
                    dest.write_bytes(r.content)
                    break
                except requests.RequestException:
                    if attempt == 4:
                        raise
        return dest.read_bytes()


def prepare(data_dir, workers=8):
    from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
    name = "CIFAR10DVS"
    e, split_path, split, files = split_contract(name)
    base = Path(data_dir)/name
    manifest = base/"TEST_manifest.json"
    if manifest.exists():
        return
    by_stem = {Path(f).stem:f for f in files}
    resources = json.loads((ROOT/"configs/dataset_sources.json").read_text())[name]
    metadata, evidence, jobs = {}, [], []
    def catalog(resource):
        remote = RemoteArchive(resource, base/"TEST_remote_zip"/resource["name"])
        with zipfile.ZipFile(remote) as z:
            selected = [info for info in z.infolist() if Path(info.filename).stem in by_stem]
        assert len(selected) == 100
        source = dict(url=resource["url"], archive_name=resource["name"],
            recorded_archive_md5=resource["md5"], server_ETag=remote.etag,
            server_version_id=remote.version, archive_bytes=remote.size,
            source_identity_matches_recorded_MD5=True, whole_archive_verified=False,
            scope="selected TEST members only; ZIP CRC verified; no full archive MD5 recomputation")
        return source, [(remote,info,by_stem[Path(info.filename).stem]) for info in selected]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for source, selected in pool.map(catalog, resources):
            evidence.append(source)
            jobs.extend(selected)
    def convert(job):
        remote, info, filename = job
        raw = base/"TEST_raw"/(Path(filename).stem+".aedat")
        raw.parent.mkdir(parents=True, exist_ok=True)
        if not raw.exists() or raw.stat().st_size != info.file_size or zlib.crc32(raw.read_bytes()) != info.CRC:
            dest = remote.cache/(Path(filename).stem+".compressed")
            data = remote.range(info.header_offset, min(remote.size-1, info.header_offset+info.compress_size+1024), dest)
            assert data[:4] == b"PK\x03\x04"
            local = struct.unpack("<4s5H3I2H", data[:30])
            assert not (local[2] & 1), "Encrypted ZIP member"
            offset = 30+local[-2]+local[-1]
            compressed = data[offset:offset+info.compress_size]
            decoded = zlib.decompress(compressed, -15) if info.compress_type == zipfile.ZIP_DEFLATED else compressed
            assert info.compress_type in (zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED)
            assert len(decoded) == info.file_size and zlib.crc32(decoded) == info.CRC
            raw.write_bytes(decoded)
        events = CIFAR10DVS.load_origin_data(str(raw))
        meta = save_frames(events, frame_root(data_dir, name)/filename, 128)
        meta.update(raw_sample_sha256=sha(raw), ZIP_CRC32=info.CRC)
        return filename, meta
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index,(f,m) in enumerate(pool.map(convert,jobs),1):
            metadata[f] = m
            if index % 100 == 0:
                print(json.dumps(dict(phase="CIFAR_TEST_restore", samples=index, total=1000)), flush=True)
    assert set(metadata) == set(files)
    with (base/"TEST_raw_event_metadata.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_file", "events", "first_timestamp_us", "last_timestamp_us",
                                              "sensor_duration_us", "raw_sample_sha256", "ZIP_CRC32"])
        writer.writeheader()
        for filename in files:
            writer.writerow(dict(sample_file=filename, **metadata[filename]))
    write_json(manifest, dict(dataset=name, scope="frozen full TEST only; no TRAIN restoration or recalibration",
        samples=len(files), sample_files=files, split_sha256=sha(split_path), class_to_idx=split["class_to_idx"],
        native_frames=16, frame_array_format="original float64 event-count arrays; lossless compressed npz",
        spikingjelly_version="0.0.0.0.14", source_evidence=evidence,
        TEST_restore_driver_sha256=sha(Path(__file__)),
        raw_sample_sha256={f:metadata[f]["raw_sample_sha256"] for f in files},
        frame_sha256={f:sha(frame_root(data_dir,name)/f) for f in files}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT/"datasets")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args(); prepare(a.data_dir,a.workers)
