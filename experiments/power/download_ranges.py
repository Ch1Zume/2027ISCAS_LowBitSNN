"""Parallel resumable HTTP ranges; accept archives only after full MD5 verification."""
import argparse
import json
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from test_data import ROOT, digest

LOCAL = threading.local()


def session():
    if not hasattr(LOCAL, "http"):
        LOCAL.http = requests.Session()
        LOCAL.http.mount("https://", HTTPAdapter(max_retries=Retry(total=4, connect=4, read=0, backoff_factor=1)))
    return LOCAL.http


def fetch(resource, directory, segments, connections):
    dest = directory / resource["name"]
    directory.mkdir(parents=True, exist_ok=True)
    if dest.exists() and digest(dest) == resource["md5"]:
        return
    response = session().head(resource["url"], allow_redirects=True, timeout=60)
    response.raise_for_status()
    size = int(response.headers["Content-Length"])
    url = response.url
    parts = directory / (dest.name + ".ranges")
    parts.mkdir(exist_ok=True)
    width = (size+segments-1)//segments
    old = dest.with_suffix(dest.suffix+".part")
    # Preserve useful bytes from the prior sequential transfer.
    if old.exists():
        with old.open("rb") as source:
            for index in range(segments):
                lo, hi = index*width, min(size, (index+1)*width)
                available = min(max(0, old.stat().st_size-lo), hi-lo)
                p = parts / str(index)
                if available and (not p.exists() or p.stat().st_size < available):
                    source.seek(lo)
                    with p.open("wb") as output:
                        remaining = available
                        while remaining:
                            block = source.read(min(1024**2, remaining))
                            output.write(block)
                            remaining -= len(block)
    def part(index):
        lo, hi = index*width, min(size, (index+1)*width)
        p = parts / str(index)
        tail = parts / f"{index}.tail"
        headers = parts / f"{index}.headers"
        # A failed transfer still contains a valid prefix of its HTTP range.
        existing = p.stat().st_size if p.exists() else 0
        if tail.exists() and headers.exists():
            ranges = re.findall(r"content-range:\s*bytes (\d+)-(\d+)/", headers.read_text().lower())
            if ranges and int(ranges[-1][0]) == lo+existing and 0 < tail.stat().st_size <= int(ranges[-1][1])-int(ranges[-1][0])+1:
                with p.open("ab") as output, tail.open("rb") as source:
                    shutil.copyfileobj(source, output)
                tail.unlink()
        failures = 0
        while True:
            existing = p.stat().st_size if p.exists() else 0
            assert existing <= hi-lo
            if existing == hi-lo:
                return p
            try:
                end = min(hi-1, lo+existing+4*1024**2-1)
                with session().get(url, headers={"Range":f"bytes={lo+existing}-{end}"}, stream=True, timeout=(30,120)) as r:
                    r.raise_for_status()
                    assert r.status_code == 206 and r.headers["Content-Range"].startswith(f"bytes {lo+existing}-{end}/"), "Server ignored Range"
                    received = 0
                    with p.open("ab") as output:
                        for block in r.iter_content(64*1024):
                            received += len(block)
                            assert received <= end-lo-existing+1
                            output.write(block)
                failures = 0
            except (requests.RequestException, AssertionError, subprocess.CalledProcessError):
                if p.exists() and p.stat().st_size > existing:
                    failures = 0
                    continue
                failures += 1
                if failures == 6:
                    raise
                time.sleep(2*failures)
    print(json.dumps(dict(phase="download_ranges", archive=dest.name, bytes=size, segments=segments)), flush=True)
    with ThreadPoolExecutor(max_workers=connections) as pool:
        ready = list(pool.map(part, range(segments)))
    combined = dest.with_suffix(dest.suffix+".verified-part")
    with combined.open("wb") as output:
        for p in ready:
            with p.open("rb") as source:
                shutil.copyfileobj(source, output)
    assert combined.stat().st_size == size
    assert digest(combined) == resource["md5"], "Whole archive MD5 mismatch"
    combined.replace(dest)
    print(json.dumps(dict(phase="archive_verified", archive=dest.name, md5=resource["md5"])), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("DVSGesture", "CIFAR10DVS"), required=True)
    p.add_argument("--data-dir", type=Path, default=ROOT / "datasets")
    p.add_argument("--segments", type=int, default=8)
    p.add_argument("--archives", type=int, default=2)
    p.add_argument("--connections", type=int, default=4)
    a = p.parse_args()
    resources = json.loads((ROOT / "migration/dataset_sources.json").read_text())[a.dataset]
    if a.dataset == "DVSGesture":
        resources = resources[:1]
    with ThreadPoolExecutor(max_workers=a.archives) as pool:
        list(pool.map(lambda r:fetch(r, a.data_dir/a.dataset/"download", a.segments, a.connections), resources))
