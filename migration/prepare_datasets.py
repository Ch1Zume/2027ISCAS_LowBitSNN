"""Rebuild source data using recorded archives, native 16 event-count bins and saved splits."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from common import ROOT, DATASETS, entry, runtime, sha

def md5(path):
    h = hashlib.md5()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=DATASETS, required=True)
    p.add_argument('--data-dir', type=Path, default=ROOT / 'datasets')
    p.add_argument('--download', action='store_true', help='Download missing recorded source archives')
    a = p.parse_args()
    base = a.data_dir.resolve()
    sources = json.loads((ROOT / 'migration/dataset_sources.json').read_text())
    if a.dataset != 'MNIST':
        directory = base / a.dataset / 'download'
        directory.mkdir(parents=True, exist_ok=True)
        for resource in sources[a.dataset]:
            path = directory / resource['name']
            if not path.exists():
                if not a.download:
                    raise FileNotFoundError('Place the archive here or add --download: ' + str(path))
                partial = path.with_name(path.name + '.part')
                subprocess.run(['curl', '-fL', '--retry', '4', '--connect-timeout', '30',
                                '--output', str(partial), resource['url']], check=True)
                if md5(partial) != resource['md5']: raise RuntimeError('Archive MD5 mismatch: ' + str(partial))
                partial.replace(path)
            if md5(path) != resource['md5']: raise RuntimeError('Archive MD5 mismatch: ' + str(path))
    e = entry('lut_final', a.dataset)
    modules = runtime(e)
    cfg = json.loads((ROOT / e['config']).read_text())
    cfg['data_dir'] = str(base)
    data = modules['data'].build_datasets(a.dataset, cfg, download=True)
    assert sha(data[3]) == e['split_sha256'], 'Rebuilt sample ordering/split differs'
    if a.dataset != 'MNIST':
        frames = base / a.dataset / 'frames_number_16_split_by_number'
        actual = [str(f.relative_to(frames)) for f in sorted(frames.rglob('*.npz'))]
        expected = json.loads((ROOT / 'migration/event_file_order.json').read_text())[a.dataset]
        assert actual == expected, 'Rebuilt event sample names differ'
    print(json.dumps(dict(dataset=a.dataset, counts=[len(d) for d in data[:3]], split_sha256=sha(data[3]),
                          full_TEST_evaluated=False)))

if __name__ == '__main__': main()
