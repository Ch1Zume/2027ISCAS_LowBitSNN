"""Package retained data and large models without deleting source files.

Run payload first. Run finalize after the final Git commit, then publish the
generated DOWNLOAD_INDEX.json through the existing SSH server. Packages are
split into independent 512 MiB transport chunks for resumable downloads.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / 'migration/reorganize_20261013'
OUT = ROOT / 'migration/releases/20261013'
CHUNK = 512 * 1024**2


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def package(name, files):
    archive = OUT / name
    manifest_file = META / (name + '.files.json')
    if archive.exists() and manifest_file.exists():
        existing = json.loads(manifest_file.read_text())
        assert sha(archive) == existing['archive_sha256']
        print('REUSE', name, flush=True)
        return
    records = {}
    tmp = archive.with_name(archive.name + '.partial')
    with tarfile.open(tmp, 'w:gz', compresslevel=1) as tar:
        for i, p in enumerate(sorted(set(files))):
            relative = p.relative_to(ROOT).as_posix()
            before = p.stat()
            digest = sha(p)
            tar.add(p, arcname='Algorithm/' + relative, recursive=False)
            after = p.stat()
            if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
                raise RuntimeError('Source changed while packing: ' + relative)
            records[relative] = {'bytes':before.st_size, 'sha256':digest}
            if i % 2000 == 0:
                print(name, 'files', i + 1, flush=True)
    tmp.replace(archive)
    write(manifest_file, {'archive':name, 'archive_sha256':sha(archive),
                         'sha256_scope':'original extracted files', 'files':records})
    print('PACKED', name, archive.stat().st_size, len(records), flush=True)


def payload():
    OUT.mkdir(parents=True, exist_ok=True)
    models = [p for p in (ROOT / 'results/accuracy').rglob('*.pth')
              if p.stat().st_size >= 100 * 1024**2]
    assert len(models) == 4
    package('SNN-large-models-20261013.tar.gz', models)
    for dataset in ['MNIST', 'NMNIST', 'CIFAR10DVS', 'DVSGesture']:
        files = [p for p in (ROOT / 'datasets' / dataset).rglob('*') if p.is_file()]
        package('SNN-data-' + dataset + '-20261013.tar.gz', files)
    caches = []
    for top in ['experiments', 'snn', 'test', 'results', 'migration', '.pytest_cache']:
        for base, dirs, names in os.walk(ROOT / top):
            dirs[:] = [d for d in dirs if not (Path(base) == ROOT / 'migration' and d in {'releases'})]
            if 'work' in Path(base).parts and 'reorganize_20261013' in Path(base).parts:
                dirs[:] = []
                continue
            for n in names:
                p = Path(base) / n
                if '__pycache__' in p.parts or '.pytest_cache' in p.parts:
                    caches.append(p)
    package('SNN-project-caches-20261013.tar.gz', caches)


def finalize():
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT).strip():
        raise RuntimeError('Commit all tracked changes before producing a commit-bound delivery.')
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    source = OUT / 'SNN-project-20261013.zip'
    subprocess.run(['git','archive','--format=zip','--prefix=Algorithm/',
                    '-o',str(source),commit], cwd=ROOT, check=True)
    bundle = OUT / 'SNN-git-20261013.bundle'
    if bundle.exists():
        bundle.unlink()
    subprocess.run(['git','bundle','create',str(bundle),'refs/heads/main'], cwd=ROOT, check=True)
    subprocess.run(['git','bundle','verify',str(bundle)], cwd=ROOT, check=True)
    archives = [source, bundle] + sorted(OUT.glob('*.tar.gz'))
    entries = []
    for archive in archives:
        digest = sha(archive)
        parts = []
        with archive.open('rb') as f:
            i = 0
            while True:
                data = f.read(CHUNK)
                if not data:
                    break
                name = archive.name + '.part%03d' % i
                part = OUT / name
                part.write_bytes(data)
                parts.append({'name':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
                i += 1
        runtime = archive.name.startswith('SNN-runtime-')
        kind = ('source' if archive == source else 'git' if archive == bundle else
                'runtime' if runtime else 'payload')
        entries.append({'name':archive.name,'bytes':archive.stat().st_size,
                        'sha256':digest,'kind':kind,'parts':parts})
        print('INDEXED',archive.name,archive.stat().st_size,len(parts),flush=True)
    all_files = {}
    for manifest in sorted(META.glob('*.files.json')):
        for name, record in json.loads(manifest.read_text())['files'].items():
            if name in all_files:
                assert all_files[name] == record
            all_files[name] = record
    tracked = subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    for name in filter(None,tracked):
        p = ROOT / name
        all_files[name] = {'bytes':p.stat().st_size,'sha256':sha(p)}
    subprocess.run(['git','diff','--quiet','HEAD'],cwd=ROOT,check=True)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==commit
    files_manifest = OUT / 'PAYLOAD_FILES.json'
    write(files_manifest, {'schema':1,'git_commit':commit,'files':all_files,
                          'runtime_scope':'archive SHA256 on Windows; relocation and imports on Linux'})
    index = {'schema':1,'release':'20261013','git_commit':commit,
             'github':'https://github.com/Ch1Zume/2027ISCAS_LowBitSNN',
             'branch':'main','destination':r'E:\IMECAS_SNN',
             'project_directory':r'E:\IMECAS_SNN\20261013\Algorithm',
             'remote_directory':str(OUT), 'archives':entries,
             'payload_manifest':{'name':files_manifest.name,'sha256':sha(files_manifest),
                                 'bytes':files_manifest.stat().st_size},
             'compressed_bytes':sum(x['bytes'] for x in entries),
             'payload_bytes':sum(x['bytes'] for x in all_files.values()),
             'Windows_execution':'backup/download only; Linux CUDA environment uses Linux/WSL2',
             'Windows_receipt':'not present until client verifies download and extracted payload'}
    write(OUT / 'DOWNLOAD_INDEX.json', index)
    shutil.copy2(ROOT / 'migration/download.ps1', OUT / 'download.ps1')
    (OUT / 'SHA256SUMS').write_text(''.join(x['sha256']+'  '+x['name']+'\n' for x in entries))
    print('DELIVERY_READY',commit,'archives',len(entries),'payload_files',len(all_files),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['payload','finalize'])
    args=p.parse_args()
    (payload if args.mode == 'payload' else finalize)()
