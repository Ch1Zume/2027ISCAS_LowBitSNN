"""Restore the retained Linux environment into a fresh project checkout.

The original server environment is never modified. Requires Linux x86_64 and a
compatible NVIDIA driver for CUDA; Windows users run this inside Linux/WSL2.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):
            h.update(b)
    return h.hexdigest()


def run(argv, **kwargs):
    subprocess.run([str(x) for x in argv], check=True, **kwargs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archives',type=Path,required=True)
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--require-cuda',action='store_true')
    args=p.parse_args()
    if platform.system()!='Linux' or platform.machine() not in ('x86_64','AMD64'):
        raise SystemExit('This archive is for Linux x86_64. Use Linux or WSL2, not native Windows Python.')
    root=args.project.resolve(); archives=args.archives.resolve()
    base=root/'.runtime/base';venv=root/'.venv'
    if base.exists() or venv.exists():
        raise SystemExit('Restore into a fresh checkout: existing .runtime/base or .venv is preserved.')
    index_path=archives/'DOWNLOAD_INDEX.json'
    index=json.loads(index_path.read_text()) if index_path.exists() else None
    names=['SNN-runtime-base-20261013.tar.gz','SNN-runtime-overlay-20261013.tar.gz']
    for name in names:
        archive=archives/name
        if not archive.is_file():raise SystemExit('Missing '+str(archive))
        if index:
            expected=next(a for a in index['archives'] if a['name']==name)
            if sha(archive)!=expected['sha256']:raise SystemExit('Archive checksum mismatch: '+name)
    base.mkdir(parents=True)
    # Archives are generated from the retained trusted environment; symbolic
    # links in conda/venv are intentional, including prefix-fixup input paths.
    run(['tar','-xzf',archives/names[0],'-C',base])
    env=dict(os.environ,TMPDIR='/tmp',PYTHONNOUSERSITE='1')
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    run([base/'bin/python',base/'bin/conda-unpack'],env=env)
    run(['tar','-xzf',archives/names[1],'-C',root])
    # Replace only old venv launchers/configuration; retain all overlay packages.
    for name in ['python','python3','python3.10']:
        q=venv/'bin'/name
        if q.is_symlink() or q.exists():q.unlink()
    run([base/'bin/python','-m','venv','--without-pip','--system-site-packages',venv],env=env)
    py=venv/'bin/python'
    for q in (venv/'bin').iterdir():
        if not q.is_file() or q.is_symlink():continue
        data=q.read_bytes()
        first,sep,rest=data.partition(b'\n')
        if first.startswith(b'#!') and b'python' in first:
            q.write_bytes(b'#!'+str(py).encode()+b'\n'+rest)
    probe='''import json,sys,torch,torchvision,numpy,scipy,matplotlib,spikingjelly
from pathlib import Path
from importlib.metadata import version
names=['torch','torchvision','numpy','scipy','matplotlib','spikingjelly','PyYAML','pytest']
r={'python':sys.version,'base_prefix':sys.base_prefix,'prefix':sys.prefix,
 'packages':{n:version(n) for n in names},'cuda_available':torch.cuda.is_available(),
 'torch_file':torch.__file__,'torchvision_file':torchvision.__file__,
 'spikingjelly_file':spikingjelly.__file__}
print(json.dumps(r,indent=2))
'''
    result=subprocess.run([str(py),'-c',probe],check=True,capture_output=True,text=True,env=env)
    record=json.loads(result.stdout)
    if args.require_cuda and not record['cuda_available']:
        raise SystemExit('Environment restored, but CUDA is unavailable; check the host GPU driver.')
    record['archive_sha256']={n:sha(archives/n) for n in names}
    record['full_TEST_performed']=False
    (root/'migration').mkdir(exist_ok=True)
    (root/'migration/ENVIRONMENT_RESTORE_RECEIPT.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
