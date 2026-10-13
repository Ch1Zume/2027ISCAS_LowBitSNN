"""Verify a downloaded project and retained datasets against PAYLOAD_FILES.json."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    args=p.parse_args();root=args.project.resolve()
    record=json.loads(args.manifest.read_text());count=0
    for name,expected in record['files'].items():
        path=(root/name).resolve()
        if root not in path.parents or not path.is_file():raise SystemExit('Missing/invalid: '+name)
        h=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
        if path.stat().st_size!=expected['bytes'] or h.hexdigest()!=expected['sha256']:
            raise SystemExit('Checksum mismatch: '+name)
        count+=1
        if count%2000==0:print('verified',count,flush=True)
    print(json.dumps({'verified_files':count,'git_commit':record['git_commit'],
                      'full_TEST_performed':False}))


if __name__=='__main__':main()
