"""Verify retained file hashes, optionally load all 12 final model conditions on CPU."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from common import ROOT, verify, write_new

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', action='store_true')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    files = verify()
    checks = []
    if a.checkpoints:
        for e in json.loads((ROOT / 'migration/catalog.json').read_text())['entries']:
            command = [sys.executable, str(ROOT / 'migration/inspect_checkpoint.py'),
                       '--bundle', e['bundle'], '--dataset', e['dataset']]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            record = json.loads(result.stdout.strip().splitlines()[-1])
            checks.append(record)
            print(json.dumps(dict(bundle=e['bundle'], dataset=e['dataset'], state_load='passed')), flush=True)
    result = dict(verified_payload_files=files, checkpoint_checks=checks, full_TEST_performed=False,
                  new_training_steps=0)
    if a.out: write_new(a.out, result)
    print(json.dumps(dict(verified_payload_files=files, checkpoint_checks=len(checks))))

if __name__ == '__main__': main()
