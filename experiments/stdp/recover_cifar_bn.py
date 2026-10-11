"""Test TRAIN-only BN moment recalibration as an explicit CIFAR condition."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    root, previous, evidence = (p.resolve() for p in (args.out, args.previous, args.evidence))
    root.mkdir(exist_ok=False)
    driver = root / 'recovery_run_frozen.py'
    shutil.copy2(ROOT / 'experiments/stdp/run.py', driver)
    stages = [dict(name='calibrated_T8', steps=8), dict(name='calibrated_T16', steps=16)]
    write(root / 'policy.json', {
        'goal': 'CIFAR10DVS FP32 >=75% on complete TEST', 'parameters': 4717903,
        'trigger': 'prior fixed condition misses goal', 'stages': stages,
        'calibration': 'one complete shuffled native TRAIN pass; cumulative BN moments; all weights, WTA threshold state and training RNG frozen',
        'training': '12 epochs maximum, patience 6, scale 0.1, STDP 0.01; calibrated BN frozen; native TRAIN without spatial augmentation',
        'teacher': 'original large TRAIN-only cache, original teacher T=8',
        'selection': 'complete VALID accuracy then loss, including calibrated initialization; no TEST ranking',
        'waiting_for': str(previous),
    })

    def status(phase, **values):
        record = dict(at=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), phase=phase, **values)
        write(root / 'status.json', record)
        print(json.dumps(record), flush=True)

    status('waiting_for_previous_GPU_campaign')
    while True:
        f = previous / 'status.json'
        if f.exists():
            state = json.loads(f.read_text())
            if state['phase'] in ('completed', 'accuracy_goal_open'):
                break
            if state['phase'] == 'not_needed_previous_goal_met':
                status('not_needed_previous_goal_met')
                return
            if state['phase'] == 'failed':
                status('failed', error='Previous campaign failed')
                raise RuntimeError('Previous campaign failed')
        time.sleep(10)
    if json.loads((previous / 'result.json').read_text())['test']['float32']['top1_accuracy'] >= 0.75:
        status('not_needed_previous_goal_met')
        return
    candidates = []
    try:
        for spec in stages:
            folder = root / spec['name']
            folder.mkdir()
            for filename in ('dependency_audit.json', 'baseline_replay_CIFAR10DVS.json'):
                shutil.copy2(evidence / filename, folder / filename)
            command = [sys.executable, str(driver), '--out', str(folder), '--datasets', 'CIFAR10DVS',
                       '--conditions', 'lut_native_trace', '--epochs', '12', '--patience', '6',
                       '--stdp-lr', '0.01', '--bp-rate-scale', '0.1', '--steps', str(spec['steps']),
                       '--teacher-source', 'original_cached', '--stdp-granularity', 'group',
                       '--workers', '4', '--calibrate-bn', '--freeze-bn', '--skip-test']
            status('TRAIN_calibration_and_full_TRAIN_VALID', stage=spec['name'], log=str(folder / 'campaign.log'))
            with (folder / 'campaign.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            selection = json.loads((folder / 'CIFAR10DVS/lut_native_trace/validation.json').read_text())
            score = selection['validation']
            candidates.append((score['top1_accuracy'], -score['loss'], folder, selection))
        _, _, chosen, selection = max(candidates, key=lambda v: v[:2])
        record = dict(bundle=str(chosen), condition='lut_native_trace', selected_epoch=selection['selected_block'],
                      checkpoint_sha256=selection['checkpoint_sha256'], validation=selection['validation'],
                      selection='VALID only', all_candidates=[str(v[2]) for v in candidates])
        write(root / 'selection_before_TEST.json', record)
        status('selected_full_TEST', selection=record)
        with (root / 'full_TEST.log').open('w') as log:
            subprocess.run([sys.executable, str(chosen / 'run_frozen.py'), '--out', str(chosen),
                            '--dataset', 'CIFAR10DVS', '--condition', 'lut_native_trace', '--test-only'],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((chosen / 'CIFAR10DVS/lut_native_trace/accuracy.json').read_text())
            write(root / 'result.json', result)
            subprocess.run([sys.executable, str(ROOT / 'experiments/stdp/reproduce.py'), '--bundle', str(chosen),
                            '--dataset', 'CIFAR10DVS', '--condition', 'lut_native_trace',
                            '--out', str(root / 'independent_selected_replay_CIFAR10DVS.json')],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        status('completed' if result['test']['float32']['top1_accuracy'] >= 0.75 else 'accuracy_goal_open',
               test_accuracy={k:v['top1_accuracy'] for k,v in result['test'].items()})
    except Exception as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
