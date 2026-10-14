"""Restart a new LUT TRAIN/VALID run; warm starts initialize a new optimizer."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from common import ROOT, DATASETS, entry, verify, write_new

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=DATASETS, required=True)
    p.add_argument('--data-dir', type=Path, default=ROOT / 'datasets')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=('warm-start-final', 'from-scratch'), default='warm-start-final')
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--baseline-comparison', choices=('exact', 'counts'), default='exact')
    a = p.parse_args()
    if a.out.exists(): p.error('Choose a new TRAIN/VALID run directory')
    if not 1 <= a.epochs <= 60: p.error('Epoch budget must be 1..60')
    files = verify()
    out = a.out.resolve()
    out.mkdir(parents=True)
    env = dict(os.environ, SNN_PROJECT_ROOT=str(ROOT), SNN_DATA_DIR=str(a.data_dir.resolve()),
               SNN_TEACHER_CACHE_DIR=str(out / 'teacher_cache'), TMPDIR='/tmp')
    replay = out / ('baseline_replay_' + a.dataset + '.json')
    subprocess.run([sys.executable, str(ROOT / 'scripts/replay.py'), '--bundle', 'compressed',
                    '--dataset', a.dataset, '--data-dir', str(a.data_dir.resolve()), '--out', str(replay),
                    '--data-scope', 'all', '--workers', str(a.workers),
                    '--comparison', a.baseline_comparison], check=True, env=env)
    write_new(out / 'dependency_audit.json', dict(verified_payload_files=files,
              scope='Retained completed compression bundle integrity and independent full TEST replay',
              baseline_replay=str(replay), project_root=str(ROOT)))
    cache = out / 'teacher_cache' / a.dataset / 'teacher_train_logits.pth'
    subprocess.run([sys.executable, str(ROOT / 'scripts/teacher_cache.py'), '--dataset', a.dataset,
                    '--data-dir', str(a.data_dir.resolve()), '--out', str(cache)], check=True, env=env)
    command = [sys.executable, str(ROOT / 'experiments/stdp/run.py'), '--out', str(out),
               '--datasets', a.dataset, '--conditions', 'lut_native_trace', '--epochs', str(a.epochs),
               '--patience', '6', '--teacher-source', 'original_cached', '--workers', str(a.workers), '--skip-test']
    if a.dataset == 'CIFAR10DVS':
        command += ['--spatial-student', '--stdp-granularity', 'group', '--stdp-lr', '0.01',
                    '--bp-rate-scale', '1.5', '--initial-row-sum', '4', '--augment-spatial']
    else:
        command += ['--stdp-lr', '0.001', '--bp-rate-scale', '0.1']
    if a.mode == 'from-scratch': command.append('--from-scratch')
    else:
        command += ['--initial-checkpoint', str(ROOT / entry('lut_final', a.dataset)['checkpoint'])]
    write_new(out / 'training_launch.json', dict(mode=a.mode, exact_interrupted_resume=False,
              optimizer='new', full_train_epoch_budget=a.epochs, command=command,
              data_dir=str(a.data_dir.resolve()), TEST_deferred=True))
    subprocess.run(command, check=True, env=env)
    print('TRAIN/VALID finished; selections_before_TEST.json freezes the candidates. TEST is deferred.')

if __name__ == '__main__': main()
