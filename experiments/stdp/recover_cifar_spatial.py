"""Conditional 4.679M spatial-readout recovery with TRAIN-only distillation."""
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
    write(root / 'policy.json', {
        'goal': 'CIFAR10DVS FP32 >=75% on complete TEST', 'trigger': 'prior fixed condition misses goal',
        'model': dict(channels=[32,64,128,256,448], hidden=[256], spatial_readout=4, T=8,
                      parameters=4679375, STDP_hidden=16, BP_hidden=240),
        'warm_initialization': 'trained compressed channel/group selection by classifier discrimination; coarse 2x2 weights copied evenly into 4x4 subcells; positive row mass restored',
        'warm_training': '24 epochs maximum; BP scale 0.5; STDP 0.01; BN frozen; spatial augmentation',
        'scratch_trigger': 'warm VALID <77.5% or selected warm full TEST misses 75%',
        'scratch_training': '40 epochs maximum; random network, row sum 4; BP scale 1.5; STDP 0.01; spatial augmentation',
        'teacher': 'original large TRAIN-only cache at T=8; teacher absent from inference',
        'selection': 'complete VALID accuracy then loss; no TEST ranking', 'patience': 6,
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

    def trial(scratch=False):
        name = 'spatial_scratch' if scratch else 'spatial_transfer'
        folder = root / name
        folder.mkdir()
        for filename in ('dependency_audit.json', 'baseline_replay_CIFAR10DVS.json'):
            shutil.copy2(evidence / filename, folder / filename)
        command = [sys.executable, str(driver), '--out', str(folder), '--datasets', 'CIFAR10DVS',
                   '--conditions', 'lut_native_trace', '--epochs', '40' if scratch else '24', '--patience', '6',
                   '--stdp-lr', '0.01', '--bp-rate-scale', '1.5' if scratch else '0.5',
                   '--teacher-source', 'original_cached', '--stdp-granularity', 'group',
                   '--workers', '4', '--spatial-student', '--augment-spatial', '--skip-test']
        command += ['--from-scratch', '--initial-row-sum', '4'] if scratch else ['--freeze-bn']
        status('full_TRAIN_VALID', stage=name, log=str(folder / 'campaign.log'))
        with (folder / 'campaign.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        selection = json.loads((folder / 'CIFAR10DVS/lut_native_trace/validation.json').read_text())
        score = selection['validation']
        candidates.append((score['top1_accuracy'], -score['loss'], folder, selection))

    def test_selected():
        _, _, chosen, selection = max(candidates, key=lambda v: v[:2])
        record = dict(bundle=str(chosen), condition='lut_native_trace', selected_epoch=selection['selected_block'],
                      checkpoint_sha256=selection['checkpoint_sha256'], validation=selection['validation'],
                      selection='VALID only', all_candidates=[str(v[2]) for v in candidates])
        write(root / 'selection_before_TEST.json', record)
        status('selected_full_TEST', selection=record)
        with (root / 'full_TEST.log').open('a') as log:
            subprocess.run([sys.executable, str(chosen / 'run_frozen.py'), '--out', str(chosen),
                            '--dataset', 'CIFAR10DVS', '--condition', 'lut_native_trace', '--test-only'],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((chosen / 'CIFAR10DVS/lut_native_trace/accuracy.json').read_text())
        write(root / 'result.json', result)
        return result, chosen

    try:
        trial()
        scratch_done = candidates[0][0] < 0.775
        if scratch_done:
            trial(scratch=True)
        result, chosen = test_selected()
        if result['test']['float32']['top1_accuracy'] < 0.75 and not scratch_done:
            shutil.copy2(root / 'result.json', root / 'warm_result_before_retraining.json')
            shutil.copy2(root / 'selection_before_TEST.json', root / 'warm_selection_before_retraining.json')
            trial(scratch=True)
            result, chosen = test_selected()
        with (root / 'full_TEST.log').open('a') as log:
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
