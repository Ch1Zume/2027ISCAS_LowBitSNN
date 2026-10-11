"""Escalate native-forward LUT recovery using VALID, including new training."""
import argparse
import json
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
    args = parser.parse_args()
    root, previous = args.out.resolve(), args.previous.resolve()
    root.mkdir(exist_ok=False)
    driver = root / 'recovery_run_frozen.py'
    shutil.copy2(ROOT / 'experiments/stdp/run.py', driver)
    stages = []

    def status(phase, **values):
        record = {'at': datetime.now(timezone.utc).isoformat(), 'pid': __import__('os').getpid(),
                  'phase': phase, **values}
        write(root / 'status.json', record)
        print(json.dumps(record), flush=True)

    policy = {'goal': 'CIFAR10DVS FP32 >=75% full TEST; original 3M–5M physical size retained',
              'stdp': 'native forward; same signed 5-bit LUT; whole competition groups assigned STDP, count remains 48/768; reset row targets to actual learned sums at reassignment',
              'teacher': 'SHA-bound original TRAIN-only teacher logits; large teacher not part of student inference',
              'selection': 'complete VALID accuracy then loss; no TEST ranking',
              'stages': [
                  {'name': 'group_head_only', 'epochs_max': 12, 'stdp_lr': 0.01, 'bp_rate_scale': 0.5,
                   'freeze': 'backbone and BP hidden; classifier BP and winner STDP still active'},
                  {'name': 'group_hidden_head', 'epochs_max': 16, 'stdp_lr': 0.01, 'bp_rate_scale': 0.5,
                   'freeze': 'backbone only'},
                  {'name': 'group_scratch_augmented', 'epochs_max': 40, 'stdp_lr': 0.01, 'bp_rate_scale': 1.5,
                   'initialization': 'random compact model, hidden initial row sum 4',
                   'training': 'all BP components train; original-cache KD, TRAIN-only flip and 4px shift'}],
              'scratch_trigger': 'best warm-start complete VALID <77.5%; also train from scratch if the fixed selected full TEST misses 75%',
              'workers': 4, 'new_training_authorized_by_user': True, 'previous': str(previous)}
    write(root / 'policy.json', policy)
    status('waiting_for_previous_GPU_campaign', previous=str(previous))
    while True:
        path = previous / 'optimization_status.json'
        if path.exists():
            state = json.loads(path.read_text())
            if state['phase'] in ('completed', 'accuracy_goals_open'):
                break
            if state['phase'] == 'failed':
                raise RuntimeError(f'Previous campaign failed: {state}')
        time.sleep(10)

    def trial(name, epochs, *, scratch=False, freeze_hidden=False):
        folder = root / name
        folder.mkdir(exist_ok=False)
        for filename in ('dependency_audit.json', 'baseline_replay_CIFAR10DVS.json'):
            shutil.copy2(previous / filename, folder / filename)
        command = [sys.executable, str(driver), '--out', str(folder), '--datasets', 'CIFAR10DVS',
                   '--conditions', 'lut_native_trace', '--epochs', str(epochs), '--patience', '6',
                   '--stdp-lr', '0.01', '--bp-rate-scale', '1.5' if scratch else '0.5',
                   '--teacher-source', 'original_cached', '--stdp-granularity', 'group',
                   '--workers', '4', '--skip-test']
        if scratch:
            command += ['--from-scratch', '--initial-row-sum', '4', '--augment-spatial']
        else:
            command += ['--freeze-backbone', '--freeze-bn']
            if freeze_hidden:
                command.append('--freeze-bp-hidden')
        status('full_TRAIN_VALID', stage=name, epochs_max=epochs, from_scratch=scratch,
               log=str(folder / 'campaign.log'))
        with (folder / 'campaign.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        stages.append(folder)
        return folder

    def best():
        candidates = []
        for folder in stages:
            selection = json.loads((folder / 'CIFAR10DVS/lut_native_trace/validation.json').read_text())
            score = selection['validation']
            candidates.append((score['top1_accuracy'], -score['loss'], folder, selection))
        return max(candidates, key=lambda v: v[:2])

    def test_selected():
        _, _, folder, selection = best()
        record = {'bundle': str(folder), 'checkpoint_sha256': selection['checkpoint_sha256'],
                  'selected_epoch': selection['selected_block'], 'validation': selection['validation'],
                  'all_candidates': [str(v) for v in stages], 'selection': 'VALID only'}
        write(root / 'selection_before_TEST.json', record)
        status('selected_full_TEST', selection=record)
        command = [sys.executable, str(folder / 'run_frozen.py'), '--out', str(folder),
                   '--dataset', 'CIFAR10DVS', '--condition', 'lut_native_trace', '--test-only']
        with (root / 'full_TEST.log').open('a') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((folder / 'CIFAR10DVS/lut_native_trace/accuracy.json').read_text())
        write(root / 'result.json', result)
        return result

    try:
        trial('group_head_only', 12, freeze_hidden=True)
        trial('group_hidden_head', 16)
        scratch_done = best()[0] < 0.775
        if scratch_done:
            trial('group_scratch_augmented', 40, scratch=True)
        result = test_selected()
        if result['test']['float32']['top1_accuracy'] < 0.75 and not scratch_done:
            shutil.copy2(root / 'result.json', root / 'warm_result_before_retraining.json')
            shutil.copy2(root / 'selection_before_TEST.json', root / 'warm_selection_before_retraining.json')
            trial('group_scratch_augmented', 40, scratch=True)
            result = test_selected()
        chosen = Path(json.loads((root / 'selection_before_TEST.json').read_text())['bundle'])
        replay = [sys.executable, str(ROOT / 'experiments/stdp/reproduce.py'), '--bundle', str(chosen),
                  '--dataset', 'CIFAR10DVS', '--condition', 'lut_native_trace',
                  '--out', str(root / 'independent_selected_replay_CIFAR10DVS.json')]
        with (root / 'full_TEST.log').open('a') as log:
            subprocess.run(replay, stdout=log, stderr=subprocess.STDOUT, check=True)
        status('completed' if result['test']['float32']['top1_accuracy'] >= 0.75 else 'accuracy_goal_open',
               test_accuracy={k:v['top1_accuracy'] for k,v in result['test'].items()},
               from_scratch_trained=scratch_done or any('scratch' in s.name for s in stages))
    except Exception as error:
        status('failed', error=str(error), completed_stages=[str(s) for s in stages])
        raise


if __name__ == '__main__':
    main()
