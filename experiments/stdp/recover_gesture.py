"""VALID-selected Gesture LUT recovery, queued behind a single-GPU campaign."""
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
    registered = [
        dict(name='head_slow', epochs=8, scale=0.1, stdp_lr=0.001, freeze_backbone=True, freeze_hidden=True),
        dict(name='hidden_head_slow', epochs=12, scale=0.1, stdp_lr=0.001, freeze_backbone=True, freeze_hidden=False),
        dict(name='all_layers_slow', epochs=12, scale=0.1, stdp_lr=0.001, freeze_backbone=False, freeze_hidden=False),
    ]
    write(root / 'policy.json', {
        'goal': 'DVSGesture FP32 >=95% on complete TEST', 'parameters': 4456656,
        'forward': 'frozen compact native coding, T=16; Bernoulli sampling only in local learning trace',
        'selection': 'complete VALID accuracy then loss; candidates registered before TEST, never TEST ranking',
        'teacher': 'SHA-bound original TRAIN-only logits cache', 'homeostasis': 'exponential, original settings',
        'stages': registered, 'patience': 5, 'freeze_bn': True,
        'existing_v2_candidate': 'not ranked against these candidates using TEST',
        'waiting_for': str(previous), 'evidence': str(evidence),
    })

    def status(phase, **values):
        record = dict(at=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), phase=phase, **values)
        write(root / 'status.json', record)
        print(json.dumps(record), flush=True)

    status('waiting_for_previous_GPU_campaign')
    while True:
        state_file = previous / 'status.json'
        if state_file.exists():
            state = json.loads(state_file.read_text())
            if state['phase'] in ('completed', 'accuracy_goal_open'):
                break
            if state['phase'] == 'failed':
                status('failed', error='Previous GPU campaign failed', previous_status=state)
                raise RuntimeError('Previous GPU campaign failed')
        time.sleep(10)
    candidates = []
    try:
        for spec in registered:
            folder = root / spec['name']
            folder.mkdir()
            for filename in ('dependency_audit.json', 'baseline_replay_DVSGesture.json'):
                shutil.copy2(evidence / filename, folder / filename)
            command = [sys.executable, str(driver), '--out', str(folder), '--datasets', 'DVSGesture',
                       '--conditions', 'lut_native_trace', '--epochs', str(spec['epochs']), '--patience', '5',
                       '--stdp-lr', str(spec['stdp_lr']), '--bp-rate-scale', str(spec['scale']),
                       '--teacher-source', 'original_cached', '--stdp-granularity', 'group',
                       '--workers', '4', '--freeze-bn', '--skip-test']
            if spec['freeze_backbone']:
                command.append('--freeze-backbone')
            if spec['freeze_hidden']:
                command.append('--freeze-bp-hidden')
            status('full_TRAIN_VALID', stage=spec['name'], log=str(folder / 'campaign.log'))
            with (folder / 'campaign.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            selection = json.loads((folder / 'DVSGesture/lut_native_trace/validation.json').read_text())
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
                            '--dataset', 'DVSGesture', '--condition', 'lut_native_trace', '--test-only'],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((chosen / 'DVSGesture/lut_native_trace/accuracy.json').read_text())
            write(root / 'result.json', result)
            subprocess.run([sys.executable, str(ROOT / 'experiments/stdp/reproduce.py'), '--bundle', str(chosen),
                            '--dataset', 'DVSGesture', '--condition', 'lut_native_trace',
                            '--out', str(root / 'independent_selected_replay_DVSGesture.json')],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        status('completed' if result['test']['float32']['top1_accuracy'] >= 0.95 else 'accuracy_goal_open',
               test_accuracy={k:v['top1_accuracy'] for k,v in result['test'].items()},
               selected_epoch=selection['selected_block'])
    except Exception as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
