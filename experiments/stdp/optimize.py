"""Sequential, VALID-selected native-forward LUT optimization on one GPU."""
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ('MNIST', 'NMNIST', 'CIFAR10DVS', 'DVSGesture')


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.out.resolve()
    assert (root / 'selections_before_TEST.json').exists(), 'Finish the paired native-forward pilot first'
    assert not (root / 'optimization_status.json').exists(), 'A campaign is already registered here'
    driver = root / 'optimization_run_frozen.py'
    shutil.copy2(ROOT / 'experiments/stdp/run.py', driver)
    stages = []
    evidence = ('dependency_audit.json', *(f'baseline_replay_{d}.json' for d in DATASETS))

    def status(phase, **values):
        record = {'at': datetime.now(timezone.utc).isoformat(), 'phase': phase, **values}
        write(root / 'optimization_status.json', record)
        print(json.dumps(record), flush=True)

    def trial(name, datasets, lr, *, epochs=None, scale=0.1):
        folder = root / name
        folder.mkdir(exist_ok=False)
        for filename in evidence:
            shutil.copy2(root / filename, folder / filename)
        command = [sys.executable, str(driver), '--out', str(folder), '--datasets', *datasets,
                   '--conditions', 'lut_native_trace', '--stdp-lr', str(lr), '--bp-rate-scale', str(scale),
                   '--blocks', '2', '--batches-per-block', '32', '--skip-test', '--freeze-bn']
        if epochs:
            command += ['--epochs', str(epochs), '--patience', '5']
        status('full_TRAIN_VALID' if epochs else 'trace_lr_VALID_screen', stage=name,
               datasets=datasets, stdp_lr=lr, epochs=epochs, log=str(folder / 'campaign.log'))
        with (folder / 'campaign.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        stages.append(folder)
        return folder

    try:
        # Register all options before accessing TEST; freeze BN in every LR screen.
        for name, lr in (('rate_0p001', 0.001), ('rate_0p01', 0.01), ('rate_0p1', 0.1)):
            trial(name, ['CIFAR10DVS', 'DVSGesture'], lr)
        def best_trial(dataset, candidates):
            values = []
            for folder in candidates:
                path = folder / dataset / 'lut_native_trace/validation.json'
                if path.exists():
                    selection = json.loads(path.read_text())
                    cfg = json.loads((path.parent / 'config.json').read_text())
                    score = selection['validation']
                    values.append((score['top1_accuracy'], -score['loss'], folder, selection, cfg))
            assert values
            # Stable tie-break follows registration order, never TEST.
            return max(values, key=lambda v: v[:2])
        rates = {d: best_trial(d, stages)[4]['stdp_lr'] for d in ('CIFAR10DVS', 'DVSGesture')}
        write(root / 'learning_rates_before_full_training.json', rates)
        trial('CIFAR_full_train', ['CIFAR10DVS'], rates['CIFAR10DVS'], epochs=20, scale=0.5)
        trial('Gesture_full_train', ['DVSGesture'], rates['DVSGesture'], epochs=12, scale=0.25)
        selected = {}
        for dataset in DATASETS:
            _, _, folder, selection, cfg = best_trial(dataset, [root, *stages])
            selected[dataset] = {'bundle': str(folder), 'condition': 'lut_native_trace',
                'validation': selection['validation'], 'selected_block': selection['selected_block'],
                'checkpoint_sha256': selection['checkpoint_sha256'], 'stdp_lr': cfg['stdp_lr']}
        write(root / 'final_selections_before_TEST.json', selected)
        results = {}
        for dataset, value in selected.items():
            folder = Path(value['bundle'])
            status('selected_full_TEST', dataset=dataset, selected=value)
            command = [sys.executable, str(folder / 'run_frozen.py'), '--out', str(folder),
                       '--dataset', dataset, '--condition', 'lut_native_trace', '--test-only']
            with (root / 'selected_TEST.log').open('a') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((folder / dataset / 'lut_native_trace/accuracy.json').read_text())
            results[dataset] = result
            write(root / 'selected_results.json', results)
            replay = [sys.executable, str(ROOT / 'experiments/stdp/reproduce.py'), '--bundle', str(folder),
                      '--dataset', dataset, '--condition', 'lut_native_trace',
                      '--out', str(root / f'independent_selected_replay_{dataset}.json')]
            with (root / 'selected_TEST.log').open('a') as log:
                subprocess.run(replay, stdout=log, stderr=subprocess.STDOUT, check=True)
        goals = dict(zip(DATASETS, (0.99, 0.98, 0.75, 0.95)))
        passed = {d: results[d]['test']['float32']['top1_accuracy'] >= goals[d] for d in DATASETS}
        status('completed' if all(passed.values()) else 'accuracy_goals_open',
               fp32_goals_passed=passed, full_TEST_datasets=4, independent_replays=4)
    except Exception as error:
        status('failed', error=str(error), completed_stages=[str(s) for s in stages])
        raise


if __name__ == '__main__':
    main()
