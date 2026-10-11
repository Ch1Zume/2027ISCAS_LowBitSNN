"""TRAIN-feature head fitting before further native/spatial LUT recovery."""
import argparse
import json
import os
import shutil
import subprocess
import sys
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
    root, previous, evidence = (p.resolve() for p in (args.out,args.previous,args.evidence))
    state = json.loads((previous/'status.json').read_text())
    assert state['phase'] == 'stopped_ineffective_transfer', state
    root.mkdir(exist_ok=False)
    driver = root/'recovery_run_frozen.py'
    shutil.copy2(ROOT/'experiments/stdp/run.py',driver)
    stages = [dict(name='native_head_fit', spatial=False, scale=0.1),
              dict(name='spatial_head_fit', spatial=True, scale=0.3)]
    write(root/'policy.json',dict(goal='CIFAR10DVS FP32 >=75% full TEST', stages=stages,
        teacher='SHA-bound original TRAIN-only cache', initialization='ridge head fit on full frozen TRAIN features, then 200 float BP steps; per-feature common class shift enforces nonnegative physical weights without changing softmax',
        warm_epochs=12,patience=6,stdp_lr=0.01,freeze_bn=True,
        weak_feature_guard='skip full-network fine-tune when post-fit complete VALID <65%',
        scratch_trigger='best warm complete VALID <77.5%, or selected full TEST misses FP32 goal',
        scratch='random 4.679M spatial student, 40 epochs max, row sum 4, BP scale 1.5, spatial augmentation',
        selection='complete VALID accuracy then loss; no TEST ranking', stopped_previous=str(previous)))

    def status(phase,**values):
        record=dict(at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),phase=phase,**values)
        write(root/'status.json',record)
        print(json.dumps(record),flush=True)

    candidates=[]
    def trial(spec,scratch=False):
        name='spatial_scratch' if scratch else spec['name']
        folder=root/name
        folder.mkdir()
        for filename in ('dependency_audit.json','baseline_replay_CIFAR10DVS.json'):
            shutil.copy2(evidence/filename,folder/filename)
        command=[sys.executable,str(driver),'--out',str(folder),'--datasets','CIFAR10DVS',
            '--conditions','lut_native_trace','--epochs','40' if scratch else '12','--patience','6',
            '--stdp-lr','0.01','--bp-rate-scale','1.5' if scratch else str(spec['scale']),
            '--teacher-source','original_cached','--stdp-granularity','group','--workers','4','--skip-test']
        if scratch or spec['spatial']:
            command.append('--spatial-student')
        if scratch:
            command+=['--from-scratch','--initial-row-sum','4','--augment-spatial']
        else:
            command+=['--fit-classifier','--freeze-bn','--minimum-initial-valid','0.65']
        status('TRAIN_head_fit_and_full_TRAIN_VALID' if not scratch else 'full_TRAIN_VALID',stage=name,log=str(folder/'campaign.log'))
        with (folder/'campaign.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        selection=json.loads((folder/'CIFAR10DVS/lut_native_trace/validation.json').read_text())
        score=selection['validation']
        candidates.append((score['top1_accuracy'],-score['loss'],folder,selection))

    def test_selected():
        _,_,chosen,selection=max(candidates,key=lambda v:v[:2])
        record=dict(bundle=str(chosen),condition='lut_native_trace',selected_epoch=selection['selected_block'],
            checkpoint_sha256=selection['checkpoint_sha256'],validation=selection['validation'],
            selection='VALID only',all_candidates=[str(v[2]) for v in candidates])
        write(root/'selection_before_TEST.json',record)
        status('selected_full_TEST',selection=record)
        with (root/'full_TEST.log').open('a') as log:
            subprocess.run([sys.executable,str(chosen/'run_frozen.py'),'--out',str(chosen),
                '--dataset','CIFAR10DVS','--condition','lut_native_trace','--test-only'],stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((chosen/'CIFAR10DVS/lut_native_trace/accuracy.json').read_text())
        write(root/'result.json',result)
        return result,chosen

    try:
        for spec in stages:
            trial(spec)
        scratch_done=max(v[0] for v in candidates)<0.775
        if scratch_done:
            trial(stages[1],scratch=True)
        result,chosen=test_selected()
        if result['test']['float32']['top1_accuracy']<0.75 and not scratch_done:
            shutil.copy2(root/'result.json',root/'warm_result_before_retraining.json')
            shutil.copy2(root/'selection_before_TEST.json',root/'warm_selection_before_retraining.json')
            trial(stages[1],scratch=True)
            result,chosen=test_selected()
        with (root/'full_TEST.log').open('a') as log:
            subprocess.run([sys.executable,str(ROOT/'experiments/stdp/reproduce.py'),'--bundle',str(chosen),
                '--dataset','CIFAR10DVS','--condition','lut_native_trace','--out',str(root/'independent_selected_replay_CIFAR10DVS.json')],
                stdout=log,stderr=subprocess.STDOUT,check=True)
        status('completed' if result['test']['float32']['top1_accuracy']>=0.75 else 'accuracy_goal_open',
            test_accuracy={k:v['top1_accuracy'] for k,v in result['test'].items()})
    except Exception as error:
        status('failed',error=str(error))
        raise


if __name__=='__main__':
    main()
