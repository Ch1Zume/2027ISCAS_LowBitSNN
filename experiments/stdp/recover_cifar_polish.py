"""Polish the trained spatial LUT student using frozen TRAIN features."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--evidence',type=Path,required=True)
    args=parser.parse_args()
    root,previous,evidence=(p.resolve() for p in (args.out,args.previous,args.evidence))
    assert json.loads((previous/'status.json').read_text())['phase']=='accuracy_goal_open'
    source=previous/'spatial_scratch/CIFAR10DVS/lut_native_trace'
    root.mkdir(exist_ok=False)
    driver=root/'recovery_run_frozen.py'
    shutil.copy2(ROOT/'experiments/stdp/run.py',driver)
    write(root/'policy.json',dict(goal='CIFAR10DVS FP32 >=75% complete TEST',
        candidates=['trained_best_head_fit','trained_last_head_fit'],
        source=str(source),initializations=['best.pth VALID-selected epoch 36','last.pth fixed final epoch 40'],
        teacher='SHA-bound original TRAIN-only cache',head='native TRAIN ridge plus 200 FP32 BP steps; nonnegative physical gauge',
        training='6 full epochs max, patience 5, BP scale 0.03, STDP 0.01; BN frozen; original LUT trace RNG restored; fresh optimizer',
        selection='complete VALID accuracy then loss; no TEST ranking'))

    def status(phase,**values):
        record=dict(at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),phase=phase,**values)
        write(root/'status.json',record)
        print(json.dumps(record),flush=True)

    candidates=[]
    try:
        for name,filename in [('trained_best_head_fit','best.pth'),('trained_last_head_fit','last.pth')]:
            folder=root/name
            folder.mkdir()
            for proof in ('dependency_audit.json','baseline_replay_CIFAR10DVS.json'):
                shutil.copy2(evidence/proof,folder/proof)
            command=[sys.executable,str(driver),'--out',str(folder),'--datasets','CIFAR10DVS',
                '--conditions','lut_native_trace','--epochs','6','--patience','5','--stdp-lr','0.01',
                '--bp-rate-scale','0.03','--teacher-source','original_cached','--stdp-granularity','group',
                '--workers','4','--spatial-student','--initial-row-sum','4','--fit-classifier','--freeze-bn',
                '--initial-checkpoint',str(source/filename),'--skip-test']
            status('TRAIN_head_fit_and_full_TRAIN_VALID',stage=name,log=str(folder/'campaign.log'))
            with (folder/'campaign.log').open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            selection=json.loads((folder/'CIFAR10DVS/lut_native_trace/validation.json').read_text())
            score=selection['validation']
            candidates.append((score['top1_accuracy'],-score['loss'],folder,selection))
        _,_,chosen,selection=max(candidates,key=lambda v:v[:2])
        record=dict(bundle=str(chosen),condition='lut_native_trace',selected_epoch=selection['selected_block'],
            checkpoint_sha256=selection['checkpoint_sha256'],validation=selection['validation'],selection='VALID only',
            all_candidates=[str(v[2]) for v in candidates])
        write(root/'selection_before_TEST.json',record)
        status('selected_full_TEST',selection=record)
        with (root/'full_TEST.log').open('w') as log:
            subprocess.run([sys.executable,str(chosen/'run_frozen.py'),'--out',str(chosen),'--dataset','CIFAR10DVS',
                '--condition','lut_native_trace','--test-only'],stdout=log,stderr=subprocess.STDOUT,check=True)
            result=json.loads((chosen/'CIFAR10DVS/lut_native_trace/accuracy.json').read_text())
            write(root/'result.json',result)
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
