"""Cold restore retained payload/environment and replay all twelve full TEST cases."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
META=ROOT/'migration/reorganize_20261013'
OUT=ROOT/'migration/releases/20261013'
DEST=META/'work/cold_restore/Algorithm'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def stage(phase,**kwargs):
    record=dict(phase=phase,**kwargs)
    (META/'restore_validation_progress.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


def main():
    if DEST.exists():raise RuntimeError('Cold restore destination already exists; use existing evidence or a new directory.')
    start=time.monotonic();DEST.mkdir(parents=True)
    manifest=json.loads((ROOT/'migration/payload_manifest.json').read_text())
    stage('copy_core_project',files=len(manifest['sha256']))
    for name in [*manifest['sha256'],'migration/payload_manifest.json']:
        source=ROOT/name
        if source.stat().st_size>=100*1024**2:continue
        target=DEST/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    total=0
    for m in sorted(META.glob('*.files.json')):
        d=json.loads(m.read_text());archive=OUT/d['archive']
        stage('extract_payload',archive=archive.name)
        assert sha(archive)==d['archive_sha256']
        subprocess.run(['tar','-xzf',str(archive),'-C',str(DEST.parent)],check=True)
        for name,expected in d['files'].items():
            p=DEST/name
            assert p.stat().st_size==expected['bytes'] and sha(p)==expected['sha256'],name
            total+=1
    env=dict(os.environ,TMPDIR='/tmp',PYTHONNOUSERSITE='1',MPLCONFIGDIR='/tmp/snn-restore-mpl')
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    stage('restore_Linux_environment',verified_payload_files=total)
    subprocess.run([str(ROOT/'.venv/bin/python'),str(DEST/'migration/restore_environment.py'),
                    '--archives',str(OUT),'--project',str(DEST),'--require-cuda'],check=True,env=env)
    py=DEST/'.venv/bin/python'
    er=json.loads((DEST/'migration/ENVIRONMENT_RESTORE_RECEIPT.json').read_text())
    for key in ['base_prefix','prefix','torch_file','torchvision_file']:
        assert Path(er[key]).resolve().is_relative_to(DEST),key
    stage('checkpoint_loading')
    checkpoints=META/'checkpoint_restore_validation.json'
    subprocess.run([str(py),str(DEST/'migration/verify.py'),'--checkpoints','--out',str(checkpoints)],check=True,env=env)
    stage('full_TEST_replay',conditions=12)
    activity=DEST/'results/new_runs/restored_environment_replay'
    with (META/'restored_full_TEST.log').open('w') as log:
        subprocess.run([str(py),str(DEST/'experiments/power/run_activity.py'),
                        '--data-dir',str(DEST/'datasets'),'--out',str(activity)],
                       cwd=DEST,check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
    conditions=[]
    for dataset in ['NMNIST','MNIST','DVSGesture','CIFAR10DVS']:
        for precision in ['uint4','uint2','float32']:
            result=json.loads((activity/dataset/precision/'summary.json').read_text())
            old=json.loads((ROOT/'results/new_runs/spike_activity_20261012/final'/dataset/precision/'summary.json').read_text())
            assert all(result['validation'].values()),(dataset,precision,result['validation'])
            assert result['score']==old['score'],(dataset,precision)
            for name,layer in old['layers'].items():
                assert result['layers'][name]['counters']==layer['counters'],(dataset,precision,name)
            conditions.append({'dataset':dataset,'precision':precision,'samples':result['samples'],
                'correct':result['score']['correct'],'accuracy':result['score']['top1_accuracy'],
                'exact_full_score_replay':True,'all_layer_counts_match':True,
                'inference_state_unchanged':result['validation']['inference_state_unchanged']})
    record={'cold_restore_passed':True,'verified_payload_files':total,
        'environment':er,'conditions':conditions,'full_TEST_conditions':len(conditions),
        'elapsed_seconds':time.monotonic()-start,'source_environment_modified':False,
        'Windows_delivery_verified':False,'note':'Independent Linux cold restore; Windows receipt is separate.'}
    (META/'restore_validation.json').write_text(json.dumps(record,indent=2)+'\n')
    stage('completed',conditions=len(conditions),verified_payload_files=total)


if __name__=='__main__':main()
