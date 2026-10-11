"""VALID-select one/two-view CIFAR inference on immutable trained LUT students."""
import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


class InferenceViews(torch.nn.Module):
    def __init__(self,base,views):
        super().__init__()
        if views not in (1,2):
            raise ValueError('CIFAR views must be one or two')
        self.base,self.views=base,views

    @property
    def hidden_layers(self):
        return self.base.hidden_layers

    @property
    def cfg(self):
        return self.base.cfg

    def reset_membranes(self):
        self.base.reset_membranes()

    def forward(self,frames):
        if self.training:
            raise RuntimeError('Multiple views are registered only for inference')
        self.base.reset_membranes()
        logits=self.base(frames)
        if self.views==2:
            self.base.reset_membranes()
            logits=(logits+self.base(frames.flip(-1)))/2
        self.base.reset_membranes()
        return logits


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--scratch',type=Path,required=True)
    parser.add_argument('--polish',type=Path,required=True)
    parser.add_argument('--replay',action='store_true')
    args=parser.parse_args()
    root=args.out.resolve()
    scratch,polish=args.scratch.resolve(),args.polish.resolve()
    assert json.loads((polish/'status.json').read_text())['phase'] in ('completed','accuracy_goal_open')
    selection=json.loads((polish/'selection_before_TEST.json').read_text())
    condition='CIFAR10DVS/lut_native_trace'
    candidates=[dict(name='scratch_best',bundle=str(scratch/'spatial_scratch'),checkpoint=str(scratch/'spatial_scratch'/condition/'best.pth')),
                dict(name='scratch_last',bundle=str(scratch/'spatial_scratch'),checkpoint=str(scratch/'spatial_scratch'/condition/'last.pth')),
                dict(name='polished',bundle=selection['bundle'],checkpoint=str(Path(selection['bundle'])/condition/'best.pth'))]
    source=Path(candidates[0]['bundle'])/'runtime/CIFAR10DVS'
    source_hashes={str(f.relative_to(source)):sha(f) for f in source.rglob('*.py')}
    for candidate in candidates:
        folder=Path(candidate['bundle'])
        protocol=json.loads((folder/'protocol.json').read_text())
        assert sha(folder/'run_frozen.py')==protocol['driver_sha256']
        for name,digest in protocol['source_sha256'].items():
            assert sha(folder/name)==digest
        other=folder/'runtime/CIFAR10DVS'
        assert {str(f.relative_to(other)):sha(f) for f in other.rglob('*.py')}==source_hashes
        cfg_path=folder/condition/'config.json'
        candidate.update(checkpoint_sha256=sha(candidate['checkpoint']),config=str(cfg_path),config_sha256=sha(cfg_path))
    if not args.replay:
        root.mkdir(exist_ok=False)
        shutil.copy2(Path(__file__),root/'inference_frozen.py')
        write(root/'policy.json',dict(dataset='CIFAR10DVS',candidates=candidates,views=[1,2],
            selection='complete FP32 VALID accuracy then loss; no TEST ranking',
            two_view='native and horizontal flip; average logits; reset membrane between views; frozen homeostasis',
            compute='same parameters; two views approximately double inference work; T=8 per view',
            training='none; all candidates inherit completed LUT training',source=str(source),source_sha256=source_hashes,
            driver_sha256=sha(root/'inference_frozen.py')))
    else:
        policy=json.loads((root/'policy.json').read_text())
        assert sha(Path(__file__))==policy['driver_sha256']
        assert candidates==policy['candidates'] and source_hashes==policy['source_sha256']
    sys.path.insert(0,str(source))
    from experiments.accuracy.models import AccuracySNN
    from experiments.accuracy.data import build_loaders
    from experiments.accuracy.run import evaluate,evaluate_precisions,seed_everything
    for name in ('experiments.accuracy.models','experiments.accuracy.data','experiments.accuracy.run','snn.wta','snn.stdp_lut'):
        assert Path(importlib.import_module(name).__file__).resolve().is_relative_to(source)
    assert torch.cuda.is_available()
    device=torch.device('cuda')
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=False

    def status(phase,**values):
        record=dict(at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),phase=phase,**values)
        if not args.replay:
            write(root/'status.json',record)
        print(json.dumps(record),flush=True)

    def construct(candidate,views):
        cfg=json.loads(Path(candidate['config']).read_text())
        seed_everything(cfg['seed'])
        base=AccuracySNN('custom','CIFAR10DVS',cfg,'exponential').to(device)
        checkpoint=torch.load(candidate['checkpoint'],map_location=device,weights_only=True)
        base.load_state_dict(checkpoint['model'],strict=True)
        loaders=build_loaders('CIFAR10DVS',dict(cfg,num_workers=4))
        assert sha(loaders[3])==json.loads((Path(candidate['bundle'])/condition/'provenance.json').read_text())['split_sha256']
        return InferenceViews(base,views).eval(),loaders,cfg

    if not args.replay:
        values=[]
        for candidate in candidates:
            for views in (1,2):
                model,loaders,cfg=construct(candidate,views)
                before={k:v.clone() for k,v in model.state_dict().items()}
                seed_everything(cfg['seed']+2000)
                score=evaluate(model,loaders[1],'CIFAR10DVS',device)
                assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
                values.append(dict(candidate=candidate,views=views,validation=score))
                write(root/'VALID_results.json',values)
                status('complete_VALID',candidate=candidate['name'],views=views,accuracy=score['top1_accuracy'])
                del model,loaders,before
        chosen=max(values,key=lambda v:(v['validation']['top1_accuracy'],-v['validation']['loss']))
        write(root/'selection_before_TEST.json',chosen)
    else:
        chosen=json.loads((root/'selection_before_TEST.json').read_text())
    candidate=chosen['candidate']
    assert sha(candidate['checkpoint'])==candidate['checkpoint_sha256']
    model,loaders,cfg=construct(candidate,chosen['views'])
    before={k:v.clone() for k,v in model.state_dict().items()}
    seed_everything(1042)
    scores=evaluate_precisions(model,loaders[2],'CIFAR10DVS',device)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert all(v['samples']==1000 for v in scores.values())
    result=dict(dataset='CIFAR10DVS',selection=chosen,parameters=sum(p.numel() for p in model.parameters()),
        test=scores,test_rng_seed=1042,inference_state_unchanged=True,full_TEST=True,training_steps=0,
        views=chosen['views'],steps_per_view=8,inherited_lut_training=True)
    if args.replay:
        expected=json.loads((root/'result.json').read_text())
        assert result==expected,'Independent full TEST differs'
        write(root/'independent_replay.json',dict(exact_full_test_replay=True,result=result,checkpoint_sha256=candidate['checkpoint_sha256']))
        status('independent_full_TEST_exact_replay')
    else:
        write(root/'result.json',result)
        status('full_TEST_complete_pending_independent_replay',test_accuracy={k:v['top1_accuracy'] for k,v in scores.items()},views=chosen['views'])


if __name__=='__main__':
    main()
