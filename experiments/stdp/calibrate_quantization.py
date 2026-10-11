"""TRAIN-moment PTQ clipping/bias correction; VALID selection, frozen FP32."""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def quantized_values(weight,bits,ratio):
    if bits not in (2,4) or weight.ndim!=2 or not torch.isfinite(weight).all() or (weight<0).any():
        raise ValueError('Unsigned hidden matrix requires finite nonnegative weights and 2/4 bits')
    ratio=torch.as_tensor(ratio,dtype=weight.dtype,device=weight.device)
    if not torch.isfinite(ratio).all() or (ratio<=0).any() or (ratio>1).any():
        raise ValueError('Clipping ratios must lie in (0,1]')
    maximum=weight.amax(1,keepdim=True)*ratio
    scale=torch.where(maximum>0,maximum/((1<<bits)-1),torch.ones_like(maximum))
    codes=(weight/scale).round().clamp(0,(1<<bits)-1).to(torch.uint8)
    return codes.to(weight.dtype)*scale,codes,scale


@contextmanager
def calibrated_weights(model,bits,ratios,bias_factor,moments):
    if model.training:
        raise RuntimeError('PTQ is inference-only')
    saved=[]
    try:
        with torch.no_grad():
            for name,layer in model.hidden_layers.items():
                original,bias=layer.weight.clone(),layer.bias.clone()
                saved.append((layer,original,bias))
                quantized,_,_=quantized_values(original,bits,ratios[name])
                mean=moments[name]['mean'].to(original)
                correction=(original-quantized) @ mean
                layer.weight.copy_(quantized)
                layer.bias.copy_(bias+bias_factor*correction)
        yield
    finally:
        with torch.no_grad():
            for layer,weight,bias in saved:
                layer.weight.copy_(weight)
                layer.bias.copy_(bias)


def row_mse_ratios(weight,bits,moments):
    mean=moments['mean'].to(weight)
    variance=(moments['second_moment'].to(weight)-mean.square()).clamp_min(0)
    ratios=(1.,0.75,0.5,0.25,0.125)
    errors=[]
    for ratio in ratios:
        quantized,_,_=quantized_values(weight,bits,ratio)
        errors.append(((weight-quantized).square()*variance.unsqueeze(0)).sum(1))
    indices=torch.stack(errors).argmin(0)
    return torch.tensor(ratios,device=weight.device)[indices].unsqueeze(1).cpu()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--views-bundle',type=Path,required=True)
    parser.add_argument('--replay',action='store_true')
    args=parser.parse_args()
    root,view_bundle=args.out.resolve(),args.views_bundle.resolve()
    replay=json.loads((view_bundle/'independent_replay.json').read_text())
    assert replay['exact_full_test_replay']
    old=json.loads((view_bundle/'result.json').read_text())
    assert old['test']['float32']['top1_accuracy']>=0.75
    chosen=old['selection']
    candidate=chosen['candidate']
    assert sha(candidate['checkpoint'])==candidate['checkpoint_sha256']
    assert sha(candidate['config'])==candidate['config_sha256']
    view_policy=json.loads((view_bundle/'policy.json').read_text())
    assert sha(view_bundle/'inference_frozen.py')==view_policy['driver_sha256']
    source=Path(view_policy['source'])
    for name,digest in view_policy['source_sha256'].items():
        assert sha(source/name)==digest
    if not args.replay:
        root.mkdir(exist_ok=False)
        shutil.copy2(Path(__file__),root/'ptq_frozen.py')
        write(root/'policy.json',dict(dataset='CIFAR10DVS',source_views_bundle=str(view_bundle),
            checkpoint_sha256=candidate['checkpoint_sha256'],config_sha256=candidate['config_sha256'],
            grid_ratios=[1.,0.75,0.5,0.25,0.125],bias_factors=[0.,0.5,1.],
            rowwise='minimize diagonal TRAIN input-variance weighted current error',
            bias='inference only: bias + factor*(W_float-W_quant) @ TRAIN_input_mean',
            calibration='one complete native TRAIN pass with fixed selected inference views; no labels or parameter updates',
            selection='complete VALID per bit, accuracy then loss; no TEST ranking',
            precision='hidden weights only unsigned 2/4 bits; convolution, classifier, bias and LIF parameters float',
            FP32='unchanged and must exactly reproduce prior full TEST',driver_sha256=sha(root/'ptq_frozen.py')))
    else:
        assert sha(Path(__file__))==json.loads((root/'policy.json').read_text())['driver_sha256']
    spec=importlib.util.spec_from_file_location('_frozen_views',view_bundle/'inference_frozen.py')
    view_module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(view_module)
    sys.path.insert(0,str(source))
    from experiments.accuracy.models import AccuracySNN
    from experiments.accuracy.data import build_loaders,prepare_batch
    from experiments.accuracy.run import evaluate,seed_everything
    for name in ('experiments.accuracy.models','experiments.accuracy.data','experiments.accuracy.run','snn.wta','snn.inference_quant'):
        assert Path(importlib.import_module(name).__file__).resolve().is_relative_to(source)
    assert torch.cuda.is_available()
    device=torch.device('cuda')
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=False
    cfg=json.loads(Path(candidate['config']).read_text())
    seed_everything(cfg['seed'])
    base=AccuracySNN('custom','CIFAR10DVS',cfg,'exponential').to(device)
    checkpoint=torch.load(candidate['checkpoint'],map_location=device,weights_only=True)
    base.load_state_dict(checkpoint['model'],strict=True)
    model=view_module.InferenceViews(base,chosen['views']).eval()
    loaders=build_loaders('CIFAR10DVS',dict(cfg,num_workers=4))
    expected_split=json.loads((Path(candidate['bundle'])/'CIFAR10DVS/lut_native_trace/provenance.json').read_text())['split_sha256']
    assert sha(loaders[3])==expected_split
    original={k:v.clone() for k,v in model.state_dict().items()}

    def unchanged():
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.items()),'PTQ changed frozen FP32 state'

    def status(phase,**values):
        record=dict(at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),phase=phase,**values)
        if not args.replay:
            write(root/'status.json',record)
        print(json.dumps(record),flush=True)

    if not args.replay:
        accumulator={name:dict(sum=torch.zeros(layer.in_features,device=device),square=torch.zeros(layer.in_features,device=device),count=0)
                     for name,layer in model.hidden_layers.items()}
        handles=[]
        for name,layer in model.hidden_layers.items():
            def hook(module,inputs,name=name):
                x=inputs[0].detach()
                item=accumulator[name]
                item['sum']+=x.sum((0,1))
                item['square']+=x.square().sum((0,1))
                item['count']+=x.shape[0]*x.shape[1]
            handles.append(layer.register_forward_pre_hook(hook))
        samples=0
        try:
            with torch.no_grad():
                for data,labels in loaders[0]:
                    model.reset_membranes()
                    model(prepare_batch(data,'CIFAR10DVS',device,cfg=cfg))
                    samples+=len(labels)
        finally:
            for handle in handles:
                handle.remove()
        unchanged()
        assert samples==len(loaders[0].dataset)
        moments={name:dict(mean=(v['sum']/v['count']).cpu(),second_moment=(v['square']/v['count']).cpu(),count=v['count'])
                 for name,v in accumulator.items()}
        torch.save(moments,root/'TRAIN_moments.pth')
        write(root/'calibration_evidence.json',dict(partition='TRAIN',samples=samples,views=chosen['views'],
            temporal_samples={k:v['count'] for k,v in moments.items()},weights_and_homeostasis_unchanged=True,
            labels_used=False,training_steps=0))
        selected={}
        all_values={}
        configurations={}
        for bits in (2,4):
            options=[]
            for ratio in (1.,0.75,0.5,0.25,0.125):
                for bias in (0.,0.5,1.):
                    options.append(dict(name=f'ratio_{ratio}_bias_{bias}',ratio=ratio,bias=bias,
                        ratios={name:ratio for name in model.hidden_layers}))
            adaptive={name:row_mse_ratios(layer.weight,bits,moments[name]) for name,layer in model.hidden_layers.items()}
            for bias in (0.5,1.):
                options.append(dict(name=f'row_MSE_bias_{bias}',ratio='row_MSE',bias=bias,ratios=adaptive))
            values=[]
            for option in options:
                seed_everything(cfg['seed']+2000)
                with calibrated_weights(model,bits,option['ratios'],option['bias'],moments):
                    score=evaluate(model,loaders[1],'CIFAR10DVS',device)
                unchanged()
                values.append(dict(name=option['name'],ratio=option['ratio'],bias=option['bias'],validation=score))
                configurations[(bits,option['name'])]=option
                status('complete_VALID',bits=bits,condition=option['name'],accuracy=score['top1_accuracy'])
            best=max(values,key=lambda v:(v['validation']['top1_accuracy'],-v['validation']['loss']))
            if best['ratio']==1. and best['bias']==0.:
                assert best['validation']['samples']==900
            selected[str(bits)]=best
            all_values[str(bits)]=values
        write(root/'VALID_results.json',all_values)
        write(root/'selection_before_TEST.json',selected)
        settings={str(bits):configurations[(bits,selected[str(bits)]['name'])] for bits in (2,4)}
        torch.save(settings,root/'quantizer_settings.pth')
        write(root/'settings_sha256.json',dict(settings=sha(root/'quantizer_settings.pth'),moments=sha(root/'TRAIN_moments.pth')))
    else:
        hashes=json.loads((root/'settings_sha256.json').read_text())
        assert sha(root/'quantizer_settings.pth')==hashes['settings'] and sha(root/'TRAIN_moments.pth')==hashes['moments']
        moments=torch.load(root/'TRAIN_moments.pth',map_location='cpu',weights_only=True)
        settings=torch.load(root/'quantizer_settings.pth',map_location='cpu',weights_only=True)
    scores={}
    seed_everything(1042)
    scores['float32']=evaluate(model,loaders[2],'CIFAR10DVS',device)
    assert scores['float32']==old['test']['float32'],'FP32 changed during PTQ'
    for bits in (2,4):
        option=settings[str(bits)]
        seed_everything(1042)
        with calibrated_weights(model,bits,option['ratios'],option['bias'],moments):
            scores[f'uint{bits}']=evaluate(model,loaders[2],'CIFAR10DVS',device)
        unchanged()
    assert all(v['samples']==1000 for v in scores.values())
    result=dict(dataset='CIFAR10DVS',parameters=sum(p.numel() for p in model.parameters()),test=scores,
        source_views_bundle=str(view_bundle),views=chosen['views'],steps_per_view=8,
        checkpoint_sha256=candidate['checkpoint_sha256'],FP32_exactly_unchanged=True,inference_state_unchanged=True,
        settings_sha256=sha(root/'quantizer_settings.pth'),moments_sha256=sha(root/'TRAIN_moments.pth'))
    if args.replay:
        assert result==json.loads((root/'result.json').read_text())
        write(root/'independent_replay.json',dict(exact_full_test_replay=True,result=result))
        status('independent_full_TEST_exact_replay')
    else:
        write(root/'result.json',result)
        status('full_TEST_complete_pending_independent_replay',test_accuracy={k:v['top1_accuracy'] for k,v in scores.items()})


if __name__=='__main__':
    main()
