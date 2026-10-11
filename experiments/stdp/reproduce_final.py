"""Replay one dataset from the final frozen four-benchmark LUT bundle."""
import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--dataset', required=True, choices=('MNIST', 'NMNIST', 'CIFAR10DVS', 'DVSGesture'))
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Independent replay output must not exist')
    root = args.bundle.resolve()
    manifest = json.loads((root / 'benchmark_manifest.json').read_text())
    for filename, digest in manifest['artifact_sha256'].items():
        assert sha(root / filename) == digest, filename
    entry = next(e for e in manifest['entries'] if e['dataset'] == args.dataset)
    source = root / entry['runtime']
    sys.path.insert(0, str(source))
    import torch
    from experiments.accuracy.models import AccuracySNN
    from experiments.accuracy.data import build_loaders
    from experiments.accuracy.run import evaluate, evaluate_precisions, seed_everything
    for name in ('experiments.accuracy.models', 'experiments.accuracy.data', 'experiments.accuracy.run', 'snn.wta', 'snn.stdp_lut'):
        assert Path(importlib.import_module(name).__file__).resolve().is_relative_to(source)
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available(), 'The measured CUDA inference environment is required'
    device = torch.device('cuda')
    cfg = json.loads((root / entry['config']).read_text())
    seed_everything(cfg['seed'])
    base = AccuracySNN('custom', args.dataset, cfg, 'exponential').to(device)
    checkpoint = torch.load(root / entry['checkpoint'], map_location=device, weights_only=True)
    base.load_state_dict(checkpoint['model'], strict=True)
    assert sum(p.numel() for p in base.parameters()) == entry['parameters']
    for module in base.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            assert torch.isfinite(module.weight).all() and (module.weight >= 0).all()
    for layer in base.hidden_layers.values():
        assert torch.isfinite(layer.weight).all() and (layer.weight >= 0).all()
    model = base
    if entry['views'] == 2:
        views = load_module('_final_views', root / 'inference_views_frozen.py')
        model = views.InferenceViews(base, 2)
    model.eval()
    loaders = build_loaders(args.dataset, dict(cfg, num_workers=4))
    assert sha(loaders[3]) == entry['split_sha256']
    before = {k: v.clone() for k, v in model.state_dict().items()}
    if 'quantizer_settings' in entry:
        ptq = load_module('_final_ptq', root / 'ptq_frozen.py')
        moments = torch.load(root / entry['TRAIN_moments'], map_location='cpu', weights_only=True)
        settings = torch.load(root / entry['quantizer_settings'], map_location='cpu', weights_only=True)
        seed_everything(entry['test_rng_seed'])
        scores = {'float32': evaluate(model, loaders[2], args.dataset, device)}
        for bits in (2, 4):
            option = settings[str(bits)]
            seed_everything(entry['test_rng_seed'])
            with ptq.calibrated_weights(model, bits, option['ratios'], option['bias'], moments):
                scores[f'uint{bits}'] = evaluate(model, loaders[2], args.dataset, device)
    else:
        seed_everything(entry['test_rng_seed'])
        scores = evaluate_precisions(model, loaders[2], args.dataset, device)
    assert scores == entry['test'], 'Independent full TEST replay differs'
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()), 'Inference changed frozen state'
    assert all(v['samples'] == entry['test_samples'] for v in scores.values())
    result = dict(dataset=args.dataset, parameters=entry['parameters'], views=entry['views'],
                  steps_per_view=entry['steps_per_view'], exact_full_test_replay=True,
                  runtime_imports_verified=True, nonnegative_physical_weights=True,
                  inference_state_unchanged=True, new_training_steps=0, scores=scores,
                  checkpoint_sha256=sha(root / entry['checkpoint']))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(dataset=args.dataset, exact_full_test_replay=True,
                          correct={k: v['correct'] for k, v in scores.items()})), flush=True)


if __name__ == '__main__':
    main()
