"""Replay the full TEST with the unchanged final model and a new data location."""
import argparse
import json
from pathlib import Path
from common import ROOT, DATASETS, BUNDLES, sha, entry, verify, runtime, load_file, configure_torch, write_new

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', choices=BUNDLES, default='lut_final')
    p.add_argument('--dataset', choices=DATASETS, required=True)
    p.add_argument('--data-dir', type=Path, default=ROOT / 'datasets')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--comparison', choices=('exact', 'counts'), default='exact',
                   help='counts records loss drift explicitly and requires identical correct/sample counts')
    a = p.parse_args()
    if a.out.exists(): p.error('Use a fresh output path')
    files = verify()
    e = entry(a.bundle, a.dataset)
    modules = runtime(e)
    torch = configure_torch()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required for full TEST replay')
    device = torch.device('cuda')
    cfg = json.loads((ROOT / e['config']).read_text())
    cfg.update(data_dir=str(a.data_dir.resolve()), num_workers=a.workers)
    run = modules['run']
    run.seed_everything(cfg['seed'])
    base = modules['models'].AccuracySNN('custom', a.dataset, cfg, 'exponential').to(device)
    checkpoint = torch.load(ROOT / e['checkpoint'], map_location=device, weights_only=True)
    base.load_state_dict(checkpoint['model'], strict=True)
    assert sum(p.numel() for p in base.parameters()) == e['parameters']
    for module in base.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            assert torch.isfinite(module.weight).all() and (module.weight >= 0).all()
    for hidden in base.hidden_layers.values():
        assert torch.isfinite(hidden.weight).all() and (hidden.weight >= 0).all()
    model = base
    if e['views'] == 2:
        model = load_file('_migration_views', ROOT / e['inference_views']).InferenceViews(base, 2)
    model.eval()
    loaders = modules['data'].build_loaders(a.dataset, cfg)
    assert sha(loaders[3]) == e['split_sha256'], 'Data split or class ordering changed'
    before = {k: v.clone() for k, v in model.state_dict().items()}
    run.seed_everything(e['test_rng_seed'])
    if 'quantizer_settings' in e:
        ptq = load_file('_migration_ptq', ROOT / e['ptq'])
        moments = torch.load(ROOT / e['TRAIN_moments'], map_location='cpu', weights_only=True)
        settings = torch.load(ROOT / e['quantizer_settings'], map_location='cpu', weights_only=True)
        scores = {'float32': run.evaluate(model, loaders[2], a.dataset, device)}
        for bits in (2, 4):
            option = settings[str(bits)]
            run.seed_everything(e['test_rng_seed'])
            with ptq.calibrated_weights(model, bits, option['ratios'], option['bias'], moments):
                scores[f'uint{bits}'] = run.evaluate(model, loaders[2], a.dataset, device)
    else:
        scores = run.evaluate_precisions(model, loaders[2], a.dataset, device)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    expected = e['test']
    counts_match = all(scores[k]['correct'] == expected[k]['correct'] and
                       scores[k]['samples'] == expected[k]['samples'] for k in expected)
    exact = scores == expected
    result = dict(bundle=a.bundle, dataset=a.dataset, scores=scores, expected=expected,
                  exact_full_test_replay=exact, counts_match=counts_match,
                  inference_state_unchanged=True, checkpoint_sha256=sha(ROOT / e['checkpoint']),
                  comparison=a.comparison, verified_payload_files=files, new_training_steps=0,
                  torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                  gpu=torch.cuda.get_device_name())
    write_new(a.out, result)  # Persist a mismatch too, so a new GPU's drift is inspectable.
    assert counts_match and (exact or a.comparison == 'counts'), 'Replay differs; see recorded output'
    print(json.dumps(dict(dataset=a.dataset, exact=exact, counts_match=counts_match)))

if __name__ == '__main__': main()
