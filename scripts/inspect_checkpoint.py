"""Check frozen imports, state loading and physical parameter counts without data or GPU."""
import argparse
import json
from pathlib import Path
from common import ROOT, DATASETS, BUNDLES, entry, runtime, configure_torch, write_new

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', choices=BUNDLES, default='lut_final')
    p.add_argument('--dataset', choices=DATASETS, required=True)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    e = entry(a.bundle, a.dataset)
    modules = runtime(e)
    torch = configure_torch()
    cfg = json.loads((ROOT / e['config']).read_text())
    cfg['data_dir'] = str(ROOT / 'datasets')
    modules['run'].seed_everything(cfg['seed'])
    model = modules['models'].AccuracySNN('custom', a.dataset, cfg, 'exponential')
    saved = torch.load(ROOT / e['checkpoint'], map_location='cpu', weights_only=True)
    model.load_state_dict(saved['model'], strict=True)
    count = sum(p.numel() for p in model.parameters())
    assert count == e['parameters']
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            assert torch.isfinite(module.weight).all() and (module.weight >= 0).all()
    for hidden in model.hidden_layers.values():
        assert torch.isfinite(hidden.weight).all() and (hidden.weight >= 0).all()
    result = dict(bundle=a.bundle, dataset=a.dataset, parameters=count, strict_state_load=True,
                  frozen_imports=True, physical_weights_nonnegative=True,
                  checkpoint_keys=list(saved), selected_epoch_or_block=saved.get('block', saved.get('epoch')),
                  full_TEST_performed=False, new_training_steps=0)
    if a.out: write_new(a.out, result)
    print(json.dumps(result))

if __name__ == '__main__': main()
