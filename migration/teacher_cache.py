"""Regenerate a TRAIN-only teacher cache from a retained original final checkpoint."""
import argparse
import json
from pathlib import Path
from common import ROOT, DATASETS, entry, runtime, sha, configure_torch

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=DATASETS, required=True)
    p.add_argument('--data-dir', type=Path, default=ROOT / 'datasets')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists(): p.error('Use a fresh cache output path')
    e = entry('original', a.dataset)
    modules = runtime(e)
    torch = configure_torch()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required for original teacher cache')
    cfg = json.loads((ROOT / e['config']).read_text())
    cfg.update(data_dir=str(a.data_dir.resolve()), num_workers=4)
    modules['run'].seed_everything(cfg['seed'])
    loaders = modules['data'].build_loaders(a.dataset, cfg)
    assert sha(loaders[3]) == e['split_sha256']
    model = modules['models'].AccuracySNN('custom', a.dataset, cfg, 'exponential').cuda().eval()
    model.load_state_dict(torch.load(ROOT / e['checkpoint'], map_location='cuda', weights_only=True)['model'], strict=True)
    model.requires_grad_(False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    modules['run'].seed_everything(cfg['seed'] + 214)
    loader = torch.utils.data.DataLoader(loaders[0].dataset, batch_size=loaders[0].batch_size,
                                        shuffle=False, num_workers=4)
    rows, labels_all = [], []
    with torch.no_grad():
        for index, (data, labels) in enumerate(loader, 1):
            model.reset_membranes()
            rows.append(model(modules['data'].prepare_batch(data, a.dataset, torch.device('cuda'), cfg=cfg)).cpu())
            labels_all.append(labels)
            if index == 1 or index % 100 == 0: print(json.dumps(dict(TRAIN_batches=index, total=len(loader))), flush=True)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    cache = dict(logits=torch.cat(rows), labels=torch.cat(labels_all), teacher_sha256=e['checkpoint_sha256'],
                 split_sha256=e['split_sha256'], source_partition='TRAIN', encoding_seed=cfg['seed'] + 214)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('xb') as stream: torch.save(cache, stream)
    print(json.dumps(dict(dataset=a.dataset, TRAIN_samples=len(cache['labels']), cache_sha256=sha(a.out),
                          TEST_samples_used=0, new_training_steps=0)))

if __name__ == '__main__': main()
