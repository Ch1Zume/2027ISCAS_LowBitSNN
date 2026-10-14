"""Repository-relative benchmark dependencies and frozen runtime isolation."""
import hashlib
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('MNIST', 'NMNIST', 'CIFAR10DVS', 'DVSGesture')
BUNDLES = ('original', 'compressed', 'lut_final')

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()

def entry(bundle, dataset):
    return next(e for e in json.loads((ROOT / 'configs/benchmarks.json').read_text())['entries']
                if e['bundle'] == bundle and e['dataset'] == dataset)

def checkpoint_files(bundle=None, dataset=None):
    files = json.loads((ROOT / 'configs/checkpoints.json').read_text())['files']
    return {name: meta for name, meta in files.items()
            if (bundle is None or meta['bundle'] == bundle)
            and (dataset is None or meta['dataset'] == dataset)}

def verify(bundle=None, dataset=None, checkpoints=True):
    manifest = json.loads((ROOT / 'configs/frozen_sources.json').read_text())['sha256']
    if checkpoints:
        manifest.update({name: meta['sha256'] for name, meta in checkpoint_files(bundle, dataset).items()})
    for name, digest in manifest.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha(path) != digest:
            raise RuntimeError('Benchmark dependency missing or changed: ' + name)
    return len(manifest)

def runtime(e):
    source = (ROOT / e['runtime']).resolve()
    sys.path.insert(0, str(source))
    # Old frozen snapshots have a namespace parent without experiments/__init__.py.
    # Anchor it explicitly so running from a checkout cannot select its regular
    # experiments package before the frozen namespace.
    for name in list(sys.modules):
        if name in ('experiments', 'snn') or name.startswith(('experiments.', 'snn.')):
            del sys.modules[name]
    package = types.ModuleType('experiments')
    package.__path__ = [str(source / 'experiments')]
    package.__package__ = 'experiments'
    package.__spec__ = importlib.machinery.ModuleSpec('experiments', loader=None, is_package=True)
    sys.modules['experiments'] = package
    names = ('experiments.accuracy.models', 'experiments.accuracy.data', 'experiments.accuracy.run', 'snn.wta')
    modules = {}
    for name in names:
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise RuntimeError('Import escaped the frozen runtime: ' + name)
        modules[name.rsplit('.', 1)[-1]] = module
    return modules

def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def configure_torch():
    import torch
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch

def write_new(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
