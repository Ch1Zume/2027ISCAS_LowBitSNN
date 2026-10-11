"""Native resolutions/coding and persisted, disjoint train/validation/test splits."""

import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

from .config import DATASETS, PROJECT_ROOT, dataset_manifest, dataset_spec


def _targets(dataset):
    if isinstance(dataset, Subset):
        return _targets(dataset.dataset)[torch.as_tensor(dataset.indices, dtype=torch.long)]
    if isinstance(dataset, TensorDataset):
        return torch.as_tensor(dataset.tensors[1], dtype=torch.long)
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = [label for _, label in dataset.samples]
    return torch.as_tensor(targets, dtype=torch.long)


PROBE_SAMPLING = "seeded_class_balanced_round_robin_v1"


def build_probe_loader(validation_loader, samples, seed):
    """Fix a class-balanced subset of VALIDATION without loading event frames.

    Round-robin allocation covers every available class when the budget allows.
    Exhausted minority classes yield their unused quota to the remaining classes.
    All randomness uses local generators, independent of training and its loader.
    """
    targets = _targets(validation_loader.dataset)
    count = min(samples, len(targets))
    if count <= 0:
        raise ValueError("Validation probe needs a positive sample budget and nonempty validation data")
    generator = torch.Generator().manual_seed(seed)
    pools = []
    for label in targets.unique(sorted=True):
        positions = torch.where(targets == label)[0]
        pools.append(positions[torch.randperm(len(positions), generator=generator)].tolist())
    indices, depth = [], 0
    while len(indices) < count:
        for pool in pools:
            if depth < len(pool):
                indices.append(pool[depth])
                if len(indices) == count:
                    break
        depth += 1
    order = torch.randperm(count, generator=generator).tolist()
    indices = [indices[index] for index in order]
    probe = DataLoader(Subset(validation_loader.dataset, indices),
                       batch_size=validation_loader.batch_size, shuffle=False,
                       num_workers=0, collate_fn=validation_loader.collate_fn,
                       generator=torch.Generator().manual_seed(seed))
    manifest = {"sampling": PROBE_SAMPLING, "seed": seed, "samples": count,
                "indices_relative_to_validation": indices,
                "label_histogram": torch.bincount(targets[indices]).tolist()}
    return probe, manifest


def stratified_split(targets, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError("Split fraction must lie strictly between zero and one")
    generator = torch.Generator().manual_seed(seed)
    first, second = [], []
    for label in targets.unique(sorted=True):
        indices = torch.where(targets == label)[0]
        if len(indices) < 2:
            raise ValueError("Each class needs at least two samples")
        indices = indices[torch.randperm(len(indices), generator=generator)]
        count = max(1, min(len(indices)-1, int(len(indices)*fraction)))
        first.extend(indices[:count].tolist())
        second.extend(indices[count:].tolist())
    return sorted(first), sorted(second)


def _event_dataset(name, root, steps, train, download):
    from spikingjelly import configure
    from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
    from spikingjelly.datasets.n_mnist import NMNIST
    configure.max_threads_number_for_datasets_preprocess = 4
    cls = {"NMNIST": NMNIST, "CIFAR10DVS": CIFAR10DVS, "DVSGesture": DVS128Gesture}[name]
    root.mkdir(parents=True, exist_ok=True)
    frames = root / f"frames_number_{steps}_split_by_number"
    if frames.exists():
        # A prepared cache should not trigger archive checks or a network fetch.
        import numpy as np
        from torchvision.datasets import DatasetFolder
        def load_frames(path):
            with np.load(path, allow_pickle=False) as array:
                return array["frames"].astype(np.float32)
        partition = frames if name == "CIFAR10DVS" else frames / ("train" if train else "test")
        return DatasetFolder(str(partition), loader=load_frames, extensions=(".npz",))
    if not frames.exists() and not download:
        raise FileNotFoundError(f"Prepared frames missing in {root}; run the prepare command first")
    if not frames.exists() and not cls.downloadable():
        (root / "download").mkdir(exist_ok=True)
        resources = cls.resource_url_md5()
        missing = [file for file, _, _ in resources if not (root / "download" / file).is_file()]
        # Already converted event files need no archive download.
        if missing and not (root / "events_np").is_dir():
            instructions = {"directory": str(root / "download"), "resources": resources, "missing": missing}
            raise FileNotFoundError("Official adapter requires dataset archives: " + json.dumps(instructions))
    kwargs = dict(root=str(root), data_type="frame", frames_number=steps, split_by="number")
    if name != "CIFAR10DVS":
        kwargs["train"] = train
    return cls(**kwargs)


def build_datasets(name, cfg, download=False):
    spec = DATASETS[name]
    root = Path(cfg.get("data_dir", PROJECT_ROOT / "datasets")).expanduser().resolve() / name
    root.mkdir(parents=True, exist_ok=True)
    if name == "MNIST":
        from torchvision.datasets import MNIST
        from torchvision.transforms import ToTensor
        full_train = MNIST(str(root), train=True, download=download, transform=ToTensor())
        test = MNIST(str(root), train=False, download=download, transform=ToTensor())
        parent_indices = list(range(len(full_train)))
        test_indices = list(range(len(test)))
    else:
        full_train = _event_dataset(name, root, spec.steps, True, download)
        if name == "CIFAR10DVS":
            parent_indices, test_indices = stratified_split(
                _targets(full_train), cfg.get("cifar_train_fraction", 0.9), cfg["seed"])
            test = Subset(full_train, test_indices)
        else:
            test = _event_dataset(name, root, spec.steps, False, download)
            parent_indices = list(range(len(full_train)))
            test_indices = list(range(len(test)))
    for dataset in (full_train, test.dataset if isinstance(test, Subset) else test):
        if len(dataset.classes) != spec.classes:
            raise ValueError(f"{name}: expected {spec.classes} classes, found {len(dataset.classes)}; check dataset preparation")
    if name != "CIFAR10DVS" and full_train.class_to_idx != test.class_to_idx:
        raise ValueError("Training/test class mappings differ")
    train_relative, val_relative = stratified_split(
        _targets(full_train)[parent_indices], 1-cfg.get("validation_fraction", 0.1), cfg["seed"]+1)
    train_indices = [parent_indices[i] for i in train_relative]
    val_indices = [parent_indices[i] for i in val_relative]
    manifest = {"dataset": dataset_manifest(spec), "seed": cfg["seed"],
                "class_to_idx": full_train.class_to_idx,
                "train_indices": train_indices, "validation_indices": val_indices,
                "test_indices": test_indices, "test_is_separate_source": name != "CIFAR10DVS"}
    serialized = json.dumps(manifest, sort_keys=True)
    split_path = root / f"split_{hashlib.sha256(serialized.encode()).hexdigest()[:16]}.json"
    split_path.write_text(serialized + "\n")
    return Subset(full_train, train_indices), Subset(full_train, val_indices), test, split_path


def build_loaders(name, cfg, download=False, batch_size=None, limit=None):
    datasets = build_datasets(name, cfg, download)
    generator = torch.Generator().manual_seed(cfg["seed"])
    loaders = []
    for index, dataset in enumerate(datasets[:3]):
        if limit:
            dataset = Subset(dataset, range(min(limit, len(dataset))))
        loaders.append(DataLoader(dataset, batch_size=batch_size or cfg.get("batch_sizes", {}).get(name, DATASETS[name].batch_size),
                                  shuffle=index == 0, drop_last=False,
                                  num_workers=cfg.get("num_workers", 0), generator=generator if index == 0 else None))
    return (*loaders, datasets[3])


def prepare_batch(data, name, device, steps=None, cfg=None):
    data = data.to(device=device, dtype=torch.float32)
    spec = DATASETS[name]
    if name == "MNIST":
        if data.ndim != 4 or tuple(data.shape[1:]) != (spec.channels, spec.resolution, spec.resolution):
            raise ValueError("Expected native MNIST images [batch, 1, 28, 28]")
        time = steps or dataset_spec(name, cfg).steps
        return (torch.rand((time, *data.shape), device=device) < data.unsqueeze(0)).float()
    runtime = dataset_spec(name, cfg)
    if steps is not None and steps != runtime.steps:
        raise ValueError("Event-frame time steps are fixed by the prepared dataset cache")
    if data.ndim != 5 or data.shape[1] != spec.steps or tuple(data.shape[2:]) != (spec.channels, spec.resolution, spec.resolution):
        raise ValueError("Expected event frames [batch, time, polarity, height, width]")
    if runtime.steps != spec.steps:
        # Exact adjacent-bin sums preserve every event and temporal ordering.
        group = spec.steps // runtime.steps
        data = data.reshape(data.shape[0], runtime.steps, group, *data.shape[2:]).sum(2)
    if runtime.resolution != spec.resolution:
        factor = spec.resolution // runtime.resolution
        data = F.avg_pool2d(data.flatten(0, 1), factor) * factor**2
        data = data.reshape(-1, runtime.steps, spec.channels, runtime.resolution, runtime.resolution)
    if (cfg or {}).get("dataset_profiles", {}).get(name, {}).get("event_transform") == "log1p":
        data = data.log1p()
    return data.transpose(0, 1).contiguous()
