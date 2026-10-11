"""Check source preservation, initialization boundaries and native EI shapes."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_model
from torch.utils.data import DataLoader, TensorDataset

from experiments.accuracy.models import AccuracySNN
from experiments.accuracy.nonnegative_backbones import AuthorBackbone
from experiments.accuracy.run import calibrate_hidden, rng_state
from experiments.accuracy.vendor.deepeisnn.models.factory import build_model
from experiments.accuracy.vendor.deepeisnn.modules.linear import SpikingEiLinear
from test_training_monitoring import compact_config


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(7)
    torch.set_num_threads(2)


@pytest.mark.parametrize("family,name", [("vgg11_bn", "CIFAR10-VGG11"), ("resnet18", "CIFAR10-ResNet18")])
def test_author_checkpoint_features_are_preserved_before_target_adaptation(family, name):
    directory = Path("checkpoints/deepeisnn")/name
    if not (directory/"model.safetensors").exists():
        pytest.skip("Requires the separately downloaded, author-verified checkpoint")
    config = json.loads((directory/"config.json").read_text())
    original = build_model(config, torch.device("cpu"), np.random.default_rng(42)).eval()
    load_model(original, directory/"model.safetensors", strict=True)
    for module in original.modules():
        if hasattr(module, "_inited"):
            module._inited = True
    observed = []
    head = next(module for module in original.modules() if isinstance(module, SpikingEiLinear))
    handle = head.register_forward_pre_hook(lambda module, inputs: observed.append(inputs[0].detach().clone()))
    data = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        original(data)
    handle.remove()
    adapted = AuthorBackbone(family, {"directory": str(directory)}, channels=3, steps=4, resolution=32).eval()
    with torch.no_grad():
        features = adapted(data.unsqueeze(0).repeat(4, 1, 1, 1, 1))
    assert torch.equal(features.flatten(0, 1), observed[0])
    assert all((module.weight >= 0).all() for module in adapted.modules() if isinstance(module, torch.nn.Conv2d))


def test_hidden_calibration_uses_train_batch_and_preserves_rng_loader_thresholds():
    cfg = compact_config()
    cfg.update(backbone_variant="positive_ei", hidden_surrogate="atan", hidden_fan_in=4)
    model = AccuracySNN("custom", "MNIST", cfg, "rate")
    data = TensorDataset(torch.rand(4, 1, 28, 28), torch.arange(4))
    loader = DataLoader(data, batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(42))
    saved_rng, saved_loader = rng_state()[0], loader.generator.get_state().clone()
    backbone = copy.deepcopy(model.backbone.state_dict())
    thresholds = [copy.deepcopy(hidden.thresholds.state_dict()) for hidden in model.hidden_layers.values()]
    result = calibrate_hidden(model, loader, "MNIST", torch.device("cpu"))
    assert torch.equal(saved_rng, rng_state()[0]) and torch.equal(saved_loader, loader.generator.get_state())
    assert model.training and set(result) == {"hidden1", "hidden2"}
    for key, value in backbone.items():
        assert torch.equal(value, model.backbone.state_dict()[key])
    for hidden, previous in zip(model.hidden_layers.values(), thresholds):
        assert all(torch.equal(value, hidden.thresholds.state_dict()[key]) for key, value in previous.items())
        assert (hidden.weight >= 0).all() and (hidden.weight <= hidden.weight_max).all()

