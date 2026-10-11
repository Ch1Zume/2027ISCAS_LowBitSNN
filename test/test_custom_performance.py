import copy
import json
from types import SimpleNamespace

import pytest
import torch
from spikingjelly.activation_based import functional

from experiments.accuracy.config import dataset_spec, epoch_budget, load_config, PROJECT_ROOT
from experiments.accuracy.data import prepare_batch
from experiments.accuracy.models import AccuracySNN
from experiments.accuracy.nonnegative_backbones import BalancedPositiveConv
from experiments.accuracy.run import calibrate_backbone_normalization, complete_test_evaluation, rng_state
from snn.inference_quant import quantized_hidden_weights


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(2)
    torch.manual_seed(7)


def test_event_rebin_preserves_counts_and_temporal_order():
    data = torch.zeros(2, 16, 2, 128, 128)
    for t in range(16):
        data[:, t, :, 3, 5] = t + 1
    cfg = {"dataset_profiles": {"DVSGesture": {"steps": 8, "resolution": 64}}}
    frames = prepare_batch(data, "DVSGesture", torch.device("cpu"), cfg=cfg)
    assert frames.shape == (8, 2, 2, 64, 64)
    assert frames.sum() == data.sum()
    assert torch.equal(frames[:, 0, 0, 1, 2], torch.arange(3, 32, 4).float())
    cfg["dataset_profiles"]["DVSGesture"]["event_transform"] = "log1p"
    assert torch.equal(prepare_batch(data, "DVSGesture", torch.device("cpu"), cfg=cfg), frames.log1p())


def test_positive_synapse_banks_match_explicit_inhibition_and_both_learn():
    conv = BalancedPositiveConv(2, 3)
    inputs = torch.randn(2, 2, 7, 7, requires_grad=True)
    effective = torch.nn.functional.conv2d(inputs, conv.excitatory.weight - conv.inhibitory.weight, padding=1)
    output = conv(inputs)
    torch.testing.assert_close(output, effective)
    assert conv.excitatory.weight.min() >= 0 and conv.inhibitory.weight.min() >= 0
    output.square().mean().backward()
    assert conv.excitatory.weight.grad.abs().sum() > 0
    torch.testing.assert_close(conv.excitatory.weight.grad, -conv.inhibitory.weight.grad)
    assert inputs.grad.abs().sum() > 0


def test_custom_profiles_keep_nonnegative_spiking_model_and_hidden_quantization():
    cfg = load_config(PROJECT_ROOT / "configs/accuracy_custom_performance_20261004.yaml")
    cfg.update(backbone_variant="balanced_ei", custom_channels=[4, 8], custom_pool_size=2,
               hidden_widths=[8, 6], learning_rules=["stdp", "bp"], hidden_surrogate="atan")
    cfg["dataset_profiles"] = {"DVSGesture": {"steps": 8, "resolution": 64, "epochs": 160}}
    assert dataset_spec("DVSGesture", cfg).steps == 8
    assert epoch_budget(cfg, "custom", "DVSGesture") == 160
    model = AccuracySNN("custom", "DVSGesture", cfg, "exponential").eval()
    frames = prepare_batch(torch.rand(2, 16, 2, 128, 128), "DVSGesture", torch.device("cpu"), cfg=cfg)
    with torch.no_grad():
        logits = model(frames)
    assert logits.shape == (2, 11) and torch.isfinite(logits).all()
    original = copy.deepcopy(model.state_dict())
    with quantized_hidden_weights(model, 2) as records:
        assert [r["layer"] for r in records] == ["hidden1", "hidden2"]
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                assert module.weight.min() >= 0, name
                torch.testing.assert_close(module.weight, original[name + ".weight"])
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key])
    functional.reset_net(model.backbone)


def test_training_only_bn_calibration_preserves_rng_parameters_and_thresholds():
    cfg = load_config(PROJECT_ROOT / "configs/accuracy_custom_performance_20261004.yaml")
    cfg.update(custom_channels=[4, 8], hidden_widths=[8, 6], hidden_fan_in=4, num_workers=0)
    model = AccuracySNN("custom", "MNIST", cfg, "exponential")
    data = torch.utils.data.TensorDataset(torch.rand(8, 1, 28, 28), torch.arange(8))
    generator = torch.Generator().manual_seed(42)
    loader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=True, generator=generator)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    thresholds = [copy.deepcopy(h.thresholds.state_dict()) for h in model.hidden_layers.values()]
    saved_rng, saved_loader = rng_state()[0], generator.get_state()
    result = calibrate_backbone_normalization(model, loader, "MNIST", torch.device("cpu"), batches=3)
    assert result["batches"] == 3 and result["samples"] == 6
    assert torch.equal(saved_rng, rng_state()[0]) and torch.equal(saved_loader, generator.get_state())
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, before[name])
    for h, state in zip(model.hidden_layers.values(), thresholds):
        for name, value in h.thresholds.state_dict().items():
            torch.testing.assert_close(value, state[name])
    assert all(int(m.num_batches_tracked) == 3 for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))
    assert model.training


def test_validation_qualification_never_accesses_test_loader(tmp_path):
    class ForbiddenTestLoader:
        def __iter__(self):
            raise AssertionError("Qualification cannot use test samples")
    model = torch.nn.Linear(2, 2)
    torch.save({"model": model.state_dict(), "epoch": 2, "best_validation_accuracy": 0.5}, tmp_path / "best.pth")
    args = SimpleNamespace(validation_only=True)
    events = []
    monitor = SimpleNamespace(emit=lambda event, **payload: events.append((event, payload)))
    complete_test_evaluation(model, (None, None, ForbiddenTestLoader()), args, {},
                             {"evidence_scope": "validation_qualification"}, tmp_path, monitor,
                             {"reason": "epoch_budget_completed"})
    assert json.loads((tmp_path / "validation.json").read_text())["test_evaluated"] is False
    assert not (tmp_path / "accuracy.json").exists()
    assert events[-1][0] == "validation_qualification_completed"
