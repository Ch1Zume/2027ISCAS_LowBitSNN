"""Causal reset tests against the exact frozen N-MNIST hidden runtime."""
import copy
import json
import types
from pathlib import Path

import pytest
import torch

from experiments.reset_ablation.behavior import reset_source

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks/lut_final/NMNIST/snn/wta.py"


def classes():
    result = []
    source = SOURCE.read_text()
    for code in (source, reset_source(source)):
        module = types.ModuleType("snn._reset_probe")
        module.__package__ = "snn"
        exec(compile(code, str(SOURCE), "exec"), module.__dict__)
        result.append(module.WTAHidden)
    return result


def pair(width=2):
    old, reset = classes()
    options = dict(rule="bp", homeostasis="off", membrane_decay=0.9,
                   surrogate_type="atan", baseline=1, delta=0)
    a, b = old(width, width, **options), reset(width, width, **options)
    with torch.no_grad():
        a.weight.copy_(torch.eye(width))
        a.bias.zero_()
    b.load_state_dict(a.state_dict(), strict=True)
    return a, b


@pytest.mark.parametrize("training", [False, True])
def test_first_winner_clears_nonwinner_and_keeps_current_spikes(training):
    a, b = pair()
    a.train(training); b.train(training)
    x = torch.tensor([[[1.1, 0.9]], [[0.0, 0.3]]])
    assert a(x)[:, 0].tolist() == [[1, 0], [0, 1]]
    assert b(x)[:, 0].tolist() == [[1, 0], [0, 0]]
    assert torch.equal(a.eligible_rows, b.eligible_rows)


def test_reset_is_per_sample_not_entire_batch():
    a, b = pair()
    a.eval(); b.eval()
    x = torch.tensor([[[1.1, 0.9], [0.0, 0.9]], [[0.0, 0.3], [0.0, 0.3]]])
    assert b(x)[1].tolist() == [[0, 0], [0, 1]]


def test_reset_occurs_once_and_keeps_simultaneous_spikes():
    a, b = pair()
    a.eval(); b.eval()
    x = torch.tensor([[[1.1, 1.1]], [[1.8, 0.9]], [[0.3, 0.3]]])
    spikes = b(x)
    assert spikes[:, 0].tolist() == [[1, 1], [1, 0], [1, 1]]


def test_no_winner_has_identical_forward_backward_and_threshold_state():
    a, b = pair()
    x = torch.full((4, 3, 2), 0.01, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    oa, ob = a(x), b(y)
    assert torch.equal(oa, ob)
    oa.sum().backward(); ob.sum().backward()
    assert torch.equal(x.grad, y.grad)
    assert torch.equal(a.weight.grad, b.weight.grad)
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


def test_reset_cuts_gradient_through_cleared_membrane():
    a, b = pair()
    x = torch.tensor([[[1.1, 0.9]], [[0.0, 0.3]]], requires_grad=True)
    b(x)[1].sum().backward()
    assert torch.equal(x.grad[0], torch.zeros_like(x.grad[0]))
    assert x.grad[1].abs().sum() > 0


def test_patch_fails_on_unexpected_source():
    with pytest.raises(ValueError):
        reset_source("not the expected runtime")


@pytest.mark.parametrize("reset", [False, True])
def test_final_checkpoint_toy_training_and_eval_freeze(reset):
    from experiments.accuracy.models import AccuracySNN
    from snn.wta import WinnerOptimizerGuard
    from snn.inference_quant import quantized_hidden_weights
    torch.set_num_threads(2)
    torch.manual_seed(47)
    cfg = json.loads((ROOT / "configs/final/lut_final/NMNIST.json").read_text())
    model = AccuracySNN("custom", "NMNIST", cfg, "exponential")
    checkpoint = ROOT / "checkpoints/lut_final/NMNIST.pth"
    if not checkpoint.exists():
        pytest.skip("Upload the N-MNIST LUT checkpoint for this checkpoint-dependent test")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    # Full checkpoint smoke uses synthetic inputs, never VALID/TEST samples.
    if reset:
        cls = classes()[1]
        for h in model.hidden_layers.values():
            h.__class__ = cls
    assert all(h.competition_size == 512 for h in model.hidden_layers.values())
    frames = torch.rand(4, 2, 2, 34, 34)
    optimizer = torch.optim.Adam(model.bp_parameters(), lr=1e-5)
    model.train()
    logits = model(frames)
    torch.nn.functional.cross_entropy(logits, torch.tensor([1, 2])).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    with WinnerOptimizerGuard(model.hidden_layers.values(), optimizer):
        optimizer.step()
    for h in model.hidden_layers.values():
        h.apply_stdp()
    model.project_nonnegative()
    assert all(p.min() >= 0 for n, p in model.named_parameters() if n.endswith("weight") and p.ndim >= 2)
    model.reset_membranes()
    model.eval()
    before = copy.deepcopy(model.state_dict())
    with torch.no_grad(), quantized_hidden_weights(model, 4):
        assert torch.isfinite(model(frames)).all()
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
