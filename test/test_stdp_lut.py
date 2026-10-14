"""Causal STDP, frozen-runtime parity, float learning and inference boundaries."""
import copy
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch

from snn.stdp_lut import countdown_step, signed_lut
from snn.wta import WTAHidden, WinnerOptimizerGuard
from snn.inference_quant import quantize_nonnegative, quantized_hidden_weights


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "benchmarks/lut_final"
EXPECTED = [-12, -11, -11, -11, -11, -11, -10, -10, -10, -10, -10, -9, -9, -9,
            -8, -8, -7, -7, -6, -6, -5, -4, -3, -3, -1, 0, 1, 2, 4, 6, 8, 10]


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(7)
    torch.set_num_threads(2)


def lut_layer(inputs=3, outputs=2, **kwargs):
    layer = WTAHidden(inputs, outputs, rule="stdp", stdp_mode="lut",
                      stdp_input_encoding="binary", stdp_lr=0.001,
                      stdp_record_stats=True, baseline=1, delta=0, **kwargs)
    with torch.no_grad():
        layer.weight.fill_(0.4)
        layer.reset_row_sum_target()
    return layer


def context(layer, pre, post):
    layer.learning_context = (pre.float(), post.float())
    layer.eligible_rows = post.bool().any(0).any(0)


def test_table_all_entries_formula_and_signed_limits():
    table = signed_lut()
    assert table.dtype == torch.int8 and table.tolist() == EXPECTED
    assert table.min() >= -16 and table.max() <= 15
    assert signed_lut(trace_decay=0)[31] == 10
    assert signed_lut(trace_decay=0)[0] == -12
    for kwargs in [{"coefficient_lsb": 0}, {"a_plus": 1}, {"trace_decay": 1}, {"a_minus": float("nan")}]:
        with pytest.raises(ValueError):
            signed_lut(**kwargs)


def test_countdown_refresh_saturating_decay_reset_and_age31():
    u = torch.zeros(1, 2, dtype=torch.uint8)
    u = countdown_step(u, torch.tensor([[True, False]]))
    assert u.tolist() == [[31, 0]]
    for age in range(1, 34):
        u = countdown_step(u, torch.zeros_like(u, dtype=torch.bool))
        assert u.tolist() == [[max(31-age, 0), 0]]
    assert countdown_step(u, torch.tensor([[True, True]])).tolist() == [[31, 31]]
    with pytest.raises(ValueError):
        countdown_step(torch.tensor([[32]], dtype=torch.uint8), torch.ones(1, 1, dtype=torch.bool))


def test_trace_only_keeps_native_forward_gradients_and_global_rng():
    opts = dict(rule="hybrid", competition_size=4, hybrid_stdp_period=2,
                baseline=0.2, delta=0.05, stdp_lr=0.01)
    native = WTAHidden(3, 4, **opts)
    lookup = WTAHidden(3, 4, stdp_mode="lut", stdp_input_encoding="bernoulli_trace",
                       stdp_trace_seed=123, **opts)
    lookup.load_state_dict(native.state_dict(), strict=True)
    x = torch.rand(4, 8, 3, requires_grad=True)
    z = x.detach().clone().requires_grad_(True)
    old, new = native(x), lookup(z)
    assert torch.equal(old, new)
    assert torch.equal(native.learning_context[1], lookup.learning_context[1])
    assert all(torch.equal(v, lookup.state_dict()[k]) for k, v in native.state_dict().items())
    old.sum().backward()
    new.sum().backward()
    assert torch.equal(x.grad, z.grad)
    assert torch.equal(native.weight.grad, lookup.weight.grad)
    before_rng = torch.random.get_rng_state().clone()
    before = lookup.weight.detach().clone()
    lookup.apply_stdp()
    assert torch.equal(torch.random.get_rng_state(), before_rng)
    protected = ~(lookup.eligible_rows & lookup.stdp_neuron_mask)
    assert torch.equal(lookup.weight[protected], before[protected])
    assert not torch.equal(lookup.weight, before)
    assert lookup.trace_rng_state() is not None


def test_trace_only_probabilities_resume_and_eval_freeze():
    h = WTAHidden(2, 2, rule="stdp", stdp_mode="lut",
                  stdp_input_encoding="bernoulli_trace", stdp_trace_seed=91)
    x = torch.full((20000, 2), 0.25)
    events = h._learning_events(x)
    assert abs(events.float().mean().item()-0.25) < 0.01
    state = h.trace_rng_state()
    expected = h._learning_events(x)
    restored = WTAHidden(2, 2, rule="stdp", stdp_mode="lut",
                         stdp_input_encoding="bernoulli_trace", stdp_trace_seed=91)
    restored.load_state_dict(h.state_dict(), strict=True)
    restored.restore_trace_rng(state)
    assert torch.equal(restored._learning_events(x), expected)
    before_rng = restored.trace_rng_state()
    restored.eval()
    restored(torch.full((3, 4, 2), 0.25))
    restored.apply_stdp()
    assert torch.equal(restored.trace_rng_state(), before_rng)
    with pytest.raises(ValueError):
        restored._learning_events(torch.tensor([[1.1, 0.0]]))
    assert restored._learning_events(torch.tensor([[0.0, 1.0]])).tolist() == [[False, True]]






def test_train_head_fit_keeps_nonnegative_physical_weights_and_detached_features():
    from experiments.stdp.head_fit import fit_nonnegative_classifier
    generator = torch.Generator().manual_seed(123)
    features = torch.rand(160,6,generator=generator,requires_grad=True)
    signed = torch.randn(3,6,generator=generator)
    teacher = (features.detach()-0.5) @ signed.T*12
    labels = teacher.argmax(1)
    assert labels.unique().numel() == 3
    head = torch.nn.Linear(6,3)
    report = fit_nonnegative_classifier(head,features,labels,teacher)
    assert features.grad is None
    assert head.weight.dtype == torch.float32 and head.weight.min() >= 0
    assert report['partition'] == 'TRAIN' and report['steps'] == 200
    assert float((head(features.detach()).argmax(1)==labels).float().mean()) > 0.9
    virtual = torch.nn.functional.linear(features.detach(),head.weight-head.weight.min(0).values,head.bias)
    physical = head(features.detach())
    assert torch.allclose(physical.log_softmax(1),virtual.log_softmax(1),rtol=1e-5,atol=1e-5)


def test_two_views_clear_temporal_state_between_views_and_batches():
    from experiments.stdp.cifar_views import InferenceViews
    class TemporalToy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_layers=torch.nn.ModuleDict()
            self.cfg={}
            self.voltage=0
        def reset_membranes(self):
            self.voltage=0
        def forward(self,frames):
            assert self.voltage==0
            self.voltage=1
            return frames[...,0].sum((0,2,3)).unsqueeze(1)
    model=InferenceViews(TemporalToy(),2).eval()
    frames=torch.arange(24,dtype=torch.float32).reshape(2,2,1,2,3)
    expected=(frames[...,0].sum((0,2,3))+frames[...,-1].sum((0,2,3))).unsqueeze(1)/2
    assert torch.equal(model(frames),expected)
    assert torch.equal(model(frames),expected)
    assert model.base.voltage==0
    model.train()
    with pytest.raises(RuntimeError,match='inference'):
        model(frames)


def test_ptq_mean_current_correction_and_state_restore_without_training():
    from experiments.stdp.calibrate_quantization import calibrated_weights,quantized_values,row_mse_ratios
    model=torch.nn.Module()
    model.hidden_layers=torch.nn.ModuleDict({'h':torch.nn.Linear(4,2)})
    model.classifier=torch.nn.Linear(2,3)
    model.conv=torch.nn.Conv2d(1,1,1)
    model.eval()
    h=model.hidden_layers['h']
    h.weight.data.copy_(torch.tensor([[1.,0.04,0.03,0.01],[0.,0.,0.,0.]]))
    moments={'h':dict(mean=torch.tensor([1.,0.5,0.5,0.5]),second_moment=torch.tensor([1.,0.5,0.5,0.5]))}
    before=copy.deepcopy(model.state_dict())
    original_current=h.weight @ moments['h']['mean']+h.bias
    ratios=row_mse_ratios(h.weight,2,moments['h'])
    assert float(ratios[0])<1
    with calibrated_weights(model,2,{'h':ratios},1.,moments):
        assert torch.allclose(h.weight @ moments['h']['mean']+h.bias,original_current)
        assert h.weight.min()>=0
        assert torch.equal(model.classifier.weight,before['classifier.weight'])
        assert torch.equal(model.conv.weight,before['conv.weight'])
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    with pytest.raises(RuntimeError,match='deliberate'):
        with calibrated_weights(model,4,{'h':0.5},1.,moments):
            raise RuntimeError('deliberate interruption')
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    q,codes,scale=quantized_values(before['hidden_layers.h.weight'],2,1.)
    expected,old_codes,old_scale=quantize_nonnegative(before['hidden_layers.h.weight'],2)
    assert torch.equal(q,expected) and torch.equal(codes,old_codes) and torch.equal(scale,old_scale)
    assert codes.dtype==torch.uint8 and codes.max()<=3
    model.train()
    with pytest.raises(RuntimeError,match='inference-only'):
        with calibrated_weights(model,2,{'h':1.},0.,moments):
            pass


def test_coincident_refresh_ltd_and_batch_mean_with_soft_bounds():
    h = lut_layer(stdp_update_rule="add_clip")
    pre = torch.tensor([[[1, 0, 0], [0, 0, 0]]])
    post = torch.tensor([[[1, 0], [0, 0]]])
    context(h, pre, post)
    before = h.weight.detach().clone()
    h.apply_stdp()
    expected = before.clone()
    expected[0] += torch.tensor([10, -12, -12]) * 1e-6 / 2
    torch.testing.assert_close(h.weight, expected, atol=0, rtol=0)
    assert h.stdp_statistics["winner_events"] == 1
    assert h.stdp_statistics["winner_trace_histogram"][31] == 1
    assert h.stdp_statistics["winner_trace_histogram"][0] == 2
    assert h.learning_context is None
    h.apply_stdp()
    torch.testing.assert_close(h.weight, expected, atol=0, rtol=0)
    h2 = lut_layer()
    context(h2, pre, post)
    before = h2.weight.detach().clone()
    delta = torch.tensor([10*0.6, -12*0.4, -12*0.4]) * 1e-6 / 2
    updated = before[0] + delta
    expected = before.clone()
    expected[0] = updated * (h2.row_sum_target[0] / updated.sum())
    h2.apply_stdp()
    torch.testing.assert_close(h2.weight, expected, atol=1e-8, rtol=0)


def test_future_events_cannot_change_frozen_winner_trace_and_presentation_resets():
    first, second = lut_layer(), lut_layer()
    second.load_state_dict(first.state_dict(), strict=True)
    pre = torch.tensor([[[1, 0, 0]], [[0, 1, 0]], [[0, 0, 1]]])
    post = torch.tensor([[[1, 0]], [[0, 0]], [[0, 0]]])
    context(first, pre, post)
    context(second, torch.cat([pre[:1], torch.ones_like(pre[1:])]), post)
    first.apply_stdp()
    second.apply_stdp()
    assert torch.equal(first.weight, second.weight)
    first.weight.data.fill_(0.4)
    first.reset_row_sum_target()
    context(first, torch.zeros(1, 1, 3), torch.tensor([[[1, 0]]]))
    first.apply_stdp()
    assert first.stdp_statistics["winner_trace_histogram"][0] == 3


def test_group_first_winners_keep_all_forward_activity_and_threshold_freeze():
    h = WTAHidden(2, 4, rule="hybrid", competition_size=2, hybrid_stdp_period=2,
                  hybrid_stdp_granularity="group", winner_tiebreak="margin",
                  stdp_mode="lut", stdp_input_encoding="binary", stdp_record_stats=True,
                  baseline=0.1, delta=0.05, minimum=0.1, stdp_lr=0.001)
    h.weight.data.fill_(0.4)
    h.reset_row_sum_target()
    out = h(torch.ones(3, 2, 2))
    assert out.sum() == 24  # Losing neurons keep firing after the first winner.
    pre, post = h.learning_context
    assert torch.equal(post.sum((0, 1)), torch.tensor([2., 0., 2., 0.]))
    before = h.weight.detach().clone()
    h.apply_stdp()
    assert h.stdp_statistics["winner_events"] == 2
    assert torch.equal(h.weight[~h.stdp_neuron_mask], before[~h.stdp_neuron_mask])
    assert torch.equal(h.weight[1], before[1])
    h.eval()
    state = copy.deepcopy(h.state_dict())
    h(torch.zeros(2, 2, 2))
    h.apply_stdp()
    for k, v in state.items():
        assert torch.equal(v, h.state_dict()[k])
    assert h.learning_context is None


def test_no_winner_means_no_write_and_no_normalization():
    h = lut_layer()
    before = copy.deepcopy(h.state_dict())
    h(torch.zeros(4, 2, 3))
    h.apply_stdp()
    assert h.stdp_statistics["nonzero_weight_writes"] == 0
    assert h.stdp_statistics["winner_events"] == 0
    assert torch.equal(h.weight, before["weight"])
    assert torch.equal(h.row_sum_target, before["row_sum_target"])


def test_bp_and_stdp_rows_are_disjoint_with_optimizer_momentum():
    h = WTAHidden(3, 4, rule="hybrid", competition_size=2, hybrid_stdp_period=2,
                  stdp_mode="lut", stdp_input_encoding="binary", baseline=0.1,
                  delta=0, stdp_lr=0.001, stdp_record_stats=True, surrogate_type="atan",
                  winner_tiebreak="margin")
    h.weight.data.fill_(0.4)
    h.weight.data[3].fill_(0.5)
    h.reset_row_sum_target()
    opt = torch.optim.Adam([h.weight, h.bias], lr=0.01, weight_decay=0.1)
    # Preexisting momentum must also be protected in STDP/loser rows.
    for param in [h.weight, h.bias]:
        opt.state[param] = {"step": torch.tensor(3.), "exp_avg": torch.ones_like(param),
                            "exp_avg_sq": torch.ones_like(param)}
    x = torch.tensor([[[1., 0., 1.]]], requires_grad=True)
    out = h(x)
    out.sum().backward()
    assert x.grad.abs().sum() > 0
    before = h.weight.detach().clone()
    saved_momentum = opt.state[h.weight]["exp_avg"].clone()
    with WinnerOptimizerGuard([h], opt):
        opt.step()
    protected = ~(h.eligible_rows & h.bp_neuron_mask)
    assert torch.equal(h.weight[protected], before[protected])
    assert torch.equal(opt.state[h.weight]["exp_avg"][protected], saved_momentum[protected])
    before = h.weight.detach().clone()
    h.apply_stdp()
    assert h.stdp_statistics["nonzero_weight_writes"] > 0
    assert torch.equal(h.weight[h.bp_neuron_mask], before[h.bp_neuron_mask])


def test_encoding_is_explicit_mean_preserving_paired_and_passes_bp():
    with pytest.raises(ValueError, match="explicit"):
        WTAHidden(2, 2, stdp_mode="lut")
    with pytest.raises(ValueError, match="first-winner"):
        WTAHidden(2, 2, stdp_mode="lut", stdp_input_encoding="binary", winner_scope="cycle")
    h = lut_layer()
    with pytest.raises(ValueError, match="Binary"):
        h(torch.full((1, 1, 3), 0.2))
    with pytest.raises(ValueError, match="amplitude"):
        b = WTAHidden(1, 1, rule="stdp", stdp_mode="lut", stdp_input_encoding="bernoulli")
        b(torch.full((1, 1, 1), 1.1))
    native = WTAHidden(1, 1, rule="stdp", stdp_input_encoding="bernoulli", baseline=0.1, delta=0)
    lookup = WTAHidden(1, 1, rule="stdp", stdp_mode="lut", stdp_input_encoding="bernoulli", baseline=0.1, delta=0)
    lookup.load_state_dict(native.state_dict(), strict=True)
    x = torch.full((1, 20000, 1), 0.25, requires_grad=True)
    rng = torch.get_rng_state()
    native(x)
    encoded = native.learning_context[0]
    assert abs(float(encoded.mean()) - 0.25) < 0.01
    torch.set_rng_state(rng)
    lookup(x).sum().backward()
    assert torch.equal(lookup.learning_context[0], encoded)
    assert x.grad.abs().sum() > 0


@pytest.mark.parametrize("runtime,rule,extras", [
    ("MNIST", "stdp", {}),
    ("MNIST", "bp", {}),
    ("CIFAR10DVS", "hybrid", {"competition_size": 4, "hybrid_stdp_period": 2, "winner_tiebreak": "margin"}),
    ("DVSGesture", "hybrid", {"competition_size": 2, "hybrid_stdp_period": 2, "hybrid_stdp_granularity": "group", "winner_tiebreak": "margin"}),
])
def test_final_frozen_source_update_forward_gradient_and_checkpoint_parity(runtime, rule, extras):
    spec = importlib.util.spec_from_file_location("snn._frozen_wta", BUNDLE / runtime / "snn/wta.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    options = dict(rule=rule, baseline=0.4, delta=0.05, initial_row_sum=2.0,
                   stdp_lr=0.001, surrogate_type="atan", **extras)
    old = module.WTAHidden(5, 8, **options)
    new = WTAHidden(5, 8, **options)
    new.load_state_dict(old.state_dict(), strict=True)
    assert old.state_dict().keys() == new.state_dict().keys()
    original, revised = torch.rand(6, 3, 5, requires_grad=True), None
    revised = original.detach().clone().requires_grad_(True)
    out0, out1 = old(original), new(revised)
    assert torch.equal(out0, out1)
    out0.sum().backward()
    out1.sum().backward()
    assert torch.equal(original.grad, revised.grad)
    old.apply_stdp()
    new.apply_stdp()
    for k, v in old.state_dict().items():
        assert torch.equal(v, new.state_dict()[k]), k


def test_add_clip_bounds_and_float_learning_smaller_than_quantization_lsb():
    h = lut_layer(stdp_update_rule="add_clip")
    h.stdp_lr = 1000
    context(h, torch.tensor([[[1, 0, 0]]]), torch.tensor([[[1, 0]]]))
    h.apply_stdp()
    assert h.weight[0].tolist() == [1., 0., 0.]
    h = lut_layer()
    model = torch.nn.Module()
    model.hidden_layers = torch.nn.ModuleDict({"h": h})
    model.eval()
    before = copy.deepcopy(model.state_dict())
    for bits in (2, 4):
        with quantized_hidden_weights(model, bits):
            assert torch.unique(h.weight[0]).numel() <= 2**bits
        for k, v in before.items():
            assert torch.equal(v, model.state_dict()[k])
    context(h, torch.tensor([[[1, 0, 0]]]), torch.tensor([[[1, 0]]]))
    h.apply_stdp()
    assert h.weight.dtype == torch.float32
    assert 0 < h.stdp_statistics["actual_delta_abs_max"] < float(h.weight[0].max()/15)


