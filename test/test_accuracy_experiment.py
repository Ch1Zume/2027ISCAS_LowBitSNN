"""Behavioral checks for winner updates, persistence, and inference boundaries."""

import copy

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from snn.homeostasis import HomeostaticThreshold
from snn.inference_quant import quantize_nonnegative, quantized_hidden_weights
from snn.wta import WTAHidden, WinnerOptimizerGuard
from experiments.accuracy.config import DATASETS, load_config, epoch_budget
from experiments.accuracy.data import build_datasets, prepare_batch, stratified_split
from experiments.accuracy.models import AccuracySNN
from experiments.accuracy.run import evaluate_precisions, matrix, rng_state, restore_rng, train_epoch


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(7)
    torch.set_num_threads(2)


def compact_config():
    cfg = load_config("configs/accuracy_matrix.yaml")
    cfg["hidden_widths"] = [8, 8]
    cfg["custom_channels"] = [4, 8, 8, 8, 16, 16, 16]
    return cfg


def test_threshold_jump_decay_and_checkpoint_persistence():
    controller = HomeostaticThreshold(2, delta=0.5, tau=2)
    controller.advance(torch.tensor([[1., 0.]]), torch.tensor([[1., 0.]]))
    assert controller.threshold.tolist() == [1.5, 1.0]
    snapshot = copy.deepcopy(controller.state_dict())
    controller.advance(torch.zeros(1, 2), torch.zeros(1, 2))
    assert 1 < controller.threshold[0] < 1.5
    controller.load_state_dict(snapshot)
    assert controller.threshold[0] == 1.5


def test_rate_controller_moves_both_directions_and_respects_bounds():
    controller = HomeostaticThreshold(2, mode="rate", target_rate=0.5, eta=1,
                                     rate_tau=1, minimum=0.8, maximum=1.2)
    for _ in range(10):
        controller.advance(torch.zeros(1, 2), torch.tensor([[1., 0.]]))
    assert torch.allclose(controller.threshold, torch.tensor([1.2, 0.8]))


def test_homeostasis_breaks_repeated_winner_dominance():
    hidden = WTAHidden(1, 2, homeostasis="exponential", delta=1, tau=100)
    with torch.no_grad():
        hidden.weight.fill_(1)
    hidden(torch.ones(1, 1, 1))
    assert hidden.eligible_rows.tolist() == [True, False]
    hidden(torch.ones(1, 1, 1))
    assert hidden.eligible_rows.tolist() == [False, True]


def test_optimizer_does_not_move_losers_after_previous_momentum():
    hidden = WTAHidden(2, 3, homeostasis="off", baseline=0.5)
    with torch.no_grad():
        hidden.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [0.1, 0.1]]))
    optimizer = torch.optim.Adam(hidden.parameters(), lr=0.1, weight_decay=0.1)
    for sample, expected in [(torch.tensor([[[1., 0.]]]), 0), (torch.tensor([[[0., 1.]]]), 1)]:
        optimizer.zero_grad()
        hidden(sample).sum().backward()
        assert hidden.eligible_rows.nonzero().flatten().tolist() == [expected]
        original = hidden.weight.detach().clone()
        losers = ~hidden.eligible_rows
        with WinnerOptimizerGuard([hidden], optimizer):
            optimizer.step()
        assert torch.equal(hidden.weight[losers], original[losers])


def test_sample_specific_bp_gate_excludes_other_samples_from_winner_gradient():
    hidden = WTAHidden(2, 2, homeostasis="off", baseline=0.5)
    with torch.no_grad():
        hidden.weight.copy_(torch.tensor([[0.7, 0.], [0., 0.7]]))
    hidden(torch.tensor([[[1., 0.], [0., 1.]]])).sum().backward()
    assert hidden.weight.grad[0, 1] == 0
    assert hidden.weight.grad[1, 0] == 0
    assert hidden.weight.grad[0, 0] > 0 and hidden.weight.grad[1, 1] > 0


@pytest.mark.parametrize("rule", ["bp", "stdp"])
def test_wta_keeps_dense_input_gradients_for_ungated_backbone(rule):
    hidden = WTAHidden(2, 2, rule=rule, homeostasis="off", baseline=0.5, surrogate_type="atan")
    with torch.no_grad():
        hidden.weight.copy_(torch.tensor([[0.7, 0.1], [0.2, 0.8]]))
    inputs = torch.tensor([[[1., 0.]]], requires_grad=True)
    hidden(inputs).sum().backward()
    assert hidden.eligible_rows.tolist() == [True, False]
    reference = inputs.detach().clone().requires_grad_()
    current = torch.nn.functional.linear(reference, hidden.weight.detach(), hidden.bias.detach())
    hidden.atan(current-0.5).sum().backward()
    assert torch.allclose(inputs.grad, reference.grad)
    if rule == "bp":
        assert torch.equal(hidden.weight.grad[1], torch.zeros(2))


def test_atan_surrogate_provides_learning_signal_for_suprathreshold_winner():
    hidden = WTAHidden(2, 2, homeostasis="off", surrogate_type="atan")
    with torch.no_grad():
        hidden.weight.fill_(3)
    hidden(torch.ones(1, 1, 2)).sum().backward()
    assert hidden.weight.grad[0].min() > 0
    assert hidden.weight.grad[1].sum() == 0


def test_sparse_stdp_matches_dense_reference_and_preserves_losers():
    hidden = WTAHidden(6, 8, rule="stdp", homeostasis="off", baseline=0.5)
    pre = torch.rand(5, 4, 6)
    hidden(pre)
    original = hidden.weight.detach().clone()
    _, post = hidden.learning_context
    pre_trace, post_trace = torch.zeros_like(pre[0]), torch.zeros_like(post[0])
    update = torch.zeros_like(original)
    for inputs, winners in zip(pre, post):
        pre_trace = hidden.trace_decay*pre_trace+inputs
        update += hidden.a_plus*(hidden.weight_max-original)*(winners.T @ pre_trace)
        update -= hidden.a_minus*original*(post_trace.T @ inputs)
        post_trace = hidden.trace_decay*post_trace+winners
    rows = hidden.eligible_rows
    expected = (original[rows]+update[rows]*hidden.stdp_lr/pre.shape[1]).clamp(0, hidden.weight_max)
    expected *= (hidden.row_sum_target[rows]/expected.sum(1).clamp_min(1e-12)).unsqueeze(1)
    hidden.apply_stdp()
    assert torch.allclose(hidden.weight[rows], expected.clamp(0, hidden.weight_max), atol=1e-7)
    assert torch.equal(hidden.weight[~rows], original[~rows])


def test_mnist_time_and_training_budgets_follow_reviewed_design():
    assert DATASETS["MNIST"].steps == 4
    cfg = load_config("configs/accuracy_launch_20261004.yaml")
    assert epoch_budget(cfg, "vgg11_bn") < epoch_budget(cfg, "custom")
    assert epoch_budget(cfg, "resnet18") < epoch_budget(cfg, "custom")


def test_stdp_changes_only_winner_and_preserves_bounds():
    hidden = WTAHidden(3, 2, rule="stdp", homeostasis="off", baseline=0.3)
    with torch.no_grad():
        hidden.weight.fill_(0.2)
        hidden.reset_row_sum_target()
    original = hidden.weight.detach().clone()
    hidden(torch.tensor([[[1., 1., 0.]]]))
    hidden.apply_stdp()
    assert not torch.equal(hidden.weight[0], original[0])
    assert torch.equal(hidden.weight[1], original[1])
    assert (hidden.weight >= 0).all() and (hidden.weight <= 1).all()
    assert torch.allclose(hidden.weight.sum(1), original.sum(1))


@pytest.mark.parametrize("bits", [2, 4])
def test_quantization_preserves_zero_and_positive_endpoints(bits):
    weights = torch.tensor([[0., 0.2, 0.3], [0., 0., 0.]])
    quantized, codes, _ = quantize_nonnegative(weights, bits)
    assert quantized[0, 0] == 0 and quantized[0, 2] == weights[0, 2]
    assert torch.equal(quantized[1], weights[1])
    assert codes.max() <= (1 << bits)-1
    assert len(quantized[0].unique()) <= 1 << bits
    with pytest.raises(ValueError):
        quantize_nonnegative(torch.tensor([[-0.1, 0.2]]), bits)


def test_quantization_only_touches_hidden_fc_and_restores_on_error():
    model = AccuracySNN("custom", "MNIST", compact_config(), "rate")
    with pytest.raises(RuntimeError):
        with quantized_hidden_weights(model, 2):
            pass
    model.eval()
    original = copy.deepcopy(model.state_dict())
    with pytest.raises(RuntimeError, match="intentional"):
        with quantized_hidden_weights(model, 2):
            assert torch.equal(model.classifier.weight, original["classifier.weight"])
            assert torch.equal(model.backbone[0].weight, original["backbone.0.weight"])
            raise RuntimeError("intentional")
    for key, value in model.state_dict().items():
        assert torch.equal(value, original[key])


@pytest.mark.parametrize("mode", ["exponential", "rate"])
def test_eval_freezes_adaptation_and_membrane_reset_preserves_it(mode):
    model = AccuracySNN("custom", "MNIST", compact_config(), mode)
    hidden = next(iter(model.hidden_layers.values()))
    hidden.thresholds.offset.fill_(0.4)
    model.reset_membranes()
    assert torch.all(hidden.thresholds.offset == 0.4)
    before = copy.deepcopy(hidden.thresholds.state_dict())
    model.eval()
    model(torch.rand(2, 2, 1, 28, 28))
    for key, value in hidden.thresholds.state_dict().items():
        assert torch.equal(value, before[key])


@pytest.mark.parametrize("family", ["vgg11_bn", "resnet18", "custom"])
@pytest.mark.parametrize("dataset", list(DATASETS))
def test_official_and_custom_models_accept_native_dataset_shapes(family, dataset):
    cfg = compact_config()
    model = AccuracySNN(family, dataset, cfg, "exponential")
    spec = DATASETS[dataset]
    model.eval()
    with torch.no_grad():
        output = model(torch.rand(1, 1, spec.channels, spec.resolution, spec.resolution))
    assert output.shape == (1, spec.classes)
    assert torch.isfinite(output).all()
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, WTAHidden)):
            assert (module.weight >= 0).all()


def test_float_training_and_paired_inference_accuracy_execute():
    model = AccuracySNN("custom", "MNIST", compact_config(), "rate")
    data = [(torch.rand(2, 1, 28, 28), torch.tensor([1, 2]))]
    optimizer = torch.optim.Adam(model.bp_parameters(), lr=0.001)
    train = train_epoch(model, data, optimizer, "MNIST", torch.device("cpu"), steps=3)
    assert train["samples"] == 2
    results = evaluate_precisions(model, data, "MNIST", torch.device("cpu"), steps=3)
    assert list(results) == ["float32", "uint2", "uint4"]
    assert all(result["samples"] == 2 for result in results.values())


def test_event_coding_preserves_counts_and_stratified_split_is_disjoint():
    data = torch.full((2, 16, 2, 34, 34), 3.)
    frames = prepare_batch(data, "NMNIST", torch.device("cpu"))
    assert frames.shape == (16, 2, 2, 34, 34)
    assert frames.min() == 3
    labels = torch.arange(3).repeat_interleave(10)
    train, test = stratified_split(labels, 0.9, 42)
    assert not set(train) & set(test)
    assert set(train) | set(test) == set(range(30))
    assert train == stratified_split(labels, 0.9, 42)[0]


def test_matrix_has_24_float_runs_and_48_quantized_cases():
    summary = matrix(load_config("configs/accuracy_matrix.yaml"))
    assert summary["float_training_runs"] == 24
    assert summary["quantized_accuracy_cases"] == 48


@pytest.mark.parametrize("name", ["NMNIST", "CIFAR10DVS", "DVSGesture"])
def test_event_adapters_read_actual_cached_npz_and_disjoint_splits(tmp_path, name):
    cfg = compact_config()
    cfg["data_dir"] = str(tmp_path)
    spec = DATASETS[name]
    root = tmp_path / name
    frames = root / "frames_number_16_split_by_number"
    partitions = [""] if name == "CIFAR10DVS" else ["train", "test"]
    values = np.full((16, 2, spec.resolution, spec.resolution), 3., dtype=np.float32)
    for partition in partitions:
        for label in range(spec.classes):
            destination = frames / partition / str(label)
            destination.mkdir(parents=True)
            for sample in range(10 if not partition else 4):
                np.savez_compressed(destination / f"{sample}.npz", frames=values)
    train, validation, test, split = build_datasets(name, cfg)
    assert not set(train.indices) & set(validation.indices)
    if name == "CIFAR10DVS":
        assert not (set(train.indices) | set(validation.indices)) & set(test.indices)
    data, label = test[0]
    assert data.shape == values.shape and data.min() == 3
    assert 0 <= label < spec.classes and split.is_file()


def test_paired_poisson_inputs_are_identical_across_precisions():
    model = AccuracySNN("custom", "MNIST", compact_config(), "rate")
    data = [(torch.rand(2, 1, 28, 28), torch.tensor([1, 2]))]
    observed = []
    handle = model.register_forward_pre_hook(lambda module, inputs: observed.append(inputs[0].clone()))
    evaluate_precisions(model, data, "MNIST", torch.device("cpu"), steps=3)
    handle.remove()
    assert len(observed) == 3
    assert torch.equal(observed[0], observed[1]) and torch.equal(observed[0], observed[2])


def test_checkpoint_resume_reproduces_uninterrupted_next_epoch(tmp_path):
    cfg = compact_config()
    data = TensorDataset(torch.rand(4, 1, 28, 28), torch.tensor([1, 2, 3, 4]))
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(data, batch_size=2, shuffle=True, generator=generator)
    model = AccuracySNN("custom", "MNIST", cfg, "rate")
    optimizer = torch.optim.Adam(model.bp_parameters(), lr=0.001, weight_decay=0.01)
    train_epoch(model, loader, optimizer, "MNIST", torch.device("cpu"), steps=3)
    path = tmp_path / "resume.pth"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "rng": rng_state(), "loader_rng": generator.get_state()}, path)
    train_epoch(model, loader, optimizer, "MNIST", torch.device("cpu"), steps=3)
    expected = copy.deepcopy(model.state_dict())
    resumed = AccuracySNN("custom", "MNIST", cfg, "rate")
    opt = torch.optim.Adam(resumed.bp_parameters(), lr=0.001, weight_decay=0.01)
    saved = torch.load(path, weights_only=True)
    resumed.load_state_dict(saved["model"])
    opt.load_state_dict(saved["optimizer"])
    restore_rng(saved["rng"])
    generator.set_state(saved["loader_rng"])
    train_epoch(resumed, loader, opt, "MNIST", torch.device("cpu"), steps=3)
    for key, value in resumed.state_dict().items():
        assert torch.equal(value, expected[key]), key
