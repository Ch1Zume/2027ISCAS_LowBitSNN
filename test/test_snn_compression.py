"""Check the compression budget, reconstructed EI currents, and learning roles."""
import copy
from types import SimpleNamespace

import pytest
import torch

from experiments.accuracy.config import load_config, PROJECT_ROOT
from experiments.accuracy.models import AccuracySNN
from experiments.compression.structural import (TrainingStatistics, assert_physical_weights,
    hidden_indices, parameter_counts, prune_from_teacher, reconstruction_map, select_channels)


@pytest.fixture(autouse=True)
def cpu_seed():
    torch.set_num_threads(2)
    torch.manual_seed(5)


def test_redundant_feature_reconstruction_preserves_signed_effective_current():
    base = torch.rand(64, 2)
    features = torch.stack([base[:, 0], base[:, 1], 2 * base[:, 0], 3 * base[:, 1]], 1)
    covariance = features.T @ features / len(features)
    kept = select_channels(covariance, 2)
    assert len(set(kept.tolist())) == 2
    reconstructed = features[:, kept] @ reconstruction_map(covariance, kept, ridge=1e-7)
    torch.testing.assert_close(reconstructed, features, atol=1e-5, rtol=1e-5)
    weights = torch.randn(3, 4)
    compact = weights @ reconstruction_map(covariance, kept, ridge=1e-7).T
    excitatory, inhibitory = compact.clamp_min(0), (-compact).clamp_min(0)
    torch.testing.assert_close(features[:, kept] @ (excitatory-inhibitory).T,
                               features @ weights.T, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("granularity", ["neuron", "group"])
def test_hybrid_pruning_preserves_original_groups_and_stdp_roles(granularity):
    def hidden(groups):
        indices = torch.arange(groups * 4)
        units = indices if granularity == "neuron" else indices // 4
        return SimpleNamespace(rule="hybrid", competition_size=4, out_features=groups*4,
                               stdp_neuron_mask=units % 2 == 0)
    source, target = hidden(8), hidden(4)
    selected = hidden_indices(source, target, torch.rand(32))
    assert len(selected.unique()) == 16
    assert torch.equal(source.stdp_neuron_mask[selected], target.stdp_neuron_mask)
    for group in selected.reshape(-1, 4):
        assert torch.equal(group, group[0] + torch.arange(4))
        assert int(group[0]) % 4 == 0


def test_dense_checkpoint_transfer_does_not_modify_teacher_or_hide_parameters():
    cfg = load_config(PROJECT_ROOT / "configs/accuracy_custom_performance_20261004.yaml")
    cfg.update(custom_channels=[4, 8], hidden_widths=[12, 8], hidden_fan_in=4,
               hidden_dropout=0, learning_rules=["stdp", "bp"])
    teacher = AccuracySNN("custom", "MNIST", cfg, "exponential").eval()
    compressed_cfg = copy.deepcopy(cfg)
    compressed_cfg.update(custom_channels=[3, 5], hidden_widths=[6, 4])
    student = AccuracySNN("custom", "MNIST", compressed_cfg, "exponential").eval()
    before = {name: value.clone() for name, value in teacher.state_dict().items()}
    stats = TrainingStatistics(teacher, max_points=256)
    with torch.no_grad():
        teacher(torch.rand(4, 12, 1, 28, 28))
    cov, activity = stats.finish()
    record = prune_from_teacher(teacher, student, cov, activity, torch.arange(12) % 10)
    assert record["hybrid_rules_preserved"]
    assert_physical_weights(student)
    for name, value in teacher.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    counts = parameter_counts(student)
    assert counts["total"] < parameter_counts(teacher)["total"]
    assert counts["total"] == sum(p.numel() for p in student.parameters())
    student.reset_membranes()
    with torch.no_grad():
        output = student(torch.rand(4, 2, 1, 28, 28))
    assert output.shape == (2, 10) and torch.isfinite(output).all()
    assert not student.hidden_layers["hidden1"].weight.requires_grad
    assert student.hidden_layers["hidden2"].weight.requires_grad
    assert not any("mask" in name for name in student.state_dict())


def test_calibration_failing_shape_and_invalid_budget_are_explicit():
    with pytest.raises(ValueError):
        select_channels(torch.eye(3), 4)
    with pytest.raises(ValueError):
        select_channels(torch.ones(3, 2), 1)
