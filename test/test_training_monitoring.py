"""Failure containment and telemetry must not perturb a valid training run."""

import copy
import json
from argparse import Namespace

import pytest
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from experiments.accuracy import campaign, run
from experiments.accuracy.config import load_config
from experiments.accuracy.data import build_probe_loader
from experiments.accuracy.models import AccuracySNN
from experiments.accuracy.monitoring import (TrainingAborted, TrainingGuard,
                                             TrainingMonitor, monitoring_config)


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(7)
    torch.set_num_threads(2)


def compact_config():
    cfg = load_config("configs/accuracy_matrix.yaml")
    cfg["hidden_widths"] = [8, 8]
    cfg["custom_channels"] = [4, 8, 8, 8, 16, 16, 16]
    return cfg


def collapsed_validation(loss=2.3):
    return {"correct": 1, "samples": 10, "top1_accuracy": 0.1, "loss": loss,
            "prediction_histogram": [0, 10]+[0]*8, "label_histogram": [1]*10,
            "dominant_prediction_fraction": 1.0, "majority_class_baseline": 0.1,
            "logit_range_max": 0.0}


def test_single_class_collapse_stops_early_and_state_survives_resume():
    guard = TrainingGuard(monitoring_config({"warmup_batches": 1}))
    guard.check_validation(collapsed_validation(), initial=True)
    guard.state["global_step"] = 100
    guard.check_validation(collapsed_validation())
    guard.check_validation(collapsed_validation())
    resumed = TrainingGuard(guard.config)
    resumed.state = copy.deepcopy(guard.state)
    resumed.check_validation(collapsed_validation(), initial=True)
    with pytest.raises(TrainingAborted, match="persistent_single_class"):
        resumed.check_validation(collapsed_validation())


def test_improving_loss_or_diverse_predictions_prevents_false_collapse_stop():
    guard = TrainingGuard(monitoring_config({"warmup_batches": 1}))
    guard.check_validation(collapsed_validation(), initial=True)
    guard.state["global_step"] = 100
    for loss in [2.25, 2.20, 2.15, 2.10]:
        guard.check_validation(collapsed_validation(loss))
    for _ in range(10):
        guard.check_validation(collapsed_validation(2.10) | {"dominant_prediction_fraction": 0.4})
    assert guard.state["validation"]["probe"]["collapse_checks"] == 0


def test_single_label_probe_cannot_establish_model_collapse():
    guard = TrainingGuard(monitoring_config({"warmup_batches": 1}))
    guard.state["global_step"] = 100
    biased = collapsed_validation() | {"correct": 0, "top1_accuracy": 0.0,
                                       "label_histogram": [10]+[0]*9, "majority_class_baseline": 1.0}
    for _ in range(10):
        guard.check_validation(biased)
    assert guard.state["validation"]["probe"]["collapse_checks"] == 0


def test_collapse_guard_can_wait_for_one_completed_epoch():
    guard = TrainingGuard(monitoring_config({"warmup_batches": 1, "collapse_warmup_epochs": 1}))
    guard.state["global_step"] = 100
    for _ in range(10):
        guard.check_validation(collapsed_validation())
    assert guard.state["validation"]["probe"]["collapse_checks"] == 0
    guard.epoch_should_stop(0.1, 1)
    guard.check_validation(collapsed_validation())
    guard.check_validation(collapsed_validation())
    with pytest.raises(TrainingAborted, match="persistent_single_class"):
        guard.check_validation(collapsed_validation())


def test_dead_activity_requires_warmup_and_repeated_blocked_gradients():
    guard = TrainingGuard(monitoring_config({"warmup_batches": 10, "dead_patience": 3}))
    dead = {"backbone_spike_rate": 0.0, "backbone_gradient_norm": 0.0}
    for _ in range(10):
        guard.check_health(dead)
    assert guard.state["dead_checks"] == 0
    guard.state["global_step"] = 10
    guard.check_health(dead)
    guard.check_health(dead | {"backbone_gradient_norm": 0.1})
    assert guard.state["dead_checks"] == 0
    guard.check_health(dead)
    guard.check_health(dead)
    with pytest.raises(TrainingAborted, match="silent_spikes"):
        guard.check_health(dead)


def test_plateau_stops_by_validation_accuracy_and_recovery_resets_patience():
    guard = TrainingGuard(monitoring_config({"early_stopping_patience": 2, "early_stopping_min_epochs": 2}))
    assert not guard.epoch_should_stop(0.5, 1)
    assert not guard.epoch_should_stop(0.5, 2)
    assert not guard.epoch_should_stop(0.6, 3)
    assert not guard.epoch_should_stop(0.6, 4)
    assert guard.epoch_should_stop(0.6, 5)


@pytest.mark.parametrize("bad", [{"probe_samples": 0}, {"log_every_seconds": float("nan")},
                                  {"collapse_fraction": 0.1}, {"unknown": 1},
                                  {"dead_patience": True}])
def test_invalid_monitor_policy_is_rejected(bad):
    with pytest.raises(ValueError):
        monitoring_config(bad)


def test_probe_and_logging_preserve_weights_thresholds_and_training_rng(tmp_path):
    cfg = compact_config()
    data = [(torch.rand(2, 1, 28, 28), torch.tensor([1, 2]))]*2
    model = AccuracySNN("custom", "MNIST", cfg, "rate")
    monitored = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.bp_parameters(), lr=0.001)
    monitored_optimizer = torch.optim.Adam(monitored.bp_parameters(), lr=0.001)
    initial_rng = run.rng_state()
    run.train_epoch(model, data, optimizer, "MNIST", torch.device("cpu"), steps=3)
    expected_rng = run.rng_state()
    run.restore_rng(initial_rng)
    monitor = TrainingMonitor(tmp_path, monitoring_config({"log_every_batches": 1, "warmup_batches": 1000,
                                                         "probe_every_batches": 1}))
    monitor.attach(monitored)
    validation = DataLoader(TensorDataset(torch.cat([item[0] for item in data]),
                                          torch.cat([item[1] for item in data])), batch_size=2)
    probe_loader, _ = build_probe_loader(validation, 3, 2042)
    def probe(epoch, batch):
        metrics = run.validation_probe(monitored, probe_loader, "MNIST", torch.device("cpu"), 3, 3, 2042)
        assert metrics["samples"] == 3
        assert monitored.training
    try:
        run.train_epoch(monitored, data, monitored_optimizer, "MNIST", torch.device("cpu"), steps=3,
                        monitor=monitor, on_probe=probe)
    finally:
        monitor.close()
    for key, value in model.state_dict().items():
        assert torch.equal(value, monitored.state_dict()[key]), key
    assert torch.equal(expected_rng[0], run.rng_state()[0])
    logs = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(logs) == 2
    assert all("backbone_gradient_norm" in log and "hidden" in log and "loss" in log for log in logs)
    assert not monitored.hidden_layers["hidden1"]._forward_hooks


def test_probe_covers_class_ordered_nested_validation_without_training_rng_changes():
    labels = torch.arange(10).repeat_interleave(600)
    dataset = TensorDataset(torch.arange(len(labels)), labels)
    validation = Subset(Subset(dataset, range(len(labels))), range(0, len(labels), 2))
    loader = DataLoader(validation, batch_size=8, generator=torch.Generator().manual_seed(99))
    initial_rng, initial_loader_rng = torch.get_rng_state().clone(), loader.generator.get_state().clone()
    probe, manifest = build_probe_loader(loader, 256, 2042)
    again, repeated = build_probe_loader(loader, 256, 2042)
    assert manifest == repeated
    assert torch.equal(initial_rng, torch.get_rng_state())
    assert torch.equal(initial_loader_rng, loader.generator.get_state())
    assert len(set(manifest["indices_relative_to_validation"])) == 256
    assert sum(manifest["label_histogram"]) == 256
    assert min(manifest["label_histogram"]) == 25 and max(manifest["label_histogram"]) == 26
    observed = torch.cat([batch_labels for _, batch_labels in probe])
    assert torch.bincount(observed).tolist() == manifest["label_histogram"]
    assert probe.dataset.dataset is validation and again.dataset.dataset is validation


def test_probe_redistributes_exhausted_classes_and_handles_small_validation_set():
    labels = torch.tensor([0]+[1]*5+[2]*10)
    loader = DataLoader(TensorDataset(torch.arange(len(labels)), labels), batch_size=2)
    _, manifest = build_probe_loader(loader, 10, 2042)
    assert manifest["label_histogram"] == [1, 5, 4]
    _, manifest = build_probe_loader(loader, 256, 2042)
    assert manifest["samples"] == len(labels) and manifest["label_histogram"] == [1, 5, 10]


class NonfiniteModel(torch.nn.Module):
    def __init__(self, bad_gradient=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.hidden_layers = torch.nn.ModuleDict()
        self.bad_gradient = bad_gradient

    def reset_membranes(self):
        pass

    def bp_parameters(self):
        return list(self.parameters())

    def forward(self, frames):
        if self.bad_gradient:
            self.weight.register_hook(lambda grad: torch.full_like(grad, float("nan")))
            return self.weight.unsqueeze(0).expand(frames.shape[1], -1)
        return self.weight.unsqueeze(0).expand(frames.shape[1], -1) * float("nan")


@pytest.mark.parametrize("bad_gradient,reason", [(False, "nonfinite_training_loss"),
                                                (True, "nonfinite_training_gradient")])
def test_nonfinite_training_aborts_before_optimizer_mutates_weights(bad_gradient, reason):
    model = NonfiniteModel(bad_gradient)
    original = model.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    data = [(torch.rand(2, 1, 28, 28), torch.tensor([0, 1]))]
    with pytest.raises(TrainingAborted, match=reason):
        run.train_epoch(model, data, optimizer, "MNIST", torch.device("cpu"), steps=2)
    assert torch.equal(model.weight, original)
    assert not optimizer.state


def test_aborted_run_saves_diagnostics_without_final_test_or_accuracy(tmp_path, monkeypatch):
    cfg = compact_config()
    cfg["out_dir"] = str(tmp_path)
    cfg["monitoring"] = monitoring_config({"warmup_batches": 1, "probe_every_batches": 1,
                                          "collapse_patience": 2, "dead_patience": 100})
    dataset = TensorDataset(torch.rand(4, 1, 28, 28), torch.tensor([0, 1, 0, 1]))
    loader = DataLoader(dataset, batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(42))
    validation, test = object(), object()
    monkeypatch.setattr(run, "build_loaders", lambda *a, **kw: (loader, validation, test, None))
    monkeypatch.setattr(run, "build_probe_loader", lambda *a: (validation, {"sampling": run.PROBE_SAMPLING}))
    def evaluate(model, source, *args, **kwargs):
        assert source is validation, "An aborted run must not evaluate the test set"
        return collapsed_validation()
    monkeypatch.setattr(run, "evaluate", evaluate)
    args = Namespace(model="custom", dataset="MNIST", homeostasis="rate", device="cpu", resume=None,
                     batch_size=2, limit=4, steps=2, epochs=2, init_checkpoint=None, download=False)
    with pytest.raises(SystemExit) as error:
        run.run_training(args, cfg)
    assert error.value.code == 2
    out = tmp_path / "custom_MNIST_rate_seed42" / "functional_check"
    assert not (out / "accuracy.json").exists()
    stopped = json.loads((out / "stopped.json").read_text())
    assert stopped["status"] == "aborted" and stopped["final_test_evaluated"] is False
    saved = torch.load(out / "diagnostic.pth", weights_only=True)
    assert saved["monitoring_state"]["global_step"] == 2
    assert saved["resume_supported"] is False


def test_plateau_completes_paired_evaluation_and_resume_does_not_train_again(tmp_path, monkeypatch):
    cfg = compact_config()
    cfg["out_dir"] = str(tmp_path)
    cfg["monitoring"] = monitoring_config({"early_stopping_patience": 2, "early_stopping_min_epochs": 2})
    dataset = TensorDataset(torch.rand(4, 1, 28, 28), torch.tensor([0, 1, 0, 1]))
    training = DataLoader(dataset, batch_size=2, generator=torch.Generator().manual_seed(42))
    validation = DataLoader(dataset, batch_size=2)
    test = DataLoader(dataset, batch_size=2)
    monkeypatch.setattr(run, "build_loaders", lambda *a, **kw: (training, validation, test, None))
    actual_evaluate, test_calls = run.evaluate, []
    def evaluate(model, source, *args, **kwargs):
        scores = actual_evaluate(model, source, *args, **kwargs)
        if source is validation:
            return scores | {"loss": 2.3, "top1_accuracy": 0.5, "dominant_prediction_fraction": 0.5}
        if source is test:
            test_calls.append(source)
        else:
            assert isinstance(source.dataset, Subset) and source.dataset.dataset is validation.dataset
        return scores
    monkeypatch.setattr(run, "evaluate", evaluate)
    args = Namespace(model="custom", dataset="MNIST", homeostasis="rate", device="cpu", resume=None,
                     batch_size=2, limit=4, steps=2, epochs=10, init_checkpoint=None, download=False)
    run.run_training(args, cfg)
    out = tmp_path / "custom_MNIST_rate_seed42" / "functional_check"
    saved = torch.load(out / "latest.pth", weights_only=True)
    assert saved["epoch"] == 2 and len(saved["history"]) == 3
    result = json.loads((out / "accuracy.json").read_text())
    assert result["training_stop"]["reason"] == "validation_accuracy_plateau"
    assert result["selected_epoch"] == 0
    assert list(result["test"]) == ["float32", "uint2", "uint4"]
    assert test_calls == [test]*3
    def unexpected_training(*args, **kwargs):
        pytest.fail("A resumed plateau checkpoint started another training epoch")
    monkeypatch.setattr(run, "train_epoch", unexpected_training)
    args.resume = str(out / "latest.pth")
    run.run_training(args, cfg)
    assert test_calls == [test]*6


def test_selected_evaluation_reports_collapsed_checkpoint_accuracy_with_explicit_outcome(tmp_path, monkeypatch):
    cfg = compact_config()
    cfg.update(out_dir=str(tmp_path), evaluate_on_collapse=True)
    cfg["monitoring"] = monitoring_config({"warmup_batches": 1, "collapse_warmup_epochs": 1,
                                          "probe_every_batches": 1, "collapse_patience": 2, "dead_patience": 100})
    dataset = TensorDataset(torch.rand(4, 1, 28, 28), torch.tensor([0, 1, 0, 1]))
    training = DataLoader(dataset, batch_size=2, generator=torch.Generator().manual_seed(42))
    validation, test = DataLoader(dataset, batch_size=2), DataLoader(dataset, batch_size=2)
    monkeypatch.setattr(run, "build_loaders", lambda *a, **kw: (training, validation, test, None))
    actual_evaluate, test_calls = run.evaluate, []
    def evaluate(model, source, *args, **kwargs):
        if source is test:
            test_calls.append(source)
            return actual_evaluate(model, source, *args, **kwargs)
        return collapsed_validation()
    monkeypatch.setattr(run, "evaluate", evaluate)
    args = Namespace(model="custom", dataset="MNIST", homeostasis="exponential", device="cpu", resume=None,
                     batch_size=2, limit=4, steps=2, epochs=10, init_checkpoint=None, download=False)
    run.run_training(args, cfg)
    out = tmp_path / "custom_MNIST_exponential_seed42" / "functional_check"
    result = json.loads((out / "accuracy.json").read_text())
    assert result["training_outcome"] == "class_collapse"
    assert result["training_stop"]["reason"] == "persistent_single_class_predictions_without_validation_improvement"
    assert result["selected_epoch"] == 0 and test_calls == [test]*3
    assert all(score["samples"] == 4 for score in result["test"].values())
    assert (out / "stopped.json").exists() and (out / "diagnostic.pth").exists()


def test_campaign_remains_paused_and_does_not_launch_any_child(tmp_path, monkeypatch):
    import yaml
    cfg = compact_config()
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    (tmp_path / "status.json").write_text(json.dumps({"status": "paused"}))
    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *a, **k: pytest.fail("Paused campaign launched a child"))
    with pytest.raises(SystemExit, match="paused"):
        campaign.run(tmp_path, "configs/accuracy_matrix.yaml")


def test_campaign_halts_queue_when_child_monitor_aborts(tmp_path, monkeypatch):
    cfg = compact_config()
    cfg.update(models=["custom"], datasets=["MNIST"], homeostasis_modes=["exponential", "rate"])
    monkeypatch.setattr(campaign, "load_config", lambda *a: cfg)
    (tmp_path / "datasets.json").write_text(json.dumps({"MNIST": {"status": "ready"}}))
    calls = []
    class AbortedProcess:
        pid = 123
        def __init__(self, *args, **kwargs):
            calls.append(args)
            directory = tmp_path / "runs" / "custom_MNIST_exponential_seed42"
            directory.mkdir(parents=True)
            (directory / "stopped.json").write_text(json.dumps({"reason": "persistent_single_class"}))
        def wait(self):
            return 2
    monkeypatch.setattr(campaign.subprocess, "Popen", AbortedProcess)
    campaign.run(tmp_path, "configs/accuracy_matrix.yaml")
    state = json.loads((tmp_path / "status.json").read_text())
    assert len(calls) == 1
    assert state["status"] == "stopped_for_review"
    assert [job["status"] for job in state["jobs"]] == ["aborted", "queued"]


def test_campaign_accuracy_table_uses_final_scores_and_leaves_pending_metrics_empty(tmp_path):
    import csv
    completed = {"dataset": "MNIST", "model": "custom", "homeostasis": "exponential",
                 "status": "completed", "result": "/verified/accuracy.json",
                 "test": {key: {"samples": 10000, "top1_accuracy": accuracy} for key, accuracy in
                          [("float32", 0.9895), ("uint2", 0.9896), ("uint4", 0.9895)]}}
    pending = {"dataset": "NMNIST", "model": "custom", "homeostasis": "exponential", "status": "queued"}
    campaign.write_json(tmp_path / "status.json", {"jobs": [completed, pending]})
    with (tmp_path / "accuracy_summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["float32_top1_percent"] == "98.9500" and rows[0]["uint2_top1_percent"] == "98.9600"
    assert rows[0]["test_samples"] == "10000" and rows[0]["result"] == completed["result"]
    assert all(rows[1][key] == "" for key in ("test_samples", "float32_top1_percent", "uint2_top1_percent", "uint4_top1_percent"))


def test_campaign_stops_after_unexpected_worker_failure(tmp_path, monkeypatch):
    cfg = compact_config()
    cfg.update(models=["custom"], datasets=["MNIST"], homeostasis_modes=["exponential", "rate"])
    monkeypatch.setattr(campaign, "load_config", lambda *a: cfg)
    (tmp_path / "datasets.json").write_text(json.dumps({"MNIST": {"status": "ready"}}))
    calls = []
    class FailedProcess:
        pid = 123
        def __init__(self, *args, **kwargs):
            calls.append(args)
        def wait(self):
            return 1
    monkeypatch.setattr(campaign.subprocess, "Popen", FailedProcess)
    campaign.run(tmp_path, "configs/accuracy_matrix.yaml")
    state = json.loads((tmp_path / "status.json").read_text())
    assert len(calls) == 1 and state["status"] == "stopped_for_review"
    assert [job["status"] for job in state["jobs"]] == ["failed", "queued"]
