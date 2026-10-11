"""Local, flushed training telemetry and persistent guards against wasted runs."""

import json
import math
import time
from datetime import datetime, timezone

import torch
from spikingjelly.activation_based import neuron


DEFAULT_MONITORING = {
    "log_every_batches": 20,
    "log_every_seconds": 30.0,
    "probe_every_batches": 100,
    "probe_samples": 256,
    "warmup_batches": 100,
    "collapse_warmup_epochs": 0,
    "dead_patience": 3,
    "collapse_patience": 3,
    "collapse_fraction": 0.99,
    "collapse_accuracy_margin": 0.02,
    "min_loss_improvement": 0.01,
    "min_accuracy_improvement": 0.002,
    "early_stopping_patience": 5,
    "early_stopping_min_epochs": 3,
}


def monitoring_config(overrides=None):
    overrides = {} if overrides is None else overrides
    if not isinstance(overrides, dict):
        raise ValueError("monitoring must be a mapping")
    if set(overrides) - DEFAULT_MONITORING.keys():
        raise ValueError(f"Unknown monitoring keys: {sorted(set(overrides)-DEFAULT_MONITORING.keys())}")
    config = DEFAULT_MONITORING | overrides
    for key, value in config.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"monitoring.{key} must be a finite number")
        if key.endswith(("batches", "samples", "patience", "epochs")):
            minimum = 0 if key == "collapse_warmup_epochs" else 1
            if not isinstance(value, int) or value < minimum:
                raise ValueError(f"monitoring.{key} must be a positive integer")
        elif value <= 0:
            raise ValueError(f"monitoring.{key} must be positive")
    if not 0.5 < config["collapse_fraction"] <= 1:
        raise ValueError("monitoring.collapse_fraction must be in (0.5, 1]")
    if config["collapse_accuracy_margin"] >= 1 or config["min_accuracy_improvement"] >= 1:
        raise ValueError("Monitoring accuracy margins must be below 1")
    return config


class TrainingAborted(RuntimeError):
    def __init__(self, reason, metrics):
        super().__init__(reason)
        self.reason, self.metrics = reason, metrics


class TrainingGuard:
    """Diagnose repeated failure; one low-accuracy batch cannot abort a run."""
    def __init__(self, config):
        self.config = config
        self.state = {"global_step": 0, "dead_checks": 0, "validation": {},
                      "best_epoch_accuracy": -1.0, "stale_epochs": 0, "completed_epochs": 0}

    def check_health(self, metrics):
        if self.state["global_step"] < self.config["warmup_batches"]:
            return
        silent = metrics.get("backbone_spike_rate") == 0 or (
            bool(metrics.get("hidden")) and
            all(layer["spike_rate"] == 0 for layer in metrics["hidden"].values()))
        blocked = silent and metrics.get("backbone_gradient_norm", float("inf")) <= 1e-12
        self.state["dead_checks"] = self.state["dead_checks"] + 1 if blocked else 0
        if self.state["dead_checks"] >= self.config["dead_patience"]:
            raise TrainingAborted("silent_spikes_and_blocked_backbone_gradient", metrics)

    def check_validation(self, metrics, stage="probe", initial=False):
        if not math.isfinite(metrics["loss"]):
            raise TrainingAborted("nonfinite_validation_loss", metrics)
        if initial and stage in self.state["validation"]:
            return  # A resume-time observation must not erase accumulated evidence.
        tracker = self.state["validation"].setdefault(stage, {
            "best_loss": float("inf"), "best_accuracy": -1.0, "collapse_checks": 0})
        improved = (metrics["loss"] < tracker["best_loss"] - self.config["min_loss_improvement"] or
                    metrics["top1_accuracy"] > tracker["best_accuracy"] + self.config["min_accuracy_improvement"])
        if metrics["loss"] < tracker["best_loss"] - self.config["min_loss_improvement"]:
            tracker["best_loss"] = metrics["loss"]
        if metrics["top1_accuracy"] > tracker["best_accuracy"] + self.config["min_accuracy_improvement"]:
            tracker["best_accuracy"] = metrics["top1_accuracy"]
        diverse_labels = sum(count > 0 for count in metrics.get("label_histogram", [])) >= 2
        collapsed = (diverse_labels and
                     metrics["dominant_prediction_fraction"] >= self.config["collapse_fraction"] and
                     metrics["top1_accuracy"] <= metrics["majority_class_baseline"] + self.config["collapse_accuracy_margin"])
        eligible = (self.state["global_step"] >= self.config["warmup_batches"] and
                    self.state.get("completed_epochs", 0) >= self.config["collapse_warmup_epochs"])
        if collapsed and eligible and not improved and not initial:
            tracker["collapse_checks"] += 1
        else:
            tracker["collapse_checks"] = 0
        if tracker["collapse_checks"] >= self.config["collapse_patience"]:
            raise TrainingAborted("persistent_single_class_predictions_without_validation_improvement", metrics)

    def epoch_should_stop(self, accuracy, completed_epochs):
        self.state["completed_epochs"] = completed_epochs
        if accuracy > self.state["best_epoch_accuracy"] + self.config["min_accuracy_improvement"]:
            self.state["best_epoch_accuracy"] = accuracy
            self.state["stale_epochs"] = 0
        else:
            self.state["stale_epochs"] += 1
        return self.plateau_reached(completed_epochs)

    def plateau_reached(self, completed_epochs):
        return (completed_epochs >= self.config["early_stopping_min_epochs"] and
                self.state["stale_epochs"] >= self.config["early_stopping_patience"])


class TrainingMonitor:
    def __init__(self, directory, config):
        self.directory, self.config = directory, config
        self.guard = TrainingGuard(config)
        self.stream = (directory / "metrics.jsonl").open("a", buffering=1)
        self.started = self.last_log = time.monotonic()
        self.capture = False
        self.activities, self.handles = {}, []

    def emit(self, event, **metrics):
        record = {"event": event, "at": datetime.now(timezone.utc).isoformat(),
                  "global_step": self.guard.state["global_step"], **metrics}
        # Diagnostics may contain NaN/Inf on failed runs; serialize them as null.
        def clean(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, dict):
                return {key: clean(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [clean(item) for item in value]
            return value
        line = json.dumps(clean(record), allow_nan=False)
        print(line, flush=True)
        self.stream.write(line + "\n")
        self.stream.flush()
        return clean(record)

    def attach(self, model):
        def activity(name):
            def hook(module, inputs, output):
                if self.capture:
                    self.activities[name] = output.detach().float().mean()
            return hook
        nodes = [module for module in model.backbone.modules() if isinstance(module, neuron.BaseNode)]
        if nodes:
            self.handles.append(nodes[-1].register_forward_hook(activity("backbone")))
        for name, hidden in model.hidden_layers.items():
            self.handles.append(hidden.register_forward_hook(activity(name)))

    def begin_batch(self, batch_index):
        self.guard.state["global_step"] += 1
        self.capture = (batch_index == 1 or
                        self.guard.state["global_step"] % self.config["log_every_batches"] == 0 or
                        time.monotonic() - self.last_log >= self.config["log_every_seconds"])
        self.activities.clear()
        return self.capture

    def should_probe(self):
        return self.guard.state["global_step"] % self.config["probe_every_batches"] == 0

    @torch.no_grad()
    def diagnostics(self, model):
        gradients = [parameter.grad.detach().norm() for parameter in model.backbone.parameters()
                     if parameter.grad is not None]
        result = {"backbone_gradient_norm": torch.stack(gradients).norm().item() if gradients else 0.0,
                  "hidden": {}}
        if "backbone" in self.activities:
            result["backbone_spike_rate"] = self.activities["backbone"].item()
        for name, hidden in model.hidden_layers.items():
            threshold = hidden.thresholds.threshold
            result["hidden"][name] = {
                "rule": hidden.rule,
                "spike_rate": self.activities[name].item(),
                "eligible_neurons": int(hidden.eligible_rows.sum().item()),
                "bp_eligible_neurons": int((hidden.eligible_rows & hidden.bp_neuron_mask).sum().item()) if hidden.training else 0,
                "stdp_eligible_neurons": int((hidden.eligible_rows & hidden.stdp_neuron_mask).sum().item()) if hidden.training else 0,
                "hidden_learning_enabled": hidden.training,
                "threshold_min": threshold.min().item(), "threshold_max": threshold.max().item()}
        return result

    def log_batch(self, **metrics):
        self.emit("train_progress", **metrics)
        self.last_log = time.monotonic()
        self.capture = False
        # Wall-clock logging must not change the deterministic guard cadence.
        if self.guard.state["global_step"] % self.config["log_every_batches"] == 0:
            self.guard.check_health(metrics)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.stream.close()
