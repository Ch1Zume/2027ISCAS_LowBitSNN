"""Verify real compressed-model TRAIN updates on an isolated checkpoint copy."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

from structural import assert_physical_weights, parameter_counts


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=3)
    args = parser.parse_args()
    if args.batches <= 0:
        parser.error("batches must be positive")
    directory = args.condition.resolve()
    provenance = json.loads((directory / "provenance.json").read_text())
    source = Path(provenance["source_root"])
    for name, expected in provenance["source_sha256"].items():
        if digest(source / name) != expected:
            raise AssertionError("Pinned model runtime changed")
    sys.path.insert(0, str(source))
    from experiments.accuracy.models import AccuracySNN
    from experiments.accuracy.data import build_loaders, prepare_batch
    from experiments.accuracy.run import seed_everything
    from snn.wta import WinnerOptimizerGuard

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    seed_everything(3142)
    path = directory / "best.pth"
    before_digest = digest(path)
    saved = torch.load(path, map_location=device, weights_only=True)
    cfg = json.loads((directory / "config.json").read_text())
    cfg["num_workers"] = 0
    model = AccuracySNN("custom", provenance["dataset"], cfg, "exponential").to(device)
    model.load_state_dict(saved["model"], strict=True)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.bp_parameters(), lr=1e-4)
    train = build_loaders(provenance["dataset"], cfg)[0]
    record = {"dataset": provenance["dataset"], "parameters": parameter_counts(model),
              "checkpoint_sha256": before_digest, "source_partition": "TRAIN", "batches": []}
    model.train()
    for index, (data, labels) in enumerate(train, 1):
        model.reset_membranes()
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        old_parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        logits = model(prepare_batch(data, provenance["dataset"], device, cfg=cfg))
        loss = F.cross_entropy(logits, labels.to(device))
        loss.backward()
        if any(parameter.grad.dtype != torch.float32 for parameter in model.parameters() if parameter.grad is not None):
            raise AssertionError("A training gradient is not FP32")
        rules = {}
        protected_states = []
        for name, hidden in model.hidden_layers.items():
            stdp_mask = getattr(hidden, "stdp_neuron_mask", torch.full_like(hidden.eligible_rows, hidden.rule == "stdp"))
            bp = hidden.eligible_rows & ~stdp_mask
            stdp = hidden.eligible_rows & stdp_mask
            rules[name] = (bp, stdp)
            for parameter in (hidden.weight, hidden.bias):
                for key, value in optimizer.state.get(parameter, {}).items():
                    if isinstance(value, torch.Tensor) and value.shape == parameter.shape:
                        protected_states.append((parameter, key, ~bp, value[~bp].clone()))
        torch.nn.utils.clip_grad_norm_(model.bp_parameters(), 1, error_if_nonfinite=True)
        with WinnerOptimizerGuard(model.hidden_layers.values(), optimizer):
            optimizer.step()
        for parameter, key, mask, values in protected_states:
            if not torch.equal(optimizer.state[parameter][key][mask], values):
                raise AssertionError("Loser or STDP optimizer momentum changed")
        for name, hidden in model.hidden_layers.items():
            bp, stdp = rules[name]
            for suffix, parameter in (("weight", hidden.weight), ("bias", hidden.bias)):
                if not torch.equal(parameter[~bp], old_parameters[f"hidden_layers.{name}.{suffix}"][~bp]):
                    raise AssertionError("BP wrote a loser or a STDP-designated row")
            hidden.apply_stdp()
        model.project_nonnegative()
        layers = {}
        for name, hidden in model.hidden_layers.items():
            bp, stdp = rules[name]
            weight_changes = (hidden.weight != old_parameters[f"hidden_layers.{name}.weight"]).any(1)
            bias_changes = hidden.bias != old_parameters[f"hidden_layers.{name}.bias"]
            if weight_changes[~(bp | stdp)].any() or bias_changes[~bp].any():
                raise AssertionError("Hidden weights or biases changed outside their assigned winners")
            layers[name] = {"bp_winners": int(bp.sum()), "stdp_winners": int(stdp.sum()),
                            "bp_rows_changed": int((weight_changes & bp).sum()),
                            "stdp_rows_changed": int((weight_changes & stdp).sum())}
        physical = assert_physical_weights(model)
        backbone_changed = sum(int((parameter != old_parameters[name]).sum())
                               for name, parameter in model.named_parameters() if name.startswith("backbone."))
        head_changed = sum(int((parameter != old_parameters[name]).sum())
                           for name, parameter in model.named_parameters() if name.startswith("classifier."))
        if backbone_changed == 0 or head_changed == 0:
            raise AssertionError("Backbone or head BP failed to update")
        record["batches"].append({"batch": index, "loss": float(loss.detach()), "layers": layers,
                                  "backbone_parameters_changed": backbone_changed,
                                  "head_parameters_changed": head_changed,
                                  "physical_layers_checked": len(physical), "loser_momentum_preserved": True})
        if index >= args.batches:
            break
    model.load_state_dict(before, strict=True)
    if any(not torch.equal(value, before[name]) for name, value in model.state_dict().items()):
        raise AssertionError("Isolated model did not restore exactly")
    if digest(path) != before_digest:
        raise AssertionError("Published compression checkpoint changed")
    record.update(passed=True, isolated_model_restored=True, published_checkpoint_unchanged=True,
                  fp32_gradients=True, nonnegative_physical_weights=True, budget_passed=3_000_000 <= record["parameters"]["total"] <= 5_000_000)
    (directory / "training_constraints_audit.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
