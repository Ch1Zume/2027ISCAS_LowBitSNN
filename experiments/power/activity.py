"""Read-only activity hooks. Logical activity is not a hardware power measurement."""
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def distribution(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(mean=float(a.mean()), p50=float(np.quantile(a, .5)), p95=float(np.quantile(a, .95)),
                p99=float(np.quantile(a, .99)), minimum=float(a.min()), maximum=float(a.max()))


@torch.no_grad()
def hidden_shadow(module, x, spikes):
    """Reconstruct local state without modifying forward, RNG, or parameters.

    Require every reconstructed threshold event to match the actual forward.
    Values remain algorithmic membrane units, not volts or charge.
    """
    currents = F.linear(x, module.weight, module.bias)
    threshold = module.thresholds.threshold.detach().clone().to(currents)
    mem = torch.zeros_like(currents[0])
    before, after, winners = [], [], []
    groups = module.out_features // module.competition_size
    chosen = torch.zeros((x.shape[1], groups), device=x.device, dtype=torch.bool)
    for t, current in enumerate(currents):
        pre = module.membrane_decay * mem + current
        predicted = (module.atan(pre - threshold) if module.surrogate_type == "atan" else
                     (pre >= threshold).to(pre))
        if not torch.equal(predicted, spikes[t]):
            raise RuntimeError("Shadow membrane differs from frozen spike forward")
        candidates = spikes[t].bool().reshape(x.shape[1], groups, module.competition_size)
        if module.winner_scope == "presentation":
            candidates = candidates & ~chosen.unsqueeze(-1)
        has = candidates.any(-1)
        if module.winner_tiebreak == "margin":
            margins = (pre - threshold).reshape_as(candidates)
            indices = margins.masked_fill(~candidates, -torch.inf).argmax(-1)
        else:
            indices = candidates.to(torch.int64).argmax(-1)
        selected = torch.zeros_like(candidates)
        selected.scatter_(2, indices.unsqueeze(-1), has.unsqueeze(-1))
        chosen |= has
        mem = pre - spikes[t].detach() * threshold
        before.append(pre)
        after.append(mem)
        winners.append(selected.reshape_as(spikes[t]))
    return currents, torch.stack(before), torch.stack(after), torch.stack(winners), threshold


class ActivityCollector:
    def __init__(self, base, out, samples, views, steps, hidden_type, neuron_type):
        self.base, self.out = base, Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.samples, self.views, self.steps = samples, views, steps
        self.offset, self.current_view = 0, -1
        self.handles, self.stats, self.batch = [], {}, {}
        self.candidates, self.traces, self.degree_cache = {}, {}, {}
        self.labels = None
        self.stream = (self.out / "per_sample_layers.csv").open("w")
        self.fields = ["sample_index", "label", "layer", "role", "input_nonzero", "addressed_terms",
                       "effective_nonzero_weight_terms", "fire", "winner", "neuron_updates",
                       "software_spike_reset", "logical_boundary_initializations", "end_nonzero_membrane",
                       "input_slots", "neuron_slots"]
        self.writer = csv.DictWriter(self.stream, fieldnames=self.fields)
        self.writer.writeheader()
        self.handles.append(base.register_forward_pre_hook(self._base_input))
        for name, module in base.named_modules():
            if isinstance(module, hidden_type):
                self.handles.append(module.register_forward_hook(self._hidden_hook(name)))
            elif isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                self.handles.append(module.register_forward_hook(self._synapse_hook(name)))
            elif isinstance(module, neuron_type):
                self.handles.append(module.register_forward_hook(self._neuron_hook(name)))

    def begin(self, labels):
        self.labels = labels.tolist()
        self.batch, self.current_view = {}, -1

    def _base_input(self, module, inputs):
        self.current_view += 1
        assert self.current_view < self.views
        self._record("network_input", "input", inputs[0].detach())

    def _record(self, name, role, x=None, spikes=None, addresses=None, effective=None,
                winners=None, end_mem=None, metadata=None):
        tensor = x if x is not None else spikes
        assert tensor.shape[:2] == (self.steps, len(self.labels))
        b = len(self.labels)
        if name not in self.stats:
            self.stats[name] = dict(role=role, binary_input=True, input_units=math.prod(x.shape[2:]) if x is not None else 0,
                                    neuron_units=math.prod(spikes.shape[2:]) if spikes is not None else 0,
                                    time={}, neuron_fire=None, input_nonzero_by_unit=None, metadata=metadata or {})
        stat = self.stats[name]
        if x is not None:
            stat["binary_input"] &= bool(((x == 0) | (x == 1)).all().item())
            unit_counts = (x != 0).sum((0, 1)).flatten().cpu().numpy().astype(np.int64)
            if stat["input_nonzero_by_unit"] is None:
                stat["input_nonzero_by_unit"] = np.zeros_like(unit_counts)
            stat["input_nonzero_by_unit"] += unit_counts
        if spikes is not None:
            assert bool(((spikes == 0) | (spikes == 1)).all().item())
            counts = spikes.sum((0, 1)).flatten().to(torch.int64).cpu().numpy()
            if stat["neuron_fire"] is None:
                stat["neuron_fire"] = np.zeros_like(counts)
            stat["neuron_fire"] += counts
        values = {}
        for key, value in (("input_nonzero", x != 0 if x is not None else None),
                           ("fire", spikes if spikes is not None else None), ("winner", winners)):
            if value is not None:
                values[key] = value.flatten(2).to(torch.int64).sum(-1).transpose(0, 1).cpu().numpy()
        for key, value in (("addressed_terms", addresses), ("effective_nonzero_weight_terms", effective)):
            if value is not None:
                values[key] = value.to(torch.int64).transpose(0, 1).cpu().numpy()
        if spikes is not None:
            values["neuron_updates"] = np.full((b, self.steps), stat["neuron_units"], np.int64)
        batch = self.batch.setdefault(name, dict(values={}, end_nonzero=np.zeros(b, np.int64)))
        for key, value in values.items():
            if key not in stat["time"]:
                stat["time"][key] = np.zeros((self.samples, self.views, self.steps), np.int64)
            stat["time"][key][self.offset:self.offset+b, self.current_view] = value
            batch["values"][key] = batch["values"].get(key, np.zeros(b, np.int64)) + value.sum(-1)
        if end_mem is not None:
            batch["end_nonzero"] += (end_mem != 0).sum(-1).cpu().numpy()

    def _synapse_hook(self, name):
        def hook(module, inputs, output):
            x = inputs[0].detach()
            nonzero = (x != 0).to(torch.int64)
            if isinstance(module, torch.nn.Linear):
                dense = nonzero.sum(-1) * module.out_features
                degree = (module.weight != 0).sum(0)
                effective = (nonzero * degree).sum(-1)
                meta = dict(kind="Linear", in_features=module.in_features, out_features=module.out_features)
            else:
                cache_key = (name, tuple(x.shape[2:]), tuple(output.shape[2:]))
                if cache_key not in self.degree_cache:
                    ones = torch.ones((1, *output.shape[2:]), device=x.device, dtype=x.dtype)
                    op = tuple(x.shape[-2+i] - ((output.shape[-2+i]-1)*module.stride[i] - 2*module.padding[i]
                               + module.dilation[i]*(module.kernel_size[i]-1) + 1) for i in (0, 1))
                    degrees = [F.conv_transpose2d(ones, weight, stride=module.stride, padding=module.padding,
                               output_padding=op, groups=module.groups, dilation=module.dilation)[0].round().to(torch.int64)
                               for weight in (torch.ones_like(module.weight), (module.weight != 0).to(module.weight))]
                    assert degrees[0].shape == x.shape[2:]
                    self.degree_cache[cache_key] = degrees
                dense_degree, effective_degree = self.degree_cache[cache_key]
                dense = (nonzero * dense_degree).flatten(2).sum(-1)
                effective = (nonzero * effective_degree).flatten(2).sum(-1)
                meta = dict(kind="Conv2d", in_channels=module.in_channels, out_channels=module.out_channels,
                            kernel=list(module.kernel_size), groups=module.groups, padding_excluded=True)
            meta["weight_zero_fraction"] = float((module.weight == 0).float().mean().item())
            self._record(name, "synapse", x=x, addresses=dense, effective=effective, metadata=meta)
        return hook

    def _neuron_hook(self, name):
        def hook(module, inputs, output):
            final = module.v.detach().flatten(1) if isinstance(module.v, torch.Tensor) else None
            self._record(name, "neuron", spikes=output.detach(), end_mem=final, metadata=dict(kind=type(module).__name__,
                         reset="subtract" if module.v_reset is None else "hard", v_reset=module.v_reset,
                         end_membrane_observed=final is not None,
                         neuron_updates_are_not_measured_SA_enables=True))
        return hook

    def _trace(self, name, key, array, units):
        name_key = (name, key)
        if name_key not in self.traces:
            self.traces[name_key] = np.lib.format.open_memmap(self.out / (name.replace('.', '_') + "_" + key + "_packbits.npy"),
                         mode="w+", dtype=np.uint8, shape=(self.samples, self.views, self.steps, (units+7)//8))
        bits = np.packbits(array.transpose(1, 0, 2), axis=-1, bitorder="little")
        self.traces[name_key][self.offset:self.offset+len(self.labels), self.current_view] = bits

    def _hidden_hook(self, name):
        def hook(module, inputs, output):
            x, spikes = inputs[0].detach(), output.detach()
            # Current frozen final conditions keep native forward activity.
            assert module.stdp_input_encoding in ("native", "bernoulli_trace") or module.rule == "bp"
            currents, before, after, winners, threshold = hidden_shadow(module, x, spikes)
            nonzero = (x != 0).to(torch.int64)
            dense = nonzero.sum(-1) * module.out_features
            effective = (nonzero * (module.weight != 0).sum(0)).sum(-1)
            self._record(name, "hidden", x=x, spikes=spikes, addresses=dense, effective=effective,
                         winners=winners, end_mem=after[-1], metadata=dict(kind=type(module).__name__,
                         in_features=module.in_features, out_features=module.out_features,
                         membrane_decay=module.membrane_decay, reset="subtract", shadow_spikes_match=True,
                         end_membrane_observed=True,
                         weight_zero_fraction=float((module.weight == 0).float().mean().item()),
                         competition_size=module.competition_size, winner_scope=module.winner_scope,
                         winner_gates_learning_only=True, membrane_units="algorithmic; not RBL volts"))
            self._trace(name, "output", spikes.bool().cpu().numpy(), module.out_features)
            binary = bool(((x == 0) | (x == 1)).all().item())
            if binary:
                self._trace(name, "input", x.bool().cpu().numpy(), module.in_features)
            candidate = self.candidates.setdefault(name, {})
            totals = nonzero.sum((0, 2)).cpu().numpy()
            cpu = None
            for i, count in enumerate(totals):
                bucket = min(100, int(count / (self.steps * module.in_features) * 100))
                if bucket not in candidate:
                    if cpu is None:
                        cpu = [a.cpu().numpy() for a in (x, spikes, currents, before, after, winners)]
                    candidate[bucket] = dict(sample_index=self.offset+i, view=self.current_view,
                         input_nonzero=int(count), label=self.labels[i], values=[a[:, i].copy() for a in cpu])
        return hook

    def end(self):
        assert self.current_view+1 == self.views
        for name, batch in self.batch.items():
            stat, v = self.stats[name], batch["values"]
            for i, label in enumerate(self.labels):
                row = dict(sample_index=self.offset+i, label=label, layer=name, role=stat["role"],
                           input_slots=stat["input_units"]*self.steps*self.views,
                           neuron_slots=stat["neuron_units"]*self.steps*self.views,
                           logical_boundary_initializations=stat["neuron_units"]*self.views,
                           end_nonzero_membrane=int(batch["end_nonzero"][i]) if stat["metadata"].get("end_membrane_observed") else None)
                row.update({key:int(v.get(key, np.zeros(len(self.labels), np.int64))[i])
                            for key in ("input_nonzero", "addressed_terms", "effective_nonzero_weight_terms",
                                        "fire", "winner", "neuron_updates")})
                row["software_spike_reset"] = row["fire"]
                self.writer.writerow(row)
        self.offset += len(self.labels)
        self.stream.flush()

    def close(self):
        for h in self.handles:
            h.remove()
        self.stream.close()
        for v in self.traces.values():
            v.flush()

    def finalize(self):
        assert self.offset == self.samples
        result, arrays = {}, {}
        slots = self.samples*self.views*self.steps
        for name, stat in self.stats.items():
            unit_in, unit_out = stat["input_units"], stat["neuron_units"]
            counters = {key:int(a.sum()) for key,a in stat["time"].items()}
            summary = dict(role=stat["role"], input_units=unit_in, neuron_units=unit_out,
                           metadata=stat["metadata"], binary_input=stat["binary_input"], counters=counters,
                           input_activity=counters.get("input_nonzero", 0)/(slots*unit_in) if unit_in else None,
                           firing_activity=counters.get("fire", 0)/(slots*unit_out) if unit_out else None,
                           SOP=counters.get("addressed_terms") if stat["binary_input"] and unit_in else None,
                           effective_SOP=counters.get("effective_nonzero_weight_terms") if stat["binary_input"] and unit_in else None,
                           hardware_RST_count=None, measured_energy_pJ=None, measured_power_W=None,
                           per_sample={key:distribution(a.sum((1,2))) for key,a in stat["time"].items()},
                           simultaneous_per_step={key:distribution(a.flatten()) for key,a in stat["time"].items()})
            if unit_out:
                rates = stat["neuron_fire"]/slots
                summary["per_neuron_firing_activity"] = distribution(rates)
                summary["silent_neuron_fraction"] = float((stat["neuron_fire"] == 0).mean())
                arrays[name+"__neuron_fire"] = stat["neuron_fire"]
            if unit_in:
                arrays[name+"__input_nonzero_by_unit"] = stat["input_nonzero_by_unit"]
            for key,a in stat["time"].items():
                arrays[name+"__"+key] = a
            if name in self.candidates:
                per_view = stat["time"]["input_nonzero"].sum(-1).flatten()
                representatives = []
                for q in (0, .5, .95, 1):
                    target = float(np.quantile(per_view, q))
                    chosen = min(self.candidates[name].values(), key=lambda c:abs(c["input_nonzero"]-target))
                    dest = self.out / f"{name.replace('.', '_')}_representative_q{q:g}.npz"
                    np.savez_compressed(dest, **dict(zip(("input", "spikes", "currents", "membrane_before_reset",
                                        "membrane_after_reset", "learning_winners"), chosen["values"])))
                    representatives.append(dict(quantile_target=q, target_input_nonzero=target,
                         sample_index=chosen["sample_index"], view=chosen["view"], actual_input_nonzero=chosen["input_nonzero"],
                         path=dest.name, selection="nearest among deterministic 1-percent activity-bin representatives"))
                summary["representative_traces"] = representatives
                summary["full_trace_format"] = dict(bitorder="little", order="sample, view, step, packed units",
                         output_units=unit_out, input_units=unit_in if stat["binary_input"] else None)
            result[name] = summary
        np.savez_compressed(self.out / "per_timestep_and_neuron_counts.npz", **arrays)
        with (self.out / "layers.json").open("w") as f:
            json.dump(result, f, indent=2, allow_nan=False)
        return result


class ObservedLoader:
    """Preserve the original DataLoader and global RNG consumption exactly."""
    def __init__(self, loader, collector, progress):
        self.loader, self.collector, self.progress = loader, collector, progress

    def __iter__(self):
        for data, labels in self.loader:
            self.collector.begin(labels)
            yield data, labels
            self.collector.end()
            self.progress(self.collector.offset)
