"""Float training with sample-specific winner gating and bounded local STDP."""

import torch
from torch import nn
from torch.nn import functional as F
from spikingjelly.activation_based import surrogate

from .homeostasis import HomeostaticThreshold
from .neuron import SpikeFunction


class _WinnerLinear(torch.autograd.Function):
    """Gate synaptic writes per sample while propagating all input errors."""
    @staticmethod
    def forward(ctx, inputs, weights, bias, eligibility):
        ctx.save_for_backward(inputs, weights)
        ctx.eligibility = eligibility
        return F.linear(inputs, weights, bias)

    @staticmethod
    def backward(ctx, gradient):
        inputs, weights = ctx.saved_tensors
        input_gradient = gradient @ weights if ctx.needs_input_grad[0] else None
        gated = gradient * ctx.eligibility["mask"].unsqueeze(0)
        weight_gradient = (gated.flatten(0, 1).T @ inputs.flatten(0, 1)) if ctx.needs_input_grad[1] else None
        bias_gradient = gated.sum((0, 1)) if ctx.needs_input_grad[2] else None
        return input_gradient, weight_gradient, bias_gradient, None


class WTAHidden(nn.Module):
    """WTA gates learning, not the forward activity of losing neurons.

    One winner per sample/group, either first in the presentation or each cycle; simultaneous crossings
    use configurable index or voltage-margin selection. Threshold adaptation persists. No winner
    means no hidden synaptic update. Classification heads are separate modules.
    """
    def __init__(self, in_features, out_features, rule="bp", homeostasis="exponential",
                 membrane_decay=0.9, weight_max=1.0, stdp_lr=0.01,
                 a_plus=0.01, a_minus=0.012, trace_decay=0.9,
                 surrogate_type="rectangle", initial_row_sum=4.0, initial_fan_in=None, competition_size=None,
                 hybrid_stdp_period=4, winner_tiebreak="index", winner_scope="presentation",
                 hybrid_stdp_granularity="neuron", **threshold_kwargs):
        super().__init__()
        if rule not in {"bp", "stdp", "hybrid"}:
            raise ValueError("rule must be bp, stdp or hybrid")
        if winner_tiebreak not in {"index", "margin"}:
            raise ValueError("winner_tiebreak must be index or margin")
        self.winner_tiebreak = winner_tiebreak
        if winner_scope not in {"presentation", "cycle"}:
            raise ValueError("winner_scope must be presentation or cycle")
        self.winner_scope = winner_scope
        if hybrid_stdp_granularity not in {"neuron", "group"}:
            raise ValueError("hybrid_stdp_granularity must be neuron or group")
        self.hybrid_stdp_granularity = hybrid_stdp_granularity
        if weight_max <= 0 or not 0 <= membrane_decay < 1 or not 0 <= trace_decay < 1:
            raise ValueError("Invalid weight bound or decay factor")
        self.in_features, self.out_features, self.rule = in_features, out_features, rule
        self.competition_size = out_features if competition_size is None else competition_size
        if (isinstance(self.competition_size, bool) or not isinstance(self.competition_size, int)
                or self.competition_size <= 0 or out_features % self.competition_size):
            raise ValueError("competition_size must be a positive divisor of hidden width")
        if rule == "hybrid":
            if (isinstance(hybrid_stdp_period, bool) or not isinstance(hybrid_stdp_period, int)
                    or hybrid_stdp_period < 2
                    or (hybrid_stdp_granularity == "neuron" and self.competition_size % hybrid_stdp_period)
                    or (hybrid_stdp_granularity == "group" and out_features % (self.competition_size * hybrid_stdp_period))):
                raise ValueError("hybrid_stdp_period must be >=2 and divide the selected neuron or group population")
            indices = torch.arange(out_features)
            units = indices // self.competition_size if hybrid_stdp_granularity == "group" else indices
            stdp_mask = units % hybrid_stdp_period == 0
        else:
            stdp_mask = torch.full((out_features,), rule == "stdp", dtype=torch.bool)
        # Derived from immutable rules/config; old pure-rule checkpoints remain loadable.
        self.register_buffer("stdp_neuron_mask", stdp_mask, persistent=False)
        self.register_buffer("bp_neuron_mask", ~stdp_mask, persistent=False)
        self.hybrid_stdp_period = hybrid_stdp_period
        self.membrane_decay, self.weight_max = membrane_decay, weight_max
        self.stdp_lr, self.a_plus, self.a_minus = stdp_lr, a_plus, a_minus
        self.trace_decay = trace_decay
        if surrogate_type not in {"rectangle", "atan"} or initial_row_sum <= 0:
            raise ValueError("Invalid hidden surrogate or initialization scale")
        self.surrogate_type = surrogate_type
        self.atan = surrogate.ATan(alpha=2.0)
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=rule != "stdp")
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=rule != "stdp")
        nn.init.uniform_(self.weight, 0, min(weight_max, 2.0*initial_row_sum / in_features))
        if initial_fan_in is not None:
            count = min(initial_fan_in, in_features)
            if count <= 0:
                raise ValueError("initial_fan_in must be positive")
            with torch.no_grad():
                indices = torch.rand(out_features, in_features).topk(count, dim=1).indices
                values = torch.rand(out_features, count)*min(weight_max, 2.0*initial_row_sum/count)
                self.weight.zero_().scatter_(1, indices, values)
        self.thresholds = HomeostaticThreshold(out_features, mode=homeostasis, **threshold_kwargs)
        self.register_buffer("row_sum_target", self.weight.detach().sum(1))
        self.eligible_rows = None
        self.learning_context = None

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.in_features:
            raise ValueError("Expected [time, batch, in_features]")
        eligibility = {}
        currents = (_WinnerLinear.apply(x, self.weight, self.bias, eligibility) if self.training else
                    F.linear(x, self.weight, self.bias))
        mem = torch.zeros_like(currents[0])
        batch = x.shape[1]
        groups = self.out_features // self.competition_size
        winner = torch.full((batch, groups), -1, dtype=torch.long, device=x.device)
        spike_seq, winning_seq = [], []
        for current in currents:
            threshold = self.thresholds.threshold.detach().clone().to(current)
            mem = self.membrane_decay * mem + current
            margin = (mem - threshold).detach() if self.winner_tiebreak == "margin" else None
            spike = (self.atan(mem-threshold) if self.surrogate_type == "atan" else
                     SpikeFunction.apply(mem, threshold, 0.5))
            mem = mem - spike.detach() * threshold
            candidates = spike.detach().reshape(batch, groups, self.competition_size) > 0
            if self.winner_scope == "presentation":
                candidates = candidates & (winner < 0).unsqueeze(2)
            has_winner = candidates.any(2)
            if self.winner_tiebreak == "margin":
                # Select the largest threshold excess among current crossings.
                # Presentation scope excludes groups that already chose a winner.
                scores = margin.reshape(batch, groups, self.competition_size)
                indices = scores.masked_fill(~candidates, -torch.inf).argmax(2)
            else:
                indices = candidates.to(torch.int64).argmax(2)
            winner[has_winner] = indices[has_winner]
            selected = torch.zeros_like(candidates, dtype=spike.dtype)
            selected.scatter_(2, indices.unsqueeze(2), has_winner.unsqueeze(2).to(spike.dtype))
            selected = selected.reshape_as(spike)
            spike_seq.append(spike)
            winning_seq.append(selected)
            if self.training:
                self.thresholds.advance(selected, spike.detach())
        spikes = torch.stack(spike_seq)
        selected = torch.stack(winning_seq)
        # BP eligibility is the union of winning neurons over the presentation.
        # STDP retains actual event times, including repeated cycle winners.
        mask = selected.sum(0).clamp_max(1)
        self.eligible_rows = mask.bool().any(0)
        if self.training:
            # Only hidden synaptic writes are gated. Dense input gradients keep
            # convolutional BP outside WTA, as required by the experiment.
            eligibility["mask"] = mask * self.bp_neuron_mask.to(mask)
            self.learning_context = (x.detach(), selected.detach()) if self.rule != "bp" else None
        else:
            self.learning_context = None
        return spikes

    @torch.no_grad()
    def apply_stdp(self):
        if self.rule == "bp" or self.learning_context is None:
            return
        pre, post = self.learning_context
        rows = self.eligible_rows & self.stdp_neuron_mask
        if not rows.any():
            self.learning_context = None
            return
        weights = self.weight[rows]
        post = post[:, :, rows]
        pre_trace = torch.zeros_like(pre[0])
        post_trace = torch.zeros_like(post[0])
        dw = torch.zeros_like(weights)
        # Include coincident pre-spikes so first-step winners can learn. Only
        # winner events contribute to the post trace and hence to weight writes.
        for x, y in zip(pre, post):
            pre_trace.mul_(self.trace_decay).add_(x.clamp_min(0))
            if y.any():
                dw.add_((self.weight_max-weights) * (y.T @ pre_trace), alpha=self.a_plus)
            if post_trace.any():
                dw.sub_(weights * (post_trace.T @ x.clamp_min(0)), alpha=self.a_minus)
            post_trace.mul_(self.trace_decay).add_(y)
        dw.mul_(self.stdp_lr / pre.shape[1])
        updated = (weights + dw).clamp(0, self.weight_max)
        # Competition plus soft bounds and winner-only normalization prevents
        # uniform potentiation toward the upper bound; losers remain untouched.
        sums = updated.sum(1).clamp_min(1e-12)
        updated *= (self.row_sum_target[rows] / sums).unsqueeze(1)
        self.weight[rows] = updated.clamp(0, self.weight_max)
        self.learning_context = None

    @torch.no_grad()
    def reset_row_sum_target(self):
        self.row_sum_target.copy_(self.weight.sum(1))


class WinnerOptimizerGuard:
    """Preserve losing BP rows, including optimizer momentum and weight decay."""
    def __init__(self, layers, optimizer):
        self.layers = [layer for layer in layers if layer.rule != "stdp"]
        self.optimizer, self.snapshots = optimizer, []

    @torch.no_grad()
    def __enter__(self):
        for layer in self.layers:
            if layer.eligible_rows is None:
                raise RuntimeError("Run a forward pass before stepping the optimizer")
            # STDP-designated rows are protected from BP optimizer writes/momentum.
            losers = ~(layer.eligible_rows & layer.bp_neuron_mask)
            for param in (layer.weight, layer.bias):
                if param.grad is not None:
                    param.grad[losers] = 0
                state = {}
                for key, value in self.optimizer.state.get(param, {}).items():
                    if isinstance(value, torch.Tensor) and value.shape == param.shape:
                        state[key] = value[losers].clone()
                self.snapshots.append((param, losers, param[losers].clone(), state))
        return self

    @torch.no_grad()
    def __exit__(self, *exc):
        for param, losers, original, state in self.snapshots:
            param[losers] = original
            for key, value in self.optimizer.state.get(param, {}).items():
                if isinstance(value, torch.Tensor) and value.shape == param.shape:
                    value[losers] = state.get(key, torch.zeros_like(value[losers]))
        self.snapshots.clear()
