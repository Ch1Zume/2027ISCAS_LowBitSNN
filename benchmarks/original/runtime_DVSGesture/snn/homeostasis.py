"""Persistent, per-neuron thresholds; membrane resets do not reset adaptation."""

import math

import torch
from torch import nn


class HomeostaticThreshold(nn.Module):
    def __init__(self, neurons, mode="exponential", baseline=1.0, delta=0.05,
                 tau=20.0, eta=0.01, target_rate=0.1, rate_tau=20.0,
                 minimum=0.1, maximum=10.0):
        super().__init__()
        if mode not in {"off", "exponential", "rate"}:
            raise ValueError(f"Unknown homeostasis mode: {mode}")
        if not minimum <= baseline <= maximum or min(tau, rate_tau) <= 0:
            raise ValueError("Invalid threshold bounds or time constants")
        if delta < 0 or eta < 0 or not 0 <= target_rate <= 1:
            raise ValueError("Invalid homeostasis gain or target rate")
        self.mode, self.baseline = mode, baseline
        self.delta, self.tau, self.eta = delta, tau, eta
        self.target_rate, self.rate_tau = target_rate, rate_tau
        self.minimum, self.maximum = minimum, maximum
        self.register_buffer("offset", torch.zeros(neurons))
        self.register_buffer("rate", torch.full((neurons,), target_rate))

    @property
    def threshold(self):
        return (self.baseline + self.offset).clamp(self.minimum, self.maximum)

    @torch.no_grad()
    def advance(self, winners, spikes, dt=1.0):
        """One simulation step. Each win adds delta; rates average over the batch."""
        if dt <= 0:
            raise ValueError("dt must be positive")
        if winners.shape != spikes.shape or spikes.ndim != 2:
            raise ValueError("Expected matching [batch, neurons] activity")
        if self.mode == "exponential":
            self.offset.mul_(math.exp(-dt / self.tau))
            self.offset.add_(self.delta * winners.sum(0).to(self.offset))
        elif self.mode == "rate":
            decay = math.exp(-dt / self.rate_tau)
            self.rate.mul_(decay).add_(spikes.float().mean(0).to(self.rate), alpha=1-decay)
            self.offset.add_(self.eta * dt * (self.rate - self.target_rate))
        self.offset.clamp_(self.minimum-self.baseline, self.maximum-self.baseline)

    @torch.no_grad()
    def reset_adaptation(self):
        self.offset.zero_()
        self.rate.fill_(self.target_rate)
