import torch
import torch.nn as nn

from .functional import heaviside, surrogate_rectangle_grad, lif_update, apply_reset


class SpikeFunction(torch.autograd.Function):
    '''
        Forward: spike accumulation
        Backward: surrogate function
    '''
    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: float = 1.0, lens: float = 0.5):
        ctx.save_for_backward(x)
        ctx.threshold = threshold
        ctx.lens = lens
        return heaviside(x, threshold)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        threshold = ctx.threshold
        lens = ctx.lens

        grad = surrogate_rectangle_grad(x, threshold=threshold, lens=lens)
        return grad_output * grad, None, None

class BaseNeuron(nn.Module):
    def __init__(self):
        super().__init__()
        self.mem = None

    def reset_state(self):
        self.mem = None

    def detach_state(self):
        if self.mem is not None:
            self.mem = self.mem.detach()

    def extra_repr(self):
        return ""

class LIFNeuron(BaseNeuron):
    def __init__(
        self,
        tau: float = 0.9,
        threshold: float = 1.0,
        lens: float = 0.5,
        reset_mode: str = "subtract",
        detach_reset: bool = False,
    ):
        super().__init__()
        self.tau = tau
        self.threshold = threshold
        self.lens = lens
        self.reset_mode = reset_mode
        self.detach_reset = detach_reset

    def forward(self, x: torch.Tensor, return_mem: bool = False):
        if x.dim() < 3:
            raise ValueError("Expected input shape [T, B, ...].")

        T = x.shape[0]
        self.mem = torch.zeros_like(x[0]) if self.mem is None else self.mem

        spike_seq = []
        mem_seq = []

        for t in range(T):
            self.mem = lif_update(self.mem, x[t], self.tau)
            spike = SpikeFunction.apply(self.mem, self.threshold, self.lens)

            spike_for_reset = spike.detach() if self.detach_reset else spike
            self.mem = apply_reset(
                self.mem,
                spike_for_reset,
                threshold=self.threshold,
                reset_mode=self.reset_mode,
            )

            spike_seq.append(spike)
            if return_mem:
                mem_seq.append(self.mem.clone())

        spike_seq = torch.stack(spike_seq, dim=0)

        if return_mem:
            mem_seq = torch.stack(mem_seq, dim=0)
            return spike_seq, mem_seq

        return spike_seq