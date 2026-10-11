import torch
import torch.nn as nn


class BaseEncoder(nn.Module):
    """
    Base class for static-to-temporal encoders.

    Input:
        static tensor: [B, ...]
    Output:
        temporal tensor: [T, B, ...]
    """
    def __init__(self, T: int):
        super().__init__()
        if T <= 0:
            raise ValueError(f"T must be positive, but got {T}.")
        self.T = T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f"T={self.T}"


class RepeatEncoder(BaseEncoder):
    """
    Repeat static input for T time steps.

    Example:
        [B, C, H, W] -> [T, B, C, H, W]
        [B, N]       -> [T, B, N]
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            raise ValueError("Expected input shape [B, ...].")

        # [1, B, ...] -> [T, B, ...]
        repeat_shape = [self.T] + [1] * x.dim()
        return x.unsqueeze(0).repeat(*repeat_shape)


class PoissonEncoder(BaseEncoder):
    """
    Bernoulli/Poisson-like rate encoding for normalized static input.

    Each time step samples spikes independently:
        s_t ~ Bernoulli(x)

    Assumes x is in [0, 1].
    Output is float tensor with values in {0, 1}.
    """
    def __init__(self, T: int, clip: bool = True):
        super().__init__(T)
        self.clip = clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            raise ValueError("Expected input shape [B, ...].")

        if self.clip:
            x = x.clamp(0.0, 1.0)
        else:
            if torch.any(x < 0) or torch.any(x > 1):
                raise ValueError("PoissonEncoder expects input in [0, 1].")

        # Expand to [T, B, ...]
        x_seq = x.unsqueeze(0).expand(self.T, *x.shape)

        # Sample Bernoulli spikes
        rand = torch.rand_like(x_seq)
        spikes = (rand <= x_seq).to(x.dtype)

        return spikes

    def extra_repr(self) -> str:
        return f"T={self.T}, clip={self.clip}"


class LatencyEncoder(BaseEncoder):
    """
    Single-spike latency encoding.

    Larger input value -> earlier spike time.
    Smaller input value -> later spike time.
    Each neuron/pixel fires at most once.

    Assumes x is in [0, 1].

    If x == 0:
        - no spike when allow_zero=False
        - last-time-step spike when allow_zero=True
    """
    def __init__(
        self,
        T: int,
        clip: bool = True,
        allow_zero: bool = False,
    ):
        super().__init__(T)
        self.clip = clip
        self.allow_zero = allow_zero

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            raise ValueError("Expected input shape [B, ...].")

        if self.clip:
            x = x.clamp(0.0, 1.0)
        else:
            if torch.any(x < 0) or torch.any(x > 1):
                raise ValueError("LatencyEncoder expects input in [0, 1].")

        # Map intensity to firing time:
        # x=1   -> t=0
        # x~0   -> t=T-1 or no spike
        #
        # firing_time shape: [B, ...], integer in [0, T-1]
        firing_time = ((1.0 - x) * (self.T - 1)).round().long()

        # Build time index tensor: [T, 1, 1, ...]
        time_shape = [self.T] + [1] * x.dim()
        t_idx = torch.arange(self.T, device=x.device).view(*time_shape)

        # Compare against firing time -> [T, B, ...]
        spikes = (t_idx == firing_time.unsqueeze(0)).to(x.dtype)

        if not self.allow_zero:
            # Zero-valued inputs produce no spike
            spikes = spikes * (x.unsqueeze(0) > 0).to(x.dtype)

        return spikes

    def extra_repr(self) -> str:
        return f"T={self.T}, clip={self.clip}, allow_zero={self.allow_zero}"