import torch
import torch.nn as nn
import torch.nn.functional as F

from .functional import merge_time_batch, split_time_batch


class Conv2dSNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [T, B, C, H, W]
        """
        if x.dim() != 5:
            raise ValueError("Expected input shape [T, B, C, H, W].")

        x_merged, T, B = merge_time_batch(x)
        y = self.conv(x_merged)
        y = split_time_batch(y, T, B)
        return y

# class LinearSNN(nn.Module):
#     def __init__(self, in_features: int, out_features: int, bias: bool = True):
#         super().__init__()
#         self.linear = nn.Linear(in_features, out_features, bias=bias)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: [T, B, N]
#         """
#         if x.dim() != 3:
#             raise ValueError("Expected input shape [T, B, N].")

#         x_merged, T, B = merge_time_batch(x)
#         y = self.linear(x_merged)
#         y = split_time_batch(y, T, B)
#         return y

class FlattenSNN(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [T, B, ...]
        return: [T, B, N]
        """
        if x.dim() < 3:
            raise ValueError("Expected input shape [T, B, ...].")
        T, B = x.shape[:2]
        return x.reshape(T, B, -1)

class AvgPool2dSNN(nn.Module):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
    ):
        super().__init__()

        self.pool = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [T, B, C, H, W]
        """

        if x.dim() != 5:
            raise ValueError("Expected input shape [T, B, C, H, W]")

        x, T, B = merge_time_batch(x)

        y = self.pool(x)

        y = split_time_batch(y, T, B)

        return y

class AdaptiveAvgPool2dSNN(nn.Module):
    def __init__(self, output_size):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x: torch.Tensor):

        if x.dim() != 5:
            raise ValueError("Expected input shape [T,B,C,H,W]")

        x, T, B = merge_time_batch(x)

        y = self.pool(x)

        y = split_time_batch(y, T, B)

        return y

class MaxPool2dSNN(nn.Module):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        ceil_mode=False,
    ):
        super().__init__()

        self.pool = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [T, B, C, H, W]
        """

        if x.dim() != 5:
            raise ValueError("Expected input shape [T, B, C, H, W]")

        x, T, B = merge_time_batch(x)

        y = self.pool(x)

        y = split_time_batch(y, T, B)

        return y

class BatchNorm2dSNN(nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super().__init__()

        self.bn = nn.BatchNorm2d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [T, B, C, H, W]
        """

        if x.dim() != 5:
            raise ValueError("Expected input shape [T, B, C, H, W]")

        x, T, B = merge_time_batch(x)

        y = self.bn(x)

        y = split_time_batch(y, T, B)

        return y

class BatchNorm1dSNN(nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super().__init__()

        self.bn = nn.BatchNorm1d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [T, B, N]
        """

        if x.dim() != 3:
            raise ValueError("Expected input shape [T, B, N]")

        x, T, B = merge_time_batch(x)

        y = self.bn(x)

        y = split_time_batch(y, T, B)

        return y

class DropoutSNN(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor):

        if x.dim() < 3:
            raise ValueError("Expected input shape [T,B,...]")

        x, T, B = merge_time_batch(x)

        y = self.dropout(x)

        y = split_time_batch(y, T, B)

        return y

class LinearSNN(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode: str = "linear",

        # WTA-related. Used only when mode="wta".
        wta_tau: float = 0.9,
        wta_threshold: float = 1.0,
        wta_lr: float = 1e-2,

        # STDP-related
        stdp_enabled: bool = False,
        tau_pre: float = 0.9,
        tau_post: float = 0.9,
        A_plus: float = 1e-3,
        A_minus: float = 1e-3,
        w_min: float = 0.0,
        w_max: float = 1.0,
        hard_clamp: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        if mode not in {"linear", "wta"}:
            raise ValueError(f"Unsupported LinearSNN mode: {mode}")
        self.mode = mode

        # WTA mode keeps learning local: one winner row and recorded pre-spikes only.
        self.wta_tau = wta_tau
        self.wta_threshold = wta_threshold
        self.wta_lr = wta_lr

        # whether this layer supports / uses STDP update
        self.stdp_enabled = stdp_enabled

        # STDP hyperparameters
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.w_min = w_min
        self.w_max = w_max
        self.hard_clamp = hard_clamp

        # parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # traces / state
        self.register_buffer("pre_trace", None)
        self.register_buffer("post_trace", None)
        self.register_buffer("mem", None)

        # record
        self.record = False
        self.history = {
            "pre_spike": [],
            "post_spike": [],
            "pre_trace": [],
            "post_trace": [],
            "weight": [],
            "dW": [],
        }

        self.register_buffer("_weight_norm_target", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.stdp_enabled:
            fan_in = self.in_features
            target_row_sum = 1.0
            bound = 2.0 * target_row_sum / fan_in
            nn.init.uniform_(self.weight, a=0.0, b=bound)
            self._weight_norm_target = self.weight.data.sum(dim=1).mean().unsqueeze(0)
        else:
            nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @torch.no_grad()
    def normalize_weights(self):
        if self._weight_norm_target is None:
            return
        self.weight.data.clamp_(min=0.0)
        row_sums = self.weight.data.sum(dim=1, keepdim=True).clamp(min=1e-8)
        self.weight.data *= self._weight_norm_target / row_sums

    def set_stdp_enabled(self, enabled: bool):
        self.stdp_enabled = enabled

    def reset_state(self):
        self.pre_trace = None
        self.post_trace = None
        self.mem = None

    def detach_state(self):
        if self.pre_trace is not None:
            self.pre_trace = self.pre_trace.detach()
        if self.post_trace is not None:
            self.post_trace = self.post_trace.detach()
        if self.mem is not None:
            self.mem = self.mem.detach()

    def clear_history(self):
        for k in self.history:
            self.history[k] = []

    def enable_recording(self):
        self.record = True
        self.clear_history()

    def disable_recording(self):
        self.record = False

    def get_history(self):
        return self.history

    def _init_trace(self, batch_size: int, device, dtype):
        if self.pre_trace is None:
            self.pre_trace = torch.zeros(
                batch_size, self.in_features, device=device, dtype=dtype
            )
        if self.post_trace is None:
            self.post_trace = torch.zeros(
                batch_size, self.out_features, device=device, dtype=dtype
            )

    def _init_wta_state(self, batch_size: int, device, dtype):
        self.mem = torch.zeros(batch_size, self.out_features, device=device, dtype=dtype)
        # Binary trace for WTA phase 1: record whether each pre-neuron fired this round.
        self.pre_trace = torch.zeros(batch_size, self.in_features, device=device, dtype=dtype)

    # =========================================================
    # forward
    # =========================================================
    def forward_single_step(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("Expected input shape [B, N].")
        return F.linear(x, self.weight, self.bias)

    def forward(self, x: torch.Tensor, learn: bool = True, return_info: bool = False):
        """
        x:
            [B, N] or [T, B, N] for mode="linear"
            [T, B, N] for mode="wta"
        """
        if self.mode == "wta":
            return self._forward_wta(x, learn=learn, return_info=return_info)

        if x.dim() == 2:
            return self.forward_single_step(x)

        if x.dim() != 3:
            raise ValueError("Expected input shape [B, N] or [T, B, N].")

        T = x.shape[0]
        out_seq = []
        for t in range(T):
            out_seq.append(self.forward_single_step(x[t]))
        return torch.stack(out_seq, dim=0)

    @torch.no_grad()
    def _forward_wta(self, pre_spike: torch.Tensor, learn: bool = True, return_info: bool = False):
        if pre_spike.dim() != 3:
            raise ValueError('WTA mode expects pre_spike shape [T, B, in_features].')
        if pre_spike.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dim {self.in_features}, got {pre_spike.shape[-1]}."
            )

        T, B, _ = pre_spike.shape
        self._init_wta_state(B, pre_spike.device, pre_spike.dtype)

        post_spike = torch.zeros(T, B, self.out_features, device=pre_spike.device, dtype=pre_spike.dtype)
        mem_seq = torch.zeros(T, B, self.out_features, device=pre_spike.device, dtype=pre_spike.dtype)
        weight_seq = torch.zeros(T + 1, self.out_features, self.in_features, device=pre_spike.device, dtype=self.weight.dtype)
        weight_seq[0] = self.weight.data
        winner_index = torch.full((B,), -1, device=pre_spike.device, dtype=torch.long)
        winner_time = torch.full((B,), -1, device=pre_spike.device, dtype=torch.long)
        update_count = torch.zeros(B, device=pre_spike.device, dtype=torch.long)
        round_done = torch.zeros(B, device=pre_spike.device, dtype=torch.bool)

        for t in range(T):
            active = ~round_done
            if not active.any():
                break

            x_t = pre_spike[t]
            # Record only pre activity seen before the first winner of this round.
            self.pre_trace[active] = torch.maximum(
                self.pre_trace[active], (x_t[active] > 0).to(pre_spike.dtype)
            )

            current = F.linear(x_t, self.weight, self.bias)
            self.mem[active] = self.wta_tau * self.mem[active] + current[active]

            # Record membrane before winner reset so threshold crossings are visible.
            mem_seq[t] = self.mem

            spike = (self.mem >= self.wta_threshold) & active.unsqueeze(1)
            has_winner = spike.any(dim=1)
            if not has_winner.any():
                weight_seq[t + 1] = self.weight.data
                continue

            batch_ids = torch.nonzero(has_winner, as_tuple=False).flatten()
            for b in batch_ids.tolist():
                # A deterministic priority-encoder tie break for phase 1.
                winner = int(torch.nonzero(spike[b], as_tuple=False).flatten()[0].item())
                winner_index[b] = winner
                winner_time[b] = t
                post_spike[t, b, winner] = 1.0

                if learn:
                    # WTA only selects the eligible post neuron; the actual local
                    # weight write is kept in the STDP update path below.
                    update_count[b] = self.stdp_update_wta_single(
                        winner_index=winner,
                        pre_trace=self.pre_trace[b],
                    )
                else:
                    update_count[b] = int((self.pre_trace[b] > 0).sum().item())

                # Winner ends this sample's round: reset all output membranes.
                self.mem[b].zero_()
                round_done[b] = True

            weight_seq[t + 1] = self.weight.data

        if return_info:
            return post_spike, {
                "winner_index": winner_index.detach().cpu(),
                "winner_time": winner_time.detach().cpu(),
                "update_count": update_count.detach().cpu(),
                "pre_trace": self.pre_trace.detach().cpu().clone(),
                "mem": self.mem.detach().cpu().clone(),
                "mem_seq": mem_seq.detach().cpu(),
                "weight_seq": weight_seq.detach().cpu(),
            }
        return post_spike

    @torch.no_grad()
    def stdp_update_wta_single(
        self,
        winner_index: int,
        pre_trace: torch.Tensor,
        scale: float = 1.0,
    ) -> int:
        """
        WTA-gated STDP update for phase 1.

        The WTA forward pass decides which post neuron is allowed to learn. This
        method then performs the local STDP-style write: constant potentiation on
        the winner row, limited to pre-synaptic addresses recorded in pre_trace.
        """
        if pre_trace.dim() != 1 or pre_trace.numel() != self.in_features:
            raise ValueError("Expected pre_trace shape [in_features].")
        if winner_index < 0 or winner_index >= self.out_features:
            raise ValueError(f"Invalid winner_index: {winner_index}")

        active_pre = pre_trace > 0
        if not active_pre.any():
            return 0

        # Local write: only winner row at recorded pre-spike addresses.
        pre_ids = torch.nonzero(active_pre, as_tuple=False).flatten()
        updated = self.weight.data[winner_index, pre_ids] + self.wta_lr * scale
        self.weight.data[winner_index, pre_ids] = updated.clamp(self.w_min, self.w_max)
        return int(pre_ids.numel())

    # =========================================================
    # STDP update
    # =========================================================
    @torch.no_grad()
    def compute_stdp_update_single_step(
        self,
        pre_spike: torch.Tensor,
        post_spike: torch.Tensor,
        reduction: str = "mean",
    ):
        if not self.stdp_enabled:
            raise RuntimeError(
                "STDP is disabled for this layer. "
                "Set stdp_enabled=True if you want to use stdp_update."
            )

        if pre_spike.dim() != 2 or post_spike.dim() != 2:
            raise ValueError("Expected pre/post spike shape [B, N].")

        B = pre_spike.shape[0]
        self._init_trace(B, pre_spike.device, pre_spike.dtype)

        pre_trace_old = self.pre_trace.clone()
        post_trace_old = self.post_trace.clone()

        dW_plus = torch.einsum("bo,bi->boi", post_spike, pre_trace_old)
        dW_minus = torch.einsum("bo,bi->boi", post_trace_old, pre_spike)

        if reduction == "mean":
            dW_plus = dW_plus.mean(dim=0)
            dW_minus = dW_minus.mean(dim=0)
        elif reduction == "sum":
            dW_plus = dW_plus.sum(dim=0)
            dW_minus = dW_minus.sum(dim=0)
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")

        dW = self.A_plus * dW_plus - self.A_minus * dW_minus

        self.pre_trace = self.tau_pre * self.pre_trace + pre_spike
        self.post_trace = self.tau_post * self.post_trace + post_spike

        return dW

    @torch.no_grad()
    def stdp_update_single_step(
        self,
        pre_spike: torch.Tensor,
        post_spike: torch.Tensor,
        reduction: str = "mean",
        on_grad: bool = False,
        scale: float = 1.0,
    ):
        dW = self.compute_stdp_update_single_step(
            pre_spike=pre_spike,
            post_spike=post_spike,
            reduction=reduction,
        )

        if on_grad:
            if self.weight.grad is None:
                self.weight.grad = torch.zeros_like(self.weight.data)
            self.weight.grad.add_(-dW * scale)
        else:
            self.weight.data.add_(dW * scale)
            if self.hard_clamp:
                self.weight.data.clamp_(self.w_min, self.w_max)

        if self.record:
            self.history["pre_spike"].append(pre_spike.detach().cpu().clone())
            self.history["post_spike"].append(post_spike.detach().cpu().clone())
            self.history["pre_trace"].append(self.pre_trace.detach().cpu().clone())
            self.history["post_trace"].append(self.post_trace.detach().cpu().clone())
            self.history["weight"].append(self.weight.data.detach().cpu().clone())
            self.history["dW"].append((dW * scale).detach().cpu().clone())

    @torch.no_grad()
    def stdp_update(
        self,
        pre_spike: torch.Tensor,
        post_spike: torch.Tensor,
        reduction: str = "mean",
        on_grad: bool = False,
        scale: float = 1.0,
    ):
        if not self.stdp_enabled:
            raise RuntimeError(
                "STDP is disabled for this layer. "
                "Set stdp_enabled=True before calling stdp_update."
            )

        if pre_spike.dim() == 2 and post_spike.dim() == 2:
            self.stdp_update_single_step(
                pre_spike,
                post_spike,
                reduction=reduction,
                on_grad=on_grad,
                scale=scale,
            )
            return

        if pre_spike.dim() != 3 or post_spike.dim() != 3:
            raise ValueError("Expected pre/post spike shape [B, N] or [T, B, N].")

        if pre_spike.shape[0] != post_spike.shape[0]:
            raise ValueError("pre/post must have the same T.")

        T = pre_spike.shape[0]
        for t in range(T):
            self.stdp_update_single_step(
                pre_spike[t],
                post_spike[t],
                reduction=reduction,
                on_grad=on_grad,
                scale=scale,
            )
