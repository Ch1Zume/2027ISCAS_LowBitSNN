import os
from typing import Optional

import torch
import matplotlib.pyplot as plt


def _to_1d_numpy(x: torch.Tensor):
    return x.detach().cpu().flatten().numpy()


def plot_membrane_and_spike(
    mem: torch.Tensor,
    spike: torch.Tensor,
    threshold: Optional[float] = None,
    title: str = "Neuron Trace",
    figsize=(10, 4),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot membrane trace and spike events for a single neuron.

    Args:
        mem:   [T] or tensor that can be flattened to [T]
        spike: [T] or tensor that can be flattened to [T]
        threshold: optional threshold line
        title: plot title
        figsize: figure size
        save_path: if not None, save figure to this path
        show: whether to call plt.show()
    """
    mem_np = _to_1d_numpy(mem)
    spike_np = _to_1d_numpy(spike)

    if mem_np.shape != spike_np.shape:
        raise ValueError(
            f"mem and spike must have the same flattened shape, "
            f"but got {mem_np.shape} and {spike_np.shape}"
        )

    T = len(mem_np)
    t = list(range(T))

    plt.figure(figsize=figsize)
    plt.plot(t, mem_np, label="membrane", linewidth=2)

    if threshold is not None:
        plt.axhline(
            y=threshold,
            linestyle="--",
            linewidth=1.5,
            label=f"threshold={threshold}"
        )

    spike_times = [i for i, s in enumerate(spike_np) if s > 0.5]
    spike_values = [mem_np[i] for i in spike_times]

    if len(spike_times) > 0:
        plt.scatter(
            spike_times,
            spike_values,
            marker="o",
            s=60,
            label="spike event"
        )

    plt.xlabel("time step")
    plt.ylabel("value")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_spike_raster(
    spike_seq: torch.Tensor,
    title: str = "Spike Raster",
    figsize=(10, 4),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot spike raster.

    Supported input shapes:
        [T, N]
        [T, B, N]
        [T, B, C, H, W]  -> flattened to [T, num_neurons]

    Notes:
        - If input is [T, B, N], batch and neuron dims are flattened together.
        - This is mainly for quick qualitative debugging.
    """
    if spike_seq.dim() < 2:
        raise ValueError("spike_seq must have at least 2 dims, e.g. [T, N]")

    T = spike_seq.shape[0]
    spike_flat = spike_seq.detach().cpu().reshape(T, -1)  # [T, num_units]
    num_units = spike_flat.shape[1]

    spike_indices_t = []
    spike_indices_n = []

    for t in range(T):
        fired = torch.nonzero(spike_flat[t] > 0.5, as_tuple=False).flatten()
        for idx in fired.tolist():
            spike_indices_t.append(t)
            spike_indices_n.append(idx)

    plt.figure(figsize=figsize)
    plt.scatter(spike_indices_t, spike_indices_n, s=8, marker="|")

    plt.xlabel("time step")
    plt.ylabel("neuron index")
    plt.title(f"{title} (units={num_units})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_firing_rate_over_time(
    spike_seq: torch.Tensor,
    title: str = "Firing Rate Over Time",
    figsize=(10, 4),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot mean firing rate at each time step.

    Args:
        spike_seq: [T, ...]
    """
    if spike_seq.dim() < 2:
        raise ValueError("spike_seq must have at least 2 dims, e.g. [T, N]")

    spike_flat = spike_seq.detach().cpu().reshape(spike_seq.shape[0], -1)  # [T, num_units]
    firing_rate = spike_flat.float().mean(dim=1).numpy()

    plt.figure(figsize=figsize)
    plt.plot(range(len(firing_rate)), firing_rate, linewidth=2)

    plt.xlabel("time step")
    plt.ylabel("mean firing rate")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

def _stack_history_list(history_list):
    """
    history_list: list of tensors, each tensor shape [B, N] or [O, I]
    return stacked tensor on dim 0
    """
    if len(history_list) == 0:
        raise ValueError("Empty history list.")
    return torch.stack(history_list, dim=0)


def plot_stdp_dynamics(
    history: dict,
    batch_idx: int = 0,
    in_idx: int = 0,
    out_idx: int = 0,
    title: str = "STDP Dynamics",
    figsize=(10, 10),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Visualize a single synapse path:
        input neuron index = in_idx
        output neuron index = out_idx
        batch sample index = batch_idx

    Plots:
        1) in_spike(t)
        2) out_spike(t)
        3) pre_trace(t)
        4) post_trace(t)
        5) weight(t)
    """

    pre_spike = _stack_history_list(history["pre_spike"])      # [T, B, I]
    post_spike = _stack_history_list(history["post_spike"])    # [T, B, O]
    pre_trace = _stack_history_list(history["pre_trace"])      # [T, B, I]
    post_trace = _stack_history_list(history["post_trace"])    # [T, B, O]
    weight = _stack_history_list(history["weight"])            # [T, O, I]

    T = pre_spike.shape[0]
    t = list(range(T))

    in_spk = pre_spike[:, batch_idx, in_idx].numpy()
    out_spk = post_spike[:, batch_idx, out_idx].numpy()
    pre_tr = pre_trace[:, batch_idx, in_idx].numpy()
    post_tr = post_trace[:, batch_idx, out_idx].numpy()
    w = weight[:, out_idx, in_idx].numpy()

    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]

    fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True)

    # 1. in spike
    spike_times = [i for i, s in enumerate(in_spk) if s > 0.5]
    axes[0].scatter(spike_times, [0.5] * len(spike_times),
                    marker="|", s=200, color=colors[0])
    axes[0].set_ylabel("in_spike", color=colors[0])
    axes[0].set_ylim(0, 1)
    axes[0].set_yticks([])
    axes[0].grid(True, axis="x", alpha=0.3)

    # 2. pre trace (between in_spike and out_spike)
    axes[1].plot(t, pre_tr, linewidth=2, color=colors[1])
    axes[1].set_ylabel("trace_pre", color=colors[1])
    axes[1].tick_params(axis="y", labelcolor=colors[1])
    axes[1].grid(True, alpha=0.3)

    # 3. out spike
    spike_times = [i for i, s in enumerate(out_spk) if s > 0.5]
    axes[2].scatter(spike_times, [0.5] * len(spike_times),
                    marker="|", s=200, color=colors[2])
    axes[2].set_ylabel("out_spike", color=colors[2])
    axes[2].set_ylim(0, 1)
    axes[2].set_yticks([])
    axes[2].grid(True, axis="x", alpha=0.3)

    # 4. post trace
    axes[3].plot(t, post_tr, linewidth=2, color=colors[3])
    axes[3].set_ylabel("trace_post", color=colors[3])
    axes[3].tick_params(axis="y", labelcolor=colors[3])
    axes[3].grid(True, alpha=0.3)

    # 5. weight (auto-scaled y-axis with margin)
    w_min_val, w_max_val = w.min(), w.max()
    w_range = w_max_val - w_min_val
    margin = max(w_range * 0.3, 1e-4)
    axes[4].set_ylim(w_min_val - margin, w_max_val + margin)
    axes[4].plot(t, w, linewidth=2, color=colors[4])
    axes[4].set_ylabel("weight", color=colors[4])
    axes[4].set_xlabel("time step")
    axes[4].tick_params(axis="y", labelcolor=colors[4])
    axes[4].grid(True, alpha=0.3)

    fig.suptitle(
        f"{title}\n(batch={batch_idx}, in_idx={in_idx}, out_idx={out_idx})"
    )
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_wta_membrane_dynamics(
    mem_seq: torch.Tensor,
    spike_seq: torch.Tensor,
    threshold: Optional[float] = None,
    batch_idx: int = 0,
    title: str = "WTA Membrane Dynamics",
    figsize=(12, 5),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot all output-neuron membrane traces for one batch sample.

    Args:
        mem_seq: [T, B, O] membrane values recorded before WTA reset
        spike_seq: [T, B, O] output spikes
    """
    if mem_seq.dim() != 3 or spike_seq.dim() != 3:
        raise ValueError("mem_seq and spike_seq must have shape [T, B, O].")
    if mem_seq.shape != spike_seq.shape:
        raise ValueError("mem_seq and spike_seq must have the same shape.")
    if batch_idx < 0 or batch_idx >= mem_seq.shape[1]:
        raise ValueError(f"Invalid batch_idx={batch_idx} for B={mem_seq.shape[1]}.")

    mem = mem_seq.detach().cpu()[:, batch_idx]
    spikes = spike_seq.detach().cpu()[:, batch_idx]
    T, O = mem.shape
    t = list(range(T))

    plt.figure(figsize=figsize)
    for out_idx in range(O):
        plt.plot(t, mem[:, out_idx].numpy(), linewidth=2, label=f"neuron {out_idx}")
        spike_times = torch.nonzero(spikes[:, out_idx] > 0.5, as_tuple=False).flatten().tolist()
        if spike_times:
            spike_values = mem[spike_times, out_idx].numpy()
            plt.scatter(spike_times, spike_values, s=55, marker="o")

    if threshold is not None:
        plt.axhline(threshold, linestyle="--", linewidth=1.5, color="black", label="threshold")

    plt.xlabel("time step")
    plt.ylabel("membrane value")
    plt.title(f"{title} (batch={batch_idx})")
    plt.legend(ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_wta_weight_dynamics(
    weight_history: torch.Tensor,
    title: str = "WTA Weight Dynamics",
    figsize=(12, 6),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot every synaptic weight over global time.

    Args:
        weight_history: [S, O, I], where S can be training rounds or flattened
            timestep snapshots. Each line is one synapse w[out_idx, in_idx].
    """
    if weight_history.dim() != 3:
        raise ValueError("weight_history must have shape [S, O, I].")

    weights = weight_history.detach().cpu()
    S, O, I = weights.shape
    x = list(range(S))

    plt.figure(figsize=figsize)
    for out_idx in range(O):
        for in_idx in range(I):
            linestyle = "-" if out_idx % 2 == 0 else "--"
            plt.plot(
                x,
                weights[:, out_idx, in_idx].numpy(),
                linewidth=1.7,
                linestyle=linestyle,
                label=f"w[{out_idx},{in_idx}]",
            )

    plt.xlabel("global time step")
    plt.ylabel("weight")
    plt.title(f"{title} (synapses={O * I})")
    plt.legend(ncol=max(1, min(I, 4)), fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

