import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from snn import LinearSNN
from visualize import plot_wta_membrane_dynamics, plot_wta_weight_dynamics


def _make_wta_corner_minibatch(T=8):
    """Return [T, B=4, I=8] spikes covering a few WTA corner cases."""
    x = torch.zeros(T, 4, 8)

    # sample 0: early pattern for neuron 0.
    x[0, 0, 0] = 1.0
    x[0, 0, 1] = 1.0

    # sample 1: delayed pattern for neuron 1, showing membrane stays quiet first.
    x[3, 1, 2] = 1.0
    x[3, 1, 3] = 1.0

    # sample 2: no input spike, so no winner and no weight update.

    # sample 3: one input pair drives two neurons over threshold at once.
    # The phase-1 WTA tie break should pick the lower neuron index.
    x[1, 3, 4] = 1.0
    x[1, 3, 5] = 1.0
    return x


def _make_wta_layer():
    layer = LinearSNN(
        in_features=8,
        out_features=4,
        mode="wta",
        bias=False,
        wta_tau=0.0,
        wta_threshold=1.0,
        wta_lr=0.05,
        w_min=0.0,
        w_max=2.0,
    )

    # Structured weights make expected winners and the tie-break case visible.
    layer.weight.data = torch.tensor([
        [0.60, 0.60, 0.05, 0.05, 0.50, 0.50, 0.05, 0.05],
        [0.05, 0.05, 0.60, 0.60, 0.05, 0.05, 0.05, 0.05],
        [0.05, 0.05, 0.05, 0.05, 0.50, 0.50, 0.05, 0.05],
        [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.60, 0.60],
    ])
    return layer


def test_wta_smoke_visual(rounds=6, save_dir="test/figures", show=False):
    print("\n=== test_wta_smoke_visual ===")
    os.makedirs(save_dir, exist_ok=True)

    layer = _make_wta_layer()
    initial_weight = layer.weight.detach().clone()
    x = _make_wta_corner_minibatch()

    weight_segments = []
    last_post_spike = None
    last_info = None

    for step in range(rounds):
        post_spike, info = layer(x, learn=True, return_info=True)
        last_post_spike = post_spike
        last_info = info

        # Avoid duplicating the boundary snapshot when concatenating rounds.
        segment = info["weight_seq"] if step == 0 else info["weight_seq"][1:]
        weight_segments.append(segment)

        print(
            f"step={step} winners={info['winner_index'].tolist()} "
            f"winner_time={info['winner_time'].tolist()} "
            f"updates={info['update_count'].tolist()}"
        )

    assert last_info is not None and last_post_spike is not None
    assert last_info["winner_index"].tolist() == [0, 1, -1, 0]
    assert last_info["winner_time"].tolist() == [0, 3, -1, 1]
    assert last_info["update_count"].tolist() == [2, 2, 0, 2]
    assert last_post_spike.sum().item() == 3.0
    assert torch.allclose(last_info["mem"][0], torch.zeros_like(last_info["mem"][0]))
    assert torch.allclose(last_info["mem"][1], torch.zeros_like(last_info["mem"][1]))
    assert torch.allclose(last_info["mem"][3], torch.zeros_like(last_info["mem"][3]))

    expected = initial_weight.clone()
    expected[0, 0] += rounds * layer.wta_lr
    expected[0, 1] += rounds * layer.wta_lr
    expected[1, 2] += rounds * layer.wta_lr
    expected[1, 3] += rounds * layer.wta_lr
    expected[0, 4] += rounds * layer.wta_lr
    expected[0, 5] += rounds * layer.wta_lr
    assert torch.allclose(layer.weight, expected), (
        "WTA should only update active pre-spike synapses on winner rows."
    )

    weight_history = torch.cat(weight_segments, dim=0)
    plot_wta_membrane_dynamics(
        mem_seq=last_info["mem_seq"],
        spike_seq=last_post_spike,
        threshold=layer.wta_threshold,
        batch_idx=3,
        title="WTA membrane dynamics - tie-break sample",
        save_path=os.path.join(save_dir, "wta_membrane_tie_break.png"),
        show=show,
    )
    plot_wta_membrane_dynamics(
        mem_seq=last_info["mem_seq"],
        spike_seq=last_post_spike,
        threshold=layer.wta_threshold,
        batch_idx=2,
        title="WTA membrane dynamics - no winner sample",
        save_path=os.path.join(save_dir, "wta_membrane_no_winner.png"),
        show=show,
    )
    plot_wta_weight_dynamics(
        weight_history=weight_history,
        title="WTA weight dynamics over repeated minibatch rounds",
        save_path=os.path.join(save_dir, "wta_weight_dynamics.png"),
        show=show,
    )

    print(f"saved figures to {save_dir}")
    print("PASS")


if __name__ == "__main__":
    test_wta_smoke_visual()
