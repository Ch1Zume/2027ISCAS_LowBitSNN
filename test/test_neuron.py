import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import torch

from snn import LIFNeuron
from visualize import (
    plot_membrane_and_spike,
    plot_spike_raster,
    plot_firing_rate_over_time,
)


def print_tensor_1d(name: str, x: torch.Tensor):
    x = x.detach().cpu().flatten()
    print(f"{name}: {[round(v.item(), 4) for v in x]}")


def test_constant_input_response(
    visualize: bool = True,
    save_dir: str = "tests/figures",
):
    print("\n=== test_constant_input_response ===")

    T = 10
    B = 1

    # constant input current
    x = torch.full((T, B, 1), 0.3)

    neuron = LIFNeuron(
        tau=0.9,
        threshold=1.0,
        lens=0.5,
        reset_mode="subtract"
    )

    spike, mem = neuron(x, return_mem=True)

    print_tensor_1d("input", x[:, 0, 0])
    print_tensor_1d("mem", mem[:, 0, 0])
    print_tensor_1d("spike", spike[:, 0, 0])

    assert spike.shape == x.shape
    assert mem.shape == x.shape
    assert torch.all((spike == 0) | (spike == 1)), "Spike should be binary."

    if visualize:
        os.makedirs(save_dir, exist_ok=True)

        plot_membrane_and_spike(
            mem=mem[:, 0, 0],
            spike=spike[:, 0, 0],
            threshold=neuron.threshold,
            title="Constant Input Response (Single Neuron)",
            save_path=os.path.join(save_dir, "constant_input_trace.png"),
            show=True,
        )

        plot_spike_raster(
            spike_seq=spike,
            title="Constant Input Spike Raster",
            save_path=os.path.join(save_dir, "constant_input_raster.png"),
            show=True,
        )

        plot_firing_rate_over_time(
            spike_seq=spike,
            title="Constant Input Firing Rate Over Time",
            save_path=os.path.join(save_dir, "constant_input_firing_rate.png"),
            show=True,
        )

    print("PASS")


def test_threshold_effect(
    visualize: bool = True,
    save_dir: str = "tests/figures",
):
    print("\n=== test_threshold_effect ===")

    T = 10
    B = 1
    x = torch.full((T, B, 1), 0.3)

    neuron_low = LIFNeuron(tau=0.9, threshold=0.6)
    neuron_high = LIFNeuron(tau=0.9, threshold=1.2)

    spike_low, mem_low = neuron_low(x, return_mem=True)
    neuron_low.reset_state()

    spike_high, mem_high = neuron_high(x, return_mem=True)
    neuron_high.reset_state()

    cnt_low = spike_low.sum().item()
    cnt_high = spike_high.sum().item()

    print(f"low threshold spike count  = {cnt_low}")
    print(f"high threshold spike count = {cnt_high}")

    assert cnt_low >= cnt_high, "Lower threshold should not fire less."

    if visualize:
        os.makedirs(save_dir, exist_ok=True)

        plot_membrane_and_spike(
            mem=mem_low[:, 0, 0],
            spike=spike_low[:, 0, 0],
            threshold=neuron_low.threshold,
            title="Low Threshold Response",
            save_path=os.path.join(save_dir, "low_threshold_trace.png"),
            show=True,
        )

        plot_membrane_and_spike(
            mem=mem_high[:, 0, 0],
            spike=spike_high[:, 0, 0],
            threshold=neuron_high.threshold,
            title="High Threshold Response",
            save_path=os.path.join(save_dir, "high_threshold_trace.png"),
            show=True,
        )

    print("PASS")


def test_tau_effect(
    visualize: bool = True,
    save_dir: str = "tests/figures",
):
    print("\n=== test_tau_effect ===")

    T = 10
    B = 1
    x = torch.full((T, B, 1), 0.2)

    neuron_small_tau = LIFNeuron(tau=0.5, threshold=1.0)
    neuron_large_tau = LIFNeuron(tau=0.95, threshold=1.0)

    spike_small, mem_small = neuron_small_tau(x, return_mem=True)
    neuron_small_tau.reset_state()

    spike_large, mem_large = neuron_large_tau(x, return_mem=True)
    neuron_large_tau.reset_state()

    print_tensor_1d("mem_small_tau", mem_small[:, 0, 0])
    print_tensor_1d("mem_large_tau", mem_large[:, 0, 0])

    assert mem_large[-1, 0, 0] >= mem_small[-1, 0, 0], \
        "Larger tau should keep more memory under constant input."

    if visualize:
        os.makedirs(save_dir, exist_ok=True)

        plot_membrane_and_spike(
            mem=mem_small[:, 0, 0],
            spike=spike_small[:, 0, 0],
            threshold=neuron_small_tau.threshold,
            title="Small Tau Response",
            save_path=os.path.join(save_dir, "small_tau_trace.png"),
            show=True,
        )

        plot_membrane_and_spike(
            mem=mem_large[:, 0, 0],
            spike=spike_large[:, 0, 0],
            threshold=neuron_large_tau.threshold,
            title="Large Tau Response",
            save_path=os.path.join(save_dir, "large_tau_trace.png"),
            show=True,
        )

    print("PASS")


def test_reset_mode_difference(
    visualize: bool = True,
    save_dir: str = "tests/figures",
):
    print("\n=== test_reset_mode_difference ===")

    T = 12
    B = 1
    x = torch.full((T, B, 1), 0.4)

    neuron_sub = LIFNeuron(tau=0.9, threshold=1.0, reset_mode="subtract")
    neuron_zero = LIFNeuron(tau=0.9, threshold=1.0, reset_mode="zero")

    spike_sub, mem_sub = neuron_sub(x, return_mem=True)
    neuron_sub.reset_state()

    spike_zero, mem_zero = neuron_zero(x, return_mem=True)
    neuron_zero.reset_state()

    print_tensor_1d("mem_sub", mem_sub[:, 0, 0])
    print_tensor_1d("mem_zero", mem_zero[:, 0, 0])
    print_tensor_1d("spike_sub", spike_sub[:, 0, 0])
    print_tensor_1d("spike_zero", spike_zero[:, 0, 0])

    same = torch.allclose(mem_sub, mem_zero)
    assert not same, "Different reset modes should generally lead to different dynamics."

    if visualize:
        os.makedirs(save_dir, exist_ok=True)

        plot_membrane_and_spike(
            mem=mem_sub[:, 0, 0],
            spike=spike_sub[:, 0, 0],
            threshold=neuron_sub.threshold,
            title="Subtract Reset Response",
            save_path=os.path.join(save_dir, "subtract_reset_trace.png"),
            show=True,
        )

        plot_membrane_and_spike(
            mem=mem_zero[:, 0, 0],
            spike=spike_zero[:, 0, 0],
            threshold=neuron_zero.threshold,
            title="Zero Reset Response",
            save_path=os.path.join(save_dir, "zero_reset_trace.png"),
            show=True,
        )

    print("PASS")


def test_batch_state_independence(
    visualize: bool = True,
    save_dir: str = "tests/figures",
):
    print("\n=== test_batch_state_independence ===")

    T = 8
    B = 2

    x = torch.zeros(T, B, 1)
    x[:, 0, 0] = 0.35
    x[:, 1, 0] = 0.10

    neuron = LIFNeuron(tau=0.9, threshold=1.0)
    spike, mem = neuron(x, return_mem=True)

    cnt0 = spike[:, 0, 0].sum().item()
    cnt1 = spike[:, 1, 0].sum().item()

    print(f"sample0 spike count = {cnt0}")
    print(f"sample1 spike count = {cnt1}")

    assert cnt0 >= cnt1, "Different batch samples should evolve independently."

    if visualize:
        os.makedirs(save_dir, exist_ok=True)

        plot_spike_raster(
            spike_seq=spike,
            title="Batch Independence Spike Raster",
            save_path=os.path.join(save_dir, "batch_independence_raster.png"),
            show=True,
        )

        plot_firing_rate_over_time(
            spike_seq=spike,
            title="Batch Independence Firing Rate",
            save_path=os.path.join(save_dir, "batch_independence_firing_rate.png"),
            show=True,
        )

    print("PASS")


if __name__ == "__main__":
    test_constant_input_response()
    test_threshold_effect()
    test_tau_effect()
    test_reset_mode_difference()
    test_batch_state_independence()