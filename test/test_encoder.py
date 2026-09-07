import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from snn import RepeatEncoder, PoissonEncoder, LatencyEncoder


def test_repeat_encoder():
    print("\n=== test_repeat_encoder ===")

    B, C, H, W = 2, 1, 4, 4
    x = torch.rand(B, C, H, W)

    encoder = RepeatEncoder(T=5)
    y = encoder(x)

    assert y.shape == (5, B, C, H, W)
    assert torch.allclose(y[0], x)
    assert torch.allclose(y[-1], x)
    print("PASS")


def test_poisson_encoder():
    print("\n=== test_poisson_encoder ===")

    B, C, H, W = 2, 1, 4, 4
    x = torch.rand(B, C, H, W)

    encoder = PoissonEncoder(T=6)
    y = encoder(x)

    assert y.shape == (6, B, C, H, W)
    assert torch.all((y == 0) | (y == 1)), "Poisson output should be binary."
    print("PASS")


def test_latency_encoder():
    print("\n=== test_latency_encoder ===")

    x = torch.tensor([[[[0.0, 0.25, 0.75, 1.0]]]])  # [B=1,C=1,H=1,W=4]

    encoder = LatencyEncoder(T=5, allow_zero=False)
    y = encoder(x)

    assert y.shape == (5, 1, 1, 1, 4)
    assert torch.all((y == 0) | (y == 1)), "Latency output should be binary."

    # each nonzero element spikes at most once
    spike_count = y.sum(dim=0)
    assert torch.all(spike_count <= 1), "Latency code should fire at most once per site."

    # zero input should produce no spike
    assert spike_count[0, 0, 0, 0].item() == 0
    print("PASS")


if __name__ == "__main__":
    test_repeat_encoder()
    test_poisson_encoder()
    test_latency_encoder()