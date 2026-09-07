import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from snn import (
    Conv2dSNN,
    LinearSNN,
    MaxPool2dSNN,
    AvgPool2dSNN,
    BatchNorm2dSNN,
    BatchNorm1dSNN,
    FlattenSNN,
)


def test_conv2d_snn_shape_and_equivalence():
    print("\n=== test_conv2d_snn_shape_and_equivalence ===")

    T, B, C, H, W = 4, 2, 3, 8, 8
    x = torch.randn(T, B, C, H, W)

    layer = Conv2dSNN(3, 5, kernel_size=3, padding=1)
    y = layer(x)

    assert y.shape == (T, B, 5, H, W)

    # compare with explicit for-loop
    y_loop = []
    for t in range(T):
        y_loop.append(layer.conv(x[t]))
    y_loop = torch.stack(y_loop, dim=0)

    assert torch.allclose(y, y_loop, atol=1e-6), "Conv2dSNN mismatch with loop version."
    print("PASS")


def test_linear_snn_shape_and_equivalence():
    print("\n=== test_linear_snn_shape_and_equivalence ===")

    T, B, N = 5, 3, 16
    x = torch.randn(T, B, N)

    layer = LinearSNN(16, 10)
    y = layer(x)

    assert y.shape == (T, B, 10)

    y_loop = []
    for t in range(T):
        y_loop.append(layer.forward_single_step(x[t]))
    y_loop = torch.stack(y_loop, dim=0)

    assert torch.allclose(y, y_loop, atol=1e-6), "LinearSNN mismatch with loop version."
    print("PASS")


def test_pooling_shape():
    print("\n=== test_pooling_shape ===")

    T, B, C, H, W = 4, 2, 3, 8, 8
    x = torch.randn(T, B, C, H, W)

    maxpool = MaxPool2dSNN(2)
    avgpool = AvgPool2dSNN(2)

    y1 = maxpool(x)
    y2 = avgpool(x)

    assert y1.shape == (T, B, C, 4, 4)
    assert y2.shape == (T, B, C, 4, 4)
    print("PASS")


def test_batchnorm2d_shape():
    print("\n=== test_batchnorm2d_shape ===")

    T, B, C, H, W = 4, 2, 3, 8, 8
    x = torch.randn(T, B, C, H, W)

    bn = BatchNorm2dSNN(C)
    y = bn(x)

    assert y.shape == x.shape
    print("PASS")


def test_batchnorm1d_shape():
    print("\n=== test_batchnorm1d_shape ===")

    T, B, N = 6, 2, 20
    x = torch.randn(T, B, N)

    bn = BatchNorm1dSNN(N)
    y = bn(x)

    assert y.shape == x.shape
    print("PASS")


def test_flatten_snn():
    print("\n=== test_flatten_snn ===")

    T, B, C, H, W = 4, 2, 3, 8, 8
    x = torch.randn(T, B, C, H, W)

    flatten = FlattenSNN()
    y = flatten(x)

    assert y.shape == (T, B, C * H * W)
    print("PASS")


if __name__ == "__main__":
    test_conv2d_snn_shape_and_equivalence()
    test_linear_snn_shape_and_equivalence()
    test_pooling_shape()
    test_batchnorm2d_shape()
    test_batchnorm1d_shape()
    test_flatten_snn()