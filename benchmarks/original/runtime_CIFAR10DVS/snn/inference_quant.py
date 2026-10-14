"""Unsigned, zero-preserving 2/4-bit inference quantization of hidden FC only."""

from contextlib import contextmanager

import torch


@torch.no_grad()
def quantize_nonnegative(weight, bits):
    if bits not in (2, 4) or weight.ndim != 2:
        raise ValueError("Expected a matrix and 2 or 4 bits")
    if not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("Quantized weights must be finite and nonnegative")
    levels = (1 << bits)-1
    maximum = weight.amax(1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / levels, torch.ones_like(maximum))
    codes = (weight / scale).round().clamp(0, levels).to(torch.uint8)
    return codes.to(weight.dtype) * scale, codes, scale


@contextmanager
def quantized_hidden_weights(model, bits):
    if model.training:
        raise RuntimeError("Weight quantization is permitted only in evaluation mode")
    saved, metadata = [], []
    try:
        with torch.no_grad():
            for name, layer in model.hidden_layers.items():
                original = layer.weight.detach().clone()
                saved.append((layer.weight, original))
                quantized, codes, scale = quantize_nonnegative(original, bits)
                layer.weight.copy_(quantized)
                metadata.append({"layer": name, "bits": bits, "codes": codes.cpu(), "scale": scale.cpu()})
        yield metadata
    finally:
        with torch.no_grad():
            for weight, original in saved:
                weight.copy_(original)
