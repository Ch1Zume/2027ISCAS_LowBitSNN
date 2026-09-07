import torch
import torch.nn as nn


def _sample_for_quantile(
    x: torch.Tensor, max_samples: int = 1_000_000
) -> torch.Tensor:
    """
    Down-sample a 1D tensor if it's too large for torch.quantile.
    """
    if x.numel() <= max_samples:
        return x
    step = x.numel() // max_samples + 1
    return x[::step]


def quantize_2bit(
    w: torch.Tensor,
    percentile: float = 99.99,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict]:
    """
    Asymmetric (affine) unsigned 2-bit quantization with zero-point (zp) shift.

    Input:
      w: FP32/FP16 weight tensor (can be any shape)
      percentile: clamp by symmetric abs-percentile (default 99.99%)
    Output:
      w_hat: dequantized weight tensor (float32) after quantize->dequantize

    Steps:
      1) clamp by abs-percentile to remove outliers (symmetric clamp)
      2) compute [w_min, w_max] on clamped tensor
      3) scale + zero-point shift (asymmetric affine)
      4) round to {0,1,2,3}
      5) dequantize back to float
    """
    if w.numel() == 0:
        raise ValueError("Empty tensor.")

    qmin, qmax = 0, 3

    # --- 1) clamp outliers by symmetric abs-percentile ---
    w_fp = w.detach()  # do not backprop through stats by default
    abs_flat = w_fp.abs().flatten()

    # torch.quantile expects float; downsample if tensor is huge to avoid errors
    abs_flat = _sample_for_quantile(abs_flat.to(torch.float32))
    a = torch.quantile(abs_flat, percentile / 100.0)  # >= 0
    a = torch.clamp(a, min=eps)

    w_c = torch.clamp(w.to(torch.float32), min=-a.item(), max=a.item())

    # --- 2) get min/max on clamped tensor ---
    w_min = w_c.min()
    w_max = w_c.max()

    # avoid degenerate range
    rng = torch.clamp(w_max - w_min, min=eps)

    # --- 3) scale + zero-point shift ---
    # scale maps real range to integer range length (qmax-qmin)
    scale = rng / float(qmax - qmin)  # scalar tensor
    # zero-point in integer domain (can be fractional before rounding)
    zp_real = qmin - w_min / scale
    zero_point = torch.round(zp_real).clamp(qmin, qmax).to(torch.int32)

    # --- 4) quantize to {0,1,2,3} ---
    q = torch.round(w_c / scale + zero_point.to(w_c.dtype))
    q = q.clamp(qmin, qmax).to(torch.int32)

    # --- 5) dequantize ---
    w_hat = (q.to(torch.float32) - zero_point.to(torch.float32)) * scale + 0.0

    return w_hat.to(w.dtype)


def quantize_4bit(
    w: torch.Tensor, percentile: float = 99.99, eps: float = 1e-8
) -> torch.Tensor:
    """
    Quantize-dequantize weight tensor to 4 bits/channel (values in {0,...,15}).
    Steps:
      1) clamp by abs-percentile to remove outliers (symmetric clamp)
      2) compute [w_min, w_max] on clamped tensor
      3) scale + zero-point shift (asymmetric affine)
      4) round to {0,1,...,15}
      5) dequantize back to float
    """
    if w.numel() == 0:
        raise ValueError("Empty tensor.")

    qmin, qmax = 0, 15

    # --- 1) clamp outliers by symmetric abs-percentile ---
    w_fp = w.detach()  # do not backprop through stats by default
    abs_flat = w_fp.abs().flatten()

    abs_flat = _sample_for_quantile(abs_flat.to(torch.float32))
    a = torch.quantile(abs_flat, percentile / 100.0)
    a = torch.clamp(a, min=eps)

    w_c = torch.clamp(w.to(torch.float32), min=-a.item(), max=a.item())

    # --- 2) get min/max on clamped tensor ---
    w_min = w_c.min()
    w_max = w_c.max()

    # avoid degenerate range
    rng = torch.clamp(w_max - w_min, min=eps)

    # --- 3) scale + zero-point shift ---
    scale = rng / float(qmax - qmin)
    zp_real = qmin - w_min / scale
    zero_point = torch.round(zp_real).clamp(qmin, qmax).to(torch.int32)

    # --- 4) quantize to {0,1,...,15} ---
    q = torch.round(w_c / scale + zero_point.to(w_c.dtype))
    q = q.clamp(qmin, qmax).to(torch.int32)

    # --- 5) dequantize ---
    w_hat = (q.to(torch.float32) - zero_point.to(torch.float32)) * scale + 0.0

    return w_hat.to(w.dtype)


@torch.no_grad()
def apply_channelwise_quantization(
    net: torch.nn.Module,
    percentile: float = 99.99,
    num_bits: int = 2,
) -> None:
    """
    Apply channel-wise quantization to all trainable weight tensors
    in the network. For a weight tensor with shape [C_out, ...], each
    output channel (index along dim 0) is quantized independently.
    Support 2bit (num_bits=2) and 4bit (num_bits=4).
    """
    # Also support SpikingJelly's Conv2d / Linear if available.
    extra_types: tuple[type, ...]
    try:
        from spikingjelly.activation_based import layer as sj_layer

        extra_types = (sj_layer.Conv2d, sj_layer.Linear)
    except Exception:
        extra_types = tuple()

    target_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear) + extra_types

    quantize_fn = None
    if num_bits == 2:
        quantize_fn = quantize_2bit
    elif num_bits == 4:
        quantize_fn = quantize_4bit
    else:
        raise ValueError(f"Only 2bit and 4bit quantization supported. Got num_bits={num_bits}.")

    for module in net.modules():
        if not isinstance(module, target_types):
            continue
        if not hasattr(module, "weight"):
            continue
        weight = getattr(module, "weight")
        if weight is None or not isinstance(weight, torch.Tensor):
            continue
        if weight.dim() < 1:
            continue

        w = weight.data
        # Channel-wise along output dimension (dim 0)
        out_channels = w.shape[0]
        w_q = torch.empty_like(w)
        for c in range(out_channels):
            w_q[c] = quantize_fn(w[c], percentile=percentile)

        weight.data.copy_(w_q)


def fake_quantize_tensor(
    x: torch.Tensor,
    num_bits: int = 8,
    percentile: float = 99.99,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Fake-quantize tensor for QAT (straight-through estimator).

    For 2/4 bits, reuse the existing weight quantizers and apply
    STE trick: y = q.detach() + (x - x.detach()) so that
    gradients are approximately identity while forward uses q.
    """
    if not x.is_floating_point():
        return x

    if num_bits == 2:
        q = quantize_2bit(x, percentile=percentile, eps=eps)
    elif num_bits == 4:
        q = quantize_4bit(x, percentile=percentile, eps=eps)
    else:
        # Generic uniform affine quantization to num_bits
        qmin, qmax = 0, (1 << num_bits) - 1

        with torch.no_grad():
            x_fp = x.detach()
            abs_flat = _sample_for_quantile(
                x_fp.abs().flatten().to(torch.float32)
            )
            a = torch.quantile(abs_flat, percentile / 100.0)
            a = torch.clamp(a, min=eps)
            x_c = torch.clamp(x.to(torch.float32), min=-a.item(), max=a.item())

            x_min = x_c.min()
            x_max = x_c.max()
            rng = torch.clamp(x_max - x_min, min=eps)

            scale = rng / float(qmax - qmin)
            zp_real = qmin - x_min / scale
            zero_point = torch.round(zp_real).clamp(qmin, qmax).to(torch.int32)

            q_int = torch.round(x_c / scale + zero_point.to(x_c.dtype))
            q_int = q_int.clamp(qmin, qmax).to(torch.int32)
            q = (q_int.to(torch.float32) - zero_point.to(torch.float32)) * scale

        q = q.to(x.dtype)

    # Straight-through estimator: forward uses q, backward ~ identity
    return q.detach() + (x - x.detach())


@torch.no_grad()
def quantize_activation_tensor(
    x: torch.Tensor,
    num_bits: int = 8,
    percentile: float = 99.99,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Real (non-STE) activation quantization for inference.

    This is similar to fake_quantize_tensor but without the straight-through
    estimator trick.
    """
    if not x.is_floating_point():
        return x

    if num_bits == 2:
        return quantize_2bit(x, percentile=percentile, eps=eps)
    if num_bits == 4:
        return quantize_4bit(x, percentile=percentile, eps=eps)

    # Generic uniform affine quantization to num_bits
    qmin, qmax = 0, (1 << num_bits) - 1

    x_fp = x.detach()
    abs_flat = _sample_for_quantile(x_fp.abs().flatten().to(torch.float32))
    a = torch.quantile(abs_flat, percentile / 100.0)
    a = torch.clamp(a, min=eps)
    x_c = torch.clamp(x.to(torch.float32), min=-a.item(), max=a.item())

    x_min = x_c.min()
    x_max = x_c.max()
    rng = torch.clamp(x_max - x_min, min=eps)

    scale = rng / float(qmax - qmin)
    zp_real = qmin - x_min / scale
    zero_point = torch.round(zp_real).clamp(qmin, qmax).to(torch.int32)

    q_int = torch.round(x_c / scale + zero_point.to(x_c.dtype))
    q_int = q_int.clamp(qmin, qmax).to(torch.int32)
    q = (q_int.to(torch.float32) - zero_point.to(torch.float32)) * scale
    return q.to(x.dtype)


def enable_qat_weight_fake_quant(
    net: torch.nn.Module,
    num_bits: int = 2,
    percentile: float = 99.99,
) -> list[torch.utils.hooks.RemovableHandle]:
    """
    Attach forward hooks that fake-quantize Conv/Linear weights channel-wise
    during training (QAT).  Original FP weights are restored after each forward
    so that the optimizer always updates full-precision weights.
    """
    extra_types: tuple[type, ...]
    try:
        from spikingjelly.activation_based import layer as sj_layer
        extra_types = (sj_layer.Conv2d, sj_layer.Linear)
    except Exception:
        extra_types = tuple()

    target_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear) + extra_types

    if num_bits == 2:
        quantize_fn = quantize_2bit
    elif num_bits == 4:
        quantize_fn = quantize_4bit
    else:
        raise ValueError(
            f"Only 2/4-bit weight quantization supported, got {num_bits}."
        )

    handles: list[torch.utils.hooks.RemovableHandle] = []

    def _pre_hook(module: nn.Module, inputs):
        weight = getattr(module, "weight", None)
        if weight is None or not isinstance(weight, torch.Tensor):
            return
        if weight.dim() < 1:
            return
        module._orig_weight_data = weight.data.clone()
        w = weight.data
        out_channels = w.shape[0]
        w_q = torch.empty_like(w)
        for c in range(out_channels):
            w_q[c] = quantize_fn(w[c], percentile=percentile)
        weight.data.copy_(w_q)

    def _post_hook(module: nn.Module, inputs, output):
        if hasattr(module, "_orig_weight_data"):
            module.weight.data.copy_(module._orig_weight_data)
            del module._orig_weight_data
        return output

    for module in net.modules():
        if isinstance(module, target_types):
            if hasattr(module, "weight") and module.weight is not None:
                h1 = module.register_forward_pre_hook(_pre_hook)
                h2 = module.register_forward_hook(_post_hook)
                handles.extend([h1, h2])

    return handles


def enable_qat_activation_fake_quant(
    net: torch.nn.Module,
    num_bits: int = 8,
    percentile: float = 99.99,
) -> list[torch.utils.hooks.RemovableHandle]:
    """
    Attach fake-quantization hooks to activations of common layers for QAT.

    The hooks quantize layer outputs with straight-through estimator,
    so training can approximate inference-time quantization.
    """
    # Also support SpikingJelly's Conv2d / Linear if available.
    extra_types: tuple[type, ...]
    try:
        from spikingjelly.activation_based import layer as sj_layer

        extra_types = (sj_layer.Conv2d, sj_layer.Linear)
    except Exception:
        extra_types = tuple()

    target_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear) + extra_types

    handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook():
        def _hook(module: nn.Module, inputs, output):
            if isinstance(output, torch.Tensor):
                return fake_quantize_tensor(
                    output, num_bits=num_bits, percentile=percentile
                )
            if isinstance(output, (tuple, list)):
                quantized = []
                for o in output:
                    if isinstance(o, torch.Tensor):
                        quantized.append(
                            fake_quantize_tensor(
                                o, num_bits=num_bits, percentile=percentile
                            )
                        )
                    else:
                        quantized.append(o)
                return type(output)(quantized)
            return output

        return _hook

    hook_fn = _make_hook()
    for module in net.modules():
        if isinstance(module, target_types):
            h = module.register_forward_hook(hook_fn)
            handles.append(h)

    return handles


@torch.no_grad()
def enable_activation_quant_inference(
    net: torch.nn.Module,
    num_bits: int = 8,
    percentile: float = 99.99,
) -> list[torch.utils.hooks.RemovableHandle]:
    """
    Attach real (non-STE) activation quantization hooks for inference.

    Quantizes outputs of Conv/Linear (pre-LIF activations), to simulate
    low-bit activation behavior at inference time.
    """
    # Also support SpikingJelly's Conv2d / Linear if available.
    extra_types: tuple[type, ...]
    try:
        from spikingjelly.activation_based import layer as sj_layer

        extra_types = (sj_layer.Conv2d, sj_layer.Linear)
    except Exception:
        extra_types = tuple()

    target_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear) + extra_types

    handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook():
        def _hook(module: nn.Module, inputs, output):
            if isinstance(output, torch.Tensor):
                return quantize_activation_tensor(
                    output, num_bits=num_bits, percentile=percentile
                )
            if isinstance(output, (tuple, list)):
                quantized = []
                for o in output:
                    if isinstance(o, torch.Tensor):
                        quantized.append(
                            quantize_activation_tensor(
                                o, num_bits=num_bits, percentile=percentile
                            )
                        )
                    else:
                        quantized.append(o)
                return type(output)(quantized)
            return output

        return _hook

    hook_fn = _make_hook()
    for module in net.modules():
        if isinstance(module, target_types):
            h = module.register_forward_hook(hook_fn)
            handles.append(h)

    return handles