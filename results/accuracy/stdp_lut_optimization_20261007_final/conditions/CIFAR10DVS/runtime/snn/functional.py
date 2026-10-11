import torch


def heaviside(x: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    return (x >= threshold).to(x.dtype)


def surrogate_rectangle_grad(
    x: torch.Tensor,
    threshold: float = 1.0,
    lens: float = 0.5
) -> torch.Tensor:
    return (torch.abs(x - threshold) < lens).to(x.dtype) / (2.0 * lens)


def lif_update(mem: torch.Tensor, x: torch.Tensor, tau: float) -> torch.Tensor:
    return tau * mem + x


def apply_reset(
    mem: torch.Tensor,
    spike: torch.Tensor,
    threshold: float = 1.0,
    reset_mode: str = "subtract"
) -> torch.Tensor:
    if reset_mode == "subtract":
        return mem - spike * threshold
    elif reset_mode == "zero":
        return mem * (1.0 - spike)
    else:
        raise ValueError(f"Unsupported reset_mode: {reset_mode}")


def merge_time_batch(x: torch.Tensor):
    if x.dim() < 2:
        raise ValueError("Input tensor must have at least 2 dims [T, B, ...].")
    T, B = x.shape[:2]
    y = x.reshape(T * B, *x.shape[2:])
    return y, T, B


def split_time_batch(x: torch.Tensor, T: int, B: int) -> torch.Tensor:
    return x.reshape(T, B, *x.shape[1:])