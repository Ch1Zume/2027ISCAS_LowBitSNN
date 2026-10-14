"""Five-bit countdown eligibility and signed coefficients for float STDP.

This is a new recency rule: absent/old pre activity causes LTD at the first
winner. It does not reproduce continuous post-before-pre depression.
"""
import math

import torch


TRACE_MAX = 31


def signed_lut(a_plus=0.01, a_minus=0.012, trace_decay=0.9, coefficient_lsb=0.001):
    if not all(math.isfinite(v) for v in (a_plus, a_minus, trace_decay, coefficient_lsb)):
        raise ValueError("LUT parameters must be finite")
    if a_plus < 0 or a_minus < 0 or not 0 <= trace_decay < 1 or coefficient_lsb <= 0:
        raise ValueError("Invalid LUT rates, decay or coefficient LSB")
    values = []
    for u in range(TRACE_MAX + 1):
        recency = 0.0 if u == 0 else trace_decay ** (TRACE_MAX - u)
        values.append(round(((a_plus + a_minus) * recency - a_minus) / coefficient_lsb))
    if min(values) < -16 or max(values) > 15:
        raise ValueError("LUT coefficient exceeds signed 5-bit range")
    return torch.tensor(values, dtype=torch.int8)


def countdown_step(trace, events):
    """Refresh precedes winner lookup; uint8 is the software storage container."""
    if trace.dtype != torch.uint8 or events.dtype != torch.bool or trace.shape != events.shape:
        raise ValueError("Countdown expects matching uint8 trace and binary events")
    if torch.any(trace > TRACE_MAX):
        raise ValueError("Trace exceeds logical 5-bit range")
    silent = torch.where(trace > 0, trace - 1, torch.zeros_like(trace))
    return torch.where(events, torch.full_like(trace, TRACE_MAX), silent)


def stdp_options(cfg):
    """Shared options for the checkout and independently frozen runtimes."""
    return {key: cfg.get(key, default) for key, default in {
        "stdp_mode": "legacy_continuous", "stdp_input_encoding": "native",
        "stdp_event_amplitude": 1.0, "stdp_coefficient_lsb": 0.001,
        "stdp_update_rule": "soft_bounds_rownorm", "stdp_record_stats": False,
        "stdp_trace_seed": 0,
        "a_plus": 0.01, "a_minus": 0.012, "trace_decay": 0.9,
    }.items()}
