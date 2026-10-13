"""Accounting tests for padding, zero weights, fractional inputs and winner gates."""
import json
import sys
from pathlib import Path

import torch
from spikingjelly.activation_based import layer

sys.path.insert(0, str(Path(__file__).parent))
from activity import ActivityCollector
from test_data import ROOT
sys.path.insert(0, str(ROOT / "migration"))
from common import entry, runtime


class Container(torch.nn.Module):
    def __init__(self, unit):
        super().__init__()
        self.unit = unit

    def forward(self, x):
        return self.unit(x)


def test_conv_padding_and_zero_weights(tmp_path):
    conv = layer.Conv2d(1, 2, 3, padding=1, bias=False, step_mode="m")
    with torch.no_grad():
        conv.weight.fill_(1)
        conv.weight[0, 0, 1, 1] = 0
    model = Container(conv).eval()
    x = torch.zeros(2, 1, 1, 3, 3)
    x[0, 0, 0, 0, 0] = 1
    x[1, 0, 0, 1, 1] = 1
    c = ActivityCollector(model, tmp_path, 1, 1, 2, (), ())
    try:
        c.begin(torch.tensor([0]))
        model(x)
        c.end()
        result = c.finalize()["unit"]
    finally:
        c.close()
    # Corner reaches four positions per output channel; center reaches nine.
    assert result["SOP"] == 4*2 + 9*2
    assert result["effective_SOP"] == result["SOP"] - 2


def test_hidden_resets_are_all_spikes_not_first_winner(tmp_path):
    Hidden = runtime(entry("lut_final", "NMNIST"))["wta"].WTAHidden
    hidden = Hidden(2, 3, rule="bp", homeostasis="off", membrane_decay=.9)
    with torch.no_grad():
        hidden.weight.fill_(1)
        hidden.bias.zero_()
    model = Container(hidden).eval()
    c = ActivityCollector(model, tmp_path, 1, 1, 2, Hidden, ())
    try:
        c.begin(torch.tensor([0]))
        model(torch.ones(2, 1, 2))
        c.end()
        result = c.finalize()["unit"]
    finally:
        c.close()
    assert result["counters"]["fire"] == 6
    assert result["counters"]["winner"] == 1
    assert result["SOP"] == 12
    assert result["hardware_RST_count"] is None


def test_fractional_activity_is_not_a_physical_spike_count(tmp_path):
    model = Container(layer.Linear(2, 3, bias=False, step_mode="m")).eval()
    c = ActivityCollector(model, tmp_path, 1, 1, 1, (), ())
    try:
        c.begin(torch.tensor([0]))
        model(torch.tensor([[[.5, 0.]]]))
        c.end()
        result = c.finalize()["unit"]
    finally:
        c.close()
    assert result["counters"]["input_nonzero"] == 1
    assert result["counters"]["addressed_terms"] == 3
    assert result["SOP"] is None
