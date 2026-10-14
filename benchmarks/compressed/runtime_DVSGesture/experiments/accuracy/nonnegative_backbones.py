"""Author EI-SNN backbones with native nonnegative synapses, plus our own CNN."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from spikingjelly.activation_based import base, layer, neuron, surrogate
from safetensors.torch import load_model

from .vendor.deepeisnn.models.factory import build_model
from .vendor.deepeisnn.modules.activation import LIF
from .vendor.deepeisnn.modules.conv2d import SpikingEiConv2d
from .vendor.deepeisnn.modules.linear import SpikingEiLinear
from .vendor.deepeisnn.modules.norm2d import SpikingEiNorm2d
from .vendor.deepeisnn.utils.dim import AddTemporalDim, MergeTemporalDim, SplitTemporalDim


class AuthorBackbone(nn.Module):
    def __init__(self, family, specification, channels, steps, resolution):
        super().__init__()
        directory = Path(specification["directory"])
        checkpoint_digest = hashlib.sha256((directory / "model.safetensors").read_bytes()).hexdigest()
        config_digest = hashlib.sha256((directory / "config.json").read_bytes()).hexdigest()
        if specification.get("sha256", checkpoint_digest) != checkpoint_digest:
            raise ValueError("Pretrained checkpoint SHA256 differs from the frozen source lock")
        if specification.get("config_sha256", config_digest) != config_digest:
            raise ValueError("Pretrained config SHA256 differs from the frozen source lock")
        config = json.loads((directory / "config.json").read_text())
        if config["type"] != "EI-SNN":
            raise ValueError("The selected pretrained backbone must have nonnegative EI-SNN synapses")
        expected = ("VGG", 11) if family == "vgg11_bn" else ("ResNet", 18)
        if (config["arch"], config["num_layers"]) != expected:
            raise ValueError("Checkpoint architecture differs from the selected model")
        model = build_model(config, torch.device("cpu"), np.random.default_rng(42))
        missing, unexpected = load_model(model, directory / "model.safetensors", strict=True)
        if missing or unexpected:
            raise ValueError("Author checkpoint failed strict loading")
        for module in model.modules():
            if hasattr(module, "_inited"):
                module._inited = True
            if hasattr(module, "T"):
                module.T = steps
        self.family, self.steps = family, steps
        self.adaptations = ["native nonnegative excitatory/inhibitory synapses and author inhibition retained",
                            "pretrained classifier replaced by two WTA hidden layers and target classifier"]
        if steps != config["T"]:
            self.adaptations.append(f"source T={config['T']} adapted to existing target coding T={steps}")
        if family == "vgg11_bn":
            parts = list(model.layers.children())
            boundary = next(index for index, item in enumerate(parts) if isinstance(item, SpikingEiLinear))
            parts = parts[:boundary]
            parts = [item for item in parts if not isinstance(item, (AddTemporalDim, nn.Flatten))]
            if resolution < 32:
                [item for item in parts if isinstance(item, nn.AvgPool2d)][-1].ceil_mode = True
                self.adaptations.append("last average pool uses ceil_mode for native 28x28 MNIST")
            first = next(item for item in parts if isinstance(item, SpikingEiConv2d))
            self.features = nn.Sequential(*parts, nn.AdaptiveAvgPool2d(1), nn.Flatten(1), SplitTemporalDim(steps))
            self.adaptations.append("global average pooling supports native resolution without 7x7 expansion")
        else:
            model.init_expand = nn.Identity()
            model.fc = nn.Identity()
            model.final_ei_norm = nn.Identity()
            first = model.conv1
            self.features = model
        for convolution in (first.conv_ee, first.conv_ie):
            old = convolution.weight.detach()
            if old.shape[1] != channels:
                replacement = nn.Conv2d(channels, convolution.out_channels, convolution.kernel_size,
                                        stride=convolution.stride, padding=convolution.padding, bias=False)
                with torch.no_grad():
                    replacement.weight.copy_(old.mean(1, keepdim=True).repeat(1, channels, 1, 1)*old.shape[1]/channels)
                if convolution is first.conv_ee:
                    first.conv_ee = replacement
                else:
                    first.conv_ie = replacement
        self.adaptations.append(f"first excitatory/inhibitory kernels adapted to {channels} target channels")
        self.source = {"kind": "author_pretrained_nonnegative_SNN", "repository": "https://github.com/vwOvOwv/DeepEISNN",
                       "architecture": f"SpikingEi{config['arch']}{config['num_layers']}",
                       "source_dataset": config["dataset"], "source_steps": config["T"],
                       "checkpoint": str((directory / "model.safetensors").resolve()),
                       "sha256": checkpoint_digest, "config_sha256": config_digest,
                       "provenance": json.loads((Path(__file__).parent / "vendor/deepeisnn/provenance.json").read_text())}

    def forward(self, frames):
        if self.family == "vgg11_bn":
            return self.features(frames)
        # The author ResNet averages T in forward(); retain temporal features here.
        model = self.features
        x = model.init_merge(frames)
        x = model.merge1(model.lif1(model.split1(model.norm1(model.conv1(x)))))
        x = model.maxpool(x)
        x = model.layer4(model.layer3(model.layer2(model.layer1(x))))
        return model.final_split(model.adaptive_avgpool(x).flatten(1))


class CustomPositiveBackbone(nn.Module):
    """Our seven-stage topology, fresh exponential weights and local inhibition."""
    def __init__(self, channels, widths, steps, seed, ei_ratio=4):
        super().__init__()
        rng = np.random.default_rng(seed)
        parts = [MergeTemporalDim(steps)]
        for index, width in enumerate(widths):
            convolution = SpikingEiConv2d(channels, width, ei_ratio, torch.device("cpu"), rng,
                                          kernel_size=3, padding=1, bias=False)
            normalization = SpikingEiNorm2d(width, channels*9, ei_ratio, torch.device("cpu"))
            # Fixed rate-0.15 moment estimate gives a reproducible fresh start;
            # no validation/test sample participates in initialization.
            with torch.no_grad():
                scale = np.sqrt(0.1275 / ((channels*9)*(0.15+0.1275)))
                for weight in (convolution.conv_ee.weight, convolution.conv_ie.weight):
                    weight.copy_(torch.from_numpy(rng.exponential(scale, size=tuple(weight.shape))).float())
                normalization.gain_i.fill_(np.sqrt(0.15+0.1275)/(np.sqrt(channels*9)*0.15))
            convolution._inited = normalization._inited = True
            parts += [convolution, normalization, SplitTemporalDim(steps),
                      LIF(tau=2.0, surrogate_function="atan"), MergeTemporalDim(steps)]
            if index in {0, 1, 3, 5}:
                parts.append(nn.AvgPool2d(2, ceil_mode=True))
            channels = width
        parts += [nn.AdaptiveAvgPool2d(1), nn.Flatten(1), SplitTemporalDim(steps)]
        self.features = nn.Sequential(*parts)

    def forward(self, frames):
        return self.features(frames)


class BalancedPositiveConv(nn.Module):
    """Two physical nonnegative synapse banks and explicit current subtraction.

    Neither convolution receives a signed weight. The initial net current
    matches a Kaiming-normal convolution, while both positive banks train by BP.
    """
    def __init__(self, inputs, outputs, kernel_size=3, stride=1):
        super().__init__()
        self.excitatory = layer.Conv2d(inputs, outputs, kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.inhibitory = layer.Conv2d(inputs, outputs, kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        with torch.no_grad():
            nn.init.kaiming_normal_(self.excitatory.weight, mode="fan_in", nonlinearity="relu")
            self.inhibitory.weight.copy_(self.excitatory.weight.neg().clamp_min(0))
            self.excitatory.weight.clamp_(min=0)

    def forward(self, inputs):
        return self.excitatory(inputs) - self.inhibitory(inputs)


class BalancedCausalPositiveConv(nn.Module):
    """Three temporal taps, nonnegative E/I banks, and no future-frame access."""
    def __init__(self, inputs, outputs, implementation="conv3d"):
        super().__init__()
        if implementation not in {"conv3d", "lagged_conv2d"}:
            raise ValueError("Unknown causal convolution implementation")
        self.implementation = implementation
        self.excitatory = nn.Conv3d(inputs, outputs, (3, 3, 3), padding=(0, 1, 1), bias=False)
        self.inhibitory = nn.Conv3d(inputs, outputs, (3, 3, 3), padding=(0, 1, 1), bias=False)
        with torch.no_grad():
            nn.init.kaiming_normal_(self.excitatory.weight, mode="fan_in", nonlinearity="relu")
            self.inhibitory.weight.copy_(self.excitatory.weight.neg().clamp_min(0))
            self.excitatory.weight.clamp_(min=0)

    def forward(self, inputs):
        if inputs.ndim != 5:
            raise ValueError("Expected [time,batch,channels,height,width]")
        if self.implementation == "lagged_conv2d":
            # Use exactly the spatial predecessor's current-tap operation.
            # History taps remain independently differentiable physical banks.
            outputs = []
            for bank in (self.excitatory, self.inhibitory):
                current = None
                for lag in (0, 1, 2):
                    shifted = inputs if lag == 0 else torch.cat((torch.zeros_like(inputs[:min(lag, len(inputs))]), inputs[:-lag]), 0)
                    value = F.conv2d(shifted.flatten(0, 1), bank.weight[:, :, 2-lag].contiguous(), padding=1)
                    value = value.reshape(inputs.shape[0], inputs.shape[1], *value.shape[1:])
                    current = value if current is None else current + value
                outputs.append(current)
            return outputs[0] - outputs[1]
        sequence = F.pad(inputs.permute(1, 2, 0, 3, 4), (0, 0, 0, 0, 2, 0))
        current = self.excitatory(sequence) - self.inhibitory(sequence)
        return current.permute(2, 0, 1, 3, 4)


class TemporalBatchNorm2d(nn.Module, base.StepModule):
    """Independent per-time-step affine parameters and population moments.

    CNN-only adaptation of BNTT; unlike the reference we retain learned beta
    so a shared-BN own checkpoint can be expanded with identical eval outputs.
    """
    def supported_step_mode(self):
        return ("m",)

    def __init__(self, channels, steps):
        super().__init__()
        self.step_mode = "m"
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError("Temporal BN requires a positive fixed step count")
        self.steps = steps
        self.banks = nn.ModuleList([nn.BatchNorm2d(channels) for _ in range(steps)])

    def forward(self, inputs):
        if inputs.ndim != 5 or inputs.shape[0] != self.steps:
            raise ValueError("Temporal BN requires the configured [T,B,C,H,W] input")
        return torch.stack([bank(inputs[t]) for t, bank in enumerate(self.banks)])


class MultiStepGroupNorm2d(nn.Module, base.StepModule):
    """Group moments within each sample and time step; no population state."""
    def supported_step_mode(self):
        return ("m",)

    def __init__(self, channels, groups):
        super().__init__()
        self.step_mode = "m"
        self.norm = nn.GroupNorm(groups, channels)

    def forward(self, inputs):
        if inputs.ndim != 5:
            raise ValueError("GroupNorm requires [T,B,C,H,W] input")
        return self.norm(inputs.flatten(0, 1)).reshape_as(inputs)


class CustomBalancedBackbone(nn.Sequential):
    """Our spatial CNN with stable BN currents and learned PLIF time constants."""
    def __init__(self, channels, widths, temporal_layers=0, time_bn_steps=None, temporal_implementation="conv3d", pool_after=None, group_norm_groups=None):
        pool_after = set(range(len(widths)-1) if pool_after is None else pool_after)
        parts = []
        for index, width in enumerate(widths):
            convolution = (BalancedCausalPositiveConv(channels, width, temporal_implementation) if index < temporal_layers
                           else BalancedPositiveConv(channels, width))
            normalization = (MultiStepGroupNorm2d(width, group_norm_groups) if group_norm_groups is not None
                             else layer.BatchNorm2d(width) if time_bn_steps is None
                             else TemporalBatchNorm2d(width, time_bn_steps))
            parts.extend([convolution, normalization,
                          neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=surrogate.ATan(),
                                                   detach_reset=True, backend="torch")])
            if index in pool_after:
                parts.append(layer.MaxPool2d(2, ceil_mode=True))
            channels = width
        super().__init__(*parts)


def residual_spike():
    return neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=surrogate.ATan(),
                                   detach_reset=True, backend="torch")


class BalancedResidualBlock(nn.Module):
    """Current or spike-add shortcuts; each spiking node emits binary spikes."""
    def __init__(self, inputs, outputs, stride=1, connection="current"):
        super().__init__()
        self.conv1 = BalancedPositiveConv(inputs, outputs, stride=stride)
        self.connection = connection
        if connection not in {"current", "spike_add"}:
            raise ValueError("Unknown custom residual connection")
        self.bn1 = layer.BatchNorm2d(outputs)
        self.spike1 = residual_spike()
        self.conv2 = BalancedPositiveConv(outputs, outputs)
        self.bn2 = layer.BatchNorm2d(outputs)
        self.shortcut = nn.Identity() if inputs == outputs and stride == 1 else nn.Sequential(
            BalancedPositiveConv(inputs, outputs, kernel_size=1, stride=stride), layer.BatchNorm2d(outputs))
        if connection == "spike_add" and not isinstance(self.shortcut, nn.Identity):
            self.shortcut.add_module("spike", residual_spike())
        self.spike2 = residual_spike()

    def forward(self, inputs):
        current = self.bn2(self.conv2(self.spike1(self.bn1(self.conv1(inputs)))))
        if self.connection == "spike_add":
            return self.spike2(current) + self.shortcut(inputs)
        return self.spike2(current + self.shortcut(inputs))


class CustomBalancedResidualBackbone(nn.Sequential):
    """Fresh four-stage residual SNN with explicit nonnegative EI banks."""
    def __init__(self, channels, widths, blocks=2, connection="current"):
        parts = [BalancedPositiveConv(channels, widths[0]), layer.BatchNorm2d(widths[0]),
                 residual_spike(), layer.MaxPool2d(2, ceil_mode=True)]
        channels = widths[0]
        for index, width in enumerate(widths):
            for block in range(blocks):
                parts.append(BalancedResidualBlock(channels, width, stride=2 if index > 0 and block == 0 else 1, connection=connection))
                channels = width
        super().__init__(*parts)

