"""Author EI-SNN backbones with native nonnegative synapses, plus our own CNN."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from spikingjelly.activation_based import layer, neuron, surrogate
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
    def __init__(self, inputs, outputs):
        super().__init__()
        self.excitatory = layer.Conv2d(inputs, outputs, 3, padding=1, bias=False)
        self.inhibitory = layer.Conv2d(inputs, outputs, 3, padding=1, bias=False)
        with torch.no_grad():
            nn.init.kaiming_normal_(self.excitatory.weight, mode="fan_in", nonlinearity="relu")
            self.inhibitory.weight.copy_(self.excitatory.weight.neg().clamp_min(0))
            self.excitatory.weight.clamp_(min=0)

    def forward(self, inputs):
        return self.excitatory(inputs) - self.inhibitory(inputs)


class CustomBalancedBackbone(nn.Sequential):
    """Our spatial CNN with stable BN currents and learned PLIF time constants."""
    def __init__(self, channels, widths):
        parts = []
        for index, width in enumerate(widths):
            parts.extend([BalancedPositiveConv(channels, width), layer.BatchNorm2d(width),
                          neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=surrogate.ATan(),
                                                   detach_reset=True, backend="torch")])
            if index < len(widths)-1:
                parts.append(layer.MaxPool2d(2, ceil_mode=True))
            channels = width
        super().__init__(*parts)

