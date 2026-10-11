# 2027ISCAS_SNN

Research code for the 2027 ISCAS SNN experiments, including model definitions,
quantization/error-injection experiments, and evaluation scripts.

## Accuracy experiment

The new accuracy workflow is documented in [experiments/accuracy/README.md](experiments/accuracy/README.md).
The reviewed campaign uses author-pretrained nonnegative Spiking EI VGG11 /
ResNet18 backbones and a custom network,
float32 BP/STDP with winner-gated hidden-layer learning, selectable exponential
or rate-based threshold homeostasis, and inference-only unsigned 2/4-bit hidden
Linear weights. Convolutions and the classification head stay float32 and use BP.
All Conv/Linear synaptic weights are nonnegative. Dataset storage is `./datasets`.
The launch config is `configs/accuracy_launch_20261004.yaml`: MNIST T=4, event
datasets T=16, at most 10 fine-tuning epochs versus 30 from-scratch epochs, with
live loss logging and automatic failure guards.

The compact custom-network and LUT STDP follow-up is documented in
[experiments/stdp/README.md](experiments/stdp/README.md). It preserves native
forward activity while sampling discrete events only for the local learning
trace, records explicit temporal/structural adaptations, and selects checkpoints
on complete validation data before full FP32/2-bit/4-bit tests and independent
replays. Current campaign conditions are recorded in each result bundle's policy
and provenance; the launch settings above describe the earlier campaign.

The older entry points below do not implement this experiment protocol.

## Repository boundary

The project-specific code is kept in this repository. The public SpikingJelly
source and the local compatibility snapshot are maintained separately under:

```text
/work/dependency/2027VLSI_SNN/spikingjelly_repo
/work/dependency/2027VLSI_SNN/spikingjelly_local_patch
```

Checkpoints, datasets, W&B runs, Python environments, and generated work
directories are intentionally excluded by `.gitignore`.

## Environment

The original environment was local to the server and is not part of this
repository. Recreate a suitable Conda environment from the package requirements
needed for the experiment before running scripts.
