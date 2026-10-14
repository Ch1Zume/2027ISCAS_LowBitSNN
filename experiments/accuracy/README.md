# Accuracy implementation

See the [Chinese runbook](../../README.md) or [English runbook](../../README_EN.md)
for installation, checkpoints, dataset preparation, full TEST replay and training.

`models.py`, `data.py`, `run.py` and `config.py` implement the maintained experiment
stack. Historical score replay uses the condition-specific frozen implementation
in `benchmarks/`, selected through `configs/benchmarks.json` and `scripts/replay.py`.
Input resolution, timestep selection, learning roles, nonnegative E/I banks and
hidden-only inference quantization are defined by each configuration.

Use `scripts/train.py` for a new LUT TRAIN/VALID experiment. Its warm-start mode
creates a new optimizer; it is not continuation of a historical training state.
The original supervised classifier uses temporal mean logits followed by argmax.

The vendored reference implementation and attribution are in `vendor/deepeisnn/`.
