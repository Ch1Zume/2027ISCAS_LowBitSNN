# Accuracy implementation

See the [Chinese runbook](../../README.md) or [English runbook](../../README_EN.md)
for installation, checkpoints, dataset preparation, full TEST replay and activity measurement.

`models.py`, `data.py`, `run.py` and `config.py` implement the maintained experiment
stack. Historical score replay uses the condition-specific frozen implementation
in `benchmarks/`, selected through `configs/benchmarks.json` and `scripts/replay.py`.
Input resolution, timestep selection, learning roles, nonnegative E/I banks and
hidden-only inference quantization are defined by each configuration.

Only the final `lut_final` models are distributed through the reproduction
manifest. Historical teacher-dependent training launchers are retained in Git history.
The original supervised classifier uses temporal mean logits followed by argmax.

The vendored reference implementation and attribution are in `vendor/deepeisnn/`.
