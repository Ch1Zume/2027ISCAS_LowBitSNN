# 2027ISCAS Low-bit SNN — final migration release

This checkout retains the original four-dataset benchmark, the selected 3M–5M
compression baseline, and the final LUT optimization. Model binaries travel in
`SNN-models-20261011.zip`; final code, configurations, scores and source audits
are tracked in Git. Datasets and discarded candidates are excluded.

Start with [迁移和复原手册](migration/RESTORE_MANUAL.md). Use
`migration/replay.py` for full TEST and `migration/train.py` to start a new
TRAIN/VALID run. `migration/catalog.json` binds each score to its unchanged
checkpoint, runtime and configuration using relative paths.

The retained frozen runtimes are intentional: different final conditions used
different implementations. The working modules in `experiments/` and `snn/`
support new training; published replay imports the condition's frozen modules.
Older absolute paths in source manifests are preserved provenance. The migration
entry points override data locations in memory without rewriting those records.

The LUT final full TEST scores are:

| Dataset | Parameters | T/view | Views | FP32 | uint2 | uint4 |
|---|---:|---:|---:|---:|---:|---:|
| MNIST | 4,452,879 | 4 | 1 | 99.26% | 99.29% | 99.29% |
| N-MNIST | 4,453,455 | 4 | 1 | 98.96% | 98.84% | 98.85% |
| CIFAR10-DVS | 4,679,375 | 8 | 2 | 76.20% | 74.40% | 75.70% |
| DVS Gesture | 4,456,656 | 16 | 1 | 95.49% | 95.14% | 95.14% |

These are preserved completed measurements, not new migration-time TEST runs.
CIFAR uses two inference views and calibrated hidden weights. Its uint2 result
remains below 75%. MNIST/N-MNIST/Gesture retain already passing compressed weights;
their selected LUT optimization epoch is zero. This is single-seed evidence.
The original VGG/ResNet comparison campaign is not claimed complete.
