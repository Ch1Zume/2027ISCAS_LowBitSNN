# 2027ISCAS Low-bit SNN — complete project delivery, 2026-10-13

This checkout retains the original four-dataset benchmark, the selected 3M–5M
compression baseline, final LUT optimization, and the complete four-dataset
spike-activity study. The default `main` branch contains final code, all frozen
runtimes/configurations, ten model/calibration binaries below 100 MiB, all
activity results, figures, input sequences, and integrity evidence.

Existing datasets, raw archives, download fragments, frame caches, project
caches and the working environment are retained on the source server. Four
original checkpoints above 100 MiB, the retained datasets/caches, and the
relocatable Linux environment are delivered as separately verified archives.
The old 2026-10-11 ZIP/bundle packages have been removed; their small historical
manifests/manuals are retained under `migration/history/20261011/`.

Start with [迁移和复原手册](migration/RESTORE_MANUAL.md). Use
`experiments/power/run_activity.py` for replay using the retained complete TEST
data, and `migration/train.py` to start a new TRAIN/VALID run after preparing the
missing TRAIN/VALID data. `migration/catalog.json` binds each score to its unchanged
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

The 2026-10-12 activity campaign independently replayed all twelve full-TEST
conditions with exact frozen score dictionaries. The 2026-10-13 environment
restore validation is recorded separately in `migration/reorganize_20261013/`.
CIFAR uses two inference views and calibrated hidden weights. Its uint2 result
remains below 75%. MNIST/N-MNIST/Gesture retain already passing compressed weights;
their selected LUT optimization epoch is zero. This is single-seed evidence.
The original VGG/ResNet comparison campaign is not claimed complete.

## Activity results and hardware boundary

See [complete activity outputs](results/new_runs/spike_activity_20261012/final/)
and [usage](experiments/power/USAGE.txt). The primary N-MNIST `hidden2` workload
has 512 input rows and 512 neurons, with uint4 input activity 16.1092% and output
activity 10.7842% per logical step. These are one-layer software activities.
The frozen WTA gates learning only and does not restrict forward spikes. Hard
output WTA or a different physical reset policy requires a separate evaluation.
Clock mapping remains unconfirmed; no physical power, RST count, MHz, pJ/SOP or
area measurement is claimed by these outputs.

## Windows download and Linux restore

The download script uses the existing `IMECAS-yufan` SSH alias and places the new
delivery under `E:\IMECAS_SNN\20261013\`, without overwriting the old local project.
It resumes at verified 512 MiB chunk boundaries and verifies both archives and
extracted files. A Windows receipt proves backup integrity; executing the Linux
environment requires a compatible Linux/WSL2 host and NVIDIA driver.

The current retained dataset state supports all four full-TEST evaluations.
MNIST also has its training raw files; event-dataset TRAIN/VALID restoration is
not complete. Existing partial downloads are preserved. The restore manual
documents both immediate TEST replay and the data preparation needed for new
training; the delivery does not claim previously deleted training data are back.
