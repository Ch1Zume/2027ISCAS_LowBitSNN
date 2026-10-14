# Low-bit SNN Runbook

English | [中文](README.md)

Reproduce the low-bit SNN experiments on MNIST, N-MNIST, CIFAR10-DVS and DVS128 Gesture.
Workflow: **clone → download models locally and upload → create the Conda environment → prepare data → run experiments**.

Git contains source code, execution configurations, fixed dataset splits, model manifests and reference scores.
Datasets, checkpoints, logs, per-sample records, figures, experiment outputs and installed environments are ignored. Git LFS is unnecessary.
The frozen source under `benchmarks/` defines the historical models and is required for reproduction.

## 1. Clone the repository

Prepare Git, Conda and a working NVIDIA GPU driver on the target Linux x86-64 server:

```bash
git clone --depth 1 https://github.com/Ch1Zume/2027ISCAS_LowBitSNN.git
cd 2027ISCAS_LowBitSNN
nvidia-smi
```

Run subsequent commands from the repository root. Full TEST evaluation and training require CUDA; file verification and model loading checks can run on CPU.

## 2. Upload locally downloaded checkpoints

Download model files to your local machine separately, then upload them to the target server using SCP/SFTP. **Models are not distributed or synchronized through this GitHub repository**.
With `/path/to/2027ISCAS_LowBitSNN` as the server-side repository root, use these exact destinations:

| Model group / file | Server destination, relative to the repository root |
|---|---|
| original, one per dataset | `checkpoints/original/<dataset>.pth` |
| compressed, one per dataset | `checkpoints/compressed/<dataset>.pth` |
| lut_final, one per dataset | `checkpoints/lut_final/<dataset>.pth` |
| CIFAR TRAIN calibration statistics | `checkpoints/lut_final/CIFAR10DVS_TRAIN_moments.pth` |
| CIFAR quantizer configuration | `checkpoints/lut_final/CIFAR10DVS_quantizer_settings.pth` |

Use exactly `MNIST`, `NMNIST`, `CIFAR10DVS` or `DVSGesture` for `<dataset>`; names are case-sensitive.
If a downloaded checkpoint is named `best.pth`, rename it according to its model group and dataset. For example, the final N-MNIST LUT model belongs at `checkpoints/lut_final/NMNIST.pth`.
Rename CIFAR `TRAIN_moments.pth` and `quantizer_settings.pth` by adding the `CIFAR10DVS_` prefix shown above.
The complete three-bundle benchmark needs **12 checkpoints + 2 calibration files**. A single-condition replay needs only that model and its calibration dependencies.

Create the directories from the repository root on the target server:

```bash
mkdir -p checkpoints/original checkpoints/compressed checkpoints/lut_final
```

Upload from your local terminal. These commands assume your downloaded files are arranged under `/local/models/` as above; replace paths, `USER` and `SERVER`:

```bash
scp /local/models/original/*.pth USER@SERVER:/path/to/2027ISCAS_LowBitSNN/checkpoints/original/
scp /local/models/compressed/*.pth USER@SERVER:/path/to/2027ISCAS_LowBitSNN/checkpoints/compressed/
scp /local/models/lut_final/*.pth USER@SERVER:/path/to/2027ISCAS_LowBitSNN/checkpoints/lut_final/
```

Example for uploading a single model:

```bash
scp /local/downloads/NMNIST/best.pth USER@SERVER:/path/to/2027ISCAS_LowBitSNN/checkpoints/lut_final/NMNIST.pth
```

On Windows, use PowerShell's `scp` or an SFTP client with the same server destinations.
`configs/checkpoints.json` records the required filenames, sizes and SHA256 hashes. Git ignores `checkpoints/`.

## 3. Create the Conda environment

```bash
CONDARC="$PWD/configs/condarc.yml" conda env create -f environment.yml
conda activate lowbitsnn
python -m pip check
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
```

`environment.yml` pins Python 3.10.20, PyTorch 2.7.1/CUDA 12.6, torchvision 0.22.1, SpikingJelly 0.0.0.0.14 and the other dependencies.
The project `configs/condarc.yml` maps the defaults alias to conda-forge; `CONDARC` scopes this setting to the invocation without editing global configuration.
The PyTorch wheel supplies CUDA userspace dependencies; the server still needs a compatible NVIDIA driver.
See the [official PyTorch installation history](https://pytorch.org/get-started/previous-versions/#v271) for the version pairing.

## 4. Check uploaded models

After uploading files and activating the environment, run on the target server:

```bash
python scripts/verify.py --source-only
python scripts/verify.py --checkpoints
```

The last command validates model files, strictly loads all 12 models on CPU, and checks parameter counts and nonnegative weights. It does not evaluate TEST accuracy.
For a single uploaded condition:

```bash
python scripts/verify.py --bundle lut_final --dataset NMNIST --checkpoints
```

## 5. Prepare TEST data

Accuracy replay and activity measurement do not require the complete TRAIN partition:

```bash
for dataset in MNIST NMNIST CIFAR10DVS DVSGesture; do
  python scripts/prepare_datasets.py --scope test --dataset "$dataset" --data-dir datasets
done
```

This downloads the required public archives and builds fixed TEST caches. Archives may contain additional samples, but only the frozen TEST selection is converted.
Archive URLs and checksums are in `configs/dataset_sources.json`; splits and file ordering are in `configs/dataset_splits/` and `configs/event_file_order.json`.
Event data retains the original 16 event-count frames (`split_by=number`); the model applies its configured timestep selection.

For an external data location, use the same `--data-dir /path/to/data` for preparation, replay and activity commands.
Interrupted downloads can be resumed by rerunning. If networking is unavailable, place the listed archives in `<data-dir>/<dataset>/download/`; MNIST uses torchvision's `MNIST/raw/` layout.

## 6. Replay accuracy

Start with the primary N-MNIST baseline:

```bash
python scripts/replay.py --bundle lut_final --dataset NMNIST --data-dir datasets --out results/replay/lut_final_NMNIST.json
```

Each invocation evaluates **FP32, uint2 and uint4 on the complete TEST partition**, checks that inference leaves the model state unchanged, and compares against the stored reference.
Replay all three model bundles and all four datasets:

```bash
for bundle in original compressed lut_final; do
  for dataset in MNIST NMNIST CIFAR10DVS DVSGesture; do
    python scripts/replay.py --bundle "$bundle" --dataset "$dataset" --data-dir datasets --out "results/replay/${bundle}_${dataset}.json"
  done
done
```

Use a new output filename for each run. The default `--comparison exact` requires identical full score dictionaries.
After changing hardware or dependencies, if only floating-point metrics differ, explicitly use `--comparison counts`: correct and sample counts must still match for every precision, and numerical differences remain recorded. Failed comparisons also save their output.

| Final LUT reference | T / view | Views | FP32 | uint2 | uint4 |
|---|---:|---:|---:|---:|---:|
| MNIST | 4 | 1 | 99.26% | 99.29% | 99.29% |
| N-MNIST | 4 | 1 | 98.96% | 98.84% | 98.85% |
| CIFAR10-DVS | 8 | 2 | 76.20% | 74.40% | 75.70% |
| DVS128 Gesture | 16 | 1 | 95.49% | 95.14% | 95.14% |

Only hidden fully connected synapse weights are quantized. CIFAR uses two views and frozen quantization calibration.
Final LUT weights for MNIST/N-MNIST/Gesture retain the selected compressed models. These are single-seed references, not a guarantee that new training produces identical scores.

## 7. Measure activity and export circuit stimuli

Prepare all four TEST datasets in step 5 and upload the complete LUT model group:

```bash
python experiments/power/run_activity.py --data-dir datasets --out results/activity --wait-data-seconds 0
python experiments/power/report_activity.py --out results/activity
python experiments/power/export_stimuli.py --out results/activity --precision uint4
python experiments/power/export_stimuli.py --out results/activity --precision uint2
```

Outputs include activity by layer/sample/timestep, figures, and N-MNIST hidden2 weight codes and input sequences.
Add `--datasets NMNIST` to evaluate only the primary baseline; the summary report requires all four datasets.
See the [activity contract](experiments/power/USAGE.txt): these are software activity and SOP counts, not measured silicon power or physical RST counts.

## 8. New training and reset ablation

Training needs complete TRAIN/VALID/TEST data. Use a separate directory to avoid treating a TEST-only cache as a complete dataset:

```bash
python scripts/prepare_datasets.py --scope all --download --dataset NMNIST --data-dir datasets_train
python scripts/train.py --dataset NMNIST --data-dir datasets_train --mode warm-start-final --epochs 6 --out results/train_nmnist
```

This entry point replays the compressed baseline, generates a TRAIN-only teacher cache from the original checkpoint, and then runs TRAIN/VALID optimization.
`warm-start-final` initializes a new optimizer from the final weights; `--mode from-scratch` randomly initializes the student but still uses the original teacher.
This is a new experiment, not continuation of the historical optimizer state or a guarantee of historical scores. TEST is deferred by default; after VALID selection, run:

```bash
SNN_DATA_DIR="$PWD/datasets_train" python results/train_nmnist/run_frozen.py --out results/train_nmnist --dataset NMNIST --condition lut_native_trace --stdp-lr 0.001 --bp-rate-scale 0.1 --test-only
```

Run the equal-budget N-MNIST reset comparison:

```bash
python experiments/reset_ablation/run.py --data-dir datasets_train --epochs 6 --out results/reset_nmnist
```

This trains both behaviors, freezes both VALID selections, evaluates complete TEST partitions, and generates comparison figures.
Upload original, compressed and lut_final weights; uploading all files from step 2 satisfies the dependencies of every entry point.

## 9. Tests and repository layout

```bash
python -m pytest -q test experiments/power/test_activity.py
```

| Path | Purpose | Git |
|---|---|---|
| `snn/`, `experiments/`, `scripts/`, `test/` | Algorithms, experiment drivers, utilities and tests | Tracked |
| `benchmarks/` | Frozen model implementations and required training settings | Tracked |
| `configs/`, `environment.yml` | Configurations, splits, manifests and environment definition | Tracked |
| `checkpoints/`, `artifacts/` | Separately uploaded weights and archives | Ignored |
| `datasets*/` | Raw downloads and generated frames | Ignored |
| `results/`, `outputs/`, `runs/`, `logs/` | Experiment records, caches, figures and outputs | Ignored |

`.gitignore` also excludes common model, array, log, image and archive formats, installed environments and editor caches.
Do not force-add generated experiment artifacts. Fixed splits and minimal reference scores are required reproduction inputs and remain in Git.
