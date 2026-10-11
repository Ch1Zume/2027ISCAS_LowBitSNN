# SNN accuracy experiment

Training and backward passes use float32. Inference alone quantizes hidden
Linear weights to 2 or 4 bits. Conv weights and the final classifier remain
float32. All Conv/Linear synaptic weights, including the classifier, are
nonnegative. Biases and BatchNorm coefficients may be signed. No activation
quantization or QAT hooks are used. This protocol has a separate entry point
from the older `our_snn.py` / `weight_bit_validate.py` experiments.

## Models and provenance

The reviewed campaign uses `configs/accuracy_launch_20261004.yaml`. VGG and
ResNet use the author's genuine, pretrained **nonnegative EI-SNN** backbones
from [DeepEISNN](https://github.com/vwOvOwv/DeepEISNN) and its
[checkpoint release](https://huggingface.co/vwOvOwv/DeepEISNN). Excitatory and
inhibitory synapses retain nonnegative weights and the author's inhibition;
pretrained signed ANN weights are not clipped into a substitute SNN.

The vendor snapshot is pinned to commit
`6b3355c074fb46fd606192c733f833efe0a68793`. Only absolute imports are rewritten
into the local package. `vendor/deepeisnn/provenance.json` records original and
local source SHA256s. Each checkpoint and its architecture config has a full
SHA256 lock in the launch config. Strict loading is required. Two tests verify
that backbone features exactly match the author model before target adaptation.

| Experiment model | Actual backbone | Checkpoint source dataset |
|---|---|---|
| `vgg11_bn`, MNIST / NMNIST | SpikingEiVGG11 | CIFAR10 |
| `vgg11_bn`, CIFAR10DVS / DVSGesture | SpikingEiVGG11 | CIFAR10DVS |
| `resnet18`, all four targets | SpikingEiResNet18 | CIFAR10 |
| `custom`, all four targets | Our seven-stage EI CNN | From scratch |

`vgg11_bn` is retained as a legacy CLI identifier; this campaign's actual VGG
is the author EI VGG11, without BatchNorm. Except for VGG on CIFAR10DVS, these
are cross-dataset SNN transfers, not target-dataset pretrained checkpoints.
Author source accuracy is not claimed for the adapted models.

Both official backbones feed two new hidden Linear layers of widths 4096/4096
and a target classifier. VGG uses global average pooling instead of expanding
to 7x7; ResNet has the explicitly accepted hidden adapter. Input kernels are
averaged/repeated/scaled to native grayscale or two-polarity channels. The last
VGG average pool uses ceil mode for 28x28 MNIST. The temporal dimension is
adapted to target coding; native input resolutions are retained. All adaptations
and source architecture names are included in run metadata.

The custom network uses channels `[64,128,256,320,512,1024,1024]`, average pools
after stages 1/2/4/6, global average pooling, two 4096-wide hidden layers, and a
target classifier. Its convolution weights use a fresh exponential
initialization and its own topology; no pretrained weights are copied.

The agreed size metric sums output channels of **all convolution modules**,
including residual projections and EI connections, plus hidden Linear widths.
The classifier is reported separately. Spiking population widths, spatial
neuron counts and parameter counts are also reported, with distinct field names.

| Model | `width_sum` excluding classifier | Hidden widths |
|---|---:|---|
| SpikingEiVGG11 | 11,632 | 4096, 4096 |
| SpikingEiResNet18 with adapter | 14,192 | 4096, 4096 |
| Custom | 12,352 | 4096, 4096 |

The custom width sum is 6.2% above VGG and 13.0% below ResNet. Add 10 classifier
outputs for MNIST, NMNIST and CIFAR10DVS, or 11 for Gesture.

The legacy SpikingJelly/ANN path remains available for compatibility in
`accuracy_matrix.yaml`; it is not the reviewed campaign configuration. Historical
ANN initialization files remain under `checkpoints/sources` and are not used in
the new campaign.

## Learning rules

Convolutions and the classifier use supervised BP without WTA. The first hidden
Linear layer uses STDP; the second uses BP. Each hidden layer's rule is
configurable. Both hidden layers use WTA to gate learning.

For each sample presentation, each hidden layer selects its first neuron to
reach threshold. Simultaneous crossings use the lowest index. Other neurons
retain forward activity; only the winner can update its synaptic row. BP
weight and bias gradients are gated per sample, and losing weight/bias rows and optimizer
momentum remain unchanged, including with Adam and weight decay. STDP parameters
do not receive BP updates. All input-error gradients propagate to earlier
layers, including errors from losing neurons, so convolutional BP is not gated
by WTA. The input-gradient regression test covers this distinction.

Hidden dynamics are `mem = 0.9 * mem + current`, threshold crossing spikes and
subtractive reset. Membranes and STDP traces reset between presentations.
Threshold adaptation persists across samples/epochs, is saved in checkpoints,
and is frozen during validation and test. A cycle is one simulated time step.
Batch samples are parallel; every observed win contributes a threshold jump.

Exponential mode:

```text
offset[k+1] = exp(-1/tau) * offset[k] + delta * number_of_wins[k]
threshold = baseline + offset
```

Rate mode:

```text
rate[k+1] = exp(-1/rate_tau) * rate[k] + (1-exp(-1/rate_tau)) * mean_batch_spike[k]
threshold[k+1] = threshold[k] + eta * (rate[k+1] - target_rate)
```

Rates are spikes per neuron per time step, averaged across the batch. Thresholds
are bounded by configurable minimum/maximum values. Defaults are initial
experimental hyperparameters, not tuned accuracy claims. Fixed-threshold `off`
mode is available for a control experiment.

Saturation is handled separately: multiplicative STDP potentiation scales with
`weight_max-weight`, depression with `weight`, followed by winner-only row-sum
normalization and clipping. Coincident pre activity is included, allowing
first-step winners to learn. Hidden BP weights are clipped to `[0,weight_max]`;
BP uses gradient norm clipping. The reviewed campaign uses SGD with momentum
0.9 and zero weight decay to preserve the author EI normalization dynamics. Conv/classifier synaptic weights and EI gains are projected nonnegative after
each optimizer step. Hidden layers use ATan surrogate gradients and sparse
positive initialization with 32 initial connections per neuron. One training
batch, without labels or optimizer updates, calibrates each hidden layer to a
mean current of 0.6 times its baseline threshold. The calibration preserves
thresholds, backbone weights, training RNG and loader RNG; it uses no validation
or test data.

## Data and evaluation

| Dataset CLI name | Native input | T | Coding | Launch batch |
|---|---|---:|---|---:|
| MNIST | 1x28x28 | 4 | Poisson, existing `our_snn.py` default | 64 |
| NMNIST | 2x34x34 | 16 | Event count frames, split by event number | 8 |
| CIFAR10DVS | 2x128x128 | 16 | Event count frames, split by event number | 2 |
| DVSGesture | 2x128x128 | 16 | Existing Gesture frame protocol | 2 |

MNIST T=100 was retired after the design review. Prepared event caches remain
T=16. Event counts are preserved, not binarized. Batch sizes are identical
across models within a dataset and are fixed before the campaign starts.

VGG and ResNet fine-tune for at most **10 epochs** at initial learning rate
0.001; custom trains from scratch for at most **30 epochs** at 0.01. Each uses
cosine decay over its own budget. Validation plateau can stop either phase
earlier. Metadata records the actual budget; legacy dataset epoch fields are
not the training budget.

MNIST/N-MNIST/Gesture retain their official train/test partitions. CIFAR10-DVS
uses a seeded, class-stratified 90/10 train/test split. A separate stratified 10%
validation partition is taken from training for every dataset. Best checkpoints
are selected by float validation top-1 accuracy, not test accuracy. Split indices
are persisted under a content-hashed filename. Evaluation reports correct/sample
counts and top-1 accuracy. FP32, 2-bit and 4-bit use the same checkpoint, frozen
threshold state and Poisson RNG sequence.

Unsigned quantization uses zero point zero and per-output-row scale
`max(weight)/(2**bits-1)`. Zero rows stay zero. Discrete weights are simulated
with dequantized float operations; there is no integer-kernel or bit-packing
claim. Float weights are restored after each inference precision check.

The matrix has 24 float training runs (3 models x 4 datasets x 2 homeostasis
modes), 48 primary quantized accuracy cases and 24 accompanying FP32 controls.

## Live monitoring and failure containment

Training prints flushed JSON records and appends them to `metrics.jsonl` in the
run directory. No remote logging account or additional package is needed. The
`monitoring` block in `configs/accuracy_launch_20261004.yaml` controls the following:

- Every 20 batches or approximately 30 seconds, at a batch boundary: current
  and sample-weighted mean loss, training top-1, learning rate, gradient norm,
  backbone gradient norm, final backbone spike rate, hidden spike rates, eligible
  neuron counts, threshold bounds, throughput, and estimated epoch time remaining.
- Before training and every 100 batches: a fixed, seeded class-balanced sample of up to 256
  validation examples, selected without decoding the full event dataset. The probe uses a fixed Poisson RNG stream, restores training RNG and
  module train/eval flags, and freezes homeostasis. It reports loss, top-1,
  prediction/label histograms, majority-class baseline, and logit variation.
  Probe indices and label histograms are stored in `validation_probe.json` and
  checkpoint metadata. A single-label probe cannot establish collapse. Probe
  scores do not select checkpoints. Full validation still selects the
  best checkpoint at each epoch; test samples never drive the guards.
- Nonfinite loss or BP gradient stops immediately, before the optimizer writes.
  After 100 warmup batches, silent backbone/all hidden spikes together with
  backbone gradient norm at most `1e-12` on three consecutive scheduled health
  checks stops the run. Health checks occur on fixed batch intervals; extra
  wall-clock log messages do not change the guard counters.
- Three consecutive validation probes dominated by one predicted class
  (at least 99%), with accuracy no better than the majority-class baseline plus
  2 percentage points and no meaningful loss/accuracy improvement, abort the
  run. Improvements of 0.01 loss or 0.2 percentage points top-1 reset the counter.
  A single low-accuracy batch cannot trigger this guard.
- Five consecutive full-validation epochs without a meaningful top-1
  improvement, after at least three completed epochs, ends training normally
  and evaluates the best checkpoint at FP32/2/4 bits.

An abnormal abort writes `stopped.json` with its reason and observed metrics,
plus `diagnostic.pth` containing the partial model/optimizer state. The latter
is for diagnosis and is **not** a supported resume checkpoint. Completed-epoch
`latest.pth` and `best.pth` are retained. By default, an aborted training run does
not publish `accuracy.json` or proceed to its final test evaluation. The campaign
marks the job `aborted`, stops with `stopped_for_review`, and leaves other jobs
queued. The selected custom/exponential campaign opts into `evaluate_on_collapse`
and allows one full training/validation epoch before the collapse guard is
eligible. For a confirmed class collapse with a completed-epoch best checkpoint,
it evaluates that checkpoint on the full test set and records
`training_outcome: class_collapse`, the stop reason, and the diagnostic path.
Such a completed evaluation is a failed-learning result. Numerical and unexpected
process failures still stop the queue without publishing successful accuracy.
If a nonfinite value is encountered during final evaluation, that evaluation
is aborted and no successful result is published. Normal validation-plateau
early stopping is recorded separately as `training_stop` in `accuracy.json`.

Guard counters are saved in training checkpoints. Resuming a checkpoint that
already reached the plateau condition does not launch more training epochs.
These thresholds are configurable initial settings, not guarantees of learning.
Tests verify failure triggers, failure containment, ordinary early stopping,
and unchanged training weights/thresholds/RNG with probes enabled.

For a running experiment, follow its `metrics.jsonl` with `tail -f`. Batch and
probe records are also present in the campaign job's log file. Numerical failures
stop the whole queue for review; collapse handling follows the frozen policy
described above.

## Running

Install `requirements-accuracy.txt` in a project environment, with a Torch wheel
appropriate to the machine. The current `.venv` borrows the existing Torch 2.7.1
installation and contains SpikingJelly 0.0.0.0.14 / torchvision 0.22.1+cpu.
Dependencies pass `pip check`. The native host has a working RTX 4090; sandboxed
processes may not see CUDA, so GPU training must run with native GPU access.
The 2026-10-02 campaign was paused by user request on 2026-10-03 and is retained
as historical evidence. The reviewed experiment uses a new campaign directory.
Final accuracy results require completed full test-set evaluations. The outcome
field distinguishes ordinary completion from evaluation of a collapsed run.

```bash
.venv/bin/python -m experiments.accuracy.run --config configs/accuracy_launch_20261004.yaml matrix
.venv/bin/python -m experiments.accuracy.run prepare --dataset MNIST
.venv/bin/python -m experiments.accuracy.run prepare --dataset NMNIST
.venv/bin/python -m experiments.accuracy.run prepare --dataset CIFAR10DVS
.venv/bin/python -m experiments.accuracy.run prepare --dataset DVSGesture
```

Storage is `datasets/<dataset>` within this repository. MNIST/CIFAR10-DVS have
automatic download adapters. The fixed official adapter version requires the
following archives for the other datasets:

- `datasets/NMNIST/download/Train.zip`, `Test.zip`, from the
  [dataset author](https://www.garrickorchard.com/datasets/n-mnist).
- `datasets/DVSGesture/download/DvsGesture.tar.gz`, `gesture_mapping.csv`,
  `LICENSE.txt`, `README.txt`, from the
  [official IBM release](https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794).

Missing-file errors provide filenames, official links and MD5s. Already converted
`events_np` directories can also be used. Existing frame caches matching the
requested T and split protocol are read directly without downloading archives.

Examples of full runs:

```bash
.venv/bin/python -m experiments.accuracy.run --config configs/accuracy_launch_20261004.yaml train --model vgg11_bn --dataset MNIST --homeostasis exponential
.venv/bin/python -m experiments.accuracy.run --config configs/accuracy_launch_20261004.yaml train --model resnet18 --dataset NMNIST --homeostasis rate
.venv/bin/python -m experiments.accuracy.run --config configs/accuracy_launch_20261004.yaml train --model custom --dataset DVSGesture --homeostasis exponential
```

Repeat the combinations listed by `matrix`. Each float run automatically
evaluates both bit widths and FP32, writes `accuracy.json`, and saves `best.pth`
and `latest.pth`. `--resume` continues an experiment checkpoint, restoring
optimizer, scheduler, thresholds, Poisson RNG and shuffled-loader RNG. Keep the
same protocol config/model/dataset/mode and `--limit`/`--steps` for resume/eval.

The reviewed campaign is stored under `results/accuracy/campaign_20261004`:

- `config.yaml` freezes the configuration; `status.json` records its SHA256,
  source hashes, runner/worker PIDs, and all 24 jobs.
- `datasets.json` records download/preparation status. Event datasets become
  ready after official archive MD5 checks, full sample-count checks and native
  frame-shape checks. The historical preparation evidence is retained, and the reviewed campaign
  records verified current counts and split manifests in its own readiness file.
- `logs/<model>_<dataset>_<mode>_seed42.log` records batch progress and epoch
  validation results. `runs/<same-name>/accuracy.json` contains final FP32/2/4-bit
  test-set accuracy; a progress-log training accuracy is not that final metric.
- `*_download_sources.json` records effective public download URLs and official
  checksums. N-MNIST uses the author's Mendeley release; CIFAR10-DVS uses
  Figshare's public storage; Gesture uses the IBM Box release. Download failures
  and retries are retained in the preparation logs.

The campaign runner processes eligible jobs sequentially on one GPU. Dataset
preparation runs in separate CPU/download processes. Its file lock prevents a
second runner from starting the same campaign. An interrupted running campaign
can continue with its frozen configuration and latest training checkpoints.
For a fresh launch, the reviewed config is frozen by this command:

```bash
.venv/bin/python -B -u -m experiments.accuracy.campaign --directory results/accuracy/campaign_20261004 --config configs/accuracy_launch_20261004.yaml
```

Explicitly `paused` or `stopped_for_review` campaigns refuse to launch children.
The historical `campaign_20261002` cannot restart automatically. Source/config changes
also require a new campaign directory; the historical campaign is retained.

Small checks are explicitly marked `functional_check` and stored in a separate
subdirectory. Their accuracy is not a full-dataset experimental result:

```bash
.venv/bin/python -m experiments.accuracy.run --config configs/accuracy_launch_20261004.yaml train --model custom --dataset MNIST --homeostasis exponential --device cuda --epochs 1 --batch-size 2 --limit 4
MPLBACKEND=Agg .venv/bin/python -m pytest -q test/test_training_monitoring.py test/test_accuracy_experiment.py test/test_nonnegative_backbones.py test/test_layer.py test/test_encoder.py test/test_stdp.py
```

The 72 passing tests cover learning gates with dense backbone error propagation,
both threshold controllers, inference quantization, paired Poisson inputs,
offline event caches, checkpoint continuation, monitoring failures, author
feature preservation and training-only calibration.

Three small MNIST learning qualifications use an 8,192-sample training prefix
and full validation, with separate `functional_check` output directories. They
establish that the training path learns; they are not final dataset results.
An earlier silent-spike variant aborted at batch 140 and retained diagnostics,
verifying that the guard can terminate an invalid run.

The launch preflight exercises native full-T/full-batch training forward and
backward, and 2/4-bit inference for all 12 model/dataset combinations on the RTX
4090. It records shapes, finite gradients, widths, actual peak CUDA allocations
and counts in `campaign_20261004/preflight_native.json`. Only training samples
are consumed and its temporary models are discarded. It does not report test
accuracy or initialize the formal campaign from qualification models.

## Selected custom/exponential evaluation

The user selected the custom network plus exponential homeostasis after the
MNIST matrix. The first selected-recipe attempt used
`configs/accuracy_custom_exponential_20261004.yaml` and
`results/accuracy/custom_exponential_20261004`. It stopped for review after
Gesture collapsed and is retained as historical evidence. MNIST's completed FP32/2/4-bit
result is reused byte-for-byte from the original campaign; its hash, original
source/config hashes, selection dataset and original probe method are recorded
in `inherited_results.json`. This is a reused result, not a new MNIST run.

That attempt queued Gesture, CIFAR10DVS and NMNIST as fresh custom networks for at most
30 epochs, in that order, with the existing T=16 event caches and native input
resolutions. Network widths, BP/STDP rules, WTA, exponential thresholds,
learning rate, batch sizes and quantization stay identical to the selected
recipe. The validation probe now covers all available classes deterministically;
full validation and test partitions are unchanged. Resume from an older
prefix-probe checkpoint is refused because its guard history is incompatible.

```bash
.venv/bin/python -B -u -m experiments.accuracy.campaign --directory results/accuracy/custom_exponential_20261004 --config configs/accuracy_custom_exponential_20261004.yaml
```

The runner regenerates `accuracy_summary.csv` with every status update. Scores
are filled only after complete FP32/2/4-bit test evaluations; queued/aborted
jobs have empty metric cells. `status.json`, per-run `metrics.jsonl` and
`accuracy.json` remain the detailed evidence. Numerical/collapse guards stop
the queue and preserve diagnostics. Historical campaigns remain stopped.

The active selected-recipe evaluation lives in
`results/accuracy/custom_exponential_eval_20261004`, with launch configuration
`configs/accuracy_custom_exponential_eval_20261004.yaml`. It keeps the original
network, initialization, learning rates, batch sizes and exponential thresholds.
Diagnostic parameter variants are archived and are not adopted in this table.
MNIST reuses its completed result. Gesture evaluates the best completed-epoch
checkpoint from the confirmed collapsed training; this is explicitly labeled
`training_outcome: class_collapse`, not converged training. NMNIST and CIFAR10DVS
train fresh models with the selected recipe.

For new selected-recipe jobs, collapse checks begin after one complete
training/validation epoch. `evaluate_on_collapse: true` permits full FP32/2/4-bit
test evaluation of a numerical-finite validation-best completed-epoch checkpoint
after a class-collapse stop. Stop reason, diagnostics, selected epoch and
`training_outcome` are retained in the result and CSV. This does not turn
collapsed training into a successful learning result. Nonfinite loss/gradients
or unexpected process failures stop the queue without a successful accuracy
result. The default policy for other configurations remains abort without test
evaluation. Historical campaign directories retain their original status.

The local report watcher refreshes `progress_report.md` and
`evaluation_audit.json` every 30 seconds, and exits when the campaign completes
or stops for review. It checks frozen source/config hashes, full test sample
counts, correct/count accuracy, prediction and label totals, paired labels,
frozen thresholds, hidden-only quantization and the byte-identical reused MNIST
result. Its audit distinguishes available evaluations from completion of all
four; learning outcomes remain visible even when a test evaluation completes.

```bash
.venv/bin/python results/accuracy/custom_exponential_eval_20261004/report.py
```
