# Compress a trained custom SNN to 3M–5M

The campaign uses the four frozen checkpoints in
`results/accuracy/custom_performance_candidates_20261004/selected_benchmark_final_20261005`.
Each condition imports its own audited runtime. The original benchmark files and
teacher checkpoints stay intact. Every dense convolution bank, hidden weight,
classifier, bias and learned neuron parameter counts towards the target.

The first candidate halves convolution channel widths. Hidden widths become
512+512 for MNIST/N-MNIST, 768 for CIFAR10-DVS, and 1024 for Gesture. Input
resolution, T, encoding, PLIF dynamics, WTA competition size, BP/STDP roles,
threshold homeostasis and inference quantization come from the teacher profile.
Pure hidden populations retain their rule. Hybrid populations retain complete
competition groups with matching BP/STDP masks, including Gesture's group roles.
The earlier matching of channel/hidden-width sums across datasets is superseded
for this compression attempt by the user's explicit 3M–5M parameter target.

TRAIN covariance statistics choose nonredundant channels. Ridge reconstruction
redistributes removed channels into retained convolution inputs; reconstructed
effective signed kernels are split into two nonnegative physical EI banks. Hidden
pruning preserves selected neurons and rescales their surviving positive input
mass. A TRAIN-only classification-head recovery and optional TRAIN BN refit precede
distillation. Hidden synaptic updates during recovery training still require WTA
eligibility and follow their assigned BP/STDP rule. Dropout is capped at 0.2.

The teacher supplies cached FP32 TRAIN logits. For MNIST the teacher sees one fixed
Poisson draw per sample, while the student continues stochastic Poisson coding.
No validation or test sample is a distillation target. The optimization objective
is 0.5×cross-entropy + 0.5×KL×temperature² at temperature 2. Original FP32 and
nonnegative projection rules remain active; there is no training quantization or
mixed precision. Validation selects the final state, then that fixed state is
tested at FP32/uint2/uint4 using identical input RNG. Both low-bit modes constrain
hidden Linear weights only.

Run on the CUDA host using the project environment:

```bash
.venv/bin/python experiments/compression/campaign.py --out results/accuracy/compression_3to5M_20261006_v1
.venv/bin/python experiments/compression/summarize.py results/accuracy/compression_3to5M_20261006_v1
```

Default recovery budgets are 12 epochs for MNIST/N-MNIST and 30 for CIFAR/Gesture.
NaN, dead gradients and persistent prediction collapse abort ineffective training;
six stale validation epochs can stop it after epoch five. Flushed JSON logs include
cross-entropy, distillation loss, gradient norm, spikes, eligible neurons, validation
accuracy and timing. `last.pth` includes optimizer, scheduler and RNG states.

Resume an interrupted condition, or independently replay a completed test:

```bash
.venv/bin/python experiments/compression/run.py --dataset CIFAR10DVS --out results/accuracy/compression_3to5M_20261006_v1/CIFAR10DVS --resume
.venv/bin/python experiments/compression/replay.py --condition results/accuracy/compression_3to5M_20261006_v1/CIFAR10DVS
```

`replay.py` verifies and imports the compression-source snapshot that actually
created the condition. `audit.py --condition <directory>` checks three isolated
TRAIN updates for FP32/nonnegative weights, disjoint BP/STDP winners, protected
loser momentum, and unchanged published checkpoints.

Alternative CIFAR candidates can retain more front channels with
`--profile front_preserved`, or warm-start a compressed state while changing T
with `--profile half --steps 16 --student-init <best.pth>`. These are distinct
recorded conditions. Compare complete validation accuracy then validation loss;
test scores do not rank candidates. `publish.py --trials <trial-directories>
--out <new-directory>` copies the selected audited checkpoints, frozen model and
compression source, checksums, and a standalone full-test `reproduce.py`.

Some teacher defaults in `config.json` (`init`, `lr`, `training_epochs`,
`monitoring`) describe the base model. The actual compression initializer,
learning rate and epoch budget are in `provenance.json` and the frozen `run.py`;
the run uses six stale validation epochs from epoch five for early stopping.

Only `accuracy.json` provides completed full-test evidence. Initialization,
validation and training progress are not a completed accuracy result. Compare
against the corresponding frozen teacher using the same test split. The campaign
is one compression attempt; the parameter target does not promise unchanged
accuracy under the additional nonnegative/WTA/STDP constraints.
