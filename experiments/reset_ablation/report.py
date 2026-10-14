"""Reproducible recovery curves and explicit full-TEST comparison."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    record = json.loads((out / "results.json").read_text())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"preserve": "#276FBF", "reset_all": "#D55E00"}
    labels = {"preserve": "Preserve other membranes", "reset_all": "Reset whole hidden layer"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    for variant, selected in record["selections"].items():
        history = selected["history"]
        x = [h["block"] for h in history]
        y = [100*h["validation"]["top1_accuracy"] for h in history]
        axes[0].plot(x, y, "o-", color=colors[variant], label=labels[variant], markersize=4)
        i = selected["selected_block"]
        axes[0].scatter([i], [y[i]], s=110, facecolors="none", edgecolors=colors[variant], linewidths=1.5)
        axes[1].plot(x, [h["validation"]["loss"] for h in history], "o-", color=colors[variant], markersize=4)
    axes[0].set(ylabel="Complete VALID accuracy (%)", xlabel="Recovery epoch (0 = initial)")
    axes[1].set(ylabel="Complete VALID cross-entropy", xlabel="Recovery epoch (0 = initial)")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle("N-MNIST: matched warm-start recovery; one seed; T=4", fontsize=11)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / ("validation_curves." + ext), dpi=200)
    lines = ["# N-MNIST winner-triggered membrane reset recovery", "",
        "Both variants start from the identical frozen final checkpoint and use fresh Adam optimizers. "
        "Six full TRAIN epochs per variant; 53,994 TRAIN / 6,006 VALID / 10,000 TEST samples. "
        "Winner-triggered reset is active in both training and inference, in both hidden layers.", "",
        "The intervention clears every membrane in the sample's 512-neuron hidden layer after its first "
        "winner in the presentation. Current-step simultaneous spikes are retained; winner eligibility "
        "is not rearmed. Threshold adaptation, learning roles, input frames and readout are unchanged.", "",
        "## Complete TEST", "",
        "| Behavior | State | Selected epoch | FP32 % | uint2 % | uint4 % |",
        "|---|---|---:|---:|---:|---:|"]
    for variant in ("preserve", "reset_all"):
        for state in ("initial", "selected"):
            score = record["results"][variant][state]
            epoch = 0 if state == "initial" else record["selections"][variant]["selected_block"]
            values = " | ".join(f"{100*score[p]['top1_accuracy']:.2f}" for p in ("float32", "uint2", "uint4"))
            lines.append(f"| {variant} | {state} | {epoch} | {values} |")
    lines += ["", "Reset minus preserve after VALID selection (percentage points): " +
              ", ".join(f"{k}: {v:+.2f}" for k, v in record["reset_minus_preserve_pp"].items()) + ".", "",
        "## Interpretation boundary", "",
        "This is a single-seed, six-epoch warm-start recovery experiment, not a from-scratch comparison "
        "or a validation of repeated WIN_CLR/rearm transactions. Every epoch uses the full TRAIN split; "
        "checkpoint selection uses only complete VALID accuracy, then loss, and allows epoch zero. "
        "Both selections were frozen before any new TEST evaluation. All TEST modes use 10,000 samples. "
        "The original baseline correct counts were replayed and all frozen delivery hashes remained unchanged.", "",
        "Training follows the existing LUT recovery driver: 0.5 CE + 0.5 original TRAIN-only teacher KD "
        "(temperature 2); initial BP learning rates backbone 2e-5, hidden BP 1e-5, head 1e-4; "
        "stdp_lr=0.001; cosine BP schedule to 1e-6. Training is FP32; only hidden FC weights use uint2/uint4 "
        "for inference. Conv/head/bias/scales/neuron parameters remain floating point.", "",
        "Evidence: contract.json; selections_before_TEST.json; results.json; comparison.csv; "
        "preserve/ and reset_all/ hold runtime snapshots, initial/best/last checkpoints, provenance, "
        "per-epoch VALID results and full TEST scores. Curves can be regenerated with "
        "`python experiments/reset_ablation/report.py --out <campaign>`.", ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(dict(event="report_complete", report=str(out / "REPORT.md"))), flush=True)


if __name__ == "__main__":
    main()
