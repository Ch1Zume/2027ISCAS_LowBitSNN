"""Audit persisted activity and produce CSV, a report, and reproducible figures."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024**2), b""):
            h.update(b)
    return h.hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def main(root):
    root = Path(root)
    contract = json.loads((root/"contract.json").read_text())
    summaries, rows, audits = {}, [], []
    for dataset in contract["datasets"]:
        for precision in contract["precisions"]:
            folder = root/dataset/precision
            if not (folder/"summary.json").exists():
                continue
            s = json.loads((folder/"summary.json").read_text())
            assert all(s["validation"].values()), f"Validation incomplete: {folder}"
            summaries[(dataset, precision)] = s
            counts = np.load(folder/"per_timestep_and_neuron_counts.npz", allow_pickle=False)
            per_sample = {}
            with (folder/"per_sample_layers.csv").open() as f:
                for row in csv.DictReader(f):
                    n = int(row["sample_index"])
                    v = per_sample.setdefault(row["layer"], {})
                    assert n not in v
                    v[n] = row
            for name, layer in s["layers"].items():
                assert len(per_sample[name]) == s["samples"]
                for key,total in layer["counters"].items():
                    a = counts[name+"__"+key]
                    assert a.shape == (s["samples"], s["views"], s["steps_per_view"])
                    assert int(a.sum()) == total
                    sample_counts = a.sum((1,2))
                    assert all(int(per_sample[name][n][key]) == int(sample_counts[n]) for n in range(s["samples"]))
                if layer["role"] == "hidden":
                    # Decode every retained output bit and independently recount spikes.
                    for kind,units in (("output", layer["neuron_units"]), ("input", layer["input_units"])):
                        p = folder/(name.replace('.', '_')+"_"+kind+"_packbits.npy")
                        if kind == "input" and not layer["binary_input"]:
                            assert not p.exists()
                            continue
                        packed = np.load(p, mmap_mode="r")
                        bits = np.unpackbits(packed, axis=-1, count=units, bitorder="little")
                        key = "fire" if kind == "output" else "input_nonzero"
                        assert np.array_equal(bits.sum(-1), counts[name+"__"+key])
                    if layer["binary_input"]:
                        assert layer["SOP"] == layer["counters"]["input_nonzero"]*layer["neuron_units"]
                    else:
                        assert layer["SOP"] is None
                pin, pout = layer["input_activity"], layer["firing_activity"]
                sample_rows = list(per_sample[name].values())
                reset_total = sum(int(r["software_spike_reset"]) for r in sample_rows)
                boundary_total = sum(int(r["logical_boundary_initializations"]) for r in sample_rows)
                end_total = (sum(int(r["end_nonzero_membrane"]) for r in sample_rows)
                             if layer["metadata"].get("end_membrane_observed") else None)
                rows.append(dict(dataset=dataset, precision=precision, layer=name, role=layer["role"],
                     TEST_samples=s["samples"], steps_per_view=s["steps_per_view"], views=s["views"],
                     accuracy_percent=100*s["score"]["top1_accuracy"],
                     input_units=layer["input_units"], neuron_units=layer["neuron_units"],
                     input_kind="binary_spikes" if layer["binary_input"] else "nonbinary_activities",
                     input_activity=pin, firing_activity=pout,
                     input_nonzero_total=layer["counters"].get("input_nonzero"),
                     firing_total=layer["counters"].get("fire"), SOP_total=layer["SOP"],
                     software_spike_reset_total=reset_total, logical_boundary_initializations_total=boundary_total,
                     end_nonzero_membrane_total=end_total,
                     effective_SOP_total=layer["effective_SOP"],
                     mean_input_nonzero_per_inference=layer["per_sample"].get("input_nonzero", {}).get("mean"),
                     mean_firings_per_inference=layer["per_sample"].get("fire", {}).get("mean"),
                     P95_firings_per_inference=layer["per_sample"].get("fire", {}).get("p95"),
                     P95_simultaneous_firings=layer["simultaneous_per_step"].get("fire", {}).get("p95"),
                     silent_neuron_fraction=layer.get("silent_neuron_fraction"),
                     hardware_RST_total=None, measured_energy_pJ=None))
            audits.append(dict(dataset=dataset, precision=precision, all_sample_time_totals_match=True,
                         all_hidden_packed_traces_recount_match=True, all_binary_hidden_SOP_counts_match=True,
                         exact_full_TEST_score=True, state_unchanged=True))
    fields = list(rows[0])
    with (root/"layer_activity.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    hidden = [r for r in rows if r["role"] == "hidden"]
    with (root/"hidden_activity.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(hidden)
    primary = summaries[("NMNIST", "uint4")]["layers"]["hidden_layers.hidden2"]
    baseline_pin, baseline_pout = primary["input_activity"], primary["firing_activity"]
    supplemental = []
    for r in hidden:
        if r["precision"] != "uint4" or r["dataset"] == "NMNIST":
            continue
        matched = r["input_units"] == 512 and r["neuron_units"] == 512 and r["input_kind"] == "binary_spikes"
        supplemental.append(dict(dataset=r["dataset"], layer=r["layer"], input_kind=r["input_kind"],
            input_activity=r["input_activity"], firing_activity=r["firing_activity"],
            matches_512x512_binary_interface=matched,
            input_activity_ratio_to_NMNIST_hidden2=r["input_activity"]/baseline_pin if matched else None,
            firing_activity_ratio_to_NMNIST_hidden2=r["firing_activity"]/baseline_pout,
            row_event_ratio_per_inference=r["mean_input_nonzero_per_inference"]/primary["per_sample"]["input_nonzero"]["mean"] if matched else None,
            software_firing_reset_ratio_per_inference=r["mean_firings_per_inference"]/primary["per_sample"]["fire"]["mean"] if matched else None,
            neuron_update_opportunity_ratio_per_inference=(r["neuron_units"]*r["steps_per_view"]*r["views"]/(512*4)) if matched else None,
            logical_boundary_initialization_ratio_per_inference=r["neuron_units"]*r["views"]/512 if matched else None,
            effective_positive_weight_SOP_ratio_per_inference=(r["effective_SOP_total"]/r["TEST_samples"]/(primary["effective_SOP"]/10000)) if matched else None,
            interpretation="different frozen model/task; activity comparison, not a universal power correction factor"))
    write(root/"supplemental_activity_comparison.json", supplemental)
    pending = [f"{d}/{p}" for d in contract["datasets"] for p in contract["precisions"] if (d,p) not in summaries]
    audit = dict(generated_at=datetime.now(timezone.utc).isoformat(), completed_conditions=len(summaries),
         expected_conditions=len(contract["datasets"])*len(contract["precisions"]), pending=pending,
         full_campaign_complete=not pending, condition_audits=audits, hardware_power_measured=False,
         report_driver_sha256=sha(Path(__file__)))
    write(root/"completion_audit.json", audit)
    text = ["SNN inference activity for neuron power characterization", "",
        f"Completed conditions: {len(summaries)}; pending: {', '.join(pending) or 'none'}",
        "Primary: frozen lut_final N-MNIST, hidden2, 512 inputs x 512 neurons, uint4, full 10000 TEST samples.",
        "Each sample has 4 logical steps. State is cleared at the sample boundary. No training or calibration.", "",
        "Primary measured algorithmic activity:",
        f"Input spike probability per row per logical step: {baseline_pin:.12f} ({baseline_pin*100:.6f}%).",
        f"Firing probability per neuron per logical step: {baseline_pout:.12f} ({baseline_pout*100:.6f}%).",
        f"Total input spikes: {primary['counters']['input_nonzero']}; total output spikes: {primary['counters']['fire']}.",
        f"Addressed SOPs including zero weights: {primary['SOP']}; nonzero-weight effective SOPs: {primary['effective_SOP']}.",
        f"Mean input spikes/inference: {primary['per_sample']['input_nonzero']['mean']:.6f}.",
        f"Mean output spikes/inference: {primary['per_sample']['fire']['mean']:.6f}.",
        f"P95 output spikes/inference: {primary['per_sample']['fire']['p95']:.6f}.",
        f"Silent-neuron fraction over full TEST: {primary['silent_neuron_fraction']:.6f}.", "",
        "Timing is unconfirmed by the user. Report logical-step activity; hardware Hz are not calculated.",
        "A future Hz conversion requires actual scheduling, active time and idle time.", "",
        "Use in circuit simulations:",
        "1. Use exact hidden2 input/output bit traces, frozen uint weight codes, per-neuron scales/biases/thresholds.",
        "2. Preserve 4-step sample boundaries; replay many full samples instead of extending one sample.",
        "3. Characterize columns across firing and weight distributions; one average column is not sufficient.",
        "4. Software subtractive reset is counted separately from initialization and remaining membrane state.",
        "5. Verify circuit reset/threshold/scale/bias behavior before treating these traces as physical RST events.",
        "6. Integrate supplies over the complete physical window, including final restoration and peripheral costs.", "",
        "Supplemental conditions:"]
    for r in supplemental:
        text.append(f"{r['dataset']} {r['layer']}: input activity {r['input_activity']*100:.6f}%, "
                    f"firing activity {r['firing_activity']*100:.6f}%; input kind {r['input_kind']}.")
    text += ["", "Nonbinary activity is not a physical spike count. Its physical SOP and input Hz remain unset.",
        "Winners gate learning; losing neurons' spikes are retained. Winner count is not firing/reset count.",
        "Neuron updates are algorithmic opportunities; physical SA gating is not simulated.",
        "Software spike resets, sample-boundary initialization and physical RBL recharge must not be conflated.",
        "Membrane traces use algorithm units; no RBL voltage or reset energy is inferred.",
        "Supplemental activity ratios include changes in model, weights, thresholds and task. They are not causal dataset-only corrections.",
        "Static power and layout area do not follow from dataset activity; they require their own circuit/layout characterization.",
        "Only TEST samples are evaluated. Original archive/adapters can also download TRAIN bytes; none are used for fitting or calibration.",
        "Voltage/clock/timing mapping, transistor/RTL energy, macro static/dynamic power and pJ/SOP remain unmeasured.",
        "Outputs: layer_activity.csv, hidden_activity.csv, per-condition per_sample_layers.csv and per_timestep_and_neuron_counts.npz.",
        "Packed trace bitorder is little; array axes are sample, view, step, packed input/output units.",
        "Representative traces are nearest choices from deterministic 1-percent input-activity bins, not exact quantile samples."]
    (root/"REPORT.txt").write_text("\n".join(text)+"\n")
    plot(root, summaries)
    hashes = {str(p.relative_to(root)):sha(p) for p in root.rglob("*")
              if p.is_file() and p.name not in ("artifacts_manifest.json", "status.json", "campaign.log")
              and p.suffix not in (".log", ".tmp") and "smoke" not in p.parts}
    write(root/"artifacts_manifest.json", dict(sha256=hashes, completion_audit_sha256=sha(root/"completion_audit.json")))
    print(json.dumps(dict(completed=len(summaries), pending=pending, primary_input_activity=baseline_pin,
                          primary_firing_activity=baseline_pout)), flush=True)


def plot(root, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10, "pdf.fonttype":42, "ps.fonttype":42})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), layout="constrained")
    folder = root/"NMNIST/uint4"
    counts = np.load(folder/"per_timestep_and_neuron_counts.npz", allow_pickle=False)
    key = "hidden_layers.hidden2"
    for metric, label, color in (("input_nonzero", "Input row", "#2563eb"), ("fire", "Output neuron", "#dc2626")):
        rates = counts[key+"__"+metric].mean((0,1))/512*100
        axes[0].plot(np.arange(1,len(rates)+1), rates, "o-", label=label, color=color)
    axes[0].set(xlabel="Logical step within one inference", ylabel="Activity (%)", title="N-MNIST uint4: step dependence")
    axes[0].set_xticks([1,2,3,4]); axes[0].legend(frameon=False)
    colors = {"uint4":"#2563eb", "uint2":"#059669", "float32":"#7c3aed"}
    for precision, color in colors.items():
        p = root/"NMNIST"/precision/"per_timestep_and_neuron_counts.npz"
        if p.exists():
            a = np.load(p, allow_pickle=False)[key+"__neuron_fire"]/(10000*4)*100
            ordered = np.sort(a)
            axes[1].step(ordered, np.arange(1,len(a)+1)/len(a), where="post", label=precision, color=color)
    axes[1].set(xlabel="Per-neuron firing activity (%)", ylabel="Fraction of neurons", title="N-MNIST: neuron distribution")
    axes[1].legend(frameon=False)
    labels, values = [], []
    for dataset in ("NMNIST", "MNIST", "DVSGesture", "CIFAR10DVS"):
        if (dataset,"uint4") in summaries:
            s = summaries[(dataset,"uint4")]
            name = "hidden_layers.hidden2" if dataset in ("NMNIST","MNIST") else "hidden_layers.hidden1"
            labels.append(dataset+"\n"+name.split('.')[-1])
            values.append(s["layers"][name]["firing_activity"]*100)
    axes[2].bar(labels, values, color=["#2563eb"]+["#94a3b8"]*(len(values)-1))
    axes[2].set(ylabel="Firing activity (%)", title="uint4: different model/task conditions")
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(root/("activity_summary."+extension), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args().out)
