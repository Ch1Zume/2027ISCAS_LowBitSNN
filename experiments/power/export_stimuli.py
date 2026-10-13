"""Export logical 512x512 inference stimuli; physical pulse timing is deliberately unset."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main(root, precision):
    folder = Path(root)/"NMNIST"/precision
    summary = json.loads((folder/"summary.json").read_text())
    layer = summary["layers"]["hidden_layers.hidden2"]
    assert layer["binary_input"] and layer["input_units"] == layer["neuron_units"] == 512
    inputs = np.load(folder/"hidden_layers_hidden2_input_packbits.npy", mmap_mode="r")
    outputs = np.load(folder/"hidden_layers_hidden2_output_packbits.npy", mmap_mode="r")
    labels = {}
    with (folder/"per_sample_layers.csv").open() as f:
        for row in csv.DictReader(f):
            if row["layer"] == "hidden_layers.hidden2":
                labels[int(row["sample_index"])] = int(row["label"])
    destination = folder/"macro_stimuli"
    destination.mkdir(exist_ok=True)
    fields = ["sample_index", "label", "view", "logical_step", "input_row_mask_hex",
              "algorithm_reference_fire_mask_hex", "input_spikes", "algorithm_firings", "SOP_including_zero_weights"]
    with (destination/"full_TEST_steps.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        total_in = total_out = total_sop = 0
        for n in range(summary["samples"]):
            for v in range(summary["views"]):
                for t in range(summary["steps_per_view"]):
                    i = int.from_bytes(inputs[n,v,t].tobytes(), "little")
                    o = int.from_bytes(outputs[n,v,t].tobytes(), "little")
                    ni, no = i.bit_count(), o.bit_count()
                    total_in += ni; total_out += no; total_sop += ni*512
                    writer.writerow(dict(sample_index=n, label=labels[n], view=v, logical_step=t,
                         input_row_mask_hex=f"{i:0128x}", algorithm_reference_fire_mask_hex=f"{o:0128x}",
                         input_spikes=ni, algorithm_firings=no, SOP_including_zero_weights=ni*512))
    assert total_in == layer["counters"]["input_nonzero"]
    assert total_out == layer["counters"]["fire"]
    assert total_sop == layer["SOP"]
    weight = np.load(folder/"hidden_weights_and_thresholds.npz", allow_pickle=False)
    key = "hidden2__"
    if precision != "float32":
        codes, scales = weight[key+"weight_codes"], weight[key+"weight_scales"].reshape(-1)
        with (destination/"weights_by_input_row.csv").open("w") as f:
            w = csv.writer(f); w.writerow(["input_row"]+[f"neuron_{i}" for i in range(512)])
            for i,row in enumerate(codes.T):
                w.writerow([i]+row.tolist())
    else:
        scales = np.ones(512)
    with (destination/"algorithm_neuron_parameters.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["neuron", "row_weight_scale", "bias", "threshold", "membrane_decay"])
        for i in range(512):
            w.writerow([i, scales[i], weight[key+"bias"][i], weight[key+"threshold"][i], layer["metadata"]["membrane_decay"]])
    counts = np.load(folder/"per_timestep_and_neuron_counts.npz", allow_pickle=False)
    firing = counts["hidden_layers.hidden2__neuron_fire"]/(summary["samples"]*summary["views"]*summary["steps_per_view"])
    input_rate = counts["hidden_layers.hidden2__input_nonzero_by_unit"]/(summary["samples"]*summary["views"]*summary["steps_per_view"])
    effective_weight = weight[key+"weight_effective"]
    with (destination/"per_neuron_activity_and_parameters.csv").open("w") as f:
        writer=csv.writer(f)
        writer.writerow(["neuron", "firing_activity", "weight_nonzero_count", "weight_sum", "weight_scale", "threshold", "bias", "mean_algorithm_input_current"])
        for i in range(512):
            writer.writerow([i, firing[i], int((effective_weight[i]!=0).sum()), float(effective_weight[i].sum()),
                             scales[i], weight[key+"threshold"][i], weight[key+"bias"][i],
                             float(effective_weight[i]@input_rate+weight[key+"bias"][i])])
    with (destination/"representative_columns.csv").open("w") as f:
        writer=csv.writer(f)
        writer.writerow(["target_firing_quantile", "target_firing_activity", "neuron", "actual_firing_activity", "weight_scale", "threshold", "bias"])
        for q in (0,.25,.5,.75,.9,.95,.99,1):
            target=np.quantile(firing,q); i=int(np.abs(firing-target).argmin())
            writer.writerow([q,target,i,firing[i],scales[i],weight[key+"threshold"][i],weight[key+"bias"][i]])
    contract = dict(dataset="NMNIST", precision=precision, layer="hidden2", shape=[512,512],
         samples=summary["samples"], steps_per_sample=4, pulse_timing_ns=None, hardware_RST_sequence=None,
         mask_format="128 hexadecimal digits; neuron/input row 0 is least significant bit",
         input_semantics="every asserted row represents one input spike in that logical step; repeated asserted rows are repeated spikes",
         output_semantics="reference algorithm firings; not measured soma outputs or physical RST signals",
         sample_boundary="zero initial algorithm state for each sample; physical initialization and end restoration must be specified separately",
         reconstruction="m_pre=0.9*m_post_previous + scale[neuron]*sum(code[neuron,input]*spike[input]) + bias[neuron]",
         reset="all threshold crossings emit one spike and subtract the threshold; WTA does not suppress losers",
         voltage_mapping="unresolved; per-neuron scale/bias/threshold values are algorithm units, not V_REF/RBL volts",
         counted_input_spikes=total_in, counted_algorithm_firings=total_out, counted_SOP=total_sop,
         source_checkpoint_sha256=summary["checkpoint_sha256"])
    (destination/"stimulus_contract.json").write_text(json.dumps(contract, indent=2)+"\n")
    print(json.dumps(dict(precision=precision, samples=summary["samples"], steps=summary["samples"]*4,
                         counted_input_spikes=total_in, counted_firings=total_out)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--precision", choices=("uint4","uint2","float32"), required=True)
    a = p.parse_args(); main(a.out, a.precision)
