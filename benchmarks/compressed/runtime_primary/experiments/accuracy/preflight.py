"""Exercise one native training batch per model/dataset before a campaign.

This is a compatibility/memory check, not an accuracy evaluation. Only training
samples are used; the temporary models are discarded, never warm-started.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from snn.inference_quant import quantized_hidden_weights
from .config import load_config
from .data import build_loaders, prepare_batch
from .models import AccuracySNN
from .run import calibrate_hidden, seed_everything


def check(config, output):
    cfg = load_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("Native CUDA access is required for memory qualification")
    result = {"evidence_scope": "functional_check", "device": torch.cuda.get_device_name(0),
              "configuration": str(Path(config).resolve()), "datasets": {}, "checks": []}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for dataset in cfg["datasets"]:
        loaders = build_loaders(dataset, cfg)
        result["datasets"][dataset] = {"counts": [len(loader.dataset) for loader in loaders[:3]],
                                        "split": str(loaders[3])}
        for family in cfg["models"]:
            seed_everything(cfg["seed"])
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            model = AccuracySNN(family, dataset, cfg, "exponential").cuda()
            if family != "custom":
                model.initialize_official()
            calibration = calibrate_hidden(model, loaders[0], dataset, torch.device("cuda"))
            data, labels = next(iter(loaders[0]))
            frames = prepare_batch(data, dataset, torch.device("cuda"))
            labels = labels.cuda()
            model.train()
            model.reset_membranes()
            loss = F.cross_entropy(model(frames), labels)
            assert torch.isfinite(loss), (family, dataset, "nonfinite loss")
            loss.backward()
            gradients = [p.grad.detach().norm() for p in model.backbone.parameters() if p.grad is not None]
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            gradient_norm = torch.stack(gradients).norm().item()
            assert gradient_norm > 1e-12, (family, dataset, "dead backbone gradient")
            inference = {}
            model.eval()
            with torch.no_grad():
                for bits in (2, 4):
                    model.reset_membranes()
                    with quantized_hidden_weights(model, bits):
                        logits = model(frames)
                        assert torch.isfinite(logits).all(), (family, dataset, bits)
                        inference[str(bits)] = list(logits.shape)
            size = model.describe_size()
            torch.cuda.synchronize()
            record = {"model": family, "dataset": dataset, "frames": list(frames.shape),
                      "train_loss": loss.item(), "backbone_gradient_norm": gradient_norm,
                      "quantized_output_shapes": inference, "hidden_calibration": calibration,
                      "width_sum": size["width_sum"], "parameters": size["parameters"],
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "elapsed_seconds": time.monotonic()-started, "status": "passed"}
            result["checks"].append(record)
            output.write_text(json.dumps(result, indent=2)+"\n")
            print(json.dumps(record), flush=True)
            del model, frames, loss, gradients, logits, data, labels
            gc.collect()
        del loaders
    result["status"] = "passed"
    output.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    check(args.config, args.output)
