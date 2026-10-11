"""Frozen own-run teachers used only to supervise TRAIN; inference is one SNN."""
import copy
import hashlib
import math
from pathlib import Path

import torch
from torch.nn import functional as F


def validate_distillation_config(settings):
    if settings is None:
        return
    if not isinstance(settings, dict) or set(settings) != {"temperature", "hard_weight", "teachers"}:
        raise ValueError("own_distillation requires temperature, hard_weight and teachers")
    for name in ("temperature", "hard_weight"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid own_distillation {name}")
    if settings["temperature"] <= 0 or not 0 < settings["hard_weight"] < 1:
        raise ValueError("Distillation requires positive temperature and hard_weight in (0,1)")
    teachers = settings["teachers"]
    if not isinstance(teachers, list) or not teachers:
        raise ValueError("Distillation requires own teachers")
    paths = []
    for teacher in teachers:
        if not isinstance(teacher, dict) or set(teacher) != {"checkpoint", "sha256", "weight"}:
            raise ValueError("Each own teacher requires checkpoint, sha256 and weight")
        if not isinstance(teacher["checkpoint"], str) or not Path(teacher["checkpoint"]).is_absolute():
            raise ValueError("Own teacher checkpoint must be an absolute path")
        digest = teacher["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid own teacher sha256")
        weight = teacher["weight"]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("Own teacher weight must be positive and finite")
        paths.append(teacher["checkpoint"])
    if len(set(paths)) != len(paths) or not math.isclose(sum(t["weight"] for t in teachers), 1., abs_tol=1e-9):
        raise ValueError("Own teacher paths must be unique and weights sum to one")


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_teacher_metadata(meta, cfg, dataset, mode, split):
    origin = meta.get("initialization", {})
    if (meta.get("model") != "custom" or not (
            origin.get("kind") == "from_scratch" or
            origin.get("kind") == "own_run_adaptation" and origin.get("from_scratch_origin") is True)):
        raise ValueError("Teacher must be this dataset's own custom model originating from scratch")
    if meta.get("final_refit") or meta.get("sample_limit") is not None or meta.get("step_override") is not None:
        raise ValueError("Teacher must preserve held-out VALID and full TRAIN protocol")
    if meta["dataset"]["name"] != dataset or meta["homeostasis"] != mode or meta["seed"] != cfg["seed"]:
        raise ValueError("Own teacher dataset, homeostasis or split seed differs")
    from .config import dataset_spec
    expected = dataset_spec(dataset, cfg)
    if any(meta["dataset"][key] != getattr(expected, key) for key in ("steps", "resolution", "channels", "classes", "coding")):
        raise ValueError("Own teacher input protocol differs")
    old = meta["config"]
    if old.get("dataset_profiles", {}).get(dataset, {}).get("event_transform", "none") != cfg.get("dataset_profiles", {}).get(dataset, {}).get("event_transform", "none"):
        raise ValueError("Own teacher event transform differs")
    if not split or not meta.get("split_manifest") or file_digest(split) != file_digest(meta["split_manifest"]):
        raise ValueError("Own teacher must use identical TRAIN/VALID/TEST partition manifest")
    if Path(old["data_dir"]).resolve() != Path(cfg["data_dir"]).resolve():
        raise ValueError("Own teacher data root differs")


class OwnTeacherBundle:
    # Deliberately not nn.Module: teachers never join student parameters,
    # checkpoints, neuron accounting, quantization or inference.
    def __init__(self, cfg, dataset, mode, split, device, model_factory=None):
        settings = cfg.get("own_distillation")
        validate_distillation_config(settings)
        if settings is None:
            raise ValueError("Missing own_distillation configuration")
        if model_factory is None:
            from .models import AccuracySNN
            model_factory = AccuracySNN
        self.settings = copy.deepcopy(settings)
        self.teachers, records = [], []
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        try:
            for item in settings["teachers"]:
                path = Path(item["checkpoint"])
                if file_digest(path) != item["sha256"]:
                    raise ValueError("Own teacher checkpoint SHA256 mismatch")
                saved = torch.load(path, map_location="cpu", weights_only=True)
                meta = saved["metadata"]
                validate_teacher_metadata(meta, cfg, dataset, mode, split)
                teacher = model_factory("custom", dataset, copy.deepcopy(meta["config"]), mode).to(device)
                teacher.load_state_dict(saved["model"])
                for name, module in teacher.named_modules():
                    if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)) or hasattr(module, "eligible_rows"):
                        weight = module.weight
                        if weight.dtype != torch.float32 or not torch.isfinite(weight).all() or (weight < 0).any():
                            raise ValueError(f"Own teacher contains invalid physical synapses: {name}")
                teacher.requires_grad_(False)
                teacher.eval()
                self.teachers.append(teacher)
                records.append({"checkpoint": str(path), "sha256": item["sha256"],
                                "weight": item["weight"], "source_epoch": saved["epoch"],
                                "initialization": meta["initialization"],
                                "split_sha256": file_digest(split), "frozen": True})
                del saved
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng:
                torch.cuda.set_rng_state_all(cuda_rng)
        self.provenance = {"kind": "own_from_scratch_train_only_distillation", "teachers": records,
                           "temperature": settings["temperature"], "hard_weight": settings["hard_weight"],
                           "inference_teachers": 0, "student_size_unchanged": True}

    @torch.no_grad()
    def probabilities(self, frames):
        mixture = None
        for teacher, item in zip(self.teachers, self.settings["teachers"]):
            teacher.eval()
            teacher.reset_membranes()
            value = (teacher(frames) / self.settings["temperature"]).softmax(1)
            teacher.reset_membranes()
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite own teacher probabilities")
            weighted = value * item["weight"]
            mixture = weighted if mixture is None else mixture + weighted
        return mixture

    def loss(self, logits, labels, probabilities):
        hard = F.cross_entropy(logits, labels)
        temperature = self.settings["temperature"]
        soft = F.kl_div(F.log_softmax(logits / temperature, 1), probabilities.detach(),
                        reduction="batchmean") * temperature ** 2
        weight = self.settings["hard_weight"]
        total = weight * hard + (1-weight) * soft
        return total, {"hard_ce_loss": hard.item(), "soft_kl_scaled_loss": soft.item(),
                       "loss_kind": "hard_ce_and_own_teacher_kl"}
