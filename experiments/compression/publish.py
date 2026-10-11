"""Publish dense 3M-5M students, selecting candidates by whole validation only."""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "results/accuracy/custom_performance_candidates_20261004/selected_benchmark_final_20261005"
GOALS = {"MNIST": 0.99, "NMNIST": 0.98, "CIFAR10DVS": 0.75, "DVSGesture": 0.95}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise AssertionError("Do not overwrite an existing published bundle")
    baseline = read(BASELINE / "benchmark_manifest.json")
    selected, comparisons = [], {}
    # Validate every candidate and audit before writing the deliverable.
    for old in baseline["entries"]:
        dataset = old["dataset"]
        candidates = []
        for trial in args.trials:
            directory = trial.resolve() / dataset
            if not directory.exists():
                continue
            if not (directory / "accuracy.json").exists():
                raise AssertionError(f"Candidate has not completed its full test: {directory}")
            row = read(directory / "accuracy.json")
            candidates.append((directory, row))
        if not candidates:
            raise AssertionError(f"Missing completed compression: {dataset}")
        directory, result = max(candidates, key=lambda pair: (
            pair[1]["validation"]["top1_accuracy"], -pair[1]["validation"]["loss"]))
        comparisons[dataset] = [{"condition": str(path), "validation": row["validation"],
                                 "selected": path == directory, "parameters": row["parameters"],
                                 "test": row["test"]} for path, row in candidates]
        provenance = read(directory / "provenance.json")
        audit = read(directory / "training_constraints_audit.json")
        replay = read(directory / "replay.json")
        if not audit["passed"] or not replay["exact_replay"]:
            raise AssertionError("Native TRAIN audit or independent full TEST replay missing")
        checkpoint_sha = sha(directory / "best.pth")
        if checkpoint_sha != result["checkpoint_sha256"] or checkpoint_sha != audit["checkpoint_sha256"]:
            raise AssertionError("Audits reference a different checkpoint")
        if replay["test"] != result["test"] or replay["checkpoint_sha256"] != checkpoint_sha:
            raise AssertionError("Independent replay result differs")
        if not 3_000_000 <= result["parameters"]["total"] <= 5_000_000:
            raise AssertionError("Physical dense parameter budget failed")
        if sha(old["checkpoint"]) != old["checkpoint_sha256"]:
            raise AssertionError("Original baseline checkpoint changed")
        selected.append((old, directory, result, provenance))

    out.mkdir(parents=True)
    shutil.copy2(Path(__file__).with_name("reproduce_selected.py"), out / "reproduce.py")
    entries = []
    for old, directory, result, provenance in selected:
        dataset = old["dataset"]
        destination = out / "conditions" / dataset
        destination.mkdir(parents=True)
        names = ["best.pth", "config.json", "provenance.json", "pruning.json", "accuracy.json",
                 "replay.json", "training_constraints_audit.json", "calibration_split.json",
                 "validation_probe.json", "metrics.jsonl", "status.json"]
        for name in names:
            shutil.copy2(directory / name, destination / name)
        runtime = out / Path(provenance["source_root"]).name
        for name, expected in provenance["source_sha256"].items():
            original = Path(provenance["source_root"]) / name
            if sha(original) != expected:
                raise AssertionError("Original model runtime changed")
            target = runtime / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(original, target)
            if sha(target) != expected:
                raise AssertionError("Frozen runtime copy is incorrect")
        algorithms = [directory / "compression_runtime", directory.parent / "compression_source_v1"]
        algorithm = next((path for path in algorithms if all(
            (path / name).exists() and sha(path / name) == expected
            for name, expected in provenance["compression_source_sha256"].items())), None)
        if algorithm is None:
            raise AssertionError("Exact compression source snapshot is missing")
        algorithm_target = destination / "compression_runtime"
        algorithm_target.mkdir()
        for name in provenance["compression_source_sha256"]:
            shutil.copy2(algorithm / name, algorithm_target / name)
        frozen_baseline = read(Path(old["original_result"]))
        write(destination / "baseline_accuracy.json", frozen_baseline)
        files = [destination / name for name in names]
        files += [algorithm_target / name for name in provenance["compression_source_sha256"]]
        files += [runtime / name for name in provenance["source_sha256"]]
        files += [destination / "baseline_accuracy.json", out / "reproduce.py"]
        entries.append(dict(
            dataset=dataset, selected_condition=str(directory), source_root=runtime.name,
            runtime_imports=list(provenance["runtime_imports"]),
            checkpoint=str((destination / "best.pth").relative_to(out)),
            checkpoint_sha256=result["checkpoint_sha256"],
            config=str((destination / "config.json").relative_to(out)),
            accuracy=str((destination / "accuracy.json").relative_to(out)),
            parameters=result["parameters"], teacher_parameters=result["teacher_parameters"],
            teacher_checkpoint=old["checkpoint"], teacher_checkpoint_sha256=old["checkpoint_sha256"],
            split_sha256=provenance["split_sha256"], test_rng_seed=old["test_rng_seed"],
            input_profile=provenance["input_profile"],
            fp32_goal=GOALS[dataset], fp32_goal_met=result["test"]["float32"]["top1_accuracy"] >= GOALS[dataset],
            bundle_sha256={str(path.relative_to(out)): sha(path) for path in files}))
    manifest = dict(at=datetime.now(timezone.utc).isoformat(),
                    selection="Maximum complete validation accuracy, then minimum validation loss; no TEST ranking",
                    parameter_accounting="All physical dense parameters, including E/I banks and classifier; no masks",
                    baseline_manifest_sha256=sha(BASELINE / "benchmark_manifest.json"),
                    all_parameter_goals_met=True, all_fp32_goals_met=all(entry["fp32_goal_met"] for entry in entries),
                    entries=entries)
    write(out / "compression_manifest.json", manifest)
    write(out / "candidate_comparison.json", comparisons)
    lines = ["# 已训练 SNN 的 3M–5M 压缩结果", "",
             "四个学生模型来自已训练检查点的结构化压缩与蒸馏恢复。参数量统计整个网络，包括两组非负 E/I 卷积权重、隐藏层、分类头、偏置及可训练 PLIF 参数。", "",
             "| 数据集 | 原参数 M | 压缩后 M | 倍数 | 原 FP32 % | 压缩 FP32 % | 变化 pp | 2-bit % | 4-bit % | TEST 样本 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for old, directory, result, provenance in selected:
        before = read(Path(old["original_result"]))["test"]["float32"]["top1_accuracy"]
        scores = result["test"]
        lines.append(f"| {old['dataset']} | {result['teacher_parameters']['total']/1e6:.3f} | "
                     f"{result['parameters']['total']/1e6:.3f} | {result['compression_ratio']:.2f}× | "
                     f"{before*100:.2f} | {scores['float32']['top1_accuracy']*100:.2f} | "
                     f"{(scores['float32']['top1_accuracy']-before)*100:+.2f} | "
                     f"{scores['uint2']['top1_accuracy']*100:.2f} | {scores['uint4']['top1_accuracy']*100:.2f} | "
                     f"{scores['float32']['samples']} |")
    lines.extend(["", "## 结构与输入", "",
                  "卷积均为 5 个非负 E/I 3×3 层，前 4 层后进行 2×2 池化，最后采用 2×2 自适应池化读出。", "",
                  "| 数据集 | 卷积输出通道 | 隐藏 Linear 宽度 | T | 分辨率 | 最佳轮次 |",
                  "|---|---|---|---:|---:|---:|"])
    for old, directory, result, provenance in selected:
        cfg = read(directory / "config.json")
        profile = provenance["input_profile"]
        lines.append(f"| {old['dataset']} | {cfg['custom_channels']} | {cfg['hidden_widths']} | "
                     f"{profile['steps']} | {profile['resolution']} | {result['best_epoch']} |")
        original_profile = read(Path(old["config_snapshot"]))["dataset_profiles"][old["dataset"]]
        if profile != original_profile:
            lines.append(f"\n{old['dataset']} 输入条件适配：原 {original_profile}，压缩后 {profile}；参数减少不代表输入条件相同时的实测延迟加速。\n")
    lines.extend(["", "## 恢复训练与约束", "",
        "训练集的协方差用于通道选择和岭回归重构。重构后的等效有符号卷积拆为两组非负物理权重；隐藏层保留完整 WTA 分组及对应 BP/STDP 分配，并恢复原输入行权重和。",
        "TRAIN-only 512 样本用于初始重构统计、分类头恢复与 BN 统计重校准。之后缓存教师在整个 TRAIN 上的浮点 logits；目标为 0.5×交叉熵 + 0.5×KL×温度²，温度 2。",
        "MNIST 教师缓存每样本一次固定 Poisson 编码，学生继续随机 Poisson 编码；事件数据保持计数合帧、空间缩放及 log1p。没有训练量化，梯度和权重是 FP32，dropout 上限 0.2，关闭额外数据增强。CuDNN TF32 开启，与原实验一致。",
        "卷积及分类头使用 BP，不参与 WTA；隐藏层每次更新受原 WTA 和 BP/STDP 分配约束。MNIST/N-MNIST 的隐藏层分别为 STDP、BP；CIFAR/Gesture 的隐藏层保留 hybrid 分组。",
        "卷积后的神经元为原 PLIF，复制选中通道对应 BN 与 PLIF 参数；隐藏层保持 LIF 更新 mem=0.9×mem+current，发放后减去阈值，使用原 surrogate。指数稳态基准阈值 1，winner 增量 0.05，衰减 tau=20，VALID/TEST 冻结阈值适应；其余阈值范围等条件见各 config.json。",
        "推理的 uint2/uint4 只约束隐藏 Linear 权重，卷积与分类头保持浮点。量化为每输出行 max/(2^b−1) 的无符号均匀量化，零点 0，先取整和截断到 0…2^b−1，再反量化进行精度模拟；尚未测量整数硬件延迟。完整测试中的量化层列表和混淆矩阵见 accuracy.json。",
        "实际恢复训练的学习率与轮次以 provenance.json 为准：隐藏 BP 学习率为骨干的 0.5 倍，分类头为 5 倍；Adam、梯度裁剪 1、无 weight decay、cosine decay 下限 1e-6。config.json 中沿用教师的 init、lr、training_epochs、monitoring 默认字段不代表本次实际启动方式或预算。",
        "恢复预算 MNIST/N-MNIST 12 轮，CIFAR/Gesture 首版 30 轮。实时 metrics.jsonl 输出 loss/CE/KD、梯度、发放及 winner 数；从第 5 轮起允许 6 个验证精度未显著提高的轮次触发早停，NaN、死梯度与持续单类预测会提前中止。具体实际轮次与终止原因见 accuracy.json 和 metrics.jsonl。", "",
        "## 选择、验证与复现", "",
        "模型和压缩条件仅按完整验证集精度选择，精度相同按验证 loss 选择。TEST 不参与条件排序。所有候选的验证分数、完整测试结果和选择标记保存在 candidate_comparison.json；未采用的试验保留在原 trial 目录。",
        "所有选中模型已通过独立完整 FP32/2-bit/4-bit 重放，以及隔离副本上真实 TRAIN 更新审计。审计检查非 winner 的权重及其优化器动量不发生更新、BP 不写入 STDP 行、STDP 仅更新其 winner、非负权重与 FP32 梯度、原检查点不变。",
        "结果为单种子压缩实验，微小精度上升不构成统计显著改善。四个模型均满足 3M–5M 的物理参数目标；此前 FP32 精度门槛的实际通过状态见 compression_manifest.json，未通过的条件不能视为精度闭环。", "",
        "包内保存选中检查点、配置、模型源代码、压缩算法源代码与 SHA256 清单；reproduce.py 独立构建学生模型，无须重新训练教师。数据仍使用当前工作区 datasets。", "",
        "```bash",
        ".venv/bin/python results/accuracy/" + out.name + "/reproduce.py --dataset CIFAR10DVS --out /tmp/compressed_cifar_replay.json",
        "```", ""])
    (out / "README.md").write_text("\n".join(lines))
    print(json.dumps(dict(bundle=str(out), all_parameter_goals_met=True,
                          all_fp32_goals_met=manifest["all_fp32_goals_met"],
                          selected={entry["dataset"]: entry["selected_condition"] for entry in entries})), flush=True)


if __name__ == "__main__":
    main()
