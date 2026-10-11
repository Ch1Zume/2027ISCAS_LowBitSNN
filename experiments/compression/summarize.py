"""Report completed compression trials; never infer accuracy from training logs."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    benchmark = root / "results/accuracy/custom_performance_candidates_20261004/selected_benchmark_final_20261005"
    manifest = json.loads((benchmark / "benchmark_manifest.json").read_text())
    lines = ["# 已训练 SNN 的 3M–5M 压缩实验", "",
             "参数统计包含整个网络的所有密集参数，两组非负 E/I 卷积权重和分类头均计入。原检查点保留。",
             "只用训练集完成通道选择、重构、分类头恢复及教师蒸馏；按完整验证集选模，再运行完整测试集。", "",
             "| 数据集 | 原参数 M | 压缩后参数 M | 压缩倍数 | 原 FP32 % | 压缩 FP32 % | 变化 pp | 2-bit % | 4-bit % | 选中轮次 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    results, pending = {}, []
    for entry in manifest["entries"]:
        name = entry["dataset"]
        path = args.directory / name / "accuracy.json"
        if not path.exists():
            pending.append(name)
            continue
        row = json.loads(path.read_text())
        if not 3_000_000 <= row["parameters"]["total"] <= 5_000_000:
            raise AssertionError(f"Dense parameter budget failed: {name}")
        old = json.loads(Path(entry["original_result"]).read_text())["test"]["float32"]
        if row["test"]["float32"]["samples"] != old["samples"]:
            raise AssertionError("Baseline and compressed full test sizes differ")
        baseline = 100 * old["top1_accuracy"]
        accuracy = 100 * row["test"]["float32"]["top1_accuracy"]
        lines.append(f"| {name} | {row['teacher_parameters']['total']/1e6:.3f} | "
                     f"{row['parameters']['total']/1e6:.3f} | {row['compression_ratio']:.2f}× | "
                     f"{baseline:.2f} | {accuracy:.2f} | {accuracy-baseline:+.2f} | "
                     f"{100*row['test']['uint2']['top1_accuracy']:.2f} | "
                     f"{100*row['test']['uint4']['top1_accuracy']:.2f} | {row['best_epoch']} |")
        results[name] = row | {"baseline_fp32": old, "accuracy_change_pp": accuracy-baseline}
    lines.extend(["", "输入 T、分辨率与编码保持各自教师检查点设置；浮点训练，隐藏层使用原 WTA/BP/STDP 规则与指数稳态阈值。",
                  "推理仅量化隐藏 Linear 权重；卷积和分类头仍使用浮点。结果为单种子压缩尝试，不代表多种子统计。", ""])
    if pending:
        lines.append("尚未完成完整测试的条件：" + "、".join(pending) + "。")
    lines.extend(["", "各条件目录包含 config.json、pruning.json、provenance.json、metrics.jsonl、best.pth 和 accuracy.json。",
                  "campaign.json 与各条件 status.json 保存执行进度；已完成条件可通过 --resume --evaluate-only 独立重放。", ""])
    (args.directory / "README.md").write_text("\n".join(lines))
    (args.directory / "summary.json").write_text(json.dumps({"results": results, "pending": pending}, indent=2) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
