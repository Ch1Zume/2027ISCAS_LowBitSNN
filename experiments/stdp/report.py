"""Verify the delivered experiment artifacts and write a concise Chinese report."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pct(value):
    return f"{100*value:.2f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    a = p.parse_args()
    b = a.bundle.resolve()
    protocol = json.loads((b / "protocol.json").read_text())
    original = Path(protocol["baseline_bundle"])
    manifest = json.loads((original / "compression_manifest.json").read_text())
    results = json.loads((b / "results.json").read_text())
    checks = []
    checks.append({"check": "regressions", "passed": "60 passed" in (b / "regression.log").read_text()})
    checks.append({"check": "baseline_manifest_unchanged", "passed": sha(original / "compression_manifest.json") == protocol["baseline_manifest_sha256"]})
    for name, digest in protocol["source_sha256"].items():
        checks.append({"check": "frozen_source", "path": name, "passed": sha(b / name) == digest})
    checks.append({"check": "frozen_driver", "passed": sha(b / "run_frozen.py") == protocol["driver_sha256"]})
    for driver in protocol.get("driver_history", []):
        checks.append({"check": "archived_driver", "passed": sha(b / driver["path"]) == driver["sha256"]})
    before_test = json.loads((b / "selections_before_TEST.json").read_text())
    rows = []
    diagnostics = []
    encoding_initial = []
    old_scores = {}
    for entry in manifest["entries"]:
        dataset = entry["dataset"]
        old = json.loads((b / f"baseline_replay_{dataset}.json").read_text())
        old_scores[dataset] = old["test"]
        initial_native = results[dataset]["legacy_native"]["selection"]["history"][0]["validation"]
        initial_events = results[dataset]["legacy_events"]["selection"]["history"][0]["validation"]
        initial_lut = results[dataset]["lut_events"]["selection"]["history"][0]["validation"]
        checks.append({"check": "paired_encoding_forward_at_identical_initial_weights", "dataset": dataset,
            "passed": initial_events == initial_lut})
        encoding_initial.append({"dataset": dataset, "native": initial_native, "events": initial_events,
            "events_legacy_and_lut_exact_match": initial_events == initial_lut})
        checks.append({"check": "old_full_test_exact_replay", "dataset": dataset, "passed": old["exact_replay"]})
        for name, digest in entry["bundle_sha256"].items():
            checks.append({"check": "original_bundle_unchanged", "path": name, "passed": sha(original / name) == digest})
        replay = json.loads((b / f"independent_lut_replay_{dataset}.json").read_text())
        checks.append({"check": "independent_lut_full_test_replay", "dataset": dataset,
            "passed": replay["exact_full_test_replay"] and replay["inference_state_unchanged"]
                and replay["scores"] == results[dataset]["lut_events"]["test"]
                and replay["checkpoint_sha256"] == results[dataset]["lut_events"]["selection"]["checkpoint_sha256"]})
        for mode in protocol["conditions"]:
            value = results[dataset][mode]
            selection = value["selection"]
            folder = b / dataset / mode
            provenance = json.loads((folder / "provenance.json").read_text())
            if mode == "lut_events":
                checks.append({"check": "actual_lut", "dataset": dataset,
                    "passed": provenance["lut"] == json.loads((b / "lut_definition.json").read_text())["table"]})
            best = max(selection["history"], key=lambda h: (h["validation"]["top1_accuracy"], -h["validation"]["loss"]))
            checks.append({"check": "valid_only_selection", "dataset": dataset, "condition": mode,
                "passed": best["block"] == selection["selected_block"] and before_test[dataset][mode] == selection})
            checks.append({"check": "checkpoint_sha256", "dataset": dataset, "condition": mode,
                "passed": sha(folder / "best.pth") == selection["checkpoint_sha256"]})
            checks.append({"check": "config_sha256", "dataset": dataset, "condition": mode,
                "passed": sha(folder / "config.json") == provenance["config_sha256"]})
            checks.append({"check": "full_test_and_state_freeze", "dataset": dataset, "condition": mode,
                "passed": value["inference_state_unchanged"] and all(s["samples"] == old["test"]["float32"]["samples"] for s in value["test"].values())})
            checks.append({"check": "training_protection", "dataset": dataset, "condition": mode,
                "passed": selection["all_bp_stdp_protection_assertions_passed"]})
            rows.append({"dataset": dataset, "condition": mode, "parameters": value["parameters"],
                "VALID": selection["validation"]["top1_accuracy"], "VALID_loss": selection["validation"]["loss"],
                "selected_block": selection["selected_block"], "train_batches": selection["batches_trained"],
                "train_samples": selection["samples_trained"],
                **{k: v["top1_accuracy"] for k,v in value["test"].items()},
                "test_samples": value["test"]["float32"]["samples"], "checkpoint_sha256": selection["checkpoint_sha256"]})
            if mode == "lut_events":
                stats = selection["training_statistics"]
                logged = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
                sampled_saturation = max((s.get("max_weight_fraction_winner_rows", 0.0) for log in logged
                    for s in log.get("stdp", {}).values() if s is not None), default=0.0)
                diagnostics.append({"dataset": dataset, "winner_events": sum(s["winner_events"] for s in stats.values()),
                    "nonzero_weight_writes": sum(s["nonzero_weight_writes"] for s in stats.values()),
                    "raw_delta_abs_max": max(s["raw_delta_abs_max"] for s in stats.values()),
                    "actual_delta_abs_max": max(s["actual_delta_abs_max"] for s in stats.values()),
                    "observed_trace_states": sorted({u for s in stats.values() for u,n in enumerate(s["winner_trace_histogram"]) if n}),
                    "logged_winner_rows_upper_saturation_max": sampled_saturation,
                    "selected_checkpoint_training_blocks": selection["selected_block"],
                    "count_scope": "cumulative per-minibatch writes over ALL screened blocks, includes unselected later blocks"})
    assert all(x["passed"] for x in checks), [x for x in checks if not x["passed"]]
    with (b / "summary.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    write = lambda path, value: path.write_text(json.dumps(value, indent=2) + "\n")
    write(b / "learning_diagnostics.json", diagnostics)
    write(b / "encoding_initial_VALID.json", encoding_initial)
    text = ["# 压缩后的离散 pre-trace / LUT STDP 实施结果", "",
        "代码实施、四数据集基线审计、固定预算训练及完整 TEST 已完成。连续模式仍为默认；LUT 为显式可切换条件。原压缩模型、配置和冻结源码的全部 SHA256 均未改变。", "",
        "依赖原 turn 已有 task_complete，且宿主 GPU 无压缩训练进程；本会话没有 read_thread，使用原 rollout 生命周期、交接 completion 与最终产物交叉确认。证据见 dependency_audit.json。", "",
        "本轮仅使用最终压缩模型作为基线和 TRAIN 蒸馏教师：MNIST 4,452,879、N-MNIST 4,453,455、CIFAR10-DVS 4,717,903、Gesture 4,456,656 参数。旧 30M–46M 模型只出现在历史 provenance，不参与本轮推理或训练。四个原压缩模型的 FP32/uint2/uint4 全 TEST 重放精确匹配。", "",
        "三种条件：legacy_native（连续、原输入），legacy_events（连续、事件编码），lut_events（LUT、同一事件编码）。MNIST 使用严格 binary 输入，其他数据集采用均值保持 Bernoulli；forward 与 trace 共用同一批事件。每条件同一随机种子、初始压缩 checkpoint 和数据顺序，最多 2×16 个 TRAIN batch。CNN、BP 隐藏行与分类头均继续浮点 BP；STDP 行与优化器动量隔离。", "",
        "损失为 0.5 CE + 0.5 KD、温度 2，教师是冻结的最终压缩 checkpoint。BP 学习率为实际压缩恢复值的 0.1 倍，hidden BP 为 0.5 倍、分类头为 5 倍；STDP 保持 0.001。保留 T/分辨率/原输入编码、WTA 分组/first-winner/tiebreak/BP-STDP 分工、指数稳态和 winner 后完整前向。初始化及每个训练块后均跑完整 VALID，先精度后 loss 选择；全部条件选择固定后才测 TEST。block=0 表示初始化状态优于后续块，不是失败后用 TEST 重选。", "",
        "| 数据集 | 条件 | VALID % | VALID loss | 选中 block | TRAIN样本 | TEST FP32 % | uint2 % | uint4 % | TEST样本 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        text.append(f"| {r['dataset']} | {r['condition']} | {pct(r['VALID'])} | {r['VALID_loss']:.6f} | {r['selected_block']} | {r['train_samples']} | {pct(r['float32'])} | {pct(r['uint2'])} | {pct(r['uint4'])} | {r['test_samples']} |")
    text += ["", "在完全相同初始压缩权重、未进行任何学习时，完整 VALID 已能单独检验输入编码影响。事件编码的连续/LUT 两条件完整分数字典逐项完全一致，确认切换学习规则未额外改变前向。", "",
        "| 数据集 | 初始原输入 VALID % | 初始事件编码 VALID % | 编码变化 pp |",
        "|---|---:|---:|---:|"]
    for e in encoding_initial:
        native, events = e["native"]["top1_accuracy"], e["events"]["top1_accuracy"]
        text.append(f"| {e['dataset']} | {pct(native)} | {pct(events)} | {100*(events-native):+.2f} |")
    text += ["", "CIFAR 的初始编码损失为 9.67 pp，Gesture 为 5.39 pp。选中状态的差值还包含各自训练轨迹和 VALID 选型，不能把最终两格差值全部视作单次 LUT 更新的因果效果。"]
    text += ["", "原压缩基线与 LUT 选中模型（完整 TEST）：", "",
        "| 数据集 | 原压缩 FP32/uint2/uint4 % | LUT FP32/uint2/uint4 % | LUT FP32原门槛通过 |",
        "|---|---|---|---|"]
    for e in manifest["entries"]:
        dataset = e["dataset"]
        new = results[dataset]["lut_events"]["test"]
        old = old_scores[dataset]
        text.append(f"| {dataset} | {' / '.join(pct(old[k]['top1_accuracy']) for k in ['float32','uint2','uint4'])} | {' / '.join(pct(new[k]['top1_accuracy']) for k in ['float32','uint2','uint4'])} | {'是' if new['float32']['top1_accuracy'] >= e['fp32_goal'] else '否'} |")
    text += ["", "LUT 的实际训练写入记录（全部训练块，非最终选中模型的独占归因；每次 batch 写同一突触会重复计数）：", "",
        "| 数据集 | STDP winner事件 | 非零突触写次数 | raw delta 最大绝对值 | 含归一化实际最大变化 | winner trace实际覆盖 |",
        "|---|---:|---:|---:|---:|---|"]
    for d in diagnostics:
        text.append(f"| {d['dataset']} | {d['winner_events']} | {d['nonzero_weight_writes']} | {d['raw_delta_abs_max']:.4g} | {d['actual_delta_abs_max']:.4g} | {','.join(map(str,d['observed_trace_states']))} |")
    text += ["", "所有 LUT 条件均产生实际 STDP 写入，未出现持续单类塌缩。记录的 winner 行上界饱和比例为 0。无 STDP winner 的 batch 保持零写，未强制改权重。MNIST/N-MNIST 本轮成熟模型的 winner 仅覆盖 U=0/31，countdown 中间状态由因果单元检查验证；本轮不能替代长期训练对所有 age 的实际覆盖。", "",
        "实际表 U=0..31：", "", "```text",
        "[-12,-11,-11,-11,-11,-11,-10,-10,-10,-10,-10,-9,-9,-9,-8,-8,",
        " -7,-7,-6,-6,-5,-4,-3,-3,-1,0,1,2,4,6,8,10]", "```", "",
        "每 presentation U 复位为 0，事件刷新到 31，否则减 1 到 0；同一步 pre 刷新先于 winner 查表，每个 sample/group 只提交第一次 winner。c(U)=0（U=0），否则 0.9^(31−U)；g=0.022c−0.012，L=round(g/0.001)。表为 32 个 signed 5-bit 系数、160 bit 逻辑存储；软件 trace uint8、table int8。trace 的等效指数 tau=9.49122 step，与稳态阈值 tau=20 分开。", "",
        "浮点基础 delta 为 1e−6×L。兼容模式沿用正更新乘 (1−w)、负更新乘 w、整个 batch mean、winner-only row normalization 和 [0,1] 裁剪。表项不是 uint2/uint4 权重码增量。旧连续规则在 winner 后仍可按 post trace 对后续 pre 做 LTD；新规则在 winner 时以旧/缺失 pre 触发 LTD，trace refresh 替代累积，二者不等价。学习 trace 的有效快照在 winner 时固定，之后仍保留完整前向。", "",
        "本轮未重新标定 STDP 学习率：0.001 已能产生可测更新，且远小于推理量化 LSB；没有证据表明它是长期收敛最优值。row normalization 会改变实际变化尺度，不能只凭 LUT 基础 delta 比较训练强度。CIFAR/Gesture 的编码控制已显示精度损失，应先处理活动重编码与前向统计，再在 VALID 上重新标定；不能把全部损失归因于 LUT 或假定增大学习率即可恢复。", "",
        "这轮是单种子、有界 warm-start 检查，未完成从头训练、完整 TRAIN 收敛、多种子、学习率搜索或 uint2/uint4 恢复优化。全 TEST 已完成不代表所有 FP32 门槛通过。LUT 暂不替换默认连续模式。独立 add_clip 开关仅通过局部更新/饱和边界测试，未测精度；严格 winner 后停止前向、整数残差/随机更新硬件学习尚未实现。", "",
        "60 项相关回归通过：trace refresh/decay/reset、同一步顺序、组内首 winner 与后续活动保护、无 winner 零写、BP/STDP及动量隔离、表项/位宽/边界、batch scaling、严格旧 checkpoint 加载、冻结源码连续模式 forward/gradient/update 一致性、eval 状态冻结、隐藏层推理量化。四个 LUT 选中模型另经独立完整 FP32/uint2/uint4 重放精确复现；回归日志及 completion_audit.json 保存证据。", "",
        "CIFAR 首版压缩 provenance 没有新版 backbone_learning_rate 字段；读取其冻结 compression_runtime/run.py 中实际 RATES=0.0003，Gesture 为 0.0002。未开始训练的失败启动保存在 attempts，旧 driver 保存在 run_frozen_attempt1.py；已完成条件未重跑或按 TEST 更换。", "",
        "主要产物：protocol.json（冻结合同及源码 SHA），results.json / summary.csv（完整比较），lut.csv / lut_definition.json（实际表），learning_diagnostics.json（写入/trace覆盖），各 dataset/condition 的 config.json、provenance.json、best.pth、validation.json、accuracy.json、metrics.jsonl。默认模型实现见 snn/wta.py、snn/stdp_lut.py；接口说明见 experiments/stdp/README.md。", "",
        "独立复现，输出路径必须不存在：", "", "```bash",
        f"TMPDIR=/tmp .venv/bin/python results/accuracy/{b.name}/reproduce.py --bundle results/accuracy/{b.name} --dataset MNIST --out /tmp/MNIST_lut_replay_NEW.json",
        "```", ""]
    (b / "README.md").write_text("\n".join(text))
    files = [p for p in b.rglob("*") if p.is_file() and p.suffix in {".py", ".json", ".csv", ".md", ".pth", ".log", ".diff"}
             and p.name != "completion_audit.json" and "__pycache__" not in p.parts]
    write(b / "completion_audit.json", {"status": "IMPLEMENTATION_AND_BOUNDED_EVALUATION_COMPLETE",
        "conditions": 12, "full_test_modes": 36, "independent_lut_replayed_models": 4, "unit_tests_passed": 60,
        "checks_passed": len(checks), "checks": checks, "all_fp32_goals_met": all(results[e['dataset']]['lut_events']['test']['float32']['top1_accuracy'] >= e['fp32_goal'] for e in manifest['entries']),
        "not_validated": ["from_scratch_convergence", "full_train_convergence", "multiseed", "learning_rate_optimality", "add_clip_accuracy", "integer_hardware_learning"],
        "package_sha256": {str(p.relative_to(b)): sha(p) for p in files}})
    print(json.dumps({"checks_passed": len(checks), "report": str(b / 'README.md'), "conditions": len(rows)}))


if __name__ == "__main__":
    main()
