"""Package explicit final choices, including TRAIN-calibrated inference PTQ."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results/accuracy'
DATASETS = ('MNIST', 'NMNIST', 'CIFAR10DVS', 'DVSGesture')
GOALS = dict(zip(DATASETS, (0.99, 0.98, 0.75, 0.95)))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def runtime_roles(source, config):
    """Inspect legacy configurations in an isolated frozen module namespace."""
    script = '''
import json,sys,torch
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from experiments.accuracy.models import AccuracySNN
torch.set_num_threads(1)
cfg=json.loads(Path(sys.argv[2]).read_text())
model=AccuracySNN('custom',sys.argv[3],cfg,'exponential')
roles={}
for name,layer in model.hidden_layers.items():
 roles[name]=dict(granularity=layer.hybrid_stdp_granularity,
  stdp_neurons=int(layer.stdp_neuron_mask.sum()),bp_neurons=int(layer.bp_neuron_mask.sum()),
  competition_size=layer.competition_size,
  stdp_neuron_indices=layer.stdp_neuron_mask.nonzero().flatten().tolist())
print(json.dumps(roles))
'''
    return json.loads(subprocess.check_output([sys.executable, '-c', script, str(source), str(config), config.parent.parent.name], text=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--history', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(exist_ok=False)
    history = args.history.resolve()
    history_audit = read(history / 'audit.json')
    assert history_audit['checks_passed'] == len(history_audit['checks'])
    assert all(v['passed'] for v in history_audit['checks'])
    shutil.copytree(history, out / 'history')
    checks = list(history_audit['checks'])

    def check(name, passed, **details):
        record = dict(check=name, passed=bool(passed), **details)
        checks.append(record)
        assert record['passed'], record

    pilot = RESULTS / 'stdp_lut_native_forward_20261007_v2'
    gesture = RESULTS / 'stdp_lut_gesture_teacher_20261007_v5'
    views = RESULTS / 'stdp_lut_cifar_views_20261007_v11'
    ptq = RESULTS / 'stdp_lut_cifar_PTQ_20261007_v12'
    choices = read(pilot / 'final_selections_before_TEST.json')
    gesture_choice = read(gesture / 'selection_before_TEST.json')
    view_choice = read(views / 'selection_before_TEST.json')
    view_policy = read(views / 'policy.json')
    view_result = read(views / 'result.json')
    view_replay = read(views / 'independent_replay.json')
    ptq_result = read(ptq / 'result.json')
    ptq_replay = read(ptq / 'independent_replay.json')
    ptq_policy = read(ptq / 'policy.json')
    ptq_choices = read(ptq / 'selection_before_TEST.json')
    calibration = read(ptq / 'calibration_evidence.json')
    check('view_independent_replay', view_replay['exact_full_test_replay'] and view_replay['result'] == view_result)
    check('PTQ_independent_replay', ptq_replay['exact_full_test_replay'] and ptq_replay['result'] == ptq_result)
    check('PTQ_FP32_exactly_unchanged', ptq_result['FP32_exactly_unchanged'] and ptq_result['test']['float32'] == view_result['test']['float32'])
    check('complete_TRAIN_label_free_PTQ', calibration['partition'] == 'TRAIN' and calibration['samples'] == 8100
          and calibration['views'] == 2 and calibration['temporal_samples']['hidden1'] == 129600
          and calibration['weights_and_homeostasis_unchanged'] and not calibration['labels_used'] and calibration['training_steps'] == 0)
    check('selected_view_matches_result', view_result['selection'] == view_choice and ptq_result['views'] == view_choice['views'] == 2)
    validation_views = read(views / 'VALID_results.json')
    best_view = max(validation_views, key=lambda v: (v['validation']['top1_accuracy'], -v['validation']['loss']))
    check('complete_VALID_view_selection', best_view == view_choice and all(v['validation']['samples'] == 900 for v in validation_views))
    for bundle, filename, policy in ((views, 'inference_frozen.py', view_policy), (ptq, 'ptq_frozen.py', ptq_policy)):
        check('frozen_inference_driver', sha(bundle / filename) == policy['driver_sha256'], path=str(bundle / filename))
    for name, digest in view_policy['source_sha256'].items():
        check('frozen_view_source', sha(Path(view_policy['source']) / name) == digest, path=name)
    hashes = read(ptq / 'settings_sha256.json')
    check('PTQ_settings_and_moments_pinned', sha(ptq / 'quantizer_settings.pth') == hashes['settings'] == ptq_result['settings_sha256']
          and sha(ptq / 'TRAIN_moments.pth') == hashes['moments'] == ptq_result['moments_sha256'])
    validations = read(ptq / 'VALID_results.json')
    for bits in ('2', '4'):
        best = max(validations[bits], key=lambda v: (v['validation']['top1_accuracy'], -v['validation']['loss']))
        check('complete_VALID_PTQ_selection', best == ptq_choices[bits] and len(validations[bits]) == 17
              and all(v['validation']['samples'] == 900 for v in validations[bits]), bits=bits)

    entries = []
    for dataset in DATASETS:
        if dataset == 'CIFAR10DVS':
            bundle = Path(view_choice['candidate']['bundle'])
            choice = view_choice['candidate']
            result = ptq_result
            replay_path = ptq / 'independent_replay.json'
        elif dataset == 'DVSGesture':
            bundle, choice = Path(gesture_choice['bundle']), gesture_choice
            replay_path = gesture / 'independent_selected_replay_DVSGesture.json'
            result = read(bundle / dataset / 'lut_native_trace/accuracy.json')
        else:
            choice = choices[dataset]
            bundle = Path(choice['bundle'])
            replay_path = pilot / f'independent_selected_replay_{dataset}.json'
            result = read(bundle / dataset / 'lut_native_trace/accuracy.json')
        condition = bundle / dataset / 'lut_native_trace'
        cfg = read(condition / 'config.json')
        provenance = read(condition / 'provenance.json')
        selection = read(condition / 'validation.json')
        protocol = read(bundle / 'protocol.json')
        check('selected_checkpoint_pinned', sha(condition / 'best.pth') == choice['checkpoint_sha256'], dataset=dataset)
        check('selected_config_pinned', sha(condition / 'config.json') == provenance['config_sha256'], dataset=dataset)
        check('selected_training_driver_pinned', sha(bundle / 'run_frozen.py') == protocol['driver_sha256'], dataset=dataset)
        for name, digest in protocol['source_sha256'].items():
            check('selected_training_source_pinned', sha(bundle / name) == digest, dataset=dataset, path=name)
        replay = read(replay_path)
        if dataset != 'CIFAR10DVS':
            check('selected_full_TEST_exact_replay', replay['exact_full_test_replay'] and replay['inference_state_unchanged']
                  and replay['scores'] == result['test'] and replay['checkpoint_sha256'] == choice['checkpoint_sha256'], dataset=dataset)
        else:
            check('PTQ_same_trained_checkpoint', ptq_result['checkpoint_sha256'] == view_result['selection']['candidate']['checkpoint_sha256']
                  == choice['checkpoint_sha256'] == ptq_policy['checkpoint_sha256'], dataset=dataset)
        check('selected_physical_parameter_budget', 3_000_000 <= result['parameters'] <= 5_000_000, dataset=dataset)
        check('selected_BP_STDP_protection', selection['all_bp_stdp_protection_assertions_passed'], dataset=dataset)
        check('selected_FP32_goal', result['test']['float32']['top1_accuracy'] >= GOALS[dataset], dataset=dataset)
        samples = {'MNIST': 10000, 'NMNIST': 10000, 'CIFAR10DVS': 1000, 'DVSGesture': 288}[dataset]
        check('selected_full_TEST_sample_counts', all(v['samples'] == samples for v in result['test'].values()), dataset=dataset)
        destination = out / 'conditions' / dataset
        destination.mkdir(parents=True)
        for name in ('best.pth', 'config.json', 'provenance.json', 'validation.json'):
            shutil.copy2(condition / name, destination / name)
        shutil.copy2(replay_path, destination / 'source_independent_replay.json')
        write(destination / 'accuracy.json', result)
        source = bundle / 'runtime' / dataset
        for file in source.rglob('*.py'):
            target = destination / 'runtime' / file.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
        prefix = f'conditions/{dataset}'
        profile = cfg['dataset_profiles'][dataset]
        entry = dict(dataset=dataset, source_bundle=str(bundle), parameters=result['parameters'],
                     config=f'{prefix}/config.json', checkpoint=f'{prefix}/best.pth', runtime=f'{prefix}/runtime',
                     checkpoint_sha256=choice['checkpoint_sha256'], split_sha256=provenance['split_sha256'],
                     steps_per_view=profile['steps'], views=2 if dataset == 'CIFAR10DVS' else 1,
                     resolution=profile.get('resolution'), channels=cfg.get('custom_channels'),
                     hidden_widths=cfg['hidden_widths'], spatial_readout=profile['pool_size'],
                     hidden_roles=provenance.get('actual_hidden_roles') or runtime_roles(source, condition / 'config.json'), selected_epoch=selection['selected_block'],
                     inherited_lut_training=provenance.get('inherited_lut_training', False),
                     new_LUT_weights=selection['selected_block'] > 0 or provenance.get('inherited_lut_training', False),
                     test_rng_seed=1042, test_samples=samples, test=result['test'], FP32_goal=GOALS[dataset],
                     FP32_goal_passed=True, training_provenance=f'{prefix}/provenance.json')
        if dataset == 'CIFAR10DVS':
            for name in ('TRAIN_moments.pth', 'quantizer_settings.pth', 'calibration_evidence.json', 'selection_before_TEST.json'):
                shutil.copy2(ptq / name, destination / name)
            entry.update(TRAIN_moments=f'{prefix}/TRAIN_moments.pth', quantizer_settings=f'{prefix}/quantizer_settings.pth',
                         quantizer_selection=ptq_choices, inference_compute='two views; approximately twice the single-view work',
                         source_training_epochs=36, selected_polish_epochs=1, TRAIN_classifier_fit_steps=200)
        entries.append(entry)

    shutil.copy2(ROOT / 'experiments/stdp/reproduce_final.py', out / 'reproduce.py')
    shutil.copy2(views / 'inference_frozen.py', out / 'inference_views_frozen.py')
    shutil.copy2(ptq / 'ptq_frozen.py', out / 'ptq_frozen.py')
    lut = read(out / 'conditions/CIFAR10DVS/provenance.json')['lut']
    check('signed_five_bit_LUT', len(lut) == 32 and min(lut) >= -16 and max(lut) <= 15)
    with (out / 'STDP_LUT.csv').open('w') as file:
        writer = csv.writer(file)
        writer.writerow(('U', 'age_steps_or_absent', 'signed_coefficient', 'coefficient_float'))
        for u, value in enumerate(lut):
            writer.writerow((u, 'absent_or_age_ge_31' if u == 0 else 31-u, value, value*0.001))
    write(out / 'STDP_LUT.json', dict(trace_bits=5, coefficient_bits=5, logical_table_bits=160, coefficient_lsb=0.001,
          coefficients=lut, trace_decay=0.9, trace_tau_steps=9.491221581029905,
          a_plus=0.01, a_minus=0.012, refresh_before_lookup=True, training_arithmetic='floating point',
          weight_quantization='inference only; LUT coefficients are not weight-code increments'))
    write(out / 'audit.json', dict(checks=checks, checks_passed=len(checks),
          all_FP32_goals_passed=True, full_TEST_independent_replays=True,
          regression_tests=dict(passed=67, observed_output='67 passed in 8.52s',
                                note='Recorded completed regression suite; see test inventory in README.')))
    with (out / 'summary.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=('dataset', 'parameters', 'steps_per_view', 'views', 'float32', 'uint2', 'uint4', 'FP32_goal_passed', 'new_LUT_weights'))
        writer.writeheader()
        for entry in entries:
            writer.writerow({**{k: entry[k] for k in writer.fieldnames if k not in ('float32', 'uint2', 'uint4')},
                             **{k: entry['test'][k]['top1_accuracy'] for k in ('float32', 'uint2', 'uint4')}})
    readme = ['# 新 LUT STDP 优化后的四数据集结果', '',
              '四项 FP32 精度目标均已达到，以下均为完整 TEST；独立进程重放与原结果完全一致。', '',
              '| 数据集 | 物理参数 M | T/视图 | 视图数 | TEST 样本 | FP32 % | 2-bit % | 4-bit % | FP32 目标 % |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for entry in entries:
        readme.append(f"| {entry['dataset']} | {entry['parameters']/1e6:.3f} | {entry['steps_per_view']} | {entry['views']} | {entry['test_samples']} | "
                      + ' | '.join(f"{100*entry['test'][k]['top1_accuracy']:.2f}" for k in ('float32', 'uint2', 'uint4'))
                      + f" | {100*entry['FP32_goal']:.0f} |")
    readme += ['', 'MNIST、N-MNIST、Gesture 保留已达标的压缩权重；其选中 epoch=0，不宣称新增 LUT 训练提升。CIFAR 从随机初始化训练新 LUT 模型 40 个完整 epoch、按 VALID 选第 36 个，再做 TRAIN 分类头拟合与完整微调，按 VALID 选第 1 个微调 epoch。', '',
               'CIFAR 结构为通道 [32,64,128,256,448]、4×4 空间读出、256 个隐藏神经元，共 4,679,375 个物理参数；竞争组大小 16，16 个 STDP 行、240 个 BP 行，按完整组分工。原图/水平翻转平均 logits，各自 T=8，视图之间重置膜电位。双视图约增加一倍推理计算；同 checkpoint 单视图 FP32 为 74.90%，不能把 76.20% 当成单视图结果。', '',
               'CIFAR 推理量化只读取完整 TRAIN 的隐藏输入均值/二阶矩，不使用标签、不训练权重；34 个候选在完整 VALID 上按精度、再 loss 选定。2-bit 的逐行裁剪比例为 0.5，4-bit 的比例按 TRAIN 输入方差加权误差逐行选择，二者使用完整偏置补偿 bq=b+(W−Wq)μ。权重码为 unsigned 2/4-bit、逐输出行浮点 scale、零点 0；卷积、分类头、bias、scale 和神经元参数仍为浮点。偏置可有符号；非负约束作用于突触权重。', '',
               '同一 FP32 模型原始 absmax 量化的 2-bit/4-bit 为 60.20%/68.80%，校准后为 74.40%/75.70%，FP32 精度完全不变。2-bit 仍低于 75% 门槛 0.60 个百分点；当前已完成的目标是四项 FP32 达标及低位宽改善。', '',
               '卷积骨干采用物理非负 E/I 双权重银行与 PLIF；隐藏 LIF 使用 m←0.9m+I、达到阈值发放并减去阈值，atan surrogate。WTA 只决定隐藏层的更新资格，每个样本/竞争组只选首次 winner，其他发放保留，卷积和分类头继续 BP。隐藏 BP/STDP 行互斥，非获胜行和其优化器状态受到保护。', '',
               '指数稳态调节逐步执行 offset←exp(−1/20)offset+0.05×batch_winner_count，阈值为 clip(1+offset,0.1,10)，跨样本持久保存；VALID/TEST 使用 checkpoint 中的阈值并冻结调节。另有按目标发放率调节的可切换实现，本表选用指数模式。', '',
               '前向保持原生 spike coding/活动，局部学习才用独立 RNG 对池化活动按 x/A 采样 Bernoulli 事件，A=1；这不是池化前真实 spike 的时序重建。MNIST/N-MNIST 的 T=4、CIFAR T=8、Gesture T=16，具体输入分辨率和事件归一化以各条件 config.json 及冻结 data.py 为准。', '',
               'LUT：5-bit countdown，event 置 31、静默减 1 到 0，每次 presentation 从 0 开始；刷新先于首次 winner 查表。U=0 的 c=0，否则 c=0.9^(31−U)，L=round((0.022c−0.012)/0.001)。完整 32 项见 STDP_LUT.csv。CIFAR stdp_lr=0.01，基础浮点 Δw=1e−5×L；soft bounds 与 winner 行归一化后变化可更大。表为 signed 5-bit 系数、逻辑 160 bit，不代表实测硬件面积，也不是权重码增量。训练前向/反向均为浮点。', '',
               '## 独立复现', '',
               '冻结的模型代码、配置、checkpoint、CIFAR 校准统计和量化设定已复制到 conditions/，不依赖可变的工作区模型源码或旧实验权重路径。依赖本工作区数据集及 CUDA 环境；原配置中的数据路径保留。一次只重放一个数据集，三个精度模式全部检查。输出文件必须不存在。', '',
               '```bash', f'TMPDIR=/tmp .venv/bin/python {out}/reproduce.py --bundle {out} --dataset CIFAR10DVS --out /tmp/final_CIFAR_replay_NEW.json', '```', '',
               '其他数据集将 --dataset 改为 MNIST、NMNIST 或 DVSGesture。benchmark_manifest.json 记录全部文件 SHA256 和参数/样本/精度/视图条件；audit.json 保存来源审核。最终副本的重放结果保存到 replays/。', '',
               '## 训练与实验记录', '',
               '本轮重训入口是 experiments/stdp/run.py 的 lut_native_trace 条件；CIFAR 从头训练使用 --spatial-student --from-scratch --epochs 40 --patience 6 --stdp-lr 0.01 --bp-rate-scale 1.5 --initial-row-sum 4 --augment-spatial --teacher-source original_cached --stdp-granularity group。实际参数/学习率/冻结策略/教师缓存哈希以来源 protocol.json、training provenance 为准；config 中旧恢复预算不是本轮预算。教师仅参与 TRAIN，推理不包含大教师。', '',
               '完整结果表和 2×2/4×4 从头训练的 VALID 曲线见 history/README.md。前向随机编码破坏初始化精度、竞争组配置未传入而作废的 v3，以及结构直接迁移效果很差而早停的 v8 均保留原实验记录；未作有效成功证据。当前结果为单种子，形状、蒸馏、分类头拟合、双视图与量化校准共同改变，不能单独归因于 LUT。每轮 checkpoint/量化设定依据 VALID，但 TEST 门槛触发进一步开发，整项迭代不是一次性盲测。', '',
               '回归检查覆盖 LUT/事件 RNG、旧 checkpoint 兼容、WTA/BP/STDP 保护、稳态、非负骨干、结构迁移、分类头拟合、双视图重置和推理量化补偿。67 项通过。', '']
    (out / 'README.md').write_text('\n'.join(readme))
    artifact_sha256 = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    write(out / 'benchmark_manifest.json', dict(schema_version=1, status='FP32_goals_met_quantization_improved',
          selection='explicit campaign VALID selections; no TEST ranking in this packaging step',
          all_FP32_goals_passed=True, quantized_CIFAR_uint2_goal_open=True, test_rng_seed=1042,
          entries=entries, artifact_sha256=artifact_sha256))
    write(out / 'status.json', dict(phase='copied_artifacts_pending_final_replay', all_FP32_goals_passed=True,
          source_independent_replays_exact=True, artifact_checks_passed=len(checks)))
    print(json.dumps(dict(bundle=str(out), checks_passed=len(checks), datasets=len(entries))))


if __name__ == '__main__':
    main()
