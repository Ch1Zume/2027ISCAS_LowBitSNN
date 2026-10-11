"""Audit explicit completed LUT campaigns; report all outcomes without TEST ranking."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ('MNIST', 'NMNIST', 'CIFAR10DVS', 'DVSGesture')
GOALS = dict(zip(DATASETS, (0.99, 0.98, 0.75, 0.95)))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--campaigns', type=Path, nargs='*', default=[])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(exist_ok=False)
    checks, rows = [], []
    curves = []
    baseline = ROOT / 'results/accuracy/compression_3to5M_20261006_selected'
    manifest = json.loads((baseline / 'compression_manifest.json').read_text())
    for entry in manifest['entries']:
        for filename, digest in entry['bundle_sha256'].items():
            checks.append(dict(check='immutable_compression_bundle', path=filename, passed=sha(baseline / filename) == digest))

    def record(dataset, folder, selection, replay_path, campaign):
        condition = folder / dataset / 'lut_native_trace'
        result = json.loads((condition / 'accuracy.json').read_text())
        provenance = json.loads((condition / 'provenance.json').read_text())
        cfg = json.loads((condition / 'config.json').read_text())
        replay = json.loads(replay_path.read_text())
        protocol = json.loads((folder / 'protocol.json').read_text())
        checks.extend([
            dict(check='checkpoint_frozen_before_TEST', campaign=campaign, dataset=dataset,
                 passed=sha(condition / 'best.pth') == selection['checkpoint_sha256'] == result['selection']['checkpoint_sha256']),
            dict(check='full_TEST_and_exact_independent_replay', campaign=campaign, dataset=dataset,
                 passed=result['inference_state_unchanged'] and replay['inference_state_unchanged']
                    and replay['exact_full_test_replay'] and replay['scores'] == result['test']
                    and replay['checkpoint_sha256'] == selection['checkpoint_sha256']),
            dict(check='config_pinned', campaign=campaign, dataset=dataset,
                 passed=sha(condition / 'config.json') == provenance['config_sha256']),
            dict(check='physical_parameter_budget', campaign=campaign, dataset=dataset,
                 passed=3_000_000 <= result['parameters'] <= 5_000_000),
            dict(check='BP_STDP_protected_updates', campaign=campaign, dataset=dataset,
                 passed=result['selection']['all_bp_stdp_protection_assertions_passed']),
        ])
        best = max(result['selection']['history'], key=lambda r: (r['validation']['top1_accuracy'], -r['validation']['loss']))
        checks.append(dict(check='within_condition_VALID_selection', campaign=campaign, dataset=dataset,
                           passed=best['block'] == result['selection']['selected_block']))
        for name, digest in protocol['source_sha256'].items():
            checks.append(dict(check='frozen_source', campaign=campaign, path=name, passed=sha(folder / name) == digest))
        checks.append(dict(check='frozen_driver', campaign=campaign,
                           passed=sha(folder / 'run_frozen.py') == protocol['driver_sha256']))
        entry = next(e for e in manifest['entries'] if e['dataset'] == dataset)
        test_samples = json.loads((baseline / entry['accuracy']).read_text())['test']['float32']['samples']
        checks.append(dict(check='complete_TEST_sample_counts', campaign=campaign, dataset=dataset,
                           passed=all(v['samples'] == test_samples for v in result['test'].values())))
        rows.append(dict(campaign=campaign, dataset=dataset, bundle=str(folder), parameters=result['parameters'],
                         initialization=provenance.get('initialization', 'random' if protocol.get('from_scratch') else 'compressed_checkpoint'), selected_epoch=result['selection']['selected_block'],
                         stdp_updated_checkpoint=result['selection']['selected_block'] > 0 or provenance.get('inherited_lut_training', False),
                         inherited_lut_training=provenance.get('inherited_lut_training', False),
                         TRAIN_head_fit=provenance.get('head_bootstrap') is not None,
                         TRAIN_BN_calibration=provenance.get('bn_calibration') is not None,
                         spatial_readout=cfg['dataset_profiles'][dataset]['pool_size'],
                         trained_epochs=len(result['selection']['history']) - 1,
                         trained_samples=result['selection']['samples_trained'],
                         VALID=result['selection']['validation']['top1_accuracy'],
                         **{k:v['top1_accuracy'] for k,v in result['test'].items()},
                         goal_passed=result['test']['float32']['top1_accuracy'] >= GOALS[dataset],
                         steps=cfg['dataset_profiles'][dataset]['steps'], stdp_lr=cfg['stdp_lr'],
                         checkpoint_sha256=selection['checkpoint_sha256']))

    pilot = args.pilot.resolve()
    choices = json.loads((pilot / 'final_selections_before_TEST.json').read_text())
    for dataset in DATASETS:
        value = choices[dataset]
        record(dataset, Path(value['bundle']), value,
               pilot / f'independent_selected_replay_{dataset}.json', pilot.name)
    for path in args.campaigns:
        campaign = path.resolve()
        status = json.loads((campaign / 'status.json').read_text())
        if status['phase'] == 'not_needed_previous_goal_met':
            continue
        assert status['phase'] in ('completed', 'accuracy_goal_open'), status
        choice = json.loads((campaign / 'selection_before_TEST.json').read_text())
        chosen = Path(choice['bundle'])
        dataset = json.loads((campaign / 'result.json').read_text())['dataset']
        record(dataset, chosen, choice, campaign / f'independent_selected_replay_{dataset}.json', campaign.name)
        for validation_file in campaign.rglob('CIFAR10DVS/lut_native_trace/validation.json'):
            condition = validation_file.parent
            provenance = json.loads((condition/'provenance.json').read_text())
            if provenance.get('initialization') != 'random':
                continue
            cfg = json.loads((condition/'config.json').read_text())
            selection = json.loads(validation_file.read_text())
            curves.append(dict(label=f"From scratch, readout {cfg['dataset_profiles']['CIFAR10DVS']['pool_size']}x{cfg['dataset_profiles']['CIFAR10DVS']['pool_size']}, hidden {cfg['hidden_widths'][0]}",
                epochs=[v['block'] for v in selection['history']],
                accuracy=[100*v['validation']['top1_accuracy'] for v in selection['history']],
                path=str(validation_file)))
    assert all(c['passed'] for c in checks), [c for c in checks if not c['passed']]
    with (root / 'summary.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (root / 'results.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    (root / 'audit.json').write_text(json.dumps(dict(checks=checks, checks_passed=len(checks),
        selection='explicit campaign choices frozen by VALID; this report never ranks TEST',
        caveats=['single seed', 'BP, teacher, roles, coding and augmentation vary across trials; no isolated LUT causal benefit claim',
                 'each campaign selects checkpoints on VALID; repeated TEST goals trigger further development, so the whole iteration is not a one-shot blind evaluation',
                 'selected epoch 0 has no new full-network STDP update in that campaign; it can inherit LUT-trained weights or include registered structural transfer, TRAIN BN calibration or TRAIN head fit, recorded in provenance']), indent=2) + '\n')
    text = ['# LUT STDP 继续优化记录', '',
            '以下为各轮独立预注册、完整 VALID 选定候选的完整 TEST 结果。表格保留失败轮次，不按 TEST 精度重选 checkpoint。', '',
            '| 轮次 | 数据集 | 参数 M | T | 选中轮次 | VALID % | FP32 % | uint2 % | uint4 % | FP32 达标 |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in rows:
        text.append(f"| {row['campaign']} | {row['dataset']} | {row['parameters']/1e6:.3f} | {row['steps']} | {row['selected_epoch']} | {100*row['VALID']:.2f} | {100*row['float32']:.2f} | {100*row['uint2']:.2f} | {100*row['uint4']:.2f} | {'是' if row['goal_passed'] else '否'} |")
    text += ['', '选中轮次为 0 表示没有该轮新增的完整网络 STDP 更新；可能继承此前 LUT 训练权重，或包括已登记的结构迁移、TRAIN BN 校准及 TRAIN 分类头拟合，详见初始化 provenance。教师仅参与 TRAIN 蒸馏，推理模型仍为表中学生尺寸。', '',
             '前向和 BP 使用原生活动；Bernoulli 仅用于局部学习 trace，并有独立、可恢复的 RNG。这是对池化活动的概率解释，不是还原池化前的 spike 时序。', '',
             'WTA 只选择隐藏更新资格，CNN 与分类头继续 BP。整组分工避免 STDP 行被成熟 BP 行长期压制；warm start 重分工时保持当前权重并重新记录行和目标。', '',
             'LUT 仍为 5-bit countdown、32 个 signed 5-bit 系数，系数 LSB=0.001；阈值稳态 tau=20 与 trace 等效 tau=9.49122 区分。仅推理隐藏 Linear 采用 unsigned 2/4-bit 权重量化。', '',
             '原压缩包哈希、冻结源码/配置、VALID 选型、完整样本数及独立全 TEST 重放已审核。这里仅有单随机种子；多项训练条件同时变化，不能把改善单独归因于 LUT。每轮 checkpoint 仅依据 VALID，但 TEST 门槛用于判断是否继续开发，整项迭代不是一次性盲测。', '']
    (root / 'README.md').write_text('\n'.join(text))
    if curves:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(8,4.5),layout='constrained')
        for curve in curves:
            axis.plot(curve['epochs'],curve['accuracy'],label=curve['label'],marker='o',markersize=3)
        axis.axhline(75.3333333333,color='gray',linestyle='--',label='Original compressed VALID')
        axis.set(xlabel='Complete TRAIN epochs',ylabel='Complete VALID accuracy (%)',
                 title='CIFAR10-DVS LUT training; native forward, T=8')
        axis.grid(alpha=0.25)
        axis.legend(loc='lower right',fontsize=8)
        figure.savefig(root/'CIFAR_VALID_learning_curves.png',dpi=180)
        plt.close(figure)
        (root/'learning_curves.json').write_text(json.dumps(curves,indent=2)+'\n')
        with (root/'README.md').open('a') as file:
            file.write('\n随机初始化条件的完整 VALID 学习曲线（结构和 STDP 行数同时改变，不是单一因素消融）：\n\n![CIFAR VALID](CIFAR_VALID_learning_curves.png)\n')
    print(json.dumps(dict(report=str(root / 'README.md'), measured_campaign_rows=len(rows), checks_passed=len(checks))))


if __name__ == '__main__':
    main()
