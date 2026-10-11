# 新 LUT STDP 优化后的四数据集结果

四项 FP32 精度目标均已达到，以下均为完整 TEST；独立进程重放与原结果完全一致。

| 数据集 | 物理参数 M | T/视图 | 视图数 | TEST 样本 | FP32 % | 2-bit % | 4-bit % | FP32 目标 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MNIST | 4.453 | 4 | 1 | 10000 | 99.26 | 99.29 | 99.29 | 99 |
| NMNIST | 4.453 | 4 | 1 | 10000 | 98.96 | 98.84 | 98.85 | 98 |
| CIFAR10DVS | 4.679 | 8 | 2 | 1000 | 76.20 | 74.40 | 75.70 | 75 |
| DVSGesture | 4.457 | 16 | 1 | 288 | 95.49 | 95.14 | 95.14 | 95 |

MNIST、N-MNIST、Gesture 保留已达标的压缩权重；其选中 epoch=0，不宣称新增 LUT 训练提升。CIFAR 从随机初始化训练新 LUT 模型 40 个完整 epoch、按 VALID 选第 36 个，再做 TRAIN 分类头拟合与完整微调，按 VALID 选第 1 个微调 epoch。

CIFAR 结构为通道 [32,64,128,256,448]、4×4 空间读出、256 个隐藏神经元，共 4,679,375 个物理参数；竞争组大小 16，16 个 STDP 行、240 个 BP 行，按完整组分工。原图/水平翻转平均 logits，各自 T=8，视图之间重置膜电位。双视图约增加一倍推理计算；同 checkpoint 单视图 FP32 为 74.90%，不能把 76.20% 当成单视图结果。

CIFAR 推理量化只读取完整 TRAIN 的隐藏输入均值/二阶矩，不使用标签、不训练权重；34 个候选在完整 VALID 上按精度、再 loss 选定。2-bit 的逐行裁剪比例为 0.5，4-bit 的比例按 TRAIN 输入方差加权误差逐行选择，二者使用完整偏置补偿 bq=b+(W−Wq)μ。权重码为 unsigned 2/4-bit、逐输出行浮点 scale、零点 0；卷积、分类头、bias、scale 和神经元参数仍为浮点。偏置可有符号；非负约束作用于突触权重。

同一 FP32 模型原始 absmax 量化的 2-bit/4-bit 为 60.20%/68.80%，校准后为 74.40%/75.70%，FP32 精度完全不变。2-bit 仍低于 75% 门槛 0.60 个百分点；当前已完成的目标是四项 FP32 达标及低位宽改善。

卷积骨干采用物理非负 E/I 双权重银行与 PLIF；隐藏 LIF 使用 m←0.9m+I、达到阈值发放并减去阈值，atan surrogate。WTA 只决定隐藏层的更新资格，每个样本/竞争组只选首次 winner，其他发放保留，卷积和分类头继续 BP。隐藏 BP/STDP 行互斥，非获胜行和其优化器状态受到保护。

指数稳态调节逐步执行 offset←exp(−1/20)offset+0.05×batch_winner_count，阈值为 clip(1+offset,0.1,10)，跨样本持久保存；VALID/TEST 使用 checkpoint 中的阈值并冻结调节。另有按目标发放率调节的可切换实现，本表选用指数模式。

前向保持原生 spike coding/活动，局部学习才用独立 RNG 对池化活动按 x/A 采样 Bernoulli 事件，A=1；这不是池化前真实 spike 的时序重建。MNIST/N-MNIST 的 T=4、CIFAR T=8、Gesture T=16，具体输入分辨率和事件归一化以各条件 config.json 及冻结 data.py 为准。

LUT：5-bit countdown，event 置 31、静默减 1 到 0，每次 presentation 从 0 开始；刷新先于首次 winner 查表。U=0 的 c=0，否则 c=0.9^(31−U)，L=round((0.022c−0.012)/0.001)。完整 32 项见 STDP_LUT.csv。CIFAR stdp_lr=0.01，基础浮点 Δw=1e−5×L；soft bounds 与 winner 行归一化后变化可更大。表为 signed 5-bit 系数、逻辑 160 bit，不代表实测硬件面积，也不是权重码增量。训练前向/反向均为浮点。

## 独立复现

冻结的模型代码、配置、checkpoint、CIFAR 校准统计和量化设定已复制到 conditions/，不依赖可变的工作区模型源码或旧实验权重路径。依赖本工作区数据集及 CUDA 环境；原配置中的数据路径保留。一次只重放一个数据集，三个精度模式全部检查。输出文件必须不存在。

```bash
TMPDIR=/tmp .venv/bin/python /home/zhangyufan/Project/1_2026Q3_T22_SNNCIM/2_Ruiqi/Algorithm/results/accuracy/stdp_lut_optimization_20261007_final/reproduce.py --bundle /home/zhangyufan/Project/1_2026Q3_T22_SNNCIM/2_Ruiqi/Algorithm/results/accuracy/stdp_lut_optimization_20261007_final --dataset CIFAR10DVS --out /tmp/final_CIFAR_replay_NEW.json
```

其他数据集将 --dataset 改为 MNIST、NMNIST 或 DVSGesture。benchmark_manifest.json 记录全部文件 SHA256 和参数/样本/精度/视图条件；audit.json 保存来源审核。最终副本的重放结果保存到 replays/。

## 训练与实验记录

本轮重训入口是 experiments/stdp/run.py 的 lut_native_trace 条件；CIFAR 从头训练使用 --spatial-student --from-scratch --epochs 40 --patience 6 --stdp-lr 0.01 --bp-rate-scale 1.5 --initial-row-sum 4 --augment-spatial --teacher-source original_cached --stdp-granularity group。实际参数/学习率/冻结策略/教师缓存哈希以来源 protocol.json、training provenance 为准；config 中旧恢复预算不是本轮预算。教师仅参与 TRAIN，推理不包含大教师。

完整结果表和 2×2/4×4 从头训练的 VALID 曲线见 history/README.md。前向随机编码破坏初始化精度、竞争组配置未传入而作废的 v3，以及结构直接迁移效果很差而早停的 v8 均保留原实验记录；未作有效成功证据。当前结果为单种子，形状、蒸馏、分类头拟合、双视图与量化校准共同改变，不能单独归因于 LUT。每轮 checkpoint/量化设定依据 VALID，但 TEST 门槛触发进一步开发，整项迭代不是一次性盲测。

回归检查覆盖 LUT/事件 RNG、旧 checkpoint 兼容、WTA/BP/STDP 保护、稳态、非负骨干、结构迁移、分类头拟合、双视图重置和推理量化补偿。67 项通过。
