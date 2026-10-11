# 自建网络的跨数据集精度目标

2026-10-04 用户确认先追求 FP32 精度，再改善推理时的 2-bit／4-bit
隐藏线性层权重量化。完整测试集的阶段目标为 MNIST 99%、N-MNIST 98%、
CIFAR10-DVS 75%、DVS128 Gesture 95%。目标尚未达到。

卷积、隐藏线性层和分类头的突触权重继续非负。卷积和分类头使用 BP，
隐藏层保留 WTA 与 BP/STDP、指数稳态调节。训练前向和反向均为 FP32，
只有推理时量化隐藏线性层。各数据集独立训练，不用 MNIST 权重初始化事件网络。

上一版本的完整测试 FP32 精度分别为 98.95%、60.11%、10%、16.67%。
N-MNIST 在第 5 轮失活，其结果来自较早的最佳验证检查点；后两项仅达到
单类别预测基线。原始日志、停止原因、检查点和源代码快照保留在
`results/accuracy/custom_exponential_eval_20261004`，不作为达标结果。

## 第一轮结构修正

新自建 CNN 使用 `[64,128,256,512,1024]` 五级卷积。每级有独立的
非负兴奋／抑制卷积权重，两支路分别计算后相减；不会把有符号的有效
卷积核作为非负突触权重执行。BN 稳定电流，PLIF 学习膜时间常数，
最大池化减少分辨率。空间读出保留 2×2 或 4×4 特征，再接两层 4096
宽的 WTA 隐藏层及分类头。初始规则为 STDP→BP。

按所有卷积算子输出通道与隐藏层宽度求和为 12160，其中兴奋与抑制
两支卷积分别计数；真正的 PLIF/WTA 发放通道宽度之和为 10176，
两者随检查点分别报告。前一口径与旧自建网络的 12352 同量级。
参数量增加：4×4 空间读出的 Gesture 版本约 96.48M 参数，旧版约
42.59M；本轮优化对象是精度，不能据此声称硬件开销不变。

| 数据集 | T | 输入分辨率 | 空间读出 | batch | 初始 lr | 最大轮数 |
|---|---:|---:|---:|---:|---:|---:|
| MNIST | 4 | 28 | 2×2 | 64 | 0.0005 | 60 |
| N-MNIST | 4 | 34 | 2×2 | 32 | 0.0005 | 80 |
| CIFAR10-DVS | 8 | 64 | 4×4 | 16 | 0.001 | 200 |
| DVS128 Gesture | 8 | 64 | 4×4 | 16 | 0.001 | 160 |

MNIST 保留 Poisson 编码。事件输入从原有 T=16 缓存按相邻时间 bin
求和，降采样按不重叠像素块求和，两个步骤都在 log1p 压缩前保持
总事件计数；没有截去后半段事件。输入变换和原始缓存形状单独记录。
数据集 train/validation/test 样本划分保持不变。

BN 初始统计来自 8 个训练批次，随后用一个训练批次校准隐藏层电流。
这两步都恢复训练 RNG 和 loader RNG，不使用验证／测试输入，也不更新
突触参数；BN 校准仅初始化运行统计，隐藏校准仅设置初始权重尺度。

真实 GPU 检查确认前后向有限、卷积和分类头梯度非零。Gesture 的
6 轮全训练／全验证资格检查取得 36.67% 最佳验证精度，未访问测试集。
这只确认开始学习，不是最终精度。

## 选择与下一轮调整

短训练资格检查只看训练动态与验证结果，使用 `--validation-only`。
完整训练在每轮完整验证集上选择最佳检查点，数值异常立即停止，
20 轮验证停滞可提前结束。训练完成才评测完整测试集并报告 FP32/2/4-bit。
未达目标会记录为未达标，不以训练准确率、验证探针或“任务完成”状态
代替达标。

后续调整由验证曲线、发放率、BP 梯度、WTA 更新覆盖率及泛化差距驱动。
允许按数据集调整 T、读出空间大小、结构、学习规则与训练超参数，
每次变更使用独立配置与输出目录，保留前次失败证据。候选筛选不使用
测试集选择超参数。

PLIF/卷积骨干的参考是作者的
[官方实现](https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron)
及 [SpikingJelly 实现](https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/activation_based/model/parametric_lif_net.py)。
它们没有本实验的非负突触和 WTA/STDP 组合约束，也有不同的模型选择
协议，公开精度不能视为本实现复现结果或直接同条件比较。

## Gesture 泛化与 BN 统计修正

第一轮新结构的 Gesture 已完成：最佳随机样本验证精度 95.00%，
完整官方测试 FP32/2-bit/4-bit 为 86.81%/86.81%/86.46%，仍未达到 95%。
后续候选把 4 位官方训练参与者整体留作验证，训练/验证/测试样本数
分别是 972/204/288，三个集合的参与者互不交叉。官方测试样本和类别不变。

候选使用原始 128×128、T=16、batch=8，隐藏层 dropout=0.5，
仅训练时做平移、旋转和尺度增强；不翻转 Gesture 输入。隐藏层每 64
个神经元形成 WTA 竞争组，以扩大有效更新覆盖，仍然保留 STDP→BP。
只看验证的 8 轮资格检查最佳精度为 40.69%，未访问测试集。

固定检查点后，仅用 16 个未增强训练批次重估 BN 运行均值与方差：
第 5 轮检查点验证精度 40.69%→44.12%，第 8 轮检查点 24.02%→44.12%。
这说明后期验证波动包含 BN 统计偏移，但并不代表最终精度达标。
重估不更新参数、阈值或优化器，并恢复 RNG 与 loader 状态。新候选
在验证探针和每轮完整验证前执行此步骤；保存检查点后的统计在测试时冻结。

相关运行源代码分别冻结在 `custom_performance_candidates_20261004/source_v2`
至 `source_v4`，活跃训练使用的根目录源码保持原版本。改进候选 80 项
核心与功能测试通过；源码、配置哈希及原始诊断文件均随实验保留。
完整 Gesture 新训练已启动，CIFAR10-DVS 原始分辨率＋增强候选先进行
8 轮验证检查。多个变更同时筛选，只报告组合效果。

跨版本实时状态见
`results/accuracy/custom_performance_candidates_20261004/progress_report.md`。
资格检查与完整测试分开列出，达标判断只使用完整测试结果。

## 已达到的完整测试目标

N-MNIST 的第一轮新结构已完成 24 轮训练，并按完整验证精度选中第
19 轮检查点（文件中 epoch 从 0 开始）。完整 10000 样本测试结果：
FP32 9875/10000=98.75%，2-bit 9877/10000=98.77%，4-bit
9878/10000=98.78%。N-MNIST ≥98% 目标已达到，其他三项尚未全部达到。
该检查点独立从零训练，使用 T=4、原生 34×34 与 log1p 事件变换。
量化结果仅报告本次测量，不据此声称量化提高了模型质量。

提前停止的有效改善门槛为 0.2 个百分点，连续 20 轮未达到此改善
门槛会停止；保存的最佳检查点仍按绝对最高的完整验证精度选择。

较小 Gesture 候选 `[64,128,256,384,512]` 已通过单个真实训练样本的
前后向与更新检查，无测试集访问。参数量 56.44M、发放通道宽度之和
9536、所有卷积算子通道与隐藏宽度之和 10880。与当前完整 Gesture
候选相比，参数量减少约 42%，真正发放宽度减少约 6%。这只是约束和
梯度检查，尚未验证其精度收益。

MNIST 优先调度保持主实验的配置、源码和输出目录完全相同。只有
CIFAR 原始分辨率短训练已退出且 GPU 空闲显存不少于 5 GiB 时才会
启动；暂挂主队列驱动器防止重复启动 MNIST，已有训练不暂停。
MNIST 结束后自动恢复队列驱动器，并从真实完整测试文件接续状态。

## 混合神经元学习规则候选

`source_v5` 增加可切换的 `hybrid` 隐藏层规则。每层 4096 个神经元
中，索引能被 4 整除的 1024 个使用 STDP，其他 3072 个使用 BP；
每组 64 个神经元继续通过 WTA 选择更新资格。BP 梯度、参数和 Adam
行级动量只更新获得资格的 BP 神经元；STDP 只更新获得资格的 STDP
神经元。误差仍通过所有神经元传向卷积，分类头继续使用 BP。

83 项相关测试通过，覆盖稠密输入梯度、两种更新的行级隔离、已有
Adam 动量、非赢家保护、旧的纯 BP/STDP 模式及仅推理量化。真实 GPU
训练样本检查中，两层分别有 49/15 和 44/20 个 BP/STDP 赢家，梯度
非零、所有突触权重非负、非赢家未改变。该检查不是精度测量。

较小拓扑的原 STDP→BP 与混合规则候选分别进行相同的 8 轮全训练/
全验证检查，共用 T=16、128×128、batch=8、同一参与者划分、增强和
BN 重估。队列先保证 MNIST 分配显存，再要求至少 11 GiB 空闲显存；
两个候选依次运行，发生训练错误会保留证据并停下该候选队列。

CIFAR 原始分辨率增强候选已完成 8 轮验证，最佳 30.67%，无测试集
访问。暂不据此切换主 CIFAR 长训练配置；两者预算和 LR 曲线不同，
不把短训练分数的差异作为最终性能排名。

## MNIST 时间步诊断

训练中的第 11 轮最佳检查点被独立冻结，使用同一 T=16 Poisson 流的
T=4/8/16 前缀在完整 6006 样本验证集上比较，阈值与参数保持固定。
编码 seed=3042，验证精度分别为 5939/6006=98.88%、5925/6006=98.65%、
5907/6006=98.35%。T=8 相对 T=4 修正 11 个样本、退化 25 个；T=16
修正 10 个、退化 42 个。全部诊断仅使用验证输入，未访问测试集。

当前训练与推理继续采用 T=4。该结果只适用于这个训练 T=4 的早期
检查点，不能推断重新以 T=8 训练的结果。如果将来改变最终推理条件，
需在最终选定检查点上重新验证并明确记录训练/推理时间步。

## MNIST 完整测试与重放

MNIST 已完成 25 轮训练，按完整验证精度选中第 24 轮检查点：
验证 99.15%，完整 10000 样本测试 FP32 9923/10000=99.23%、
2-bit 9927/10000=99.27%、4-bit 9925/10000=99.25%。保持 T=4、
28×28 Poisson 编码和原 STDP→BP 配置；从零训练，没有使用测试集
选择检查点或修改训练条件。目前 MNIST 与 N-MNIST 两项 FP32 达标。

独立重放采用正式 `complete_test_evaluation` 的固定 seed=1042，
三种精度的正确数、预测类别直方图和 loss 均与首次测试完全相同。
证据保存在该运行的 `test_replay.json` 与 `test_rng.pth`。初次重放
错误使用末轮训练 RNG，得到 99.17%；该检查错误保留为
`test_replay_incorrect_rng_attempt.json`，首次正式测试未被覆盖。

优先训练结束后主队列驱动器已恢复，已有 CIFAR 与 Gesture 训练继续。
较小 Gesture 的 STDP→BP 候选已完成相同 8 轮全训练/全验证检查，
最佳参与者独立验证精度 48.04%；混合规则候选继续按相同条件检查。
资格检查未访问测试集，不能据此认定 Gesture 达到 95% 目标。

N-MNIST 的保存检查点也已独立重放：正式 seed=1042 下 FP32、2-bit、
4-bit 的正确数、预测类别直方图和 loss 均与首次完整测试完全一致。
重放用于验证结果，不用于选择编码种子或模型。

## CIFAR 泛化候选

主 CIFAR 第 29 轮训练精度 97.63%、验证 62.33%，此前最佳完整验证
63.89%，显示出训练/验证差距。登记两项相同预算候选，均从零开始：
较小 `[64,128,256,384,512]` 骨干、T=8、64×64、隐藏 dropout=0.5、
每 64 个神经元 WTA、训练几何增强、16 批未增强训练输入 BN 重估。
第一项采用 STDP→BP；第二项仅将首个隐藏层改为 75% BP/25% STDP，
第二个隐藏层保持 BP。两项均进行 8 轮全训练/全验证，未访问测试。

队列等待 Gesture 短训练释放资源及至少 11 GiB 空闲显存。每项在正式
启动前，需用其实际 batch=16 完成训练输入上的浮点前后向、Adam 与
STDP 更新检查，验证非赢家保护、非负突触和显存余量；临时模型丢弃，
不会作为训练初始化。现有 CIFAR 与 Gesture 长训练继续，运行源码冻结。

## 验证停滞时的学习率与 BP 覆盖

较小 Gesture 的两层混合候选完成同一 8 轮资格检查，最佳验证
46.08%，原 STDP→BP 为 48.04%。没有观察到该两层混合组合提高验证
精度，暂不直接推广为长训练配方；单次短训练也不足以声称因果优劣。

独立 `source_v6` 增加可切换的 `lr_schedule=plateau`，默认旧的 cosine
行为不变。新模式根据每轮完整验证 top-1，在停滞时降低 BP 优化器
学习率，阈值为精度分数的绝对改善量。调度器与优化器状态一起保存，
恢复运行要求其配置一致；验证探针和测试分数不参与 LR 调整。
调度日志明确记录改变前后的学习率，既有数值/静默/坍塌/早停保护保留。

已登记原生 128×128、T=16、batch=8 的较小 Gesture 新资格候选：
竞争组 64→16，首隐藏层 3840 BP/256 STDP，第二隐藏层全 4096 BP，
学习率 factor=0.5、patience=3、min_lr=1e-5、threshold=0.002。
该组合同时改变更新覆盖与调度，只以完整验证评估组合行为，不分解
声称单因素收益。保持训练增强、16 批 TRAIN-only BN 与指数稳态调节。

新版本 84 项不同检查通过，覆盖完整验证驱动的 LR、调度状态恢复、
严格配置恢复、默认 cosine 一致性、资格检查排除测试数据及现有突触
约束。队列等待 CIFAR 已真实分配显存且至少 13 GiB 空闲，再用实际
B8 训练批次检查后才启动 8 轮资格训练。测试通过与候选登记均不代表
Gesture 精度达标。CIFAR 第一个较小增强候选已通过实际 B16 检查并训练。

## 第 88 轮 Gesture BN 数据量对照

固定第 88 轮检查点和完整 204 样本参与者独立验证：保存的 16 批
统计为 175/204=85.78%、loss=0.5282；仅用未增强 TRAIN 的 64 批
(512 样本) 重估得到 172/204=84.31%、loss=0.5588；122 批完整
TRAIN (972 样本) 重估得到 170/204=83.33%、loss=0.5754。
三种条件权重、阈值均未改变，无 TEST 访问。扩大重估样本量没有改善
该检查点的验证结果，保留当前 16 批策略。不能据此推断其他统计方法
或其他检查点的效果。检查点、原始完整分数与参数不变检查保存在
`gesture_bn_population_diagnostic`，不覆盖主训练或其后续检查点。

## Gesture 完整测试与 WTA 同步发放对照

参与者独立验证的原生 Gesture 长训练已结束，选择第 88 轮检查点。
完整官方 288 样本测试 FP32 257/288=89.24%，2-bit 与 4-bit 均
262/288=90.97%。FP32 距 ≥95% 所需的 274 个正确样本还差 17 个；
该配置未达标，原始配置、检查点和测试结果保留。

训练批次和代码检查发现：旧 WTA 在同一步中多个神经元发放时，总
按最低索引选一个，混合规则的 STDP 神经元位于每个周期的第一个索引。
新增可切换的 `hidden_winner_tiebreak=margin`：仍由最早发放时刻确定
赢家，但在该时刻已发放的候选中选择复位前 `V-Vth` 最大者；两者
严格相等时才使用较小索引。赢家资格控制突触写入，向卷积的误差保持
稠密；推理阈值冻结，旧 `index` 模式仍是默认值。

`source_v7` 87 项检查通过，涵盖同步发放选择、首次发放优先、真实
相等、无赢家、动态阈值差、BP 行梯度与原规则兼容。真实 B8 训练
检查中首层 2048 次赢家资格为 BP 1900、STDP 148（7.23%）；BP
具有资格的不同神经元 1291 个，STDP 为 95 个。峰值 6237 MiB，
所有突触非负、非赢家未写入。资格检查与测试通过不是精度收益证明。

原 `index` 候选已在显存释放后启动 8 轮训练，因此保留为对照；
新的 `margin` 候选在它结束后启动。两项具有相同 T=16、128×128、
batch=8、较小骨干、每 16 神经元 WTA、首层 3840 BP/256 STDP、
第二层全 BP、增强、BN16、plateau LR、seed42 与 204 样本完整验证。
一次相同 seed 的比较只说明本次行为差异，不据此声称通用性能收益。
新候选显存门槛根据实际 B8 峰值 ×1.25＋1024 MiB 设为 8821 MiB，
并等待 CIFAR 已完成显存分配；已有长训练与配置保持其冻结协议。

## 同条件 WTA 对照结果与行权重总量

8 轮原生 Gesture 对照完成：index 最佳 96/204=47.06%，margin
111/204=54.41%。两项均为完整验证，无 TEST 访问。margin 的最佳
结果在第 8 轮，旧选择的最佳在第 5 轮；只报告本次同 seed 比较。

较小增强 CIFAR 的 STDP→BP 与首层 75% BP/25% STDP→BP 也完成
同条件 8 轮验证，最佳分别 28.44% 与 27.89%，未看到后者带来提高。
这些短训练的 cosine 总周期为 8，不把它们与 200 轮配方直接排名。

第 8 轮训练检查点显示部分 BP 行的权重总量大幅漂移：Gesture
第二隐藏层总量/初始化目标的第 90 百分位为 23.41 倍，最大为
90.64 倍；单个权重没有达到上限 1。原始分组统计保存在
`weight_mass_diagnostic.json`。这说明单个权重 clipping 不能限制
整行总输入强度；其对精度的作用需要通过验证对照确认。

新候选启用既有 `bp_row_normalize=true`，只将获得 WTA 资格的 BP
行投影回各自校准后的初始非负 L1 总量。STDP 继续使用独立软边界
和行归一化，非赢家权重与 Adam 行状态仍保护。实际配置 B16/B8
检查通过，行总量比最大约 1.00000024，权重非负、BP 梯度非零。

## 可续跑的资格前缀

`source_v8` 新增 `--qualification-epochs N --validation-only`，仅
限制本次训练窗口，`--epochs 200` 的总预算与 LR 调度不变。资格
阶段保存完整 latest/best、优化器、调度器、阈值、TRAIN loader RNG
和全局 RNG；全验证决定下一步，资格阶段不读 TEST。

CIFAR 新前缀为 20 轮，使用 T=8、64×64、batch=16；Gesture 为
16 轮，使用 T=16、128×128、batch=8。两者使用较小骨干、margin
WTA 每组 16、首层 3840 BP/256 STDP、第二层全 BP、dropout=0.5、
训练增强、BN16 和 plateau LR。两个数据集分别从零训练，不相互初始化。

90 项检查通过，包括小型 CPU SNN 在 cosine 和 plateau 下的前缀
续跑与连续训练的模型/优化器/调度/RNG/loader/保护状态完全相同。
CUDA 续跑另行核验；这些检查不表示大模型达标。前缀验证后继续训练
时使用自己的 latest.pth 和原 200 轮预算，在独立 full-case 输出目录
保存证据，并在完整测试前明确记录所选配方；不重新训练前缀，也不
覆盖资格阶段的 validation.json。

## 归一化前缀结果与自身检查点接续

Gesture 归一化前缀16轮完成，最佳108/204=52.94%，第13轮；前8轮
最佳95/204=46.57%，低于不归一化同条件8轮111/204=54.41%。本次
归一化未带来验证提高，暂不延长。保留配置与训练证据；CIFAR归一化
候选继续完成预定20轮窗口，无TEST访问。

新完整Gesture配方选择此前margin-WTA且不归一化的配置，接续其
自己从零训练的8轮latest.pth，没有重新训练前缀。source_v9提供
明确的--extend-plateau-budget，将总预算8→200，保留模型、阈值、
Adam、LR、监控、loader和RNG，仍受原早停规则控制。plateau日程
当前LR与剩余总轮数无关，cosine禁止这样延长。除out_dir外学习配置
完全相同。dataset_assignments.json在新完整TEST前登记该配方，
选择依据是完整验证而非测试分数。

原普通CUDA比较在两个卷积张量出现至多3.49e-10差异，全局/loader
RNG与LR状态相同；启用确定性cuDNN后模型与优化器数值完全一致。
另发现整个检查点映射到GPU会把Adam计步张量也迁移，新版本先在
CPU加载，再由模型/优化器正确安置。96项CPU/原生CUDA检查通过，
包含活跃BP/STDP/阈值、cosine/plateau恢复以及plateau明确延长后
与连续训练逐位一致。该证明使用测试内确定性cuDNN；正在训练的
模型保留原数值内核，不声称普通CUDA任意接续轨迹都逐位相同。
resume_gpu_diagnostic保留量化差异诊断和第二次失败检查点；第一次
临时检查点在归档前被pytest自动保留策略删除，不能声称其仍在磁盘。

自身第8轮检查点真实B8 TRAIN批次预检通过，BP/STDP均有更新、
突触非负、非赢家权重不变，Adam计步保持CPU，峰值6669MiB。
接续训练已启动，显存门槛9361MiB，实时监测及原早停规则生效。
最终仅由完整288样本TEST判断Gesture≥95%目标；检查和验证进度
均不替代最终测试精度。

## CIFAR 过拟合诊断

主CIFAR训练第71轮训练99.62%、loss0.011，完整验证64.22%、loss1.993；
最佳第53轮66.11%。固定该最佳检查点，仅用未增强TRAIN重估BN：
保存统计595/900=66.11%；16批TRAIN578/900=64.22%；64批TRAIN
583/900=64.78%。参数/阈值逐项不变，无TEST输入访问，未见单独BN
重估提高。保留源检查点和完整分数，下一步针对训练/验证差距使用
自己的从零训练检查点进行明确记录的增强/正则化训练阶段。

## CIFAR 完整测试与自己的增强/正则化新阶段

原主CIFAR在第73轮因20轮验证平台早停，选择第53轮。完整1000样本
TEST为FP32 668/1000=66.80%，2-bit 677/1000=67.70%，4-bit
669/1000=66.90%，未达≥75%。原测试记录与检查点保留。新阶段选择
依据仍为此前训练/完整验证差距和TRAIN-only BN诊断，没有按测试
类别或错误样本调参。

source_v10验证了明确的--own-init-checkpoint新阶段入口：只有同数据
集、同形状、同输入编码、同划分、同STDP/BP样式、FP32且非负的
自建、来自从零训练的检查点可用；保留权重与稳态阈值，跳过重新
校准输入/隐藏权重尺度。新阶段明确重新建立优化器、LR及监控，不
将它包装成原训练轨迹连续续跑。一般--resume仍保留严格身份规则。
106项CPU/原生CUDA检查通过。

仅调整增强/正则化的global-WTA阶段未启动训练。真实B16预检显示
第二隐藏层仅1/4096个BP赢家，梯度范数2.675e-5，行L1/初始目标
最大298.25。保留该预检记录，据此加入分组WTA：source_v11只
扩展新阶段可更改资格规则，保持STDP/BP更新样式及其它基准约束。
107项CPU/原生CUDA检查通过。

实际训练候选为cifar_regularized_broadwta_own_phase_prefix：从自己
第53轮冻结检查点开始，T8/64×64/B16和原骨干/4096×2隐藏层不变，
首层全STDP、次层全BP不变；WTA全局/index→每16个/margin。
只增强TRAIN：平移0.0625、旋转0、缩放0.95..1.05、水平反射0.5；
dropout0.2→0.5，LR0.001→0.0001，plateau因子0.5、patience5、
min_lr1e-6，早停patience15/min10。BN保持原训练/保存统计策略，
暂不使用单独验证未提高的额外BN重估。多个改动组成新阶段，不声称
单项精度因果收益，也没有外部预训练模型。

真实B16预检中第二隐藏层2413个BP赢家、2406行实际变化，梯度
范数19.324；首层2317个STDP赢家。所有突触非负且非赢家未写入，
峰值2278MiB。新阶段总预算80、资格窗口12，显存门槛6144MiB，
已启动实时监测训练，无TEST访问。表内轮数指新阶段轮数，原53轮
计算不会重做。若完整验证支持延长，将自己的latest.pth按原80总
预算接续，在另一个输出目录预先登记完整测试配方。

CIFAR归一化20/200前缀已结束，最佳368/900=40.89%，无TEST。
暂不延长，优先推进自身第53轮检查点的增强/分组WTA阶段；保留原
20轮检查点和200预算，可按后续完整验证需要继续，无需重训前缀。
当前新阶段3轮最佳572/900=63.56%；Gesture完整接续33轮最佳
154/204=75.49%。两项均仍在训练，不能据验证进度判定完整TEST达标。

## 后续完整结果与固定最终拟合（2026-10-04 18:37 CST）

以上训练进度已成为历史记录。CIFAR新阶段12轮资格结束后，完整
验证614/900=68.22%超过原66.11%，按预登记条件保留同一新阶段
80轮总预算；恢复原优化器/调度器/阈值/RNG/早停状态，未
重训前缀。第21个新阶段轮因平台早停，选择第6轮，完整TEST1000
样本：FP32 678/1000=67.80%，2-bit679/1000=67.90%，4-bit
672/1000=67.20%。相比原主实验FP32提高1个百分点，仍未达75%。

Gesture紧凑hybrid/margin候选完整训练于第81轮平台早停，选择第61
轮；完整VAL166/204=81.37%，独立TEST FP32 266/288=92.36%，
2-bit236/288=81.94%，4-bit260/288=90.28%。未达95%；FP32优先，
现阶段不根据量化测试类别或误分类样本调整配置。

source_v12新增固定最终拟合：在完整VAL选择配置与检查点后，将官方
TRAIN内972训练+204验证合并为1176样本。原完整VAL更高的自建
gesture_subject_native_bnrefit_full第88轮(175/204=85.78%)为已
预登记前驱；保持所有学习配置与原cosine200调度状态，固定再训练
ceil(0.1*88)=9轮至第97轮。合并后不做验证，不按训练或测试成绩
选择端点。官方独立TEST288样本仅在端点进行FP32/2/4bit评测。
checkpoint禁止再进入原留出VAL模型选择，避免训练数据复用泄漏。
23项相关CPU/原生CUDA检查通过，真实1176TRAIN/B8预检通过；
检查仅证明机制接线，不代表测试目标达成。原所有失败结果保留。

新增source_v13自建残差候选仅做CIFAR完整VAL前缀：四阶段宽度
32/64/128/256，每阶段两块，电流shortcut在PLIF前相加，输出仍
为spike；卷积含非负E/I两银行，形状变化使用非负1x1两银行投影。
计所有物理卷积输出通道+4096x2隐藏层为12992，原12160，增加
6.84%；实际脉冲宽度10144接近原10176；参数目标小于4500万。
首隐藏3840BP/256STDP，第二全BP，group16/margin、指数稳态、
推理仅隐藏线性权重2/4bit约束保持。输入CIFAR T16/64分辨率，
从零训练，不载入外部或既有模型权重；其它同一训练/验证/测试划分。
总预算200、资格15轮，完整VAL900；前缀禁止TEST。多个条件组成
候选，不能把精度变化归因于单个因素；据完整VAL曲线决定接续。

固定Gesture最终拟合已完成：FP32 250/288=86.81%，2-bit
236/288=81.94%，4-bit245/288=85.07%，未改善、未达标。该阶段
没有新增验证成绩，表中的85.78%仅为合并前前驱成绩；失败证据和
独立测试均保留。后续新模型仍只使用未参与训练的原204完整VAL
选配置，不能用已在1176样本拟合的检查点返回该留出验证选择。

source_v13最终119项CPU/原生CUDA检查通过(12.32秒)，真实CIFAR
B16/T16/64预检峰值2438MiB、Gesture B8/T16/128峰值4434MiB；
两者隐藏层BP/STDP赢家实际写入、非赢家保护与非负约束均通过。
正式训练与预检均在隐藏校准前执行相同TRAIN-only初始BN估计。
两项已从零开始训练，CIFAR前缀15/200、Gesture前缀20/200，实时
打印loss、发放率、梯度、赢家及类别坍塌探测；没有前缀TEST。
Gesture的残差候选保持原参与者划分、T16/128/B8、同一增强和
16批TRAIN-only BN重估，使用原LR0.001的cosine200配方；架构
改变，因此此处不称作旧模型续训。两者预检和测试都不代表精度达标。

## 残差候选的早期否定结果与修订（2026-10-04 19:12 CST）

source_v13的窄/深电流残差配方未改善：Gesture20轮完整VAL最佳
97/204=47.55%，期间TRAIN loss1.68升到2.98、VAL降到7.35%，
随后恢复部分；抽样梯度范数达1.6万，暂不接续。CIFAR第7轮保存
完整检查点后主动停止：完整VAL最佳156/900=17.33%、TRAIN不足
20%、loss约2.18，数轮改善弱、梯度明显放大。只停止该候选，目标
仍推进；未训练的剩余窗口不算完成，两个前缀都未访问TEST。

Gesture第16轮模型的64个TRAIN样本诊断显示，第二隐藏层327个
不同赢家、平均组赢家占比90.54%；只在临时模型将delta.05→.2，
不同赢家340、占比89.60%，发放率下降。没有优化参数或读取VAL/
TEST；为记录BP层winner而保留trace的临时调整已验证前向等价。
这些机制数据不能证明精度改善，因此没有采用更大delta，也没有
把行权重和增大直接当成饱和并加入任意行质量上限。

source_v14为自己的新配方加入SEW式脉冲相加shortcut，参考
[Deep Residual Learning in Spiking Neural Networks](https://arxiv.org/abs/2102.04159)
和SpikingJelly官方实现。各PLIF节点仍输出0/1，两个分支相加为
spike count，直通路径不额外经过surrogate导数。检查零残差分支时
恒等输入与精确单位输入梯度，形状变化含E/I非负投影和PLIF。
这证明实现路径，不能证明本项目精度收益。122项全回归13.07秒，
额外1项连接身份保护2.12秒，共123项唯一检查通过；旧current模式
保留，resume禁止更改连接类型。

新候选四阶段64/128/256/512、每阶段一块，主路径9个spiking节点，
连同投影12个；物理卷积通道+隐藏宽度13952，较原12160高14.74%；
实际spiking宽度11072，较原10176高8.81%，仍为相近规模。CIFAR
60170134参数、Gesture60174231参数，原大网络约9600万参数。
降低LR.001→.0005；CIFAR T16→原T8，Gesture仍T16，原增强、
dropout.5、group16/margin、指数delta.05/tau20与划分保持。两者
从零开始，CIFAR15/200、Gesture20/200完整VAL资格，无外部权重、
前缀TEST。多个变化作为一项候选，不声称单因素性能因果收益。

另外登记同骨干CIFAR对照，仅首隐藏hybrid→全STDP，第二层仍BP，
其余配置、初始化种子、数据划分、15/200预算完全相同。真实首批
TRAIN前向loss2.30360055、卷积/分类头/第二隐藏层反向梯度与控制
逐项相同，首层STDP赢家行和保持初始目标；正式训练已启动。
这用于区分局部更新方式的影响，不根据旧TEST类别或误分类调参。
当前全部为训练/验证进度，完整TEST目标仍只有MNIST与N-MNIST达成。

2026-10-04 19:34 CST：用户确认先让FP32达标，再改善2/4bit。
CIFAR匹配15轮资格已完成：首层hybrid最佳224/900=24.89%，
首层全STDP最佳257/900=28.56%；两项均不做前缀TEST，不追加
长训练。Gesture新脉冲残差20轮完整VAL最佳118/204=57.84%，
晚期仍改善；新目录保留所有训练状态及原cosine200预算，固定
再观察20轮至第40轮，仍只用原204留出VAL。

冻结CIFAR卷积编码器的非负浮点线性分类器诊断仅使用TRAIN8100/
VAL900。新脉冲残差编码器固定20轮探针VAL25%；原自建第53轮
编码器探针TRAIN99.90%而VAL61.78%，提示泛化不足。探针超参数
未经优化；旧编码器的原始尺度分类头折叠FP32等价检查失败，保留
失败记录，不能据此声称部署分类头有效或完整benchmark结果。

因此新登记cifar_regularized_root_stdpbp_from_scratch_prefix：
恢复原非负E/I卷积宽度64/128/256/512/1024与4096x2隐藏宽度，
从零训练，没有已训练权重。T8/64/B16，首层STDP、第二BP，
group16/margin、指数delta.05/tau20；从第一轮使用dropout.5、
轻微位移/缩放与水平翻转、Adam weight_decay1e-4、LR.001。
固定20轮资格，cosine200总预算，完整VAL900和原划分保持。
这些变化组成一项候选，不归因于单个因素。预检临时更新已丢弃；
实际训练FP32、实时loss/梯度/发放/WTA/类别坍塌监测。两项新实验
仅验证阶段，完整测试FP32目标仍为2/4，不声称已经达标。

2026-10-04 19:52 CST：Gesture新脉冲残差已完成40轮完整VAL资格，
最佳134/204=65.69%，未访问TEST；仍低于自己的既有验证方案，
暂不继续。CIFAR原骨干正则化从零训练仍在其20轮资格窗口内。

已选自建紧凑Gesture第61轮（原完整VAL166/204=81.37%）的64个
TRAIN样本诊断：第二隐藏层530个不同赢家，76971/91216个活跃
样本-神经元组合不是赢家；参数未变，trace记录前向等价检查通过。
此结果仅提出学习覆盖率假设，不能证明精度改善。

source_v15可切换hidden_winner_scope=cycle/presentation；原模式
保留。cycle在每个仿真周期、每个样本和每个group选一个发放赢家，
同时发放用margin、严格同值选低索引；没有发放则没有赢家。BP
只更新该呈现中曾胜出的神经元，卷积输入误差仍不受WTA门控；
STDP保留真实赢家事件时刻；每次胜利按原delta.05增加阈值，
指数tau20逐周期衰减，推理冻结。训练前向仍保留败者spike。
scope变更不能冒充普通resume，只允许显式自建模型新训练阶段。
最终138项CPU/原生CUDA检查34.43秒通过，含逐周期模式实际BP/
STDP/稳态状态的确定性续训一致性、非赢家Adam/衰减保护和真实
YAML开关加载。两项实际B8/T16/128训练步预检通过；这些检查不
代表精度达标，预检首批没有STDP赢家也按实记录。

匹配阶段从同一个自建、从零来源的第61轮参数和阈值开始；相同
原划分、增强、dropout.5、BN16、group16/margin与神经元BP/STDP
分工，均重新登记Adam/LR1e-4/plateau80预算及监测状态，资格8轮。
配置差异只有winner_scope和输出目录。只用原204完整留出VAL做
选择，不按已见TEST类别/错误选择配置，弱前缀不做完整TEST。

第一次两个新阶段并发启动因显存检查竞争在首个训练步OOM：没有
完成轮次或测试，没有改变前驱检查点。各自attempt1_oom保存
日志和失败记录；修复为全局flock持有至worker结束，同配置重启，
一项训练、另一项排队。现有CIFAR未受影响，Gesture40资格正常
完成。监测报告现在显示排队阶段，并保持目标未达成状态。

2026-10-04 20:04 CST：匹配8轮结果已齐，presentation最佳
162/204=79.41%，cycle156/204=76.47%，都低于共同前驱
166/204=81.37%；两个新阶段均不接续、不访问TEST。这个单种子
窗口不能推出WTA模式普遍优劣。固定静默0/5/10/20/40周期的完整
VAL筛选也没有改善：原大骨干正确175/175/174/175/172，原紧凑
骨干166/164/164/161/158。零静默原成绩精确复现，参数未变；
未采用静默条件，逐周期阶段在诊断到达时尚未完成而被跳过。

CIFAR登记续训门槛：先完成20轮完整VAL900，最佳VAL至少50%，
最后3轮平均TRAIN loss低于第8..10轮，再在新目录保留全部状态
与原cosine200预算接续20轮至第40轮；真实恢复训练步预检通过后
才启动，整个窗口禁止TEST。门槛在第20轮结束前登记，后续仍按
完整留出VAL学习曲线判定，不把资格精度当成75%测试目标达成。

2026-10-04 20:25 CST：CIFAR原骨干正则化已完成40轮资格，最佳完整
VAL574/900=63.78%，较20轮56.78%提高；第40轮TRAIN loss1.0433、
TRAIN64.44%，末期仍学习，不能称为已经停滞。当前成绩仍低于自己的
614/900=68.22%验证方案，也没有新的TEST。先准备保留已有特征的
显式首隐藏层学习分工适配，避免直接重跑更大的从零结构。

source_v16已冻结：前两卷积层可换为因果三抽头的非负E/I Conv3d，
只读当前与前两帧；原自建第88轮卷积参数复制到当前抽头，历史抽头
置零。卷积全部BP、无WTA/量化；原隐藏4096x2、神经元宽度预算、
T16/128、TRAIN/VAL划分、group64/index/presentation、指数稳态
参数保持。增加299520个卷积参数，物理bank宽度12160、发放宽度
10176均保持。自己的已训练权重源于从零训练，无外部模型权重。
148项CPU/原生CUDA检查通过，包括因果性、历史抽头梯度、非负
投影、严格结构迁移、仅隐藏量化、实际BP/STDP/稳态续训一致性。
两项真实B8/T16/128训练步预检均通过，峰值约6.6GiB；临时更新
丢弃。空间对照和时间候选均恢复相同前驱RNG/训练loader，重新
登记Adam LR1e-4/plateau80，先各8轮完整VAL204，禁止资格TEST。

初始真实TRAIN loss并非相同：空间0.1223118603、时间0.07003792375，
原因尚未定位；不能宣称初始FP32前向逐位等价或把后续差异仅归因
于时间上下文。此限制记录在gesture_temporal_matched_protocol.json。
全局GPU锁串行调度：空间worker184773训练、时间supervisor179903
排队，防止此前显存竞争重复发生。当前完整TEST达标仍为MNIST和
N-MNIST两项，其余两项未达标，2/4bit调优继续置后。

2026-10-04 20:45 CST：Gesture时间卷积/空间匹配8轮资格均完成，
最佳173/204=84.80%与170/204=83.33%，均低于原175/204=85.78%。
保留实现、数值限制与负结果，不延长、不评测TEST。

source_v17冻结32项运行源码。显式自建新阶段允许首隐藏层STDP
转为hybrid（按索引每16个保留一个STDP，共256STDP/3840BP），
第二层4096BP；复制所有原参数、阈值与行和目标，没有重新校准。
普通resume禁止改变分工。160项回归与2项真实CPU/原生CUDA新
阶段续训检查通过，涵盖BP/STDP赢家更新、败者/Adam保护、完整
参数/优化器/阈值/RNG/loader/scheduler的续训一致性。CIFAR从
既有最高完整VAL614/900的自建检查点开始，T8/64、原增强与
输入划分保持；纯STDP对照和hybrid首批loss均0.7595491409。
纯STDP对照初次预检因误改无效hybrid周期被严格身份检查拒绝，
恢复前驱原设置后通过；失败证据保留，运行源码未改动。

Gesture采用自身最高VAL的第88轮检查点，T16/128、原增强、
dropout.5与BN16保持；共同改为group16/margin，比较首层STDP
和hybrid，相同新Adam LR1e-4/plateau80与前驱RNG/loader。
两个真实训练步均通过，首批loss相同0.1127085835，hybrid实际
BP/STDP赢家事件1098/84。先各8轮完整VAL204，禁止资格TEST。
这些阶段仍从自己的从零训练祖先来，没有外部训练权重。

CIFAR首层hybrid8轮最佳601/900=66.78%，尚未超过原614/900；
末期有改善，待匹配对照后判定，不把当前窗口宣称成功。第3轮
保存TRAIN状态显示首层行和中位数2.53倍、最大5.17倍，但没有
单个权重达到上界1；只能提出尺度漂移假设，不能称权重上界饱和。
source_v18增加按隐藏层名称开启BP赢家行和投影，先仅hidden1
保留前驱STDP行和目标。171项独立CPU/原生CUDA检查通过，确认
不投影STDP行/败者行/第二隐藏层，实际续训状态一致。真实训练
步与未约束hybrid首批loss和BP梯度完全一致；更新后第一层最大
行和比1.00000024、第二层原比280.435仍保持其原学习策略。该
8轮新候选已登记排队，采用相同祖先、配置及RNG，比较仅首层行和
机制。此前全隐藏层归一化的失败仍保留，没有重用该全层策略。
所有新窗口只用完整VAL选择；FP32完整TEST达标仍2/4，2/4bit
调优继续置后。实时日志和retained_features曲线持续记录进度。

2026-10-04 20:56 CST：CIFAR纯STDP/混合首层匹配8轮已经齐全：
最佳608/900=67.56%与601/900=66.78%，两者均低于前驱614/900。
只是在这一种子/窗口中hybrid少7个正确，不能推出普遍优劣；不评测
资格TEST。首层行和约束窗口仍运行，保留负结果及完整训练状态。
在该窗口结束前登记cifar_next_window_policy：三项8轮都完成后，
若最高VAL至少632/900=70.22%且最近3轮VAL均值至少68%，保留
全部状态和原plateau80预算再观察12轮到阶段20；否则验证原正则化
从零模型40轮的末5轮TRAIN loss低于前5轮、最好VAL至少63%，在
原cosine200预算下保留全部状态继续20轮到60。该条件只读TRAIN/
VAL，不读测试结果；恢复真实训练步通过后才进入全局GPU串行队列。
控制器记录waiting/selected/queued/failed状态，遇到资格失败不
自动启动新训练。整个后续窗口仍为完整验证，不宣称75%TEST目标。

2026-10-04 21:14 CST：首层行和约束8轮已完成，最佳582/900=64.67%，
低于未约束hybrid601/900和前驱614/900，不采用、不延长、不访问
TEST。预先登记的条件选择正则化从零CIFAR模型40->60，真实恢复
步通过；原cosine200、模型/Adam/阈值/RNG/loader/监测状态保持。
让此前排队的Gesture8轮先完成后，CIFAR正式worker229965于21:06
开始；第41轮完整VAL580/900=64.44%提高原574/900=63.78%，仍
不代表75%完整TEST目标达成。更换的只是等待中的supervisor，没有
停止任何训练worker，调度记录保留before_gesture_priority证据。

Gesture纯STDP与首层hybrid匹配8轮最佳172/204=84.31%与
173/204=84.80%，均未超过前驱175/204=85.78%，不继续这些阶段。
只用完整VAL的自己的双检查点诊断准确复现原root88的175正确和
compact61的166正确，参数/BN/阈值均未变；两者同时正确152、
同时错误15，root独有正确23、compact独有正确14。概率混合在
T2、各.5权重时179/204=87.75%，仅作为辅助训练依据，不作为单个
网络性能或95%TEST达标。该诊断没有TRAIN更新或TEST评测。
下一项待实现：使用这些同TRAIN/VAL划分、从零来源的自己模型做
训练时的软目标，保留一个原尺寸SNN学生与原BP/STDP/WTA/指数
稳态、T16/128条件。计划.5硬标签CE+.5*T²KL、T2/教师各.5，必须
拒绝TRAIN+VAL合并教师、冻结教师所有状态、保留学生RNG、单独
测试真实BP/STDP/续训与原尺寸预算，先8轮完整VAL。该计划尚未
实现、登记或启动；现有source_v2..v18运行源码保持冻结。

2026-10-04：CIFAR正则化从零模型40->60窗口完成，最佳587/900=65.22%
（第47轮）；第60轮541/900=60.11%，TRAIN CE0.94483。仍低于
保留特征前驱614/900=68.22%，不自动延长到80轮、不评测新TEST。
下一项Gesture训练时自蒸馏正在新source_v19实现，教师来自相同
TRAIN/VALID划分的root88和compact61，从零来源、无合并VALID；
目标是一个原尺寸学生。正式运行需先完成真实更新、冻结状态和续训检查。

2026-10-04：source_v19的自建教师训练辅助已经实现。183项独立检查
通过（182项初次通过，一项测试夹具配置共享修正后通过）；CPU和
原生CUDA实际联合损失的学生/优化器/调度器/RNG/loader续训一致。
教师SHA、来源、完整输入、数据根、TRAIN/VALID/TEST划分严格匹配，
拒绝外部来源和TRAIN+VALID重训教师。教师构造保留学生RNG，冻结
所有参数、BN、阈值；不进入学生state_dict/尺寸/量化/推理。
真实TRAIN8样本、T16/128预检CE0.12231、scaledKL1.78988、总损失
0.95610；CNN/classifier/第二隐藏BP有梯度，首层250个STDP赢家行
更新，第二层157个BP行更新、败者受保护；显存峰7180MiB。
正式8轮完整VAL204已登记并启动，原root88学生/尺寸/g64index/
STDP-BP/指数稳态保留；新阶段Adam1e-4、plateau80保留。先看
单学生是否最佳至少178/204且末3轮均值至少175/204才考虑延长，
否则保留负结果。训练记录分别打印hardCE/scaledKL/combinedloss，
VALID始终只算CE；此窗口不评测TEST。运行源码33文件SHA冻结。

2026-10-04：source_v20实现单独显式CIFAR64->128自建检查点输入阶段，
普通续训严守分辨率和新阶段标志，不能同时改T/事件变换/池化或
使用TRAIN+VALID重训模型。192项检查通过，包括CPU/原生CUDA
实际原生输入阶段的全参数/Adam/调度器/BN/RNG/loader续训一致。
候选使用最高VAL614/900的原正则化自建检查点，参数、阈值、行和
目标全部保留；T8/log1p/pool4、STDP-BP/g16margin/指数稳态不变。
原生128保留更多空间事件；通道和线性宽度及参数量不变，卷积
空间展开神经元约4倍，明确计入额外激活量。TRAIN16批BN重估
后真实TRAIN16样本预检CE2.2917、显存6652MiB，CNN/classifier/
第二隐藏BP有梯度、首层1425个STDP赢家行更新，第二层518个BP
行更新，败者保护和非负通过。较64输入CE0.7595明显增大，只能
说明输入适配尚需学习，不宣称预检提高精度。登记8轮完整VAL900，
全局GPU队列在Gesture窗口之后运行。与历史64输入对照同时改变
分辨率和BN策略，不归因于孤立单因素。所有33个运行文件冻结。
两项新窗口实时曲线fp32_recovery_curves.png分开TRAIN硬CE/教师
KL和VALID CE；VALID只有单学生，不含教师混合推理；未访问新TEST。

2026-10-04 21:45 CST：Gesture自蒸馏8轮完成，最佳170/204=83.33%
（第2轮），与原source16普通CE自身微调8轮170正确持平，低于
自身root88的175正确。预先登记的延长门槛未满足，不延长、不
评测新TEST；双教师混合179正确没有转化为单网络收益。负结果
及hardCE/scaledKL/联合损失轨迹保留gesture_own_distillation_result。
学生仍原尺寸，未改变非负/BP-STDP/WTA/指数稳态约束。
CIFAR原生128、T8候选正式worker263100于21:44:48启动，继续
完整VAL900短窗口。当前FP32完整TEST达标只有MNIST99.23%和
N-MNIST98.75%；CIFAR75%与Gesture95%仍未完成，量化调优置后。

2026-10-04 21:52 CST：Gesture完整VAL204缓存误差分析确认root175与
compact166，按实际lexical文件夹class_to_idx映射动作名（class2
是其他随机动作，class8是两类合并的前臂滚动），未更改标签。
root在参与者17上47/60，其他40/48、46/48、42/48；拍手与左手
顺时针各11/17、随机动作12/17。仅分析VAL，不分析TEST错误。

source_v21新增分类头非负投影模式common_offset，Adam之后对每
个输入特征列减去该列所有类别的最小权重。各类共享相同输入相关
logit偏移，保留相对logit/softmax/CE和输入梯度（FP32容差），
所有物理权重非负，不新增参数/银行/神经元；head仍不参与WTA和
量化。旧模式clamp默认行为保持，变更只允许显式own阶段，普通
续训严格匹配。201项检查通过，包括CPU/CUDA实际BP-STDP/WTA
续训状态一致。原root头9.64%权重为0，CIFAR头1.50%为0，只说明
投影边界活跃，不能认定是泛化失败原因。真实TRAIN步中root4330
个、CIFAR358个Adam更新后权重为负；新投影使min=0并保持所有
预测，相对logit最大误差两者都2.2888e-5，CE误差均小于6e-7。
初始完整真实批loss/梯度与原匹配控制一致，rootCE0.12231186、
CIFAR CE0.75954914；BP/STDP赢家行写入、败者保护和非负通过。
分别从最高VAL的自建root88/正则化614模型保留参数/阈值/行目标/
RNG开启8轮完整VAL窗口，Gesture原T16/128/g64index、CIFAR原
T8/64/g16margin等都保持。原生CIFAR窗口结束后先运行Gesture，
再运行CIFAR分类头窗口；顺序wrapper和全局GPU锁防止抢占或
重复启动。各33运行文件SHA冻结；没有宣称单因素精度收益。
继续门槛仍Gesture最佳>=178/204且末3均值>=175/204，CIFAR
最佳>=632/900且末3均值>=68%；未满足不延长、不评测新TEST。
classification_projection_curves单独保存硬CE和完整VALID曲线。

2026-10-04 22:00 CST：追加自建结构候选singlehidden8192。五层CNN
64/128/256/512/1024保留，两层4096隐藏层改成一层8192，BP7680
和STDP512按period16分工、g16margin/presentation WTA及指数稳态
保留。神经元计数口径完全匹配：spiking通道+线性宽度10176、
物理E/I银行+线性宽度12160。空间读出pool4->2减少第一线性层
fan-in；分类头保持默认非负clamp，此结构不叠加common_offset。
分别在GestureT16/128和CIFART8/64从零训练，原数据划分/编码保持，
TRAIN初始BN8批/隐藏校准及每次VALID前BN16批、Adam.001和原
cosine160/200预算。先各20轮完整VAL，未达到预登记学习门槛不
直接转成长训练。多项结构/学习分工/池化共同变化，不作孤立归因。
真实训练步G CE2.4019/显存6275MiB，2311BP+146STDP赢家行均
写入；C CE2.3106/显存1957MiB，4478BP+300STDP eligible共
4747行写入（因此至少4447BP和269STDP行写入），败者和非负
通过。fresh单隐藏层CPU/原生CUDA额外2项实际训练/Adam/调度/
BN/RNG/loader续训一致通过，source_v21共203独立检查。
C预检进程退出时有multiprocessing临时目录清理warning，计算
及断言完成、进程exit0和preflight passed；不据此宣称性能。
20轮窗口已进入确定顺序的GPU队列，等待两项分类头短窗口结束。
报告的量化范围审计现在从真实hidden_widths派生层名；旧四项
完整TEST审计仍通过、单层fixture允许hidden1而拒绝classifier和
不存在的hidden2，共7项报告检查。该fixture不代表单层网络精度。

2026-10-04 22:01 CST：CIFAR原生128保留特征8轮窗口完成，最佳
485/900=53.89%（第7轮），低于自身原64输入
614/900和原64匹配8轮608/900；末3轮均值51.89%。预登记延长
条件未满足，不延长、不评测新TEST。完整失败轨迹与比较限制写入
cifar_native128_retained_features_result；不据单种子及BN共同变化
认定原生输入普遍较差。Gesture分类头common_offset窗口正式
worker279674于22:01:04启动，第1轮完整VAL170/204=83.33%，
仍低于原175/204，尚无延长或95%TEST达标证据。后续队列保持
CIFAR分类头8轮、Gesture单隐藏20轮、CIFAR单隐藏20轮。


2026-10-04T14:31:06.998378+00:00: Common-offset8-round windows finished: Gesture170/204, CIFAR607/900; neither promoted. Event gain-only validation variants failed the predeclared improvement-over-identity175/204 rule, so normalized warm training rejected before launch. Singlehidden Gesture20 best159/204 at round20 misses original80% gate; recorded as failure. A separate adaptive VALID-only20->40 continuation is registered from latest all-state with original160 cosine, based on falling TRAIN loss and improving fullVALID trend. CIFAR singlehidden20 remains running. No new official TEST evaluation.

2026-10-04T14:34:52.563554+00:00: Registered conditional CIFAR singlehidden20->40 launcher before20-round results: unchanged original best>=540/900, last5TRAINCE<first5 and last5VALIDmean>=previous5. Source21/allstate/original200cosine retained; execute after Gesture exploratory40 only if gate passes. No TEST. Driver/helper/config/protocol hashes pinned and partial-count/short-window guards checked.

2026-10-04T14:46:08.817059+00:00: CIFAR singlehidden20 finished650/900=72.2222% best round19; all predeclared20-round gates pass, conditional40 driver314319 waits for Gesture. Gesture continuous20->40 worker324131 started at epoch21 and globalstep2440; original160 cosine retained. No new official TEST.

2026-10-04T14:59:14.571525+00:00: Gesture singlehidden40 best176/204 at37, late5mean164.6 vs previous5mean166.6, so no further continuation. Separately fixed global highestVALID checkpoint37 fullTEST evaluation; datasetassignment updated beforeTEST and all parameter/input/split/source hashes pinned. CIFAR qualified40 worker337126 resumed after Gesture. Source23 optional per-stepCE validated254checks, realTRAIN preflight passed and matchedmeanCE .12231186; no formal temporalCE phase launched.

2026-10-04T15:02:12.298663+00:00: FixedGesture37 completeTEST FP32/uint2/uint4=260/256/259 of288 (90.2778/88.8889/89.9306%). Statehash before/after inference exact, no newtraining.95%goal notmet. Registered-before-TEST temporalCE ownroot phase queuedafterCIFAR40 with8VALIDround gate>=178/204 and last3mean>=175/204; onlyTRAIN/VALID and officialloss-code basis, no TESTerror-driven tuning. Refreshedscientificliveplot joins original and currentCIFARwindow; prior plot/launch artifacts archived.

2026-10-04T15:21:06.364627+00:00: CIFAR40 best674/900 at34, all existing614/trend gates pass; conditional60 startedworker361870 from originalepoch41 with complete state/original200cosine. GesturetemporalCE8 best172/204 at6 vs175ancestor/176currenthighestVALID, registered178+last3>=175 gatefails; noextension/noTEST. TrainingperstepCE decreased withoutaccuracygain. Conditional60 controller resultfilename corrected beforedecision (old20evidencepreserved, originalworker337126 uninterrupted). NextG candidate mayuse freshonehidden with existing causaltemporalEI features; notyetimplemented/launched.

2026-10-04T15:25:53.226186+00:00: Registered freshGesture temporal2+singlehidden8192 fromScratch20 afterCIFAR60: causalE/IConv3d firsttwo stages, same12160physical/10176spiking widthbudget, T16/res128/pool2/B8/meanCE/exp/group16margin/hybridperiod16 unchanged.2 nativeCPU/CUDA freshresumechecks added outside33frozenruntime manifest (total256unique); realTRAINpreflight all4historybankgradients positive,2338BP+157STDProws updatedwinneronly, allsynapticweightsnonnegative, peak6311MiB. No new TEST or externalweights.

2026-10-04T15:35 UTC: Conditional CIFAR60->80 policy registered before60 completion; unchanged675/900 + full20round/train-validation trends gate. If passed, wait fresh Gesture temporal2 qualification then resume complete state/original200cosine for20 rounds; stop candidate at80 or earlier nativeplateau and separately freeze highestVALID for one fullTEST. If60 gate fails, no extension and freeze highestVALID for fullTEST. Policy/driver hashes pinned; controller375805 active. No new official TEST.

2026-10-04T15:38:25.421770+00:00: CIFAR60 complete best688/900 at47, all four unchanged extensionflags pass; last5VALID664.8 vs663.4 previous5 and TRAINCE.24098 vs.27323. Conditional80 now waits freshGesture20 worker381289 started15:37:38UTC. No new TEST; fresh temporal+singlehidden loss/VALID plotted by native379901.

2026-10-04T15:40 UTC: BeforeCIFAR80 observations, registered fixedhighestVALID post80/earliernativeplateau fullTEST1000 (FP32/uint2/uint4). Native selector384251 waits originalcontinuationcontroller375805 thenactualsuccessfultrainingterminal, freezescheckpointSHA/epoch, checks33sourcehashes/splitcounts and updatesdatasetassignment beforeTEST. Originalusedevalhelper preserved; v2onlyreads correctparent60gate provenance (true). NoTEST opened now.

2026-10-04T15:42:29.590403+00:00: Before freshGesture20 completion, original164/204+fallingTRAINCE+nondecliningVALID gate translated to conditional40 controller386174.5 gateboundary checks reject partial/short/repeated/wrongaccuracy windows. Ifpassed, waitCIFAR80 successfulterminal thenresumeallstate original160cosine; iffailed noextension. NoTEST and no gate relaxation.

2026-10-04T15:52:49.581775+00:00: FreshGesture causaltemporal2 qualification20 complete133/204=65.1961%, original164accuracygatefails despiteTRAIN/VALIDtrends; noextension/noTEST. CIFAR80native396336 running. Source24optionalCNNtimeBN adaptedauthorBNTT, retainsbeta andclonesownsharedBN beforeTRAIN-only16batchmoments; full272CPU/nativeCUDAchecks59.07s. RealTRAINpreflightCE1.68455/backbonegrad139.71/hidden2BP.61087/class8.23765,271STDP+171BPeligibleupdated, loserprotection/nonnegative,6598MiB. Ownroot88CNN/timeBN8windowqueuedafterCIFAR80, target178/204+last3mean175. Samewidthbudget, noTESTerror-driven tuning.

2026-10-04T15:53:47.872535+00:00: CIFAR bounded80 route stoppedat62 byoriginal15staleepochplateau guard; only2new rounds61/62 VALID672/645, no improvementover688/900 selected47. FixedpostwindowfullTEST evaluator384251 registeredSHA/epoch/assignment beforeTEST, nowrunning (no furthertraining); GesturetimeBNnative398684 started15:53:05UTC,8VALIDrounds.

2026-10-04T15:54:37.290136+00:00: FixedCIFAR47 completeTEST1000 FP32/uint2/uint4=735/740/737 (73.5/74.0/73.7%); correctmodelstate/hash preservedbefore/after, singlehiddenonlyquant.15correctshortFP32target750.Original62epochplateaustop retained; targetnotclosed. Nextcandidate should be basedonVALID15epochplateau (cosineLR~.0008), not TESTperclass/errors.

2026-10-04T15:57 UTC: CIFAR postplateau ownphase registered withfreshAdam1e-4/plateau80 from fixedown47; alltopology/input/WTA/STDPrules retained.8VALIDwindow gate700/900+last3>=688; prefCE.37299,1958MiB,1330BPeligibleupdated; firstbatchSTDPeligible0 reportedexplicitly (512STDPrule neurons remain, no claimofSTDPupdate inthatbatch). Nativequeue402289 waitsGesturetimeBN8. Liveplot402353 showsbothphases. FullTESTmodelassignmentC73.5% remains untilanynewVALIDselectedeval fixed.

2026-10-04T16:00:09.122445+00:00: GesturetimeBN8 completebest169/204 at5, last3mean159.333 vs175 required; both original178/last3 gatesfail. MatchedsharedBNwarm170 andancestor175 remain stronger onVALID; noextension/noTEST. CIFAR lowerLR8worker404976 started15:59:07; assignmentremains fixedCIFAR735fullTEST. Reporterprevious277952 nativehandle missingconfirmed and new404141refresh includes currentCIFAR assignment withnoauditerrors.

2026-10-04T16:05:46.736578+00:00: MatchedGestureownsingle37 pair registeredsource24: freshAdam1e-4/plateau80,meanCEcontrol versus per-stepCE, sameT16/res128/pool2/g16margin/8192hybrid/drop.5/aug/BN16/weights/theta/ownRNG/loader. RealTRAINinitialmeanCEboth.40713936 exact, temporalobjective.93152797;CNN/class/h1BPgradspositive,1412eligibleBProwsupdatedeach; firstbatchSTDp0explicit despite512styleSTDpneuronsretained. Peaks6276MiB. Native410837/411636 orderedafterCIFAR8. Gates180/204+last3>=176 per8window, noTEST. CIFARown8 best709/900 at5; original700+last3>=688 gate copiedwith4boundaryguards; conditional16controller410838 waits matchedGpairifgatepasses, sameallstateplateau80/outdironly; bounded16thenfreezehighestVALIDfullTEST.

2026-10-04T16:07:24.590269+00:00: CIFARown8 completeVALIDbest709/900 at5, last3mean698.333; all unchanged700+688 gatespassed. Controller410838 waits matchedGesturepair thenallstateownphase9..16(originalplateau80); noextra owninit retainedinchild protocol. Fixedpost16 fullTESTpolicy registeredbeforefuture16 observations, mininheritedVALID709/900; freezehighestVALID checkpointandassignment beforeonefullTEST. GesturemeanCEnative413210 started16:06:30.904UTC, own37 unchangedinputs/WTA/exp; temporalCE queued aftercontrol.

2026-10-04T16:12:47.988689+00:00: GesturemeanCEown37 complete8 best183/204 at3 vs176ancestor, late3mean175<176 so originalextensiongatefails. Controlbudgetnotextended; awaitper-stepCE matched8 thenapplypreregistered selection. Selectiondriver417559 stoppedwhilewaiting/beforechild, oldpolicy/launch/status archived; replacementv2/418988 bindsresume preflight to selectedbranchdeclaredhelper (meanoriginal vs temporal0f), criteria/order unchanged. No newTEST.

2026-10-04T16:16:28.179004+00:00: Gesturematched8 terminal: meanCE183/204 at3 butlast3mean175<176 so noextension; per-stepCE182/204 at8, last3mean180.333, alloriginal180+176 gatespass. Preregisteredeligiblebranchselection choosesper-stepCE, waitsCIFAR16 thenownphase9..16/allstateplateau80. GlobalhighestVALID183fromfailedbudgetgatecontrol remains in finalcheckpoint selection pool; gatescontrolbudgetonly. NoTEST now.

2026-10-04T16:20:11.014151+00:00: Gesturepostmatched16 fullTESTselector426874 registered before16 observations; frozen selectionpoolboth8branches+selected16, maxfullVALID accuracy (tieprefermeanCE thenearliest), minimum183/204 inheritedfromcontrol. Runtime/config/split/checkpointSHA andassignment fixed beforeTEST288; no newTRAIN/BNrefit/threshold changes. Budgetgatefailure doesnotexcludecontrol183checkpoint.2 actualranking guards pass. Reporter404141 missingnativehandle confirmedafterqueuegap; oldhelper archived, persistent425581now--watch--persist keeps updatingthroughpreflight/registration gaps; no trainingcode mutation.

2026-10-04T16:25:21.831193+00:00: CIFARownphase16 completesbest709/900 fromphase5, selected beforeTEST; fullTEST FP32/uint2/uint4=755/759/755 of1000 (75.5/75.9/75.5%). Statehash beforeafter exact/quantonlyhidden1,zeroevaltraining; FP3275targetclosed. ConditionalofficialTRAINfixed5 policy skippedbecausecondition<750 false; noextraCIFARtraining. Gestureselectedtemporalownphase16 native430818 runningafterCIFAR, originalplateau80/allstate. OverallM/N/C FP32targetsmet, Gesture95stillopen.

2026-10-05 00:33 CST: CIFAR singlehidden lowerLR ownphase complete TEST FP32 755/1000, uint2 759/1000, uint4 755/1000; independent replay exact. Gesture global complete VALID winner meanCE phase3 (183/204) fixed TEST 270/288, 272/288, 270/288; not FP32 target274. Triggered official TRAIN union fixed5 policy registered 2026-10-04 16:23 UTC before prior TEST: retain selected Adam/LR/RNG, train1176, no merged VALID selection, fixed end epoch7 from parent2, full TEST288 at endpoint. Policy/result/config/source33 hashes and real TRAIN preflight recorded in gesture_singlehidden37_official_train_fixed5_final. No TEST class/error tuning.

2026-10-05 00:38 CST: Gesture fixed5 official TRAIN union endpoint FP32 268/288, uint2 268, uint4 269; independent replay all scores exact, state unchanged; target remains NOT MET. CuDNN allow_tf32=True and matmul allow_tf32=False recorded, parameters FP32. No claim of full IEEE FP32 arithmetic. Mature heldVALID teachers meanCE3/temporalCE9 each183/204; six preregistered mixture conditions maximum184 at T2/control.25. New matched8 ownphase student/control registered from same183 model/RNG; KD .5hardCE+.5KL*T2, teachers .25/.75; no external teacher or ensemble final inference. Extend only best>=184 and last3mean>=183, no TEST during qualification.

2026-10-05 00:55 CST: Mature183 ownKD/control eightrounds both bestVALID180/204, failed184/trend183 gates, noextension/noTEST. TRAIN-only BNcoverage16/32/64/122batches VALID183/177/176/178, noextension; spatialfiveview means at +/-4,+/-8 pixels VALID182/183 vs single183, noTEST. Newsource25 nested2x2-to4x4 pool readout expansion, same8192hidden/channels/12160physical widths, firsthidden coefficients repeat/4 preserving currents/rowmass. Full CPU/nativeCUDA284passed31.69s; initial completeVALID204 allpredictions identical183/204, maxlogitdiff0. Parameters46,192,784->146,856,080, TRAINpeak6275.91->7433.91MiB. Newmatchedownphase8 pool2/pool4 fromsame183parent, freshAdam1e-5/plateau80, T16/128/countcoding/WTA/nonnegative/BP-STDP/exp unchanged; gated16 continuation onlybest>=184 andlast3mean>=183. TESTindependentandnotusedforselection.

2026-10-05 01:12 CST: Spatial pool2/pool4 eight-round lowLR1e-5 matched phases both best182/204, failed184/trend183, noextension/noTEST. Original mature183warm causalConv3D initialVALID183but predictionsdiffered fromspatialpredecessor, TRAINCE.2173vs.1924; nottrained, retainedinitialization_rejected. Source26 adds explicit laggedConv2D implementation of same3causaltap physicalEI banks; defaultConv3D retained, stricteval/resumeidentity.293fullCPU/nativeCUDApassed32.02s; currenttap nativeexact parity and realhistorygradients/no future access. Mature183 owncausalnewphase initialcompleteVALID204 allpredictionsidentical183,maxlogitdifference0. ActualTRAINhardCE.1924437881 andWTAevents exactcontrol, historytapgradient norms4.5586/1.3216, peak6854.76MiB. Params46,192,784->46,492,304, widthbudget12160same. Eight-roundownphase freshAdam1e-4/plateau80, T16/128/pool2/WTA/BP-STDP/exp unchanged, validateddriver extends16 onlybestVALID>=184 andlast3mean>=183. Allnewruns useheldVALID, priorTRAINunion model excluded.

2026-10-05 01:21 CST: Maturewarmcausal8 VALID[178,176,182,175,177,178,175,180], best182<183, both184/trend183 gatesfailed, noextension/noTEST. Freshown uniform5x256CNN +9600hybrid (9000BP600STDP) maxpool4readout, T8/res128/log1p nowrunning. Physicalwidth12160same, logical10880 versus10176 +6.918%. Native16countframesmergedadjacentpairsbeforelog, conservedcounts. Seed42/splits972/204/288/exp/WTA/nonnegative/classheadFP unchanged. FreshAdam5e-4/plateau80 patience5min5e-6; initial20TRAINVALIDrounds, gate40 onlybest>=153 andlast5VALIDmean>=previous5 andlast5CE<first5; stage40before80 gatebest176,last5mean164. Source27 fullCPU/nativeCUDA300pass30.37s/runtime33SHA. RealTRAINpreflight loss2.40342, BPeligible3087/STDPeligible194, changed3275 so at least188STDProwschanged; CNN/head/hiddengradientspositive; peak8517.53MiB. OfficialPLIF equalwidth/maxpool onlyarchitectureinspiration, no externalweights/reproductionclaim. Nativeworker507077/supervisor507049/controller507311, realtimereporter425581. FP32targetsremain3/4; newfreshtrainprogressnotqualityproof.

2026-10-05 01:40 CST: Fresh uniform256/maxpool/T8 scratch20 ended best VALID140/204 (epoch19); original153 best gate failed, so controllers skipped40/80 and no improvedVALID TEST. All20 TRAIN CE1.934->.712 remained logged; best width-matched increasing-channel predecessor still183. Refreshed complete20 descriptive learning plot; actual expanded spatial activity accounting retained. CPU checkpoint audits confirmed M9923/10000,N9875/10000,C755/1000 FP32 fullTEST with exact independent replay, finite nonnegative float32 physical weight tensors and hidden-only 2/4-bit quantization. Old M/N score files lack saved confusion matrices; no false confusion claim. Goal remains3/4.

2026-10-05 01:41 CST: Source28 label-smoothing epsilon0.1 TRAIN loss-only own phase preregistered from heldsubjectVALID183 model (sha6639b935...), freshAdam1e-4/plateau80 with same parent RNG/loader, T16/res128/log1p/pool2/8192hybrid/g16/period16/margin/presentation/exp/drop.5/aug/BN state. 317 full CPU/nativeCUDA tests passed31.62s; analytical hard/uniform CE gradient, hard VALID, explicit ownphase/strictresume and actual WTA/STDP training-state equality checks included. Complete initialVALID204 predictions/logits exactly unchanged183 (maxlogitdiff0); same46,192,784params. Completed prior matched plain8 control reused (best180); no new TEST error/subject/class tuning. Actual objective preflight cloned newhelper instead of altering SHA-pinned hardCE helper. First8 rounds extend to16 onlybest>=184 andlast3mean>=183, predeclared fixed highestVALID fullTEST onlyif>=184. Failure records retained; no new TEST yet.

2026-10-05 01:42 CST: Label smoothing actualTRAIN preflight smoothedCE1.1945753/hardCE.1924437881 (hardCE exactly matched parent control), allpositiveCNN/head/hiddenBPgrads,1456eligibleBProws changed andno loserchanges, STDPeligible0 thisbatch explicitly; peak6275.91MiB, backendCuDNNTF32true/matmulfalse. First actualtraining batch exactlypreflight CE, initialfullVALID183/hardCE.438594. Worker527051/supervisor527028 nativeconfirmed. Conditionaldriver startup-beforequeue race stopped onlywaiter527030, archivedoldlaunch/status/log, restarteddriver527769 andselector527770 afterverifiedlive worker/supervisor; no trainingrestart/helperSHA/policychanges.6 boundarygatechecks passed; improvedVALID selection/continuation thresholdsunchanged.

2026-10-05 01:47 CST: Source28 strong label smoothing0.1 complete8 VALID[157,164,171,161,170,172,168,173], best173/204 (84.8039%) below matchedplain180 andparent183; both184+last3mean183 gates failed. SmoothedTRAINCE1.082->.717 decreased withoutaccuracygain. Nativeconditionalcontrollers terminalnoextend/noTEST, GPU idleconfirmed. Loss curves distinguish smoothed objective from hard CE. Preregistered weaker0.01 sameparent/freshAdam/RNG/source28/conditions, same8->16gate andonefixedTEST onlyVALID>=184; no TESTerrors/classes/subjects used. No extra implementation/regression reruns because runtime33 unchanged from317passed snapshot; actualinitialVALID andTRAINpreflight rerun for newconfig.

2026-10-05 01:48 CST: Weaker0.01 initialVALID204 exact183 predictions andlogits(maxdiff0), same46,192,784parameters. ActualTRAIN hardCE.1924437881 exactlyprior, smoothedCE.2926569283, CNN/head/hiddenBPgradientspositive,1456BPeligiblewrites/no loserupdates, firstbatchSTDpeligible0explicit, float32weights/nonnegative6275.91MiB; CuDNNTF32true/matmulfalse. Supervisor533462/worker533473/controller533478/selector533479 launched onlyafter actualqueue/nativehandles existed (startup race prevention inlauncher). Same317passed source28/runtime33 and parentSHA retained; source/default implementation unchanged. Budget8 first, best184/204+last3mean183 toextend16; onefixedTEST onlyhighestVALID>=184. Closed3 FP32goals remain M99.23,N98.75,C75.50; G95notclosed. Trainingcurrentneverclaimsbenchmarkquality.

2026-10-05 01:52 CST: Weak label smoothing0.01 ownphase completes8 VALID[181,180,180,175,177,172,171,179], best181/204 atfirstepoch <parent183; original184/trend183 continuation gates fail. Strong0.1 best173 andweak0.01 best181 both failed against matchedplaincontrol180/parent183. No TEST opened for either. Validation hardCE occasionally improved (.4399 at3) withoutcorrect-count gain; further epsilon sweep not selected. Next hypothesis is stronger TRAIN-only geometry augmentation based on recorded train/heldsubject generalization gap; 31 priorqualifiedGesture configs shared identical0.1translate/10degree/0.85..1.15scale profile. Runtime source28/317checks can be reused unchanged; no implementation change needed.

2026-10-05 02:00 CST: Broader TRAIN augmentation(translation.15/rotation15/scale.75..1.25) complete8 VALID[180,178,179,177,173,181,178,183], best183 at8 equalsparent; original184+last3mean183 gatesfail. Bothconditionalcontrollers noextend/noTEST, no improvement claimed. FirstrealTRAINprefCE.365516 versusplain.192444/1451eligibleBPwrites andno loserwrites; fullinitialVALID204 logits/predictions exact183. Source28/317checks unchanged. Registered ownalignedparameteraverage VALIDdiagnostic afterthisterminal: maturemeanCE3 andtemporalCE9 ownbranchesboth183, common37parentSHAd8dc29..., exactshape/style/input/split/seed/physicalFP32checked. ConvexA coefficients.25/.5/.75, each averagedBN versusTRAIN-only16batchBNrefit; sixconditions, oneinfermodel/samewidth, no optimizer steps/no officialTRAINunion. PureCPU25tests passed1.33s for negativeweights/dtype/masks/commonorigin/split/state/noalias/linearcurrentparity. This is custom alignedbranchaveraging, not a reproducedSWA schedule; PyTorch primaryguidance consulted forBNstatistics. SelectfullVALID highestearliestties; futureTEST onlyif>=184, no TESTperclass/errorselection.

2026-10-05 02:02 CST: Six alignedweightaverage fullVALIDconditions complete: A.25 noBN183/TRAINBN181, A.5 noBN183/TRAINBN180, A.75 noBN184/TRAINBN176 of204. SelectedA.75/B.25+averagedfloatingBNbuffers (no newTRAINBNrefit), oneinferSNN46,192,784params/same12160physicalwidth. BaselinesA/B eachreplayedexact183, stateunchanged; allcandidateVALID204exactcounts, signedBNaffinesallowed/physicalweightsnonnegativeFP32. TRAINBNrefitonlybackbone, actualparameters andallhiddenweights/thresholdbuffers unchangedverified; averagehelpers25checks/source28317 bound. GainonlyoneVALIDcorrect(.4902percentagepoint), marginal andnotTESTproof. Preregistered184gatepasses; selectedaveragedcheckpoint epoch=-1/evaluationonly/nooptimizer, rawfloatstatehash frozen, metadata records allownsourceSHAs/coefficients. Assignmentandconfig/selectedSHA fixed beforefuturefullTEST288; noTEST tuning.

2026-10-05 02:03 CST: VALID184-selected A.75/B.25 singlemodel fullTEST completes FP32 271/288=94.0972%, uint2/uint4 272/288=94.4444%; newinference statehashunchanged, no newTRAIN/BN/thresholdupdates. FP32 95goal274stillthreecorrectshort. OnefixedTESTselected only byVALID, no TESTerror/class/subjecttuning; no inferenceensemble or extradimension. CheckpointSHA cf6aa72188153936653f763f24d3f72d50a349de095408ad728acbe896ae2fec frozen beforeTEST and assignmentupdated inadvance. Independent samecheckpoint/sameinputstream precisionreplay registered as auditcase, no modelselection/assignmentchange.

2026-10-05 02:05 CST: Independent replay of exact selectedaverageCP verified all fullscorefields/confusions/losses identical FP32/uint2/uint4=271/272/272 of288; bothbefore/aftermodelstateexact andzero newTRAIN. test_replay_verification.json saved; CPsha cf6aa721... andallhelpers/source33 guards retained. LatestfixedmodelassignmentG94.0972%,95goalremainsopen(threecorrectshort), no materiallynegative2/4-bitgap. M99.23/N98.75/C75.5 remainclosed. Parameteraveraging is explicitly recordedaspostprocessing ofsame-parent WTA/BP-STDP-trained ownbranches; oneinferenceSNN, no ensemble. Any futureaveraged-own-init training must explicitly recordfresh optimizer/RNG policy because syntheticCP hasnooptimizer/RNG andepoch=-1/evaluationonly; do not silently pretendtoresume.

2026-10-05 02:09 CST: Reporter425581 exited on VALID-diagnostic launch.json lacking trainingconfig; renamed diagnostic_launch.json. Firstrestart552808 alsoexited because averaging parent hasresults.json ratherthantrainingvalidation.json. Unfrozenreporthelper oldversion archived; nowinherits declared averaging diagnostic byfrozenresultsSHA andexactselectedcondition, verifiesVALID204/newTRAINsteps0/oneinfermodel, includingindependentreplayoriginalbasis. No fabricated trainingepochs/validation.json. One-shot fullrefreshpassed allcaseaudits[], currentassignmentG94.0972/94.4444/94.4444 with0newtrainingepochs andVALID184explicitparametersaveraginglabel. Newpersistent reporter554386 sourceSHA84ca8864... launched; training snapshots/configs/results/selectedCHECKPOINTs unchanged. AllfullTEST andindependentreplay complete; goalremains3/4active.

2026-10-05 02:20 CST: Source29 explicitownaverageadaptation full332passed35.51s (15 new guards/freshownphase/realCPU-CUDA BP-STDP/resume); frozenruntime33. Registered originalfreshAdam1e-5 average184ownphase8, fullinitialVALID184/predictions/logits exact andactualTRAINprefpassed; NOTlaunched. Recordedinitialization_rejected becauseexistingaverage diagnosticTRAIN16-BNreset184->176, so avoiddestructiveBNrefit beforefitting. Source30 freeze_backbone_bn configurableonlyexplicitownphase/resume, BNrefit0 enforced, BNrunningstatsfrozen duringTRAIN viaAccuracySNN.train override; affineBP trainable, remainingnetwork/dropout/spiking/WTA/homeostasis TRAIN normal. FinalfullCPU/nativeCUDA341passed33.84s (9more: momentsimmutable acrossrealgradientforwards, signedaffinesgradients, strictconfigurationidentity, no freshscratch frozenBN, actualownavgWTA/BP-STDP optimizer/RNG/resume). Onefixture initiallysharedsource/currentconfig invalidatedguardtest; fixeddeepcopy, runtimeunchanged; failureevidence retained bytestsession. NewfrozenBNcandidatefromsameVALID184average registered beforeobservations, freshAdam1e-5/plateau80/min5e-6, freshseed42/loader42, T16/128/pool2/8192hybrid/exp/plainCE/ordinaryaug/drop.5 unchanged; budget8, extend16 onlybest>=185 andlast3mean>=184, fullTEST onlyhighestVALID>=185. Originalsource29case andallprevioushashes retained.

2026-10-05 02:21 CST: FrozenBNsource30 realinitialcompleteVALID204 predictions/logits exactlyselected184 (maxdiff0), same46,192,784parameters/12160physicalwidth. ActualTRAINprefhardCE.00426548, CNNgrad3.124/head.10368/hiddenBP.0042497, allfiveBN weight+biasgradientspositive,1534BPeligible rowschanged/no loserwrites, STDPeligible0firstbatch explicitly; allBNmean/variance/intcounters unchanged verified through actualoptimizerstep. Peak6275.91MiB; float32params/CuDNNallowTF32true/matmulfalse, noAMP/quantTRAIN. BNcalibration/hiddencalibration fields inpref are inheritedhistoricalmetadata, notnewcalibration; runtimeBNstats arefrozenaveragedmoments. Freshseed42/loader42/freshAdam1e-5 andoriginalplateau80; registeredbudget8->16 onlybest185 andlast3mean184, anynewTESTonlyhighestVALID>=185. OriginalBNrefit16source29case staysrejected/unlaunched; all prior source/protocol/CP hashes preserved.

2026-10-05 02:25 CST: FrozenBNlowLRownaverage8 complete VALID[182,176,176,176,175,173,172,174], best182 below initial184; TRAIN CE.09556->.06154. Both185/last3mean184 gates fail, no16continuation/noTEST. Six gate boundary checks passed; allcase/protocol/helper hashes unchanged. Actualall15 backbone BN mean/variance/integercounters exact throughout8rounds verified againstownparent, so freezingworked butdidnot improve generalization. Recordedfailure retained; currentfixedfullTEST G271/288=94.0972 remains assignment. Next bounded VALID-only frozenmodel temporalreadout windows use sameT16forward once, start0/2/4/8; no newTRAIN/BN/theta updates, no inferenceensemble orTESTbased selection.

2026-10-05 02:32 CST: Frozenownaverage184 singlemodel time-readoutVALIDdiagnostic starts0/2/4/8 scores184/185/183/180 of204, same16simulation steps; start2selected byhighestVALID beforeTEST. Eachbatch defaultCUDAreadout exactlyoldmodel; allweights/BN/homeostasisstatesunchanged. Twohelpermetadata/device-reduction failures archived; fixedsamegridwithnoTEST/HPchange. Source31 adds stricttemporal_readout_start default0; fullCPU/nativeCUDA358passed34.61s afterfixone-step describe_size hookprobe toforward_sequence. RealnewmodelcompleteVALID185/confusionsexactdiag, lossdifference<1e-6 andallstatesexactbefore/after. Savedmetadata-only suffix2CPshab93828a9... evaluationonlyepoch-1/nooptimizer/RNG. Same46,192,784params/12160physicalwidth/WTA/BP-STDP/exp/nonnegative/hiddenonlyquant; simulationT16, meanlast14classification steps. FP32/2/4fullTEST fixedcondition preregistered andassignmentadvancedbeforeTEST.

2026-10-05 02:33 CST: Fixedsuffix2 VALID185-selected singleSNN fullTEST288 FP32 272=94.4444%,uint2 271=94.0972%,uint4 273=94.7917%. Allmodelstate unchanged; zeroTRAIN/BN/thetaupdates. FP3295target274 stilltwo correctshort; no wholegoalcompletionclaim. T16simulation/last14readout andCPshab93828a9... assignmentfixedbeforeTEST. Independentexact fixedcheckpoint replay audit preregistered, noassignmentchange, noTESTsubject/class/error-based tuning.

2026-10-05 02:37 CST: ExactsameCP suffix2 independentfullTEST replay verified allscores/confusions/losses FP/2/4=272/271/273 of288; goalstill3/4. Newsource32 freeze_backbone_parameters requiresownphase+freezeBN, excludesallCNN/BNaffine/PLIFparams fromAdam whilehiddenWTA/BP-STDP andheadBP remaintrainable. 367fullCPU/nativeCUDAchecks passed34.49s, includingactualBP/STDP/optimizer/RNG resumeandbackboneallstateunchanged. Two directgradienttest randomfixtures hadnohiddenwinner; addedexplicitactiveBPfixture, runtimeunchanged/failure retained. Registeredfreshownphase fromfixedVALID185suffix2 CPb93828a9..., freshseed42/loader42/Adam1e-4/plateau80/min5e-6, sameT16/res128/log1p/pool2/8192hybrid/g16/exp/aug/drop.5/suffix2. InitialcompleteVALID204 allpredictions/logits exact185(maxdiff0), all46,192,784physicalparametersstateunchanged. ActualdisposableTRAINpref verifiespositivehidden/headgrads,zeroCNNgrad,noCNN/BN/PLIFparameter or bufferchanges, WTAonlyeligibleupdates/nonnegativefloat32. First8 budget, extend16 onlybest>=186+last3mean>=185; newfullTEST onlyhighestVALID>=186. Newpref/runner clonedratherthaneditingoldSHA-pinnedhelpers; sixboundarygatechecks passed.

2026-10-05 02:38 CST: FrozenCNNownphase launched native supervisor581459/worker581469, gatecontroller581538/selector581539, persistentreporter575738. Firstactualepoch complete VALID185/204, TRAIN CE.11886; no gain claim. NativeGPU2780MiB, disposableTRAINprefpeak1944.97MiB; headgradient.06890/hiddenBP.002818, 1534eligibleBProwschanged/no loserwrites, firstbatchSTDpeligible0 explicitly, backbonegrad0intentional/allstatesexact. Currentofficial fullTEST FP/2/4 G94.4444/94.0972/94.7917 fixedsuffix2(singleT16SNN), M99.23/N98.75/C75.5FP32 remainclosed; goalactive3/4.

2026-10-05 02:40 CST: FrozenCNNhidden/head8 complete VALID[185,181,184,179,181,181,182,183], best185 equalsinitialparent, TRAIN CE.11886->.07310. Original186best+last3mean185 gates bothfailed; conditionalcontrollers no16/noTEST. Allbackbonephysicalweights/BNaffines/PLIFparams/BNbuffers exact againstparent afteractual8TRAINepochs; runtime33SHAintact, latestallstateCP saved. CurrentfixedfullTEST GestureFP272/288=94.4444, uint2 271=94.0972,uint4 273=94.7917 independentreplayexact. Goalremainsactive3/4, FP3295requires274 so two correctmissing. NewTRAINprogressneverclaimedasTESTquality.

2026-10-05 02:44 CST: Directparent frozenCNNchild interpolation pureCPU39checks passed1.40s; parentCPb93828a9... childCP8317c110... directinitSHA/alignment/input/split/style/allCNN/BN/PLIFstateexact verified. Parentcoefficients.75/.5/.25 eachfullVALID185/204; neitherbeats185 parent. NoTRAIN/BNrefit/TEST andnoinferensemble, everyfrozenbackbonetensorclonedexact. Newdiagnosticmetadata parent-child provenance andallfailedconditions retained. Nextsource33explicit classifier-only ownphase will freezeallhiddenparameters/theta/dropout andbackbone whileclassifierfloatBP remainsactive; originalscratch-origin WTA/BP-STDP trainingretained. NoTESTclass/error/subject tuning.

2026-10-05 02:49 CST: Source33 classifier_only_phase explicitowninit/resume, requiresfrozenCNN+BN andforbidshiddenrowprojection. HiddenrequiresgradFalse, hidden/dropout/backboneevalduringTRAIN, onlyclassifierweight/bias enterAdam; originalWTA/BP-STDP scratchtraining preservedhistorically, nohiddenwrites/thresholdadvance inthisstage. Logs potentialforwardwinnersseparatelyfromactualBP/STDPeligibilityzero; emptyWinnerGuard avoidsunnecessaryhiddenstatesnapshots.375fullCPU/nativeCUDApass35.57s, includingheadrealupdate/nonheadallstatesexact andactualownaveragecontinuousvsresumed optimizer/RNG/loaderexact. First374pass1testdevicecomparison errorfixedtestCPUcomparisononly; runtimeunchanged/failure retained. Newownclassifierstagefromsameb93828a9... VALID185suffix2 parentregistered: freshseed42/Adam1e-4/plateau80/min5e-6, T16/res128/countlog1p/pool2/8192hybrid/exp/suffix2/trainingaug unchanged. InitialfullVALID204 predictions/logits exact185 maxdiff0; actualTRAINprefpositiveclassifiergrad,zeroCNN/hiddengrad/STDPwrite andallnonclassifierparameters/buffers exact. Sixgatechecks; budget8, extend16 onlybestVALID186+last3mean185; newTESTonlyhighestVALID186.

2026-10-05 02:52 CST: Classifier-onlyclamp8 complete VALID[185,182,184,184,184,185,183,181], best185equalsinitialparent, no16/noTEST. TRAIN CE.11230->.096999; allnonclassifierweights/BN/thetabuffers exactafter8. Newown classifier-onlycommon_offset headprojection fromsameb93828a9... parent/freshseed42/Adam1e-4/aug/source33 registered. Onlyheadprojectionchanged, sharedfeature-column shift preservesrelativeclass logits butphysicalheadweightsnonnegative, alreadyimplemented/testedinexisting375passed source. Newprefclone usesactualsuffix2 objective forprojectionverification; nooldSHAhelpersedited. Budget8/186best+last3mean185 gates unchanged; completeVALIDselectiononly/noTESTperclass use.

2026-10-05 02:54 CST: CommonoffsetownheadphaseinitialVALID204 exact185/alllogitsidentical maxdiff0; same46,192,784params. ActualTRAINpref objectiveCE.002644437 equalsclampcontrol, headgrad.0661743, peak2121.08MiB; 2219headweights transientnegative afterAdam. Commonoffsetprojection relativeclasslogitmaxdiff9.5367e-6, CEafterstep.002199564 exactlyequalbefore/afterprojection, predictionsexact, physicalmin0. Allnonclassifierweights/BN/theta unchanged; nohiddenlearning/writes. Source33/runtime33/375checksunchanged, sixgateboundarychecks. Nativeworker andcontrollers launched afteractualqueue handle verified; first8/186best+last3mean185 gatesunchanged, nonewTESTyet.

2026-10-05 03:00 CST: FrozenownVALID185 virtualtwo-cycle/frame T32 diagnostic complete: T16/start2 185, repeat2/start2 170, repeat2/start4 169 of204, allweights/BN/thetaunchanged; noTRAIN/TEST. Repeateddrive doublesprocessedinput exposure, explicitlynotnew32timestamp bins/nocountconservationclaim; noextensionselected. NextpreregisteredT8countpreserving rebin usesnativecodeadjacentrawcountsumbeforelog1p, sameownmodel/frozenweights: T16/start2, T8/start1 andstart2, perbatchrawmass conservationandcodingorderchecked. NoTESTclass/error/subject tuning.

2026-10-05 03:04 CST: T8count-preserving frozenweights VALID163/start1,164/start2 versusT16/start2 185; raw eventmassandprelogrebin orderexactcheckedallbatches, noTRAIN/TEST. Newfreshown hierarchicalhidden7168hybrid(6720BP448STDP)+1024BP, sameCNN64/128/256/512/1024, pool2/T16/res128/log1p/suffix2/exp/g16/margin/presentation, physicalwidth12160/logical10176same. Dropout.3eachhidden, freshseed42/Adam5e-4/plateau80patience5/min5e-6, BN16. Source33/runtime33/375checks unchanged. Allweightsrandomfromscratch, noexternal/ownweightload/teacher. First20 gatebest153+last5VALIDmean>=previous5+last5CE<first5; 40->80 gatebest176+last5mean164/full40. FinalfixedTEST onlyifhighestVALID>=186, current185reference unchanged. Multiplearchitecture/regularizationchanges explicitlyrecorded; noreproduction/superiority/singlefactorclaim.

2026-10-05 03:06 CST: Hierarchical7168+1024 freshpref passes actualCNNgrad.101509/head6.7103/H1BP.009529/H2BP.198812, H1eligible2165(2009BP156STDP) all2165rowchanged, H2BPeligible381allchanged; no loserwrites, STDPactual228events, peak6307.93MiB. Calibrationbothhiddenmean.6 fromTRAIN8; nooutsideweights. Actualarchitecture49,259,664params(+3,066,880/+6.639percent vs46,192,784), physical12160/logical10176same; spatialunfoldedneurons recordedindependentartifact. Source33/375checks/runtime33unchanged; six20roundgateboundarychecks pass. Nativeworker+20/40/fullTESTcontrollers launched afterqueue verified; first20qual only, original80plateaubudget/RNG preservedonconditionalcontinuations. CurrentfullTESTassignment remainsoldVALID185suffix2G94.4444%; no newTESTopened.

2026-10-05 03:18 CST: Fresh twohidden7168hybrid+1024BP complete20 VALID[56,55,49,57,59,62,59,64,70,81,69,69,72,74,87,81,69,73,74,78], best87/204=42.6471percent; TRAINCE2.21913->1.59452. Original153gatefails despite loss/trend passing; no40/noTEST. Preregistered one independent seed17 replicate of archived singlehidden8192 seed42 recipe, sameT16/res128/pool2/full16readout/drop0.5/Adam1e-3/cosine160/g16/hybridperiod16/exp/aug/BN16. Source34 splits training seed from new split_seed defaulting to oldseed; cfgseed17/split_seed42 actualnative dataset972/204/288, heldsubjects14/17/18/21 and manifestSHAe94be76f... byteexact.386 fullCPU/nativeCUDA tests pass66.32s, actualsampler17/differentorder, strictsplitidentity/defaultcompatibility/resume allchecked. One newfixturehardcodedresolutionfixedtestonly andfailurearchived. Six20gateboundarychecks pass. Bounded20->40->80 retainsoriginal160cosinedenominator; finalTESTonlyVALID>=186; no TESTerror/class/subject tuning. CurrentassignmentunchangedFP32G272/288=94.4444percent, goalactive3/4.

2026-10-05 03:22 CST: Seed17 actualpref CE2.417217/allCNN0.48179/head18.19489/hiddenBP0.042992 gradientspositive;2636eligible rows allchanged (2478BP158STDP), no loserupdates,258STDPwinningevents, peak6275.28MiB/nonnegativefloat32. Native launcher623146 completed anddetached supervisor623432/worker623436/controllers623506,623507,623508. ActualfirsttwoepochsCE2.27037->1.51860, VALID83->101/204; progressonly, TESTnotopened. Newstatusreporter623981 withgeneralizedparentseed path passesallcaseaudit_errors[]; historicaldiagnosticparentswithoutconfigretaincurrentchildseed fallback, nofabricatedtrainingepochs. Allpreregisteredsource/helper/config/policySHAsunchanged. One combinedlaunchcall automaticapprovalreviewtimedout withoutaction; retryonlytraininglauncher executedsuccessfully; separatelyidentityverifiedoldreporter575738beforereplacement. CurrentfixedfullTESTscoresandassignmentunchanged3/4; hierarchy20gatefailedandallconditionalcontrollersnoextend/noTEST.

2026-10-05 03:28 CST: Registered boundedreadoutneighbors starts[2baseline,1,3] on immutablefullVALID204sequencecacheSHA1a33befa... fromalignedaveragecf6aa...; currentb938... modelstateexactagainstcachecheckpoint. Originalper-batch CUDAshapes/contiguity recovered; coarse0/2/4/8 correct/confusions/losses allreplayedexact. New1=184/204,3=183 versus2=185; no186gain, no materializedcheckpoint/TRAIN/TEST. Helper/protocol/resultSHAs retained; no source34runtimechange or extra regressionrerun. Seed17 first9 actualVALID83,101,118,130,131,135,98,123,122 withCE2.270->.816; still20qualification, nativeworkerunchanged, noTEST.

2026-10-04T19:31:11.320551+00:00: Registered suffix2alignedaverage VALIDdiagnostic withcoefficientsA[.75baseline,1rawA,0rawB,.625,.875]; sameA/B owncommonparentSHAd8dc29..., source34/386 runtimechecks andpureCPU25averagechecks reused unchanged. Baselinecurrentb938... weights/BN/theta exact .75 blend andfullVALID185/confusion requiredbeforeunseenconditions. Rawendpointscloneallstatesexact (ratherthan maxBNcounter from averagehelper). NoTRAINBNrefit/optimizer/externalweights/teacher/TEST; oneinferSNN46,192,784params/samewidth. GlobalGPUqueue pending631852 while623436stillnativealive at15epochs. SelectionfullVALIDhighest/earliestlistedties, fixed186gatebeforeanyfutureTEST.

2026-10-04T19:35:19.737375+00:00: Seed17 complete20 allpreregisteredgateflags pass: VALID[83,101,118,130,131,135,98,123,122,122,114,143,126,130,133,134,144,132,155,156], best156/204=76.4706percent at20, CE2.270366->.430826. Last5VALIDmean144.2>=previous5129.2, last5CE<first5. Conditional20->40 preflight actualpassed thenoriginal20latest/allstate resume, nativeworker635830/supervisor623506/controllers623507/623508 confirmed, first21VALID164/204=80.3922percent. No TEST. Queuedfive suffix2averageconditions complete A.75=185, rawA=184/rawB=183, A.625=184/A.875=185, allfullVALID204/stateunchanged, baselineexactcurrentb938... confusions; no186gain/noTEST, oneSNN46,192,784params/noTRAINBNrefit. Newreporthelper nowinherits bestVALID and completedpredecessor rounds for explicitcontinuousresume parent_case, withgenericparentseed config; oldhelperarchived, one-shot validation/restart pending. Core source34/386 frozen unchanged.

2026-10-04T20:09:39.084425+00:00: Seed17 bounded scratch branch terminated at66 rounds (plateau15), highest independent VALID181/204 at51, below186 gate: no TEST. Source35 395 CPU/nativeCUDA checks passed, fresh averaged-own initialization final_refit explicitly supports fresh Adam/RNG without copying synthetic optimizer. Registered VALID185-selected suffix2 average own parent b93828a9... exact initial VALID185/confusion/loss verified. Fixed5 officialTRAIN1176 complete with frozen BN moments, fresh Adam1e-5, no postmergeVALID selection; TRAIN CE [.164247,.160697,.141505,.165633,.136135], actual preflight1431 BP rows changed/no losers; STDP eligible0 throughout this phase, no new STDP-write claim. Complete TEST FP32/uint2/uint4=269/269/270 of288, latest assignment93.4028percent and goal remains3/4. Native training/supervisor terminal/GPUfree verified; failures retained. Independent exact-checkpoint replay registered, plus independent VALID-only exponentialoffset [1,.5,0] inference settling on original unmergedVALID185 parent, all physical weights/BN/rate unchanged; gate186 for any future TEST.

2026-10-04T20:14:25.641686+00:00: Freshfixed5 selectedCP independent fullTEST replay exact allFP32/uint2/uint4 scorefields/loss/confusion=269/269/270; noTRAIN, stateunchanged. New independent VALID185 parent exponentialtheta offsetscales[1,.5,0] best185, no186/noTEST; weights/BN/rate/stateunrelatedbuffers allunchanged. CachedVALID singlemodel temporalweight slopes[0,-.5,.5] VALID[185,180,183], originalperbatchCUDA reduction/confusions/losses coarse grid exact; noTRAIN/TEST/newCP. Preserveuniformsuffix2/no cooldown. Source36 adds TRAIN-only convexencoded-input MixUp and exactsamecoefficient softtarget CE, customsupervisedonly, disablednoRNG/bitwiselegacy, one WTApresentation perbatch; targetweightedTRAINaccuracy explicitlyrecorded, VALID/TESThardlabelsunchanged. Full414 CPU/nativeCUDA tests pass37.49s including actual averaged-ownBP-STDP learning+exactresumedmodel/optimizer/RNG/loader/monitor. OwnunmergedVALID185 parent freshAdam1e-4/alpha.2/p1/frozenBNmoments registered8->eligible16, best186+last3mean185 gate andfutureTEST onlyVALID186, allconfig/runtime/policy/helper SHAs frozen. Fourgate boundary checks pass; initialfullVALID/stateverification and actual TRAIN preflight beforelaunch.

2026-10-04T20:19:33.908154+00:00: StrongencodedMixUpalpha.2/p1,freshAdam1e-4,frozenBN ownVALID185 parent completed8VALID[151,150,152,149,140,130,124,123], best152/204; TRAINsoftCE.82237->.43640. Original186/last3mean185 gatesfail; no16/noTEST, latestTESTassignmentunchanged269/288. InitialVALID185 state/confusion/loss exact; prefactualmix λ.00071152, CNNgrad1.8713/head.06916/H1BP.002664/1537BProwwrites/no losers, STDPeligible0 thisbatch. Full414regressions includesrealSTDP fixtures, notclaimthisprefixSTDPwrites. Conditioncontroller firststartup hit race beforeauthoritativeworkerpublished; archived firstlaunch/failure, sameimmutablehelperretry afteridentityverified676974, traininguninterrupted; nativeoldselector676964 identitychecked/stopped/replaced678410, driver678409 bothterminal noextend/noTEST. Registered lighteralpha.1/p.25 freshAdam1e-5 fromoriginalb938... beforestrong8completed (earlyVALIDnegativebasis); same frozenBN/T16/suffix2/width/source36/414 checks,8->16 gatesunchanged186/mean185. NoTESTperclass/erroradaptation andno currentfailedbranchweights reused.

2026-10-04T20:29:49.296353+00:00: LighterencodedMixUpalpha.1/p.25/freshAdam1e-5 complete8VALID[184,180,176,176,174,175,174,177], best184<186, no16/noTEST. Firstactualprefmixupnottriggered (reportedfalse) butformalTRAINloggedtrueMiXUp coefficients, gradientspositive1534BProwwrites/no losers; initialVALID185 exact. SingleSNN mirrorreadout VALIDonly3 fixedcoefficients[0,.25,.5] gives185/185/183, losses.418773/.360598/.369487; baselinefullconfusion/loss exact, physicalallstateunchanged. Nonzero mirrorweight explicitlytwoinferencepresentations/T16each,total32 andaugmentationensemblecost~2x, notclaimedoneview/inferfreegain. Semanticmapping verifiedofficialCSV MD5 109b2ae... and localDatasetFolderclassindex permutation[0,3,2,1,7,6,5,4,8,9,10]; changesleftright/CWCCW inferredfromgeometry/officialnames, noTESTtrue-labelselection. Reflectionhelper8pureCPUchecks pass1.31s. No186gain/no newTEST; unusedv3two-view evaluator cannotbeinterpretedasmeasuredresult. Nextsource37 TRAINonlysemanticmirror+labelswapimplemented withprobability/permutationvalidation andactualloaderindexguard, disablednoRNG/legacyexact, oneTRAINpresentation/unmodifiedsingleviewVALID/TEST, allolderruntimeSHAsunchanged. Focusfirst11pass1regexfailure recognizedearlierexistingRNG-policyguard; fullfirst425pass1 CPUfixturewinnerpopulation failure archived (randomallSTDPwinner, notruntimebug). Fixtureseed1234/lowfirsthiddenweight.01/knownBP1-STDP4bias.5 makeseligibilitydeterministic; focusedandfull rerun pending. No source37formalTRAINuntilcompletechecks/preflight/initialVALID pass.

2026-10-04T20:31:19.210847+00:00: Source37 corrected deterministic fixture full426CPU/nativeCUDA tests pass41.09s; all33runtimeSHA frozen, originalsource36/414 unchanged. OwnoriginalVALID185 semantic-reflection TRAIN p.5 registered beforeformalTRAIN; freshAdam1e-5/frozenBNmoments, noMixUp, sameT16/res128/pool2/suffix2 andsingleviewinference. Labels transformed simultaneously underauditedclasspermutation[0,3,2,1,7,6,5,4,8,9,10], actualDatasetFoldermap guard beforebatch. 8roundscreening, eligible16 onlybest186+last3mean185, finalTEST onlyhighestVALID186/earliestties. InitialVALIDexactreplay andactualTRAINprefbeforelaunch; no officialTRAINunion, noTEST class/errors/subjecttuning.

2026-10-04T20:36:47.166697+00:00: SemanticreflectionTRAIN p.5 freshAdam1e-5/frozenBN complete8VALID[185,178,179,177,175,174,173,177], best185equalparent; TRAINhardCE.42119->.31159. InitialfullVALID185 exact/allstatesunchanged; actualpref4/8 reflected, CE.711484/CNNgrad43.2875/head4.12328/H1BP.347274/1453BP rowwrites/no losers; STDpeligible0 inpref andallloggedbatches. Bound186/last3mean185 gatesfail,no16/noTEST; originalsingleviewTESTassignmentstill269/288. NewownSGD1e-3 momentum.9/frozenBN/freshseed42ordinarygeometry/noMixUp/no semanticreflection registeredbeforelaunchfromsameoriginalb938parent. Source37 all426checks unchanged; extra2 CPU/nativeCUDA averagephaseSGD integrationchecks pass5.90s actualBP/STDPeligibility andexactmodel/momentum/scheduler/RNG/loader/monitorresume. SGSscalarLR differsfromAdam coordinateadaptiveLR, no matchedstep-size/singlefactorcausal claim. Same8->16 gates186/mean185 andTESTonlyVALID186, noTESTselection. New optimizer-awarepreflight actualSGD, initialVALIDbeforeformalTRAIN.

2026-10-04T20:52:31.459809+00:00: SGDplain ownVALID185 phase8complete VALID[185,175,184,175,169,169,171,168], best185; firstVALIDloss.38996<parent.41877 but accuracy gate186fails, no16/noTEST. ActualSGDpref1367/1534eligibleBPweightrows changed/no losers, STDpeligible0; allsource37/426 andextraSGD2checks unchanged. Source38 explicit custom_pool_after enables10 narrowerCNN layers/five spatialstages; full438CPU/nativeCUDA tests pass41.88s, legacyexplicitdefaultweights/logits bitexact, shape/rules/resume guards/actual freshBP-STDP allstate checked. NewfreshCNNwidths[32,32,64,64,128,128,256,256,512,512], poolafter[1,3,5,7], single8192hybrid, sameT16/res128/countlog1p/pool2/suffix2/exp/g16/presentation/margin/period16/drop.5. FreshAdam5e-4/plateau80, noown/externalweights/teacher/MixUp/reflection/frozenBN, BN16TRAINonly. Actualcompared5CNNsingle8192 budgetsphysical12160/logical10176/unfolded2039808 exactlysame; params46,192,784->26,299,413 (-43.06597percent). Nativepref20CNNbanks eachpositivegradient (min.007827/max.242708), CNNaggregate.470613/head19.55558/H1BP.029694;2347eligible rows allchanged =2189BP+158STDP,268STDPwinningevents/no losers. Peak5688.6875MiB=5.5554GiB (commentary earlier5.69GiB wasunits typo). First20gate153+recent5VALIDmean>=previous5+CErecent5<first5,40gate176+recentmean164, finalTEST onlyVALID186; allrulesfixedbeforeformalTRAIN. Controlleradaptation completedbeforelaunch: correct per-bankpreflight for40->80 and newstructural labels; old policy drafts archived, no HP/gates altered. Firstboundaryfixture wronglytreated [160]*10+[153]*10 as fallinglast5 (actuallyflatlatest10); archiveandcorrect to[160]*15+[153]*5, fourgatechecks pass without changingrule. Supervisor709224/nativeworker709228/stage20driver709294/stage40driver709295/selector709298 launched20:49:45Z afterrealworkerhandleverified. First3TRAINCE[2.2259,1.8316,1.5984]/VALID[77,82,91]/204, ongoingqualificationonly, noTEST. Reporter638363/audit_errors[] stillactive; goal3/4currentG269/288 remainsnotclosed.

### 2026-10-05 GroupNorm scratch candidate, after failed deeper CNN screen

The ten-layer two-half-width CNN completed its full 20-round screen. VALID correct counts: 77,82,91,113,104,112,129,107,117,127,137,130,130,142,134,124,122,125,137,127. Best142/204, last5mean127 vs previous5mean134.6; the frozen153 and trend gates both failed. No40/80 continuation or TEST; all native handles terminal.

Source_v39 adds optional per-sample/per-time GroupNorm using32 groups in the original five-CNN/single8192 scratch topology. Physical synaptic nonnegativity, WTA row eligibility, disjointBP/STDP, exponential theta and inference-only hidden2/4 quantization retained. No BatchNorm population state/refit; affineGroupNorm remains BP trainable. Reference https://arxiv.org/abs/1803.08494 establishes ANN small-batch behavior, not current SNN accuracy. CPU/nativeCUDA full454 checks passed40.09s, including defaultBN bitexact, temporal/sample independence and freshBP/STDP exactallstate resume. No accuracy proof from these checks.

Actual constructors matched all original size counts:46,192,784parameters;12,160physical bank width;10,176logicalspiking width;2,039,808unfoldedspatialneurons/time. Frozen before formalTRAIN: Adam1e-3/cosine160denominator/T16/countlog1p/pool2/full16mean/drop.5. Bounded actual20/40/80 rounds:20 gate153+recentVALIDtrend+TRAINCElower;40 gate176+last5mean164; TEST onlyfullVALID186. No own/external checkpoint, teacher, MixUp/reflection or TEST-selected changes.

Native actualTRAIN batch prefpassed:CE2.4040; all10 Convbanks and5 GroupNormaffine gradients positive;1357BP+103STDP eligibleweightrows all1460 changed, no loserwrites;295STDP events;6275.40MiB peak. GroupNorm didnotconsumeTRAIN for BN refit. Disposableupdatesdiscarded.

Native formalTRAIN launched2026-10-04T23:34:17Z: supervisor836243/worker836253/stage20driver836317/stage40driver836318/finalselector836319. Persistent goalreporter638363 remains responsiblefor status. Currentassignments unchanged:MNIST99.23/NMNIST98.75/CIFAR10DVS75.50 closed; latestGesture269/28893.40 remainsopen. HistoricalGesture272/28894.44 isnot currentassignment.

### 2026-10-05 Temporal classification banks, bounded own phase

GroupNorm32 fresh20 screen completed:VALID84,80,103,104,117,114,120,123,93,128,136,118,126,121,125,107,132,124,128,98. Best136/204, both153 andrecenttrend gatesfailed; no40/80 orTEST. Nativeallhandles terminal.

Source_v40 adds optional classification-head timebanks. Each time step has independent nonnegative classifier weights; sharedbias/configuredsuffixmean retained. The old shared coefficients are copied into all16 banks in an explicit own-from-scratch-origin phase. Conv/hidden/PLIF/BN/theta remainexactly frozen in this phase; no newhiddenBP/STDP writes claimed. OriginalWTA/BP-STDP training retained. Native/CPU full470 checks passed41.44s:disabledmodelegacybitexact, temporalorderdiscrimination, initialownpredictions, quantheadexclusion, freshBP/STDP+allstate resume, classifieronlyphase fullfrozenstate+exactresume. First CPUfocuscaught sizeauditone-step probing incompatibility and2fixtureassertion errors; fixedbeforefreeze, initialreport/source/tests archived.

Actualwidth/neuroncountunchanged:physical12160/logical10176/unfolded2039808. Classifierweights storage16x; wholemodel47,544,464params vs46,192,784 (+1,351,680=2.92617percent). Full16step classificationMACcount unchanged; not a runtime ensemble/multipleviews.

Candidate gesture_suffix2_temporal_classifier_only_lr1e5_prefix:fixedsourceb938 VALID185 average/suffix2 ownorigin; freshAdam1e-5/plateau16; first8fullTRAIN972/VALID204, extend16 onlybestVALID186+last3mean185. FinalTESTonlyhighestVALID186, earliestties andassignmentbeforeTEST. Native actualinitialfullVALID original andexpanded both185/204/exactconfusion; losses.418773229179 vs.418773100657, allmodelstateunchanged. RealTRAINdisposableCE.0026444/headgrad.02073; all14activebanksgradientpositive,2excludedprefixbankszero; allnonheadstatesexactfrozen. MeasuredcuDNNtf32true/matmulfalse, float32tensors/noAMP/noTRAINquant;peak2132.47MiB.

FormalnativeTRAIN launched2026-10-04T23:50:40Z supervisor852521/worker852531/selection852595/finalselector852596. Persistentgoalreporter638363. CurrentFP32gates3/4;Gesturelatest269/28893.40 remainsopen untilnewfixedfullTEST.

### 2026-10-05 Temporal head fullTEST and active encoder phase

The independenttime-bank classifier-only phase completed8+8 rounds:VALID185,185,186,186,185,185,185,185 then185,185,185,186,186,185,185,186. Registeredearliesttie selectedepoch3 beforeTEST; datasetassignmentupdated beforeTEST. FullTEST:FP32 273/28894.79167percent, uint2 272/28894.44444, uint4 273/28894.79167. FixedCP21361ad87ef4cc982d789807ab8ac22177cd284b9b778cf3a298603e33d0d6ed. Independentreplay858872 verifiedeveryfullprecision scorefield/confusion/loss exact; noTRAIN/newselection/statechange. CurrentGestureassignment gesture_suffix2_temporal_classifier_only_lr1e5_bounded_selected_full_eval. FP32 needs274correct, stillone short; otherthreeclosed. Quantgaponecorrect2bit/zero4bit; FP32first. No TESTclass/error/subject-based tuning.

Source_v41 allows a newownphase to retain trainedtimebanks andclear priorarchitecture-expansionmarker; ordinaryresume identityunchanged. All474 CPU/nativeCUDA checks passed44.32s, includingactualactiveencoderBP/STDP+exactallstate resume. Noaccuracyproof fromtests. Newphase retainsCP21361/timebanks, clearsaveraged-source/expansion flags, re-enablesCNN/PLIF/BNaffineBP andhiddenWTA BP/STDP, frozenBNrunningmoments, dropout.5, freshAdam1e-5/plateau16. First8only:VALID187pluslast3mean186 toextend16; finalTESTonlyhighestVALID187/earliestties. Candidate/controlconditionsfixedbeforeTRAIN; inheritedmodelwas selectedon VALIDbeforepreviousTEST. No externalorigin or newquantizedTRAIN.

ActualnativeprefinitialfullVALID186/204; all10convbanks+14activeheadbankspositivegradients; backbone1.32668/head.01739/hiddenBP.00191. All1534eligibleBProws changed, no losers; zeroeligibleSTDP/events in thisbatch, no newSTDPwriteclaim. BNaffinepositivegradients/runningstatsunchanged. Peak6285.60MiB, dtypefloat32, measuredcuDNNtf32true/matmulfalse/noAMP.

Original launchattempt timedout inautomaticapprovalbeforeprocesscreation; verifiedno launch/supervisorregistry. Onepermittedretry withsaved launch_active_temporal_phase.py succeeded. FormalnativeTRAIN actuallystarted2026-10-05T05:02:20Z (CST13:02:20):supervisor1089640/worker1089650/selection1089714/finalselector1089715. Source/config/gatesremainedfrozen during approvaldelay. Reporter638363 persists.

### 2026-10-05 Bounded higher-LR temporal head phase

Activeencoderphase completed8:VALID181,176,176,175,174,173,175,175; trainingCE.10245->.06149. Best181 belowparent186; registered187/last3mean186 gatesfailed, no16 ornewTEST. STDPeligibilityzero inpref; no positiveSTDPwriteclaim in thismaturephase. All failuresretained. CurrentfixedfullTEST remainsFP273/28894.79/uint2 272/uint4 273.

Newclassifieronlyphase gesture_temporal186_classifier_only_lr1e4_prefix retainsVALID186 CP21361/timebanks andallnonheadstates, clears sourcephase markers; source_v41 existing474 checks reused, no runtimechanges or redundanttests. FreshAdam1e-4 instead1e-5, plateau16, dropoutoff, frozenBN/Conv/hidden/PLIF/theta; first8 gate187+last3mean186, optionalcontinuous16, finalTESTonlyVALID187 earliestties. Code/helper/config/gatesfixedbeforeTRAIN. Realnativepref fullVALIDoriginal/new186 andallnonheadstatesexact; CE.0019665/headgrad.0125568, nohiddenupdates.

ActualnativeformalTRAIN started2026-10-05T06:07:22Z:CST14:07:22 supervisor1140320/worker1140324/selection1140393/finalselector1140394; reporter638363. Sourcecp selectedonVALID beforepriorTEST; noTESTclass/error/subject-guided tuning.

### 2026-10-05 Nonnegative temporal-head common-offset projection

Higher-LR clamp head phase completed8 VALID185,185,183,185,185,185,185,185; best185 belowparent186, registered187gatefailed. No16 orTEST. Fixed-parent/head-child interpolation coefficients1,0,.75,.5,.25 gavefullVALID186,185,186,186,185; all45nonheadstate tensors exact, no syntheticcheckpoint orTEST. Adjacent temporalreadout starts2,1,3 gave186,183,183; noTRAIN/TEST/checkpoint. FailuresandimmutableVALIDcache retained.

Newboundedclassifieronly condition gesture_temporal186_classifier_commonoffset_lr1e4_prefix retainsparentCP21361/VALID186, frozenallnonheadstates, source_v41/full474checks. FreshAdam1e-4/plateau16; onlychangevsfailedhigherLRhead is columnwisecommonoffsetprojection preservingclass-relative logits whilekeepingphysicalweightsnonnegative. Full8TRAIN972/VALID204; extend16 onlybest187+last3mean186; highestVALID187 earliesttie beforeanyTEST. ConditionsfixedbeforeTRAIN; noTESTclass/error/subjecttuning. Actualnativepref original/newVALID186, allnonheadfrozen, headgradient.0125568, CE.0019665; projectedrelative-logitmaxdifference1.52588e-5, CEexact/predictionspreserved, physicalmin0, peak2137.63MiB. Float32tensors/grads, cuDNNtf32true/matmulfalse, noAMP. No newhiddenBP/STDPwrites.

ActualnativeformalTRAIN launched2026-10-05T06:22:39.997409Z (CST14:22:39): supervisor1153925/worker1153929/selection1153999/fullTESTselector1154000. Persistentreporter638363. CurrentfullTEST remainsM99.23/N98.75/C75.50/G94.79167percent, 3/4FP32gates; Gesturestillrequires274/288ratherthan273.

### 2026-10-05 Per-step temporal-head BP condition

Common-offset Adam1e-4 phase completed8 fullVALID185,185,183,185,186,186,183,184; best186, fixed187gatefailed. No16/noTEST, currentGesture273/288 remainsopen. Next temporal-head condition uses per-step CE over all16 time banks while inference preserves the suffix2 mean; freshAdam1e-5/common-offset, frozenallnonheadstates. This supervises unusedprefixheads duringTRAIN too, explicitly recorded, not an inference change. Firstpreflight source41 configuration rejected this combination before anyTRAIN; failurelog/draftprotocol archived.

Source_v42 permits explicit custom per-step temporal-head CE while retaining mean-CE default, distillation prohibition andordinaryresume identity. All479 CPU/nativeCUDA checks passed43.44s (XML43.433s):independentlycomputed per-time CE/gradients onCPU/CUDA, actualownheadBP/frozennonhead/optimizerRNGloaderexactresume plus existingregressions. No accuracyproof fromchecks. Source41 frozen unchanged. Protocol/helpers/source42runtimechecksums fixedbefore newTRAINpreflight. Still bounded8/eligible16, continuationbestVALID187+last3mean186, fullTESTonlyhighestVALID187 earliesttie. Nativepreflight nowrunning; noformalTRAINclaim until liveworkerpublished.

ActualnativeperstepTRAIN started2026-10-05T06:28:51.881597Z (CST14:28:51):supervisor1160154/worker1160166/selection1160240/fullTESTselector1160241. Pref original/newfullVALID186; realbatch perstepCE.622117 vsmeanheadCE.0019665, headgradientpositive, allnonheadfrozen/physicalnonnegative. NativeGPUpref verifiedconfiguredloss against independentlyrecomputedflattenedCE; projectionrelative-logitmaxdifference1.14441e-5/predictionspreserved. Source42/full479 frozen. No newhiddenBP/STDPwrites inthisheadphase.

Persteptimehead completed8VALID185,183,184,184,183,183,183,183; TRAINperstepCE.8186899956->.7663735173, best185<187; no16/noTEST. Source42/full479 retained. Persistentreporter638363 wasfoundnativelyabsent, exited05:02:46 withAttributeError ontextualselection_basis ofnewfixed-evaluationparent. Fixedreportinghelper followsimmutableparent protocols tooriginalVALIDrecord, neverreadsTESTtoinferVALID; actualallcase refresh passesaudit_errors[]. Nativepersistentwatchrestored2026-10-05T06:30:10.690188Z pid1161831, SHA0925de3172e137776c7d237348479307fd383844d7feb0ba7777905499ab0871; originalfailure/helperarchived. PriortrainingandTESTartifactsunchanged.

Registeredhiddenbaselinecalibration[1,.95,1.05] fullVALID[186,187,184], allstoredparameter/BN/offset/rate tensors exact; onlycfgbaselinechanged. Selected.95beforeTEST, independentnativeVALIDreload exactly187/allscorefields, fullTEST273/272/273of288unchangedgoal3/4. IndependentfixedCPfullTESTreplay allprecision scorefields/confusions/histograms/loss exact, noTRAIN/statechange. Currentassignment gesture_temporal186_baseline0p95_VALID187_selected_full_eval selectedbyVALID; noTESTclass/error/subjecttuning. Reporter1167435withgenericthresholddiagnosticvalidation/current-assignmenttable, auditedsource helperSHAe570cc40faaadba543cf6072ea6586fb5de0a10adec5da3a5533022990912e16. NextregisteredfullVALIDthresholdneighbors[.95,.90,.975], required188; usesonlypriorVALIDdirection187vs186vs184, no posthocTESTconditionselection. Source42/full479 unchanged.

### 2026-10-05 Joint from-scratch temporal-bank model

VALIDthresholdneighbors[.95,.90,.975] gave187,187,186; fixed188gatefailed/noTEST, .95 assignmentretainedbyearliesttie. LatestfullTEST273/272/273of288 independentexactreplayverifiedandrecordedinoriginalrun/test_replay.json.

Newroot gesture_temporalbank_single8192_fromscratch_from_scratch_prefix trainsoriginal5CNN/single8192WTAhybridwithtime-bankclassifier jointlyfromscratch; noown/externalcheckpoint/teacher. T16/full16meanCE/countlog1p/res128/pool2/drop.5/BNrefit16/Adam1e-3, common-offsetnonnegativeheadprojection. Samephysical12160/logical10176/unfolded2039808, actualCPUconstructor47,544,464params verified (+2.926percentheadstorage), oneSNN/oneview. Source42/full479runtimechecks reusedunchanged; modelseed42/splitseed42. Boundedactual20/conditional40/conditional80, original160cosinedenominator retained; noactual160extension.20best153+last5VALIDtrend+TRAINCElower,40best176+last5mean164; fullTESTonlyhighestVALID188/earliestties. Multiplejointchangesfromoldroot declared, no isolatedcausalclaim; allconditionsfrozenbeforeTRAIN.

RealnativeTRAINpreflight:CE2.39543, all10physicalConvbankspositivegradients, head7.50029/CNN.20704/hiddenBP.02266;2457eligibleweightrows allchanged,2311BP+146STDP,234winningSTDpevents/noloserwrites.15timebankspositivegradients, firststepzerospikefeatures/zerograd recorded. Common-offsetrelative-logitmaxdifference1.90735e-6, CEexact/predictionspreserved/physicalmin0, peak6285.59MiB. Float32tensors/grads; cuDNNtf32true/matmulfalse/noAMP/noTRAINquant. Disposableupdatesdiscarded.

FormalnativeTRAIN actuallystarted2026-10-05T06:39:43.386838Z (CST14:39:43):supervisor1171057/worker1171061/stage20driver1171126/stage40driver1171127/fullTESTselector1171128. Reporter1167435runningwiththresholdscope/currentassignmenttable, audit_errors[]. Fulltestgoalremains3/4; Gesturestill273ratherthanrequired274/288.

2026-10-05T06:50:59.385624Z: Independent-initialized fresh timehead jointCNN completed20VALID[74,99,108,109,136,112,101,101,116,107,129,124,136,145,116,139,135,127,145,131]; best145/204at14. Recent5mean135.4>=prior5130 andTRAINCElower2.09->.61449, butfixed153gatefailed. No40/80/TEST. Nativeallfivehandles terminalverified. Goalremains3/4/currentG273/288.

Preparing source_v43 shared fresh initialization: independenttimebanks begin as exactcopiesof one newlyrandom legacyhead, then remainindependentlytrainable; noexternal/ownCP, no tiedweights. CPU focused10checks passed4.66s; additional explicit-default bitexacttest addedafterfocusedrun, tobeincluded infullsuite. Legacybody/head coefficients andCPU_RNG exact, initialCPUlogits close, mathematicalfeaturegradients consistentacrosstime yetweightbanksreceive differentBPgradients andcan specialize. Ordinaryresume cannotchangeinitidentity. Default independentbehaviorretained. FullCPU/nativeCUDA suite launchedonlyafteroriginal20branchterminal. No accuracyproof fromtests, nextformalTRAIN pendingfullnativeverification.

Source_v43 full492 CPU/nativeCUDA checks passed44.19s; allruntimefiles frozen. Freshsharedtimehead init preserveslegacyrandomhead/body/CPU_RNG; bankweightsindependentlytrainable, actualBP/STDP/exactallstate resume verified, defaultindependentbitexact. Nativepref confirmed all16initialbanks identical andinitialfeaturegradientdifference0.0; CNNgradient.484237/head7.50913/hiddenBP.054379, all10Convbankspositive;2457rows allchanged (2311BP+146STDP),234STDPevents/noloserwrites. Commonoffsetrelative-logitmaxdifference2.861e-6, CEerror4.47e-8/predictionspreserved, peak6289.59MiB, nonnegativefloat32/cuDNNtf32true/matmulfalse/noAMP. Onlyinitmodechanges vscompletedindependenttimebankroot; T16/countlog1p/res128/width/optimizer/augmentation and20/40/80gates unchanged. One SNN/noexternal orownCP. Actuallaunch 2026-10-05T06:54:50.729716+00:00 supervisor1186501/worker1186511/controllers[1186575, 1186576, 1186577]. Reporter1167435; goalactive3/4/currentfullG273/288.

2026-10-05 current-selected fullTEST/physical-weight audit: M9923/9927/9925,N9875/9877/9878,C755/759/755,G273/272/273; exact10000/10000/1000/288 denominators, no sample/step limits, exponentialmode, actualall50physicalsynaptic tensors finite nonnegativefloat32 (13/13/12/12), hiddenonlyquantization/headandCNNexempt. Allfourindependentfixed-checkpointreplaysverified; C previouslycompletednative replayalreadyexact butmarkerabsentfromoriginalrun. ComparedallprecisionfieldsandCPsha9cebc... again readonly andlinked test_replay.json without newTRAIN/TEST. QuantdropM/N/Cnonpositive; G2bitonecorrect0.34722pp,4bitzero. NoFP32completion: Gstillonecorrectshort. Saved selected_goal_requirements_audit.json explicitlydefersruntimeBP/STDP/provenanceaudit, not a falsecompletionflag.

Shared-init freshtimehead20 completeVALID[98,100,104,118,123,124,101,106,128,130,124,129,134,139,99,138,128,111,147,131]; best147at19<153, no40/80/TEST; source43full492 andnativeallhandles terminal. NewcachedVALID temporalreadout mean186, class-centered logsumexp temperature16=187/temperature4=183; fixed188gatefailed/no implementation orTEST yet. Classcenteringremoves per-time class-common logitshift; nophysicalweightchange/extraforward orneurons. PreregisterednextVALID-only combination on independentVALID-selectedthreshold.95 withtemperatures16/8/24plusmeanbaseline, mustreplay187 baselineexact andallmodelstateunchanged; futureTESTonlyifVALID188. ChoiceusesonlyearlierVALIDdirection, noTESTerror/class/subjecttuning.

NativefullVALIDthreshold.95+readoutcombination replayedbaseline187 allscorefields exact; temp16=187,temp8=186,temp24=187; no188, noimplementation/TRAIN/TEST/newCP. Allstoredweights/BN/offset/rateexact andGPUprocesscompleted0. NewregisteredCPUcachedoldthreshold1 neighbors16/8/24 refineVALID187temperature16 only; noTEST class/error/subjecttuning.

Originaltheta1 cachedsoftreadout neighbors16/8/24 completedVALID187/186/186 vsmean186, min188failed. No runtimeimplementation/newCP/TRAIN/TEST.

Registered frozenbody shared-classifier bias-only calibration on currenttheta.95 CP3538 beforeoptimization: fullTRAIN972 noaugmentation/dropout, frozenBN/theta/physicalweights, only11sharedbias float32 BP; objectiveTRAINCE+.001*sum(delta^2), LBFGS50maxiterations/70maxeval, zero-mean canonicaldelta; native completeVALID204 candidates[0,.25,.5,1], highestcorrect earliestties, required188beforeTEST. Nativebiasgradient/logit vsfrozen-cache proof requiredbeforeoptimization, disposableprobe/allstateexactrestored. Source43/full492 reused unchanged; noTESTclass/error/subjecttuning, no newhiddenBP/STDPwrites.

Bias-only calibration nativecompleted0: disposablebiasgradient/cachemaxdifference3.31784e-9/logit8.39308e-6; allnonbiasgradientsabsent/allstateprobe restoredexact; fullTRAIN972 CE.04398806020617485->.041359253227710724, LBFGS9closures. NativeVALIDscale0/.25/.5/1 scores187/187/185/185; required188failed. No syntheticCP/noTEST. Originalassigned273/288retained.

Registeredoriginal5CNN/single8192/sharedhead seed42 scratch condition with exponentialdelta.2 instead.05, tau20/baseline1; no temporalbank, same originalclamp/noCP/teacher. Physicalwidth12160/logical10176/unfolded2039808/46,192,784params. Source43/full492 reused. TRAIN972/VALID204, T16/res128/log1p/pool2/drop.5/Adam1e-3/originalcosine160 denominator; actualbounded20/eligible40/eligible80 only. Fixed20best153+last5VALIDtrend+TRAINCElower,40best176+last5mean164, TESTonlyhighestVALID188earliestties. HypothesisredistributeWTAeligibility; strongerdelta efficacyunproven. Code/config/policies/helpersSHA fixedbeforepreflight/TRAIN.

NativeformalstrongerdeltaTRAIN actuallylaunched 2026-10-05T07:22:57.094049+00:00: supervisor1211949/worker1211958/controllers1212022,1212023,1212024; source43/full492. ActualCPUconstructor46,192,784params/12160physical/10176logical. ActualnativeTRAINpref CE2.40245/CNNgrad.488475/head16.44989/hiddenBP.055326;2835eligibleweightrows allchanged,2662BP+173STDP/248winningSTDp events/no loserwrites/nonnegativefloat32/peak6275.28MiB. Discardedprefupdates; noTRAIN/TESTquant, noCP/teacher. Reporter1167435 remainsnativealive.

Preregistered temporal calibration onVALIDselectedtheta.95 CP3538: TRAIN972 frozen eval-mode suffix2 temporal sequences, 14 positive factorssoftmax*14, CE+.01*mean((factor-1)^2), LBFGS50/70; scales[0,.25,.5,1], fullnativeVALID204 highestcorrect earliestties188required. Learnedfactorsabsorb directly inexisting classifierbanks, sharedbias/Conv/hidden/PLIF/BN/theta exact, nonnegative/oneSNN/noextra neurons/runtimeparameters/forward. Nativeactualheadgradient dottedwithoriginalbank vscachedfactorBP/logitequivalence proof first, disposableprobe restoredallstate; source43/full492 unchanged. Queuedbehindcurrentstrongerdelta fullprefix, noTESTclass/error/subjecttuning.

Persistentreporter1167435 crashed07:28approx because standalonecalibration launch.json wronglymatchedstandardtrainingcase discoveryandhadno config.yaml. NativeTRAIN/queuedcalibration unaffected. Preservederrorlog; renamedonlystandaloneregistryto calibration_launch.json, savedregistry_naming_correction.json, helper/runtime/protocol unchanged. Actualrefresh audit_errors[]; persistentwatch restartedfromsame e570 SHA (goal_report_after_calibration_registry_fix_launch.json).

Strongerhomeostasis20VALID[72,81,120,105,117,135,85,98,115,137,133,142,141,122,125,134,135,132,163,142], best163at19; last5mean141.2vsprior5132.6, TRAINCElower, allfixed20gatespassed. Nativecontinuous40preflight/continuation launched byregistereddriver; noTEST. TemporalBPcalibration nativecompleteVALID [187, 187, 185, 182] forscales[0,.25,.5,1], no188/noCP/noTEST. Nativegradient/logit equivalence andallnonheadstate exact verified; positivefactors absorbedinhead, noextra neurons/runtimeparams; seeexactpref/results.

Preregisterednext original5CNN/single8192 sharedhead scratch T8 seed42/split42 condition beforeanyT8TRAIN: onlyT8 vsoriginalT16 root; delta.05tau20baseline1/drop.5/Adam1e-3/originalcosine160/BNrefit16same; allrawadjacentbins summedbeforelog1p, noeventdiscard. Same12160physical/10176logical/46,192,784params; forwardstepshalved, actualruntimeaccuracyunproven. FullTRAIN972/VALID204, samebounded20/conditional40/conditional80gates153/trend/CE then176/recent164, TESTonlyhighestVALID188earliesttie. Source43/full492 reusedunchanged. Plannedstartonlyafterstrongerdelta branch fullVALID screen fails188; noTESTclass/error/subjectselection, no T16 checkpoint reused.

T8actualCPUconstructor46,192,784params/12160physical/10176logical andactualfirstTRAIN8 countaudit passed: raw andpaired1,726,049events exact, raw16frames to8pairbins beforelog1p. CPUparallelad-hocaudit hit sandboxIPC PermissionError; identityverified stoppedonly itsCPUparent1228133/fourchildren, serialauditsucceeded withoutchangingexperimentalnum_workers4. Deferrednativecontroller1230740 launched2026-10-05T07:40:24.193246Z, policySHA/helper/prereglocked, currentlywaiting onconfirmedliveparentselector1212024. T8TRAINhasnotstarted; ifparentVALID188qualifieddefer instead. Reporter1221238 unchangede570SHA.

Strongerdelta.2 continuous40completed fullVALID21-40[144, 143, 155, 143, 161, 152, 145, 164, 161, 161, 149, 159, 148, 163, 164, 149, 143, 141, 152, 167], best167<176 andlast5mean150.4<164; bothcontinuationgatesfailed. No80/noTEST. All20newroundsTRAIN972/VALID204 verified, parentmax167<registered188TESTgate. DeferredT8controller executesregisterednativepreflightthenfreshTRAINonlyafterparentselector terminal.

T8initialactualnativepreflight passed, butboundedlauncher overstrictlyrequired everyeligibleweightrowchange, sofirstdeferredlauncher endedbeforeTRAIN withAssertionError; no supervisor/workerwascreated. Archivedoldhelpers/protocols/policies/prefbeforepreTRAIN amendment. Addedseparate actualBP/STDProwcounters innew candidate_T8_training_preflight.py SHA00bc705b0089faaf5feaf6aaa33686a7a9f44619e887dcbe74fc6234ad959e50; boundedlauncher nowrequires0<changed<=eligible pluspositiveactualBP/STDP changedrows, preservesnative loser-weight/stateprotection. Source43/all492checks unchanged. PreTRAINnewpref:2577eligible/2562changed,2405BP+157STDP;15BPeligiblerowszeroweightupdate legitimate; all no-loserwrite checks retained. CNNgrad.3052665/head16.98756/hiddenBP.034435/peak3397.01MiB/allphysicalnonnegativefloat32. ActualfreshTRAIN launched 2026-10-05T07:47:32.590716+00:00 supervisor1237693/worker1237697/controllers1237766,1237767,1237768; discardedprefstates. ParentfullVALID167failed188, noTEST.

Preregisterednext conditional T8single8192 time-specific CNN BN fromscratch: original5CNN/single8192/sharedhead, sameT8countcoding/delta.05tau20/seed42/split42/drop.5/Adam1e-3/cosine160/BNrefit16/aug, onlysharedBN->8independent gamma/beta/moments perCNN layer. CurrentruntimehasexistingTemporalBatchNorm2d/source43/full492coverage; no runtimeedits. Physical/logical/unfoldedneuronsunchanged, 46,220,560params (+27,776/0.06013%). Realnativeprefmustproveall40timeBNgamma banks gradientspositive plusBP/STDPactualchangedrows/loserprotection/nonnegative. Oldnegative timeBNwarm8 hadtwolayerstdp/bp4096/group64/pool4, not thisfreshcondition. StartonlyaftercurrentT8 fullVALID20/40/80screen fails188; sameboundedgates153/trend/CE then176/recent164. NoTESTclass/error/subjecttuning.

OrdinaryT8sharedBN20completeVALID[83, 76, 95, 112, 118, 119, 122, 101, 111, 117, 112, 132, 138, 137, 131, 123, 137, 126, 140, 134], best140at19<153; no40/80/TEST. Nativeparentallhandles terminalverified. PreregisteredtimeBNCPU actual46,220,560params/12160physical/10176logical/2,039,808unfolded/40gamma-beta banks, identicalauditedT8coding. Conditionalcontroller1245619 launched07:54:14.748459Z; parentVALID140failed188, executesregisterednativepreflight. Source43/full492 unchanged.

Actualnative T8timeBNpreflightpassed:all40per-timegamma-bankgradientspositive, CNNgrad.337442/head17.13025/hiddenBP.041782;2603eligible/2598changed,2436BP+162STDP/249winningSTDp events,5eligibleBProwszeroupdate/noloserwrites. Peak3397.23MiB/float32physicalnonnegative, disposableprefupdatesdiscarded. FormalfreshTRAIN launched 2026-10-05T07:54:36.054934+00:00 supervisor1246103/worker1246108/controllers1246172,1246173,1246174; parentordinaryT8VALID140failed188, allparametersfromscratch/noown/externalCP/teacher. Source43/full492 unchanged; reporter1221238.

Preregisterednext originalT16/sharedBN/single8192/shared-head common-offset FROM SCRATCH: onlyheadprojection changesvsoriginalclamp root; delta.05tau20/T16/log1p/pool2/drop.5/seed42/split42/Adam1e-3/cosine160/BNrefit16/augmentation unchanged. Per-input-column commonoffset preservesclass-relative logits/softmaxCE whileheadphysicalweightsremainnonnegative; accuracybenefitunproven. Differentfromold2hiddenwarm commonoffset andfresh16time-bankroots. Same46,192,784params/12160physical/10176logical/2039808unfolded, oneSNN/oneview. Existingruntime/source43/full492 reused. Nativeprefrequiresall10physicalCNNgradientspositive/BP+STDPactualchangedrows/noloserwrites/commonoffsetequivalence. ConditionstartonlyafterT8timeBNwholefullVALIDscreen fails188, samefixed20/40/80gates153/trend/CE then176/recent164; noTESTclass/error/subjecttuning.

T16sharedheadcommonoffset actualCPUconstructor46,192,784params/12160physical/10176logical/2039808unfolded verified. Actualoriginalrootconfig differences onlyheadprojection/outdir/explicit split_seed42 matchingoriginaldefault42; source21->43 unrelatedconditionalextensions tracked, defaultbehaviorregressions retained. Deferrednativecontroller1252273 launched08:00:00.974661Z, waiting onconfirmedliveT8timeBNselector1246174; allconfig/protocol/policy/helper/sourceSHA re-audited, source43full492 unchanged. No commonoffsetTRAIN yet/no TEST.

T8timeBN20completeVALID[88, 72, 95, 113, 117, 111, 114, 122, 126, 128, 127, 127, 135, 126, 120, 125, 142, 125, 144, 125], best144at19<153; no40/80/TEST. Nativeallpriorhandles terminal. NextoriginalT16commonoffsetnativeprefpassed:all10CNNbankgradientspositive/BP+STDPactualrowschanged/noloserwrites and commonoffsetrelative-logit/CE/prediction preservation checked; exactfieldsinpreflight.json, backendTF32cuDNNtrue/matmulfalse/noAMP. FreshformalTRAIN started 2026-10-05T08:02:52.194386+00:00 supervisor1255290/worker1255299; controllerhandlesfromdeferred_launch.log. OneSNN/sharedhead/T16; some templatedebugstage strings retainT8 labels, externalprotocol_label_clarification.json documentsactualconfig/nativeframes withoutmutatingregisteredhelpers. Source43/full492 unchanged/no TEST.

Read-onlyauditable figure/CSV/evidence gesture_first20_screening_curves generatedfromactualcompleteTRAIN972/VALID204epochs oforiginalT16clamp/strongerdelta/T8sharedBN/T8timeBN/currentT16commonoffset. LastpartialJSONrecord ignored; nativefirst20scores/loss/counts exact, metric snapshotsboundSHA, zeroTRAIN/TESTsteps. Originalhistoricalgate differs; figurecurrent153gate doesnotretrospectivelychangeoldpolicy. Epochsnotmatchedcompute, no isolatedcausal/superiority claim. Figure visuallyinspected, no truncatedlabels.

T16commonoffsetseed42 first20VALID[91,96,117,122,111,144,128,124,131,133,160,142,144,159,127,127,136,142,161,163], best163at20; last5mean145.8<previous5146.4, despite153precision/CEgatespass, registeredtrend failedby3correct across5VALIDpasses. Stoppedno40/80/TEST; no posthocgatechange. Curves/CSV/evidence retained. Preregisterednext independentrandomseed7 commonoffsetscratch replica, sameT16/sharedBN/width/delta.05/optimizer/coding/HP andfixedsplit42/972204288. Seed7 changesweights/TRAINshuffle/augmentation/dropoutRNG together, not onlyinitialization. Explicitnewrunner/selector/evaluator pathsseed7/TESTseed1007; no runtimechanges/source43full492 reused. Same20/40/80gates/min188beforeTEST. No TESTclass/error/subjecttuning.

Seed7replica actualCPUconstructor46,192,784params/12160physical/10176logical; allnewrunner/selector/continuation/evaluator runpathsseed7 staticASTchecked/nohardcodedseed42paths, TESTcfgseed+1000=1007, split_seed42samee94manifest. Actualnativepref CNNgrad.448809/all10Convbankgradpositive;2702eligibleallchanged,2539BP+163STDP/noloserwrites/nonnegativefloat32;commonoffsetrelative-logitmaxerror1.43051e-6/CEdifference1.19e-7/predictionspreserved. FreshformalTRAIN actuallylaunched 2026-10-05T08:18:31.356738+00:00 supervisor1270499/worker1270503/controllers1270567,1270568,1270569. Source43/full492 reusedunchanged, alldisposableprefupdatesdiscarded/noTEST. Goalremains3/4/currentG273of288.

2026-10-05T08:29:55.355654+00:00 preregistered conditional rebalancedCNN freshT16 candidate: fivechannels[128,256,512,576,512], sum1984 unchanged vsoriginal[64,128,256,512,1024]; single8192/sharedclassifier retained, common-offset/seed42/split42/BN/augmentation/delta.05tau20/Adam1e-3/cosine160 same. ActualCPUconstructor30,450,064params (-34.08%), physicalwidth12160/logical10176 unchanged, spatialneurons3,858,432 (+89.16%; compute/activation notmatched). Hiddenfanin4096->2048; jointstructurechange, no isolatedcausalclaim. Source43/full492 reusedunchanged. Startonlyafter currentseed7 allboundedVALIDscreen fails188; fixed20gate153/trend/CE and40gate176/recent164, finalTESTonlyhighestVALID188 earliestties. Nativepref mustprove all10CNNbanks positivegrad/BP+STDPactualchangedrows/loserprotection/commonoffset/nonnegativefloat32 andmeasurememory; no TESTclass/error/subjecttuning. Allnewconfig/protocol/helper/policySHA fixedbeforepref/TRAIN.

Seed7originalT16commonoffset20completed VALID[65, 109, 122, 117, 112, 125, 136, 116, 144, 142, 121, 134, 133, 108, 143, 134, 148, 148, 152, 139]; best152at19<153, despite trend/CEpasses, stopped no40/80/TEST. Full972TRAIN/204VALID all20epochs; TRAINCE2.56467->.508853. Nativeallfivehandles terminalbeforeconditionalrebalancedlaunch; no posthocgatechange.

RebalancedCNN actualnativeprefpassed: all10physicalConvbankgradientnormspositive; CNNgrad0.537856936454773/head17.253633499145508/hiddenBP0.03835885599255562;2515eligibleallchanged2373BP+142STDP/243winningSTDp events/noloserwrites. Commonoffsetrelmax1.90735e-6/CEerror1.19e-7/predictionspreserved/min0, allfloat32/cuDNNtf32true/matmulfalse/noAMP; actualpeakallocated11252.3486328125MiB. Disposableprefdiscarded; actualfreshTRAINlaunch2026-10-05T08:30:44.611400+00:00 supervisor1282776/worker1282780/controllers1282844,1282845,1282846. GPUprocessallocation15542MiB atfirstminute distinctfromprefpeakallocated. Source43/full492 unchanged; reporter1221238 live, allaudit_errors[]. NoTEST.

Selectedbenchmark snapshot selected_benchmark_snapshot_20261005_0834 frozenfourcurrentVALID-selected completeTEST results/CP/configs/splitmanifest/replay evidence, originalM/N31-file runtime copiedverbatim checkedlaunchSHA, C/G33-file pinnedsource24/42. CPUverify_selected_benchmark_snapshot actualallhashes/fullcounts/scorefractions/configs/exactreplays passed; noTRAIN/TEST/CUDA. CPshaM c632fe77..., N e22b60b9..., C9cebc..., G3538...; goal3/4/G273of288, all_FP32_goals_met false. Scopeartifactconsistency, notnewaccuracyproof orruntimeweightbehavior. Packageversionsrecorded, originalartifactspreserved.

RebalancedCNN fullfirst20VALID[96, 88, 117, 122, 132, 142, 130, 109, 125, 129, 135, 138, 143, 145, 161, 148, 145, 163, 151, 165]; best165at20, last5mean154.4>=previous5144.4, last5TRAINCEbelowfirst5: allregistered20gatespassed. Actualcontinuous40resume launched08:49:51Z driver1282844/worker1300868; originalweights/Adam/cosine/BN/theta/RNG/loader/guard retained. Fullround21-40VALID[153, 149, 156, 139, 166, 158, 163, 170, 161, 157, 181, 162, 160, 156, 173, 161, 174, 163, 177, 187]; best187at40, last5mean172.4>=164 andbest>=176, allstage40gatespassed. Nativecontinuous80registeredresume actuallylaunched2026-10-05T09:08:53.358742+00:00 driver1282845/worker1318020; source43/full492 unchanged, noTEST; finalTESTonlyhighestfullVALID188earliestties. Lastread44epochs, currentnewstage[181,162,173,177], bestglobal187.

Registeredconditional standalone TRAIN-only BNaffine/PLIF calibration on fixedtheta.95 CP3538: only15tensors/3973scalars trainable, allnonnegativephysicalConv/hidden/timehead/headbias weights/BNrunningmoments/hiddenthetaexact frozen. ModelPLIFtraining surrogate/BNmoments frozen/hiddeneval/dropoutoff/noaugmentation; no new WTA/STDP/hiddenBP writes, originalscratch-origin learningretained. FreshAdam1e-4/anchorL2.001/TRAIN972,4-roundgatebest185andlast2CEbelowfirst2 to8max; nativefullVALID204 highestcorrectearliesttie, required188beforeTEST. CPUconstructor47,544,464params/15tensors3973 exact; actualinheritedsource43/full492 unchanged (initial479 label correctedbeforeanynativepref/TRAIN, archivedoldprotocol). Standalonephase explicitlyoverridesoriginalconstructor trainingfreeze flags, inferencecfgunchanged; helper/protocol/SHA/nativegradient-and-frozenstatepreflight fixedbeforeTRAIN. StartonlyaftercurrentrebalancedboundedVALIDscreen fails188; defer ifqualifies. NoTESTclass/error/subjecttuning.

Nativeconditional normalization/PLIF controller launched onlywaitingon confirmedparentselector1282846; registry deferred_launch.json (notstandardlaunch.json), afterterminalVALIDfailure actorregistrycalibration_launch.json toavoidstandardreporter misclassification. No calibration TRAIN/TEST started atlaunch; guarddefer ifrebalancedVALID188 qualifies, source43/full492/CPU3973/47,544,464 fixed.

RebalancedCNN plateau55 terminal09:22:55Z/stale15, fullround41-55VALID[181, 162, 173, 177, 174, 156, 165, 179, 180, 175, 177, 143, 161, 161, 163]; bestglobal187at40<188, finalselector no_improved_VALID_checkpoint_no_TEST. All55fullTRAIN972/VALID204, lastTRAINCE0.07634369152679346; nativealloldhandles terminal. ConditionalnormalizationPLIF calibration actuallyexecuted09:23:12Z native1330592. NativeactualTRAINpreflightpassed all15BNaffine/PLIF gradientspositive/finite andall15changed; trainingmodevsinference logitsdifference0.0, allphysicalConv/hidden/timehead/headbias/BNrunningmoments/hiddenthetaexact andallnonselectedgradientsabsent; disposablefullstaterestored. Source43/full492 unchanged/noAMP/float32/cuDNNtf32true/matmulfalse; bounded4/8 calibration nowactive/noTEST.

NormalizationPLIF calibration full4VALID[185,184,182,184], TRAINCE[.040576,.039974,.040713,.040693]; recent2CEworsefirst2, fixed4gatefailed/no8/noCP/noTEST, native1330592 terminal. Newregisteredrebalanced187 temporalclassifieronly phase: fixednewbody bestVALID187at40 fromscratch, copysharedhead into16independentbanks, sharedbias/all16mean retained(start0), commonoffset retained. AllConv/BN/PLIF/hidden/thetaexact frozen; onlyheadfloatBP/dropoutoff/augunchanged/freshAdam1e-5/plateau16. First8, extend16 iffbest188+last3mean187, TESTonlyhighestVALID188earliesttie. ActualCPUconstructor31,801,744params (+1,351,680/4.438%headstorage), physicalwidth12160/logical10176/spatial3,858,432 unchanged; sameheadMACcount, runtimenotmatched. Allnonheadstate andsharedbias exact/copyweights verified; source43/full492 unchanged. NativeinitialVALID187/equivalence/headgrad/allfrozenstate proofbeforeTRAIN. No TESTclass/error/subjecttuning.

Rebalancedtemporalheadinitial nativepref rejectedunconditionalall16bankgradientpositive beforeTRAIN; fullinitialVALID original/expanded187 andconfusionexact alreadypassed. Sourcebodynewall16mean includesearlyunfiredtimebins; preTRAIN guardamended to checkactualperbankhiddeninputactivity: everyactivebankgradpositive, everyinactivebankgradzero. Newpref_v2 copiedoldSHAhelper, archivedalloriginalprotocol/policy/launch/error/logs, nativeold1338497terminal/noTRAINworker existed. HP/source43/full492/data/weights/VALID188gate unchanged; actualactivitydiagnostic will verifymechanism ratherthan assume.

Rebalancedtemporalhead nativeactualTRAIN activity confirmedstep0hiddeninputmass0/weightgradient0; remaining15banks inputactivitypositive andBPgradientpositive. Original/expandedfullVALID187/confusionexact/losswithin1e-5/allstateunchanged; allnonheadfrozen/CNN-hiddennograd-orwrites/BNmoments-andthetaexact. Headgrad.0745369/commonoffsetrelmax3.05176e-5/CEdifference2.887e-8/predictionspreserved/min0; peak3524.06MiB/float32/cuDNNtf32true/matmulfalse. Originalprefstrictall16 failure correctedbeforeTRAINwithactualinactive-bankguard andarchivedevidence. Formalfreshheadphase actuallylaunched09:31:38.26955Z supervisor1340790/worker1340794/driver1340858/selector1340859; nativeallalive. Source43/full492 unchanged, noTEST.

Rebalanced187temporalheadfirst8 completeVALID[187, 185, 183, 183, 183, 183, 182, 182]; max187<188/recent3gatefails, no16/noTEST; fullTRAIN972/VALID204, frozenallnonheadstate/source43full492. NextregisteredVALID-onlynewrebalancedsharedhead CP392f1f... (best187at40) baselines[1,.95,1.05], maxfullVALID204 earliestties, min188beforeTEST. Allphysicalweights/BN/offset/rate tensors exact; onlynonpersistent hiddenbaselinecfg changes, expdelta.05tau20 retained, noTRAIN. Control187/confusion/lossreplay required. Source43/full492 unchanged; noTESTclass/error/subjecttuning.

Newrebalanced187 fixedthresholdscan nativeVALID[187,187,186] atbaselines[1,.95,1.05], no188/noCP/noTEST; allmodeltensors exact. NextregisteredfrozenVALID-onlysuffixreadoutstart[0,1,2] onoriginalrebalancedCP392f1f...; all16timestepsstill simulated, oneSNN/noextra forwardsatfinalinference, source43/full492 unchanged. FullVALID204 control187/everysequence fullmean exact/nativeallstateunchanged; min188 earliestties beforeTEST; noTESTclass/error/subjecttuning.

Newrebalancedreadoutstart[0,1,2] fullVALID187/187/186; no188/noCP/noTEST, allstateexact. Nextregisteredownfromscratch rebalancedVALID187best40 activewarmphase: architecture/T16/log1p/res128/pool2/sharedhead/thresholddelta.05tau20/drop.5/aug/WTAhybridperiod16 unchanged; freshAdam1e-5/plateau16 andfrozenBNrunningmoments. CNN/PLIF/BNaffine/headBP andhiddenWTA BP-STDP active; BNrunningmoments frozen. InitialCPUallmodelstateexact/30,450,064params, source43/full492 unchanged. First8, optional16 iffbest188+last3mean187; fixedfullTESTonlyhighestVALID188 earliestties. NativeprefrequiresfullVALID187 identity/all10CNNgradpositive/actualWTAprotectedupdates/nonnegativefloat32/commonoffset/BNmomentsexact. NoTESTclass/error/subjecttuning, combinedoptimizationnotisolatedattribution.

RebalancedactivefrozenBN nativeprefpassed initialfullVALID187/allinitialmodeltensors exact; all10physicalCNNbanks gradientspositive, CNNnorm4.86357/head.152118/hiddenBP.00894163. 1324eligible/1322BProwschanged/2eligibleBPzero updates/no loserwrites; actualfirstbatchSTDPeligible0/writes0 explicitly, earlierfreshroot2373BP+142STDP updatesretained. BNrunningmoments exact, CNN/BNaffine/PLIF/hidden/head active, nonnegativefloat32, commonoffsetrelmax2.28882e-5/CEdiff8.708e-8/predictionspreserved, peak11252.35MiB. Formalfreshownphase launched09:43:07.034748Z supervisor1351120/worker1351135/driver1351199/selector1351200; source43/full492 unchanged/cuDNNtf32true/matmulfalse/noAMP/noTRAINquant. CompleteVALIDonly8/eligible16, noTEST; currentFP32goal3/4/G273of288.

Registerednext direct-parent weights/theta2x2 VALID-only diagnostic afteractivefrozenBN188 gatefailure: originalrebalanced40 CP392f1f... versushighestcompletedVALID warm8/eligible16 earliesttie child. Fourfixedstates parentweights/parenttheta, child/child, childweights/parenttheta, parentweights/childtheta; everytensor exactfromdeclaredsameorigin CPs, BNrunningmoments mustidentical. CPUcurrentbestchild1(thetaauditonly):1560offsetschanged/max.101505, rates/BNmomentsunchanged. ActualfullVALID204 controlsbothparents/child exact beforeinterpretingcombinations; min188 beforeTEST, noTRAIN/TEST/newruntime source43/full492. StartonlyifparentfailsVALID188; defer ifqualifies.

### 2026-10-05: rebalanced warm phase / weights-theta diagnostic / group assignment

The rebalanced own active frozen-BN Adam1e-5 phase finished all eight full TRAIN972 / VALID204 epochs with correct counts 182,181,176,180,179,178,178,179. The fixed continuation and TEST gates failed. No 16-epoch extension or TEST was run. BN running moments were unchanged; logged STDP eligibility was 0 or 1 in 56 progress records, so STDP was mostly absent, not universally zero.

The predeclared complete-VALID weights/theta 2x2 diagnostic scored 187 (parent/parent), 182 (child/child), 182 (child weights/parent theta), 187 (parent weights/child theta). Both controls exactly reproduce recorded validation scores. For this checkpoint pair and VALID set, theta exchange did not recover changed weights. Every non-theta tensor belongs exactly to the declared weights source, and BN moments are identical. No TRAIN, checkpoint promotion or TEST.

Source_v44 adds immutable `hybrid_stdp_granularity: neuron|group`; neuron is the existing default. Group mode assigns every 16th whole WTA group to STDP, keeping 512/8192 STDP and 7680/8192 BP neurons, 512 groups of16 and existing WTA/BP/STDP/exp-theta update mathematics. Ordinary own phases and resume reject reassignment. Full CPU/native CUDA regression passed 505 tests in46.06sec, including real disjoint writes, protected Adam loser rows/states, dense input gradients, unchanged default RNG/state/forward and exact all-state resume. This verifies implementation, not accuracy.

A fresh source44 rebalanced group-assignment condition is preregistered at `gesture_rebalanced128_512_single8192_T16_groupSTDP_commonoffset_seed42_from_scratch_prefix/protocol.json`. Same CNN [128,256,512,576,512], hidden8192, T16, encoding/augmentation/LR/cosine160 denominator/seed42/split42 as the previous rebalanced scratch condition. Constructor confirms 30,450,064 parameters, physical width12160, spiking width10176, spatial neurons3,858,432. The only controlled config change is granularity. Actual budget20, extend40 only full20 best153 plus recent trend and declining TRAIN CE, extend80 only full40 best176 and recent5mean164. Highest complete VALID at least188/204, earliest ties, required before any fixed TEST. Existing selected benchmark results remain authoritative. Native disposable preflight precedes formal TRAIN; no TEST-based class/error/subject tuning.

Group-assignment native disposable preflight passed: all10CNN gradients positive; 2366 BP/149 STDP eligible rows changed, all2515 winner rows changed, no losing rows changed. Actual4096 selection events split3840BP/256STDP. Common-offset projection preserves predictions, relative-logit maximum error1.90735e-6 and CE difference1.19209e-7; all physical weights nonnegative, float32 params/grads, cuDNN TF32 enabled/matmul TF32 disabled. Peak11,252.35MiB. Scratch TRAIN launched natively at2026-10-05T10:05:28.200110+00:00, supervisor1370693, worker1370697; fixed stage20/stage40/fullTEST controllers1370761/1370762/1370763. Reporter1221238 verified alive. No new TEST result.

At2026-10-05T10:08:44Z, `groupSTDP_progress_20261005_100844` freezes both metric snapshots, configs/source hashes and common complete first3 epochs. Controlled config difference is only granularity/output path.21 common logged TRAIN steps have mean STDP-eligible neurons18.2381 (neuron) versus137.2857 (group), population512 each. These are eligibility counts, not actual writes nor whole-epoch coverage. First3 VALID[96,88,117] versus[77,89,119], TRAIN CE[1.95291,1.40209,1.09582] versus[2.36739,1.44506,1.15860]. Newgroup qualification continues under existing gates; no superiority or TEST improvement claim. Scientific figure visually checked, no TRAIN/TEST forwards performed by read-only audit.

At2026-10-05T10:18Z, first13 complete group-assignment epochs scored VALID[77,89,119,113,128,145,132,113,118,130,137,142,136]. The read-only metric snapshot `groupSTDP_progress_20261005_101804` compares92 identical logged globalsteps: meanSTDPeligibility137.663(group)/5.6413(neuron), zero-eligibilitysteps0/12. These are sampled potential updates, not total writes or final quality.
CPU-only immutable compact latest-TRAIN snapshots atepochs9/13 show actual weight changes in470/512STDProws, maxabsolutechange.00377378, meanabsolutechange1.61102e-5, exactunchangedSTDPbias/rowtargets, rowmassmaxerror1.90735e-6, noSTDPweightatupperbound and maxweight.384209. BPweightdigests also differ. SourceCP/config/split/helper hashes are bound; no TRAIN/VALID/TEST forwards or model selection.

A conditional backup `gesture_rebalanced128_512_single8192_T16_groupSTDP_ownKD0p25_seed42_from_scratch_prefix` is preregistered using unchanged source44/full505. Student remains fully fromscratch, same topology/T/coding/augmentation/seed42/split42. Add25%temperature2KL and75%hardCE from one frozen original rebalanced custom teacher (earliestVALID187/204 at40, CPsha392f1fb1a47f7bc42ad7c3a65e5c1c743112a2215a8e531a4193795e40b433ac, neverTESTed). Same manifest and teacher input/origin contract verified onCPU. No coefficient transfer, externalweights, teacherat inference or additionalstudentparameters. TRAIN logs separately report hardCE/totalKD; round20 lossgate useshardCE. Same20/40/80VALIDgates and188minimumfixedTESTgate. Nativefrozen-teacher/BP/STDPpreflight mandatory. Deferredstarter only runs after parentselector's authoritative noimprovedVALID/noTEST decision, holdingGPUqueueslot duringpreflight; skipsifparent qualifies. Earlier unsuccessful warmKDphases remainnegativeevidence; freshstudent with stronger singleteacher is a distinct, unproven condition.

Conditional ownKD deferred process launched natively at2026-10-05T10:21:54.514852+00:00, PID1386516, registry`deferred_launch.json`. Its actual status confirms waiting on live parentselector1370763, no newTRAIN or GPUreservation while waiting. If parent qualifies it skips; on authoritative VALID rejection it runs frozen-teacher nativepreflight in the existing GPUqueue slot and only then launches preregistered20/40/80student. Currentgroup TRAIN remainsunaltered.

Complete20 group-assignment window finished: VALID[77, 89, 119, 113, 128, 145, 132, 113, 118, 130, 137, 142, 136, 145, 131, 131, 147, 152, 157, 144], best157/204 at19, TRAINCE2.36739->.584087, recent5mean146.2 vs prior5mean138.2. All preregistered20 gates pass (best153, recenttrend, declining hardCE); noTEST. Nativecontinuous20->40preflight passed: all10CNNgradientspositive, hiddenBPgradient.243802;1407BP/172STDProwseligibleandchanged, no losingrowschanged. Actualresume launches at2026-10-05T10:24:36.233864+00:00, worker1389066, supervisor/driver1370761; source44/config/Adam/cosine160/BN/theta/RNG/loader/guard preserved, outputdirectory only. Parentinitialworker1370697 andsupervisor1370693 terminal/missing natively verified. Stage40 gate remainsbest176 andrecent5mean164. FixedTEST still requireshighestVALID188/204 andearliestties. Conditional ownKDwaiting1386516 stayspending anddoesnot occupyGPU.

Complete40 group-assignment window finished at2026-10-05T10:43:32Z: round21-40 VALID[143, 156, 156, 146, 163, 148, 152, 175, 160, 163, 166, 156, 171, 167, 165, 158, 170, 160, 160, 179], best179/204 at40; last5mean165.4. All frozen40 gates pass; nativecontinuous40->80preflight passed, all10CNNgradientspositive, hiddenBPgradient.268432;1286BP/139STDProwseligibleandchanged, no losingrowschanged. Peak11,485.55MiB. Resume launches at2026-10-05T10:43:52.289440+00:00, worker1406509 andsupervisor1370762. Model/Adam/cosine160denominator/BN/exp-theta/RNG/loader/guard retained. Source44 runtime unchanged. Actualboundedbudget80, earlyplateauguard remains15. FixedTESTrequireshighestVALID188/204 earliesttie. NoTEST; ownKDwaiting1386516 remainsdormant. Same-seed neuron-assignment predecessor reached187/204 by40; currentgroup179 shows no accuracy improvement yet despite maintainedSTDPupdates. First40completecounts/decision/checkpoint/sourcehashes archived in`gesture_rebalanced128_512_single8192_T16_groupSTDP_commonoffset_seed42_first40_screening_audit.json`.

At 2026-10-05T11:08Z, the group-assignment run has completed 66 full TRAIN972 / VALID204 rounds. Round64 improves the best complete VALID from182 to185/204; this resets the existing 15-round plateau guard. Round66 TRAIN CE is0.0433006 and VALID170. The bounded maximum remains80 and TEST admission remains188/204; no new TEST or changed condition. Native worker1406509, supervisor1370762, selector1370763, dormant ownKD controller1386516 and reporter1221238 were verified alive at11:08:02Z; reporter audit_errors is empty. Best-round50 and best-round64 models are preserved in `gesture_rebalanced128_512_single8192_T16_groupSTDP_commonoffset_seed42_qualified80_prefix/validated_model_snapshots/epoch050_VALID182` and `epoch064_VALID185`, with source checkpoint/protocol/model SHA256, exact completed validation records and all-state bitexact reload checks. These are compact model snapshots without optimizer states; original full resume checkpoints remain intact. Snapshotting performs no forwards, training or TEST and does not change checkpoint selection. Existing full TEST FP32 results remain MNIST9923/10000, N-MNIST9875/10000, CIFAR10-DVS755/1000, Gesture273/288; the fourth95% gate requires274/288 and remains unmet.

At 2026-10-05T11:11:42Z, the still-deferred ownKD dependency audit confirms all7 registered helpers and33 runtime files retain their registered hashes, protocol/config unchanged, and the frozen teacher CP retains SHA392f1fb1a47f7bc42ad7c3a65e5c1c743112a2215a8e531a4193795e40b433ac. Actual checkpoint metadata confirms custom from-scratch origin, selected round40 VALID187/204, no TEST result, same input profile/topology/seed42/data root/partition configuration. The actual source44 teacher metadata validator passes. Manifest SHAe94be76f1838d302734893b73be3913d6d5733ec73469ebf7d3dad264ddf7546 is unchanged. Proof: `groupSTDP_ownKD_deferred_integrity_20261005_1110.json`. This is a CPU read-only hash/metadata audit without any model forward or TRAIN/VALID/TEST evaluation; mandatory native training-gradient/frozen-teacher preflight is still pending until an authoritative parent VALID rejection. No registered condition or trigger was changed.

### 2026-10-05: group-assignment terminal result and scratch distillation entry repair

The group-assignment condition stopped at79 complete epochs at11:20:29Z, with15 stale epochs after best round64 VALID185/204. The selector sealed `no_improved_VALID_checkpoint_no_TEST` at11:20:33Z. Every epoch covers TRAIN972 / VALID204; final TRAIN CE0.0149233, final VALID184. The existing188/204 gate fails. Source44/config hashes remain unchanged; the selected checkpoint exactly matches the preserved epoch064_VALID185 model snapshot. Full79 curves, source/checkpoint hashes, stop and selector evidence are recorded in `gesture_rebalanced128_512_single8192_T16_groupSTDP_commonoffset_seed42_final_screening_audit.json`. This condition provides no new TEST or accuracy improvement over the selected benchmark model.

The deferred ownKD native preflight passed, but its original CLI worker1440528 failed at11:20:53Z before data/model/TRAIN creation: `Own distillation requires own initialization or continuous resume`. The disposable preflight had bypassed this entrance restriction. Original case/config/protocol/logs/registries remain intact, and `failed_before_training_audit.json` verifies zero formal training steps and all original worker/supervisor/controller handles terminated. The published launch event alone did not establish successful training.

Source45 changes only the `run_training` entrance guard: a custom `init=from_scratch` student without external initialization can use the existing frozen own-teacher bundle. Existing own-initialization/resume paths, teacher provenance and partition checks, float training, WTA/BP/STDP/exp-theta mathematics, quantization and evaluation are unchanged. New CPU/CUDA integration exercises actual scratch TRAIN, real group BP/STDP weight changes, exact model/Adam/scheduler/guard/RNG/loader-RNG resume, frozen teacher state/gradients, no own checkpoint copying and no TEST; invalid official/external/non-scratch initialization is rejected. Full native regression passes510 tests, zero failures/errors/skips,45.35sec; `source_v45_verification.json` freezes33 runtime hashes, with only `experiments/accuracy/run.py` differing from44. These are implementation checks, not benchmark accuracy.

The recovered case is `gesture_rebalanced128_512_single8192_T16_groupSTDP_scratchKD0p25_seed42_from_scratch_prefix`, registered11:26:46Z. Compared with the failed case, configuration differs only in output directory. Teacher CP392f1fb1a47f7bc42ad7c3a65e5c1c743112a2215a8e531a4193795e40b433ac,75%hardCE/25%temperature2KL, from-scratch student initialization, topology/T16/log1p/res128/pool2/augmentation/LR/seeds/split and fixed20/40/80/188VALID gates are retained. Constructor exactly matches the prior30,450,064 parameters, physical width12160 and spiking width10176. Native preflight is rerun under the GPU queue reservation; all10CNN gradients are positive,2366BP/149STDP winner rows change, no losing rows change, teacher states remain exact, common-offset projection preserves predictions, peak11,368.52MiB, float32 tensors/grads/cuDNN TF32 enabled/matmul TF32 disabled. Student remains the sole inference model.

Actual recovered TRAIN starts11:27:37Z with initialization `from_scratch`, supervisor1446984/worker1446988 and stage20/stage40/fixedTEST controllers1447052/1447053/1447054. Native handles and real first training batch are verified, rather than inferred from launch status. First batch loss and CNN gradient exactly match the configured disposable preflight. Full first epoch finishes11:28:43Z: TRAIN972 hardCE1.9051134 / totalKD2.9294556, VALID109/204. Loss components are separately recorded; continuation lossgate uses supervised hardCE. `startup_verified.json` preserves first actual TRAIN/VALID and native invocation evidence. Reporter1221238 remains live with audit_errors empty. There is no new TEST result; selected FP32 benchmark gates remain3/4, Gesture273/288 still below274/288.

### 2026-10-05: deferred continuous-checkpoint averaging

While source45 scratchKD remains active, a bounded VALID-only fallback is registered at `gesture_groupSTDP_continuous64_79_average_VALID_diagnostic/protocol.json`. It uses only the completed source44 group-assignment run's own best64 VALID185 and latest79 VALID184. Actual CPU checkpoint inspection confirms identical complete metadata/config/roles, from-scratch origin, all79 full TRAIN972/VALID204 records and an exact64-round history prefix. These are temporally aligned snapshots in one continuous run, not independently trained branches. Source checkpoint/protocol/helper/split/runtime SHA256 are bound before any new GPU VALID forward.

Three fixed peak coefficients[.5,.75,.25], each with either serialized averaged BN moments or16-batch TRAIN-only BN refit, define six candidate conditions. Both original controls must reproduce all recorded VALID score fields and leave model state unchanged. The existing convex FP32 helper checks finite states/nonnegative physical weights; actual CPU preflight validates lineage, all planned blends and endpoint numerical identity, with no model forward/TRAIN/VALID/TEST. Candidate parameter and hidden states must remain exact during BN refit; no gradients or optimizer updates are introduced. The architecture/T16/coding/role masks/width remain unchanged and inference uses one SNN. Best fullVALID with earliest fixed-order ties must reach188/204; only then reload the selected checkpoint, exactly replay fullVALID and freeze one fullTEST condition. Any future TEST result requires independent replay before an accuracy claim.

Native fallback1461066 launched11:42:21Z and was verified at11:42:39Z to be waiting on actual selector1447054. It runs only after the parent's authoritative `no_improved_VALID_checkpoint_no_TEST` decision, skips if the parent qualifies, and requires review on a failed or missing parent. It holds the existing GPU queue reservation during any later diagnostic and evaluation; while waiting it performs no GPU VALID or TEST. The original diagnostic registration and CPU preflight are archived before a prelaunch correction to the future child's registration timestamp; conditions/trigger unchanged and new helper/protocol revalidated before launch. Source45 runtime stays frozen/full510.

At11:42:39Z scratchKD has completed13 full epochs with VALID[109,99,116,124,130,134,128,129,125,147,153,138,141], best153 at11 and latest hardCE0.736150. Native worker1446988, supervisor1446984, all three selectors and reporter1221238 remain live. The complete20 recent-trend/hardCE gate is still pending; early VALID progress is not new TEST accuracy. Selected benchmark FP32 gates remain3/4.

ScratchKD completed its first20 full TRAIN972/VALID204 epochs at11:49:29Z. VALID[109,99,116,124,130,134,128,129,125,147,153,138,141,143,153,149,140,156,161,166], best166 at20. Recent5 mean154.4 versus previous5 mean145.6; first5 supervised hardCE mean1.3145461 versus last5 mean0.5576742. All fixed20 gates pass. Recorded per-epoch totalKD is numerically consistent with75%hardCE+25%scaledteacherKL; these losses are not conflated. Every20-round metric, decision, source and continuous-resume checkpoint hash is preserved in `gesture_rebalanced128_512_single8192_T16_groupSTDP_scratchKD0p25_seed42_first20_screening_audit.json`. Source45/full510 remains unchanged and no TEST is evaluated.

Native20->40 preflight passes with teacher states exactly unchanged, all10CNN gradients positive, CNNnorm8.574625/head2.913884/hiddenBP0.262639. All1353BP/182STDP eligible rows change and no losing rows change; peak11,602.59MiB. Continuous resume worker1468054 starts at11:49:45Z under supervisor1447052, actualstart_epoch20/global_step2440; model/Adam/cosine160 denominator/BN/exp-theta/RNG/loader/guard and teacher configuration are retained. Original scratch worker1446988 has terminated. Stage40 best176/recent5mean164 gates and final188VALID gate remain fixed. At11:51:45Z round21 completes VALID142, hardCE0.442103; current best remains166. Native worker/supervisor/remaining selectors/reporting processes and conditional-average waiter1461066 are confirmed live. The fallback continues waiting on final selector1447054; it does not start at the20-round transition. No new TEST result or FP32 target achievement.

### 2026-10-05: deferred convolution-arithmetic validation

At12:06:08Z native VALID-only waiter1483393 launches `diagnose_rebalanced187_no_tf32_VALID.py` (SHA3ce7f3c575819f97306af9de3c69d524ba7407ed87add92e9351f1ec915a87bc). Native identity and waiting status are verified at12:09:13Z. Registration is `gesture_rebalanced187_no_tf32_VALID_diagnostic/protocol.json`. It depends exclusively on the continuous-average waiter's authoritative no-improved-VALID terminal result; it skips an earlier VALID qualification, and failed or missing dependencies require review. While waiting, there is no GPU model forward or TEST access.

The fixed checkpoint is the original rebalanced neuron-assignment scratch round40 VALID187 teacher (SHA392f1fb1a47f7bc42ad7c3a65e5c1c743112a2215a8e531a4193795e40b433ac), which has never been TESTed. Its original source43 runtime remains frozen with492 passing checks. CPU-only preflight verifies actual scratch metadata, complete original VALID record, finite nonnegative float32 physical weights, all runtime hashes, settable backend flags and helper syntax; this does not validate GPU arithmetic or accuracy. The GPU sequence first reproduces the complete original VALID score with convolution TF32 enabled and matmul TF32 disabled, then evaluates one fixed candidate with both flags disabled. All model states, weights, BN, PLIF, thresholds, topology, coding, T16 and split remain identical; no gradients, training or BN refit are introduced. This changes the requested PyTorch convolution arithmetic flag and does not establish bit-exact IEEE arithmetic or an accuracy improvement.

Only a full204-sample VALID result of at least188 qualifies. The selected checkpoint must serialize and reload and reproduce every complete VALID score field with unchanged model states before one fixed fullTEST condition is launched. The dedicated evaluator `evaluate_registered_checkpoint_no_tf32.py` (SHA4d12db91527da947d40cbc3732a33f625c1eb9ebd8cc3abdd0e72378e9974667) binds and checks both disabled flags and records them with the result; any eventual independent TEST replay must use the same evaluator and arithmetic. Original active training/evaluation helpers are unchanged. The prelaunch serialization/reload-guard amendment and archived original protocol/preflight precede any new GPU VALID or TEST.

At12:09:16Z scratchKD completes round38 (zero-based epoch37), full TRAIN972/VALID204, hardCE0.2136721, totalKD0.3252169, VALID169/204. The round38 score is169; complete-history inspection at12:10:54Z confirms the global best is round37 VALID177, correcting the earlier recent-row-only report. Native worker1468054, supervisor1447052, stage40/final selectors1447053/1447054, continuous-average waiter1461066, new arithmetic waiter1483393 and reporter1221238 are live at12:09:13Z. The fixed40 gate remains best176 and recent5mean164; it has not yet been assessed on40 complete rounds. Reporter refresh12:09:29Z has audit_errors empty. Selected completeTEST results remain M9923/10000, N9875/10000, C755/1000, G273/288:3/4 FP32 goals, no new TEST. User priority is first all FP32 goals, then improve unsigned2/4-bit inference results.

ScratchKD round40 completes12:11:26Z: fullTRAIN972/VALID204, latestVALID171, hardCE0.2136387. First40 global best177 at37 and last5mean168 pass all fixed40 gates at12:11:34Z. `gesture_rebalanced128_512_single8192_T16_groupSTDP_scratchKD0p25_seed42_first40_screening_audit.json` records all40 per-round scores/losses, count checks, numerical75%hardCE/25%KL identity, immutable source/checkpoint/config hashes and actual native process identities. The next configuration differs only in output directory. Native40->80 preflight passes: all10CNN gradients positive, CNNnorm15.405032/head8.026713/hiddenBP0.207797;1292BP/161STDP eligible rows all change, no losing rows change, frozen teacher states exact, peak11,602.59MiB. Actual TRAIN starts12:11:46.805169Z, worker1488385/supervisor1447053, start_epoch40/global_step4880. Weights, Adam, cosine160 denominator, BN, exponential thresholds, RNG, loader and guard continue; the actual total cap remains80 with15-round plateau stopping. Previous stage40 worker1468054/supervisor1447052 are terminal. Final selector1447054 and both VALID fallback waiters remain live and waiting; no new TEST. Best177 is validation progress and does not establish the Gesture95% TEST goal.

### 2026-10-05: TRAIN-only event-count gain condition

Read-only event statistics over exactly registered TRAIN972/VALID204 complete presentations are saved in `gesture_TRAIN_VALID_event_activity_audit_20261005.json`, with split manifest hash and no TEST data or model forwards. TRAIN mass10/50/90 percentiles120580.5/309674.5/660032.4 and VALID108657.3/332380/681446.5 substantially overlap; these statistics do not prove a domain shift. Event transformations such as random dropping are documented by Tonic (https://tonic.readthedocs.io/en/develop/autoapi/tonic/transforms/index.html); the proposed experiment uses fractional count scaling rather than reproducing that event-drop operation or claiming its results. Hypothesis: mild presentation-consistent training count variation can improve activity robustness.

Isolated `source_v46` changes only `experiments/accuracy/data.py` and `config.py` relative to frozen source45. Optional TRAIN geometric augmentation settings accept finite positive ordered `event_gain_min/max` for event data; MNIST Poisson coding is rejected. Each sample draws one gain shared across time, polarity and space. For encoded log1p frames, stable x+log1p((g-1)*(-expm1(-x))) exactly represents log1p(g*expm1(x)), preceding the spatial warp; raw count coding multiplies by g. This rescales real-valued counts, not individual-event Bernoulli dropping. Zero inputs remain zero, shape/dtype/nonnegativity/order are retained and omitted/identity gain consumes no extra RNG. VALID/TEST do not call this helper. No model, WTA, homeostasis, quantizer or training loop change.

Native targeted21 checks pass7.59s. Full CPU/nativeCUDA528 checks pass123.69s with no failure/error/skip; slower runtime includes concurrent existing TRAIN GPU use. New checks cover previous-source bitexact geometry/RNG compatibility, independent sample/shared presentation scaling in raw/log coding, numerical stability up to encoded1000, invalid configuration rejection, actual event-input scratch BP/STDP weight changes and exact all-model/Adam/scheduler/monitor/RNG/loader resume, strict changed-gain resume rejection and actual evaluation bypass. This proves implementation behavior, not benchmark quality. Runtime33 hashes and test/JUnit hashes are frozen in `source_v46_verification.json`; checkpoint assets link to the immutable source45 vendor directory, and the proposed custom scratch condition loads no checkpoint.

Registered12:22:31Z case `gesture_rebalanced128_512_single8192_T16_eventgain0p75_1p25_seed42_from_scratch_prefix` differs from the original rebalanced neuron-assignment source43 seed42 condition only by output directory and TRAIN gain U[.75,1.25]. There is no own initialization, distillation or inference ensemble. CNN[128,256,512,576,512]/hidden8192, parameter count30,450,064, physical width12160/logical10176/spatial3,858,432 are remeasured with the actual source46 CPU constructor and zero shape probe, exactly matching the original. T16/res128/log1p/pool2, neuron-level hybrid1/16 assignment, competition16/presentation/margin, shared head/common-offset, Adam.001/cosine160, drop.5,16TRAIN-batch BN refit, original spatial augmentation, seeds42 and exponential baseline1/delta.05/tau20 remain. Fixed20 best153+recenttrend+decreasinghardCE;40 best176+last5mean164; cap80/plateau15; final fullVALID188 admission and earliest ties are retained.

Native conditional waiter1499937 launches12:23:19Z. Dependency is actual1483393 `diagnose_rebalanced187_no_tf32_VALID.py`; the helper and status identity, configuration, policies, source and helpers are bound before launch. Trigger only its authoritative `no_improved_VALID_checkpoint_no_TEST`, which already follows the earlier scratchKD and continuous-average negative decisions. Earlier VALID qualification skips this training; missing/failed dependencies require review. Waiting has no GPU forward/TEST access. After rejection, the actual configured native TRAIN-step preflight runs under the GPU queue reservation and requires positive CNN/head/hidden gradients, real BP/STDP winner-row changes, no loser writes, finite nonnegative float32 parameters and projection invariance before the existing bounded runner/controllers may start. Mandatory native preflight and first actual TRAIN are still pending at registration. Existing live helpers/source and selected TEST results are unchanged; independent replay remains required for any future fullTEST.

At12:25:13Z scratchKD completes52 full rounds. New global best178/204 occurs at50, resetting the existing15-round plateau guard; latest52 VALID159/hardCE0.13774. Native worker1488385/supervisor1447053/finalselector1447054 and waiters1461066/1483393/1499937/reporter1221238 are live. Deferred event-gain registration/runtime/helper/policy hashes are reverified unchanged; no actual preflight/TRAIN or TEST has started for it (`deferred_integrity_audit.json`). Current completeTEST goals remain3/4; Gesture273/288 below274. This turn makes implementation/registration progress and verifies the existing TRAIN wait; there is no new benchmark quality claim.

### 2026-10-05: matched plain-group versus scratchKD progress

`gesture_group_vs_scratchKD_matched50_VALID_audit_20261005.json` verifies matched complete20/40/50 prefixes, each TRAIN972/VALID204, same architecture/input/seeds/role masks/LR/schedule. Config differs only by output directory and TRAIN-only own_distillation; source44->45 runtime differs only at the repaired run.py entrance guard. Plain versus scratchKD bestVALID at20:157/166, at40:179/177, at50:182/178. Recent5VALID means at50:170.4/163.4, supervised hardCE means0.1276769/0.1570545. Teacher/KL total objective is not substituted for comparable hardCE. These are one-seed correlated validation observations, not proof of a causal mechanism or new TEST quality; KD's early advantage is not sustained at50. Current bounded scratchKD remains governed by its unchanged early stopping and80 cap. No new train condition, TEST selection or budget is inferred from this audit. At12:26:44Z53 full rounds are complete with current best178 at50, and native worker1488385 plus all queue controllers/reporter remain live. Previous goal turn made actual source46/checking/conditional-registration progress; this turn additionally records the matched-prefix evidence and verifies the ongoing TRAIN wait.
