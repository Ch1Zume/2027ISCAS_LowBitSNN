在完成压缩任务的最终 3M–5M 模型上比较连续 STDP 与 countdown/LUT STDP。
权威压缩基线为 `results/accuracy/compression_3to5M_20261006_selected`。

`snn.wta.WTAHidden` 默认 `stdp_mode="legacy_continuous"`，其前向、连续更新的
运算次序与旧 checkpoint 键保持兼容。实际压缩源码支持的竞争组、margin
tiebreak、hybrid neuron/group 分工均已保留。旧 `snn/stdp.py` 未用于压缩模型，
本次实际训练入口修改在 `WTAHidden.apply_stdp()`。

LUT 条件配置示例（JSON/YAML 字段）：

```yaml
stdp_mode: lut
stdp_input_encoding: bernoulli_trace
stdp_event_amplitude: 1.0
stdp_coefficient_lsb: 0.001
stdp_update_rule: soft_bounds_rownorm
stdp_record_stats: true
stdp_lr: 0.001
a_plus: 0.01
a_minus: 0.012
trace_decay: 0.9
```

初版对照实验中，MNIST 最终压缩模型的首隐藏层空间输入实际为 binary，使用
`stdp_input_encoding: binary`；严格检查输入只能为 0 或 event amplitude。
其余数据集的平均池化输入有分数值。`bernoulli` 用概率 x/A 产生事件，
前向值为 A×event，期望等于 x，trace 与前向使用同一事件。要求 0≤x≤A，
超出界限时明确报错。反向对随机编码使用 identity surrogate，保持 CNN
BP 通路；这属于新的输入编码条件，需要独立于学习规则消融。
训练和评估使用受控的 Torch CPU/CUDA RNG，检查点保存训练 RNG，推理按
固定种子复现。纯 BP 层保持原输入。legacy 模式也能开启相同编码。

优化阶段新增 `stdp_input_encoding: bernoulli_trace`：前向和 BP 使用原始
连续活动，仅在 `apply_stdp` 中按 x/A 采样 learning event。这与前向也做
Bernoulli 的旧实验是两个独立条件；它保留前向统计，但学习事件是对池化
活动的概率解释，不是还原池化前的真实 spike 时序。采样采用每层独立
Torch generator，`stdp_trace_seed` 控制初始状态，不消耗输入编码/dropout
使用的全局 RNG。训练检查点另外保存各隐藏层的 `trace_rng_state()`，恢复
时调用 `restore_trace_rng(state)`；model state_dict 不增加持久键。eval 不
采样或推进学习 RNG。LUT、winner 选择、阈值稳态及浮点训练约束均保留。

每次 presentation 初始化 U=0；当前 pre event 置 31，静默减 1 到 0。
刷新先于同一步 winner lookup。每个 sample/competition group 只选第一次
winner，立即确定其学习系数，后续 pre 不再影响该次学习。实现可以在完整
前向之后提交已确定的事件；forward 中的非 winner 和之后的发放均保留。
LUT 模式拒绝 cycle-winner 条件。指数稳态阈值调节保持原实现，eval 不更新。

U=0 表示 absent 或 age≥31；其他 age=31−U。
c=0（U=0），否则 c=0.9^(31−U)；g=0.022c−0.012；L=round(g/0.001)。
实际表按 U=0..31：

```text
[-12,-11,-11,-11,-11,-11,-10,-10,-10,-10,-10,-9,-9,-9,-8,-8,
 -7,-7,-6,-6,-5,-4,-3,-3,-1,0,1,2,4,6,8,10]
```

逻辑 trace 为 5 bit，32 个 signed 5-bit 系数共 160 bit；软件分别使用 uint8
和 int8 容器，逻辑位宽不代表实测内存或硬件面积。trace 的等效指数时间常数
9.49122 step 与稳态阈值 tau=20 分开。

浮点基础 delta=stdp_lr×coefficient_lsb×L，当前为 1e−6×L。
兼容模式正系数乘 (weight_max−w)，负系数乘 w，按整个 batch 平均，
仅对实际 STDP winner 行归一化到原 row_sum_target，再非负裁剪。
旧连续模式在 first winner 后仍可通过 post trace 对之后 pre 产生 LTD；
新规则在 winner 时对旧/缺失 pre 产生 LTD，刷新会替换而非累积 trace。
两者不等价。row normalization 会耦合同一 winner 行的突触，实际最终
更新可大于基础 delta，日志分别保存 raw delta 与归一化后的实际变化。

独立 `stdp_update_rule: add_clip` 为纯加法/裁剪参考，不使用 soft bounds 和
row normalization；仅做了单元级边界验证，本轮精度实验使用兼容模式。
未实施 winner 后停止前向或严格整数学习。表项是有符号学习系数，不是
2/4-bit 权重码增量；训练前向/反向均为浮点，仅推理隐藏 Linear 量化。

审计、固定预算实验及独立重放：

```bash
TMPDIR=/tmp .venv/bin/python experiments/stdp/audit.py --out results/accuracy/stdp_lut_after_compression_NEW
TMPDIR=/tmp .venv/bin/python experiments/stdp/run.py --out results/accuracy/stdp_lut_after_compression_NEW --blocks 2 --batches-per-block 16
TMPDIR=/tmp .venv/bin/python experiments/stdp/reproduce.py --bundle results/accuracy/stdp_lut_after_compression_20261007_v1 --dataset MNIST --out /tmp/MNIST_lut_replay_NEW.json
```

使用新的输出路径。run.py 按每条件完整 VALID 精度、再 VALID loss 选择，
包含未继续训练的初始化检查点；全部选择固定后才访问完整 TEST。
`--resume` 保留完成的条件，归档未完成启动；预算和冻结模型源码必须一致。
初版 v1 是单种子、成熟压缩 checkpoint warm start、最多 32 batch 的小规模
检查；训练样本不是完整 TRAIN epoch。BP 同时更新 CNN、BP 隐藏行和分类头，
STDP 学习率保持 0.001。BP 学习率采用实际压缩恢复值的 0.1 倍，hidden BP
为其 0.5 倍、分类头为 5 倍，固定不扫描。教师为冻结的同一压缩 checkpoint，
TRAIN 上 0.5 CE + 0.5 KD、温度 2。结果不代表从头训练收敛或学习率最优。

检查记录和全 TEST 结果见实验目录的 `README.md`、`protocol.json`、
`results.json`，每条件保留 config、checkpoint、源码 SHA256、VALID 历史和
实时 loss/winner/STDP 写入/饱和统计。baseline 重放和旧产物哈希独立保存。

2026-10-07 后续优化采用 `lut_native_trace` 条件。初版前向 Bernoulli 会明显
破坏 CIFAR/Gesture 的初始化精度，因此后续保持原生前向，仅局部学习采用
独立随机事件。`--epochs N` 表示完整 TRAIN epoch；`--skip-test` 先固定完整
VALID 选择，之后 `--test-only` 才读取 TEST。初始化 checkpoint 也参与 VALID
比较；若最终 `selected_block=0`，必须报告其未包含本轮 STDP 权重更新。

`--stdp-granularity group` 把完整竞争组交给 STDP，避免少数 STDP 神经元和
已成熟 BP 神经元混在同一组时几乎从不胜利。warm start 重新分工后将
row_sum_target 设为对应行当前权重和，保留前向权重尺度。运行时断言模型
实际采用所请求的分工，`provenance.json.actual_hidden_roles` 保存真实索引。
CIFAR 仍为 48/768 个 STDP 隐藏神经元，只改变其位置与竞争关系。

`--teacher-source original_cached` 使用原大模型的 TRAIN-only logits 缓存，
检查教师 checkpoint 哈希、数据划分哈希和每 batch 样本索引/标签。大教师
不进入学生推理模型。`--from-scratch` 随机初始化相同尺寸的紧凑网络；
`--augment-spatial` 仅对 CIFAR TRAIN 做水平翻转和四像素平移，同一变换
用于所有时间步与极性，验证/测试维持原预处理。缓存教师对应未增强原图，
作为保持类别的训练目标。`--freeze-backbone`、`--freeze-bp-hidden` 与
`--freeze-bn` 分别控制骨干参数、BP 隐藏参数和 BN 统计；STDP winner
更新始终独立于 BP。所有条件维持物理非负权重和仅推理隐藏 Linear 量化。

优化控制器为 `optimize.py`、`recover_cifar.py`、`recover_gesture.py`。
它们保存预注册 policy、冻结训练入口、实时 status、各轮 VALID 与 checkpoint，
在固定候选后进行完整 FP32/uint2/uint4 TEST 及独立重放。是否从头训练、
实际学习率/epoch 上限以每条件 provenance 为准，旧 config 中的恢复预算
字段不能当作本轮预算。新的 LUT 学习率可独立调整；表的 LSB=0.001 不变。

`--steps` 显式登记时间步条件，事件数据只允许原缓存长度的整除分帧。
改变 T 的初始化精度须重新测量，不宣称与旧条件前向相同。
`--calibrate-bn` 在首次 VALID 前，仅用完整 native TRAIN 重估 BN 统计；
所有权重、dropout、稳态阈值和 STDP 处于冻结状态。BN 使用累计 batch
均值/方差，独立 shuffle generator，不推进正式训练的数据流；检查非 BN
持久状态逐项不变。校准不读取教师、VALID、TEST，不加入空间增强。
后续可用 `--freeze-bn` 固定校准统计。`recover_cifar_temporal.py` 与
`recover_cifar_bn.py` 将这些条件登记为前序失败时的后续试验。

`--spatial-student` 是 CIFAR 的显式 4,679,375 参数候选：通道
`[32,64,128,256,448]`、隐藏宽度 256、空间读出 4×4，16 个 STDP 行与
240 个 BP 行按完整竞争组分工。warm start 选取已训练的通道/组，将原
2×2 权重平均复制到对应 4×4 子块，并恢复正权重行和；改变了结构与分类
头，不宣称旧输出保持一致。卷积和 BN 保留所选已训练参数。新结构也可
通过 `--from-scratch` 使用新的 LUT 方案从头训练。

`--fit-classifier` 先冻结模型提取完整 native TRAIN 的隐藏发放率，再用
ridge 初始化和 200 步 FP32 BP 拟合分类头。目标为 TRAIN 标签与原教师
TRAIN-only 缓存的 CE/KD；不访问 VALID/TEST、不更新骨干、隐藏权重、稳态
阈值或学习 RNG。分类头通过给所有类别添加相同的每特征系数，保持
logit 差值与 softmax 不变，同时保证所有物理权重非负；分类头仍不量化。
这是单独的 BP 初始化阶段，不是 STDP 精度收益。`--minimum-initial-valid`
在拟合后完整 VALID 太差时跳过无效的整网恢复。`recover_cifar_head.py`
登记原结构/空间结构的分类头拟合、完整微调以及必要的从头训练。

`summarize_optimization.py` 汇总各轮已固定候选、完整 TEST 及重放审核，
保留失败轮次与初始化候选。每轮 checkpoint 只依据 VALID 选定，但测试
结果用于判断是否继续开发，因此整项迭代不是一次性盲测，尚需独立新种子
或最终锁定后的确认实验评估泛化稳定性。

最终汇总包为 `results/accuracy/stdp_lut_optimization_20261007_final`。
MNIST/N-MNIST/Gesture 保留已达标压缩权重；CIFAR 采用新的 4.679M 空间网络，
从头完成 40 个 TRAIN epoch，完整 VALID 选第 36 个，再做 TRAIN 分类头拟合
和微调。`--initial-checkpoint` 可以从已经训练的同结构学生继续训练，严格检查
checkpoint、结构和 private trace RNG，并登记继承的 LUT 训练；优化器重新初始化。
`recover_cifar_polish.py` 登记具体的继续训练条件。

`cifar_views.py` 仅在推理时比较原图和水平翻转，平均 logits，视图之间重置
膜电位并冻结阈值。双视图每视图 T=8，参数量不变，推理计算约为单视图两倍；
此条件必须明确报告，不能把双视图精度当作单视图精度。

`calibrate_quantization.py` 只用完整 TRAIN 的隐藏输入均值与二阶矩做推理
校准，不读取 TRAIN 标签、不更新任何训练权重或稳态阈值。权重逐输出行量化：
`scale = max(W_row) * clipping_ratio / (2**bits-1)`，零点 0，unsigned 码裁剪到
合法范围；scale 保持浮点。补偿为 `bq = b + factor*(W-Wq) @ TRAIN_mean`。
比率和补偿系数按完整 VALID 精度、再 loss 选定，固定后才进行完整 TEST。
这不是 QAT；卷积、分类头、bias 与神经元参数均保持浮点。

最终 CIFAR FP32/2-bit/4-bit 为 76.20%/74.40%/75.70%，分别独立重放一致。
其中 2-bit 用 ratio=0.5、factor=1；4-bit 按 TRAIN 输入方差加权量化误差逐行
选 ratio、factor=1。FP32 输出与校准前完全相同；原始 absmax 量化精度为
60.20%/68.80%，不能用校准后的结果替换原始量化条件的历史记录。

`finalize_optimization.py` 审核已固定候选及完整 TEST 重放，复制各数据集的
独立冻结 runtime、配置、checkpoint 和校准统计。最终 `reproduce.py` 支持四个
数据集，逐项检查 SHA256、完整样本数、三个精度模式、非负物理权重及推理
状态不变。复现命令与训练来源见最终包 README，实验过程中失败和作废的
记录均保留。所有 FP32 目标通过，CIFAR 2-bit 尚低于 75%。
