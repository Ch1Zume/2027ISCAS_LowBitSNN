# 迁移断点

迁移范围仅为本 Algorithm 项目。2026-10-11 在真实服务器进程命名空间中按
工作目录和命令参数核查，没有发现本项目训练、评测或候选调度进程；因此没有
需要发送终止信号的目标，也没有新增的运行中训练断点。原始快照见
`process_snapshot.json`，打包前复核见迁移交付目录的 `delivery_status.json`。

最终可恢复状态共有三套、十二个数据集条件，见 `catalog.json`：

1. original：原始自建网络四数据集完整 TEST 基准，保留最终大模型，可用于教师。
2. compressed：选定的 3M–5M 压缩基准；CIFAR FP32 73.70%，未达到 75%。
3. lut_final：四项 FP32 目标均通过；CIFAR uint2 74.40%，此门槛仍未达到。

每个条件保留实际 checkpoint、冻结源码、原配置、完整 TEST 分数、数据划分
SHA256、推理随机种子和既有独立重放证据。CIFAR 校准统计和量化设置也保留。
最终 checkpoint 中存在的优化器、全局 RNG、loader RNG 和局部 trace RNG
不被剥离；所有二进制保持原 SHA256。

迁移不保留失败候选、逐批日志、临时恢复队列或其最后状态。没有把过时 PID/
queue 文件当作仍在运行的任务。原始 VGG/ResNet 比较未完成，不计为最终结论。

后续可以重放最终结果，或从最终权重开始新的训练。`migration/train.py` 的
warm-start 使用新的优化器，属于新训练，不能表述为未中断的原训练续跑。
原驱动的 `--resume` 会保留已完成条件并重启未完成条件，也不是任意 batch
边界的严格续跑。当前迁移没有运行中 batch/epoch 可严格续接。
