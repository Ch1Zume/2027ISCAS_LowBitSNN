# Low-bit SNN 运行手册

[English](README_EN.md) | 中文

用于复现 MNIST、N-MNIST、CIFAR10-DVS 和 DVS128 Gesture 的最终版低比特 SNN 实验，仅保留 `lut_final`。
基本流程：**clone 代码 → 本地下载模型并上传 → 创建 Conda 环境 → 准备数据 → 运行实验**。

Git 只保存代码、运行配置、固定数据划分、模型文件清单和精度比对基准。
数据集、checkpoint、日志、逐样本记录、图表、实验结果和环境全部忽略；不需要 Git LFS。
`benchmarks/` 中的冻结源码定义了历史模型的实际行为，是复现依赖。

## 1. 下载代码

在目标 Linux x86-64 服务器上准备 Git、Conda 和可用的 NVIDIA GPU 驱动，然后执行：

```bash
git clone --depth 1 https://github.com/Ch1Zume/2027ISCAS_LowBitSNN.git
cd 2027ISCAS_LowBitSNN
nvidia-smi
```

后续命令均在仓库根目录运行。完整 TEST 和训练需要 CUDA；CPU 可用于文件验证和模型加载检查。

## 2. 从本地上传 checkpoint

模型文件单独下载到本地，再通过 SCP/SFTP 上传到目标服务器。**GitHub 不提供或同步模型文件**。
以 `/path/to/2027ISCAS_LowBitSNN` 为目标服务器上的仓库根目录，必须使用以下路径：

| 模型组 / 文件 | 目标服务器路径（相对仓库根目录） |
|---|---|
| lut_final，四个数据集各一个 | `checkpoints/lut_final/<dataset>.pth` |
| CIFAR TRAIN 统计校准文件 | `checkpoints/lut_final/CIFAR10DVS_TRAIN_moments.pth` |
| CIFAR 量化配置文件 | `checkpoints/lut_final/CIFAR10DVS_quantizer_settings.pth` |

`<dataset>` 严格使用 `MNIST`、`NMNIST`、`CIFAR10DVS`、`DVSGesture`，区分大小写。
下载的 checkpoint 如果叫 `best.pth`，请按所属数据集重命名后上传，例如 N-MNIST 的最终 LUT 模型对应 `checkpoints/lut_final/NMNIST.pth`。
下载的 CIFAR `TRAIN_moments.pth` 和 `quantizer_settings.pth` 分别按上表添加 `CIFAR10DVS_` 前缀。
最终版需要 **4 个 checkpoint + 2 个 CIFAR 校准文件**；只重放一个条件时，只上传该条件的模型及校准文件即可。

从持有最终文件的源服务器下载到本地（将 SSH 地址和源仓库路径替换为实际值）：

```bash
mkdir -p checkpoints
scp -r USER@SOURCE_SERVER:/path/to/source-repo/checkpoints/lut_final checkpoints/
```

下载后本地 `checkpoints/lut_final/` 应包含：

```text
MNIST.pth
NMNIST.pth
CIFAR10DVS.pth
DVSGesture.pth
CIFAR10DVS_TRAIN_moments.pth
CIFAR10DVS_quantizer_settings.pth
```

在目标服务器的仓库根目录创建目录：

```bash
mkdir -p checkpoints/lut_final
```

在本地终端上传。这里假设本地模型已按上述结构整理到 `/local/models/`；将路径及 `USER`、`SERVER` 替换为实际值：

```bash
scp /local/models/lut_final/*.pth USER@SERVER:/path/to/2027ISCAS_LowBitSNN/checkpoints/lut_final/
```

单独上传一个模型的例子：

```bash
scp /local/downloads/NMNIST/best.pth USER@SERVER:/path/to/2027ISCAS_LowBitSNN/checkpoints/lut_final/NMNIST.pth
```

Windows 本地可用 PowerShell 的 `scp` 或 SFTP 客户端，目标服务器路径与上表相同。
模型文件名、大小及 SHA256 在 `configs/checkpoints.json` 中定义。`checkpoints/` 已被 Git 忽略。

## 3. 创建 Conda 环境

```bash
CONDARC="$PWD/configs/condarc.yml" conda env create -f environment.yml
conda activate lowbitsnn
python -m pip check
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
```

`environment.yml` 固定 Python 3.10.20、PyTorch 2.7.1/CUDA 12.6、torchvision 0.22.1、SpikingJelly 0.0.0.0.14 等依赖。
项目级 `configs/condarc.yml` 将 defaults 别名指向 conda-forge；`CONDARC` 仅作用于本次调用，不修改全局设置。
PyTorch wheel 自带 CUDA 用户态依赖；服务器仍需支持该运行时的 NVIDIA 驱动。
版本组合参见 [PyTorch 官方安装记录](https://pytorch.org/get-started/previous-versions/#v271)。

## 4. 检查已上传的模型

完成上传并激活环境后，在目标服务器执行：

```bash
python scripts/verify.py --source-only
python scripts/verify.py --checkpoints
```

最后一条命令校验模型文件，并在 CPU 上严格加载全部 4 个最终模型，检查参数量和非负权重。这一步不计算 TEST 精度。
若只上传一个条件，使用：

```bash
python scripts/verify.py --bundle lut_final --dataset NMNIST --checkpoints
```

## 5. 准备 TEST 数据

只做精度重放和活动统计时，不需要恢复完整 TRAIN 数据：

```bash
for dataset in MNIST NMNIST CIFAR10DVS DVSGesture; do
  python scripts/prepare_datasets.py --scope test --dataset "$dataset" --data-dir datasets
done
```

此命令会下载所需公开归档并生成固定 TEST 缓存。原始归档可能包含额外样本，但只转换固定 TEST 选择。
下载地址和归档校验值在 `configs/dataset_sources.json`；固定划分和文件顺序在
`configs/dataset_splits/`、`configs/event_file_order.json`。
事件数据保持原始 16 个 event-count 帧（`split_by=number`），模型再按配置使用对应的步数。

需要自定义数据位置时，对准备、重放和活动统计命令使用相同的 `--data-dir /path/to/data`。
下载中断可以重跑。网络不可达时，将清单里的归档手工放入 `<data-dir>/<dataset>/download/`；MNIST 使用 torchvision 的 `MNIST/raw/` 布局。

## 6. 运行精度重放

先运行主基准 N-MNIST：

```bash
python scripts/replay.py --bundle lut_final --dataset NMNIST --data-dir datasets --out results/replay/lut_final_NMNIST.json
```

每次调用都会计算该模型的 **FP32、uint2、uint4 完整 TEST**，检查推理前后模型状态不变，并与保存的基准比较。
复现最终版的四个数据集：

```bash
for dataset in MNIST NMNIST CIFAR10DVS DVSGesture; do
  python scripts/replay.py --dataset "$dataset" --data-dir datasets --out "results/replay/lut_final_${dataset}.json"
done
```

输出必须使用新文件名。默认 `--comparison exact` 要求完整分数字典一致；更换 GPU 或依赖后，若只出现浮点指标差异，可显式使用 `--comparison counts`：仍要求每种精度的正确数、样本数完全一致，同时记录全部数值差异。失败也会保存输出。

| 最终 LUT 基准 | T / view | Views | FP32 | uint2 | uint4 |
|---|---:|---:|---:|---:|---:|
| MNIST | 4 | 1 | 99.26% | 99.29% | 99.29% |
| N-MNIST | 4 | 1 | 98.96% | 98.84% | 98.85% |
| CIFAR10-DVS | 8 | 2 | 76.20% | 74.40% | 75.70% |
| DVS128 Gesture | 16 | 1 | 95.49% | 95.14% | 95.14% |

量化仅作用于隐藏全连接突触权重。CIFAR 使用两个视图及冻结的量化校准；MNIST/N-MNIST/Gesture 的最终 LUT 权重保留已选压缩模型。表格是单 seed 的参考结果，不代表新训练必然得到同一分数。

## 7. 运行活动统计与电路输入导出

先完成第 5 步的四个 TEST 数据集准备，并上传完整 LUT 模型组：

```bash
python experiments/power/run_activity.py --data-dir datasets --out results/activity --wait-data-seconds 0
python experiments/power/report_activity.py --out results/activity
python experiments/power/export_stimuli.py --out results/activity --precision uint4
python experiments/power/export_stimuli.py --out results/activity --precision uint2
```

输出包含逐层/逐样本/逐时间步活动、统计图和 N-MNIST hidden2 的权重码与输入序列。
仅跑主基准可加 `--datasets NMNIST`；汇总报告命令需要完整的四数据集结果。
统计口径见 [活动说明](experiments/power/USAGE.txt)：这些是软件活动与 SOP，不是硅片功耗或实际 RST 次数。
