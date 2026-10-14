# Low-bit SNN 运行手册

[English](README_EN.md) | 中文

用于复现 MNIST、N-MNIST、CIFAR10-DVS 和 DVS128 Gesture 的低比特 SNN 实验。
基本流程：**clone 代码 → 单独上传 checkpoint 包 → 创建 Conda 环境 → 导入权重 → 准备数据 → 运行实验**。

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

## 2. 单独上传 checkpoint

在已经持有模型的源服务器上打包。使用源服务器可用的 Python 3.10；此命令只依赖标准库：

```bash
python scripts/checkpoints.py pack --out artifacts/snn-checkpoints.tar.gz
```

将包上传到目标服务器。将 `USER`、`SERVER` 和 `/path/to/2027ISCAS_LowBitSNN` 替换为实际值：

```bash
ssh USER@SERVER 'mkdir -p /path/to/2027ISCAS_LowBitSNN/artifacts'
scp artifacts/snn-checkpoints.tar.gz USER@SERVER:/path/to/2027ISCAS_LowBitSNN/artifacts/
```

完整包包含 **12 个 checkpoint + 2 个 CIFAR 量化校准文件**，不包含数据集或环境。
模型布局如下；导入工具会按清单验证大小与 SHA256，并放到正确位置：

```text
checkpoints/
├── original/{MNIST,NMNIST,CIFAR10DVS,DVSGesture}.pth
├── compressed/{MNIST,NMNIST,CIFAR10DVS,DVSGesture}.pth
└── lut_final/
    ├── {MNIST,NMNIST,CIFAR10DVS,DVSGesture}.pth
    ├── CIFAR10DVS_TRAIN_moments.pth
    └── CIFAR10DVS_quantizer_settings.pth
```

只复现一个条件时，可只打包对应模型；CIFAR 的校准文件会一起包含：

```bash
python scripts/checkpoints.py pack --bundle lut_final --dataset NMNIST --out artifacts/nmnist-lut.tar.gz
```

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

## 4. 导入并检查模型

在目标服务器执行：

```bash
python scripts/verify.py --source-only
python scripts/checkpoints.py unpack --archive artifacts/snn-checkpoints.tar.gz
python scripts/verify.py --checkpoints
```

最后一条命令校验模型文件，并在 CPU 上严格加载全部 12 个模型，检查参数量和非负权重。
这一步不计算 TEST 精度。若只上传一个条件，改用：

```bash
python scripts/checkpoints.py unpack --archive artifacts/nmnist-lut.tar.gz
python scripts/verify.py --bundle lut_final --dataset NMNIST --checkpoints
```

也可以直接上传 `checkpoints/` 目录，保留上述布局，然后执行 `verify.py`。
文件名、大小和 SHA256 的唯一清单为 `configs/checkpoints.json`。

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
复现全部三套模型、四个数据集：

```bash
for bundle in original compressed lut_final; do
  for dataset in MNIST NMNIST CIFAR10DVS DVSGesture; do
    python scripts/replay.py --bundle "$bundle" --dataset "$dataset" --data-dir datasets --out "results/replay/${bundle}_${dataset}.json"
  done
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

## 8. 新训练与 reset 消融

训练需要完整 TRAIN/VALID/TEST。使用单独目录，避免将 TEST-only 缓存误用为完整数据：

```bash
python scripts/prepare_datasets.py --scope all --download --dataset NMNIST --data-dir datasets_train
python scripts/train.py --dataset NMNIST --data-dir datasets_train --mode warm-start-final --epochs 6 --out results/train_nmnist
```

该入口先重放压缩基准，再从 original checkpoint 生成 TRAIN-only 教师缓存，最后进行 TRAIN/VALID 训练。
`warm-start-final` 从最终权重启动新的优化器；`--mode from-scratch` 随机初始化学生，但仍使用原始教师。
这是新实验，不是原训练状态的续跑，也不保证恢复历史分数。TEST 默认推迟；VALID 选定后执行：

```bash
SNN_DATA_DIR="$PWD/datasets_train" python results/train_nmnist/run_frozen.py --out results/train_nmnist --dataset NMNIST --condition lut_native_trace --stdp-lr 0.001 --bp-rate-scale 0.1 --test-only
```

N-MNIST 两种 reset 行为的相同预算对照：

```bash
python experiments/reset_ablation/run.py --data-dir datasets_train --epochs 6 --out results/reset_nmnist
```

该命令为两种行为分别训练、完成 VALID 选择后运行完整 TEST，并生成对比图表。需上传 original、compressed 和 lut_final 权重；完整模型包满足所有入口的依赖。

## 9. 代码检查与目录说明

```bash
python -m pytest -q test experiments/power/test_activity.py
```

| 目录 / 文件 | 用途 | Git |
|---|---|---|
| `snn/`、`experiments/`、`scripts/`、`test/` | 算法、实验入口、工具和测试 | 跟踪 |
| `benchmarks/` | 各条件的冻结模型实现与必要训练参数 | 跟踪 |
| `configs/`、`environment.yml` | 配置、划分、校验清单和环境定义 | 跟踪 |
| `checkpoints/`、`artifacts/` | 单独上传的权重及压缩包 | 忽略 |
| `datasets*/` | 下载的原始数据及生成的帧 | 忽略 |
| `results/`、`outputs/`、`runs/`、`logs/` | 实验记录、缓存、图表和输出 | 忽略 |

`.gitignore` 同时忽略常见模型、数组、日志、图像、压缩包、环境和编辑器缓存。
不要用 `git add -f` 添加实验产物。固定划分和最小精度比对基准属于必要的复现输入，保留在 Git 中。
