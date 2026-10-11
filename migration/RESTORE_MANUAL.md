# SNNCIM 迁移与复原手册

## 1. 交付内容

本次只迁移 `/home/zhangyufan/Project/1_2026Q3_T22_SNNCIM/2_Ruiqi/Algorithm`。
代码仓库为 `Ch1Zume/2027ISCAS_LowBitSNN`，迁移分支为 `codex/migration-20261011`。

| 文件 | 用途 |
|---|---|
| SNN-code-20261011.zip | 最终代码、配置、表格、审计、冻结 runtime 和本手册 |
| SNN-models-20261011.zip | 十二个最终 checkpoint 及 CIFAR 校准统计/量化设置 |
| SNN-git-20261011.bundle | Git 仓库离线副本，可离线克隆迁移分支及其历史 |
| SHA256SUMS | 以上交付文件的 SHA256 |
| download.ps1 | 从 IMECAS-yufan 下载到 Windows 并解压、校验 |

两个 ZIP 均以 `Algorithm/` 为顶层目录，解压到同一父目录即可合并。模型包不
包含数据集。不要修改冻结 checkpoint、config 和 runtime；未来输出使用新的
`results/new_runs/` 目录。最终成绩和条件见 `FINAL_RESULTS.csv` 与 `catalog.json`。

## 2. 下载到 Windows

电脑端目标为 `E:\IMECAS Project`。在 Windows PowerShell 中执行：

```powershell
New-Item -ItemType Directory -Force -Path 'E:\IMECAS Project' | Out-Null
scp 'IMECAS-yufan:/home/zhangyufan/Project/1_2026Q3_T22_SNNCIM/2_Ruiqi/Algorithm/migration/downloads/download.ps1' 'E:\IMECAS Project\download.ps1'
& 'E:\IMECAS Project\download.ps1'
```

如果当前 PowerShell 执行策略拒绝本地脚本，可仅针对此次进程运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'E:\IMECAS Project\download.ps1'
```

SSH 别名若未配置，使用实际地址：

```powershell
& 'E:\IMECAS Project\download.ps1' -Server 'zhangyufan@100.102.159.102'
```

脚本使用已有 OpenSSH 身份验证，不保存密码或私钥。已存在且哈希正确的 ZIP
会跳过下载；中断后重新执行即可。全部包与解压后每个保留文件都校验通过后，
生成 `E:\IMECAS Project\Algorithm\migration\WINDOWS_RECEIPT.json`。

服务器会保留最终交付包。只有电脑端生成成功回执并确认备份后，才适合删除
服务器上的最终副本。当前会话无法直接写入 Windows 的 E: 盘，下载脚本须在
电脑上运行；服务器生成 ZIP 不等于电脑端已收到。

## 3. 放到新 Linux 服务器

推荐目录 `/srv/projects/Algorithm`，也可以用用户有写权限的其它目录。

联网恢复：

```bash
cd /srv/projects
git clone --branch codex/migration-20261011 git@github.com:Ch1Zume/2027ISCAS_LowBitSNN.git Algorithm
unzip /path/to/SNN-models-20261011.zip -d /srv/projects
cd Algorithm
```

无需 GitHub 的离线恢复：

```bash
cd /srv/projects
git clone --branch codex/migration-20261011 /path/to/SNN-git-20261011.bundle Algorithm
unzip /path/to/SNN-models-20261011.zip -d /srv/projects
cd Algorithm
```

也可直接解压两个 ZIP 到同一父目录。这种方式可运行项目，但不包含 `.git`。
迁移上传的提交号以交付目录 `delivery_status.json` 为准。解压前可使用
`sha256sum -c SHA256SUMS` 检查包文件。

## 4. 重建环境

实测原环境：Python 3.10.20、torch 2.7.1+cu126、torchvision 0.22.1+cpu、
SpikingJelly 0.0.0.0.14；GPU 为 RTX 4090，驱动 565.57.01。完整版本快照见
`environment_snapshot.json`。旧 `.venv` 指向旧服务器 Conda，不可直接搬走。

在新服务器安装 Python 3.10 后，在项目根目录创建环境：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu --no-deps
python -m pip install -r requirements-accuracy.txt
python -m pip install pytest==8.4.2
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python migration/verify.py --checkpoints --out /tmp/snn_restore_checkpoint_check.json
```

主环境需要 CUDA 可用才能做完整 TEST；CPU 检查仅验证导入、结构和权重加载，
不构成精度重放。不同 GPU、驱动或库版本可能改变浮点 loss。默认严格重放会
保存差异并停止；不能将这种情况报为精确复现。

## 5. 重建数据集

原数据集、下载归档、解压副本和 frame 缓存已按要求排除。保留的
`dataset_sources.json` 包含实际使用过的资源地址及 MD5；`dataset_splits/`
保存划分索引，`event_file_order.json` 保存事件样本文件名顺序，不含样本数据。

```bash
python migration/prepare_datasets.py --dataset MNIST --data-dir ./datasets --download
python migration/prepare_datasets.py --dataset NMNIST --data-dir ./datasets --download
python migration/prepare_datasets.py --dataset CIFAR10DVS --data-dir ./datasets --download
python migration/prepare_datasets.py --dataset DVSGesture --data-dir ./datasets --download
```

资源地址是迁移时保留的历史来源，不保证未来持续可下载。如果作者网站需要
人工下载，按 `dataset_sources.json` 的名称放到 `datasets/<数据集>/download/`，
再执行相同命令并去掉 `--download`。脚本先检查归档 MD5，再用同一版本适配器
重建事件数据、按事件数量切成原生 16 帧。运行时相邻合并成所需 T，并按原配置
做 log1p/分辨率调整。MNIST 使用原 Poisson 图像编码；三个事件数据集为
event-count frames, split by event number, with log1p compression。

数据重建后必须匹配样本名称、类别顺序和划分 SHA256。CIFAR TEST 为 1000，
MNIST/NMNIST 为 10000，Gesture 为 288。数据路径可放到单独大盘并通过
`--data-dir /path/to/datasets` 指定。恢复源归档后临时空间需求可能超过原先
约 49 GiB 的存量，需为下载、解压与 frame 生成留空间。

## 6. 重放最终完整 TEST

```bash
python migration/replay.py --bundle lut_final --dataset MNIST --data-dir ./datasets --out /tmp/replay_MNIST_NEW.json
python migration/replay.py --bundle lut_final --dataset NMNIST --data-dir ./datasets --out /tmp/replay_NMNIST_NEW.json
python migration/replay.py --bundle lut_final --dataset CIFAR10DVS --data-dir ./datasets --out /tmp/replay_CIFAR10DVS_NEW.json
python migration/replay.py --bundle lut_final --dataset DVSGesture --data-dir ./datasets --out /tmp/replay_DVSGesture_NEW.json
```

一次一个数据集，三个精度模式全部评测。若要检查原大模型或压缩基线，将
`--bundle` 改为 `original` 或 `compressed`。脚本验证全部迁移保留文件哈希、
冻结源码来源、严格 state_dict 加载、物理参数量、非负物理权重、数据划分、
完整 TEST 样本数及推理状态不变。CIFAR LUT 使用两个独立重置的视图与原校准
统计；不会重新校准或重新训练。

默认要求整个分数字典精确相同。若新 GPU 只导致 loss 漂移，可另建输出路径并
显式加 `--comparison counts`；此模式仍要求三种精度的正确数和样本数全部相同，
会记录 `exact_full_test_replay: false`，不能当作精确复现。新训练的基线验证若只存在此类 loss 漂移，可显式使用 `migration/train.py --baseline-comparison counts`。

## 7. 从最终模型开始新训练

先完成数据重建和 CUDA 检查，再运行：

```bash
python migration/train.py --dataset CIFAR10DVS --mode warm-start-final --epochs 6 --data-dir ./datasets --out results/new_runs/CIFAR_restart_NEW
```

这会检查完整性、重放压缩基线、重新生成 TRAIN-only 原教师缓存，再从最终
CIFAR 模型权重开始新的 TRAIN/VALID，使用新优化器。新 run 保存启动参数、
学习 RNG 和选择记录；完整 VALID 选定后 TEST 保持 deferred。其它数据集只需
替换 `--dataset`。教师缓存是可重建中间文件，不需要从旧服务器迁移。

从头训练新的 CIFAR 空间模型：

```bash
python migration/train.py --dataset CIFAR10DVS --mode from-scratch --epochs 40 --data-dir ./datasets --out results/new_runs/CIFAR_scratch_NEW
```

这是相同方案的新训练，不能承诺单种子历史分数完全重现。既有 40 个从头训练
epoch 与后续 6 个微调 epoch 分开登记；MNIST/NMNIST/Gesture 的最终 LUT 优化
选中 epoch=0，不宣称由新增 LUT 学习提升。发布权重用于重放；新训练结果需在
VALID 冻结后重新完成全 TEST，不能复用历史表格。

完成新 run 的 VALID 冻结后，可以用其冻结驱动评测新候选：

```bash
SNN_PROJECT_ROOT="$PWD" SNN_DATA_DIR="$PWD/datasets" SNN_TEACHER_CACHE_DIR="$PWD/results/new_runs/CIFAR_restart_NEW/teacher_cache" python results/new_runs/CIFAR_restart_NEW/run_frozen.py --out results/new_runs/CIFAR_restart_NEW --dataset CIFAR10DVS --condition lut_native_trace --test-only --spatial-student
```

这个新 run 的 TEST 使用训练驱动原始单视图/absmax 量化设置，未包含发布 LUT
最终版的双视图或 PTQ 校准；要比较发布条件，应另行执行保留的
`experiments/stdp/cifar_views.py` 与 `calibrate_quantization.py`，固定 VALID
选定设置后再完整 TEST。运行中断的重启语义见 `BREAKPOINT.md`。

## 8. 迁移验收边界

服务器完成的校验只说明交付包完整、最终模型可以加载及代码回归通过。
历史完整 TEST 及精确重放证据原样保留；本次迁移不会将 CPU 检查当成新精度结果。
Windows 收到并通过 SHA256、在新服务器重建数据且完成完整 TEST 重放，是后续
两项验收。不要在电脑端收到之前删除服务器最终包。
