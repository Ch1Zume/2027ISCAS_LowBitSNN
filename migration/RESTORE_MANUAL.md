# SNNCIM 完整项目整理与恢复手册（2026-10-13）

## 本次版本与保留范围

仓库：https://github.com/Ch1Zume/2027ISCAS_LowBitSNN ，使用默认 `main` 分支。
上次最终代码只在 `codex/migration-20261011`，默认分支没有同步；本次将最终版本
更新到默认分支，并纳入 2026-10-12 的全部正式脉冲活动结果。

服务器项目：`/home/zhangyufan/Project/1_2026Q3_T22_SNNCIM/2_Ruiqi/Algorithm`。
本次保留全部现有数据集、原始归档、未完成下载分片、frame 缓存、Python 缓存、
`.venv`、其外部 Conda 基础环境，以及三套最终模型/冻结 runtime。只清理过期
`pilot`、`validation_v1`、外围中间日志和上次交付包。清单见
`reorganize_20261013/cleanup_audit.json`。历史校验清单保留于 `history/20261011/`。

| 保存位置 | 内容 |
|---|---|
| GitHub main | 代码、配置、冻结 runtime、最终表格、全部活动结果/序列/图表、十个小于 100 MiB 的模型/校准文件、手册与审计 |
| SNN-large-models-20261013.tar.gz | 四个超过普通 GitHub 单文件限制的 original checkpoint |
| SNN-data-数据集名-20261013.tar.gz | 对应数据集全部现存文件，包括缓存和未完成下载 |
| SNN-project-caches-20261013.tar.gz | 项目 Python/pytest 缓存 |
| SNN-runtime-base-20261013.tar.gz | 当前 `.venv` 所依赖的 Linux Conda 基础环境，使用 conda-pack 制作 |
| SNN-runtime-overlay-20261013.tar.gz | 当前 `.venv` 的附加包与原始环境文件 |
| SNN-project-20261013.zip / SNN-git-20261013.bundle | GitHub 同一提交的方便下载副本 / 离线 Git 副本 |

交付目录为服务器 `migration/releases/20261013/`。`DOWNLOAD_INDEX.json` 记录
Git 提交、每个包及每个 512 MiB 分片的 SHA256；`PAYLOAD_FILES.json` 记录项目、
模型、数据和缓存的逐文件 SHA256。运行环境先按归档校验，再在 Linux 验证。

## 下载到 E:\IMECAS_SNN

在你的 Windows PowerShell 中执行以下三行。使用已配置的 SSH 别名：

```powershell
New-Item -ItemType Directory -Force -Path 'E:\IMECAS_SNN' | Out-Null
scp 'IMECAS-yufan:/home/zhangyufan/Project/1_2026Q3_T22_SNNCIM/2_Ruiqi/Algorithm/migration/releases/20261013/download.ps1' 'E:\IMECAS_SNN\download-20261013.ps1'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'E:\IMECAS_SNN\download-20261013.ps1'
```

没有 SSH 别名时，将第一次 `scp` 的 `IMECAS-yufan` 换成
`zhangyufan@100.102.159.102`，并在最后一行加
`-Server 'zhangyufan@100.102.159.102'`。需要电脑已能访问这台服务器。

结果位置：

- 项目：`E:\IMECAS_SNN\20261013\Algorithm`
- 下载包：`E:\IMECAS_SNN\20261013\archives`
- 接收回执：`E:\IMECAS_SNN\20261013\WINDOWS_RECEIPT.json`

使用日期目录避免覆盖旧项目。脚本每次刷新下载索引，跳过已验证的包/分片；
中断后重新执行同一个脚本即可。完整包通过校验后才删除冗余传输分片。
脚本解压代码、模型、数据和项目缓存，并逐文件校验。Linux 环境包在 Windows
保留为归档，避免符号链接和权限丢失。成功回执也会尝试传回服务器。

服务器上的打包/模拟接收验证不代表 Windows 已收到。只有上述 Windows 回执
证明电脑端实际完成下载。该回执不代表电脑已执行 CUDA 或完整 TEST。

## 在新 Linux 主机或 WSL2 恢复

离线环境包适用于兼容的 **Linux x86_64**，不能直接作为 Windows 原生 Python
使用。GPU 驱动不在包中；完整 TEST 需要可用的 NVIDIA GPU/CUDA 驱动。
WSL2 建议把运行目录放到 Linux 文件系统，Windows E 盘作为交付包保存位置。

可以直接使用解压后的 `Algorithm`，或者从 GitHub 获取：

```bash
git clone --branch main git@github.com:Ch1Zume/2027ISCAS_LowBitSNN.git Algorithm
```

完全离线时：

```bash
git clone --branch main /path/to/archives/SNN-git-20261013.bundle Algorithm
```

假设新项目为 `/srv/projects/Algorithm`，把六个模型/数据/缓存包解压到父目录
`/srv/projects`（这些包内部带有 `Algorithm/` 前缀）。已有 Windows 脚本解压结果
可直接复制，无需重复解压。不要把 runtime 包当成这些 payload 包解压。

```bash
python3 Algorithm/migration/verify_delivery.py --project Algorithm --manifest /path/to/archives/PAYLOAD_FILES.json
python3 Algorithm/migration/restore_environment.py --project Algorithm --archives /path/to/archives --require-cuda
```

先校验、后运行，因为 Python/pytest 缓存会在执行后正常改变。环境恢复脚本会
解压基础环境到 `Algorithm/.runtime/base`，执行 conda-unpack，再恢复 `.venv`
附加包并重建启动路径。它拒绝覆盖已有 `.venv` 或 `.runtime/base`，因此不会
损坏源服务器正在使用的环境。必须在最终运行路径执行恢复，完成后不要直接
移动环境；需要移动时从归档重新恢复。

恢复结果保存于 `migration/ENVIRONMENT_RESTORE_RECEIPT.json`，含实际包版本、
加载路径及 CUDA 可用性。以项目 Python 执行后续操作：

```bash
cd /srv/projects/Algorithm
TMPDIR=/tmp .venv/bin/python migration/verify.py --checkpoints --out /tmp/snn_checkpoint_verify_NEW.json
```

源环境版本为 Python 3.10.20、torch 2.7.1+cu126、torchvision 0.22.1+cpu、
SpikingJelly 0.0.0.0.14。完整版本及包位置见 `reorganize_20261013/environment.json`。
只复制旧 `.venv` 不够，因为它依赖外部 Conda；本交付同时保存了两部分。

## 立即重放完整 TEST 与活动统计

当前保留数据可支持 MNIST/N-MNIST/CIFAR10-DVS/DVSGesture 全部 TEST。事件数据
使用冻结样本顺序和 event-count frames，按事件数量分帧并做 log1p 压缩。
保留的完整 TEST 数量分别为 10000、10000、1000、288。

```bash
TMPDIR=/tmp .venv/bin/python experiments/power/run_activity.py --out results/new_runs/activity_replay_NEW
TMPDIR=/tmp MPLCONFIGDIR=/tmp/snn-mpl .venv/bin/python experiments/power/report_activity.py --out results/new_runs/activity_replay_NEW
TMPDIR=/tmp .venv/bin/python experiments/power/export_stimuli.py --out results/new_runs/activity_replay_NEW --precision uint4
TMPDIR=/tmp .venv/bin/python experiments/power/export_stimuli.py --out results/new_runs/activity_replay_NEW --precision uint2
```

一次重放四个数据集、三种精度，使用冻结模型且不训练/重新校准。新输出必须放
新目录。完整分数字典、计数及状态对照见各条件 `summary.json`；不同 GPU/软件
版本可能导致 loss 浮点差异，应分别报告正确数一致和完全精确重放。

现有正式输出在 `results/new_runs/spike_activity_20261012/final/`；画图、审计和
使用导出的刺激不需要重跑模型。该目录的冻结源码快照和历史校验不改写。

WTA 口径：冻结模型只让 winner 获得学习资格，保留全部阈值发放；其 16.1092%
输入活动率与 10.7842% 发放率对应 N-MNIST uint4 的 512×512 hidden2。
它们不是硬 WTA 的单 winner 输出，也不是实测复位或功耗。逻辑时间步到芯片
周期的映射仍未确认；本次不报告 MHz、实际 RST、pJ/SOP 或面积。

## 新训练与完整数据准备

“保留现有数据”不等于恢复上次已删除的全部训练数据。目前 MNIST 有训练原始
文件，三个事件数据集主要是本次恢复的完整 TEST；下载分片也全部保留。进行
TRAIN/VALID 或旧通用数据加载路径的评测前，在独立目录准备完整数据。原适配器
可能把已有的 TEST-only frame 目录当作完整缓存，因此不要对当前 `datasets`
直接运行完整数据准备，也不要删除现存缓存来绕过它：

```bash
TMPDIR=/tmp .venv/bin/python migration/prepare_datasets.py --dataset NMNIST --data-dir ./datasets_full --download
```

按需把数据集名替换为 MNIST、CIFAR10DVS 或 DVSGesture。准备脚本使用保存的
官方资源和冻结划分，不更改最终模型。完成数据准备后，可运行：

```bash
TMPDIR=/tmp .venv/bin/python migration/train.py --dataset CIFAR10DVS --mode warm-start-final --epochs 6 --data-dir ./datasets_full --out results/new_runs/CIFAR_restart_NEW
```

这是从最终权重开始的新训练，会创建新优化器；不代表原训练无缝续跑，也不承诺
复现相同单种子分数。TRAIN-only 缓存按需生成，新 VALID 选定后再做新的 TEST。
`migration/replay.py` 保留原有完整数据适配器入口；现有 TEST-only 数据的直接
重放请使用上面的 `experiments/power/run_activity.py`。

## 校验边界

`reorganize_20261013/` 保存清理、环境快照、归档成员清单及恢复验证证据。
`migration/payload_manifest.json` 验证核心项目与最终模型；交付目录中的
`PAYLOAD_FILES.json` 额外覆盖数据与项目缓存。历史 checkpoint/runtime 和正式
活动证据保持原字节。Windows 接收、环境可运行和完整 TEST 是三个不同验收项，
必须分别查看对应证据。
