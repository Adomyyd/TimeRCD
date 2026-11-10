# 原始 TimeRCD：400k 本地复现

> 迁移整理注（2026-11-10）：本页已统一路径表示与环境表述，实验数值和结论保留。旧清单中的文档摘要对应整理前版本；历史命令不作为当前用法，当前入口见两路线README。

本目录与 `experiments/ablation_v9/` 并列。它保存基于原训练流程的 400k 本地复现，不代表官方训练数据和预算的完全复刻。

## 目录

```text
original_400k/
├── run_experiment.py       # 生成、训练、比较的统一入口
├── src/
│   ├── generate_400k.py
│   ├── train_400k.py
│   └── compare_400k.py
├── data/                   # 数据及原始 manifest，v9 只读共用
├── runs/
│   └── time_rcd_400k_20261108/
└── migrations/             # 路径迁移记录及整理前脚本
```

`src/train_400k.py` 继续调用仓库根目录 `training.py`，不复制或改写原模型与损失。编码器 patch=16、遮挡采样块=16、遮挡比例=0.15，batch size=16，学习率=5e-4。此轮目录整理只调整路径，训练数值配置不变。

## 命令

现有 400k 已在续训，不要重复启动。以后恢复同一运行时使用：

```bash
python \
  experiments/original_400k/run_experiment.py \
  --mode train
```

该命令仍使用原脚本指定的物理 GPU 0。目录整理没有增加或改变设备参数。

入口支持 `--mode generate` 和 `--mode compare`，分别调用原生成脚本与比较脚本。数据已存在时生成脚本拒绝覆盖；训练和比较不会自动串联，运行前应登记并获得相应任务授权。只查看帮助不会执行这些阶段：

```bash
python experiments/original_400k/run_experiment.py --help
```

## 旧路径兼容

`experiments/` 根目录的旧脚本、数据文件、manifest、runs 路径保留为符号链接。当前已启动进程仍持有旧路径，通过链接访问原数据和同一输出目录。旧 run 的配置快照、原 manifest 的 `data_path`、v9 历史 audit 源码保留原文，路径对应关系见 `migrations/layout_20261110.json`。

后续代码和命令使用本目录的新路径。不要删除兼容链接，直到相关运行和历史引用已另行处理。
