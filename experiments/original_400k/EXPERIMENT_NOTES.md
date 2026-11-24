# 原流程 400k 实验记录

原流程用本地 400k 单变量数据复现训练，语料和预算以本路线配置为准。v9 共用数据，使用自己的入口与模型快照。命令在 [README](README.md)。

| 文件 | 职责 |
|---|---|
| `run_experiment.py` | 统一 `generate`、`train`、`compare` |
| `src/generate_400k.py` | 生成数据和 manifest，已有文件时拒绝覆盖 |
| `src/train_400k.py` | 调仓库根 `training.py`，用既有检查点恢复原 run |
| `src/compare_400k.py` | 比较本地 best 和官方权重，评测 TSB-AD-U 的 11 组，逐序列错误写入结果，汇总使用成功行 |

保留原分类 padding 行为。历史配置为 batch 16、编码器 patch 16、遮挡块 16、比例 0.15、seed72、FP32、AdamW 学习率 `5e-4`、weight decay `1e-5`、最多 50 轮、patience 7、workers 0、每 100 批打印、保存间隔 5。入口固定使用物理 GPU 0。续训恢复模型和优化器，历史完整 RNG 和早停状态不可恢复。

首跑中断后从 epoch 10 恢复，于 2026-11-10 停于 epoch 12/batch 2100。epoch 11 平均总 loss 155.5050、重构 155.4366、分类 0.0685、验证总 loss 0.0948。日志间隔未捕捉具体异常批次，根因未定。按保存逻辑，latest 为 epoch 11，best 为本次续训的 epoch 10。见[停止报告](audits/stop_20261110/REPORT.md)。

`data/`、`runs/time_rcd_400k_20261108/` 分别保留数据和运行资产。旧 `migrations/` 及一次性停止脚本已归档，原 `20261110_cleanup/` 目录已打包为 `experiments/_archive/20261110_cleanup.tar.gz`，见[归档说明](../_archive/README.md)。本路线 `compare` 与 v9 合成加真实的统一评测是两个入口，详见 [v9 运行说明](../ablation_v9/RUNTIME_NOTES.md)。
