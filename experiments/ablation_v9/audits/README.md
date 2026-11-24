# 审计证据索引

当前目录只保留仍在使用的证据：三个回归测试依赖的源码快照、稳定版 P0/P1 的探针，以及 2026-11-21 的四组探针。其余历史审计已归并到 `experiments/_archive/audits_history_20261110_20261121.tar.gz`。

| 目录 | 内容 |
|---|---|
| `patch_20261110/sources/`、`eval_20261112/sources/`、`spike_20261111/sources_cleanup_20261114/` | 回归测试依赖的历史源码快照，原位保留，不随归档移动 |
| `stability_20261117/` | 共同优化控制、稳定版真实评测与两个只读探针，见 [说明](stability_20261117/README.md) 和 [真实评测](stability_20261117/REAL700_RESULTS.md) |
| `frozen_classifier_20261121/` | 冻结表示上的成对逐点分类探针，见 [说明](frozen_classifier_20261121/README.md) |
| `frozen_readout_20261121/` | 冻结编码器的小型读出探针，见 [说明](frozen_readout_20261121/README.md) |
| `context_ablation_20261121/` | 固定异常点与冻结探针的配对上下文消融，见 [说明](context_ablation_20261121/README.md) |
| `residual_readout_20261121/` | 残差读出探针，见 [说明](residual_readout_20261121/README.md) |

## 归档包

`experiments/_archive/audits_history_20261110_20261121.tar.gz` 含 177 个文件：

- 日期目录：`bench_20261111`、`bucket_review_20261111`、`docs_review_20261111`、`padding_20261110`、`parallel_20261110`、`eval_20261114`、`spike_20261114`、`p1_20261115`、`p1_failure_20261116`。
- 根级清理与提案：`CLEANUP_20261116.md`、`CLEANUP_20261121.md`、`RUN_CLEANUP_PROPOSAL.md`、`cleanup_20261116.json`、`cleanup_20261121.{json,py}`、`run_cleanup_execution_20261116.json`、`run_cleanup_proposal_20261116.json`、`run_cleanup_proposal_checkpoint_checks_20261116.json`。
- `stability_20261117/` 中除两份文档和两份探针外的启动、日志、结构映射与队列文件。

解包后按同名相对路径引用。日期目录里的路径、参数、状态和章节号按报告日期解释。本次清理的逐文件账本见 [cleanup_20261124.json](../../_archive/cleanup_20261124.json)，摘要见 [CLEANUP_20261124.md](../../_archive/CLEANUP_20261124.md)。

当前运行、方法和检测成绩分别见[实验记录](../../../实验记录.md)、[工程规范](../EXPERIMENT_SPEC.md)和[结果表](../RESULTS.md)。
