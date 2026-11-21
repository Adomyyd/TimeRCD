# 审计证据索引

| 目录 | 内容 |
|---|---|
| `patch_20261110`、`parallel_20261110`、`padding_20261110` | 遮挡、并行与分类 padding 对齐 |
| `bucket_review_20261111`、`bench_20261111`、`docs_review_20261111` | 分桶、测速与旧文档核查 |
| `eval_20261112`、`eval_20261114` | 官方协议来源、实现巡检、初评、NaN 和性能修复 |
| `spike_20261111`、`spike_20261114` | P0 尖峰、诊断和检查点保留 |
| `p1_20261115`、`p1_failure_20261116` | P1 改版、续训、失稳分析、诊断提速和官方对齐 |
| `stability_20261117` | 共同优化控制、模块归并、重启、[稳定版真实评测](stability_20261117/REAL700_RESULTS.md)及 [run 清理记录](stability_20261117/README.md#运行目录清理) |

回归依赖 `patch_20261110/sources/`、`eval_20261112/sources/`、`spike_20261111/sources_cleanup_20261114/`，分别由 patch、benchmark eval 和 diagnostics 测试使用。迁移时随代码保留。

日期目录记录当时情况。临时文件清理、日志无损压缩和替代证据见 [CLEANUP_20261116.md](CLEANUP_20261116.md)。三份诊断 run 和原 P1 25 份中间权重的删除已执行，见[账本](run_cleanup_execution_20261116.json)。当前运行、方法和检测成绩分别见[实验记录](../../../实验记录.md)、[工程规范](../EXPERIMENT_SPEC.md)和[结果表](../RESULTS.md)。
