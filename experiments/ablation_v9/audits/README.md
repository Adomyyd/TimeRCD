# v9审计证据

> 迁移整理注（2026-11-10）：本页已统一路径表示与环境表述，实验数值和结论保留。旧清单中的文档摘要对应整理前版本；历史命令不作为当前用法，当前入口见两路线README。

这些目录用于核实来源和实现，不是训练入口：patch_20261110（官方与本地patch/遮挡）、parallel_20261110（DataParallel）、padding_20261110（padding开关）。各次REPORT及XML记录当时状态，不能当作当前进程状态或方法性能结论。

patch_20261110/sources中的官方/历史源码被tests/test_patch_parity.py按AST读取，并验证默认配置和spawn行为，保留原文。不要把这些文件当作待清理的重复训练代码。

padding_20261110/REPORT.md历史所指before/已移至`experiments/_archive/20261110_cleanup/padding_before/`，归档代码保持原哈希；文档展示内容已做迁移整理，旧文档哈希对应整理前版本，报告/diff/manifest/XML保留原位。原400k迁移备份也位于同次归档original_400k_migrations。原始结果文件保留；报告中的路径表示已更新，统一按该映射查找。

早期patch报告对远程Python执行能力的概括已更正；后续远端套件已通过。当前从仓库根目录使用项目环境的`python`运行`experiments/ablation_v9/tests/run_checks.py`。

## 2026-11-11索引补充

- bucket_review_20261111/REPORT.md：分桶覆盖、配对与bench计时问题审查；当时的文档待办已由后续文档整理处理，算法问题仍未修复。
- bench_20261111/bench_summary.json：他次短程测速摘要，原四个run目录已移除；不视为完整轮次测量或通用设备结论。
- docs_review_20261111/REPORT.md：当前代码与全文档核对、失效资产及能力边界、文档检查结果。
- spike_20261111/REPORT.md：三次loss尖峰（400k epoch11、旧P0 epoch3、当前P0 official epoch1）的日志定位、标准化量级上界核验，以及与官方/论文公式的三处差异；不构成根因结论，逐batch诊断当时未实现，现已实现可选模块；原报告中MSE上界表述已在实验记录第26节纠正。

旧patch/parallel/padding报告里的batch16、三配置、固定顺序及“未测默认规模”均是当时事实；当前为六配置、源batch32、length_bucket，可选batch覆盖，执行状态见根实验记录。旧P0目录本次未找到，审计不保证其检查点仍可用。

## 2026-11-12索引补充

- eval_20261112/REPORT.md：合成留出与官方单变量TSB-AD-U统一检测评测；固定官方d458549源码、数据清单/推理/16指标数值核对、CPU回归及实际checkpoint兼容检查。sources被test_benchmark_eval.py读取，保留；不把此小测试当成全量检测成绩。

- eval_20261112/CODE_REVIEW.md：v9全部活动代码、隐藏备份、审计来源与文档核查；记录精确范围、保留理由及cleanup_proposal.diff。用户确认后已清理两隐藏备份共11文件和三处无用代码；执行及CPU回归证据见同目录cleanup_execution.json、cleanup_cpu_tests.txt。

## 2026-11-14索引补充

- eval_20261114/REPORT.md：两个P0训练完成核对、初始全量及后续范围变更；旧execution.json为已停止full队列。当前固定合成200/真实700协议见SUBSET200_PROTOCOL.md、执行见execution_subset200.json；small_process_metric_timing.json记录同分数重算的耗时及数值一致性。控制台日志与run产物保留，不以启动记录代替最终检测成绩。

- eval_20261114/performance/REPORT.md：用户要求停止后的官方调用链对比、完整数据常驻的fork开销定位、独立CPU指标工作进程修复、同分数16指标回归及CPU/GPU小型验证；两个正式队列均已停止，未恢复。

- eval_20261114重新执行记录：旧aborted评测输出已清除，固定200合成/700真实P0队列的restart2已完成，成绩见RESULTS.md和实验记录第22节；run-id、范围与状态见restart_cleanup.json、restart_eval_queue.json及对应run目录。先前REPORT和performance证据保留作历史诊断，不代表新结果。

- restart_nan_failure.json记录官方允许的NaN与旧入口拒绝逻辑差异；attempt1的不完整输出已删除。NaN兼容修正经14项benchmark测试后，restart2的ignore/official队列已重新启动。

- 最新完成状态：P0 restart2固定200合成/700真实评测均已完成，对照表见[../RESULTS.md](../RESULTS.md)，根实验记录第22节；前述队列启动/停止描述均为历史阶段。
- [spike_20261111/CHECKPOINT_RETENTION_20261114.md](spike_20261111/CHECKPOINT_RETENTION_20261114.md)：用户授权清理两个P0训练权重，删除46文件约19.06GiB，各保留best/last及两份尖峰前驱；载荷、保留摘要及删除账本已核验。该目录的sources_cleanup_20261114是复现分析证据，不是训练入口；旧checkpoint无RNG状态，不能保证精确重放。

- [spike_20261114/REPORT.md](spike_20261114/REPORT.md)：可拆卸逐batch诊断、CPU/双卡数值及RNG对照、pre-step现场重放和epoch0门槛。用户授权从初始化重跑P0 official两轮，最新状态见execution.json与run；早期spike报告中的“未实现诊断”为当时事实，现已实现可选模块，不代表根因已查明。

## 2026-11-15：新P1

p1_20261115保存旧P1停止、退役配置清单、新P1代码和文档差异、合成/双卡验证与正式启动证据。历史run保持原语义；只有P0与新P1是当前执行方案。
