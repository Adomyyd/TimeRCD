# v9审计证据

> 迁移整理注（2026-11-10）：本页已统一路径表示与环境表述，实验数值和结论保留。旧清单中的文档摘要对应整理前版本；历史命令不作为当前用法，当前入口见两路线README。

这些目录用于核实来源和实现，不是训练入口：patch_20261110（官方与本地patch/遮挡）、parallel_20261110（DataParallel）、padding_20261110（padding开关）。各次REPORT及XML记录当时状态，不能当作当前进程状态或方法性能结论。

patch_20261110/sources中的官方/历史源码被tests/test_patch_parity.py按AST读取，并验证默认配置和spawn行为，保留原文。不要把这些文件当作待清理的重复训练代码。

padding_20261110/REPORT.md历史所指before/已移至`experiments/_archive/20261110_cleanup/padding_before/`，归档代码保持原哈希；文档展示内容已做迁移整理，旧文档哈希对应整理前版本，报告/diff/manifest/XML保留原位。原400k迁移备份也位于同次归档original_400k_migrations。原始结果文件保留；报告中的路径表示已更新，统一按该映射查找。

早期patch报告对远程Python执行能力的概括已更正；后续远端套件已通过。当前从仓库根目录使用项目环境的`python`运行`experiments/ablation_v9/tests/run_checks.py`。
