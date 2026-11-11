# 实验维护归档

> 迁移整理注（2026-11-10）：本页已统一路径表示与环境表述，实验数值和结论保留。旧清单中的文档摘要对应整理前版本；历史命令不作为当前用法，当前入口见两路线README。

本目录只保存整理前原文、旧代码备份、路径迁移记录与一次性操作，不作为当前实验入口。当前工作流分别位于../original_400k与../ablation_v9，使用各自run_experiment.py。

- 20261110_cleanup/：2026-11-10二次目录/文档整理；before保留整理前文档与两份审计相关代码；original_400k_migrations保留首次迁移记录；padding_before保留padding修改前备份；one_off_scripts保留PID限定的旧停止脚本；operations保存本次有边界保护的维护脚本。

不得从归档脚本自动重启、杀进程、生成数据或执行训练。数据/权重/run不在此次归档范围，原路径保留。详情见该次README、plan/result/verification与REPORT。

迁移补注（2026-11-10）：归档中的文档展示内容也已统一为仓库相对路径，去除机器绑定表述；数值与结论保留。旧清单的文档摘要描述当时的原始版本，不适用于后续整理后的展示内容。未修改归档代码、JSON、diff和XML原始证据。当前运行方法以两路线README和`experiments/MIGRATION.md`为准。

> 当前性说明（2026-11-11）：本文是指定日期的历史记录，不随当前实现改写。旧入口、配置生成步骤、batch16/固定顺序及当时的run保留说明不作当前事实；当前使用两路线README，代码差异见ablation_v9/IMPLEMENTATION_DIFF.md（仓库位置：experiments/ablation_v9/IMPLEMENTATION_DIFF.md），最新资产存在性与执行状态见根实验记录第12节。
