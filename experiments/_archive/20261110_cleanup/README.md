# 2026-11-10目录与文档整理

> 迁移整理注（2026-11-10）：本页已统一路径表示与环境表述，实验数值和结论保留。旧清单中的文档摘要对应整理前版本；历史命令不作为当前用法，当前入口见两路线README。

用户要求继续整理experiments代码、删除不需要的内容并同步协作文档。此次不改变训练算法或研究路线，不启动训练/生成/耗时评测。

## 路径映射

| 原位置（相对experiments） | 现位置（相对此目录） |
|---|---|
| original_400k/migrations/ | original_400k_migrations/ |
| ablation_v9/audits/padding_20261110/before/ | padding_before/ |
| original_400k/audits/stop_20261110/stop_and_inspect.py | one_off_scripts/stop_and_inspect.py |
| 根协作文档、README及相关审计代码整理前原文 | before/（保留仓库相对目录） |

从活动目录移除四个旧入口链接：experiments/{train_400k.py,generate_400k.py,compare_400k.py,EXPERIMENT_SPEC.md}。它们的真实目标保留，当前入口不依赖这些链接。link原目标记录在plan.json，便于核查/恢复。

保留三个历史资产链接：experiments/runs、dataset_400000...pkl、同名manifest。保留所有data、runs、registry、官方权重及必要审计，patch审计sources仍被回归测试读取。

## 证据

- plan.json：变更前28个资产文件的身份/小文件哈希、30个活动代码/配置/测试哈希、移动清单、链接目标及引用检查。
- before/：文档原文与audit.py/test_parallel_contract.py原文。实验记录原文包含过去逐次计划与完成登记，已保留但不作为当前状态。
- result.json、removed_caches.json：实际归档/删除操作。
- verification.json：资产保护、活动代码变化白名单、归档逐字节一致性、兼容链接与旧路径引用核查。
- changes.diff、source_manifest.json：本次文档与两份代码的差异/哈希。
- pytest_remote.xml、entry_help.json：远端合成回归及两入口帮助检查。
- REPORT.md：本次完成事实与限制。

audit.py的唯一行为修正是审计元数据随padding模式变化，相关现有入口测试增加断言。模型、损失、数据、采样、并行、训练、评价、六配置及两个正式入口保持原内容。缓存可重建，不作为研究证据保留。大型数据/权重没有复制或重算全文件哈希，使用inode/大小/mtime核对未改写。

迁移补注（2026-11-10）：归档中的文档展示内容也已统一为仓库相对路径，去除机器绑定表述；数值与结论保留。旧清单的文档摘要描述当时的原始版本，不适用于后续整理后的展示内容。未修改归档代码、JSON、diff和XML原始证据。当前运行方法以两路线README和`experiments/MIGRATION.md`为准。

> 当前性说明（2026-11-11）：本文是指定日期的历史记录，不随当前实现改写。旧入口、配置生成步骤、batch16/固定顺序及当时的run保留说明不作当前事实；当前使用两路线README，代码差异见ablation_v9/IMPLEMENTATION_DIFF.md（仓库位置：experiments/ablation_v9/IMPLEMENTATION_DIFF.md），最新资产存在性与执行状态见根实验记录第12节。
