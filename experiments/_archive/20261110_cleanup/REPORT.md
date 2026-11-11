# experiments代码与文档整理报告

> 迁移整理注（2026-11-10）：本页已统一路径表示与环境表述，实验数值和结论保留。旧清单中的文档摘要对应整理前版本；历史命令不作为当前用法，当前入口见两路线README。

完成日期：2026-11-10；下列路径均相对于仓库根目录。

## 结果

保留original_400k与ablation_v9两条并列实验路线，各使用自身run_experiment.py统一入口。六份P0/P1/P2 × ignore/official配置及训练、损失、模型、数据与评价算法均未修改。未启动正式训练、恢复训练、生成数据或执行耗时评价。

## 目录清理

| 操作 | 原位置（相对experiments） | 当前归档位置（相对本目录） |
|---|---|---|
| 归档7个旧文件 | original_400k/migrations/ | original_400k_migrations/ |
| 归档12个修改前备份 | ablation_v9/audits/padding_20261110/before/ | padding_before/ |
| 归档一次性停止脚本 | original_400k/audits/stop_20261110/stop_and_inspect.py | one_off_scripts/stop_and_inspect.py |

实际删除：experiments根目录train_400k.py、generate_400k.py、compare_400k.py、EXPERIMENT_SPEC.md四个符号链接；四个可重建缓存目录，共21个文件、105791字节。链接指向的正式源码保留。

保留数据、manifest、runs三个资产兼容链接，历史run快照仍可引用它们。保留ablation_v9/audits/patch_20261110/sources/，其为test_patch_parity.py的回归依赖。既有run、registry、数据、权重、日志及审计报告未移动或改写。

## 文档与代码

整理14份文档：根目录AGENTS.md、研究问题和思路.md、方法设计.md、实验记录.md和README.md；experiments总览、两路线README、v9规范/实现差异、两路线审计索引及两份归档说明。职责分开，明确历史中断状态、padding独立对照轴、patch16/遮挡16、统一入口、GPU用法和当前实现边界。VUS、密度分层、上下文干预与v9恢复训练等未实现事项列为后续工作，all模式不再被写成会自动生成报告。

唯一运行代码改动：ablation_v9/src/audit.py的padding审计元数据随ignore/official配置变化，修正official模式被错误标为过滤padding；在test_parallel_contract.py补相应断言。训练与损失计算不变。

## 验证

- 远端合成回归：75 passed in 8.11s，无跳过，含真实双GPU小模型检查；不加载400k数据。见pytest_remote.log、pytest_remote.xml和validation.json。
- 两路线入口--help均返回0，见entry_help.json。
- 修改后16个文件已逐一下载回读，SHA256与本地部署源全部一致；见source_manifest.json与readback_verification.json。
- 28项受保护资产身份与小文件摘要均未变化。大型数据和权重只核对inode、字节数、mtime，未完整哈希或复制。
- 原30项生产代码/配置/测试清单中仅audit.py与test_parallel_contract.py变化；两入口及六份配置保持原摘要。
- 20个归档文件逐一校验字节摘要一致；四个旧入口链接消失，三个资产链接仍有效；未发现活跃代码引用已移除路径。
- 检查时未发现相关正式训练进程；本次没有持续监控。

## 追溯

before/保存修改前12个既有文档/代码文件；changes.diff为16项部署文件的完整差异，source_manifest.json记录修改前后SHA256。plan.json、result.json、removed_caches.json、verification.json保存目录操作与保护校验。operations/下为本次维护脚本，不是训练入口。原迁移记录和旧文档仅作历史证据，不应当作当前执行规范。

> 当前性说明（2026-11-11）：本文是指定日期的历史记录，不随当前实现改写。旧入口、配置生成步骤、batch16/固定顺序及当时的run保留说明不作当前事实；当前使用两路线README，代码差异见ablation_v9/IMPLEMENTATION_DIFF.md（仓库位置：experiments/ablation_v9/IMPLEMENTATION_DIFF.md），最新资产存在性与执行状态见根实验记录第12节。
