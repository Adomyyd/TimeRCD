# v9消融工程规范

更新时间：2026-11-10。本文面向Agent，约束当前实现与后续实验；用户命令见README，完整运行语义见RUNTIME_NOTES.md，研究假设及推进条件见根方法设计.md。不将计划功能写成已实现能力。

## 1. 目录与职责

所有新v9代码位于`experiments/ablation_v9/`，只有run_experiment.py一个执行入口；original_400k为并列原流程。新增算法前先列目录、文件职责和调用链，再写代码。

| 位置 | 职责 |
|---|---|
| configs/ | 六份P0/P1/P2 × ignore/official JSON，实验协议与数值参数 |
| src/config.py | 读取配置，校验当前schema/变体/关键数值/并行/padding；缺省padding为ignore并写入生效配置 |
| src/data.py | 读取配对数据、随机划分、全batch标准化与补齐 |
| src/sampling.py | 依据有效长度生成M，应用输入置零 |
| src/models.py | 骨干及双头，仅接收观测/可见mask，不读取监督 |
| src/losses.py | 重构位置、分类padding策略及明确分母；空集合返回可反传0 |
| src/parallel.py | 设备校验、DataParallel封装/解封装、并行信息 |
| src/train.py | 已解析协议下的训练、日志、优化器、早停与检查点 |
| src/evaluate.py | 固定验证M、当前分区域误差及点级检测指标 |
| src/audit.py | 文件/manifest/协议核查与参数量；记录实际padding策略 |
| src/reporting.py | 只读取eval.json生成列表，不重新训练或评价 |
| tests/ | 小规模合成行为与回归检查；run_checks.py为测试入口 |
| audits/ | 不可混作运行代码的来源、差异和验证证据；patch sources被测试读取 |
| registry/、runs/ | 执行登记与独立产物，保留既有内容 |

重复旧代码备份、文档原文和一次性操作统一放`experiments/_archive/`。数据/manifest/runs旧资产链接保持可达；不再提供experiments根目录旧脚本/规范入口。

## 2. 配置与单因素约束

配置至少描述experiment_id、data、split、input、model、loss、optimization、runtime、evaluation及schema_version。当前严格拒绝未知顶层字段，并检查关键字段组合；并非所有子对象都已实现未知键拒绝，不声称校验覆盖所有误拼。

| 因素 | 当前合同 |
|---|---|
| P0 | masked_x，encoder V&~M，target=x |
| P1 | full_x，encoder V，target=x |
| P2 | full_x，encoder V，target=x0 |
| patch | model.patch_size与input.mask_patch_size独立；当前默认16/16 |
| M | 有效块均匀抽样，ratio0.15，按块数取整；不按Y选择，不越过V |
| padding | loss.classification_padding只允许ignore/official；配对配置仅该项不同 |
| parallel | runtime.parallel为none/data_parallel；单卡gpu_ids=null，多卡至少两个唯一非负可见ID |

主方法比较固定相同padding策略；padding比较固定P组、初始化、数据划分/顺序、种子、标准化、M、总batch、优化器、设备和预算。历史400k有重排/随机源等差异，不能单独承担padding因果对照。

默认数据只读共用original_400k/data，不重新生成。当前split为随机95/5 seed42，尚未审计背景分组隔离。X与X^0用全batch观测X的同一均值/标准差变换，padding后不另改变统计；X^0和Y不得进入编码器。

## 3. 并行与损失

DataParallel只切batch维，不切时间维或Transformer层；总batch不随卡数增加。完整batch标准化、padding和采样之后再分卡，预测汇总后计算CE/MSE，不能改为每卡均值等权平均。单多卡随机和浮点轨迹可能不同，各组采用相同设备协议。

ignore：先按valid及非负标签过滤，再二值化；official：先按官方阈值二值化，再过滤-1，padding因此被当正常点。开关只改变分类损失，不改变attention或重构mask。P0/P1/P2重构均仅在M有效位置计算，目标遵循配置。

验证同时保留目标classification/total和共同valid_classification/valid_total；两模式统一按valid_total早停/选best，保留配置权重、原重构项及batch均值汇总。ignore时与既有total相同。检测指标始终过滤padding，分母与模式显式记录。不同目标/分母的total不可直接作性能结论。

## 4. 执行与产物

从仓库根目录执行，激活项目环境后使用`python`和相对脚本/配置路径；用户配置准备命令在本目录README；Agent迁移核查见`experiments/MIGRATION.md`。入口模式的实际行为见RUNTIME_NOTES.md：all当前为audit→smoke→train→eval，不自动report；v9当前无resume。不要直接运行src/train.py或另建未登记训练入口。

run-id必须唯一，已有目录拒绝覆盖。CLI并行覆盖在快照前完成；run保存实际config、command、environment、data_manifest、audit、status、smoke、日志、检查点、metrics和registry事件。保存未封装完整state_dict及优化器/epoch/验证/config，单多卡可加载。run失败不伪报完成，历史快照原文不改写。

新正式训练应先完成相应接口检查，登记初始化、预算、种子、数据版本、评价/停止条件和产物位置。用户未要求时，不自动生成数据、恢复训练或耗时测评；只读审计和合成小测试明确标注范围。

## 5. 验证与评价边界

当前测试覆盖配置、输入/遮挡、官方采样一致性、padding的CE/梯度、六组合合成微型训练/验证、共同选择标准、检查点、入口快照、真实双GPU小模型/尾批次等。生产算法修改只补必要检查；不为目录或文案修改制造镜像测试。

当前evaluate仅提供现有随机验证侧的区域重构/复制基线与点级分类指标；VUS、外部测试适配、密度分层、上下文干预、背景分组审计与多种子统计是后续要求。默认规模双卡峰值/吞吐尚未测，不能以合成测试宣称训练提速或方法收益。

正式研究评价还需按方法设计.md报告异常区域正常参考恢复、共同检测指标、误报、局部密度/事件长度/频率分层、合法上下文干预及容量匹配。阈值只由训练/验证确定，未获重复证据不推进方法主张。
