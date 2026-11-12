# v9消融工程规范

更新时间：2026-11-12。本文面向Agent，约束当前实现与后续实验；用户命令见README，完整运行语义见RUNTIME_NOTES.md，研究假设及推进条件见根方法设计.md。不将计划功能写成已实现能力。

## 1. 目录与职责

所有新v9代码位于`experiments/ablation_v9/`，只有run_experiment.py一个执行入口；original_400k为并列原流程。新增算法前先列目录、文件职责和调用链，再写代码。

| 位置 | 职责 |
|---|---|
| configs/ | 六份P0/P1/P2 × ignore/official训练JSON，另有eval_official.json统一检测评测协议 |
| src/config.py | 读取配置，校验当前schema/变体/关键数值/并行/padding；缺省padding为ignore并写入生效配置 |
| src/data.py | 读取配对数据、随机划分、长度缓存/分桶采样器、全batch标准化与补齐 |
| src/sampling.py | 依据有效长度生成M，应用输入置零 |
| src/models.py | 骨干及双头，仅接收观测/可见mask，不读取监督 |
| src/losses.py | 重构位置、分类padding策略及明确分母；空集合返回可反传0 |
| src/parallel.py | 设备校验、DataParallel封装/解封装、并行信息 |
| src/train.py | 构造训练loader、合成smoke、训练/短程bench、日志、优化器、早停与检查点 |
| src/evaluate.py | 训练时固定验证M、区域误差及点级指标；不因完整eval改变 |
| src/benchmark_eval.py | 合成留出/官方TSB-AD-U统一检测推理、官方指标、清单及汇总 |
| src/audit.py | 文件/manifest/协议核查与参数量；记录实际padding策略 |
| src/reporting.py | 只读取eval.json生成列表，不重新训练或评价 |
| tests/ | 小规模合成行为与回归检查；run_checks.py为测试入口 |
| audits/ | 不可混作运行代码的来源、差异和验证证据；patch/eval sources被测试读取 |
| registry/、runs/ | 执行登记与独立产物，保留既有内容 |

维护归档统一放`experiments/_archive/`；历史备份`.backup_bucket_20261110/`与`.backup_spike_20261111/`已于2026-11-12经用户确认清理（共11文件），不再是可用目录；范围与证据见audits/eval_20261112/CODE_REVIEW.md和cleanup_execution.json。数据/manifest/runs旧资产链接保持可达；不再提供experiments根目录旧脚本/规范入口。

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
| 采样 | 源配置length_bucket、总batch32、bucket_multiplier50；sequential可选；验证固定顺序不分桶 |
| parallel | runtime.parallel为none/data_parallel；单卡gpu_ids=null，多卡至少两个唯一非负可见ID |

主方法比较固定相同padding策略；padding比较固定P组、初始化、数据划分/顺序、种子、标准化、M、总batch、优化器、设备和预算。历史400k有重排/随机源等差异，不能单独承担padding因果对照。

默认数据只读共用original_400k/data，不重新生成。六份源配置使用仓库相对数据路径，从仓库根目录直接读取，不另生成本机配置副本。当前split为随机95/5 seed42，尚未审计背景分组隔离。X与X^0用全batch观测X的同一均值/标准差变换，padding后不另改变统计；X^0和Y不得进入编码器。分桶只处理train_indices对应的子集局部位置，每轮set_epoch，无重复/遗漏，尾批保留；不改变split。分桶改变batch标准化坐标、SGD顺序及padding量，所有对照同步；同epoch相对历史batch16不等于同更新步数。

## 3. 并行与损失

DataParallel只切batch维，不切时间维或Transformer层；总batch不随卡数增加。完整batch标准化、padding和采样之后再分卡，预测汇总后计算CE/MSE，不能改为每卡均值等权平均。单多卡随机和浮点轨迹可能不同，各组采用相同设备协议。

ignore：先按valid及非负标签过滤，再二值化；official：先按官方阈值二值化，再过滤-1，padding因此被当正常点。开关只改变分类损失，不改变attention或重构mask。P0/P1/P2重构均仅在M有效位置计算，目标遵循配置。

验证同时保留目标classification/total和共同valid_classification/valid_total；两模式统一按valid_total早停/选best，保留配置权重、原重构项及batch均值汇总。ignore时与既有total相同。检测指标始终过滤padding，分母与模式显式记录。不同目标/分母的total不可直接作性能结论。

## 4. 执行与产物

从仓库根目录执行，激活项目环境后使用`python`和相对脚本/配置路径；用户数据准备与执行命令在本目录README；Agent迁移核查见`experiments/MIGRATION.md`。入口模式的实际行为见RUNTIME_NOTES.md：all当前为audit→smoke→train→eval，不自动report；v9当前无resume。不要直接运行src/train.py或另建未登记训练入口。

run-id必须唯一，已有目录拒绝覆盖。CLI并行与batch覆盖在快照前完成。各run共有config、command、environment、status及registry事件；audit成功后写data_manifest/audit。smoke/train/all写smoke.json；train/all写train.log、逐epoch与best检查点、metrics/train.json；bench只写metrics/bench.json；eval/all写eval.log、eval_config.json、evaluation_manifest.json、逐序列/组/域CSV及metrics/eval.json，可选scores/；report写metrics/report.json。产物按模式生成，不能承诺每个run都有日志或检查点。保存未封装完整state_dict及优化器/epoch/验证/config，单多卡可加载。run失败不伪报完成，历史快照原文不改写。

新正式训练应先完成相应接口检查，登记初始化、预算、种子、数据版本、评价/停止条件和产物位置。用户未要求时，不自动生成数据、恢复训练或耗时测评；只读审计和合成小测试明确标注范围。

## 5. 验证与评价边界

当前测试覆盖分桶完整覆盖/尾批/局部索引与bench接口、配置、输入/遮挡、官方采样一致性、padding的CE/梯度、六组合合成微型训练/验证、共同选择标准、检查点、入口快照、真实双GPU小模型/尾批次等。生产算法修改只补必要检查；不为目录或文案修改制造镜像测试。

evaluate.py保留训练验证的区域误差/复制基线与点级指标；eval/all已由benchmark_eval.py提供合成留出与官方单变量TSB-AD-U/VUS。密度分层、上下文干预、背景分组审计与多种子统计仍是后续要求。已有默认规模短程测速摘要，不能视为完整训练/验证/存盘的整轮测量；其计时边界及外推限制见RUNTIME_NOTES。不能以合成测试宣称训练提速或方法收益。

正式研究评价还需按方法设计.md报告异常区域正常参考恢复、共同检测指标、误报、局部密度/事件长度/频率分层、合法上下文干预及容量匹配。部署阈值只由训练/验证确定；官方复现型F1保留oracle阈值搜索，须独立标注，不能冒充部署指标。未获重复证据不推进方法主张。

## 6. 当前能力边界

实际入口支持audit/smoke/bench/train/eval/report/all；--max-batches仅限制bench，不限制train/all。bench会加载完整pickle并更新参数，不保存模型，不当作低成本环境测试。无resume、AMP或自动阈值拟合；runtime.mixed_precision/evaluation.threshold_source目前仅有配置记录，具体限制见RUNTIME_NOTES。训练始终从随机初始化开始，没有加载既有权重的CLI。历史审计只代表其日期版本，当前代码变化后不得沿用旧哈希或旧测试数作为本次验证。
