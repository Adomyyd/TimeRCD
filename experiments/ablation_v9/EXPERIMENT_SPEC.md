# v9消融工程规范

更新时间：2026-11-15。本文面向Agent，约束当前实现与后续实验；用户命令见README，完整运行语义见RUNTIME_NOTES.md，研究假设及推进条件见根方法设计.md。不将计划功能写成已实现能力。

## 1. 目录与职责

所有新v9代码位于`experiments/ablation_v9/`，只有run_experiment.py一个执行入口；original_400k为并列原流程。新增算法前先列目录、文件职责和调用链，再写代码。

| 位置 | 职责 |
|---|---|
| configs/ | 四份历史训练配置、两份official_stable配置及eval_official.json |
| src/config.py | schema/数值校验及启动数据协议审计 |
| src/data.py | 数据读取、划分、标准化、分桶及遮挡采样 |
| src/models.py | 模型、设备选择与DataParallel封装 |
| src/losses.py | 训练目标、padding策略与损失分母 |
| src/train.py | 训练/smoke/bench、OptimizationControl、轮末恢复及权重保留 |
| src/diagnostics.py | BatchDiagnostics与DetailedObserver；逐批/分频统计与事件现场 |
| src/evaluate.py | 训练期间固定验证 |
| src/benchmark_eval.py | 完整检测评测与已有结果汇总 |
| src/metric_worker.py | 独立指标子进程，避免继承大训练数据/GPU状态 |
| tests/ | 12份回归模块；统一run_checks.py入口 |
| audits/ | 历史数值回归依赖及执行证据 |
| runs/ | 每次实验配置、日志、权重、指标与诊断 |
| registry/ | 运行登记 |

维护归档统一放`experiments/_archive/`；历史备份`.backup_bucket_20261110/`与`.backup_spike_20261111/`已于2026-11-12经用户确认清理（共11文件），不再是可用目录；范围与证据见audits/eval_20261112/CODE_REVIEW.md和cleanup_execution.json。数据/manifest/runs旧资产链接保持可达；不再提供experiments根目录旧脚本/规范入口。

## 2. 配置与单因素约束

配置至少描述experiment_id、data、split、input、model、loss、optimization、runtime、evaluation及schema_version。新增可选diagnostics诊断对象；当前严格拒绝其他未知顶层字段，并检查关键字段组合；并非所有子对象都已实现未知键拒绝，不声称校验覆盖所有误拼。

| 因素 | 当前合同 |
|---|---|
| P0 | masked_x，encoder V&~M，target=x |
| P1 | masked_x，encoder V&~M，观测target=x，并以独立头预测x0 |
| patch | model.patch_size与input.mask_patch_size独立；当前默认16/16 |
| M | 有效块均匀抽样，ratio0.15，按块数取整；不按Y选择，不越过V |
| padding | loss.classification_padding只允许ignore/official；配对配置仅该项不同 |
| 采样 | 源配置length_bucket、总batch32、bucket_multiplier50；sequential可选；验证固定顺序不分桶 |
| parallel | runtime.parallel为none/data_parallel；单卡gpu_ids=null，多卡至少两个唯一非负可见ID |

主方法比较固定相同padding策略；padding比较固定P组、初始化、数据划分/顺序、种子、标准化、M、总batch、优化器、设备和预算。历史400k有重排/随机源等差异，不能单独承担padding因果对照。

默认数据只读共用original_400k/data，不重新生成。全部源配置使用仓库相对数据路径，从仓库根目录直接读取，不另生成本机配置副本。当前split为随机95/5 seed42，尚未审计背景分组隔离。X与X^0用全batch观测X的同一均值/标准差变换，padding后不另改变统计；X^0和Y不得进入编码器。分桶只处理train_indices对应的子集局部位置，每轮set_epoch，无重复/遗漏，尾批保留；不改变split。分桶改变batch标准化坐标、SGD顺序及padding量，所有对照同步；同epoch相对历史batch16不等于同更新步数。

## 3. 并行与损失

DataParallel只切batch维，不切时间维或Transformer层；总batch不随卡数增加。完整batch标准化、padding和采样之后再分卡，预测汇总后计算CE/MSE，不能改为每卡均值等权平均。单多卡随机和浮点轨迹可能不同，各组采用相同设备协议。

ignore：先按valid及非负标签过滤，再二值化；official：先按官方阈值二值化，再过滤-1，padding因此被当正常点。开关只改变分类损失，不改变attention或重构mask。P0/P1重构均仅在M有效位置计算，P0/P1保留观测X目标；P1在相同集合另计算X^0正常参考损失，三项权重默认均1。

验证同时保留目标classification/total和共同valid_classification/valid_total；两模式统一按valid_total早停/选best，保留配置权重及batch均值汇总；P1额外含正常参考项，跨组总loss不直接比较。ignore时与既有total相同。检测指标始终过滤padding，分母与模式显式记录。不同目标/分母的total不可直接作性能结论。

## 4. 执行与产物

从仓库根目录执行，激活项目环境后使用`python`和相对脚本/配置路径；用户数据准备与执行命令在本目录README；Agent迁移核查见`experiments/MIGRATION.md`。入口模式的实际行为见RUNTIME_NOTES.md：all当前为audit→smoke→train→eval，不自动report；v9支持train轮末--resume-from。不要直接运行src/train.py或另建未登记训练入口。

run-id必须唯一，已有目录拒绝覆盖。CLI并行与batch覆盖在快照前完成。各run共有config、command、environment、status及registry事件；audit成功后写data_manifest/audit。smoke/train/all写smoke.json；train/all写train.log、逐epoch与best检查点、metrics/train.json；bench只写metrics/bench.json；eval/all写eval.log、eval_config.json、evaluation_manifest.json、逐序列/组/域CSV及metrics/eval.json，可选scores/；report写metrics/report.json。产物按模式生成，不能承诺每个run都有日志或检查点。保存未封装完整state_dict及优化器/epoch/验证/config，单多卡可加载。run失败不伪报完成，历史快照原文不改写。

新正式训练应先完成相应接口检查，登记初始化、预算、种子、数据版本、评价/停止条件和产物位置。用户未要求时，不自动生成数据、恢复训练或耗时测评；只读审计和合成小测试明确标注范围。

## 5. 验证与评价边界

当前测试覆盖分桶完整覆盖/尾批/局部索引与bench接口、配置、输入/遮挡、官方采样一致性、padding的CE/梯度、四组合合成微型训练/验证、共同选择标准、检查点、入口快照、真实双GPU小模型/尾批次等。生产算法修改只补必要检查；不为目录或文案修改制造镜像测试。

evaluate.py保留训练验证的区域误差/复制基线与点级指标；eval/all已由benchmark_eval.py提供合成留出与官方单变量TSB-AD-U/VUS。密度分层、上下文干预、背景分组审计与多种子统计仍是后续要求。已有默认规模短程测速摘要，不能视为完整训练/验证/存盘的整轮测量；其计时边界及外推限制见RUNTIME_NOTES。不能以合成测试宣称训练提速或方法收益。

正式研究评价还需按方法设计.md报告异常区域正常参考恢复、共同检测指标、误报、局部密度/事件长度/频率分层、合法上下文干预及容量匹配。部署阈值只由训练/验证确定；官方复现型F1保留oracle阈值搜索，须独立标注，不能冒充部署指标。未获重复证据不推进方法主张。

## 6. 当前能力边界

实际入口支持audit/smoke/bench/train/eval/report/all；--max-batches仅限制bench，不限制train/all。bench会加载完整pickle并更新参数，不保存模型，不当作低成本环境测试。无batch中途resume、AMP或自动阈值拟合；runtime.mixed_precision/evaluation.threshold_source目前仅有配置记录，具体限制见RUNTIME_NOTES。新训练从随机初始化开始；--resume-from只支持完整训练状态恢复，不是任意预训练权重初始化。历史审计只代表其日期版本，当前代码变化后不得沿用旧哈希或旧测试数作为本次验证。

2026-11-14性能修复：指标执行边界见RUNTIME_NOTES及audits/eval_20261114/performance。初始慢队列曾停止，修复后restart2配对评测已完成，见实验记录第22节；不自动另启评测。原数值协议不变；单体pickle仍全量加载。


## 7. 可选训练诊断（2026-11-14）

仅train通过--diagnostics启用；统一入口支持--epochs/--log-freq快照前覆盖，不修改四份源配置。普通训练不实例化诊断。src/train.py中所有调用均在OPTIONAL DIAGNOSTICS块内，删除全部块仍保留原训练循环与早停；src/diagnostics.py不包含训练循环或优化器更新。

每batch诊断输出独立JSONL，控制台仍由log_freq决定。保存首次达到阈值或非有限事件的更新前模型/AdamW、前向前完整RNG、batch/mask；不逐batch存权重。有限事件继续，非有限事件保存后失败停止。可选epoch0参考逐张量核对模型和AdamW，默认记录差异继续；开启--diagnostic-stop-on-mismatch时不匹配不得进入epoch1。旧checkpoint缺RNG，不把匹配门槛等同历史事故必然重现。实现/回归与此次两轮重跑证据见audits/spike_20261114/；轮末resume后续已实现，见第8节。

诊断归约统计在CPU执行，避免日志额外创建CUDA归约临时张量；默认模型规模的双卡数值/RNG一致性也有受控合成检查。首次正式试跑在周期loss出现差异后已停止，第二次_cpu试跑也已停止；最终为_r3新run，默认记录epoch0差异继续；不从受控测试外推历史整轮必然一致。


## 8. 轮末续跑（2026-11-14）

用户明确授权延长P0诊断。统一入口新增train专用--resume-from和--checkpoint-keep-last，恢复到新run，不原地改源run或尝试给正在运行的range改上限。总epochs默认配置/CLI指定，恢复必须小于总上限；数据/split/model/input/loss/optimizer/sampler/runtime严格一致，仅epochs/log_freq/diagnostics可变。

新P1普通检查点也含training_state；以下为P0诊断时期实现记录，用户现已暂缓该诊断。src/train.py恢复模型、AdamW、完整RNG及best/stale；setup/smoke之后、训练迭代前恢复RNG，采样按绝对epoch。诊断或续跑检查点写training_state；旧r3按同轮RNG及完整日志和best兼容恢复。缺完整RNG、配置冲突或已满足早停均拒绝。best复制到新run维持历史选择；已完成epoch1后从2继续，上限50仍允许原patience7早停。

可选保留策略只作用新run：最近N>=2轮、best、首次事件前驱；每次删除记账，首次现场独立保留。普通训练默认逐轮全留。6项恢复/配置/保留检查通过（含CPU和双卡）；全套CPU98 passed、10 GPU项跳过。并发正式训练时整套GPU测试有1项因显存不足失败，107通过；不是数值回归成功，不重复争抢生产GPU。详情见audits/spike_20261114/resume_verification.json。

## 9. 新P1与历史隔离（2026-11-15）

新P1标识p1_mask_x_normal，新增正常头在共享参数初始化后构造并保存/恢复CPU RNG；共享参数初值匹配P0，不承诺后续Dropout轨迹一致。删除旧full-input分支及四份旧P1/P2配置，旧变体schema拒绝。普通P1轮末默认含完整RNG/best/stale；P0仍按诊断/续跑条件保存。新P1训练由本轮指令授权，旧P1已停止且资产保留。新测试、差异与清理清单见audits/p1_20261115。


显存执行说明（2026-11-15）：最长10000点/总batch32的双卡合成检查发现两个完整重构头保留激活会OOM。新P1训练模式对两个重构MLP启用non-reentrant激活检查点，preserve_rng_state=True，反向时重算头部；骨干仍只前向一次，不改变M/目标/权重/batch/FP32，P0执行路径不变。验证/检测不启用重算。CPU检查覆盖输出/梯度/Dropout RNG与普通头计算一致；最长序列完整前后向及AdamW更新已通过，峰值约8.86/8.51GB（十进制）。重算增加计算开销，不将其当作数值等价提速。

## 共同优化控制（2026-11-17）

新增src/train.py和两份official_stable配置，由同一run_experiment.py执行。保留四份历史配置；不改模型/损失/初始化/采样。控制项、诊断字段、固定horizon与恢复规则见RUNTIME_NOTES。tests/test_optimization.py覆盖裁剪方向/非有限拒绝、关闭路径、调度端点、固定计划恢复，以及P0/P1 CPU与双卡完整小数据训练/诊断/恢复。新控制逻辑独立于可移除诊断块。当前授权为两组从头配对训练及原验证，非完整检测评测。
