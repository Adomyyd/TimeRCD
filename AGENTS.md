# Time-RCD 项目协作规范

适用于本仓库；用户明确要求优先。文档中的路径以仓库根目录为基准，不绑定机器名称、用户名、挂载目录或GPU型号。README面向用户和其他使用者；除官方文档外，其他文档面向Agent，用于项目长期记忆。

## 工作入口与执行

- 开始工作先读本文件、`研究问题和思路.md`、`方法设计.md`、`实验记录.md`；实际运行状态以日志和进程为准，旧记录不代表仍在运行。
- 文档命令从仓库根目录执行，先激活项目环境，再使用`python`与相对脚本/参数路径。代理实际执行遵循当前会话指定的解释器与连接约束；此类本机设置不写入共享文档。
- 新实验代码写在`experiments/`；写代码前先说明目录、文件职责及统一入口。
- 当前v9源配置为总batch32、length_bucket、bucket_multiplier50；配对组必须统一采样器、总batch、种子和并行方式。分桶会改变batch标准化与padding比例，不是数值等价优化；验证仍固定顺序。单卡可通过--batch-size覆盖，但对照组须一致。
- v9直接使用configs中的四份历史P0/P1源配置及两份official_stable配置，数据字段采用仓库相对路径；不为迁移另生成本机配置副本。每次运行由入口保存实际配置快照。
- 原流程使用`experiments/original_400k/run_experiment.py`；P0/P1使用`experiments/ablation_v9/run_experiment.py`。不另建重复训练入口。
- 未获当前任务授权，不启动或恢复正式训练、生成数据、耗时评测，不替换官方权重。历史运行及计划不会自动授权重启。合成接口检查与正式训练必须区分。

## 代码与实验协议

- 配置与逻辑分离，P0/P1共用实现；注释只解释不明显的约束，不做无关重写和过度抽象。
- 配置校验、失败路径和损失分母明确，不吞异常。核心算法应可脱离完整流程测试。
- 模型patch与遮挡块独立配置，默认16/16，比例0.15。配置类初始默认4不是当前单卡400k的运行值。
- padding是独立消融轴：ignore过滤分类padding，official复现官方行为；同一配对除该项外保持一致。历史400k不能单独隔离padding效应。
- DataParallel的batch_size指总batch，标准化和mask在分卡前生成、损失在汇总后计算。检查点保存完整未封装模型状态。
- 完整检测评测由v9统一入口的eval/all执行，使用独立eval_official.json；训练验证仍在evaluate.py。官方协议固定到d458549，合成留出与单变量TSB-AD-U逐序列计算相同指标；oracle F1与部署阈值区分，限制条数明确标子集。audits/eval_20261112/sources也是数值回归依赖，不当作无用源码清理。
- 完整eval的指标计算由src/metric_worker.py在持续复用的独立CPU进程执行；外层spawn隔离训练数据/GPU，Linux内部指标池fork小CPU进程。不得将官方进程池直接移回持有完整pickle/GPU的进程；公式/阈值不因执行优化改变。2026-11-14两个P0评测队列均按用户要求停止，修复后不自动重启；证据见audits/eval_20261114/performance。
- 数据生成配置维持当前决定。实验初始化、划分、种子、权重、步数、评价与停止条件必须可追溯；不把重构改善当作检测、解耦或上下文利用的证明。

## 文档与资产

- 读者分工是长期约定：README简洁说明“做什么、怎么用”，实验README必须直接列出可复制执行的完整命令、前提、GPU选择与输出位置；不要求用户先读Agent文档或自行拼接关键参数。
- 除官方文档外，其他文档均为Agent长期记忆。保留结构/调用链、配置语义、全部实验的计划与状态、结果/证据、失败与纠正、未完成事项和决策理由；不因精简README删去这些信息。
- 文档全量核查时，逐项核对入口参数、配置是否真正被消费、路径是否存在及run证据；删除的资产不能继续写成可用。审计/归档保留历史事实并显式标日期与失效映射；生成的包元数据不当作维护文档。
- 用户操作命令以README为准；内部事实以实现、原始产物和Agent实验记录交叉核对。修改实现后同步相应README与Agent记录，避免把历史计划或历史进程写成当前事实。
- 官方说明和论文资料保持其原有读者定位，不按Agent记忆格式改写。

- 对外文档、历史报告和归档展示文档统一使用相对路径，不写机器名称或本机目录。用户准备步骤写在实验README；Agent迁移核查见`experiments/MIGRATION.md`；历史原始产物与当前文档的展示内容应分开核对。

- `研究问题和思路.md`：问题、假设、相关工作与结论边界；`方法设计.md`：计算图、监督与对照；`实验记录.md`：当前状态、执行事实、证据与下一步。
- 用户用法写在各路线README；v9结构规范、运行语义、原版差异分别写在EXPERIMENT_SPEC、RUNTIME_NOTES、IMPLEMENTATION_DIFF。原流程细节写在EXPERIMENT_NOTES；实验进展汇总到根实验记录，避免持续追加相互矛盾的旧说明。
- v9冗余代码核查已完成，范围与精确补丁见`experiments/ablation_v9/audits/eval_20261112/CODE_REVIEW.md`；用户确认后已删除11个历史备份文件及MODES/config_to_json/Optional三处无用代码；两个占位配置字段保留，执行证据见同目录cleanup_execution.json。自己的任务临时文件按下述收尾规则处理。
- 数据、权重、日志、run配置快照、registry和审计证据视为用户资产。清理前核实引用和影响；不为整理而删除、改写或移动已有run。
- `experiments/_archive/`存放旧文档、重复备份及一次性操作脚本，不是训练入口。旧数据/manifest/runs兼容链接仍被历史快照引用，须保留。
- `experiments/ablation_v9/audits/patch_20261110/sources/`是回归测试依赖的官方/历史源码快照，不是无用副本；不得当作当前实现执行或随意删除。

## 修改与验证

- 临时脚本、压缩包和校验中间文件集中放在项目内明确的临时目录，任务结束时及时清理，不散落在仓库父目录。需长期保留的研究证据放入对应实验audits；不为一次性维护持续新增归档。

- 修改前检查调用关系与已有未提交改动，只处理当前范围，不覆盖他人修改。
- 为关键行为和回归风险做必要检查；未验证的内容明确说明。通过适当检查后不无故重复扩大测试。
- 每次实验前登记问题、对照、数据和计划；结束、中断或失败后记录执行事实、产物及解释边界。状态优先写清“已停止/历史中断/仅合成测试/尚未正式运行”。

## 可选诊断补充（2026-11-14）

- v9 train可用--diagnostics开启独立逐batch观测；train.py中OPTIONAL DIAGNOSTICS块可整体移除。不得用诊断加入额外forward/backward、采样或梯度修正；普通训练默认关闭。日志/现场/RNG在新run的diagnostics/，详见RUNTIME_NOTES与audits/spike_20261114。
- 最初获用户授权重跑P0 official的epoch0/1：p0_official_spike_diag_r3_20261114。epoch0模型/AdamW默认只核对并报告差异，继续作为同配置重跑；显式--diagnostic-stop-on-mismatch才在不匹配时停止。首次有限异常保存pre-step现场后继续，非有限值保存后报错。历史checkpoint缺RNG，匹配也不能预先承诺尖峰精确复现；用户随后明确授权继续诊断至原50轮上限（保留patience7），通过轮末恢复从epoch2继续到新run p0_official_spike_diag_continue_20261114；不包含评测授权。

- train现支持--resume-from到新run，恢复模型/AdamW/完整RNG/best/stale；数据/模型/损失/批次/采样/运行条件须一致。缺完整RNG拒绝；r3同轮独立RNG+完整历史兼容。--checkpoint-keep-last 2只清理新run逐轮权重，保留best/最近两轮/首次事件前驱及独立现场，删除记账。实际是否已续跑查continuation_queue.json，不把等待当启动。

## 当前研究与训练授权（2026-11-15）

- 当前只有P0和新P1。P1保留P0遮挡输入、观测X重构与异常分类，同一次H新增独立正常参考头预测X^0，两个MSE在同一M∩V计算；首轮三损失权重均1，主线official padding。没有B/Z、显式偏差头或双视图。
- 新P1标识p1_mask_x_normal，配置同名；旧p1_full_x与p2_full_x0实现/源配置已移除，历史run/配置快照不可按新语义解释。旧P1已按本次指令停止，保留产物。用户授权本次通过检查后启动新P1双卡正式训练；不自动启动完整评测或恢复P0诊断。
- 新P1普通轮末检查点默认含完整RNG/best/stale，可经统一入口恢复到新run；P0保持普通历史语义，诊断/续跑时含完整状态。P1的valid_total含两项重构与有效点CE，不能与P0总loss数值直接比较方法优劣。
- 文档以当前运行语义为准，历史执行集中实验记录与审计。清理范围是退出的代码/源配置及自己的临时文件，不删除既有run、数据、权重、registry或回归依赖的官方sources。


## P1事故重放授权与详细诊断（2026-11-16）

- 原新P1已由用户停止。用户明确授权从epoch23轮末重放epoch24；当前新run p1_epoch24_replay_20261116_153436，总上限25，只运行该轮及原训练验证，不自动完整检测评测或延长训练。运行状态查实际进程/日志与audits/p1_failure_20261116/REPLAY.md。
- --diagnostic-detailed为可选分频详细观察，src/diagnostics.py内DetailedObserver经OPTIONAL DIAGNOSTICS接入；主循环不额外前后向、不修正梯度。逐batch保留loss/梯度/H范围与RMS/分类GELU哨兵及非有限检查；完整参数/Adam前后与区域统计、逐层激活默认每100批，滚动批次和新事件立即详细采集。CUDA范数/极值在源设备归约，仅传回标量；CPU详细统计。事件保存实际完整输出/梯度、更新前后模型/Adam/RNG，滚动两槽并固定事件前驱；任务梯度分解在离线现场进行。
- 滚动/事件快照仅作离线batch分析，不能传给只支持轮末的--resume-from；有限升级事件有上限，非有限后续现场仍必须保存。诊断覆盖及限制、测试/启动证据见同目录REPLAY.md，不承诺历史CUDA轨迹逐位一致。

## 诊断效率与记录解释（2026-11-16）

- --diagnostic-detail-interval控制详细统计间隔，默认100，1为每批详细；与console --log-freq及滚动--diagnostic-snapshot-interval独立。不通过降低batch、改采样或关闭异常检查来提速。
- batches/updates中的observation_level标记light/detailed；轻量行省略区域/参数Adam/实际更新等字段，不把缺字段解释成0。非定期事件含实际完整输出/梯度与前后状态，但此前中间层激活只能离线补查。
- 当前运行进程仍用已加载旧实现；新代码只对后续启动生效，不因性能修改自动中断、重启或延长现有训练。受控数值与性能证据见audits/p1_failure_20261116/EFFICIENCY.md；短测不证明历史轨迹一致或整轮加速比例。


## 审计与测试维护（2026-11-16）

- audits用途与有效依赖以其README和CLEANUP_20261116.md为准；三处sources（patch_20261110、eval_20261112、spike_20261111/sources_cleanup_20261114）仍是回归依赖。历史JSON/报告只代表当时事实，已删除的临时脚本/清单通过cleanup_20261116.json追溯。长控制台日志可无损压缩，核对解压SHA256；不因清理删除实验原始证据。
- tests当前11份test_模块；早期test_loss_contract/test_sampling_contract已去重，映射见清理记录。退役配置拒绝检查仍须保留；缺必要历史源码应失败，不能静默跳过。测试用法见tests/README.md。


## 运行资产清理与新的续训授权（2026-11-16，以此为准）

- 用户确认删除p0_official_spike_diag_continue_20261114、p0_official_spike_diag_r3_20261114、p1_epoch24_replay_20261116_153436整个run；已有删除账本，旧章节的运行/产物可达性描述已失效。两P0各4份权重和完整eval保留。原新P1仅保留best(epoch22)、epoch23/24/27，其余25份权重已删除。详情见audits/run_cleanup_execution_20261116.json。
- 用户随后明确授权从原P1 epoch23完整模型/Adam/RNG再次训练epoch24及后续，恢复原50轮总上限/patience7，双卡/batch32/official/seed72/分桶50/16-16-0.15/FP32及优化器保持。使用优化后逐batch诊断、详细间隔100、滚动250、最多6有限事件；新run仅保留best/最近2轮/首次事件前驱。授权包括逐轮训练验证，不包括完整检测评测。启动/状态记录见audits/p1_failure_20261116/REPLAY_CONTINUE.md及replay_continue_launch_20261116.json；以实际日志为准。

- 用户误中断后授权删除上一续训p1_epoch24_continue_20261116_210441并从原epoch23重来，已删除旧run及对应启动/控制台记录。新的续训run为p1_epoch24_continue_20261116_211434，tmux p1_epoch24_continue，设置沿用上一条授权；实际恢复与首批以replay_continue_launch_20261116.json和日志为准。

- 历史启动已核实进入epoch24；该run现已停止，当前授权以2026-11-17条目为准。


## 当前停止状态与事故证据（2026-11-17，以此为准）

- 用户已要求停止P1续训并分析；p1_epoch24_continue_20261116_211434于11:03:36停止，最后epoch32/b11450，最近完整轮31。未经新的明确授权不续训。best仍为epoch27，验证F1是训练验证口径，不能作为官方检测成绩。
- 真实事件的离线前后向已完成，没有optimizer.step或算法改动。三个关键现场输出/总梯度逐位复现，CPU局部复核支持注意力/门控MLP高增益、局部重构大输出和后期GELU饱和解释；不称已查明最初训练根因或排除全部硬件因素。
- 证据入口audits/p1_failure_20261116/ANALYSIS_20261117.md（位于experiments/ablation_v9下）；全量106451批、有限6事件及原始资产保留。后续同现场裁剪/lr干预仅为建议，未启动。

## 新的配对训练授权与优化控制（2026-11-17，以此为准）

用户授权加入梯度防护和学习率控制，重新训练P0/P1并保持诊断。两份official_stable源配置独立于四份历史配置；AdamW/wd1e-5、全局L2裁剪5，第1轮5e-5→5e-4，随后固定50轮计划余弦降至5e-5；patience7、双卡DP、batch32、seed72、分桶50及模型/损失不变。阈值5是首轮实验设定，不是由历史日志求得的最优值。

train.py内OptimizationControl属于训练逻辑，不能放入可删除的OPTIONAL DIAGNOSTICS块。诊断仅观察：pre-step现场和gradient_l2保留未裁剪梯度，optimization.jsonl记录裁剪实测后范数/系数/实际lr/更新序号；详细updates追加该控制信息。非有限值在step前保存诊断并停止，不跳过batch。恢复核对固定schedule、每轮步数、next_update；--epochs可缩短本次运行，不能修改lr_schedule.total_epochs或超过其上限。诊断从0开始、detail100/rolling250/max_events12，每批轻量记录。仅best/最近2轮/首次事件前驱保留。

两组依次占用相同双卡，P0正常结束后自动开始P1；任一失败停止队列。只训练及原验证，不自动完整eval。启动与状态以audits/stability_20261117及实际进程为准；旧停止run和事故证据保留。

## src结构整理（2026-11-17，以此为准）

用户要求减少src碎片文件，现9个实现文件加__init__。职责与映射见EXPERIMENT_SPEC及RUNTIME_NOTES。OptimizationControl/轮末恢复在train，优化参数校验在config；两诊断类同处diagnostics。无兼容转发文件；历史审计源码保留不动。训练仍以run_experiment.py统一入口执行，不能因归并新建独立训练循环。
