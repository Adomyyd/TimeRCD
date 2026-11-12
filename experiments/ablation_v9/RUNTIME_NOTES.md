# v9运行语义与Agent记忆

受众：Agent。保存README精简后仍需长期保留的执行语义、限制与解释边界；用户命令集中在本目录README。更新实现时同步本文、EXPERIMENT_SPEC及根实验记录，不能由历史计划自动启动实验。

## 协议与配置


| 组别 | 模型输入 | encoder_mask | M选中位置的重构目标 |
|---|---|---|---|
| P0 | M处置零的X | V & ~M | X |
| P1 | 完整X | V | X |
| P2 | 完整X | V | 配对正常X^0 |

三组模型、原结构双头与损失权重相同。P1/P2前向相同，仅重构目标不同；X^0/Y仅监督。P1−P0共同取消输入置零和attention屏蔽；P2−P1检验正常参考监督，不自动证明解耦、反事实或正确使用上下文。

`model.patch_size=16`是编码器patch；`input.mask_patch_size=16`是遮挡/抽样块，独立配置；`input.mask_ratio=0.15`按有效块数取整抽样。历史官方单卡入口和本地400k在同进程也为16/16/0.15，类默认4不是该运行值。变更遮挡块只改input.mask_patch_size。

配置均在`experiments/ablation_v9/configs/`：

| 组别 | ignore（默认） | official |
|---|---|---|
| P0 | p0_mask_x.json | p0_mask_x_padding_official.json |
| P1 | p1_full_x.json | p1_full_x_padding_official.json |
| P2 | p2_full_x0.json | p2_full_x0_padding_official.json |

每对只差`loss.classification_padding`。ignore先筛选V及labels>=0，padding不参与CE；official严格复现先(labels>0.5).long()再过滤-1，使padding变成正常0，全部B×T参与CE。开关不改变实际padding、attention、重构、采样、初始化或优化器。主方法比较固定padding模式，padding比较固定P组与其他条件；历史400k不作单因素对照。

数据只读共用original_400k/data。六份源配置的数据字段均使用仓库相对路径；从仓库根目录直接使用configs中的JSON，不生成本机副本。默认d_model512、d_proj256、8层8头、单变量，总batch32、length_bucket、bucket_multiplier50、lr5e-4、seed72、50轮上限，完整设置以JSON为准。旧run快照不改写，字段少于当前schema时不直接用于新运行。

## 入口模式

| mode | 当前实际行为 |
|---|---|
| audit | 校验数据文件/manifest声明、划分数量、协议字段，构建CPU模型统计参数；不读取全部训练数据 |
| smoke | audit后用合成batch做前向、损失和反向 |
| bench | audit后读取完整pickle，按训练loader做前向/反向/AdamW更新，输出metrics/bench.json；不做smoke、验证或检查点 |
| train | audit → smoke → 训练及每轮验证/检查点 |
| eval | audit后核对训练配置/--checkpoint，统一评测合成留出与官方TSB-AD-U，输出概率/logit完整指标 |
| report | audit后汇总runs/*/metrics/eval.json；不会重新评测 |
| all | audit → smoke → train → best检查点eval；当前没有自动调用report |

eval使用新的run-id，显式给出相对于仓库根目录的`--checkpoint`路径；当前输入/模型协议来自所选JSON，操作者须与检查点一致。train/bench/all/eval会读取完整pickle，不是小测试。report目前只是指标列表，不提供统计检验或自动配对归因。

## GPU与检查点

默认runtime.parallel=none、gpu_ids=null。多卡在同一入口后增加：

```text
--parallel data_parallel --gpu-ids 0 1
```

并设置CUDA_VISIBLE_DEVICES=0,1；也可在JSON设置runtime.parallel=data_parallel、runtime.gpu_ids=[0,1]、runtime.device=cuda。编号相对于可见GPU列表，首卡为主卡。CLI覆盖在保存配置快照之前生效，--batch-size可同样覆盖总batch。单卡总batch32已有一次显存不足记录，不保证所有设备可运行；改batch时所有配对组同步。

单进程DataParallel按batch维分配完整序列，每卡完整模型。batch_size始终指总batch，32在两卡分摊；标准化/补齐/M在分卡前完成，输出汇总后按完整batch计算损失，一次优化器更新。主卡汇总输出并持有优化器；已有默认规模短程测速摘要，完整轮次吞吐未验证；单多卡dropout及浮点归约不保证逐位一致。没有DDP/FSDP/多机实现。

保存完整未封装模型state_dict（编码器+重构头+异常头，无module.前缀）、优化器、epoch、验证和实际配置，单多卡加载兼容。

## 日志、评价与选择标准

训练在完整pickle加载、split及loader构造完成后才创建train.log并打印Data ready；此前无训练日志不等于进程停滞，也不能仅凭此认定已开始batch更新。optimization.log_freq默认100，在batch索引0、100、200…打印当前batch损失，日志含Total/Recon/Anomaly、分类loss点数/有效点数/padding点数，每轮汇总分项、训练padding_fraction、训练/验证耗时。周期Timing为相邻日志窗口估计；samples_per_s使用窗口批数×当前批大小，尾批时不是精确样本计数。仅周期日志可能漏掉短暂尖峰；逐batch异常触发保存尚未实现。

- classification/total：所选padding模式的目标损失；两模式分母不同，绝对值不直接比较优劣。
- valid_classification/valid_total：共同有效位置CE与加权总损失，保留重构项和既有batch均值汇总。
- 两模式统一使用valid_total早停和选best；ignore时与原total相同。训练结果selection_metric=valid_total。
- 分类accuracy/precision/recall/F1始终仅用有效标签点；classification_loss_points记录当前CE分母，classification_points为检测指标点数，padding_points为补齐点数。
- 另报告M选中/全有效、正常/异常区域的重构误差，以及复制X作为正常预测的基线误差。

训练验证函数evaluate_model保留上述行为；2026-11-12起，入口eval/all另调用benchmark_eval.evaluate_benchmarks，完成合成留出集与TSB-AD-U的完整检测指标。密度分层、分组split审计、多种子统计和上下文干预仍未实现。P2对X^0、P0/P1对X，各组目标total也不能直接横向判断方法优劣。正式对照需在共同检测协议、同预算/固定epoch等条件下评价。

## 证据与状态

合成验证与历史训练状态见根`实验记录.md`；本目录`audits/`保留patch、DataParallel与padding核查证据，patch sources是测试依赖。2026-11-11核查已见P0 official双卡run启动，详见根实验记录的最新快照；不能继续概括六组均未开展。README精简不删除旧P0、400k的loss尖峰记录，也不把合成测试转述为方法收益。

## 分桶与短程测速

训练先按seed42做95/5随机split，再按训练子集长度排序，以总batch×bucket_multiplier分池；当前为32×50=1600条。每轮使用runtime.seed×1000003+epoch随机池及池内顺序，保留尾批，每样本出现一次。训练M使用runtime.seed+epoch×1000003+batch_index；相同协议下各P组同批同M。sequential保留split后的固定顺序，不作每轮shuffle；验证始终固定顺序、不分桶。改变batch也会改变验证分批、标准化及M的随机流，必须统一。

bench的--max-batches默认300；--batch-size可覆盖JSON，快照记录实际值。预热默认20，代码按max_batches−1裁剪，尚未按实际loader长度裁剪；短数据可能timed_batches=0仍返回completed。首个计时批的collate/搬运/mask未计入，达到limit后可能多取一批；吞吐按配置batch计算会高估尾批样本数；同步所有可见CUDA设备而峰值只重置当前卡。少数连续长度池不代表全轮；epoch_estimate_hours仅训练批次外推，不含验证、存盘或加载。问题已审查，尚未修复，不用该值承诺完整轮次耗时。

audits/bench_20261111/bench_summary.json保留他次测速汇总：单卡batch16为0.1612秒/批，双卡batch16为0.1496秒/批，双卡batch32为0.2756秒/批；单卡batch32有OOM。四个原测速run已移除，本次无法重核原日志。摘要中的2.8倍、整轮小时及“batch32必须双卡”是当时外推或特定设备观察，不作为普遍结论。没有同设备/同batch/同计时协议的当前sequential完整轮次对照。

## 配置中尚未生效的能力

runtime.mixed_precision未被v9训练/bench读取，无AMP实现；设true不会启用混合精度。evaluation.threshold_source为训练协议占位，训练验证点级指标仍对logits argmax；完整eval改用官方oracle阈值搜索，仍未拟合验证集部署阈值。schema_version未校验具体版本号；未使用的MODES常量已清理，入口choices支持bench。run_smoke虽含反向，不调用optimizer.step；正式train在其后重置seed/清梯度，仍从随机模型初始化训练，没有权重初始化或resume入口。

status/registry在入口开始时写running；KeyboardInterrupt会标aborted，其他捕获异常标failed，但外部SIGTERM/SIGKILL可能留下running。不得将状态文件代替进程/日志核查。

## 统一检测评测（2026-11-12）

代码边界：evaluate.py及训练调用保持原样；benchmark_eval.py承接eval/all，统一入口不变。评测依赖见requirements-eval.txt；eval/all在训练开始前检查评测依赖与真实基准清单。eval必须提供checkpoint，且检查checkpoint保存的experiment_id/model/input/loss/split及合成数据位置与传入配置一致；建议直接使用训练run的config.json。CPU加载检查点并释放优化器payload后再推理；评测从完整权重恢复，不再随机初始化后当作待评模型。评测seed2024独立于训练seed72。

对齐版本：thu-sail-lab/Time-RCD@d458549f6962029a5c5241c43a5baa1ec5eccbb2，是轻量包替换前最后包含main.py的官方版本。官方最新main已无原基准脚本，不把新轻量API当作原指标协议。main.py与本项目逐字节一致；utils/dataset.py一致；models/TimeRCD.py只调整包导入。根evaluation下两处已有F1_T串行执行/传入n_splits修正保留；同800分位点下与原并行路径做数值对照，证据见audits/eval_20261112。官方快照只作证据和测试依赖，不作为新入口。

数据与推理：真实基准复用官方单变量排除表，纳入IOPS/MGAB/NAB/NEK/Power/SED/Stock/TODS/UCR/WSD/YAHOO，共700条；按组数量检查完整性。CSV先dropna再按文件名倒数第三段读取训练边界，仅测试后缀进入模型；ACF周期用完整清理后的序列计算。合成侧仅用与训练相同split的留出索引，每条完整原始X独立评测，X0不输入、不计检测分数。两侧均使用TimeRCDDataset：整个测试序列每通道numpy总体std标准化，std为0才换1e-8；窗口min(5000,T)、stride=window，末尾重复最后值且mask为false；展开并截回原长度。编码器只接有效mask，不按P0训练遮蔽；检测头只读取encoder/anomaly_head，不执行重构头。DataParallel只分窗口batch；默认推理batch64，不使用optimization.batch_size。

指标：概率为softmax(mean_channel_logits)[...,1]，logit为l1-l0，分别逐序列调用官方get_metrics_optimized（默认fast，F1_T分位点800、chunk25、worker4/4）或get_metrics（default模式）。后者与fast不是可混用的同一预算。保留官方mean+3std的pred参数传法；具体阈值型函数仍搜索测试标签上的oracle阈值，不能称validation-only，也不修改训练早停。官方某些内部计算异常返回0，本入口保留该行为；返回字段缺失/非有限数则失败，不把失败自动剔除后报完整结果。PA/事件指标不是固定0.5阈值F1。VUS沿用version=opt、阈值数250；不拼接不同序列、不降采样。

产物：metrics/per_series.csv逐完成行落盘，包含domain/group/series/score_kind/长度/异常点数/周期/边界及16指标；group_mean_metrics.csv按组，domain_summary.csv分列series_macro与group_macro，eval.json保存相同嵌套汇总。真实侧group_macro是先组内平均再11组等权平均，series_macro会受各组条数影响；官方main只输出逐序列CSV，额外汇总属于本项目透明展示。合成侧为序列等权均值，不是全点拼接分数。合成与真实不合并为一个总分。

evaluation_manifest.json记录评测参数、checkpoint哈希/epoch、执行源码哈希、逐真实CSV哈希/清单、合成留出索引哈希、阈值/异常回退约定；eval_config.json为实际协议。scores/*.npz包含value/label/anomaly_score/logits（默认保存，可通过save_scores关闭）。eval.log先记录准备阶段，再记录序列推理/两类分数计算及完成，完整pickle准备阶段可能耗时。CPU线程数1只在本次评测进程设置，不修改其他训练进程；没有自动更改训练batch、采样器、mask或损失。

--eval-max-synthetic/--eval-max-real只截取稳定顺序前N条；真实清单仍先检查完整700条，合成pickle仍全量读取。full_benchmark表示两侧均完整；改变窗口/batch/metrics_mode/F1_T参数/评测seed时official_default_parameters=false。这些标记不替代比较时的完整参数核对。逐序列处理异常时写eval_failure.json并保留已完成CSV/分数；入口捕获异常并标failed，KeyboardInterrupt标aborted。依赖/准备阶段失败可能尚无eval_failure.json；不生成成功eval.json，无自动跳过或resume。all仍会训练后跑完整eval，train不会。当前只有CPU合成与官方数值对照，不声称完整数据或GPU评测已完成。

## 2026-11-12代码清理完成

全量引用核查及逐文件保留理由见audits/eval_20261112/CODE_REVIEW.md，cleanup_proposal.diff对应用户确认后已应用的代码改动。已删除MODES/config_to_json/Optional及两隐藏备份目录共11文件；不生效配置字段本轮保留。执行证据见cleanup_execution.json，CPU回归87通过、6项因需GPU跳过。bench既有计时限制、schema校验边界及未实现功能不在本次评测实现中顺带修改。本次清理未操作P0 ignore进程或其产物，现存run/registry不属于清理对象。
