# P0 padding初步对照：合成200/真实700

受众：Agent。2026-11-14用户确认本范围；取代初始全量评测计划。**之后用户要求停止并先修复耗时，当前队列已停止，本协议仅保留未来约定范围，不是执行授权。**

- 数据：同一seed42、95/5合成留出顺序的前200条；真实为官方TSB-AD-U全部700条/11组。固定索引不根据模型表现选择。
- 权重：ignore训练run p0_ignore_dp_20261112_170107 的best epoch26；official训练run p0_official_dp_20261111_003608 的best epoch31。训练配置仅classification_padding不同，实际早停轮数34/39不同。
- 入口：仍用run_experiment.py --mode eval及eval_official.json；CLI只加--eval-max-synthetic 200，真实无条数限制；seed2024、推理batch64、DataParallel逻辑0/1；概率与logit分别计算16项官方指标，参数不缩减。
- 执行：先ignore、成功后official，后台与SSH会话分离。当前状态是execution_subset200.json；原execution.json对应已停止的full队列。
- 当前run：p0_ignore_dp_20261112_170107_eval_20261114_s200_r700、p0_official_dp_20261111_003608_eval_20261114_s200_r700。
- 解释：full_benchmark=false是正确的合成子集标记；真实全部700，official_default_parameters=true。合成用于选best且仅200条，不能称独立或全量测试；F1为官方oracle阈值，非部署成绩；一个训练种子不证明稳定改善。

两组均完成后，预期1800分数行/模型（合成400、真实1400）；调度作业最后一步核查完成标记、条数、16指标有限值、唯一键，比较合成索引哈希、真实文件清单、协议/源码摘要及对应序列元数据，生成comparison_s200_r700.csv/json和COMPARISON_s200_r700.md。真实同时报告序列均值和11组等权均值，合成单列；不混成一个总分。未完成或失败时不报告完整对照。

范围变更停止记录：scope_change_stop.json；full ignore保留83行，标aborted，full official未启动。旧run不覆盖、不作为新run的断点。

耗时诊断：small_process_metric_timing.json以同一已保存合成序列、相同标签/周期/参数在小进程重算，两类分数耗时3.04/2.17秒，16指标与旧full行最大差值0；旧大数据常驻进程约14/13秒。不能将现有额外执行开销当成指标算法或模型必需耗时。本轮仅诊断，未修改执行源码；真实长序列总时间待实测，先前20k全量的约12天粗估已不适用。

最新状态：subset ignore保留27行并aborted，official未启动；性能修复及验证见performance/REPORT.md。不得执行旧后台脚本恢复评测，需新指令及新run-id。


## 2026-11-14重新执行状态

此前中断的ignore评测run及其部分分数已按用户要求删除，详见同目录restart_cleanup.json。耗时修复后的新队列已获用户明确授权启动：

- ignore：p0_ignore_dp_20261112_170107_eval_20261114_restart_s200_r700
- official：p0_official_dp_20261111_003608_eval_20261114_restart_s200_r700

固定合成留出前200条＋全部真实700条/11组，其余官方完整指标参数不变。两组顺序执行，official等待ignore退出码0。当前ignore处于running/数据准备阶段，official尚未开始。旧run删除清单见restart_cleanup.json；新队列证据见restart_launch.json和restart_eval_queue.json。不得把partial行数描述成完整评测结果。


## 重新执行更新（attempt2）

attempt1完成合成200及真实3条后因官方Affiliation-P返回NaN而退出；官方main允许NaN写入逐序列CSV，入口已改为保留NaN并在宏平均中跳过，同时记录有效数。修正后14项benchmark测试通过。attempt1不完整run已删除，失败详情仅保留restart_nan_failure.json。

当前attempt2 run-id：p0_ignore_dp_20261112_170107_eval_20261114_restart2_s200_r700、p0_official_dp_20261111_003608_eval_20261114_restart2_s200_r700。ignore先执行，成功才启动official；范围、指标、权重、共同合成索引和真实700条协议不变。