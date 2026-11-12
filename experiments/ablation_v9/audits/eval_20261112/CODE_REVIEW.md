# v9代码与文档核查（2026-11-12）

受众：Agent。状态：用户已确认，A组清理完成，B组占位字段保留。本文第1–5节保留清理前的引用/用途核查依据，表中行号对应修改前版本；cleanup_proposal.diff现为已应用补丁，执行记录见cleanup_execution.json。统一eval实现与测试见REPORT.md，不能将本报告解释为完成全量检测评测。

## 1. 范围与方法

逐项查看run_experiment.py、全部src和tests、六训练JSON、独立eval协议及依赖；AST解析并交叉检索全仓活动Python引用，核对两个隐藏备份与现版本的差异。共41份Python：24份活动代码（入口1、src12、tests11）、4份隐藏旧副本、13份审计源码；未发现整份活动模块完全脱离入口/测试的情况。静态引用仅作线索，同时检查间接调用、动态加载、配置消费和历史证据用途。

文档核对覆盖42份既有Markdown（含历史审计/归档/旧索引，排除缓存README及生成包元数据），新增本文后43份。历史资料保留日期边界；官方Tutorial不按Agent格式重写。code_inventory.json、document_inventory.json为相对路径清单；cleanup_proposal.json记录拟删文件、字节数与SHA256，待执行补丁只涉及下面A组的三处代码。

## 2. A组：已按用户确认清理

| 精确对象（相对于experiments/ablation_v9） | 核查结论 | 拟执行动作与影响 |
|---|---|---|
| src/config.py:13 的MODES | 全仓活动Python仅定义处出现；入口自己声明argparse choices，旧常量还漏bench | 删除这一行；入口模式不变 |
| src/config.py:110 的config_to_json | 全仓活动Python只有定义，无调用；快照由入口write_json实现 | 删除函数；config读取/快照行为不变，json导入仍保留 |
| src/models.py:3 的Optional导入 | AST与全文引用均确认未使用 | 删除导入及多余空行，不改模型 |
| .backup_bucket_20261110/，共10文件 | 分桶前旧版本，不参与任何活动Python导入/读取；与当前并非相同文件 | 已删除目录内这10文件及空目录，不改当前src/configs |
| .backup_spike_20261111/audits_README.md | 只有旧审计索引，当前audits/README已包含其内容并增加spike/eval索引；无代码引用 | 已删除文件及空目录，保留spike正式审计 |

bucket备份10文件为audit.py、config.py、data.py、train.py，以及p0_mask_x.json、p0_mask_x_padding_official.json、p1_full_x.json、p1_full_x_padding_official.json、p2_full_x0.json、p2_full_x0_padding_official.json。旧JSON带历史数据位置、缺少分桶设置，P1/P2仍为旧batch16；不能误当成现在六配置的备份。当前实现、历史run快照、registry、数据、权重及审计证据不在删除范围。

A组执行前应核对proposal中的摘要仍匹配；如用户/其他进程改动候选，停止对应删除并重新判断。无需为少量无用内容再新增一层重复归档；若用户希望保留备份历史，改为归档方案后再执行，不能自行选择。

## 3. B组：不生效的配置记录，建议本轮暂保留

六份源训练JSON各有runtime.mixed_precision=false、evaluation.threshold_source=validation_only，共12个键值；v9没有消费这两个字段，也没有AMP或部署阈值拟合。它们是历史协议占位，不能当作开关。根原训练代码确实消费自己的mixed_precision，与v9 JSON不同。

本轮建议保留并明确未生效。若用户要求进一步简化，可以另行从六份源JSON删除这12个字段并同步说明；只改变未来配置记录，不改既有run/检查点。不得借此顺带启用AMP、改阈值算法或重跑实验。schema_version虽然尚未校验具体版本，仍为结构记录，不直接归为可删字段。

## 4. 需要保留的代码和资产

| 对象 | 保留理由 |
|---|---|
| src/evaluate.py、evaluate_model、_make_eval_inputs | train_model每轮验证、区域误差、共同valid_total选best及回归检查仍调用；新benchmark_eval不替代它 |
| src/reporting.py | --mode report真实调用，读取既有eval.json；虽简单但不是废弃入口 |
| src/train.py的_device | train/bench仍调用，并转交统一resolve_device，不是早期报错后的残留 |
| src/data.py的_DATA_CACHE/_LENGTH_CACHE与sample_lengths | 训练/分桶/验证及完整eval的加载路径使用，避免反复读取大pickle |
| Sample.attribute、collate的attribute/lengths、模型输出h | 当前优化器/检测汇总未消费部分元数据，但属于明确的数据/模型输出契约，且未来分层分析需要属性；不把它们当作已实现密度分析，也不在本轮改接口 |
| src/__init__.py | 包标识，不按零字节判断无用 |
| run_experiment.py eval/all分支的指标/ACF预导入 | 在all训练前暴露缺失评测依赖，属于有意的预检；不能按未引用变量直接删除 |
| tests中的动态官方模块加载与导入 | 为固定官方源码数值对照建立包上下文；静态未使用警告不足以作为删除依据 |
| audits/patch_20261110/sources/ | upstream_training/local_training/train_400k/time_rcd_config被test_patch_parity直接读取；encoder/main_library即使未被当前测试直接加载也保留官方实现/全链路核查证据 |
| audits/eval_20261112/sources/ | main/HP_list/TimeRCD/metrics/basic_metrics被test_benchmark_eval读取；dataset/slidingWindows虽主要由实际仓库模块复用，仍是逐字节来源证据 |
| 其余audits、registry、runs及旧资产链接 | 实验来源、记录和用户资产；不因不参与forward而删除 |

train.make_batch_inputs与evaluate._make_eval_inputs有相同输入规则，但各自参与实际调用且受一致性测试约束；合并属于重构，不是删除死代码，本轮不做。审计源码中的未使用imports保持原文，避免破坏来源哈希。未自动删除训练期间可重建的缓存。

## 5. 发现但不夹带修复的问题

- bench：预热未按实际loader长度裁剪，可能零计时仍成功；首批计时/尾批样本计数、多卡同步和峰值重置范围、少数长度池外推等限制仍存在。已由bucket_review_20261111和RUNTIME_NOTES记录；测速功能仍有用途，不能整段删除。
- 配置：schema_version只要求存在，部分未知子键及数值尚未严格校验；两个占位字段不生效。修改校验属于后续独立变更。
- 诊断/恢复：逐batch尖峰触发记录、v9训练resume未实现；周期日志不保证捕获异常batch。
- 入口：部分只对特定模式有意义的参数在其他模式仍可解析；report仍先audit且只输出结果列表。元数据写入在主try之前，极早失败可能没有完整status记录。
- 完整eval：本次只有CPU合成/官方数值对照及真实清单和实际检查点兼容核对，未做全量或真实GPU推理。逐序列异常写eval_failure；准备阶段失败或KeyboardInterrupt可能只有入口status，没有该文件。官方指标内部异常返回0的行为仍保留并披露；无评测resume。
- 新eval属于官方协议编排，不证明P0/P1/P2检测收益。合成留出已用于选best，不能声称独立测试性能。

## 6. 文档与验证

根README/experiments总览补充统一eval；总览和MIGRATION补充评测/测试依赖与真实数据用途，六训练配置与独立eval协议分开表述。研究/方法/记录与v9规范、运行语义、差异文档同步；纠正训练验证写成完整eval的旧措辞，明确部署阈值和官方oracle指标不同。原流程、生成器用法核对后保留有效内容；历史审计/归档不改写原日期事实。

统一eval最终CPU测试87 passed、6 skipped（需GPU），17项原训练src/六配置摘要保持；具体证据见verification.json。后续仅文档与方案修改，不重复运行整套测试。final_review_checks.json记录最终AST、README bash语法、文档机器路径筛查、候选未执行及保护摘要检查。未启动生产评测、未占用训练GPU、未操作P0 ignore、未更改六训练配置或既有run/registry。

后续若继续简化配置或修复bench等问题，按独立任务处理，不从本次清理授权扩大范围。

## 7. 用户确认后的实际执行

用户回复“按照你的建议清理”后，执行前逐一匹配11文件和2处源文件的SHA256，并核对备份目录没有清单以外文件。已删除11个文件（30167字节）及两个空目录；src/config.py删除MODES和config_to_json，src/models.py删除Optional导入，结果摘要与已审阅补丁完全匹配。未改六训练JSON；runtime.mixed_precision、evaluation.threshold_source保留。42份其他活动代码/配置/测试及官方来源摘要保持。

清理后CPU回归87 passed、6 skipped，10.59秒；跳过项需要GPU，测试进程隐藏CUDA。执行记录和原始输出分别为cleanup_execution.json、cleanup_cpu_tests.txt。文档中的当前待确认/目录存在性说明已更新；本报告前文及final_review_checks.json、verification.json保留清理前的审查/验证事实，不拿旧摘要冒充清理后摘要。最新检查见cleanup_execution.json。未操作训练进程、已有run/registry、数据和权重，未启动生产评测。
