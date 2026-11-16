# 分频诊断效率修订（2026-11-16）

> 当前资产状态（2026-11-16）：用户已确认删除两份P0诊断run和首份P1 epoch24重放run；下文是历史事实，原run内日志/诊断/权重已不可用，汇总见上级run_cleanup_execution_20261116.json。新P1原训练保留best、epoch23/24/27，仍可作为恢复/前后对比来源。

受众：Agent。用户授权修改诊断提速；没有授权本次自动停止/重启/延长已运行的epoch24重放。统一入口仍为run_experiment.py，训练调用BatchDiagnostics，后者按需委托DetailedObserver。未改变模型、损失、数据、mask、学习率或优化器更新。

## 实现与证据覆盖

- --diagnostic-detail-interval默认100、必须为正，控制完整统计频率；设1可恢复每批详细采集频率，归约实现仍为新版。console日志和滚动现场间隔分别独立控制。
- 每批保留分项loss、样本ID/长度/mask种子计数、输入目标和输出范围、观测逐样本重构误差、全参数/模块梯度、完整H极值/RMS、分类GELU固定位置抽样。更新后每批检查模型和Adam非有限值。
- CUDA梯度/状态L2采用FP64设备归约，分设备批量回传标量；H仅回传完整极值和RMS。CPU路径用NumPy。异常非有限H回退完整CPU有限子集统计。减少每批完整梯度、参数、Adam和H的CPU搬运。
- 定期详细或滚动批次，采集全套已定义模块hook及区域/目标/分类统计、输入mask摘要、参数Adam前后统计与实际更新；总梯度始终不是分任务梯度。
- 反向后新阈值事件立即提升详细采集，并保存实际完整输入/输出/梯度、模型Adam/RNG和更新后状态；同级持续有限事件、超过有限事件预算不触发每批全量采集。非有限事件不受预算或此前同级事件阻挡。
- 非定期事件发生前未采各层hook，activations_complete_for_schedule=false；完整中间激活需离线现场重放，不宣称在线已经记录。分类GELU哨兵每批仍存在，其抽样不是全张量极值。
- 两JSONL均保留逐batch行，observation_level明确light/detailed；轻量updates没有实际更新范数/Adam统计，缺值不是0。现场依然不能直接作为轮末resume输入。

## 检查与性能边界

CPU完整测试108 passed/13 skipped（GPU条件）；补充4项小模型双卡/CUDA检查全部通过，含诊断频率变化下模型/Adam/验证/RNG一致、P0兼容、有限大幅值/NaN归约。事件提升/重复阈值保持轻量、非有限更新保存、现场实际更新重放及CLI配置均有回归覆盖。见efficiency_cpu_tests.log和efficiency_gpu_tests.log。现有重放占用双卡，未并发重新执行正式模型最长序列GPU测试；此前旧诊断的正式尺寸通过不能冒充本次验证。

合成CPU纯观测基准：38370180参数、batch32、长度1024；每种路径预热2次、测20次并重复一组。两组中位耗时：

| 路径 | 第一组秒/批 | 第二组秒/批 |
|---|---:|---:|
| 旧每批全量诊断 | 0.9286 | 0.9110 |
| 新非定期轻量诊断 | 0.2212 | 0.2226 |

普通批次诊断部分减少约76%；定期详细批0.8856–0.9232秒。未执行真实数据训练更新，不含CUDA拷贝/hook、前后向、快照保存，不能外推整轮GPU训练4倍提速。详细数据见efficiency_benchmark.json。运行旧进程仍每批详细且未热更新；新实现只对后续启动生效。

变更精确补丁见efficiency_changes.patch，相对于本次修改前已有未提交状态，不能把此前重放改动误认为本次新增。保留原run、配置快照、registry、检查点与诊断证据。

最终审查补充：修复持续hidden_scale超过事件预算时轻量行缺hidden_moments的边界，回归覆盖同级持续阈值仍保持轻量且不报KeyError；诊断两文件复测11 passed/7 GPU条件跳过，见efficiency_final_diagnostic_tests.log。新归约内核及训练逻辑未在此修订改变；Python语法、README全部7个bash块和git diff --check通过，见efficiency_final_checks.json。
