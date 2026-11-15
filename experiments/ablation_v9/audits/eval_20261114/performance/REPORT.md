# 评测耗时定位与修复（2026-11-14）

受众：Agent。当前正式评测全部停止，未因修复完成而恢复。本报告仅记录执行开销修复及有界验证，不包含完整P0检测对照成绩。

## 停止状态

用户要求“不要再继续了，先修复耗时问题”，随后于13:48:42（UTC+8）停止subset200调度及同组子进程。ignore保留27行，official尚未启动；execution_subset200.json已记aborted。此前full队列也已停止，ignore保留83行、official未启动；execution.json已记aborted。分数、日志、权重、数据均保留，不将部分行转换成成功结果。精确身份与停止证据见stop_before_fix.json和上一层scope_change_stop.json。

未来约定范围仍是每组固定200条合成留出＋全部700条真实，但当前不执行；重新开始应使用新run-id。旧调度脚本不再作为入口，统一入口仍为run_experiment.py。

## 与官方流程的对比

来源为thu-sail-lab/Time-RCD@d458549f6962029a5c5241c43a5baa1ec5eccbb2；固定源码在audits/eval_20261112/sources。仓库main.py与固定官方版本逐字节一致。

| 环节 | 官方main | 修复前v9 | 修复后v9 |
|---|---|---|---|
| 数据常驻 | 当前真实CSV和模型，不加载训练pickle | 为合成留出划分缓存整个400k，即使只取200条 | 仍保留一次完整pickle读取 |
| 指标调用位置 | 相对较小的推理进程 | RSS约42.5GiB的父进程 | 持续复用的独立CPU工作进程，约480MiB |
| 指标内部进程池 | Linux fork，多层池反复创建 | 反复fork持有完整pickle的地址空间 | 只fork小CPU工作进程 |
| 评测工作量 | 官方真实单变量基准 | 增加合成留出，并比较两个训练权重 | 工作量及数值协议不变 |

主要额外耗时来自大地址空间下重复fork。仅将最外层改为spawn仍不充分：若内部池也沿用spawn，会反复导入依赖，诊断约6.8秒/次。最终方案在独立CPU进程初始化后，仅在Linux将内部池启动方式设为fork；外层仍显式spawn，避免继承父进程大缓存及CUDA状态。

官方默认heavy/light worker数为auto，本项目既有协议固定4/4；这一执行设置差异保留并披露。根指标函数相对官方仅有既有_compute_f1_t串行阈值执行修正，本次不改变它。fast模式、F1_T 800分位点、Standard F1 500阈值、PA分位点0.7至0.99步长0.001、Affiliation 300阈值、VUS opt/250均未变。七份官方源码与修复前清单摘要一致，详见official_flow_comparison.json。

## 代码范围

只有以下三份代码发生本次修改，精确差异见performance_fix.diff：

- src/metric_worker.py：新增OfficialMetricWorker；一个持续复用的spawn CPU进程，按协议设种子/线程，Linux内部fork。仅传分数、标签、周期和协议；异常传播，退出关闭。
- src/benchmark_eval.py：两类分数及全部序列复用同一指标工作进程；日志拆分准备、启动、推理、周期/存分数、指标耗时；manifest新增执行后端及worker源码摘要。
- tests/test_benchmark_eval.py：增加真实进程的指标一致性、复用、父缓存隔离、CPU状态、异常和关闭验证；原流程失败注入用轻量fixture。

训练、模型、损失、采样、六份训练配置、独立评测协议、原始官方函数及已有权重未因本修复修改。

## 受控性能与数值结果

固定两个已保存合成序列98915（1394点）、111967（5478点），各测probability和logit，标签、周期和指标协议相同。计时单位为每类分数的一次完整16指标计算，不是整条序列的两类总和。

| 场景 | 四次墙钟时间（秒） | 均值 |
|---|---|---:|
| 不加载训练pickle，直接调用 | 3.408 / 2.159 / 2.507 / 2.810 | 2.72 |
| 父进程加载400k，直接调用旧路径 | 13.156 / 11.901 / 12.649 / 13.087 | 12.70 |
| 父进程仍加载400k，最终独立工作进程 | 2.980 / 2.126 / 2.455 / 2.286 | 2.46 |

最终方案在同类大缓存场景中约加速5.2倍；所有16指标与保存的原结果最大绝对差均为0。工作进程全程同PID，cached_datasets=0、cuda_initialized=false、内部fork。父进程仍约42.5GiB，说明改善来自隔离指标执行，而不是减小评测数据范围或减少指标。

64次fork的累计调用耗时在加载数据前后为0.969秒和45.197秒；它是并发累计量，不可直接从总体墙钟时间中相减。证据见controlled_timing.json、fork_events.jsonl和fixed_loaded_parent.json。

单体pickle读取仍需约223–227秒，未在本次改变存储格式。少数样本不能估计所有700条真实数据的成本；此前基于慢路径的数天/半天外推不再是修复后ETA。

## 验证及限制

- 最终CPU回归：88 passed、6 skipped，18.44秒，见cpu_regression_final.log；跳过项需要GPU。
- 实际best权重有界GPU检查：两条上述合成与一条NAB真实序列。单卡同批次下probability/logit与固定官方zero_shot逐元素一致；两卡重放与原保存的合成分数逐元素一致。
- 两卡与单卡本身存在原有浮点差异（logit最大约6.1e-5），不将其归因于本修复。早期gpu_parity.log、gpu_parity_final.log的断言失败属于诊断脚本：分别因官方tester对短序列修改win_size后被复用、以及直接严格比较不同并行批次。修正诊断为逐文件重设窗口、同批次比较后通过，最终证据为gpu_parity_verified.json/log。
- NAB诊断测试后缀13537点，推理约0.063秒，两类指标约0.759/0.795秒；不代表全部真实数据吞吐。
- 实际统一入口＋实际spawn worker＋实际官方指标的微型CPU流程通过：一条合成、一条伪真实，产生4行并completed，见unified_entry_process_check.json/log。该fixture没有写生产runs或registry。
- 非Linux内部进程启动方式及性能未实测；不会将Linux加速幅度推广至其他平台。
- 没有重跑正式P0基准，也没有训练更新；没有方法增益、padding检测优劣或完整总耗时结论。

## 证据索引与收尾

- 停止：stop_before_fix.json、subset_queue_before_stop.json。
- 官方对照及补丁：official_flow_comparison.json、performance_fix.diff。
- 机制与最终性能：controlled_timing.json/log、fork_events.jsonl、fixed_loaded_parent.json/log。
- 其他探索：actual_worker_profile.json/log；controlled_timing中的第三场景为嵌套spawn探索，不作为最终加速数字。
- 数值/流程：gpu_parity_verified.json/log、cpu_regression_final.log、unified_entry_process_check.json/log。
- 文档与收尾：document_updates.json、final_checks.json、cleanup.json。

本次临时脚本与微型fixture在诊断进程退出后清理；长期保留上列证据、正式run及训练资产。历史其他清理候选不属于本次范围。
