# 共同稳定训练与模块归并（2026-11-17）

用户授权P0/P1从头训练并启用诊断，后要求减少src文件。本目录为计划、检查及启动证据，运行状态以queue_status.txt、实际进程和各run日志为准。

## 协议

两组official padding、双卡DataParallel、总batch32、seed72、split42、分桶50、patch/遮挡16/16、比例0.15、FP32。AdamW峰值5e-4/wd1e-5，所有参数梯度合并计算L2范数，超过5则同系数缩放。第1轮从5e-5预热至5e-4，后49轮逐更新步余弦衰减至5e-5；50轮固定计划，patience7可能提前结束。裁剪阈值是实验选择，未证明最优。模型/损失/初始化次序保持既有v9语义。

逐batch基本诊断，详细100、两槽rolling250、最多12有限升级事件、非有限现场不限且保存后停止。先保存原始梯度现场，再裁剪，再AdamW更新。optimization.jsonl阶段为step前，updates.jsonl核对实际执行后的参数/Adam状态。epoch检查点保存完整RNG、优化位置和固定计划；新run只保留best、最近2轮、首次事件前驱。

P0先运行，正常完成后P1启动；失败或中断停止队列。两组启动前校验同一源码和配置SHA256。只含训练与原验证，未授权自动完整检测评测。源码与配置快照见launch_sources.zip；命令及队列身份见launch.json。

## 文件归并

本轮整理前src有17个Python文件（含新增优化控制及__init__），归并后10个：config、data、models、losses、train、diagnostics、evaluate、benchmark_eval、metric_worker和__init__。不保留兼容转发模块。映射和原hash见structure_mapping.json。历史审计源码保留原字节，测试使用隔离命名空间解析其旧导入。

## 检查证据

- tests.log：初次123项通过，旧配置清单断言未包含新两份配置；修正后的config_recheck.log为124项全通过。
- consolidated_tests.log：归并后123项通过，剩余全尺寸测试调用遗漏新夹具参数；仅修正该调用，structure_recheck.log为该项通过。合并后的124项均已通过，未把失败日志删除。
- test_optimization覆盖CPU/双卡P0/P1连续与恢复模型/Adam/RNG及调度位置一致，裁剪方向与范数、非有限拒绝更新、历史关闭路径与配对配置。
- fullsize_smoke.json：正式模型、双卡batch32×10000、前后向/裁剪/AdamW/详细诊断/rolling现场两组均通过，峰值约8.8–8.9GB。首次短测脚本发生临时文件inspect.py遮蔽标准库的启动错误，改名后通过；未影响正式入口。
- 源码迁移后的活动导入无退役模块；训练期间不修改启动快照中的源码/配置。

这些检查验证实现与短程执行，不证明长期无尖峰，也不能分别估计clip与调度的因果贡献。未来比较采用同一官方评测协议；P0/P1总loss不同，不据此判优。

## 启动核实

2026-11-17T12:58:00.935283：P0已完成100批更新，首步lr5e-5、原始梯度6.479746→裁剪后4.999999，pre/post诊断和rolling现场存在，已观察更新无非有限状态。P1仍排队。精确数值见startup_verification.json；这是启动时快照，后续以实际日志为准。任务临时脚本及短测现场已清理。
