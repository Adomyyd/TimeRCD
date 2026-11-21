# 共同稳定训练与模块归并（2026-11-17）

用户授权P0/P1从头训练并启用诊断，后要求减少src文件。两组训练及 700 条真实序列的配对评测均已完成。[检测结果](REAL700_RESULTS.md)与[总表](../../RESULTS.md)使用各自 best。

## 协议与证据

P0 正常完成后自动启动 P1。失败或中断时队列停止。两组在启动前均校验源码和配置 SHA256，本次任务为训练和逐轮验证。固定设置、调度、诊断及检查点定义见 [运行说明](../../RUNTIME_NOTES.md)和[启动记录](launch.json)。源码及配置见 `launch_sources.zip`，实时状态查 `queue_status.txt`、进程和各 run 日志。

## 文件归并

本轮整理前src有17个Python文件（含新增优化控制及__init__），归并后10个：config、data、models、losses、train、diagnostics、evaluate、benchmark_eval、metric_worker和__init__。不保留兼容转发模块。映射和原hash见structure_mapping.json。历史审计源码保留原字节，测试使用隔离命名空间解析其旧导入。

## 检查证据

- tests.log：初次123项通过，旧配置清单断言未包含新两份配置；修正后的config_recheck.log为124项全通过。
- consolidated_tests.log：归并后123项通过，剩余全尺寸测试调用遗漏新夹具参数；仅修正该调用，structure_recheck.log为该项通过。合并后的124项均已通过，未把失败日志删除。
- test_optimization覆盖CPU/双卡P0/P1连续与恢复模型/Adam/RNG及调度位置一致，裁剪方向与范数、非有限拒绝更新、历史关闭路径与配对配置。
- fullsize_smoke.json：正式模型、双卡batch32×10000、前后向/裁剪/AdamW/详细诊断/rolling现场两组均通过，峰值约8.8–8.9GB。首次短测脚本发生临时文件inspect.py遮蔽标准库的启动错误，改名后通过；未影响正式入口。
- 源码迁移后的活动导入无退役模块；训练期间不修改启动快照中的源码/配置。

这些检查验证实现与短程执行，不证明长期无尖峰，也不能分别估计clip与调度的因果贡献。11 月 21 日已按共同协议完成真实数据比较；P0/P1总loss不同，不据此判优。

## 用户要求清理重启

原125330队列已停止，本次P0 run及日志/登记和启动文件已清理；原P1未启动。删除账本为restart_cleanup.json。日志现仅P1显示正常参考loss（batch/epoch/validation及权重提示），P0省略。计算和配置不变，4项CPU/双卡P0/P1检查通过。

新队列：p0_official_stable_20261117_130401 → p1_official_stable_20261117_130401，于2026-11-17T13:04:01.184759+08:00启动，启动核查时 P0 已进入训练，P1 在等待。当前源码快照和命令已重新生成，首批核实见startup_verification.json。

启动核实（2026-11-17T13:08:09.616201+08:00）：新P0已完成16批更新，实际日志不含Normal Reference Loss，梯度防护和诊断生效；P1显示行为由CPU/双卡测试验证，正式 P1 当时仍在等待。旧run与对应registry已清理，历史实验保留。

## 运行目录清理

2026-11-21 核对后，`runs/` 共保留 10 个目录。稳定 P0 保留 best（19）和最后一轮（26），稳定 P1 保留 best（23）、首次事件前一轮（0）和最后一轮（30）。删除稳定 P0 的 epoch 25 和稳定 P1 的 epoch 29，合计 905,447,692 字节。原 padding 对照、固定学习率 P1 事故及重放权重、诊断现场和正式逐序列结果继续保留。

退役的完整输入 P1（`p1_official_dp_20261115_005859`）和中断的混合评测（`p0_official_stable_20261117_130401_eval_20261120_s200_r700`）已移出 runs。服务器审计包 `retired_runs_20261121.tar.gz` 保存两目录的 36 份文件，移除前逐文件核对 SHA-256。旧 registry 记录保留用于追溯。需要复查合成序列 31454 时，可从包内恢复混合评测目录至 `runs/` 供原探针读取。中断评测不计入结果表。
