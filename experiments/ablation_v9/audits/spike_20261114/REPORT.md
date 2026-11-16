# P0 official epoch1逐batch诊断（2026-11-14）

> 当前资产状态（2026-11-16）：用户已确认删除两份P0诊断run和首份P1 epoch24重放run；下文是历史事实，原run内日志/诊断/权重已不可用，汇总见上级run_cleanup_execution_20261116.json。新P1原训练保留best、epoch23/24/27，仍可作为恢复/前后对比来源。

受众：Agent。用户已授权修改日志并重跑；本次研究目标仅定位历史epoch1的爆炸，不调整训练稳定化策略。

## 代码与边界

- 独立模块：src/diagnostics.py；train.py中所有插入点均以OPTIONAL DIAGNOSTICS标记，默认关闭，可整体移除。
- 统一入口增加--diagnostics、--epochs、--log-freq与诊断参考/阈值参数；有效值在config快照前写入。六份源配置、模型、损失、遮挡与数据处理不改。
- 每batch写diagnostics/batches.jsonl，包括loss/累计均值、原始样本索引/长度、原始均值/标准差、输入/隐藏/预测/logit/遮挡平方误差范围、各样本遮挡MSE、有效/padding/分类点数及总梯度范数。
- 触发默认从epoch1开始：total>=10或gradient_l2>=1000；非有限值全程监测。第一次触发在optimizer.step前保存first_event.pth和摘要，含未更新模型/AdamW、前向前Python/NumPy/CPU/所有CUDA RNG、batch与实际输入/mask、损失及逐参数梯度范数。不会每batch保存权重。
- 有限值触发后继续；非有限值保存后报错，入口标failed。阈值不改loss、梯度、学习率、mask或采样。
- 每轮验证后保存RNG。epoch0模型与AdamW状态逐张量/递归精确比较旧epoch0000；不匹配则抛错停止，不进入epoch1。
- 旧checkpoint没有RNG；即使epoch0参数匹配，仍需比较epoch1轨迹，不能预先宣称历史爆炸已精确复现。
- CUDA同步/统计/写盘会增加时间；小模型数值不变验证不保证所有规模或设备绝无浮点差异，正式epoch0门槛提供进一步检查。

## 验证

- 全套合成回归：100 passed，18.53秒（包含首版4项诊断检查）。
- 额外补充CLI快照/合法性后：5 passed，96 deselected，4.94秒。
- CPU和真实双卡均比较历史源码循环、当前关闭诊断、当前开启诊断的最终模型/AdamW/验证值/RNG，逐项完全一致；只使用微型合成数据。
- 异常现场检查证实保存的是更新前状态；CPU Dropout重放损失和梯度范数一致；第二次事件不覆盖第一次；epoch0不匹配正确阻止继续。
- CLI --help检查通过。不以这些微型测试宣称正式事故已复现。

## 执行计划

新run：p0_official_spike_diag_20261114。
读取runs/p0_official_dp_20261111_003608/config.json，从初始化起保持audit→smoke→train顺序，总batch32、length_bucket、seed72、双卡DataParallel；只覆盖epochs=2与新增诊断配置，控制台log_freq仍100，JSONL每batch一条。参考为同run的checkpoints/epoch_0000.pth。不会重评测，也不会覆盖旧run。

运行状态以后续execution.json、对应run/status.json、日志及进程为准。本节启动计划不代表已完成。

补充静态核验：移除train.py全部5处OPTIONAL DIAGNOSTICS块后，AST与修改前训练源码完全相同；有效新旧配置只差optimization.epochs与新增diagnostics对象。语法及共享文档机器路径检查通过，见verification.json。新作业已启动，过程以execution.json及run为准。

## 启动检查修订：CPU统计

初次p0_official_spike_diag_20261114已在epoch0主动停止：batch0/100匹配周期日志，batch200 total0.5090（旧0.5084）、batch300为0.5654（旧0.5752）。没有epoch0完成检查点，不能确定偏差来源；该试跑保留原逐batch证据并补记aborted。归约统计现将观测tensor/grad复制到CPU计算，不再为日志在CUDA创建归约临时张量。新增正式512维8层/batch32双卡合成对照，模型/AdamW/验证/RNG与历史及关闭诊断完全一致。修改后诊断套件6 passed，96 deselected，19.61秒。先前100通过是修改前全套，不冒充修改后全套结果。

当前新run为p0_official_spike_diag_cpu_20261114，仍使用旧配置、两轮预算和epoch0精确停止门槛。前次差异不等于已确定日志副作用，也不等于已诊断训练爆炸。当前作业及状态见execution.json。

## 最终方案及运行（以本节为准）

第二次CPU Torch归约试跑在epoch0 batch200仍偏离旧日志（total0.5085/recon0.2650/cls0.2435，旧0.5084/0.2643/0.2441），主动停止并补记aborted。此结果不能证明漂移由观测造成；受控对照没有发现数值/RNG改变，但并未覆盖历史完整CUDA轨迹。

最终run为p0_official_spike_diag_r3_20261114。默认核对epoch0后只记录差异并继续两轮，出现差异时报告same_config_rerun_not_exact_replay；严格不匹配即停改为显式--diagnostic-stop-on-mismatch，最终run未启用该项。该调整已在启动前告知用户，目的是完成同配置epoch1诊断，不将精确重放作为未经证实的前提。没有载入/替换旧模型或改变训练计算。

归约统一CPU NumPy，避免CUDA临时分配及Torch CPU线程池额外开销。最终全套102 passed（32.31秒）；补充eval快照复用覆盖后诊断6 passed（18.42秒），含正式模型规模双卡对照。5处可拆卸块去除后AST仍与原训练完全一致。前两次日志保留作启动检查，不算完整实验。当前状态见execution.json，后续应先判断epoch0是否匹配，再区分新轨迹异常与历史事故精确复现。


用户随后授权删除两次无用启动试跑目录：p0_official_spike_diag_20261114、p0_official_spike_diag_cpu_20261114。已核实作业停止和aborted状态后删除；此前关于保留两次run日志的表述为清理前状态，原始日志现已删除。摘要、逐文件清单/哈希、批次数及删除核验保留在experiments/ablation_v9/audits/spike_20261114/startup_runs_cleanup.json；registry追加deleted事件。当前r3目录与进程保留，未修改训练配置或代码。


P0诊断进展核实：r3的epoch0已完成11875批，train total=0.24642795、recon=0.11046466、classification=0.13596329，valid_total=0.17488710（历史epoch0为0.18542509）。模型110项、Adam一阶矩110项、二阶矩110项与旧状态不精确匹配，共330项；因此明确为同配置重跑，不能称历史事故精确重放。epoch0权重及轮末RNG已保存，按当前授权继续epoch1；本次没有更改训练代码或配置。检查时epoch1已记录到batch1356，最大total=0.446252，阈值触发0批、非有限值0批。进度快照见experiments/ablation_v9/audits/spike_20261114/epoch0_progress.json；该数值只覆盖当时已完成batch，不代表epoch1最终结果。


## 延长授权： 用户授权延长P0诊断与轮末续跑（2026-11-14）

用户要求epoch1后继续观察是否出现爆炸。r3内存中的两轮range不会被修改JSON延长，因此让源run正常完成；新增统一入口train --resume-from，准备从epoch1的模型/AdamW和同轮完整RNG恢复到新run p0_official_spike_diag_continue_20261114，绝对epoch2起，总上限恢复原50轮，patience7不变。采样/batch32/DP/seed72/lr/mask/loss及逐batch诊断保持；缺RNG或配置变化拒绝，不把旧P0缺RNG权重当完整续训。

src/resume.py负责校验/恢复/保留策略，tests/test_resume.py验证连续与恢复的最终模型、优化器、RNG、验证/best，包括r3旧载荷兼容；6 passed，8.22秒。全CPU98 passed/10GPU跳过，16.60秒；并发生产训练的全GPU107 passed/1显存不足失败，非数值断言失败，未为此停止r3或改变训练设置。新诊断/续跑检查点增加training_state；普通关闭诊断且未resume的检查点不额外记录RNG。

新run采用--checkpoint-keep-last 2保留best/最新两轮/首次事件前驱及独立现场，删除逐项写checkpoint_retention.jsonl；源run不清理。resume.json记录来源hash/epoch/RNG/best/stale。README已有直接命令，各Agent规范更新。一次性调度仅等待源completed且退出再调用统一入口，无重复训练循环；源失败即取消，不能把排队状态当继续训练。队列作业job-mwgiyjeu-863f，实时状态见audits/spike_20261114/continuation_queue.json，正式日志continuation_console.log；首次等待时尚未续跑。诊断不保证一定出现爆炸，也不自动做完整评测。
