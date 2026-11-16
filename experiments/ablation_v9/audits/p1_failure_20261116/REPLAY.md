# P1 epoch24重放与诊断覆盖（2026-11-16）

> 当前资产状态（2026-11-16）：用户已确认删除两份P0诊断run和首份P1 epoch24重放run；下文是历史事实，原run内日志/诊断/权重已不可用，汇总见上级run_cleanup_execution_20261116.json。新P1原训练保留best、epoch23/24/27，仍可作为恢复/前后对比来源。

受众：Agent。用户已停止原训练并明确授权重放和核查诊断。原run为p1_mask_normal_official_dp_20261115_015137；本次新run为p1_epoch24_replay_20261116_153436，tmux会话p1_epoch24_replay。

## 运行边界

从原epoch23轮末恢复完整模型/AdamW/RNG/best/stale，实际配置仅改变总轮数上限25、诊断选项；console日志仍每100批。仅epoch24，共11875批，结束后进行原有训练验证并保存轮末checkpoint，不做官方检测评测，不进入epoch25。双卡DataParallel、总batch32、seed72、length_bucket50、FP32、lr5e-4、patch16/mask16/ratio0.15、三损失权重和official padding保持。

15:34:36已启动新进程，PID1383699，初始加载完整数据；尚未将加载状态当作恢复完成。后续核验见本文件追加记录与replay_startup.json。旧run/best/epoch23权重未改写。CUDA执行仍不保证历史逐位相同。

## 文件职责与实现

- src/diagnostics.py：原诊断入口与兼容语义，逐batch主日志、阈值及现场调度。
- src/diagnostic_detail.py：可选详细观察；hook、CPU统计、更新统计、滚动和升级现场。
- src/train.py：可整体移除的OPTIONAL DIAGNOSTICS接入；主损失/采样/更新语义不变。
- run_experiment.py与src/config.py：详细诊断开关/滚动间隔/事件上限校验。
- tests/test_diagnostic_detail.py：P1 CPU/双卡数值惰性、现场重放、非有限事件、CLI及正式尺寸测试。
- tests/test_padding_contract.py：移除先前删除Classification points后遗留的一条打印字符串断言；算法/padding测试保留。
- README与RUNTIME_NOTES：可执行命令及详细产物语义。

统一入口参数：--diagnostics --diagnostic-detailed --diagnostic-start-epoch24 --diagnostic-loss-threshold1 --diagnostic-grad-threshold1000 --diagnostic-snapshot-interval250 --diagnostic-max-events6（实际命令参数和值之间有空格，见README）。开始前源run已aborted且原PID退出。

## 信息覆盖与限制

| 分析问题 | 每batch证据 | 完整事件/离线分析 |
|---|---|---|
| 是否某条数据/正常目标异常 | 样本ID、长度、X统计、标准化X/X0范围、选中目标均方/零预测基线、正常/异常/实际变化区域误差、逐样本两重构误差 | 完整标准化batch包含X/X0/Y/V/attribute，原始数据依ID读取 |
| 分母/采样是否改变 | mask数量、mask种子、分类点数、valid/padding数、X/X0/Y/V/M SHA256 | 实际mask、encoder_mask、输入与配置 |
| 哪个任务loss先变化 | 三项loss及总loss，H均值/RMS和输出范围，区域误差 | 实际完整H、两重构输出与logits |
| 哪个模块出现异常梯度 | 总梯度、每个参数及每层/每头梯度范数 | 保存全部实际参数梯度；离线可逐任务反向分解 |
| 是否一次更新破坏参数 | 更新前后参数范数、Adam一/二阶矩统计，实际更新绝对/相对范数 | 更新前模型/Adam/梯度及更新后模型/Adam |
| 是否GELU饱和、哪层特征放大 | 每个GPU副本各骨干层/投影/头部Linear与GELU固定位置抽样统计 | 以完整现场离线重放完整激活 |
| 分类是否塌缩 | 有效点TP/FP/FN、异常比例与预测概率分布 | 原始完整logits和Y |
| 能否复查原轨迹/随机性 | 配置、样本与mask摘要、原日志对应loss | 前向前和反向后RNG、实际输出/梯度、更新后RNG、源码哈希 |

每batch的层激活只按时间固定步长取最多8位置，覆盖所有batch行和通道；明确标记形状、步长及设备，不把抽样极值冒充全张量极值。完整H的均值/RMS及输出范围另作全张量统计。hook在前向结束后关闭，避免记录checkpoint反向重算，不增加任何前向/反向或随机采样。

逐模块梯度是三任务总loss的梯度，不能据此分解每项任务对共享骨干的贡献。任务梯度分解可能额外反向，留在离线现场分析，不改变在线训练。异常现场包含实际输出与实际梯度，即使离线CUDA重算存在微差，也能检查原始数值以及使用原梯度重演Adam更新。

## 快照与停止规则

- batches.jsonl：每batch更新前观测；updates.jsonl：同一batch实际更新后观测。匹配epoch和batch。
- rolling_0/1.pth：每250批的pre-step完整现场，交替原子替换，始终只留两槽。每次事件把最近滚动槽硬链接为独立前驱，滚动替换不改变已固定inode。
- first_event.pth保持首次事件；event_NNN.pth记录新原因/跨十倍阈值等级事件，有限事件最多6个。loss>=1、grad>=1000、H RMS>=1000、分类GELU抽样全零可触发。触发不是自动证明该batch为首次根因。
- *_post.pth保存该次实际更新后的模型/Adam/RNG；若非有限值在更新前被发现，不执行step，因此没有对应post。
- 后续非有限事件不受有限事件上限阻挡，也不覆盖first_event；更新后参数或Adam非有限另保存nonfinite_update_post后报错。
- 有限事件保存后继续；非有限保存后失败退出。滚动/事件是离线batch现场，不是统一入口--resume-from支持的轮末checkpoint。

## 验证

第一轮诊断/恢复17项通过。全套116项检查中112通过，4项失败来自同一过期的Classification points打印断言；移除该断言后，padding文件23项复测全部通过，没有遗留失败。详细测试中的正式模型、双卡、总batch32、长度1024和10000，对比开关观测后的完整模型、Adam状态与RNG一致；CPU事件前后梯度/Adam更新可精确重演。不会把受控测试等价性推断为历史8000步轨迹已匹配。

证据：初步17项结果见本节记录（中间replay_tests.log已清理）；replay_tests_final.log（含发现的旧断言失败）、replay_padding_recheck.log、replay_code_changes.patch、replay_launch.json。源码哈希与源checkpoint/best摘要保留在启动JSON。一次性辅助脚本完成后自删。

## 2026-11-16 20:07进展核查：未复现原转折

逐batch读取截至epoch24 batch8070，共8071条完整记录；总loss最大0.5888566971（batch1676），总梯度L2最大11.8550637353（batch793），triggers均为空。原run在batch7900总loss0.3052、8000为5.3098，随后持续退化；本次8000为0.1412、8070为0.1400176287，已平稳经过原日志转折窗口。证据见replay_progress_8000.json。

本次恢复状态不等于历史轨迹精确重演：先前核验batch200总loss已由原0.2539偏为0.2567，远早于原事故区间。不能将本次无异常解释为问题已修复、诊断具有稳定作用或原异常由某条数据必然触发。多卡/CUDA非确定性、随机状态消费及诊断同步对执行的影响均需通过首次分歧对齐核查，尚未确定来源。保留当前已授权的epoch24及轮末验证边界，不自动延长训练或启动第二次重跑。本次仅读取状态并记录证据，没有修改训练代码或进程。
