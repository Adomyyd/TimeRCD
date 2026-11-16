# 新 P1 实施与验证（2026-11-15）

> 维护注（2026-11-16）：本页保留当时事实；部分临时清单/测试日志已清理，长控制台记录改为 `.log.gz`，映射见上级 `CLEANUP_20261116.md` 与 `cleanup_20261116.json`。

本次按用户指令停止旧完整输入 P1，保留其 run、配置快照、日志和权重；退出当前 P2。P0 保留为已完成基线，不因这次修改要求重跑。

## 实现

- P1 标识 p1_mask_x_normal，输入与 P0 一样遮挡。原观测重构头和异常分类头保留，新增参数独立、结构相同的正常参考重构头。
- 同一次骨干 H、同一选中有效集合 M∩V 分别预测 X/X^0；三项损失权重均1，正常参考权重为 loss.normal_reference_weight。
- X^0/Y 仅作监督；沿用 X 的 batch 标准化统计。推理完整输入，DetectionHead 只执行骨干与分类头。
- 新增头在 P0 初始化后保存/恢复 CPU RNG 构造，共享参数初值一致；后续新增 Dropout 不承诺训练轨迹逐位相同。
- 日志区分观测重构、正常参考和分类；验证新增正常参考区域/实际变化误差、计数与复制基线。P1 valid_total 含正常参考项，不能横比两组总loss。
- P1普通轮末保存完整RNG/best/stale及模型/AdamW，支持统一入口轮末恢复到新run。

## 显存核查与修正

两个全长重构头最初在 batch32、长度10000 的双卡合成前向发生 OOM（在正常头 Dropout 分配约626MiB时失败，已分配约10.07GiB）。因此新P1训练对两个重构MLP使用 non-reentrant activation checkpoint，保留Dropout RNG，反向重算头部。骨干仍一次前向，未改变batch、采样、损失、标准化、FP32或P0路径；验证/推理不重算。

修正后同一上限规模完成前向、反向、AdamW更新，梯度有限，峰值十进制8.855/8.507GB。记录见full_length_preflight.json。重算增加计算，不能当作提速或方法增益。

## 验证

最终109 passed、无失败，33.61秒，包含真实双卡、P0回归、新P1监督位置/梯度、共享初始化、便携权重、检测不执行重构头、P1连续/恢复等价，以及激活检查点与普通计算的输出/梯度/Dropout RNG精确一致。完整日志tests_final.log。

首次106通过/1失败是批量重命名误改了“拒绝旧变体”测试的输入，已修正；随后108项通过。因增加显存优化及相应数值回归，最终为109项，不把早期失败当通过。

文档13份的命令路径与README链接已核对，无机器型号/本机目录表述。当前只有P0/P1，历史日期审计/实验记录中旧方案作为历史保留。代码差异见code_changes.patch，主体文档变化见document_changes.patch，最终摘要/逐文件身份见verification.json。

## 清理与运行

删除四份旧 P1/P2 源配置，清理 full-input/旧重构目标分流，测试只执行当前两变体并保留旧变体拒绝检查。源配置保留 P0/P1 各 ignore/official 共四份及独立eval协议。回归依赖的官方来源、数据、旧run、日志和权重未删。before.zip仅用于生成精确差异，生成后已删除；本轮临时脚本在结束前删除。

正式训练通过唯一run_experiment.py入口，从头初始化，official padding、双卡、batch32、length_bucket50、seed72、split seed42、patch16/遮挡16/0.15、FP32 AdamW5e-4/wd1e-5，最多50轮、patience7，日志100步。新run与tmux身份及启动状态以launch.json、正式run config/status/train.log为准。只授权训练，不自动执行完整检测评测；P0诊断保持暂缓。

启动核验：新run p1_mask_normal_official_dp_20261115_015137 已进入真实400k训练，核实时epoch0、batch200/11875；观测重构=0.2662、正常参考=0.4006、异常分类=0.2409、总loss=0.9076，三项相加一致。双卡DataParallel/总batch32/seed72/official/masked_x与配置快照一致。训练继续运行，尚未完成，不代表检测结果；查看会话用 tmux attach -t p1_mask_normal。
