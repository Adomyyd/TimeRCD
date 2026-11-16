# P0 padding配对检测评测

> 维护注（2026-11-16）：本页保留当时事实；部分临时清单/测试日志已清理，长控制台记录改为 `.log.gz`，映射见上级 `CLEANUP_20261116.md` 与 `cleanup_20261116.json`。

受众：Agent。日期：2026-11-14。**当前两个正式评测队列均已按用户要求停止；先完成性能修复，未恢复。未来约定范围仍为每组固定合成200条＋全部真实700条。当前协议见SUBSET200_PROTOCOL.md，状态见execution_subset200.json。下文全量启动/耗时段保留为初始历史，不是当前计划。**

本文记录计划及启动事实，结果以execution.json与各run产物为准，未完成时不作完整检测性能结论。

## 问题与训练来源

固定P0及已有训练协议，比较分类padding参与损失（official）与排除padding（ignore）后的检测表现。两个训练run均已核实completed，配置递归比较仅loss.classification_padding不同。

| padding | 训练run（位于experiments/ablation_v9/runs） | 完成轮次（从0计） | best epoch | best valid_total |
|---|---|---|---:|---:|
| ignore | p0_ignore_dp_20261112_170107 | 0–33，共34轮 | 26 | 0.0811527285337448 |
| official | p0_official_dp_20261111_003608 | 0–38，共39轮 | 31 | 0.07948285787403583 |

两组按相同valid_total早停/选best，但实际轮数不同，不将该比较描述为固定epoch或固定总更新次数实验。ignore本地时间2026-11-14 02:01:27完成；official为2026-11-12 14:31:18（均UTC+8）。验证loss不能替代检测指标或直接证明padding策略优劣。

## 评测协议与执行

统一使用run_experiment.py --mode eval，读取各训练run/config.json及checkpoints/best.pth，参数来自configs/eval_official.json。全部合成留出20000条、官方TSB-AD-U 700条/11组；完整输入、同官方预处理/窗口与16项逐序列指标，概率与logit分别计算；seed2024、推理batch64、DataParallel逻辑0/1。无子集限制，不重新训练、不改训练数据或权重。

先ignore再official顺序执行，每组成功退出后才启动下一组。后台一次性调度与SSH短命令会话分离，实际工作仍通过统一入口，不新增实验算法入口。exec状态、子进程标识、起止时间和退出码见execution.json；各组独立保存控制台日志。

| padding | 正式评测run |
|---|---|
| ignore | p0_ignore_dp_20261112_170107_eval_20261114_full |
| official | p0_official_dp_20261111_003608_eval_20261114_full |

队列于2026-11-14 13:16:57（UTC+8）启动。official未轮到时没有run目录，不能写成已开始。初始不带_full的ignore eval在准备pickle时被SSH 30秒超时终止；已核实无评测进程，仅有准备日志，无逐序列结果，status与registry补记aborted，保留原目录。后台正式运行使用新run-id，没有覆写或恢复该中断run。

## 完整性与解释边界

结束后核对两组status=completed与metrics/eval.json，full_benchmark及official_default_parameters为true；各per_series.csv应有41400行，等于(20000+700)×2类分数；合成索引、真实文件清单、协议与源码摘要保持一致，仅权重来源和training padding不同。每个domain/series_id/score_kind唯一、无缺行或非有限指标；分别展示合成与真实的series_macro及真实11组group_macro，不混成一个总分。

合成留出已用于选best，不是独立测试集；官方F1保留oracle测试标签阈值搜索，不能称部署固定阈值成绩。官方部分指标内部异常返0的既有行为沿用，不把有限值完整性检查等同于内部无异常。两组仅一个训练种子，不能据此给出多种子稳定收益结论。

本次不修改模型、训练、采样、损失或指标代码，也不启动P1/P2；实验记录第17节及RUNTIME_NOTES同步启动状态。未完成或失败时保留已完成行，不把部分数据均值报告为全量成绩。

## 启动后检查快照

2026-11-14 13:23:33（UTC+8）核查：ignore评测进程存在，已过完整pickle准备阶段，进入20000/20000合成及700/700真实的full=True流程；当时逐分数CSV已写13行，official尚在队列中。此前已完成行检查无重复domain/series_id/score_kind组合，16项指标均为有限值；完整性检查仍未完成。

ignore manifest确认checkpoint epoch26、full_benchmark=true、official_default_parameters=true，选中合成20000及真实700，与既定协议相符；参数、源码摘要及早期逐行检查见preflight_and_partial_check.json。first-rows仅为流程验证，不报告其均值为全量性能。

最初7次单类分数指标调用各耗时12.21–13.99秒，需对每条序列调用两次。粗略线性外推，两组仅合成部分约12天，真实评测另计；极早期样本不代表稳定吞吐或可靠完成时间。已向用户提示可选择维持全量或改为共同固定合成子集加全部真实；未收到变更指令前不改变全量队列、指标参数或样本范围。

## 最新状态：停止后先修复（2026-11-14）

用户要求不再继续评测。subset队列13:48:42（UTC+8）停止，ignore保留27行、official未启动，execution_subset200.json为aborted；正式任务未恢复。性能问题已定位并修复，只做受控验证，详见performance/REPORT.md。当前没有完整P0对照。
