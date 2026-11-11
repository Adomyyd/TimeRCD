# padding 分类损失配对消融审计（2026-11-10）

> 迁移整理注（2026-11-10）：本页已统一路径表示与环境表述，实验数值和结论保留。旧清单中的文档摘要对应整理前版本；历史命令不作为当前用法，当前入口见两路线README。

## 结果与边界

已为 P0/P1/P2 增加 loss.classification_padding=ignore/official。默认ignore维持原v9行为；official严格复现官方先将标签二值化再过滤-1、使padding参与正常类CE的行为。此开关只控制分类损失，不表示完整复刻官方训练。

三份原配置增加显式ignore，三份新配置为 configs/p0_mask_x_padding_official.json、p1_full_x_padding_official.json、p2_full_x0_padding_official.json。每对配置除该字段外完全相同，原配置其他值均未改变。

两模式同时输出所选目标loss和共同有效点指标；检测指标始终排除padding。早停和最佳模型选择统一使用valid_total（原权重重构项+有效点分类CE，保留既有batch均值汇总），ignore下与修改前total数值相同。classification_loss_points、classification_points、padding_points记录分母差异。日志、快照和检查点可追溯所选模式。

本次没有启动正式训练、加载400k训练数据、修改original_400k或根training.py、干预既有进程及run/权重。没有测得padding影响性能的结论。历史400k包含其他流程差异，只作参考，不作为隔离padding影响的对照。

## 验证

- 本地CPU：69 passed, 6 skipped in 8.96s；6个skip是实际双GPU检查。
- 实验服务器 / time-rcd：75 passed in 9.88s，无跳过。命令：

```bash
python experiments/ablation_v9/tests/run_checks.py --junitxml experiments/ablation_v9/audits/padding_20261110/pytest_remote.xml
```

覆盖：原版CE/梯度严格复现；ignore的padding梯度为0；无padding时两模式loss/梯度相同；全padding/空集合；非法枚举与旧配置默认；三组配置单因素差异；模型初始化、输入、mask和重构项不变；同权重共同验证指标不变；全部六种组合的合成数据微型训练、验证、checkpoint；共同早停/选best；统一入口smoke快照；两模式各P组真实双GPU汇总loss/梯度、优化器更新、单样本尾批次及dropout smoke。

未测默认512维模型的双GPU正式训练吞吐与峰值显存。多卡与单卡仍不承诺浮点及dropout轨迹逐位一致，对照应保持设备和并行设置相同。

## 修改范围与证据

共16个文件：src/{config,losses,train,evaluate}.py；六份configs；tests/test_padding_contract.py及现有配置/并行测试；README、IMPLEMENTATION_DIFF、EXPERIMENT_SPEC。

run_experiment.py、src/{models,data,sampling,parallel,audit,reporting}.py与实施前快照哈希相同。部署前检查相关远程文件未发生并发修改；部署后16文件逐字节回读一致。

- 实施前快照：原before/现位于experiments/_archive/20261110_cleanup/padding_before/（保留此前已有改动），本审计目录不再有before/。
- changes.diff：相对于上述快照的本次独立diff。
- source_manifest.json：16文件修改前/后SHA256。
- pytest_local.xml / pytest_remote.xml：测试证据。

本地首次全套收集因尚未同步历史patch审计源码而失败，补齐只读依赖后继续；一次新增严格相等检查发现Python浮点求和与张量求和约6e-8差异，已改为沿用张量算术，保证ignore下valid_total与原total相同。最终测试如上，未放宽该相等断言。

> 当前性说明（2026-11-11）：本文保留该次核查的历史事实，旧batch、顺序、配置准备及资产路径不能直接作为当前协议。当前代码语义见experiments/ablation_v9/RUNTIME_NOTES.md；旧P0和测速run缺失、当前训练启动及文档纠正见根实验记录第12节。原始JSON/diff/XML未改写。
