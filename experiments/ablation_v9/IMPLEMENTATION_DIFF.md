# v9与原始TimeRCD的差异（Agent）

更新：2026-11-15。当前仅P0/P1。原流程original_400k与v9并列；P0是v9内部基线，不宣称官方全流程逐项复制。历史官方源码回归依赖保留在audits/patch_20261110/sources。

## 模型与损失

| 项目 | P0 | 新P1 |
|---|---|---|
| 输入/attention | X选中块置零，V&~M | 与P0相同 |
| 骨干/原头 | 原TimeSeriesEncoder、观测重构MLP和异常分类MLP | 全部保留，新增独立正常参考MLP |
| 重构目标 | M∩V上预测X | 同集合分别预测X和X^0 |
| 首轮loss | L_x+L_cls | L_x+L_n+L_cls |
| 检测推理 | 完整X，异常头概率/logit | 相同，不执行重构头 |

src/models.py正常参考头结构复用原d→4d→4d→1，两隐藏层GELU/Dropout、输出无界Linear，参数独立。新增头在共享参数初始化后构造，保存/恢复CPU RNG，保证同种子原参数初值一致；训练额外Dropout不保证历史随机轨迹一致。forward只接收观测和encoder_mask，同一H供三头，不读X^0/Y，无B/Z/偏差头/双视图。

src/losses.py保留reconstruction为观测误差，新增normal_reference字段及normal_reference_weight；三项按配置加权，空选中集合显式零。config.py只接受p0_mask_x/p1_mask_x_normal；删除旧full-input与normal-only目标分流，旧P1/P2标识拒绝。删除四份旧源配置，保留四份P0/P1×padding配置及独立eval协议；不修改旧run快照。

## 与原官方训练的共同点及工程差异

原历史入口同进程patch/遮挡实际16/16、ratio0.15，非初始默认4；证据见patch审计。重构只在选中位置MSE，分类同次遮挡前向。模型patch线性投影、双向注意力、时间投影和原两头结构保留。官方训练输入标准化但不重新标准化输出；没有新增clip/学习率调度/AMP/尖峰跳过，历史不稳定根因未解决。

official复现标签先二值化使padding成为正常点参与CE；ignore先过滤padding再CE。当前主线official，attention和重构仍排除padding。同组两配置仅该字段不同。原训练DistributedSampler/全局随机源与v9仍有差异，历史400k不能单独隔离padding影响。

v9随机95/5 seed42、运行seed72、分桶batch32、bucket_multiplier50；原400k历史batch16及顺序/步数不同。分桶改变同批统计、padding与更新顺序，不只是加速。训练X/X^0共同使用观测batch统计，推理沿用官方逐序列标准化。背景分组与生成器配对语义未完全审计。

## 日志、选择与状态

训练每100batch报告总loss、观测重构、正常参考和异常分类，P1验证额外报告分区域/实际变化及复制基线；P0原输出和参数键保持兼容，正常参考日志为0。各组按自身valid_total选best和patience7早停，P1多含L_n，数值不可跨组直接判断方法优劣，需共同检测协议与实际预算。

单进程DataParallel分batch、gather后全batch损失，完整未封装checkpoint；与官方多进程路径非逐项一致。统一入口负责run身份、快照、status/registry、audit与失败传播。新P1普通轮末默认保存完整RNG/best/stale；P0保留诊断/续跑时完整状态的历史语义。src/resume只做轮末恢复与保留策略，无训练循环；不把源检查点当新组预训练初始化。

可选诊断独立、默认关闭，不改变损失/梯度或增加采样/前后向；P1新增正常参考loss/预测/误差观测。P0历史事故缺RNG，不保证重现；用户已暂缓。

## 评测与历史

benchmark_eval/metric_worker对接固定官方d458549及TSB-AD-U，额外统一合成留出评价。F1 oracle、两种分数、逐序列/组宏平均及NaN规则见RUNTIME_NOTES；不改变公式来加速。进程优化让持有大pickle的父进程复用小CPU worker，相关source是数值回归依赖，不能清理。两个P0配对评测已完成，RESULTS保留原数值。

旧full-input P1已停止并保留run，不计为新P1；旧P2配置和未使用分流退出当前实现。新代码、退役配置与文档差异/检查/清理证据在audits/p1_20261115。旧日期审计代表历史，不回填新语义、不沿用旧测试数充当本次验证。


显存执行说明（2026-11-15）：最长10000点/总batch32的双卡合成检查发现两个完整重构头保留激活会OOM。新P1训练模式对两个重构MLP启用non-reentrant激活检查点，preserve_rng_state=True，反向时重算头部；骨干仍只前向一次，不改变M/目标/权重/batch/FP32，P0执行路径不变。验证/检测不启用重算。CPU检查覆盖输出/梯度/Dropout RNG与普通头计算一致；最长序列完整前后向及AdamW更新已通过，峰值约8.86/8.51GB（十进制）。重算增加计算开销，不将其当作数值等价提速。
