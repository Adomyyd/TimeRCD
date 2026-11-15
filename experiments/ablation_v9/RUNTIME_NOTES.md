# v9运行语义（Agent）

更新：2026-11-15。当前只有P0和新P1；命令见README，算法见根方法设计，源差异见IMPLEMENTATION_DIFF。历史运行与逐次失败/恢复集中于根实验记录及audits，不将历史状态写成当前能力。

## 配置和计算路径

四份源配置为P0/P1各ignore/official；另有eval_official.json。P1标识p1_mask_x_normal，两种实验都在make_batch_inputs生成M后将X置零并传V&~M。P0只输出h/reconstruction/logits；P1额外normal_reference。X^0/Y仅作监督，两MSE均M∩V，首轮L=L_x+L_n+L_cls。观测与正常参考使用相同X统计坐标；不重新标准化预测。

P1正常头在P0共享参数初始化后构造并保存/恢复CPU RNG；原共享初值一致，额外Dropout会改变运行随机轨迹。独立从头训练，未加载已完成P0或旧P1。数据380000/20000随机划分seed42，运行seed72，length_bucket50、batch32、11875步/轮，AdamW5e-4/wd1e-5、FP32、50轮/patience7。分桶改变统计与顺序，非数值等价加速。

official先二值化标签，padding成为正常点参与CE；ignore过滤有效非负标签。两种模式attention仍屏蔽padding，重构不计算padding。训练验证固定M与顺序，记录模式total/CE和共同过滤padding的valid_CE。P1 valid_total含L_n，按各自valid_total选best并早停；不同任务组总loss不能直接比较。

## 入口与产物

唯一run_experiment.py入口：audit核对数据manifest/协议和参数量；smoke用随机合成batch前后向，不加载400k；train先audit/smoke再正式训练；bench完整加载pickle并更新参数但不存权重/验证；eval从checkpoint读取并校验训练配置，检测合成留出和真实数据；all为audit→smoke→train→eval；report只整理已有eval列表。--max-batches仅限制bench。

参数覆盖在config快照前生效；run-id唯一，不覆盖目录。按模式写config/command/environment/status、registry/audit、train.log/checkpoints/metrics/train.json、bench.json或eval产物。加载完整pickle约数分钟，数据就绪后打印train.log。默认每100batch分别打印观测重构、正常参考、分类、总loss、点数和耗时；P0正常参考显示0且无该头。

验证保留观测重构各区域误差、P1正常参考selected/full正常/异常/实际变化区域误差、计数与复制基线。变化按X!=X^0，非标签替代。空区域均值0需结合计数解释；重构指标按点统计，total按batch均值，不能混用。

## 并行、状态与诊断

单进程DataParallel按batch分完整序列；标准化/padding/M先于scatter，全batch输出gather后CE/MSE，保存无module.完整模型。设备相对于CUDA_VISIBLE_DEVICES。不同设备/归约/dropout不保证逐位一致，无DDP/FSDP/AMP。

新P1普通轮末默认含training_state（模型/AdamW外的完整RNG、best/stale）；P0仅在诊断或resume时记录完整状态。--resume-from只允许轮末到新run，不是预训练权重初始化；条件严格一致，仅epochs/log_freq/diagnostics可变；缺RNG拒绝，r3可兼容同轮独立RNG。--epochs是总轮数上限，早停仍生效。--checkpoint-keep-last N>=2保留新run最近N轮、best及首次事件前驱，删除逐项记账；默认逐轮全留。

诊断默认关闭，train.py OPTIONAL DIAGNOSTICS块可整体移除。--diagnostics每batch记录分项loss（P1含正常参考）、预测/输入/梯度/样本及首次异常更新前现场；不增加前后向/采样/梯度修正。统计在CPU，默认epoch1起total>=10或grad>=1000触发首次保存；有限事件继续，非有限保存后报错。epoch0参考默认报告差异，--diagnostic-stop-on-mismatch才停止。不匹配只能称同配置重跑，不冒充历史事故精确复现。用户暂缓P0诊断，不自动恢复。

## 检测评测

固定官方d458549；完整输入仅调用DetectionHead（骨干+分类头），不执行两个重构头。真实TSB-AD-U 700条/11组，排除表/清理/测试边界/ACF周期/序列标准化/5000点非重叠窗口/尾padding/16指标按官方。合成沿用留出划分，当前初步对照前200条，概率与logit分别记录；合成与真实不混平均。合成留出参与过best，不是独立测试；F1 oracle搜索使用测试标签，不当作部署阈值。

持续复用spawn小CPU指标进程，不继承400k/GPU；Linux内部官方池fork小进程。不得把官方进程池放回大pickle进程。参数/公式不因执行优化改变；manifest记录后端/源码/输入摘要，eval.log拆分准备/推理/指标耗时。逐序列NaN保留、汇总只对有限值取均值并带有效数，全空JSON为null。不支持评测中途resume。性能修复受控结果见audits/eval_20261114/performance，不从短样本外推全量耗时。

两个P0评测已完成，结果见RESULTS与实验记录第22节。official best31、ignore best26；权重已清理，各仅4文件（best/最后/尖峰前驱），不能再引用任意逐轮权重。旧P1已停止，旧P2无正式结果；旧配置快照、run和证据不改写。当前normal-only/full-input变体均不支持，新P1不同于旧P1。

mixed_precision、threshold_source仍是记录占位，没有AMP或部署阈值拟合。背景分组split、密度分层、多种子、合法上下文干预与B/Z/偏差/少样本均未实现或执行，不能将计划写成能力。


显存执行说明（2026-11-15）：最长10000点/总batch32的双卡合成检查发现两个完整重构头保留激活会OOM。新P1训练模式对两个重构MLP启用non-reentrant激活检查点，preserve_rng_state=True，反向时重算头部；骨干仍只前向一次，不改变M/目标/权重/batch/FP32，P0执行路径不变。验证/检测不启用重算。CPU检查覆盖输出/梯度/Dropout RNG与普通头计算一致；最长序列完整前后向及AdamW更新已通过，峰值约8.86/8.51GB（十进制）。重算增加计算开销，不将其当作数值等价提速。
