# v9运行语义（Agent）

更新：2026-11-15。当前只有P0和新P1；命令见README，算法见根方法设计，源差异见IMPLEMENTATION_DIFF。历史运行与逐次失败/恢复集中于根实验记录及audits，不将历史状态写成当前能力。

## 配置和计算路径

源配置含四份历史P0/P1各ignore/official，新增两份official_stable；另有eval_official.json。P1标识p1_mask_x_normal，两种实验都在make_batch_inputs生成M后将X置零并传V&~M。P0只输出h/reconstruction/logits；P1额外normal_reference。X^0/Y仅作监督，两MSE均M∩V，首轮L=L_x+L_n+L_cls。观测与正常参考使用相同X统计坐标；不重新标准化预测。

P1正常头在P0共享参数初始化后构造并保存/恢复CPU RNG；原共享初值一致，额外Dropout会改变运行随机轨迹。独立从头训练，未加载已完成P0或旧P1。数据380000/20000随机划分seed42，运行seed72，length_bucket50、batch32、11875步/轮，AdamW5e-4/wd1e-5、FP32、50轮/patience7。分桶改变统计与顺序，非数值等价加速。

official先二值化标签，padding成为正常点参与CE；ignore过滤有效非负标签。两种模式attention仍屏蔽padding，重构不计算padding。训练验证固定M与顺序，记录模式total/CE和共同过滤padding的valid_CE。P1 valid_total含L_n，按各自valid_total选best并早停；不同任务组总loss不能直接比较。

## 入口与产物

唯一run_experiment.py入口：audit核对数据manifest/协议和参数量；smoke用随机合成batch前后向，不加载400k；train先audit/smoke再正式训练；bench完整加载pickle并更新参数但不存权重/验证；eval从checkpoint读取并校验训练配置，检测合成留出和真实数据；all为audit→smoke→train→eval；report只整理已有eval列表。--max-batches仅限制bench。

参数覆盖在config快照前生效；run-id唯一，不覆盖目录。按模式写config/command/environment/status、registry/audit、train.log/checkpoints/metrics/train.json、bench.json或eval产物。加载完整pickle约数分钟，数据就绪后打印train.log。默认每100batch分别打印观测重构、正常参考、分类、总loss和耗时；P0正常参考显示0且无该头。

验证保留观测重构各区域误差、P1正常参考selected/full正常/异常/实际变化区域误差、计数与复制基线。变化按X!=X^0，非标签替代。空区域均值0需结合计数解释；重构指标按点统计，total按batch均值，不能混用。

## 并行、状态与诊断

单进程DataParallel按batch分完整序列；标准化/padding/M先于scatter，全batch输出gather后CE/MSE，保存无module.完整模型。设备相对于CUDA_VISIBLE_DEVICES。不同设备/归约/dropout不保证逐位一致，无DDP/FSDP/AMP。

新P1普通轮末默认含training_state（模型/AdamW外的完整RNG、best/stale）；P0在诊断、resume或优化控制开启时记录完整状态。--resume-from只允许轮末到新run，不是预训练权重初始化；条件严格一致，仅epochs/log_freq/diagnostics可变；缺RNG拒绝，r3可兼容同轮独立RNG。--epochs是总轮数上限，早停仍生效。--checkpoint-keep-last N>=2保留新run最近N轮、best及首次事件前驱，删除逐项记账；默认逐轮全留。

诊断默认关闭，train.py OPTIONAL DIAGNOSTICS块可整体移除。--diagnostics每batch记录分项loss（P1含正常参考）、预测/输入/梯度/样本及首次异常更新前现场；不增加前后向/采样/梯度修正。梯度范数与完整H极值/RMS在源设备归约，仅回传标量，详细区域统计在CPU；默认epoch1起total>=10或grad>=1000触发首次保存；有限事件继续，非有限保存后报错。epoch0参考默认报告差异，--diagnostic-stop-on-mismatch才停止。不匹配只能称同配置重跑，不冒充历史事故精确复现。用户暂缓P0诊断，不自动恢复。

## 检测评测

固定官方d458549；完整输入仅调用DetectionHead（骨干+分类头），不执行两个重构头。真实TSB-AD-U 700条/11组，排除表/清理/测试边界/ACF周期/序列标准化/5000点非重叠窗口/尾padding/16指标按官方。合成沿用留出划分，当前初步对照前200条，概率与logit分别记录；合成与真实不混平均。合成留出参与过best，不是独立测试；F1 oracle搜索使用测试标签，不当作部署阈值。

持续复用spawn小CPU指标进程，不继承400k/GPU；Linux内部官方池fork小进程。不得把官方进程池放回大pickle进程。参数/公式不因执行优化改变；manifest记录后端/源码/输入摘要，eval.log拆分准备/推理/指标耗时。逐序列NaN保留、汇总只对有限值取均值并带有效数，全空JSON为null。不支持评测中途resume。性能修复受控结果见audits/eval_20261114/performance，不从短样本外推全量耗时。

两个P0评测已完成，结果见RESULTS与实验记录第22节。official best31、ignore best26；权重已清理，各仅4文件（best/最后/尖峰前驱），不能再引用任意逐轮权重。旧P1已停止，旧P2无正式结果；旧配置快照、run和证据不改写。当前normal-only/full-input变体均不支持，新P1不同于旧P1。

mixed_precision、threshold_source仍是记录占位，没有AMP或部署阈值拟合。背景分组split、密度分层、多种子、合法上下文干预与B/Z/偏差/少样本均未实现或执行，不能将计划写成能力。


显存执行说明（2026-11-15）：最长10000点/总batch32的双卡合成检查发现两个完整重构头保留激活会OOM。新P1训练模式对两个重构MLP启用non-reentrant激活检查点，preserve_rng_state=True，反向时重算头部；骨干仍只前向一次，不改变M/目标/权重/batch/FP32，P0执行路径不变。验证/检测不启用重算。CPU检查覆盖输出/梯度/Dropout RNG与普通头计算一致；最长序列完整前后向及AdamW更新已通过，峰值约8.86/8.51GB（十进制）。重算增加计算开销，不将其当作数值等价提速。


## P1事故重放诊断补充（2026-11-16）

用户授权从新P1 epoch23轮末恢复，仅运行epoch24（总上限--epochs25），保持模型/AdamW/RNG/双卡/batch32/分桶/seed/lr/mask/损失不变。新run与原run隔离，不自动评测或继续epoch25。--diagnostic-detailed启用src/diagnostics.py，普通训练默认关闭；接入均在OPTIONAL DIAGNOSTICS块，不增加forward/backward或修改梯度。

当前采用分频诊断（2026-11-16效率修订）。batches.jsonl每批保留样本ID/长度/mask种子及计数、分项loss、观测逐样本重构误差、输入/目标/输出范围、完整H极值/RMS、各参数及模块总梯度。CUDA梯度范数采用FP64设备归约并批量回传标量；不复制完整梯度到CPU做常规统计。梯度仍是总loss梯度，不是任务分解。

--diagnostic-detail-interval默认100、必须为正；详细统计在batch%interval==0、滚动快照批次或新异常事件执行，包括X/X0目标能量、区域误差/基线、正常参考逐样本误差、分类混淆计数/概率、H均值、输入/mask摘要、参数/Adam前后统计和实际更新。updates.jsonl轻量行只记录step完成及参数/Adam非有限检查；详细行才有update_by_module/state_after。各行observation_level=light或detailed；省略字段不能解释成0。设interval=1可每批详细统计，但设备归约的实现仍为新版。

原p1_epoch24_replay_20261116_153436已加载旧实现，仍每批详细统计；本次修改不会热更新该进程。旧run快照及其日志语义保留。

定期详细批次中，每个副本的骨干各层/投影/输出头Linear与GELU按时间固定步长抽样最多8位置、所有batch行和通道，记录范围/近零比例/极负比例。非定期批次仅保留分类GELU抽样作为饱和哨兵；阈值在反向后触发时无法补采此前各层激活，activations_complete_for_schedule=false，须离线重放现场获取。抽样统计不代表全张量极值；hook在原forward结束后关闭，跳过activation-checkpoint的反向重算与验证。完整H/输出/输入保存于事件现场，可离线做完整激活检查和任务梯度分解。

--diagnostic-snapshot-interval250保留两份滚动pre-step完整现场，原子替换槽位；事件用硬链接固定最近前驱，后续滚动不覆盖固定前驱。--diagnostic-max-events6限制有限事件；按loss/梯度/H尺度阈值跨十倍等级、分类GELU抽样全零新增原因保存升级事件，非有限证据不受有限事件上限限制。first_event始终保留，后续event_NNN；每个事件pre现场含模型、Adam、完整梯度、前向前/反向后RNG、实际输出、batch/mask/ID和配置，post现场含更新后模型/Adam/RNG。有限事件继续，非有限在观测点保存后报错。

滚动和事件快照是离线batch重放证据，不能直接传给仅支持轮末的--resume-from；恢复轮末仍用checkpoints/epoch_XXXX.pth。每批仍检查loss、梯度、H/输出及更新后的参数/Adam非有限值；定期详细统计检查其余抽样激活。有限事件超过上限或相同严重度持续时仅保留逐batch轻量证据，不反复全量拷贝；每次非有限事件不受上限或已见等级阻挡。观察仍增加计算/磁盘时间，不保证历史CUDA轨迹逐位一致；不得为追求确定性临时切换原训练算法。测试涵盖诊断开关轨迹与实际更新重放，正式启动/现场状态以audits/p1_failure_20261116/REPLAY.md和对应run为准。

## 可选优化控制（2026-11-17）

src/train.py被train和bench共同调用。历史配置缺gradient_clip_norm/lr_schedule时无数值干预；smoke仍仅前后向，不能代替优化器集成测试。stable配置：clip5，warmup_epochs1，total_epochs50，min_lr5e-5，start_factor0.1，learning_rate5e-4。0基更新u=e*len(loader)+batch；首步为start_factor*peak，预热最后一步为peak，最后计划更新为min_lr。early stopping可提前结束；--epochs不改变调度horizon。

顺序为设当前lr→forward/backward→测原始梯度→诊断pre-clip现场→全局同系数裁剪→optimization.jsonl→AdamW.step→诊断post-step。原始梯度范数FP64设备归约并与诊断共用；裁剪后再次实测，不额外forward/backward。optimization.jsonl阶段明确为step前，是否完成更新须交叉核对updates.jsonl。非有限原始loss/梯度拒绝step；诊断开启时先保存证据。training_state.optimization_control保存next_update/steps_per_epoch及固定策略，恢复不重置预热。

当前两组诊断从epoch0开始，基本记录每批、详细100、两槽rolling250、有限升级事件12。事件有上限，不承诺保存每次尖峰完整现场；非有限事件不受上限约束。日志每100批，独立于诊断频率。两组fresh/official/DP双卡/batch32/seed72按P0→P1顺序运行；失败停止队列。固定计划和裁剪值为实验初始选择，不代表最佳参数。

## 模块合并映射（2026-11-17）

src从17个Python文件（含__init__）减为10个：sampling→data，parallel→models，optimization/resume→train，diagnostic_detail→diagnostics，audit→config，reporting→benchmark_eval。优化参数validate_optimization归config，其余控制类归train。旧转发文件已删除；入口CLI与run产物格式不因搬迁改变。历史审计源码不改写，回归测试用隔离命名空间解析历史相对导入。历史settings.json的源码文件名/hash仍代表当时布局。新布局以EXPERIMENT_SPEC职责表为准，结构映射证据见audits/stability_20261117/structure_mapping.json。
