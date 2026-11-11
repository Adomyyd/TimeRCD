# 分桶采样代码核查（2026-11-11）

受众：Agent。用户要求核查新增分桶是否可行、逻辑是否正确。本次只读审查训练实现、执行CPU合成测试并保留核查记录，没有修改训练代码/配置、读取完整400k数据或启动正式训练/真实数据测速。

## 结论与实现映射

分桶方向可行，核心覆盖与索引映射正确。src/data.py:111的LengthBucketBatchSampler按长度排序，以batch_size * bucket_multiplier划池，每轮随机池顺序和池内顺序，尾项携带，最后保留尾批。src/train.py:97使用训练子集的局部位置作索引，按train_indices映射长度；训练每轮调用set_epoch(epoch)，并非固定同一批次。

六份configs均为batch_size=32、batch_sampler=length_bucket、bucket_multiplier=50。每池1600条；380000条训练样本每轮11875步（原batch16为23750步）。六配置的采样相关设置一致；改变总batch后，同epoch不等于与历史训练同更新步数。模型patch16、mask块16和ratio0.15未因分桶改变。训练/验证划分先完成，分桶只作用于训练集；evaluate.py仍按固定顺序验证，保持既有验证分组及mask生成协议。

## 需要处理或披露的事项

1. 分桶不是数值等价的纯性能优化。collate_samples按同批X全部有效点计算均值和标准差，共同应用于X/X0；批次组成变化会改变标准化坐标、按batch平均损失的权重以及SGD顺序。P0/P1/P2及padding配对必须保持相同采样器、batch、种子和预算。没有发现X/X0/Y错配；不能直接把新旧loss差异归因于方法。
2. padding对照范围改变。读取既有runs/bench_single_b16/metrics/bench.json：batch16、150批（20批预热，130批计时）、padding_fraction=0.0026、s_per_batch=0.1612、samples_per_s=99.25、epoch_estimate_hours=1.063。这是既有产物，非本次实测。分桶使训练padding减少，official/ignore差异也会缩小；如要研究历史大量padding的影响，应另用sequential且固定其他条件做配对，或登记sampler×padding交互。不能用接近无padding时的差异小，证明历史padding无影响。
3. 测速短数据失败（已复现）。benchmark的warmup只受max_batches约束，未受len(train_loader)约束。实际batch不足20时timing_start不设，timed_batches=0，却生成成功结果，速度/耗时/整轮估计为0。应按实际执行批数裁剪预热，且零计时批次明确失败。
4. 测速计时边界和计数。计时从首个计时批的collate/搬运/mask之后开始；达到limit判断在取出下一批之后，可能多取一批；samples_per_s按timed_batches*配置batch计算，尾批不足时高估。应明确是否包含完整数据准备，用实际样本数并精确限定迭代。_sync_devices同步所有可见GPU，而peak reset只重置当前卡，多卡/同进程多次测速也需限定参与设备并逐卡重置。
5. 前若干批未必代表整轮。池顺序随机，但一个池的50批连续输出，150/300批只覆盖少数长度区间；长度相关开销不同，不能把现有整轮外推当成已证实的整轮时间。尚无同设备同batch的sequential对照，不能定量宣称加速倍数。验证不分桶，因此训练加速也不等于包含验证/存盘的整轮同比加速。
6. 文档未同步。README、RUNTIME_NOTES、方法设计及实验记录仍有当前默认batch16的旧表述；README未说明分桶配置、bench模式和--batch-size覆盖。src/config.py中的MODES未含bench（目前未被入口使用，入口自身choices已含bench）。outputs/portable_configs若早已生成，不会随源JSON自动更新，正式命令应检查实际run/config.json。已有测试修改及备份目录均保留，本次不擅自清理他人新增内容。

当前建议：保留分桶用于新的方法比较，并固定三P组协议；padding问题按明确的采样协议另作配对。先修正测速边界并同步文档，再据相同设备/总batch的测速比较确定性能收益。无需为本次审查启动正式训练。

## 验证

- test_bucket_sampler.py与test_bench_contract.py：7 passed，3.57秒，CPU。
- 405组组合覆盖：N=1/2/7/15/17/31/33/101/199，batch=1/2/8/16/32，multiplier=1/2/5，epoch=0/1/9；每轮无重复/无遗漏、步数ceil(N/B)、尾批正确。
- 用非连续乱序原数据索引[8,2,11,5,1,7,0]接入真实_build_train_loader、PairedTimeSeriesDataset和collate_samples，两轮核对attribute、长度及完整覆盖通过。
- 仅9条训练样本、batch3、max_batches300、默认warmup20，复现timed_batches=0与s_per_batch=0却返回结果。
- 未重跑默认规模GPU训练、完整400k统计或真实吞吐对照，不能从合成padding测试推断实际数据整轮加速。
- 临时核查目录在结束前删除；审查没有改写既有run、registry或原始测速产物。

> 当前性说明（2026-11-11）：本文保留该次核查的历史事实，旧batch、顺序、配置准备及资产路径不能直接作为当前协议。当前代码语义见experiments/ablation_v9/RUNTIME_NOTES.md；旧P0和测速run缺失、当前训练启动及文档纠正见根实验记录第12节。原始JSON/diff/XML未改写。
