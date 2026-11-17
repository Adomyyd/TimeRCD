# 与官方历史训练实现的复核（2026-11-17）

范围：官方thu-sail-lab/Time-RCD提交ebf4f056b571f7ea90602ef666ae56f7d717ebfe的training.py及models/time_rcd源码；不是泛指所有官方版本或发布权重的完整训练履历。本次重新下载该提交源码并核对现存快照。训练保持停止。

## 核实结论

当前P1并非官方完整复现；P0也只是v9内部基线。编码器实现完全相同，但训练随机轨迹、batch、采样、并行、模型选择不同，P1另有新增头/损失和激活重算。

- 编码器time_rcd/_core/ts_encoder_bi_bias.py与官方文件逐字节一致，SHA256为04f07035dc99940db923c3524a479056b601d192ab937cdc6ca00499962f4c4c。注意力、pre-RMSNorm残差、门控MLP、无最终H归一化均来自官方。
- 原观测重构头和分类头的层结构、GELU/Dropout/末层Linear、Xavier normal初始化方法一致。
- 新发现：官方先重新初始化anomaly_head，再reconstruction_head；v9 src/models.py:59反向执行。相同seed72，编码器101个state张量相同，但两个头的5个weight张量全部不同，bias相同。P0/P1共享参数初值则全部一致。不能把“P0/P1初值一致”写成“与官方初值一致”。本次仅记录，不修改初始化顺序，避免悄然改变未来基线。
- 当前模型尺寸512/256、8层8头、dropout0.1与该官方配置一致；patch16、遮挡16、ratio0.15对应历史官方默认单进程路径，不能推广到spawn重建默认对象路径。
- batch观测均值/std+1e-4标准化、遮挡置零/V&~M、选中有效位置MSE、official标签二值化后的padding CE规则一致；具体同批样本和mask不相同。
- AdamW lr5e-4、wd1e-5、默认betas/eps、FP32、无梯度裁剪/学习率调度一致。

## 非等价部分

| 项目 | 官方历史入口 | 当前v9 P0/P1 |
|---|---|---|
|任务|观测重构+分类|P0同任务；P1另加正常参考重构|
|batch|每进程64，默认单卡总64|总32；历史本地400k为16|
|采样|DistributedSampler(shuffle=True)，逐epoch重排|length_bucket，bucket_multiplier50|
|标准化所在批|各进程local batch|总batch先标准化再分卡|
|并行|DDP，含默认单卡路径|DataParallel，汇总输出后算全batch损失|
|训练mask随机源|collate中全局CPU RNG|每batch独立seed+epoch*1000003+batch|
|验证mask|随全局RNG变化|固定validation_mask_seed|
|best/早停指标|重构+含padding的分类CE|valid_total：分类仅有效点，P1再加正常重构|
|初始化随机顺序|分类头→重构头|重构头→分类头|
|激活checkpoint|无|P1两重构头non-reentrant重算，保留RNG|
|数据|脚本指定100万文件名，不证明发布权重实际数据|本地生成40万，训练38万/验证2万|

分桶改变共同标准化统计、padding比例和更新顺序，不是数值等价加速。DDP平均各rank局部均值与DP全局点均值在分母不同时也不必一致。额外正常头Dropout和单独RNG路径进一步改变随机轨迹。

## 本次短测试

直接AST提取官方历史TimeSeriesPretrainModel类，使用当前字节相同编码器；完整默认模型尺寸，CPU合成2条96点序列，包含padding、遮挡、正常/异常标签。
1. 同seed独立构建官方/P0/P1，核实上述初始化差异。
2. 将官方权重映射加载给P0，固定同一输入、mask和训练模式RNG。
3. H、重构/分类/总loss、全部110个可训练参数梯度、前后RNG均逐位相同。
4. 在内存分别进行一次AdamW更新，全部state张量逐位相同。

该测试没有读取/覆盖训练权重、没有启动正式训练。它证明这个受控单batch下P0计算与更新一致，不证明全数据、双卡、长期训练或官方发布权重履历等价。证据official_parity_probe_20261117.json、official_parity_sources_20261117.json。

## 对异常解释的约束

已定位的注意力/门控MLP高增益确实发生在未改动的官方编码器；大输出头结构、无clip等也与官方相同。但当前参数是本地不同训练过程得到的，不能据此声称官方原训练一定同样崩溃，也不能排除batch/采样/初始化/P1目标等差异对进入敏感状态的影响。此前若用“完全一致”描述整个训练，应予纠正。
