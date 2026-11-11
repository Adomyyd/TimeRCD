# v9相对原始TimeRCD的实现差异

更新时间：2026-11-11。用于核实当前代码，不将历史补丁按时间堆叠成互相冲突的协议。原始根training.py未因v9改写；400k原流程继续调用它。模型、训练设置和监督需分别核对，P0不是官方全流程复制。

## 1. 结构与输入/目标

复用time_rcd/_core/ts_encoder_bi_bias.py的TimeSeriesEncoder；重构头d→4d→4d→1，异常头d→floor(d/2)→2，隐藏层GELU/Dropout与原结构一致；logits按通道平均。当前num_features1，双向RoPE，patch投影后输出逐时间特征。

| 组别 | 相对原遮蔽流程 |
|---|---|
| P0 | M处X置零、V&~M、M上重构X；保持原训练任务形态 |
| P1 | 取消输入置零及attention遮蔽，完整X/V；仍仅同规则M上重构X |
| P2 | P1相同前向，重构目标改为X^0 |

共同forward只接收x/encoder_mask并返回h/reconstruction/logits。P0/P1/P2在train.make_batch_inputs、evaluate._make_eval_inputs及loss目标处分流。X^0/Y仅监督；没有B/Z、偏离头、基元、真实标签门控或额外时序聚合。

## 2. patch/遮挡来源

官方历史训练入口thu-sail-lab/Time-RCD@ebf4f056b571f7ea90602ef666ae56f7d717ebfe/training.py及本地400k均config=default_config后原地改patch16；当前单卡、workers0时原mask函数也读到16。不能用类默认4推断运行值。v9显式分为model.patch_size=16与input.mask_patch_size=16，mask_ratio0.15。

原函数以ceil(valid_length/patch_size)块数选floor(P×ratio)块，范围裁到有效长度，M内X置零；v9保留该采样语义，训练采样使用独立显式种子。编码器把点可见性聚合为patch内任一点可见即token可见，只屏蔽全不可见token的Key/Value，Queries保留，不删除token。

官方当前main库默认4/噪声替换与历史训练入口不同；spawn重导入全局默认4也是另一条路径，不等同当前400k或v9 DataParallel。论文附录D.5写patch16，正文mask15%，没有独立mask4依据。证据和回归来源见audits/patch_20261110/，其中sources仍是测试依赖。

## 3. 分类padding独立开关

src/losses.py的classification_padding默认ignore：先用valid及labels>=0筛选再二值化；official复现先(labels>0.5).long()再labels!=-1，使padding=-1成为正常0，全部B×T参与CE。三份默认配置显式ignore，三份*_padding_official.json仅该字段不同；旧配置缺省ignore会写入生效快照。

开关不改模型、输入、attention、补齐、重构项、初始化或优化器。train/smoke/eval传递模式，日志、检查点、验证和结果记录模式及分母；audit协议字段随模式变化，避免official仍误报已过滤padding。

两模式共同按valid_total早停/选best，保留目标loss供诊断；ignore时与原total一致。检测指标始终排除padding。该选择协议用于隔离训练padding作用，与原脚本直接按其验证目标loss选best的行为须区分。

## 4. 数据/随机性与评价

| 项目 | 原400k流程 | v9 |
|---|---|---|
| 训练顺序 | DistributedSampler每轮shuffle | 默认length_bucket按长度分池，每轮重排池及池内；可选sequential固定split后顺序 |
| 总batch | 16 | 源配置32；CLI --batch-size可覆盖 |
| 验证分批 | 原验证loader | 固定split后顺序、不分桶；使用同一总batch配置 |
| 标准化 | batch观测X统计，配对共用 | 保持全batch观测统计、配对共用 |
| 采样RNG | 原全局随机源 | 训练按seed/epoch/batch显式生成器 |
| 验证M | 随全局随机流变化 | evaluation.validation_mask_seed固定 |
| padding CE | 官方行为 | ignore默认/official可选 |
| 最佳选择 | 原目标验证loss | 共同valid_total |
| eval | 原比较脚本可做外部TSB-AD | 当前仅合成语料随机验证侧、分区域误差/复制基线与点级指标 |

分桶由src/data.py的LengthBucketBatchSampler及src/train.py的_build_train_loader实现；池大小为batch_size×bucket_multiplier，当前32×50=1600，每轮覆盖一次并保留尾批。380000条训练数据对应11875步。批组成会改变全batch标准化、损失加权和SGD顺序；训练padding减少也改变padding实验的效应范围。

因此历史400k与P0的loss差异不能只归因于padding或mask大小。不同目标的总loss也不能直接说明P2优于P1。分组split、外部VUS、密度分层、合法上下文干预与多种子统计尚未在v9完整实现/执行。

## 5. 可选并行、日志与运行管理

src/parallel.py实现单进程DataParallel，入口支持--parallel/--gpu-ids；原400k走根training.py的DDP路径（当前单进程world_size1）。v9只切batch维，全batch预处理/采样，汇总预测后计算loss，总batch和优化器步数不随卡数增加。首逻辑GPU为主卡。无DDP/FSDP/多机实现；已有默认规模短程测速汇总，未完成整轮吞吐验证；见audits/bench_20261111/bench_summary.json及RUNTIME_NOTES中的限制，单多卡不承诺逐位一致。

v9检查点保存未封装完整state_dict、优化器、epoch、验证和config；支持单多卡eval加载，但当前没有训练resume接口。400k入口支持本run模型/优化器恢复，也不是完整RNG/早停恢复。

v9统一入口负责audit/smoke/bench/train/eval/report/all与独立run登记；all实际为audit→smoke→train→best eval，不自动report。默认每100batch输出分项loss和分类点数，epoch记录分项/验证。尚未加入逐batch尖峰触发保存，不能承诺周期日志捕获所有异常。

## 6. 核实位置与整理边界

src/{models,sampling,losses,train,evaluate,parallel,audit,config}.py分别承担上述职责；README给出可执行命令，EXPERIMENT_SPEC约束工程行为。audits/patch_20261110、parallel_20261110、padding_20261110保留来源、差异、哈希与XML证据。

2026-11-10整理只修正audit元数据及其断言、归档旧备份并移除冗余入口/缓存。当时未改变训练数值逻辑；之后用户新增分桶、测速和总batch32，当前以本文件及源代码为准。2026-11-11本次只更新文档，不修改代码、配置或原始产物。原migrations、padding的before备份和一次性停止脚本移入`experiments/_archive/20261110_cleanup/`；run、registry、数据、权重保留原位。

## 7. 配置字段与未实现能力

入口另支持--batch-size（在实际config快照前覆盖总batch）和bench专用--max-batches（默认300）。bench读取完整训练数据并执行反向及优化器更新，不验证、不保存训练检查点，也不是smoke；实现限制见RUNTIME_NOTES。

runtime.mixed_precision虽存在于JSON，v9未读取该字段，也没有autocast/GradScaler，不能靠设为true启用AMP。evaluation.threshold_source只是协议记录，当前eval使用logits argmax，没有验证集阈值搜索。schema_version只要求字段存在，未校验具体版本。src/config.py中的MODES未含bench且未被入口使用，实际模式以run_experiment.py的argparse choices为准。当前没有预训练权重初始化、训练resume、B/Z或P3–P5入口；方案中的这些部分仍是后续计划。
