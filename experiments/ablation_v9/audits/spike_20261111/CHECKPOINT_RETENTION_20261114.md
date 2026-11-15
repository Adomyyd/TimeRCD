# P0 检查点保留与清理（2026-11-14）

受众：Agent。用户要求对两个已完成 P0 训练 run，仅保留最佳、最后一轮及 loss 爆炸前一轮权重，供后续重放分析。本次已执行；没有启动训练或评测。

## 保留结果

epoch 编号均从 0 开始；下表文件位于相应训练 run 的 `checkpoints/`，没有改名。

| 训练 run | 最佳 | 最后一轮 | 尖峰/爆炸前一轮 | 本次删除 |
| --- | --- | --- | --- | ---: |
| `p0_official_dp_20261111_003608` | `best.pth`，内部 epoch31 | `epoch_0038.pth` | `epoch_0000.pth`、`epoch_0016.pth` | 15 |
| `p0_ignore_dp_20261112_170107` | `best.pth`，内部 epoch26 | `epoch_0033.pth` | `epoch_0001.pth`、`epoch_0030.pth` | 31 |

两个目录清理前共54个检查点，现各4个，共8个。删除46个逐轮文件，合计20,461,835,874字节，约19.06 GiB。最佳轮次的同名逐轮副本已删除，完整状态仍在各自 `best.pth` 中。清理仅作用于上述两个目录中已审核的 `epoch_XXXX.pth` 文件；现有训练日志、配置、评测结果、registry、数据、官方权重及其他 run 保留。

## 保留前驱的日志依据

| 训练策略 | 异常 epoch | 整轮平均 train total | 每100 batch采样的最大 total | 对应保留前驱 |
| --- | ---: | ---: | --- | --- |
| official | 1 | 2908.18585937 | batch6200：9.7193（recon9.5175、classification0.2018） | epoch0 |
| official | 17 | 1.52080747 | batch5500：0.8207（recon0.5901、classification0.2305） | epoch16 |
| ignore | 2 | 0.13677572 | batch100：8.4666（recon8.3409、classification0.1257） | epoch1 |
| ignore | 31 | 6.35217318 | batch2200：0.9072（recon0.7164、classification0.1909） | epoch30 |

前三处整轮异常通过 `train_total>=1` 筛出（official的epoch1/17、ignore的epoch31）；ignore epoch2虽无整轮爆炸，但存在显著单batch尖峰，也保留其前驱。补充筛查规则为非初始化epoch中已打印batch total>=2。该阈值仅用于本次保守资产保留，不是异常机制定义。每100batch打印一次，不能据采样最大值还原真实单batch极值、精确定位未打印事故batch，或声称其他轮次绝无尖峰。

## checkpoint 完整性与复现边界

8个保留文件均已用CPU加载检查，各含：

- `epoch`、`config`、`validation`；
- `model_state_dict`，111个模型状态条目；
- `optimizer_state_dict`，110个优化器参数状态及参数组。

各文件内部epoch与保留角色一致，checkpoint配置与训练run配置一致；两份best的SHA-256与已完成评测manifest完全一致。删除前后保留文件SHA-256一致，文件内容没有重存或修改。当前代码及日志显示未使用学习率调度器或AMP，不存在对应遗漏状态的额外推断。

这些旧checkpoint未保存Python、NumPy、Torch CPU或各设备CUDA随机数状态，也未保存显式的复现控制状态。现有训练实现中，分桶顺序由seed/epoch决定，重构mask由seed/epoch/batch决定，可以重建；模型Dropout所需的历史CUDA随机数状态没有落盘。因此：

- 可以从前一轮的模型和AdamW状态开始重放、加诊断，尝试重现下一轮尖峰。
- 不能保证重放过程与历史事故逐步或逐bit一致；当前保存不足的问题无法通过清理后补写一个“当时随机状态”解决。
- v9统一入口当前没有resume能力，本次没有新增恢复入口或修改训练逻辑。真正重放时应恢复模型、优化器、绝对epoch编号、采样与mask规则，并遵循统一入口规范另行实现诊断。
- 未来新训练如要求精确重放，应在保存时同时记录各随机数状态及运行控制状态。不能将此需求写成已有checkpoint已具备的能力。

## 审计与后续使用

- [checkpoint_retention_20261114.json](checkpoint_retention_20261114.json)：清理前全部文件清单、尺寸/修改时间、保留文件摘要与实际载荷键、完整逐轮均值、事件证据和训练源码摘要。
- [checkpoint_cleanup_20261114.json](checkpoint_cleanup_20261114.json)：逐文件删除账本、实际保留文件及校验值、完成状态和删除字节数。
- `sources_cleanup_20261114/`：清理时核对的统一入口、train/data/sampling/losses/models/evaluate/parallel/config及底层encoder源码快照，供后续分析使用；不作为新的训练入口，也不声称它弥补了历史RNG缺失。
- 原有 [REPORT.md](REPORT.md) 是早期尖峰分析，不据此认定根因。

本次一次性操作脚本在成功核验后已自行删除。后续先阅读本清单，勿再假设这些run仍保留所有逐轮权重；特别是同epoch配对分析只能使用实际留存的状态，不能引用已删除文件。
