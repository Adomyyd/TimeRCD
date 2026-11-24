# v9 与原始 TimeRCD 的差异

v9 的 P0 用作内部基线。新 P1 在 `src/models.py` 增加独立正常参考头，在 `src/losses.py` 增加同一有效遮挡位置上的配对正常 MSE。观测遮挡重构、异常分类与检测打分路径照旧。P0/P1 共享参数初值相同，P1 多出的 Dropout 会改变训练随机轨迹。P2 复用 P1 的模型，在 `src/losses.py` 仅筛选异常编码器 patch 内的有效遮挡点计算正常参考 MSE。模型细节见[方法设计](../../方法设计.md)。

| 方面 | 所审计的官方训练 | v9 |
|---|---|---|
| 编码器与原两头 | TimeSeriesEncoder、观测重构 MLP、异常分类 MLP | 编码器源码一致，原头结构和损失公式一致。P1 训练重构头启用激活检查点。 |
| patch 与训练目标 | 历史入口实际 patch 16、遮挡块 16、比例 0.15。输出在标准化目标坐标上直接求误差。 | 同样使用 16/16/0.15，两个 patch 参数独立。P1 正常目标沿用观测的 batch 统计量。 |
| 原头初始化 | 分类头先于重构头 | 重构头先于分类头。相同种子下编码器一致，5 个头部权重张量不同。 |
| 采样与并行 | batch 64、DistributedSampler、DDP、全局 mask RNG | 总 batch 32、长度分桶、单进程 DataParallel、独立 mask RNG。原 400k 本地路线另用 batch 16。 |
| 分类 padding | 先二值化标签，padding 进入正常类 | `official` 保留该行为；`ignore` 排除分类 padding。注意力与重构两模式均排除 padding。 |
| 验证与优化 | 含 padding 的验证损失选权重，固定 AdamW 学习率 | 各组按自身 `valid_total` 选 best。四份历史配置固定学习率，稳定配对另加裁剪、预热和余弦衰减。 |
| 运行与评测 | 原流程脚本和官方检测协议 | 统一入口，按 run 保存快照、日志、权重及诊断。检测固定官方协议，支持单独评测真实数据或同时评测合成留出与真实数据。 |

官方编码器字节比对与对齐权重、输入、mask、RNG 的 CPU 单步检查均已有证据。对齐条件下 P0 的原两项 loss、梯度及一次 AdamW 更新逐位一致。结论适用于受控单步。来源见归档包 `experiments/_archive/audits_history_20261110_20261121.tar.gz` 内的 `p1_failure_20261116/OFFICIAL_PARITY_20261117.md` 与 `patch_20261110/REPORT.md`。

运行、恢复与评测细节见[运行说明](RUNTIME_NOTES.md)。旧 P1/P2 方案只用于历史追溯，见归档包内 `p1_20261115/REPORT.md`。
