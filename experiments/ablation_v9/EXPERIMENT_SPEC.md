# v9 消融工程规范

用户命令见 [README](README.md)，算法见[方法设计](../../方法设计.md)，运行边界见[运行说明](RUNTIME_NOTES.md)。原流程 400k 与 v9 目录并列，每条路线只有一个 `run_experiment.py` 入口。

| 目录或文件 | 职责 |
|---|---|
| `configs/` | P0/P1 各两份历史 padding 配置、三份稳定训练配置、独立检测协议 |
| `src/config.py` | 配置校验、启动数据协议审计 |
| `src/data.py` | 数据读取、划分、标准化、分桶和遮挡 |
| `src/models.py`、`src/losses.py` | 模型、DataParallel、损失和分类 padding 策略 |
| `src/train.py`、`src/diagnostics.py` | 训练、优化控制、轮末恢复及可选观察 |
| `src/evaluate.py` | 训练期间固定验证 |
| `src/benchmark_eval.py`、`src/metric_worker.py` | 正式检测、结果汇总和指标子进程 |
| `tests/`、`audits/` | 回归入口、历史证据和回归引用的源码快照 |
| `runs/`、`registry/` | 运行配置、日志、权重、指标、诊断和登记 |

## 对照约束

- 当前 P0、P1、P2 都是 `masked_x`，观测重构目标 `x`。P1 另外预测 `x0`，两项重构共享有效遮挡位置。P2 复用 P1，仅将正常参考项限制到异常编码器 patch 内的有效遮挡点，按这些点数取均值。历史完整输入方案已退役，旧 run 仍按自己的快照解释。
- 训练共享 400k 数据，固定 95/5 split seed42、运行 seed72、总 batch 32、`length_bucket` 与倍数 50。分桶基于训练子集的局部索引，每轮调用 `set_epoch`，保留尾批，验证固定顺序。`sequential` 可选。对照须统一采样、归一化、模型初始化、设备、并行与预算。
- `model.patch_size` 和 `input.mask_patch_size` 独立，当前均为 16。抽样比例 0.15，限制在有效位置。normal reference 与分类标签只参与目标。
- `official` 先把标签二值化，padding 作为正常类参与分类；`ignore` 先排除无效位置。两者均屏蔽 attention padding，重构只用有效遮挡点。主方法比较固定同一 padding；历史 400k 有其他差异，不能用来隔离 padding。
- 用完整 batch 计算标准化和 mask，再按样本分卡。汇总预测后再算损失，保留真实分母。检查点保存未包装的完整模型。

## 运行与复现

从仓库根目录用源 JSON 执行。入口校验顶层配置和关键字段组合，保存 CLI 覆盖后的快照，拒绝覆盖已有 run。子对象校验尚未覆盖全部未知键。`mixed_precision` 和 `threshold_source` 是记录字段。

- 验证保留模式损失和过滤分类 padding 后的 `valid_total`；后者用于各组选 best/早停。P1、P2 的 `valid_total` 含正常参考项。外部检测比较统一使用正式评测，记录是否使用 `--eval-real-only`、各域样本数及官方 oracle F1。
- 可选诊断保持在 `OPTIONAL DIAGNOSTICS` 块；学习率调度、裁剪和非有限拒绝属于优化路径。报告短测、受控重放及正式轨迹时须标明范围。
- 未完成背景分组隔离、密度和事件分层、上下文干预、多种子及容量匹配。保留用于回归的三处历史源码。证据和当前测试入口见[审计索引](audits/README.md)与[测试说明](tests/README.md)。
