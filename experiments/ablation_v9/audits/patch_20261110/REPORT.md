# patch / 遮挡全链路核查（2026-11-10）

> 迁移整理注（2026-11-10）：本页已统一路径表示与环境表述，实验数值和结论保留。旧清单中的文档摘要对应整理前版本；历史命令不作为当前用法，当前入口见两路线README。

结论：本项目当前应以官方历史单 GPU 训练入口及现有 400k 单 GPU 复现为基准，统一编码器 patch=16、采样块=16、采样比例=0.15。此前仅依据配置类默认值认定 400k 遮挡块为 4，是错误结论。P0/P1/P2 三份默认 JSON 已由上次误设的 16/4 改回显式 16/16；两个字段保持独立。

## 1. 区分来源与实际执行路径

| 来源或执行路径 | 编码器 patch | 采样块长度 | 抽样比例 | 抽中位置的输入处理 |
|---|---:|---:|---:|---|
| 官方历史 `training.py` 默认单 GPU、workers=0 | 16 | 16 | 0.15 | 置零，传入 V & ~M |
| 当时400k单卡入口 `experiments/train_400k.py`（现为`experiments/original_400k/src/train_400k.py`） | 16 | 16 | 0.15 | 同上 |
| v9 初次 P0 旧实现（从 model.patch_size 采样） | 16 | 16 | 0.15 | 同上；本项并未比 400k 更大 |
| 上次修改后的三份默认配置（本次修改前） | 16 | 4 | 0.15 | P0 置零；P1/P2 完整输入 |
| 本次修正后 P0 | 16 | 16 | 0.15 | 置零，传入 V & ~M，选中处预测 X |
| 本次修正后 P1 | 16 | 16 | 0.15 | 完整 X 和 V，M 仅选损失位置，预测 X |
| 本次修正后 P2 | 16 | 16 | 0.15 | 完整 X 和 V，M 仅选损失位置，预测 X^0 |
| 官方当前 main 库模块单独使用未修改默认配置 | 4 | 4 | 0.15 | 以 0.1×高斯噪声替换，与历史训练入口不同 |
| 官方历史多 GPU `mp.spawn` 路径，workers=0 | 16 | 可能为重新初始化的 4 | 0.15 | 历史训练函数置零；须核查具体进程配置 |

当前 main 库默认值不代表发布权重真实训练超参，也不能据此认定全部“官方代码”只有一种执行行为。编码器构造函数签名还带 `patch_size=32`，但模型构建显式传参，本项目不会使用该回退值。

## 2. 官方代码的可追溯来源

- 官方历史训练源码：[thu-sail-lab/Time-RCD@ebf4f056b571f7ea90602ef666ae56f7d717ebfe/training.py](https://github.com/thu-sail-lab/Time-RCD/blob/ebf4f056b571f7ea90602ef666ae56f7d717ebfe/training.py)。该脚本后来被上游删除；本项目最初合并仓库 README 记载从这个版本恢复。现已独立下载该历史源码核实，不依赖 README 的超参推断。
- 同提交的 `models/time_rcd/time_rcd_config.py` 和 `models/time_rcd/ts_encoder_bi_bias.py` 与服务器当前 `time_rcd/_core/` 对应文件 SHA-256 完全相同，具体哈希见 `source_manifest.json`。
- 官方当前 main 核查版本为 `b031773e323bc0e6c9dcd8c9119b800ec4d70f8c`；其中没有根目录 training.py，库文件 `time_rcd/_core/TimeRCD_pretrain_multi.py` 保留另一套噪声替换函数。本项目实际 400k 导入的是根目录 training.py。
- 本地服务器 Git HEAD 为 `1c2d34cc0fb0192490b9006888f394ca26423c88`；这是定位信息，当前工作树修改不由 HEAD 单独代表。源码快照及哈希单独保留。
- `sources/` 保存历史官方训练函数、配置类、服务器当前 training.py、400k 入口及当前官方 main 库快照。它们只作核查证据，不是新的实验入口。

## 3. 为什么默认 4 在当前训练中变成 16

历史官方 training.py 的 main（779、785 行）与当前 400k 入口（60、65 行）均执行：

```python
config = default_config
config.ts_config.patch_size = 16
```

这里没有创建配置副本。`config is default_config`，因此全局对象也变成 16。模型构造读取 `config.ts_config.patch_size`；`create_random_mask()`（历史源码 188–224 行）读取 `default_config.ts_config.patch_size`，同进程内读到的也是 16。400k 直接调用 `train_worker(0, 1, config, DATASET)`，显式 workers=0，collate 和遮挡采样均在同一进程。训练与验证都使用同一 collate_fn。

`spawn` 则会重新导入模块并重建全局 `default_config=TimeRCDConfig()`，而传入子进程的 config 是另一个反序列化对象。因此“传入模型 16、函数全局 4”只在相应进程路径下成立。测试以真实配置类对象传入 spawn 子进程验证 `(16,4)`，没有启动 DDP/GPU 或数据训练。Linux DataLoader fork 通常继承父进程状态；spawn worker 需另核，不能跨路径推断。

## 4. 掩码的大小、抽样和注意力含义

- 时间点位置标志 M 是 `[batch, padded_length]`，采样块长度不等于这个张量的形状。
- 有效长度 L、采样块长 b 时，块数为 `ceil(L/b)`，无放回选择 `floor(0.15 * ceil(L/b))` 个块。最后一块截断至 L，padding 不选中，多通道共用同一 M。
- L=1024、b=16：64 块选 9 块，144 个时间点；b=4：256 块选 38 块，152 个时间点。15% 是块数比例，受取整和尾块影响，并非固定精确 15% 时间点。
- P0 将 M 对应的 X 置零，并传入 V & ~M；P1/P2 保持完整 X，只在同样规则抽出的 M 上计算数值损失。
- 编码器先按 16 点生成 token，再把时间点可见标志按 `mask.view(B, num_patches, 16).sum(-1) > 0` 聚合。只要 token 中还有可见点，其 key/value 就仍可访问；整个 token 无可见点时才屏蔽该 key/value，查询不删除。
- 所以 16/4 也会改变注意力屏蔽行为：抽中一个 4 点块不等于屏蔽整个 16 点 token。这是独立消融，不是当前基线的等价写法。

## 5. 本次修改范围与限制

- 三份默认 JSON 的 `input.mask_patch_size` 从 4 统一改回 16，`model.patch_size=16` 不变。训练、验证继续读取显式独立字段，不依赖全局默认。
- audit.json、smoke.json 和新训练日志补充实际两种 patch 大小及比例。保留每 100 batch 的分项 loss 日志。
- 补充配置、原函数同种子采样/置零、三组训练/验证输入、真实编码器掩码聚合、spawn 行为及默认完整模型 CPU 前向/反向测试。
- 不修改官方 training.py、400k 脚本、生成器、已有 run 的 config/log/checkpoint 或登记状态，不启动或停止任何正式训练。
- 旧 P0 的 16 点采样与 400k 在本项上相同。撤回“P0 遮挡块更大导致 epoch 3 loss 异常”的诊断；没有 batch 级证据，不能据此判断不稳定原因，也不能仅因本项要求重跑。
- P0 仍不是整个原训练过程的逐项复制：v9 修复 padding CE、使用不同的数据重排与独立掩码随机源，并固定验证 M。本次不修改这些因素。patch 一致不保证 loss 轨迹一致，历史总 loss 不可据此直接归因。

## 6. 验证

本地 Windows、Python 3.12、PyTorch 2.15.0+cpu：20 项 pytest 通过，JUnit 结果为 `pytest_local.xml`。三组均使用默认模型尺寸进行 CPU 合成数据前向、loss 和反向，梯度有限；没有加载 400k 数据集。

本报告初次核查时仅完成本地CPU测试，尚未完成远端GPU验证；当时连接命令格式不匹配，不能据此认定Python不可远程执行。后续远端双GPU测试已通过，见并行和padding审计报告。

服务器可复核命令（只运行测试，不启动正式训练）：

```bash
python experiments/ablation_v9/tests/run_checks.py
```

> 当前性说明（2026-11-11）：本文保留该次核查的历史事实，旧batch、顺序、配置准备及资产路径不能直接作为当前协议。当前代码语义见experiments/ablation_v9/RUNTIME_NOTES.md；旧P0和测速run缺失、当前训练启动及文档纠正见根实验记录第12节。原始JSON/diff/XML未改写。
