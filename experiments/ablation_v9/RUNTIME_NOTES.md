# v9 运行说明

命令见 [README](README.md)，对照协议见[工程规范](EXPERIMENT_SPEC.md)，执行历史见[实验记录](../../实验记录.md)。

## 入口和产物

| 模式 | 实际行为及主要产物 |
|---|---|
| `audit` | 核对数据清单、协议与参数量 |
| `smoke` | 核对资产后用合成 batch 前后向，无需读完整 pickle |
| `bench` | 读取完整数据并更新参数，`--max-batches` 限制更新数，产出 `metrics/bench.json` |
| `train` | 先 audit/smoke，再训练和逐轮验证，产出 `train.log`、`checkpoints/`、`metrics/train.json` |
| `eval` | 读取匹配配置及模型，产出 `eval.log`、协议、清单、逐序列/分组/分域指标及可选分数 |
| `all` | audit → smoke → train → eval |
| `report` | 汇总登记中已有 eval，产出 `metrics/report.json` |

run 自带 `config.json`、命令、环境和状态，入口登记到 `registry/`。训练日志通常在完整数据加载后出现。正常参考损失只在 P1 的启动、batch、轮末和验证日志显示。训练验证固定顺序及遮挡种子，保留区域误差、计数及复制基线。`valid_total` 按 batch 汇总，区域重构误差按点统计。

## 优化、恢复和诊断

- 稳定配置使用 AdamW、weight decay `1e-5`、全局 L2 裁剪 5。每次更新前设学习率。以当前 11875 步/轮计，epoch 0/batch 0 为 `5e-5`，batch 11874 达到 `5e-4`，余下计划按步余弦降至 epoch 49 最后一步 `5e-5`。早停或较小的 `--epochs` 可提前结束，调度仍按固定 50 轮计划计算。四份历史配置保留固定学习率路径。
- `--resume-from` 从轮末完整状态恢复到新 run，包含模型、AdamW、RNG、best/stale。条件须匹配，允许调整总轮数上限、日志和诊断设置。稳定配置额外核对每轮步数、调度策略与下一次更新序号。新 P1 普通检查点含完整状态；P0 在诊断、续训或优化控制开启时保存完整状态。早期缺 RNG 的模型权重只用于检测或离线检查，历史兼容格式需有同轮独立 RNG 与完整记录。实际 GPU 随机轨迹仍可能分叉。
- `--checkpoint-keep-last N`（N≥2）只作用于新 run，保留 best、最近 N 轮和首次诊断事件前驱并记录删除。默认保留全部逐轮权重。滚动和事件现场用于离线 batch 重放，正式恢复只读轮末检查点。
- 优化顺序为设 LR → 前后向 → 记录原始梯度与 pre-step 证据 → 裁剪 → 写 `optimization.jsonl` → AdamW 更新 → 检查更新后参数及 Adam 状态。优化日志是 step 前记录，完成情况查 `diagnostics/updates.jsonl`。非有限原始 loss/梯度拒绝更新，诊断开启时先保存现场。
- `--diagnostics` 逐批记录分项损失、样本/遮挡、H 范围及梯度。原始范数在设备归约，诊断不改梯度。默认从 epoch 1 起检查有限阈值（loss 10 或梯度 1000）；稳定队列显式从 0 开始。`--diagnostic-detailed` 启用详细统计和升级事件；默认详细间隔 100，滚动默认关闭，打开后保留两槽。稳定配对使用滚动 250、有限升级上限 12。
- `observation_level=light` 的行省去部分统计。详细行记录区域误差、模块梯度、前后参数/Adam 和实际更新。定期激活只抽样，非定期事件保留实际输入、输出、梯度和前后状态，未采集中间激活需离线重放。非有限事件不受有限事件上限约束。诊断频率与控制台 `--log-freq` 独立。

## 正式检测

[固定到 d458549 的官方协议](audits/eval_20261112/REPORT.md)使用单变量 TSB-AD-U 700 条/11 组，按原规则预处理、切分测试段、统计周期、标准化并使用 5000 点非重叠窗口。输入完整观测，检测只调用编码器与分类头。合成用原留出划分，两个分域和 probability/logit 分别汇总。子集参数限制参与评测的条数，完整 pickle 仍需加载。

指标用持续复用的独立 CPU 进程，Linux 内部指标池只从该小进程派生，避免继承训练数据和 GPU。保持原指标与阈值协议。逐序列 NaN 保留，汇总排除非有限值并记录有效数，空汇总在 JSON 中为 `null`。评测暂无中途续跑，正式 F1 采用测试标签 oracle 阈值。来源、成绩及比较范围见 [RESULTS.md](RESULTS.md)。
