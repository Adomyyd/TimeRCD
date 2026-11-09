# v9 消融实验工程规范

版本：v1。更新时间：2026-11-09。

本文只约束 v9 新消融实验的代码、配置、运行和产物组织，不改变现有 400k 原流程脚本、数据、权重或运行结果。研究问题、方法边界和阶段推进条件仍以根目录的 `研究问题和思路.md`、`方法设计.md`、`实验记录.md` 为准。

## 1. 总原则

1. 所有新消融代码、配置、测试和运行登记均放在 `/diskA/YYD/TimeRCD/experiments/ablation_v9/`。
2. 新消融实验只有一个执行入口：`ablation_v9/run_experiment.py`。训练、评估、审计和报告都通过该入口调用。
3. P0/P1/P2 必须由配置区分，不复制三套近似训练脚本。配置中显式记录输入路径、目标、采样规则、模型结构、损失、优化器、种子和预算。
4. 运行产物写入独立的 run 目录，不覆盖已有数据、权重、日志或检查点。已完成的 run 目录视为不可变证据。
5. 代码、配置、数据 manifest、运行日志和指标必须能够互相追溯。任何结果都不能只存在于终端输出或临时文件中。
6. 未通过预检、接口检查和小批量梯度检查，不得启动正式训练。

## 2. 目录规划

目标目录如下：

```
experiments/
├── README.md
├── ablation_v9/
│   ├── README.md
│   ├── run_experiment.py          # 唯一执行入口
│   ├── configs/                   # 版本化实验配置，只放 JSON/YAML
│   │   ├── p0_mask_x.json
│   │   ├── p1_full_x.json
│   │   └── p2_full_x0.json
│   ├── src/                      # 可复用实现
│   │   ├── __init__.py
│   │   ├── config.py              # 配置读取、校验、展开
│   │   ├── data.py                # 数据读取、配对检查、split
│   │   ├── sampling.py            # 有效位置和 M 采样
│   │   ├── models.py              # P0/P1/P2 共享模型及头
│   │   ├── losses.py              # 重构和分类损失
│   │   ├── train.py               # 单个 run 的训练循环
│   │   ├── evaluate.py            # 统一评估协议
│   │   ├── audit.py               # 数据、配置、模型和产物预检
│   │   └── reporting.py           # 指标汇总、表格和图
│   ├── tests/                     # 不依赖长训练的快速检查
│   │   ├── test_config_contract.py
│   │   ├── test_data_contract.py
│   │   ├── test_sampling_contract.py
│   │   ├── test_model_shapes.py
│   │   └── test_loss_contract.py
│   ├── registry/
│   │   └── experiments.jsonl      # 每个正式 run 的登记和状态
│   └── runs/                      # 运行产物；不放源代码
│       └── <run_id>/
│           ├── config.json        # 实际使用的配置快照
│           ├── command.txt        # 完整执行命令
│           ├── environment.json   # Python、PyTorch、CUDA、GPU 等
│           ├── data_manifest.json # 数据路径、哈希、样本和长度摘要
│           ├── audit.json         # 预检结果
│           ├── train.log
│           ├── checkpoints/
│           ├── metrics/
│           ├── figures/
│           └── status.json
```

现有的 `experiments/train_400k.py`、`generate_400k.py`、`compare_400k.py` 和 400k 数据、日志、检查点继续留在原位置，属于既有原流程资产，不迁移、不改写、不由新入口隐式调用。

## 3. 文件职责

### 3.1 配置

`configs/*.json` 是实验协议的唯一来源。每个配置至少包含：

- `experiment_id`：`p0_mask_x`、`p1_full_x` 或 `p2_full_x0`；
- 数据路径、数据 manifest 路径及预期哈希；
- 数据划分、划分种子和事件隔离规则；
- `input_mode`、`target`、`mask_ratio`、`mask_unit`；
- 模型和头结构、dropout、参数初始化策略；
- 损失项及权重；
- 优化器、学习率、batch size、训练步数、早停和验证频率；
- 随机种子集合、设备和混合精度设置；
- 评估指标、阈值来源和输出目录规则。

配置读取后必须做严格校验。未知字段、缺失字段、冲突字段和不允许的 P0/P1/P2 组合直接报错，不能静默使用默认值。

### 3.2 数据与采样

`data.py` 只负责读取和验证 `X`、配对正常值 `X^0`、时间异常标签 `Y`、有效标志 `V` 及划分，不负责模型逻辑。

`sampling.py` 负责生成有效 patch 位置标志 `M`，并记录每批正常、异常和 padding 位置数量。P0 的输入遮蔽和 P1/P2 的损失位置选择必须使用明确不同的接口，避免把 `M` 误传给模型。

P2 的硬约束：

- 完整 `X` 进入 Transformer；
- `M` 只选择正常参考损失位置；
- 不清零输入，不把 `M` 作为注意力键值标志；
- `X^0` 和 `Y` 只能作为监督；
- 正常和异常位置都可能被抽样。

### 3.3 模型与损失

`models.py` 只定义模型结构和前向输出，不在模型内部读取标签或决定损失位置。

`losses.py` 只根据前向输出、监督张量和有效标志计算损失。分母为零时必须显式跳过对应损失并记录，不能产生 NaN。

P0/P1/P2 的原重构头和异常头结构、输出维度、通道聚合方式必须与方法设计中的协议一致。不得在实现中偷偷加入 B/Z、偏离头、额外掩码或新的时序聚合。

### 3.4 训练、评估和报告

`train.py` 只处理一个已解析配置下的训练生命周期；它不创建另一套实验协议，也不自行改变数据划分、损失权重或保存规则。

`evaluate.py` 统一报告：

- 正常区域和异常区域的正常参考误差；
- 复制观测 `X` 作为正常预测的基线误差；
- 选中位置与全有效区域误差；
- 异常检测指标和正常误报指标；
- 按局部异常占比、事件长度和事件频率分层的结果；
- 每个 run 的样本数、有效点数和抽样覆盖率。

`reporting.py` 只读取已完成 run 的结构化产物生成汇总，不重新训练、不修改原始指标。

## 4. 统一入口

入口命令形式固定为：

```bash
python /diskA/YYD/TimeRCD/experiments/ablation_v9/run_experiment.py \
  --config /diskA/YYD/TimeRCD/experiments/ablation_v9/configs/p2_full_x0.json \
  --run-id p2_full_x0_seed072 \
  --mode audit
```

允许的模式：

- `audit`：检查数据、配置、路径、模型形状、有效标志和对照协议；
- `smoke`：用小批量完成一次前向、损失和反向传播；
- `train`：仅在 audit 和 smoke 通过后训练；
- `eval`：加载指定不可变检查点并按统一协议评估；
- `report`：汇总已完成 run；
- `all`：按 audit → smoke → train → eval → report 顺序执行，任何一步失败立即停止。

入口必须：

1. 解析并冻结配置；
2. 创建不可重复使用的 run 目录；
3. 保存命令、环境、数据 manifest 和代码版本；
4. 登记 `registry/experiments.jsonl`；
5. 对失败、终止和完成状态写入 `status.json`；
6. 禁止覆盖已有 run 目录，除非显式使用新的 run id。

不允许直接运行 `src/train.py`、在 notebook 中启动正式训练，或通过未登记的临时参数改变实验协议。

## 5. 命名和运行规则

- 实验变体使用 `p0_mask_x`、`p1_full_x`、`p2_full_x0` 等稳定名称。
- run id 至少包含变体和种子，例如 `p2_full_x0_seed072`；重复运行必须增加明确后缀。
- 代码文件使用小写下划线命名，类名使用 PascalCase，配置字段使用小写下划线。
- 一个 run 只对应一个配置快照、一个数据 manifest 和一个检查点选择过程。
- 训练数据、官方权重和已有检查点只读使用；新代码不得覆盖它们。
- 大数据集不复制进 run 目录，只保存绝对路径、文件大小和哈希。
- 日志中必须记录随机种子、有效样本数、损失分母、学习率、验证指标和检查点路径。

## 6. 实现顺序

在写模型代码前按以下顺序完成：

1. 创建目录和三个配置草案；
2. 实现配置校验和数据 manifest 读取；
3. 实现 P0/P1/P2 的输入、目标和 `M` 语义检查；
4. 实现最小模型前向、损失和梯度 smoke test；
5. 实现统一入口的 `audit`、`smoke` 和产物登记；
6. 先做小规模接口检查；
7. 完成实验记录登记后，才允许正式训练；
8. 训练和评估完成后，再由 `report` 生成比较结果。

本规范本身不授权启动训练、生成数据、修改官方权重或替换既有产物。
