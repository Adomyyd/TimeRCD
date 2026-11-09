# P0/P1/P2 相对原始 TimeRCD 的实现差异

本文用于代码核对。原始文件不修改；新实现全部位于：

`/diskA/YYD/TimeRCD/experiments/ablation_v9/`

## 1. 文件映射

| 新文件 | 对应原始职责 | 变化 |
|---|---|---|
| `src/data.py` | `training.py` 中的 `ChatTSAnomalyPretrainDataset` 和 `collate_fn` | 保留 batch 观测 `X` 统计并用于 `X^0`，改为显式返回 `x`、`x0`、`y`、`valid`；数据文件使用进程内缓存，避免 train/test 重复加载；95/5 划分使用与原始代码相同的 Python `random.shuffle` 规则 |
| `src/sampling.py` | `training.py:create_random_mask` | 将“patch 位置选择”和“输入置零”拆成两个函数；只有 P0 调用输入置零 |
| `src/models.py` | `training.py:TimeSeriesPretrainModel` | 复用同一个 `TimeSeriesEncoder` 和原重构头、异常头结构；暂不加入 B/Z、偏离头或新时序聚合 |
| `src/losses.py` | `training.py:masked_reconstruction_loss`、`anomaly_detection_loss` | 重构目标由配置选择 `X` 或 `X^0`；分类损失按显式 `valid` 和非负标签过滤 |
| `src/train.py` | `training.py:train_epoch` 和入口训练循环 | 三个变体共用同一训练循环，输入模式和目标由配置决定 |
| `src/evaluate.py` | 原始验证逻辑及项目外部评测入口 | 增加正常/异常区域、选中/全有效区域误差、复制基线和分类统计 |
| `run_experiment.py` | 原始 `experiments/train_400k.py` | 唯一入口，负责 audit、smoke、train、eval、report、配置快照和 run 登记 |

## 2. P0/P1/P2 的实际差异

### P0：`p0_mask_x.json`

与原始训练协议对应：

- 选取约 15% 有效 patch；
- 将选中的 `X` 输入值置零；
- 将对应位置从 encoder 的键值有效标志中排除；
- 重构目标是观测 `X`；
- 异常分类使用同一次前向的 `H`。

与原始 400k 入口相比，P0 有两处为消融协议做的统一化：

1. patch size 从配置读取，当前配置为 16，与 `experiments/train_400k.py` 的训练配置一致；原始 `create_random_mask()` 实际读取全局 `default_config` 的默认值 4，因此新 P0 是按已登记协议固定为 16 的基线，不是原脚本的字节级复制；
2. padding 标签通过 `valid & (label >= 0)` 过滤，不把 padding 当正常标签。原始 `training.py` 先将标签二值化，再用 `labels != -1` 判断，导致 padding 可能被当成正常位置。P0 因此是“统一有效位置处理后的原流程基线”，不是原脚本的字节级复制。

### P1：`p1_full_x.json`

只改变输入可见性：

- 输入为完整 `X`；
- 不置零；
- encoder 的键值标志为全部有效位置 `V`；
- 仍只在同样约 15% patch 上重构；
- 重构目标仍为 `X`；
- 分类头、头结构、损失权重和训练预算保持 P0 一致。

P1−P0 用于估计“取消输入值遮蔽和注意力键值屏蔽”的合并影响，不能继续拆成两个独立因果结论。

### P2：`p2_full_x0.json`

只在 P1 基础上改变数值监督目标：

- 输入仍为完整 `X`；
- 不置零，encoder 读取全部有效上下文；
- 仍使用同样的 15% patch 损失位置；
- 重构目标改为配对正常序列 `X^0`；
- 异常分类仍直接读取 `H`，此阶段没有 Z 分支。

P2−P1 用于估计“正常参考监督”的增量。P2 不把 `X^0` 或真实标签作为模型输入。

## 3. 沿用的原始结构

新模型中的以下结构与原始 `training.py` 保持一致：

- `TimeSeriesEncoder`；
- 重构头：`d_proj → 4d_proj → 4d_proj → 1`，两层 GELU/Dropout；
- 异常头：`d_proj → floor(d_proj/2) → 2`，GELU/Dropout；
- 异常 logits 的特征维均值；
- 二分类交叉熵；
- 当前 400k 配置的 patch size、dropout、batch size、学习率、weight decay、epoch 上限和 seed。

新增的不是新的模型结构，而是把 P0/P1/P2 的输入与目标差异显式化，并让它们共用同一套训练和评估代码。

## 4. 运行方式

配置：

- `configs/p0_mask_x.json`
- `configs/p1_full_x.json`
- `configs/p2_full_x0.json`

统一入口：

```bash
python /diskA/YYD/TimeRCD/experiments/ablation_v9/run_experiment.py \
  --config /diskA/YYD/TimeRCD/experiments/ablation_v9/configs/p2_full_x0.json \
  --run-id p2_full_x0_seed072_audit \
  --mode audit
```

`audit` 不读取 37 GiB pickle 数据，只核对配置、manifest 和模型结构。`smoke` 使用合成小批量验证前向、损失和反向传播。`train`、`eval` 和 `all` 才会读取 400k 数据快照。

## 5. 当前没有改变的内容

- 根目录原始 `training.py` 和模型文件；
- 400k 数据快照、manifest、官方权重和已有运行产物；
- 数据生成配置；
- 原始 400k 训练进程；
- 最终 B/Z、偏离重构和异常基元方案。
