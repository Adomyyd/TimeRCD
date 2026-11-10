# Time-RCD

> 迁移整理注（2026-11-10）：本页已统一路径表示与环境表述，实验数值和结论保留。旧清单中的文档摘要对应整理前版本；历史命令不作为当前用法，当前入口见两路线README。

Time-RCD 是一个用于时间序列异常检测的零样本基础模型项目，支持单变量和多变量序列推理，并保留训练、合成数据生成和 TSB-AD 评测代码。

## 当前状态（2026-11-09）

- `time_rcd.TimeRCDDetector` 是当前面向用户的零样本推理 API。
- `best_model/` 中保留单变量和多变量官方初始化权重，可直接进行本地推理。
- 当前科研路线以根目录三份 v9 文档为准：`研究问题和思路.md`、`方法设计.md`、`实验记录.md`。
- v9 的 P0/P1/P2 代码已实现，但尚未训练或测评；最终 B/Z 方案仍是后续计划。不要从历史记录自动启动新实验。
- 既有原流程 400k 训练的运行事实和产物记录在 `实验记录.md`，不等同于 v9 新方案结果。

## 快速开始

建议使用 Python 3.10：

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
```

在 Windows PowerShell 中可使用：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install .
```

### 使用项目内权重

```python
import numpy as np
from time_rcd import TimeRCDDetector

data = np.load("series.npy")  # (T,) 或 (T, 1)
detector = TimeRCDDetector.from_project(variant="uni")
scores = detector.predict(data)
print(scores.shape)  # (T,)
```

多变量序列的形状必须是 `(T, C)` 且 `C > 1`，并使用 `variant="multi"`：

```python
detector = TimeRCDDetector.from_project(variant="multi")
scores = detector.predict(multivariate_data)
```

`scores` 为每个时间点一个 `[0, 1]` 区间内的异常概率。模型不会替用户固定阈值；阈值应使用验证数据或已知正常数据校准。

### 从 Hugging Face 加载权重

```python
detector = TimeRCDDetector.from_pretrained(variant="uni")
scores = detector.predict(data)
```

离线或指定权重文件时，可以使用：

```python
detector = TimeRCDDetector.from_local(
    "best_model/pretrain_checkpoint_best_uni.pth",
    variant="uni",
)
```

`from_project()` 默认查找 `best_model/`，也支持通过 `TIMERCD_WEIGHTS_DIR` 指定外部权重目录。详细参数和 CSV 示例见 `Tutorial.md`。

## 常用参数

```python
detector = TimeRCDDetector.from_pretrained(
    variant="uni",
    win_size=5000,
    batch_size=64,
    device="cuda",  # 或 "cpu"
)
scores, logits = detector.predict(data, return_logits=True)
```

- `win_size`：滑动窗口长度，默认 `5000`。
- `batch_size`：推理批大小，单变量默认 `64`，多变量默认 `1`。
- `device`：未指定时自动使用 CUDA（可用时）或 CPU。
- `return_logits=True`：同时返回原始异常 logits。

## 项目结构

- `time_rcd/`：公开推理 API 和推理后端。
- `best_model/`：单变量、多变量预训练权重。
- `training.py`、`model_wrapper.py`、`models/`：训练和模型实现。
- `dataset_gen/`：合成时间序列异常数据生成器。
- `evaluation/`：指标和可视化工具。
- `experiments/`：当前实验工作区，保存实验脚本、数据、配置、日志和运行产物；入口说明见 `experiments/README.md`。
- `examples/quickstart.py`：最小推理示例。

## 内部协作资料

以下文件主要用于 Codex 和项目协作者获取研究上下文、执行约束与状态，不是面向普通用户的使用文档：

1. `AGENTS.md`
2. `研究问题和思路.md`
3. `方法设计.md`
4. `实验记录.md`

这些文档记录当前问题定义、方法边界、数据和权重约束、已核实实验事实及后续推进条件。具体实验文件和运行记录以 `experiments/` 为准；协作者执行实验前应先读取这些约束，未经登记不修改数据生成配置、不替换官方权重、不启动耗时训练或评测。

## 跨机器使用

从仓库根目录执行文档命令，使用当前机器的Python环境。实验迁移步骤见[迁移说明](../../../MIGRATION.md)。数据与权重不随Git仓库分发，需单独准备。

## 许可证

本项目采用 Apache License 2.0，详见 `LICENSE`。
