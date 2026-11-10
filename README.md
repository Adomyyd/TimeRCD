# Time-RCD

Time-RCD 用于时间序列异常检测，支持单变量和多变量零样本推理。本仓库还提供合成数据生成、原流程400k复现，以及P0/P1/P2消融实验。

## 安装

建议使用Python 3.10和与本机驱动兼容的PyTorch/CUDA。在仓库根目录激活项目环境后安装：

```bash
python -m pip install .
```

## 零样本检测

以下示例从Hugging Face获取单变量权重，并直接处理一条合成序列：

```python
import numpy as np
from time_rcd import TimeRCDDetector

data = np.sin(np.linspace(0, 40, 1000))
detector = TimeRCDDetector.from_pretrained(variant="uni")
scores = detector.predict(data)
print(scores.shape)  # (1000,)
```

已有本地权重时使用：

```python
detector = TimeRCDDetector.from_local(
    "best_model/pretrain_checkpoint_best_uni.pth", variant="uni"
)
scores = detector.predict(data)
```

单变量输入为`(T,)`或`(T, 1)`；多变量输入为`(T, C)`且`C > 1`，对应`variant="multi"`。输出是逐时间点的异常概率，阈值由验证数据或正常数据确定。

## 实验

所有命令均从仓库根目录执行。数据、权重和完整检查点需单独准备。

| 用途 | 使用说明 |
|---|---|
| 原流程400k数据生成、断点恢复、检测对比 | [original_400k](experiments/original_400k/README.md) |
| P0/P1/P2、padding对照、单卡或多卡训练 | [ablation_v9](experiments/ablation_v9/README.md) |
| 实验入口与快速检查 | [experiments](experiments/README.md) |

详细推理参数见[Tutorial](Tutorial.md)，数据生成器用法见[dataset_gen](dataset_gen/README.md)。

## 许可证

Apache License 2.0，详见[LICENSE](LICENSE)。
