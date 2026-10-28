# Time-RCD

*Towards Foundation Models for Zero-Shot Time Series Anomaly Detection: Leveraging Synthetic Data and Relative Context Discrepancy*

[![arXiv](https://img.shields.io/badge/arXiv-2509.21190-b31b1b.svg)](https://arxiv.org/abs/2509.21190)

本仓库把 Time-RCD 的**推理、训练、TSB-AD 评测、合成数据生成**四部分合并到同一个工程里，代码来自以下上游仓库：

| 来源 | 内容 |
|---|---|
| [`thu-sail-lab/Time-RCD`](https://github.com/thu-sail-lab/Time-RCD) `main` 分支 | 推理 API（`time_rcd/`） |
| 同上 `tsb-ad-integration` 分支 | 评测框架、核心模型、`training.py` |
| [`thu-sail-lab/Datasets-RCD`](https://github.com/thu-sail-lab/Datasets-RCD) | 合成数据生成（`dataset_gen/`） |

在上游基础上做的整理：

- 核心模型代码只保留**一份**（`time_rcd/_core/`），`models/`、`training.py`、`model_wrapper.py` 的 import 全部指向它，不再有 `models/time_rcd/` 的副本。
- `training.py` 在上游仓库中被删除（提交 `609476a`），本仓库从历史版本 `ebf4f05` 恢复，并把硬编码的数据路径改成 `./data/pretrain/`。
- 去掉与本项目无关的 50+ baseline 实现（TimesFM / Chronos / Moirai / granite_tsfm 等），只保留 Time-RCD 与评测指标。需要跑基线对比时请回上游 `tsb-ad-integration` 分支。

## 📁 项目结构

```
Time-RCD/
├── time_rcd/                  # ★ 推理 API + 核心模型（唯一来源）
│   ├── detector.py            #   TimeRCDDetector：from_pretrained / predict
│   ├── _inference.py          #   滑窗切分、标准化、前向推理
│   └── _core/
│       ├── time_rcd_config.py          # 超参
│       ├── ts_encoder_bi_bias.py       # Patch 编码器（RoPE + 变量注意力偏置）
│       ├── TimeRCD_pretrain_multi.py   # 模型本体（编码器 + 重建头 + 异常头）
│       ├── full_reconstruction.py      # 仅重建任务的模型变体
│       └── dataset.py                  # 预训练数据集
├── models/TimeRCD.py          # 评测框架中的 Time-RCD 适配层
├── training.py                # ★ 预训练入口（单/多数据集、DDP、早停、断点续训）
├── main.py                    # ★ TSB-AD 评测入口
├── model_wrapper.py           # 各算法统一封装
├── HP_list.py                 # 超参表
├── evaluation/                # 指标（VUS-PR / affiliation 等）与可视化
├── utils/                     # 滑窗、指标、训练工具
├── dataset_gen/               # ★ 合成数据生成（原 Datasets-RCD）
│   ├── src/generate_dataset.py
│   ├── config/synthetic.json
│   └── README.md
├── examples/quickstart.py
├── Tutorial.md
└── assets/zero-shot.png
```

## ⚙️ 环境

Python ≥ 3.10。

```bash
conda create -n Time-RCD python=3.10 -y
conda activate Time-RCD

# 推理 / 训练
pip install -e .

# 评测 + 数据生成额外依赖
pip install pandas scikit-learn statsmodels matplotlib seaborn tqdm "huggingface_hub[cli]"
```

国内服务器建议先设 Hugging Face 镜像，否则首次下载权重会很慢：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 🚀 一、推理（零样本检测）

权重首次使用时自动从 Hugging Face 下载并缓存。

```python
import numpy as np
from time_rcd import TimeRCDDetector

data = np.load("my_series.npy")                      # (T,) 或 (T, C)

det = TimeRCDDetector.from_pretrained(variant="uni") # 多变量用 "multi"
scores = det.predict(data)                           # (T,) 异常分数，取值 [0, 1]
```

```bash
python examples/quickstart.py
```

更多用法（CSV 读取、本地权重、超参调整）见 [Tutorial.md](Tutorial.md)。

## 🏋️ 二、训练

### 1. 生成合成数据

```bash
cd dataset_gen
pip install -r requirements.txt
python src/generate_dataset.py --config config/synthetic.json
```

输出目录由 `config/synthetic.json` 控制，产物是 `.pkl`，每个样本包含 `time_series` / `normal_time_series` / `labels` / `attribute`。

### 2. 预训练

把生成的 `.pkl` 放到 `data/pretrain/`，然后：

```bash
# 单数据集
python training.py --mode single --gpus 0

# 多数据集（顺序训练，权重接力）
python training.py --mode multi --gpus 0

# 断点续训
python training.py --mode single --gpus 0 --resume auto
```

`training.py` 里有两个常量需要按你的实际情况修改（`main()` 末尾）：

- `single_dataset_filename`：单数据集模式读取的 `.pkl` 文件名
- `dataset_filenames`：多数据集模式的 7 个文件名列表

默认超参：`epochs=50, lr=5e-4, batch_size=64, patch_size=16`，与官方发布权重一致。checkpoint 保存在 `checkpoints/`。

## 📊 三、TSB-AD 评测

```bash
# 单变量
python main.py --AD_Name Time_RCD

# 多变量
python main.py --AD_Name Time_RCD --mode multi
```

数据集从官方下载：

```bash
mkdir -p datasets && cd datasets
wget https://www.thedatum.org/datasets/TSB-AD-U.zip
wget https://www.thedatum.org/datasets/TSB-AD-M.zip
unzip TSB-AD-U.zip && unzip TSB-AD-M.zip && cd ..
```

权重下载（注意：评测代码默认读 `checkpoints/`）：

```bash
hf download thu-sail-lab/Time-RCD --include "best_model/*.pth" --local-dir checkpoints
```

## 🔗 Citation

```bibtex
@misc{lan2025foundationmodelszeroshottime,
      title={Towards Foundation Models for Zero-Shot Time Series Anomaly Detection: Leveraging Synthetic Data and Relative Context Discrepancy},
      author={Tian Lan and Hao Duong Le and Jinbo Li and Wenjun He and Meng Wang and Chenghao Liu and Chen Zhang},
      year={2025},
      eprint={2509.21190},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.21190},
}
```

## License

Apache-2.0，见 [LICENSE](LICENSE)。
