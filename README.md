# Time-RCD

Time-RCD 是一个用于时间序列异常检测的项目，包含零样本推理、模型训练和 TSB-AD 评测。

## 项目结构

```text
Time-RCD/
├── time_rcd/                  # 推理 API 与核心模型
│   ├── detector.py            # TimeRCDDetector 推理接口
│   ├── _inference.py           # 窗口处理与推理逻辑
│   └── _core/                  # 模型、编码器、配置和数据集
├── models/TimeRCD.py           # TSB-AD 评测适配
├── training.py                 # 预训练入口
├── main.py                     # TSB-AD 评测入口
├── model_wrapper.py            # 评测模型封装
├── HP_list.py                  # 评测超参数
├── evaluation/                 # 评测指标与可视化
├── utils/                      # 数据、滑窗及训练工具
├── dataset_gen/                # 合成训练数据生成
├── best_model/                 # 本地权重目录（已在 .gitignore 中忽略）
├── examples/quickstart.py      # 推理示例
└── Tutorial.md                 # 推理使用说明
```

## 环境

需要 Python 3.10 或更高版本。

```bash
conda create -n time-rcd python=3.10 -y
conda activate time-rcd
pip install -e .
```

评测和合成数据生成还需要额外依赖：

```bash
pip install pandas scikit-learn scipy statsmodels PyWavelets networkx matplotlib seaborn tqdm "huggingface_hub[cli]"
```

## 使用场景

### 1. 零样本推理

权重默认放在项目内的 `best_model/`，不会写入 Hugging Face 缓存。输入可以是一维序列 `(T,)`，或多变量序列 `(T, C)`。

```python
import numpy as np
from time_rcd import TimeRCDDetector

data = np.load("my_series.npy")
detector = TimeRCDDetector.from_project(variant="uni")  # 多变量序列使用 "multi"
scores = detector.predict(data)  # 每个时间点的异常分数
```

`from_project()` 依次查找：`weights_dir` 参数、`TIMERCD_WEIGHTS_DIR` 环境变量、`<repo>/best_model`。本地文件不存在时会回落到 `from_pretrained()` 下载并缓存；传入 `fallback_to_hub=False` 可改为直接报错。

手动下载权重：

```bash
mkdir -p best_model
curl -L -o best_model/pretrain_checkpoint_best_uni.pth \
  https://huggingface.co/thu-sail-lab/Time-RCD/resolve/main/best_model/pretrain_checkpoint_best_uni.pth
curl -L -o best_model/pretrain_checkpoint_best_multi.pth \
  https://huggingface.co/thu-sail-lab/Time-RCD/resolve/main/best_model/pretrain_checkpoint_best_multi.pth
```

也可以运行示例：

```bash
python examples/quickstart.py
```

### 2. 训练模型

训练入口是 `training.py`，数据放在 `data/pretrain/`，权重输出到 `checkpoints/`。

#### 2.1 生成合成数据

`dataset_gen` 没有命令行参数，需要在 `dataset_gen/` 目录下用 `PYTHONPATH=src` 调用（配置里的 `config/synthetic.json` 是相对路径）：

```bash
pip install -r dataset_gen/requirements.txt
cd dataset_gen

PYTHONPATH=src python - <<'PY'
import pickle, os
from generate_dataset import generate_dataset

ds = generate_dataset(
    num_samples=20000,
    seq_len=None,          # None 表示 1000–10000 之间随机
    anomaly_sample_ratio=0.5,
    is_multivariate=True,
    num_features=5,
    use_attribute_set=True,   # 推荐的生成模式
    num_workers=8,
)
os.makedirs("../data/pretrain", exist_ok=True)
with open("../data/pretrain/dataset_5.pkl", "wb") as f:
    pickle.dump(ds, f)
PY
```

每个样本是 `{'time_series', 'normal_time_series', 'labels', 'attribute'}` 四个字段，正好对应训练时的 `ChatTSAnomalyPretrainDataset`。

#### 2.2 文件名规则

`training.py` 里文件名是硬编码的，必须对得上：

| 模式 | 位置 | 规则 |
|---|---|---|
| `single` | `training.py` 的 `single_dataset_filename` | 现为 `dataset_1000000_samples_Nonelen_1.0ratio_univariate_attr_set.pkl`，改成自己的文件名 |
| `multi` | `training.py` 的 `dataset_filenames` 列表 | 文件名最后一个 `_` 后的数字会被当作变量数，例如 `dataset_5.pkl` → 5 个变量 |

`single` 模式固定 `num_features=1`（`training.py` 中写死）；训练多变量请用 `--mode multi`，或自己改 `config.ts_config.num_features`。

#### 2.3 启动训练

```bash
# 单数据集
python training.py --mode single --gpus 0

# 多数据集顺序训练（权重在数据集之间延续）
python training.py --mode multi --gpus 0,1

# 从最新 checkpoint 续训
python training.py --mode single --gpus 0 --resume auto
```

主要超参在 `training.py` 的 `main()` 开头（50 epoch、lr 5e-4、batch 64、`checkpoint_dir="checkpoints/"`）。训练产物：

```text
checkpoints/
├── pretrain_checkpoint_latest.pth
├── pretrain_checkpoint_best.pth
├── pretrain_checkpoint_epoch_{N}.pth
└── pretrained_ts_encoder.pth        # multi 模式按数据集再分一层子目录
```

训练好的权重可以直接用 `TimeRCDDetector.from_local("checkpoints/pretrain_checkpoint_best.pth", variant="multi")` 推理。

### 3. TSB-AD 评测

评测入口是 `main.py`，数据目录固定为 `datasets/TSB-AD-U/`（单变量）或 `datasets/TSB-AD-M/`（多变量），权重从项目根目录的 `best_model/` 读取。

CSV 文件名需符合 TSB-AD 命名，`main.py` 用 `split('_')[-3]` 取训练段长度：

```text
datasets/TSB-AD-U/001_Synthetic_tr_2000_1st_2020-01-01-00-00-00.csv
                               ^^^^ train 长度 = 2000
```

前 N 行作训练段，其余作测试段；最后一列必须是 `Label`。

```bash
# 单变量（HP_list 中 win_size=7000）
python main.py --AD_Name Time_RCD

# 多变量（win_size=5000, batch_size=1）
python main.py --AD_Name Time_RCD --mode multi

# 指定窗口做对比试验
python main.py --AD_Name Time_RCD --test_window_sizes 1000,3000,5000
```

输出：

```text
Uni_Time_RCD_v4.csv            # 指标汇总（AUC-PR/ROC、VUS、F1、PA-F1、Affiliation、F1_T）
Uni_Time_RCD_v4_logits.csv     # logits 指标
Uni_Time_RCD_v4/<数据集>_results.pkl   # value / label / anomaly_score / logits
Multi_..._v4*                  # 多变量同理
```

`main.py` 顶部把 `CUDA_VISIBLE_DEVICES` 写死为 `"0"`，换卡需要改这一行。`filter_list` 用来跳过特定数据集，目录下没有匹配文件时会打印提示并退出。
