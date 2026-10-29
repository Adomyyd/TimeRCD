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
pip install pandas scikit-learn statsmodels matplotlib seaborn tqdm "huggingface_hub[cli]"
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

先生成合成数据：

```bash
pip install -r dataset_gen/requirements.txt
python dataset_gen/src/generate_dataset.py --config dataset_gen/config/synthetic.json
```

将生成的 `.pkl` 数据集放入 `data/pretrain/`，并按需修改 `training.py` 中的数据集文件名，然后启动训练：

```bash
python training.py --mode single --gpus 0
# 多数据集顺序训练
python training.py --mode multi --gpus 0
# 从最新 checkpoint 续训
python training.py --mode single --gpus 0 --resume auto
```

### 3. TSB-AD 评测

下载并解压 TSB-AD 数据集到 `datasets/TSB-AD-U/` 或 `datasets/TSB-AD-M/`，并将模型权重放到项目根目录的 `best_model/`。运行：

```bash
# 单变量
python main.py --AD_Name Time_RCD
# 多变量
python main.py --AD_Name Time_RCD --mode multi
```

评测结果会保存为当前目录下的 CSV 文件；逐数据集的异常分数会保存为 pickle 文件。
