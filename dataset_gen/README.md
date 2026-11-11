# 合成时序数据生成器

生成单变量或多变量序列，返回观测序列、配对正常序列、时间异常标签和属性。支持属性集合及metric配置两种方式；现有400k实验使用属性集合。

所有命令从仓库根目录执行。安装生成器依赖：

```bash
python -m pip install -r dataset_gen/requirements.txt
```

## 生成实验使用的400k数据

使用统一实验入口；目标已存在时拒绝覆盖，输出约39GB，不需要为已有实验重新生成：

```bash
python experiments/original_400k/run_experiment.py --mode generate
```

数据和manifest写到`experiments/original_400k/data/`。具体用法见[原流程实验](../experiments/original_400k/README.md)。

## 在Python中生成小数据

下面代码从仓库根目录运行。生成器内部使用平级模块导入，并读取相对于工作目录的metric配置，因此先设置模块位置与工作目录。

```python
from pathlib import Path
import os
import pickle
import sys

repo = Path.cwd()
generator_dir = repo / "dataset_gen"
sys.path.insert(0, str(generator_dir / "src"))
os.chdir(generator_dir)
from generate_dataset import generate_dataset

dataset = generate_dataset(
    num_samples=10,
    seq_len=1000,
    anomaly_sample_ratio=1.0,
    is_multivariate=False,
    use_attribute_set=True,
    num_workers=1,
)
output = repo / "experiments" / "original_400k" / "data" / "example_10.pkl"
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("xb") as stream:
    pickle.dump(dataset, stream)
os.chdir(repo)
```

多变量时将`is_multivariate=True, num_features=5`传给同一函数。多进程调用应放在脚本的`if __name__ == "__main__":`中。

| 参数 | 默认值 | 含义 |
|---|---|---|
| num_samples | 1000 | 样本数 |
| seq_len | None | 随机长度1000–10000；指定整数时固定长度 |
| anomaly_sample_ratio | 0.5 | 样本保留异常的概率；不代表异常时间点比例 |
| is_multivariate | False | 是否生成多变量数据 |
| num_features | None | 多变量通道数；None时随机2–50，单变量不靠此参数切换模式 |
| activate_function | False | 多变量依赖中的激活函数开关 |
| metrics | None | metric候选，仅非属性集合路径有效 |
| use_attribute_set | False | True使用ALL_ATTRIBUTE_SET；False使用metric配置 |
| num_workers | 80 | 工作进程数；1为单进程，None按CPU核数 |

每个样本包含`normal_time_series`、`time_series`、`labels`、`attribute`。配对序列共享生成过程及数值变换，不自动等于独立因果反事实；异常样本比例1.0在单变量路径按异常属性非空重试，不表示所有时间点异常，多变量路径不提供相同的强制重试保证。

源码位于`dataset_gen/src/`，metric配置在`dataset_gen/config/synthetic.json`。底层`generate_dataset.py`的直接执行段仍有历史机器绑定输出设置，不作为当前使用入口。

生成器参考并改编自[ChatTS](https://github.com/NetManAIOps/ChatTS)。
