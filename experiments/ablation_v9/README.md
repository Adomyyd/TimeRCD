# P0/P1/P2消融实验

比较输入遮挡与正常参考监督对异常检测的影响，支持padding对照及单卡/多卡训练。

| 组别 | 输入 | 重构目标 |
|---|---|---|
| P0 | 遮挡后的观测序列 | 原观测序列 |
| P1 | 完整观测序列 | 原观测序列 |
| P2 | 完整观测序列 | 配对正常序列 |

默认编码器patch和遮挡块均为16，遮挡/抽样比例15%，总batch为16。`ignore`排除分类padding；`official`复现原版将padding作为正常点计算分类损失的行为。

## 1. 准备配置

以下为Linux shell命令，全部从仓库根目录执行。先激活项目环境，并将已有400k数据及同名manifest放在`experiments/original_400k/data/`；数据不随Git下载。

下面一次生成六份本机配置，仅替换数据位置，保存到`outputs/portable_configs/`。已有配置会报错，避免覆盖自行修改的设置：

```bash
python - <<'PY'
import json
from pathlib import Path

source_dir = Path("experiments/ablation_v9/configs")
output_dir = Path("outputs/portable_configs")
data_dir = Path("experiments/original_400k/data")
stem = "dataset_400000_samples_Nonelen_1.0ratio_univariate_attr_set"
data_path = data_dir / (stem + ".pkl")
manifest_path = data_dir / (stem + ".manifest.json")
assert data_path.is_file() and manifest_path.is_file(), "先准备数据与manifest"
sources = sorted(source_dir.glob("*.json"))
assert len(sources) == 6, "应为P0/P1/P2各两份padding配置"
output_dir.mkdir(parents=True, exist_ok=True)
for source in sources:
    config = json.loads(source.read_text(encoding="utf-8"))
    config["data"]["path"] = data_path.as_posix()
    config["data"]["manifest_path"] = manifest_path.as_posix()
    target = output_dir / source.name
    if target.exists():
        raise FileExistsError(f"已有本机配置，先核对：{target}")
    target.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(target)
PY
```

## 2. 检查能否运行

合成回归不需要400k数据；smoke需要数据文件和manifest就位，但不加载完整数据：

```bash
python experiments/ablation_v9/tests/run_checks.py

CUDA_VISIBLE_DEVICES=0 python experiments/ablation_v9/run_experiment.py \
  --config outputs/portable_configs/p0_mask_x.json \
  --run-id "p0_smoke_$(date +%Y%m%d_%H%M%S)" --mode smoke
```

## 3. 单卡训练P0、P1、P2

以下默认使用GPU0和`ignore`模式，按需分别运行。将`CUDA_VISIBLE_DEVICES=0`改为`1`即可使用GPU1：

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/ablation_v9/run_experiment.py \
  --config outputs/portable_configs/p0_mask_x.json \
  --run-id "p0_ignore_$(date +%Y%m%d_%H%M%S)" --mode train

CUDA_VISIBLE_DEVICES=0 python experiments/ablation_v9/run_experiment.py \
  --config outputs/portable_configs/p1_full_x.json \
  --run-id "p1_ignore_$(date +%Y%m%d_%H%M%S)" --mode train

CUDA_VISIBLE_DEVICES=0 python experiments/ablation_v9/run_experiment.py \
  --config outputs/portable_configs/p2_full_x0.json \
  --run-id "p2_ignore_$(date +%Y%m%d_%H%M%S)" --mode train
```

## 4. 比较padding处理

同一P组分别运行`ignore`和`official`，其他设置保持一致。下面三条是`official`模式，对应上面的三条`ignore`命令：

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/ablation_v9/run_experiment.py \
  --config outputs/portable_configs/p0_mask_x_padding_official.json \
  --run-id "p0_official_$(date +%Y%m%d_%H%M%S)" --mode train

CUDA_VISIBLE_DEVICES=0 python experiments/ablation_v9/run_experiment.py \
  --config outputs/portable_configs/p1_full_x_padding_official.json \
  --run-id "p1_official_$(date +%Y%m%d_%H%M%S)" --mode train

CUDA_VISIBLE_DEVICES=0 python experiments/ablation_v9/run_experiment.py \
  --config outputs/portable_configs/p2_full_x0_padding_official.json \
  --run-id "p2_official_$(date +%Y%m%d_%H%M%S)" --mode train
```

## 5. 双卡训练

以P0为例，使用GPU0和GPU1：

```bash
CUDA_VISIBLE_DEVICES=0,1 python experiments/ablation_v9/run_experiment.py \
  --config outputs/portable_configs/p0_mask_x.json \
  --run-id "p0_ignore_dp_$(date +%Y%m%d_%H%M%S)" \
  --parallel data_parallel --gpu-ids 0 1 --mode train
```

采用DataParallel，总batch仍为16，保存的检查点包含完整模型。GPU编号相对于`CUDA_VISIBLE_DEVICES`；比较实验时保持设备和并行方式一致。

## 输出与常用设置

结果保存到`experiments/ablation_v9/runs/`下对应run-id目录，包括`train.log`、实际配置、检查点和指标。每100个batch打印分项损失；两种padding模式统一按有效点验证总损失`valid_total`选择最佳模型。

需要调整时，修改`outputs/portable_configs/`中的对应JSON：

| 设置 | 字段 |
|---|---|
| 总batch | optimization.batch_size |
| 日志间隔 | optimization.log_freq |
| 训练轮数 | optimization.epochs |
| 遮挡块长度 | input.mask_patch_size |
| 遮挡/抽样比例 | input.mask_ratio |

当前v9不支持断点续训；已有run目录不会被覆盖。检测评价目前限于既有随机验证划分，尚无完整外部VUS评测流程。
