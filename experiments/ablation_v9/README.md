# P0 / P1 / P2 消融实验

P0 用遮挡后的观测训练观测重构和异常分类。P1 另在相同遮挡位置预测配对正常值。P2 只在含异常点的编码器 patch 内，用有效遮挡点预测配对正常值。正式检测都用未做训练遮挡的观测和分类头。结果见 [RESULTS.md](RESULTS.md)。

## 准备与检查

在仓库根目录激活项目环境。训练和 smoke 需要把已有 400k 数据及同名 manifest 放到 `experiments/original_400k/data/`。下面的回归测试用合成数据，smoke 核对资产后做合成前后向。

```bash
python -m pip install . pytest -r experiments/ablation_v9/requirements-eval.txt
python experiments/ablation_v9/tests/run_checks.py

CUDA_VISIBLE_DEVICES=0 python experiments/ablation_v9/run_experiment.py \
  --config experiments/ablation_v9/configs/p1_mask_x_normal_official_stable.json \
  --run-id "p1_smoke_$(date +%Y%m%d_%H%M%S)" --mode smoke
```

## 双卡训练

三组共同使用 official padding、总 batch 32、长度分桶和 seed72。梯度裁剪上限为 5，学习率第 1 轮由 `5e-5` 升至 `5e-4`，随后按更新步余弦降至第 50 轮末 `5e-5`，允许早停。按需选择一条命令。P2 正式训练关闭诊断，保留每 100 批的常规损失日志。

```bash
CUDA_VISIBLE_DEVICES=0,1 python -u experiments/ablation_v9/run_experiment.py \
  --config experiments/ablation_v9/configs/p0_mask_x_official_stable.json \
  --run-id "p0_official_stable_$(date +%Y%m%d_%H%M%S)" \
  --parallel data_parallel --gpu-ids 0 1 --batch-size 32 --mode train \
  --diagnostics --diagnostic-detailed --diagnostic-start-epoch 0 \
  --diagnostic-detail-interval 100 --diagnostic-snapshot-interval 250 \
  --diagnostic-max-events 12 --checkpoint-keep-last 2

CUDA_VISIBLE_DEVICES=0,1 python -u experiments/ablation_v9/run_experiment.py \
  --config experiments/ablation_v9/configs/p1_mask_x_normal_official_stable.json \
  --run-id "p1_official_stable_$(date +%Y%m%d_%H%M%S)" \
  --parallel data_parallel --gpu-ids 0 1 --batch-size 32 --mode train \
  --diagnostics --diagnostic-detailed --diagnostic-start-epoch 0 \
  --diagnostic-detail-interval 100 --diagnostic-snapshot-interval 250 \
  --diagnostic-max-events 12 --checkpoint-keep-last 2

CUDA_VISIBLE_DEVICES=0,1 python -u experiments/ablation_v9/run_experiment.py \
  --config experiments/ablation_v9/configs/p2_mask_x_normal_anomaly_patch_official_stable.json \
  --run-id "p2_official_stable_$(date +%Y%m%d_%H%M%S)" \
  --parallel data_parallel --gpu-ids 0 1 --batch-size 32 --mode train --checkpoint-keep-last 2
```

换卡时改 `CUDA_VISIBLE_DEVICES` 中的物理卡编号，`--gpu-ids 0 1` 是两张可见卡的相对编号。产物在 `experiments/ablation_v9/runs/<run-id>/`，已有目录拒绝覆盖。数据加载后写 `train.log`，正常参考损失在 P1 和 P2 显示。命令保留 best、最近两轮，开启诊断时还保留首次事件前驱。梯度及现场字段见[运行说明](RUNTIME_NOTES.md#优化恢复和诊断)。

## 单卡与 padding 对照

以下单卡示例采用历史固定学习率配置。显存不足可加 `--batch-size 16`，配对组同步调整。

```bash
CUDA_VISIBLE_DEVICES=0 python -u experiments/ablation_v9/run_experiment.py \
  --config experiments/ablation_v9/configs/p1_mask_x_normal_padding_official.json \
  --run-id "p1_single_$(date +%Y%m%d_%H%M%S)" --parallel none --mode train
```

`official` 把分类 padding 计为正常点，`ignore` 从分类损失排除。历史配置均保留固定学习率及无裁剪训练方式。把命令的配置换成所需的一份。

| 实验 | ignore | official |
|---|---|---|
| P0 | `p0_mask_x.json` | `p0_mask_x_padding_official.json` |
| P1 | `p1_mask_x_normal.json` | `p1_mask_x_normal_padding_official.json` |

## 评测已有模型

稳定版对照评测 TSB-AD-U 全部 700 条真实序列。准备 `datasets/TSB-AD-U/`、三组训练目录及各自的 best 权重，保留训练配置引用的数据与 manifest。以下命令从仓库根目录依次评测三组。

```bash
v9=experiments/ablation_v9
for trained_run in p0_official_stable_20261117_130401 p1_official_stable_20261117_130401 p2_official_stable_20261121_2320; do
  CUDA_VISIBLE_DEVICES=0,1 python -u "$v9/run_experiment.py" \
    --config "$v9/runs/$trained_run/config.json" \
    --checkpoint "$v9/runs/$trained_run/checkpoints/best.pth" \
    --eval-config "$v9/configs/eval_official.json" \
    --run-id "${trained_run}_eval_real700_$(date +%Y%m%d_%H%M%S)" \
    --parallel data_parallel --gpu-ids 0 1 --eval-real-only --mode eval || break
done
```

历史 padding 对照同时评测 200 条合成留出和全部真实序列。复用该范围时，将 `--eval-real-only` 换成 `--eval-max-synthetic 200`，并相应修改 run 名。

指标写入新 run 的 `metrics/eval.json`、`per_series.csv` 及分组、分域 CSV。合成留出参与选 best，官方 F1 使用测试标签搜索阈值。NaN 保留在逐序列结果中，汇总记录有效数。

## 续训

源进程退出后，从含完整状态的轮末检查点恢复到新 run。把变量改成实际 run 和保留的检查点文件名。示例对应原双卡训练。

```bash
trained_run=p1_official_stable_20261117_130401
checkpoint=epoch_0030.pth
CUDA_VISIBLE_DEVICES=0,1 python -u experiments/ablation_v9/run_experiment.py \
  --config "experiments/ablation_v9/runs/$trained_run/config.json" \
  --resume-from "experiments/ablation_v9/runs/$trained_run/checkpoints/$checkpoint" \
  --run-id "p1_resume_$(date +%Y%m%d_%H%M%S)" --epochs 50 --mode train
```

`--epochs` 是总上限，稳定版恢复原调度位置。当前 run 只保留 best 与末轮检查点，示例用末轮 `epoch_0030.pth`。训练输出、开关细节和 `bench` 等模式见[运行说明](RUNTIME_NOTES.md)。

历史审计正文集中在 `experiments/_archive/audits_history_20261110_20261121.tar.gz`，按包内同名路径引用；当前保留的证据见[审计索引](audits/README.md)。
