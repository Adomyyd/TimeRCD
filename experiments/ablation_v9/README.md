# P0/P1/P2消融实验

比较输入遮挡与正常参考监督对异常检测的影响，支持padding对照及单卡/多卡训练。

| 组别 | 输入 | 重构目标 |
|---|---|---|
| P0 | 遮挡后的观测序列 | 原观测序列 |
| P1 | 完整观测序列 | 原观测序列 |
| P2 | 完整观测序列 | 配对正常序列 |

默认编码器patch和遮挡块均为16，遮挡/抽样比例15%，总batch为32，默认启用长度分桶采样。`ignore`排除分类padding；`official`复现原版将padding作为正常点计算分类损失的行为。

## 1. 准备数据

以下为Linux shell命令，全部从仓库根目录执行。先激活项目环境，并将已有400k数据及同名manifest放在`experiments/original_400k/data/`；数据不随Git下载。

直接使用`experiments/ablation_v9/configs/`中的六份配置，无需生成或复制配置。数据路径以仓库根目录为基准。

## 2. 检查能否运行

先安装项目与测试依赖：

```bash
python -m pip install . pytest
```

合成回归不需要400k数据；smoke需要数据文件和manifest就位，但不加载完整数据：

```bash
python experiments/ablation_v9/tests/run_checks.py

CUDA_VISIBLE_DEVICES=0 python experiments/ablation_v9/run_experiment.py \
  --config experiments/ablation_v9/configs/p0_mask_x.json \
  --run-id "p0_smoke_$(date +%Y%m%d_%H%M%S)" --mode smoke
```

## 3. 单卡训练P0、P1、P2

以下单卡示例统一覆盖总batch为16，以降低显存需求；源配置总batch仍为32。默认使用GPU0和`ignore`模式，按需分别运行。将`CUDA_VISIBLE_DEVICES=0`改为`1`即可使用GPU1：

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/ablation_v9/run_experiment.py \
  --config experiments/ablation_v9/configs/p0_mask_x.json \
  --run-id "p0_ignore_$(date +%Y%m%d_%H%M%S)" --batch-size 16 --mode train

CUDA_VISIBLE_DEVICES=0 python experiments/ablation_v9/run_experiment.py \
  --config experiments/ablation_v9/configs/p1_full_x.json \
  --run-id "p1_ignore_$(date +%Y%m%d_%H%M%S)" --batch-size 16 --mode train

CUDA_VISIBLE_DEVICES=0 python experiments/ablation_v9/run_experiment.py \
  --config experiments/ablation_v9/configs/p2_full_x0.json \
  --run-id "p2_ignore_$(date +%Y%m%d_%H%M%S)" --batch-size 16 --mode train
```

## 4. 比较padding处理

同一P组分别运行`ignore`和`official`，其他设置保持一致。下面三条是`official`模式，对应上面的三条`ignore`命令：

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/ablation_v9/run_experiment.py \
  --config experiments/ablation_v9/configs/p0_mask_x_padding_official.json \
  --run-id "p0_official_$(date +%Y%m%d_%H%M%S)" --batch-size 16 --mode train

CUDA_VISIBLE_DEVICES=0 python experiments/ablation_v9/run_experiment.py \
  --config experiments/ablation_v9/configs/p1_full_x_padding_official.json \
  --run-id "p1_official_$(date +%Y%m%d_%H%M%S)" --batch-size 16 --mode train

CUDA_VISIBLE_DEVICES=0 python experiments/ablation_v9/run_experiment.py \
  --config experiments/ablation_v9/configs/p2_full_x0_padding_official.json \
  --run-id "p2_official_$(date +%Y%m%d_%H%M%S)" --batch-size 16 --mode train
```

## 5. 双卡训练

P0的两个padding对照均使用GPU0和GPU1、总batch32；分别运行，避免同时争用两张卡：

```bash
CUDA_VISIBLE_DEVICES=0,1 python experiments/ablation_v9/run_experiment.py \
  --config experiments/ablation_v9/configs/p0_mask_x.json \
  --run-id "p0_ignore_dp_$(date +%Y%m%d_%H%M%S)" \
  --parallel data_parallel --gpu-ids 0 1 --mode train

CUDA_VISIBLE_DEVICES=0,1 python experiments/ablation_v9/run_experiment.py \
  --config experiments/ablation_v9/configs/p0_mask_x_padding_official.json \
  --run-id "p0_official_dp_$(date +%Y%m%d_%H%M%S)" \
  --parallel data_parallel --gpu-ids 0 1 --mode train
```

采用DataParallel，总batch仍为32，保存的检查点包含完整模型。GPU编号相对于`CUDA_VISIBLE_DEVICES`；比较实验时保持设备和并行方式一致。

## 6. 短程测速和已有模型评价

测速会加载完整数据并执行训练更新，不保存模型，也不做验证。以下使用双卡和源配置总batch32，最多300个batch：

```bash
CUDA_VISIBLE_DEVICES=0,1 python experiments/ablation_v9/run_experiment.py \
  --config experiments/ablation_v9/configs/p0_mask_x.json \
  --run-id "p0_bench_dp_$(date +%Y%m%d_%H%M%S)" \
  --parallel data_parallel --gpu-ids 0 1 --mode bench --max-batches 300
```

测速输出`metrics/bench.json`。整轮时间只是部分批次外推，不含验证和存盘；短数据若`timed_batches=0`，该次测速无效。

评价已有P0 ignore模型时，修改第一行的run-id为实际训练目录名，并保持配置、总batch和并行方式与训练一致：

```bash
trained_run=p0_ignore_dp_YOUR_RUN_ID
CUDA_VISIBLE_DEVICES=0,1 python experiments/ablation_v9/run_experiment.py \
  --config experiments/ablation_v9/configs/p0_mask_x.json \
  --run-id "p0_eval_$(date +%Y%m%d_%H%M%S)" \
  --checkpoint "experiments/ablation_v9/runs/$trained_run/checkpoints/best.pth" \
  --parallel data_parallel --gpu-ids 0 1 --mode eval
```

## 输出与常用设置

结果保存到`experiments/ablation_v9/runs/`下对应run-id目录，训练run包括`train.log`、实际配置、检查点和指标；其他模式只生成对应产物。训练日志在数据加载完成后开始写入，之后每100个batch打印分项损失及耗时；两种padding模式统一按有效点验证总损失`valid_total`选择最佳模型。

需要调整时，修改`experiments/ablation_v9/configs/`中的对应JSON：

| 设置 | 字段 |
|---|---|
| 总batch | optimization.batch_size；或命令行--batch-size覆盖 |
| 采样器 | optimization.batch_sampler（length_bucket／sequential） |
| 分桶系数 | optimization.bucket_multiplier（默认50） |
| 日志间隔 | optimization.log_freq |
| 训练轮数 | optimization.epochs |
| 遮挡块长度 | input.mask_patch_size |
| 遮挡/抽样比例 | input.mask_ratio |

P0/P1/P2比较时保持采样器、总batch、种子、padding模式和并行方式一致。分桶会减少padding，因此padding对照的结果对应当前采样设置。

当前训练使用FP32，不支持通过配置启用AMP。v9不支持断点续训；已有run目录不会被覆盖。检测评价目前限于既有随机验证划分，尚无完整外部VUS评测流程。
