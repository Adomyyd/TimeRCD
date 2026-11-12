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
python -m pip install . pytest -r experiments/ablation_v9/requirements-eval.txt
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

评价一次会依次处理合成留出集与官方单变量基准：沿用训练时95/5划分，默认约20000条合成序列；真实数据使用`datasets/TSB-AD-U/`中的700条序列、11类数据集。需先准备完整TSB-AD-U；缺组或数量不符会报错，不会静默跳过。

下面评价已完成的P0 official。等训练设备空闲后，从仓库根目录执行；评价其他模型时只修改`trained_run`。直接读取该训练run的配置，避免混用P组或padding策略：

```bash
trained_run=p0_official_dp_20261111_003608
CUDA_VISIBLE_DEVICES=0,1 python experiments/ablation_v9/run_experiment.py \
  --config "experiments/ablation_v9/runs/$trained_run/config.json" \
  --checkpoint "experiments/ablation_v9/runs/$trained_run/checkpoints/best.pth" \
  --eval-config experiments/ablation_v9/configs/eval_official.json \
  --run-id "${trained_run}_eval_$(date +%Y%m%d_%H%M%S)" \
  --parallel data_parallel --gpu-ids 0 1 --mode eval
```

两侧均用完整输入检测；按官方测试序列标准化、5000点非重叠窗口和尾部补齐规则，分别计算异常概率与logit差的16项指标，包括AUC-PR/ROC、VUS-PR/ROC、Standard-F1、PA-F1、Affiliation和F1_T。指标复用仓库官方实现，默认fast、F1_T分位点800；F1类指标沿用官方最优阈值搜索，不能当作部署时可用的固定阈值成绩。合成侧仍是选择模型用过的留出集，不是独立测试集。

完整评测按序列计算全部指标，明显慢于训练期间的验证。先检查流程时，可在同一命令末尾追加`--eval-max-synthetic 2 --eval-max-real 1`；输出会标记`full_benchmark=false`，不能当作完整成绩。限制条数仍需要加载完整合成pickle。

评测参数只需修改`configs/eval_official.json`，无需生成配置。默认推理batch64与训练总batch32相互独立；显存不足可降低`inference_batch_size`，实际参数会记录。更改官方默认参数会标记`official_default_parameters=false`。单卡评测可使用`CUDA_VISIBLE_DEVICES=0`并将并行参数改为`--parallel none`，去掉`--gpu-ids`。

新评测run保存：`eval.log`、`eval_config.json`、`evaluation_manifest.json`、`metrics/per_series.csv`、`metrics/group_mean_metrics.csv`、`metrics/domain_summary.csv`、`metrics/eval.json`和`scores/*.npz`。CSV同时提供按序列均值和按数据集先平均再等权平均；原始概率/logit可追溯。`save_scores=false`可关闭分数文件。中途失败保留已完成行并标记failed，不伪报完成；当前不支持评测断点续跑。

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

当前训练使用FP32，不支持通过配置启用AMP。v9不支持断点续训；已有run目录不会被覆盖。`eval`/`all`包含合成留出集与官方TSB-AD-U评测；`train`仍只做原验证及选best，不自动执行完整基准。
