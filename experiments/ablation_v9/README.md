# P0 / P1 消融实验

比较“保留遮挡观测重构时，增加正常参考监督”对异常检测的影响。

| 实验 | 训练输入       | 重构任务                           |
| ---- | -------------- | ---------------------------------- |
| P0   | 遮挡后的观测 X | 恢复观测 X                         |
| P1   | 与P0相同       | 同时恢复观测 X、预测配对正常值 X⁰ |

P1两个重构头读取同一次Transformer输出，在相同抽样位置计算损失；另保留原异常分类任务，三项权重默认均为1。默认patch16、遮挡块16、抽样15%、总batch32、长度分桶。推理完整输入，只使用异常分类头。

## 准备与检查

以下Linux命令从仓库根目录执行，先激活项目环境。已有400k数据及同名manifest放在`experiments/original_400k/data/`；数据和模型产物不随Git下载。

```bash
python -m pip install . pytest -r experiments/ablation_v9/requirements-eval.txt
python experiments/ablation_v9/tests/run_checks.py

CUDA_VISIBLE_DEVICES=0 python experiments/ablation_v9/run_experiment.py \
  --config experiments/ablation_v9/configs/p1_mask_x_normal_padding_official.json \
  --run-id "p1_smoke_$(date +%Y%m%d_%H%M%S)" --mode smoke
```

测试只用合成小数据；smoke核对数据/manifest是否就位，不加载完整400k。

## 双卡训练（梯度防护＋学习率控制）

当前配对实验从头重训P0和P1，使用official padding、双卡、总batch32和seed72。两组统一采用AdamW、全局梯度范数裁剪5；第1轮学习率从5e-5线性预热至5e-4，随后按更新步余弦衰减至第50轮末5e-5；最多50轮，patience7。两份stable配置可直接修改裁剪阈值及lr_schedule，无需生成配置。

建议在tmux内依次执行以下命令，第一组正常结束后再运行第二组，不同时争用两张卡：

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
```

逐batch诊断从epoch0开启；每100批详细统计、每250批保存两槽滚动现场，最多12个有限升级事件；非有限值始终保存现场并停止。每100批打印loss、学习率和裁剪信息。新run只保留best、最近两轮和首次事件前一轮权重。P1重构头使用激活检查点降低显存。数据加载需数分钟。

`optimization.jsonl`记录裁剪前后范数、系数、学习率和更新序号；`diagnostics/batches.jsonl`与事件快照中的梯度始终是裁剪前值；`updates.jsonl`记录实际更新后状态。裁剪限制梯度，不保证前向loss不会出现尖峰。

## 单卡与padding对照

单卡可使用下列命令；总batch仍为32，显存不足可加`--batch-size 16`，相应P0对照也必须一致修改。

```bash
CUDA_VISIBLE_DEVICES=0 python -u experiments/ablation_v9/run_experiment.py \
  --config experiments/ablation_v9/configs/p1_mask_x_normal_padding_official.json \
  --run-id "p1_mask_normal_single_$(date +%Y%m%d_%H%M%S)" --parallel none --mode train
```

以下四份配置保留历史固定学习率、无裁剪训练方式；另有上述两份stable配置。所有配置均直接使用，无需生成本机副本：

| 实验 | ignore配置            | official配置                           |
| ---- | --------------------- | -------------------------------------- |
| P0   | p0_mask_x.json        | p0_mask_x_padding_official.json        |
| P1   | p1_mask_x_normal.json | p1_mask_x_normal_padding_official.json |

需要ignore对照时直接运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -u experiments/ablation_v9/run_experiment.py \
  --config experiments/ablation_v9/configs/p1_mask_x_normal.json \
  --run-id "p1_mask_normal_ignore_dp_$(date +%Y%m%d_%H%M%S)" \
  --parallel data_parallel --gpu-ids 0 1 --batch-size 32 --mode train
```

official把分类padding视为正常点；ignore排除这些点。两模式都保持attention屏蔽padding，重构损失只用有效选中位置。同组两配置仅padding字段不同。

## 评测已有模型

准备完整`datasets/TSB-AD-U/`（700条、11类）及训练数据。把下列`trained_run`设为实际训练目录名：

```bash
trained_run=p0_official_dp_20261111_003608
CUDA_VISIBLE_DEVICES=0,1 python experiments/ablation_v9/run_experiment.py \
  --config "experiments/ablation_v9/runs/$trained_run/config.json" \
  --checkpoint "experiments/ablation_v9/runs/$trained_run/checkpoints/best.pth" \
  --eval-config experiments/ablation_v9/configs/eval_official.json \
  --run-id "${trained_run}_eval_s200_r700_$(date +%Y%m%d_%H%M%S)" \
  --parallel data_parallel --gpu-ids 0 1 --eval-max-synthetic 200 --mode eval
```

同一次评测固定200条合成留出＋全部700条真实序列，计算官方16指标及probability/logit两类分数。去掉`--eval-max-synthetic 200`会评测全部约20000条合成留出；默认仍需完整pickle读取。检查流程可另加`--eval-max-real 1`并把合成限制设为2，结果会明确标为子集。

合成留出参与过best选择，不是独立测试集；官方F1阈值用测试标签oracle搜索，不能视为部署阈值成绩。逐序列NaN保留原值，汇总排除非有限值并记录有效数。评测产物含`metrics/per_series.csv`、`metrics/eval.json`及分组/分域CSV。论文与已完成P0对照见[结果表](RESULTS.md)；新P1取得同协议成绩后在末行追加。

## 续训、诊断与常用参数

新P1轮末检查点默认保存完整续训状态。以下路径使用同一训练run：

```bash
trained_run=替换为实际P1训练目录名
CUDA_VISIBLE_DEVICES=0,1 python -u experiments/ablation_v9/run_experiment.py \
  --config "experiments/ablation_v9/runs/$trained_run/config.json" \
  --resume-from "experiments/ablation_v9/runs/$trained_run/checkpoints/epoch_0000.pth" \
  --run-id "p1_resume_$(date +%Y%m%d_%H%M%S)" --epochs 50 --mode train
```

源进程须先退出；`--epochs`是总轮数上限，已完成epoch0则从1继续。恢复校验要求训练条件一致；只允许改变运行轮数上限、日志和诊断设置。stable检查点还核对每轮步数和全局更新位置；lr_schedule.total_epochs固定为50，缩短本次--epochs不会压缩原计划，续训也不会重新预热。早期P0权重缺完整RNG，不能直接用它精确续训。

| 参数                               | 用途                                           |
| ---------------------------------- | ---------------------------------------------- |
| --log-freq 100                     | 控制台/train.log的batch间隔                    |
| --diagnostics                      | 可选逐batch独立诊断，不更改损失或梯度          |
| --diagnostic-detailed              | 启用定期详细统计和异常升级现场                 |
| --diagnostic-detail-interval 100   | 完整统计每100批一次；设为1可每批采集，开销较大 |
| --diagnostic-snapshot-interval 250 | 详细诊断每250步保存滚动现场，保留最近两份      |
| --checkpoint-keep-last 2           | 新run只保留best、最近两轮及首次诊断事件前驱    |
| --epochs 50                        | 总轮数上限                                     |
| loss.normal_reference_weight       | P1正常参考损失权重，JSON中默认1且须为正        |
| input.mask_patch_size / mask_ratio | 遮挡块长度/比例，默认16/0.15                   |

`bench --max-batches 300`会执行训练更新但不保存模型，用于短程测速；不是smoke。实际只支持FP32，没有AMP/DDP/FSDP或batch中途恢复。

所有产物位于`runs/<run-id>/`，已有目录拒绝覆盖。训练保存实际配置、日志、逐轮/best完整模型及指标；训练验证按各自valid_total选best，P1含额外正常参考项，不能直接横比P0/P1总loss。旧完整输入P1/P2已退役，旧run和配置快照保留为历史，不使用旧命令启动新P1。

## 代码入口

统一入口为run_experiment.py。src按职责整理为9个实现文件：config（配置与审计）、data（数据与遮挡）、models（模型与并行）、losses、train（训练/优化/续训）、diagnostics、evaluate（训练验证）、benchmark_eval（完整评测/汇总）、metric_worker（指标子进程）；另有空的__init__.py。
