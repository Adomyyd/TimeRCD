# v9 P0/P1/P2 消融实验

> 迁移整理注（2026-11-10）：本页已统一路径表示与环境表述，实验数值和结论保留。旧清单中的文档摘要对应整理前版本；历史命令不作为当前用法，当前入口见两路线README。

本目录实现登记的 P0/P1/P2 消融协议。相对原始 TimeRCD 的实现差异见 `IMPLEMENTATION_DIFF.md`。统一入口是 `run_experiment.py`，配置与运行产物均位于本目录。

本目录与 `experiments/original_400k/` 并列。默认数据路径为 `experiments/original_400k/data/`，工程规范见本目录 `EXPERIMENT_SPEC.md`。原流程和 v9 各有自己的统一入口和 runs；历史 run 快照不改写。

三个配置分别为：P0 遮蔽输入并重构观测 `X`；P1 输入完整 `X` 并重构 `X`；P2 输入完整 `X` 并重构配对正常值 `X^0`。

## 训练日志

训练日志默认每 100 个 batch 写一次，格式与原版一致，包含 Total、Recon、Anomaly Loss；每个 epoch 还记录三项平均值和验证总损失。通过 `optimization.log_freq` 调整间隔。

## 编码器 patch 与遮挡粒度

二者是独立配置：

- `model.patch_size=16` 是 TimeSeriesEncoder 的 patch 大小，与原版 400k 入口一致；
- `input.mask_patch_size=16` 是遮挡函数选择连续时间点块的长度，与官方历史训练入口及当前单 GPU 400k 复现一致；
- `input.mask_ratio=0.15` 是按遮挡块数量计算的选择比例。

当前三组配置保持编码器 patch 为 16、遮挡块长度为 16、比例为 0.15。原配置类的默认值虽然是 4，但官方训练入口和 400k 入口使用 `config = default_config` 后将其原地改为 16，同进程遮挡函数也读到 16。之前“400k 是 16/4”的说明已更正。源码来源、多进程例外和测试证据见 `audits/patch_20261110/REPORT.md`。

调整遮挡粒度时只改 `input.mask_patch_size`；调整遮挡比例时只改 `input.mask_ratio`。不要为改变遮挡块长度修改 `model.patch_size`。16/4 可作为独立消融，但不能称为当前 400k 的复现设置。复制配置到新文件并使用新 run-id，保留既有 run 的配置快照。

例如，要测试遮挡比例 0.05，复制 P0 配置并只改 `input.mask_ratio`：

```bash
cp experiments/ablation_v9/configs/p0_mask_x.json \
   experiments/ablation_v9/configs/p0_mask_x_ratio05.json
```

随后通过统一入口使用新配置和唯一 run-id：

```bash
CUDA_VISIBLE_DEVICES=1 python \
  experiments/ablation_v9/run_experiment.py \
  --config experiments/ablation_v9/configs/p0_mask_x_ratio05.json \
  --run-id p0_mask_x_ratio05_seed072 \
  --mode all
```

示例使用已激活项目环境中的`python`，从仓库根目录执行。正式训练由用户决定启动。

## 可选多 GPU

默认 `runtime.parallel="none"`、`gpu_ids=null`，原单卡命令继续有效。
同机多卡采用单进程 `torch.nn.DataParallel`，通过统一入口启用：

```bash
CUDA_VISIBLE_DEVICES=0,1 python \
  experiments/ablation_v9/run_experiment.py \
  --config experiments/ablation_v9/configs/p0_mask_x.json \
  --run-id p0_mask_x_seed072_dp01 \
  --mode train --parallel data_parallel --gpu-ids 0 1
```

P1/P2 分别替换配置为 `p1_full_x.json` / `p2_full_x0.json`，并使用新的 run-id。
先将 `--mode train` 改成 `--mode smoke` 并换一个 run-id，可只做合成小批量前向/反向检查，不读取 400k pickle。

也可在复制的配置中设置 `runtime.parallel="data_parallel"`、`runtime.gpu_ids=[0,1]`，保持 `runtime.device="cuda"`。CLI 覆盖优先于 JSON，并将生效设置写入 run 的 `config.json`。设备编号相对于 `CUDA_VISIBLE_DEVICES`；如设置 `CUDA_VISIBLE_DEVICES=1,0`，`--gpu-ids 0 1` 的主卡是物理 GPU1。列表首卡保存参数、汇总输出和计算损失，显存使用可能更高；请先确认所选卡空闲。不指定可见设备环境变量时，编号对应当前进程可见的 CUDA 设备。

`optimization.batch_size` 始终是**总 batch**。例如 16 在两张卡间分摊，而非每卡 16；最后不足 GPU 数量的小 batch 自动使用可分配到的卡。标准化、padding 和随机 M 在完整 batch 上生成，损失在汇总预测上按全 batch 有效点计算，不对各卡均值再平均。因此模型、监督、学习率、样本顺序和更新步数不随卡数改变。并行归约和 dropout 随机流仍可能使数值轨迹不同，不能保证单卡/多卡逐位相同；比较 P0/P1/P2 时建议三组采用相同设备配置。

训练、验证、独立 eval 与 smoke 均复用同一封装。只由一个进程记录日志、早停和保存权重；检查点不带 `module.` 前缀，可通过 `--mode eval --checkpoint experiments/ablation_v9/runs/<run-id>/checkpoints/best.pth` 在单卡/多卡加载。当前入口仍不支持恢复训练，本次没有增加 resume。

新增模块 `src/parallel.py` 负责配置校验、设备选择、封装与解封装。模型结构及 `sampling.py`、`data.py`、`losses.py`、`evaluate.py` 的算法未改；不会出现原始 spawn 路径重新导入默认 mask 大小的问题，采样仍明确读取 `input.mask_patch_size=16`。

## 检查命令

```bash
python \
  experiments/ablation_v9/tests/run_checks.py
```
