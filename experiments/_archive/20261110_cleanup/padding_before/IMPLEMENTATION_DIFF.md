# P0/P1/P2 相对原始 TimeRCD 的实现差异

> 迁移整理注（2026-11-10）：本页已统一路径表示与环境表述，实验数值和结论保留。旧清单中的文档摘要对应整理前版本；历史命令不作为当前用法，当前入口见两路线README。

本文用于代码核对。新消融实现位于 `experiments/ablation_v9/`，原始实现文件未被修改。

## 关键 patch 配置对应

必须区分两个含义：

- `model.patch_size=16`：传给 `TimeSeriesEncoder`，决定编码器 patch 大小；
- `input.mask_patch_size=16`：传给遮挡采样函数，决定每个连续遮挡块的时间点数。

2026-11-10 全链路更正：官方历史训练入口和当前 400k 入口都执行 `config = default_config`，再执行 `config.ts_config.patch_size = 16`。这是对同一个对象的原地修改；单 GPU 直接调用、DataLoader workers=0 时，遮挡函数实际读取 16。不能将配置类初始化默认值 4 当成训练运行值。三份默认配置已统一为 16/16/0.15，仍保留两个字段以支持独立消融。

上游历史训练源码为 `thu-sail-lab/Time-RCD@ebf4f056b571f7ea90602ef666ae56f7d717ebfe/training.py`；当前官方 main 的库模块不是同一训练入口，默认 4 且以 0.1 倍高斯噪声替换。多 GPU spawn 也可能重新导入全局默认 4。详细来源、哈希和测试见 `audits/patch_20261110/REPORT.md`，不从发布权重的形状反推历史训练遮挡设置。

原版采样每条序列的块数为 `ceil(valid_length / mask_patch_size)`，选取 `int(num_blocks * mask_ratio)` 个块，并把对应 X 置零；遮挡比例因整数取整略有偏差。v9 使用显式种子生成块位置，保持独立配置但具备可复现性。

## P0/P1/P2

- P0：遮挡 `X` 输入，传入 `V & ~M`，重构观测 `X`。编码器按每 16 点聚合可见标志，只在整个 token 无可见点时屏蔽该 key/value；部分被遮的 token 仍可见。
- P1：完整 `X` 输入，在相同遮挡采样位置重构 `X`。
- P2：完整 `X` 输入，在相同遮挡采样位置重构配对正常值 `X^0`。

三组共用相同模型和损失权重。P0/P1/P2 分类损失均仅使用有效标签位置；这一点修正了原版先把 padding 标签二值化、再尝试过滤的实现问题，因此 P0 不是字节级复制。

patch 设置一致也不代表训练轨迹一致：v9 训练 DataLoader 为 `shuffle=False`，原版使用每轮重排的 DistributedSampler；v9 独立种子采样并固定验证 M，原版使用全局随机源。上述差异本次只核实披露，不混入 patch 修正；不能将历史两次训练的 loss 差异归因于遮挡长度。

## 原始模型结构复用

复用原始 `TimeSeriesEncoder`、重构头 `d_proj → 4d_proj → 4d_proj → 1`、异常头 `d_proj → floor(d_proj/2) → 2`、特征维 logits 均值和二分类交叉熵。新实验唯一入口是 `run_experiment.py`，负责 audit、smoke、train、eval、report 和运行登记。

## Batch 日志

`optimization.log_freq` 控制训练 batch 日志频率，默认 100，记录 Total/Recon/Anomaly Loss；每轮结束写平均分项损失和验证总损失。

原始实现与新实现的全部行为差异仍以实际代码为准；本文只记录当前 P0/P1/P2 协议及已核实差异。

## 2026-11-10 可选 DataParallel

- `src/parallel.py`：新增单进程多 GPU 封装；显式检查可见 GPU 编号和总 batch 大小。保留默认单卡，未移植原版的多进程全局配置读取方式。
- `src/config.py`、`configs/*.json`：增加 `runtime.parallel` / `gpu_ids`，旧配置缺省时仍按单卡处理。
- `run_experiment.py`：增加 `--parallel` / `--gpu-ids`，保存覆盖后的实际配置与可见 GPU 信息；统一封装训练、验证、eval 和 smoke，加载权重前解封装。
- `src/train.py`：通过解封装访问编码器属性、保存无 `module.` 前缀的权重；记录实际并行模式、主卡和总 batch；smoke 按所选卡数提供足够样本。
- 标准化和采样在分卡前执行，loss 在汇总后执行。没有各卡独立标准化、各卡均值等权平均或按 GPU 数量增加总 batch。P0/P1/P2 的输入、mask、目标、模型和学习率保持原协议。
- 多卡 dropout 随机流及浮点归约顺序可能不同，不承诺单卡和多卡训练逐位一致。DataParallel 主卡负责汇总，速度和显存收益需实测；本次不提供 DDP / 多机模式，也不增加 resume。
- `tests/test_parallel_contract.py`：合成变长 batch 的汇总 loss/梯度、检查点与训练流程回归；双 GPU 测试在可见卡数不足时显式 skip。

> 当前性说明（2026-11-11）：本文是指定日期的历史记录，不随当前实现改写。旧入口、配置生成步骤、batch16/固定顺序及当时的run保留说明不作当前事实；当前使用两路线README，代码差异见ablation_v9/IMPLEMENTATION_DIFF.md（仓库位置：experiments/ablation_v9/IMPLEMENTATION_DIFF.md），最新资产存在性与执行状态见根实验记录第12节。
