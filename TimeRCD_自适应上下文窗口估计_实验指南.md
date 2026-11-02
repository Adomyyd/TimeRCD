# TimeRCD 自适应上下文窗口估计：代码与实验指南

本指南把 [`TimeRCD_自适应上下文窗口估计_研究方案.md`](./TimeRCD_自适应上下文窗口估计_研究方案.md) 转成项目内可逐阶段实现、验证和复现的工程计划。研究方案中的候选窗口、教师定义、学生损失、三关验证与推翻条件仍是方法依据；本文件补充代码边界、数据产物、测试点和执行顺序。

## 1. 目标、范围与硬约束

### 1.1 目标

冻结官方 TimeRCD 权重，离线生成无标签窗口区间教师，再训练只读原始序列的轻量估计器。部署时估计器选择一个窗口，TimeRCD 使用该窗口完成官方分块推理。研究的主要问题是：在同候选集和同等标签权限下，学生是否能稳定超过开发集确定的最佳全局固定窗口。

### 1.2 本轮范围

- **先做单变量**，使用当前官方 uni checkpoint；多变量是后续独立研究，不把本指南的 uni 结果外推到 multi。
- **不重新训练 TimeRCD**。训练的只有窗口学生；教师和窗口扫描都是冻结模型推理。
- 主候选集固定为 `W5 = [1000, 2000, 4000, 8000, 16000]`。每条长度为 `T` 的序列将候选变为 `min(W, T)` 并去重；训练、教师、预测和评估共用同一个有效候选映射。
- **主测试标签只用于最终效用评估**，不得用于教师标签、学生特征/训练、规则或阈值选择。真实 Tuning 标签标定只能作为单列实验版本。
- 官方最终检测输出使用 `offset=0` 单窗口分数。偏移/裁剪输出仅用于教师诊断和敏感性分析，不把扰动平均当成单窗口主结果。
- `W_eval = [1000, 2000, ..., 16000]` 的密集扫描只提供诊断 oracle，不扩充学生可选窗口。

## 2. 当前仓库结构与建议新增模块

### 2.1 现有实现映射

| 责任 | 当前文件 | 在本实验中的用法/限制 |
|---|---|---|
| 官方用户推理 API | `time_rcd/detector.py` | `TimeRCDDetector.from_local/from_project(..., win_size=...)`；适合最终单窗口推理与一致性回归测试。 |
| 分块、全局标准化、异常头输出 | `time_rcd/_inference.py` | `_WindowDataset`、`TimeRCDPretrainTester.zero_shot` 是官方分数协议的核心；offset 扰动需另实现并测试。 |
| 编码器、重建头、随机掩码 | `time_rcd/_core/TimeRCD_pretrain_multi.py` | 冻结模型及重建头可复用；不要未经核实直接调用现有随机掩码函数。 |
| patch embedding/padding | `time_rcd/_core/ts_encoder_bi_bias.py` | 核对 patch 网格、尾部 padding、mask 语义和输出长度。 |
| 配置 | `time_rcd/_core/time_rcd_config.py` | 模块默认 `patch_size=4`，不能据此认定官方 checkpoint 的 patch 单位。 |
| 基准封装 | `models/TimeRCD.py`, `model_wrapper.py` | 旧 TSB-AD 接口；`run_Time_RCD` 会新建测试器。研究代码优先使用 `time_rcd` API，避免基准全局状态。 |
| 现有窗口扫描入口 | `main.py --test_window_sizes` | 可用于小规模交叉核验，不是教师/学生实验管线；PKL 写入共享输出目录，同名结果会被覆盖，不同运行的产物也容易混淆。 |
| 基准指标 | `evaluation/metrics.py`, `evaluation/basic_metrics.py` | 使用当前项目评估口径；主指标 VUS-PR，保留逐序列预测与失败日志。 |
| 合成数据 | `dataset_gen/src/` | 扩展窗口敏感因素时，保留生成器参数、随机种子及配置版本；不要在 `training.py` 中训练基础模型。 |
| 固定超参 | `HP_list.py` | uni 默认窗口为 7000，multi 为 5000；本研究明确传入候选，不依赖默认值。 |

### 2.2 建议新增代码布局

把研究管线与通用检测基准入口解耦，建议新增：

```text
adaptive_window/
  __init__.py
  config.py             # 固化候选集、种子、阈值、版本号
  series_io.py          # TSB/合成样本读取、ID、来源与分割审计
  candidate.py          # 截断去重及原始候选索引映射
  frozen_model.py       # 加载 uni checkpoint、冻结、统一推理接口
  reconstruction.py     # 锚点、固定 patch 目标、掩码重建误差
  perturbation.py       # 分块偏移/短序列裁剪、时间坐标对齐
  teacher.py            # W_lo/W_hi、弃权、区间标签与诊断字段
  features.py           # phi(x)、psi(x,W)、特征缓存
  student.py            # GBDT 候选组损失；后续可加 CNN
  metrics.py            # 官方指标包装、逐条结果和汇总
  benchmark.py          # 缓存官方窗口分数、计时/显存
  cli.py                # prepare / teacher / features / fit / evaluate 子命令
tests/
  test_candidate.py
  test_reconstruction.py
  test_alignment.py
  test_teacher.py
  test_partial_label_loss.py
  test_inference_equivalence.py
```

建议输出与代码分开版本化，不将大型 PKL/分数张量提交到 Git：

```text
adaptive_window_runs/<run_id>/
  manifest.json              # Git commit、权重哈希、数据清单哈希、配置、环境
  splits.csv                 # series_id、source、split、长度、通道数
  scores/W_eval/*.npz        # 官方 offset=0 分数及 logits
  teacher/*.parquet          # 每条序列的误差、区间、弃权和诊断字段
  features/*.parquet         # 一行一个序列，带列清单/版本
  predictions/*.csv          # 每种方法逐序列分数和窗口
  metrics/per_series.csv
  metrics/summary.csv
  logs/
```

所有缓存键至少包括 `series_id + checkpoint_sha256 + preprocessing_version + window + perturbation + seed`。缓存命中时校验 key、长度、有限值和模型哈希；不能仅按文件名判断可复用。

## 3. 开始写教师代码前的 P0 阻断检查

未完成 P0 前，不批量生成教师标签，也不训练学生。

### 3.1 锁定模型权重和推理等价性

1. 记录 uni checkpoint 的绝对路径、SHA256、来源/版本、Git commit、PyTorch/CUDA 版本和设备。
2. 检查 state dict 是否含 `reconstruction_head.*`、`anomaly_head.*`、encoder 参数；确认 strict load 成功。重建头缺失时按研究方案记录弃权，不要静默换成其他代理。
3. 对一条固定、有限、非恒定序列，分别用现有 TSB wrapper 与 `TimeRCDDetector.from_local(..., win_size=W)` 运行官方 offset=0；比较分数/长度（允许预先记录的数值容差）。再验证分数是否与异常 logits 的 softmax 定义一致。
4. 确认学生/教师传给 API 的是原始数值；官方 `_WindowDataset` 会自行对整条序列按通道全局 z-score。不能先在外面标准化后又被 API 二次标准化。

### 3.2 核实 patch 与掩码语义（当前代码中的明确风险）

- 用户 API 在 `detector.py` 中设置 `config.ts_config.patch_size=16`；`time_rcd/_core/time_rcd_config.py` 的 `default_config` 则默认为 4。
- `create_random_mask()` 在 `TimeRCD_pretrain_multi.py` 中从**模块级 `default_config`** 读 patch size，而不是从当前模型配置读；本项目其他副本也有类似实现。因此如果加载的模型实际使用 16 点 patch，不能无检查地复用该函数产生教师掩码。
- 在所用权重与对应训练配置中核实：patch size、patch 切分起点、最后不足 patch 的处理、mask 是“被遮挡位置为 True/1”还是相反、无效/padding mask 的定义、填充值及比例。以 checkpoint/训练实现为准，不以注释或当前默认值为准。
- 单元测试逐点打印一个小型序列的 patch 索引、掩码和有效位置；验证 masked reconstruction loss 只对目标位置计分。固定目标在各候选窗口间重用同一原始位置、噪声和目标值。
- 方案主方法初设 `rho=0.15`、`M=4`、`q_norm=0.5`，在 P0 核实前都只是配置假设。每条教师记录应存实际 patch 单位、有效锚点数、共同目标数、种子和填充噪声种子。

### 3.3 输入与退化样本

统一定义并测试：空序列、长度 1、`T < patch_size`、常数序列、NaN/Inf、多变量输入（本阶段应拒绝）、无完整 patch、mask 目标不足。数据 `dropna()` 不应悄悄改变时间索引或标签对齐；在实验 loader 中遇到非法值要明确报错或用预注册的处理规则，并记录被排除序列。

## 4. 数据准备、隔离与实验台账

### 4.1 当前训练数据状态与构建计划

**当前没有可直接用于自适应窗口学生训练的训练集。** `datasets/TSB-AD-U/` 是异常检测评测数据，不能因为文件在仓库里就自动当作训练语料；尤其严格设定下，目标评测来源不得进入训练。方案中的 LOTSA 等外部真实语料也尚未落实，不能把它们写成现成数据。现阶段应先建立受控合成训练集；外部无标签真实数据待来源、许可、下载和去重审计完成后，作为单独版本加入。

要区分三个层次：

1. **原始训练序列**：无标签输入 `x`。合成生成器会额外产出真值 label/attribute，但主学生的教师监督必须来自冻结 TimeRCD 的无标签探针，而不是这些真值。
2. **教师训练表**：对原始训练序列运行冻结模型探针生成的 `S(x)`、弃权状态和诊断记录。这一步完成后，才能训练主方法学生。
3. **合成效用标签表**：可在合成数据已知真值上计算各窗口 VUS-PR，供窗口选择上界、机制验证及“合成效用标签学生”基线使用；不得混成主教师标签。

#### 首批合成数据的建议顺序

不要一开始生成大规模数据。先按以下规模逐级验证：

| 批次 | 建议规模 | 目的 | 放行条件 |
|---|---:|---|---|
| D0 | 100–300 条 | 检查生成、schema、随机种子、教师耗时与失败率 | 每条可复现；无 NaN/Inf；标签/序列长度一致；关键生成属性有记录。 |
| D1 | 1,000–2,000 条 | 训练与诊断学生原型、画学习曲线 | teacher 有效覆盖合理；按生成机制/长度分层均有样本；学生不只预测众数窗口。 |
| D2 | 5,000–10,000 条起，按学习曲线扩充 | 正式合成训练版本 | 加数据后验证/留出机制性能仍有改善或稳定；报告教师生成计算成本。 |

以上是资源可控的工程起步规模，不是统计功效保证；最终数量由按组留出的开发性能学习曲线决定。方案中“万级”只是建模设想，不能当作现成数据或预设充分样本量。

#### 先适配现有生成器，再扩窗口敏感机制

仓库 `dataset_gen/src/generate_dataset.py` 已能生成 uni 序列，字段为 `time_series / normal_time_series / labels / attribute`，且 `seq_len` 可指定，`seq_len=None` 走当前随机长度函数。它**不是已经准备好的研究训练集**：

- `generate_dataset()` 没有显式 `seed` 参数；多进程路径会重新初始化 NumPy RNG，并为 worker 取随机 seed；生成样本也没有稳定 `sample_id/seed` 字段。首版数据生成应新增 root seed 和每样本 seed 的显式参数/metadata；完成前用 `num_workers=1` 并在进程入口同时固定 Python `random` 与 NumPy RNG，仅用于原型，不宣称多进程可完全复现。
- 现有生成器不直接提供方案指定的可控周期扫描、段长切换和异常重复次数接口。先审计 `ts_generator.py`、`config.py` 中周期/季节性采样与异常注入，再实现带 `generation_spec` 的受控生成器；不要通过事后重采样、筛选输出冒充已控制机制。
- 当前随机长度范围和论文/预训练分布要以实际 `random_seq_len()` 代码核对。训练样本需覆盖短序列、候选边界附近长度、候选窗口内长度，以及 10k 以上的外推长度；把超过 10k 的部分标记为有意的长度外推，不与训练范围混报。
- 建议 `generation_spec` 至少存 `period P`、状态切换数/段长 L、噪声/振幅 regime、异常类型/密度/重复次数 r、序列长度、全局 seed、样本 seed、生成器 Git commit 和配置哈希。真实/合成来源一律有稳定 `series_id`。
- `generate_dataset()` 当前一次性把列表保存在内存。先用小数据量；需要放大时改成分片写盘（例如每 shard 500–1,000 条），校验后再合并索引，避免长序列样本导致内存峰值。

#### 合成集合划分与窗口机制覆盖

- 按**随机种子及生成参数组合**划分 train/dev/test，而不是先生成样本再随机逐条切分。相同参数实例、派生裁剪段或同一随机基础序列不得跨 split。
- 推荐先把方案里的周期 `P∈{50,200,800,2000,5000}`、状态段长 `L∈{1k,2k,4k,8k,16k}`、重复异常 `r∈{1,2,4,8}` 做成实验因素；先用全因子或受控分层抽样保证覆盖，不要求笛卡尔积全量。另留出部分参数组合只做机制外推测试。
- 生成器 label 只用于可控性检查及合成效用标签基线。训练主学生时，屏蔽生成器 label 和 attribute 中的 anomaly 描述；特征缓存也不得包含它们。
- 每个 split 输出清单、样本数、长度分布、周期/段长/重复次数分布、异常密度、失败/重试数和 manifest；检查同一输入 seed + 版本能再生相同数据（或明确记录浮点/平台容差）。

可以先在 `dataset_gen/` 目录下用 `PYTHONPATH=src python`，以 `num_workers=1` 调用现有 API 生成 D0 原型（这是已有函数的使用示例；root seed/metadata 支持仍需先实现才适合正式数据集）：

```python
import random
import numpy as np
from generate_dataset import generate_dataset

seed = 20261102
random.seed(seed)
np.random.seed(seed)
pilot = generate_dataset(
    num_samples=100,
    seq_len=5000,
    anomaly_sample_ratio=0.5,
    is_multivariate=False,
    use_attribute_set=True,
    num_workers=1,
)
```

这个例子只验证生成器基本可用；**不代表**已经实现了 P/L/r 的受控扫描、稳定逐样本复现、训练/验证拆分或 TimeRCD 教师标签。

### 4.2 拆分设计

先建立不可变的 `splits.csv`，以数据来源/数据集为分组单位，防止同一来源的高度相似序列跨开发与评测泄漏。

- **严格主设定**：训练语料可用受控合成数据和经来源审计、不与目标评测重叠的无标签真实数据；教师和特征不读取异常标签。教师阈值/学生超参只在独立合成开发集确定。
- **最终评测**：TSB-AD-U-Eva（或项目当前锁定的 11 个数据集、700 条评测清单）在方案锁定后只评一次确认性结果。实际使用哪份清单、哈希、是否属于 Eva，要在 manifest 写清楚；不能把“700 条”与完整 TSB-AD-U-Eva 混称。
- **基准内无标签迁移**：单列设定，按数据集/来源留一，当前留出的来源所有序列不得进入学生训练、标定或特征统计。
- **真实标签标定**：若使用 TSB tuning labels，版本单列为“有标签标定”；核查 Tuning/Eva ID 是否重叠。Tuning 标签不得又被称为严格无目标域标签。
- **oracle**：允许在标签上后验计算，但只报告为不可部署上界；不反馈到训练或规则调整。

方案提及 LOTSA 与评测源的重叠、Tuning 与 Eva 是否不相交、旧权重是否可得，这些是来源审计任务，不可默认视为已解决。

### 4.3 复现当前固定窗口基线

当前项目已有固定 5000 的单变量测评产物 `Uni_Time_RCD_v4_wins-5000.csv`（此前完成 700 条）。将它作为**既有参考**，并把对应日志、数据清单与 checkpoint 哈希复制进新实验 manifest；正式对照仍需在同一 run pipeline 重算固定 5000 和所有比较方法，确保数据清单、测试段切分、指标实现和有效候选映射一致，避免不同版本指标文件混用。

额外核对 `HP_list.py` uni 的当前默认 `win_size=7000` 与论文协议 5000 不同；固定 5000 必须显式传参。不要用默认配置误标为 fixed-5k。

## 5. 推荐实现顺序与每步交付

### P0：模型/掩码/数据契约（先做）

交付：`docs/implementation_contract.md` 或 manifest 初版、checkpoint 哈希、推理等价测试、patch/mask 对照表、最小单元测试。通过标准：官方 API 可复现；重建头能前向；mask/padding/patch 网格有代码证据；异常输入行为明确。

### P1：候选集与官方窗口扫描（先跑少量，再批量）

1. 实现纯函数 `effective_candidates(T, raw_candidates)`，返回去重的有效候选及每个有效候选对应的原始窗口集合；测试 `T<1000`、恰好边界、候选重复和只剩一个候选。
2. 实现冻结推理器。每个候选独立设 `win_size=min(W,T)`，记录实际窗口与 API 分块数，`eval()`、`no_grad()`，固定 seed。
3. 先选 5–10 条覆盖短/长/常数风险的序列做 smoke test；检查输出长度等于测试段长度、所有分数有限、标签时间戳对齐。
4. 先对开发集构建 `W_eval` 官方 `offset=0` 分数缓存；最终评测缓存只在协议锁定后生成并封存，不能查看其标签效用后再调整方法。相同序列各窗口的原始输入和标签切分必须一致。
5. `main.py --test_window_sizes` 只能用于小规模交叉核验：其 PKL 写到共享 `Uni_/Multi_<AD_Name>_v4` 目录，同名路径会被覆盖，CSV 也按窗口列表固定命名覆盖。用独立实验工作目录/自建 run ID 保存产物，并校验 checkpoint 和内容哈希；不要对用户已有结果目录做清理或覆盖。

交付：一行一个序列-窗口的预测缓存及 manifest；每条记录含 `T, W_requested, W_effective, n_blocks, score_length, checkpoint_hash`。

### P2：逐序列窗口效用表及关卡 1（是否值得自适应）

- 对预测缓存使用项目官方指标计算函数；VUS-PR 为主，AUC-PR、F1-T、Standard-F1、Affiliation-F 辅助。每条序列保留指标原值、错误/缺失状态，不能把失败值自动当有效 0 而不标记。
- 计算固定 5k、开发集锁定的 `W_fix`、同候选集逐序列 oracle、密集网格 oracle；有效候选只有一个的样本仍参与固定方法和覆盖率统计，但不参与“窗口可选择”训练损失。
- 主报告必须双口径：逐序列平均、逐数据集平均；不把小数据集被放大的数据集宏平均当作唯一主结果。
- 按数据集/来源簇做配对 bootstrap 置信区间；逐序列 Wilcoxon 作为补充。开发阶段先锁定最小实际意义阈值，再看最终标签结果。
- 单独报告偏移后验增益与裁剪子集，不从窗口 oracle 增益中相减“校正”。

交付：`metrics/per_series_window.csv`、关卡 1 报告及决策。若 W5 选择余量接近零，或简单长度/周期规则已取走大部分余量，则停止“自适应窗口提升检测”主张，不进入昂贵教师/学生阶段。

### P3：教师探针与关卡 2（教师目标是否可信）

教师建议拆成三个可单测的部分：

1. `reconstruction.py`：官方默认窗口 `min(5000,T)` 只用于筛疑似正常 patch；在完整 patch 均值分数的 `q_norm` 分位内选目标。按序列固定锚点、掩码位置和替换噪声；对全部候选使用相同目标。输出每个候选 `R_patch(W)`、有效目标数及失败原因。
2. `perturbation.py`：对每个候选生成方案规定的 offset 集合；短序列共同有效位置不足时，对所有候选一致切换到 head/tail crop。所有分数回映射到原时间坐标，在相同有效位置集合计算 `delta_W`、`Delta_jk`。比较无定义时应标无效/弃权，不得当作差异 0。
3. `teacher.py`：按锁定的 `epsilon, kappa, q_top` 计算 `W_lo/W_hi/S(x)`。无信息/平坦探针明确弃权为全部有效候选；区间的单点代表按方案规定的对数距离规则确定，平局取较小窗口。记录区间内部每个候选效用只作评估，不回流到标签。

**先做诊断小样本**：在开发集抽取覆盖不同来源、长度、效用曲线形态的序列，至少做多随机种子/重采样稳定性测试；重复次数先在开发阶段确定并写入配置（不临时挑选有利种子）。确认有效覆盖、标签稳定性、patch 误差与 oracle 效用关系后，再批量算教师表。

**关卡 2 评估**：同候选集真实近优集合连续性；教师区间命中率与同宽度随机连续区间比较；区间内部近优占比和最差遗憾；单点教师代表相对 `W_fix` 的 VUS-PR 与遗憾；平坦/弃权率；按数据集/来源分层结果。教师“命中 oracle”不能单独证明可信，弃权样本不计入有效样本命中率且需单列全样本结果。

交付：逐序列教师表、探针稳定性报告、阈值锁定记录、关卡 2 决策。失败时停止或修订并重新注册版本，不能扩大区间掩盖失败，也不能直接归因于异常污染。

### P4：合成训练集/特征与学生

#### 合成数据

这是当前的首要数据工作，而不是假设训练集已存在。先实现上文 D0 生成/校验与 split manifest；再按 `train/dev/mechanism-test` 生成可复现的 D1/D2。文件建议分成 `synthetic_raw/<split>/shard-*.pkl` 和 `synthetic_index.csv`，每条样本包含 `series_id, time_series, labels, normal_time_series, generation_spec, root_seed, sample_seed, generator_version`。严格模式训练时学生 loader 只暴露 `time_series` 与 `series_id`；labels/attribute 仅由单独的合成效用评估器读取。

数据就绪后运行顺序为：原始合成序列 → 冻结 TimeRCD 离线教师 → 带来源/版本的教师表 → 仅训练 split 的特征 → 学生训练；dev 只用于选学生超参/检查学习曲线，mechanism-test 留到机制结论锁定后。不要把基础模型预训练入口 `training.py` 当成这个学生训练流程，也不要改动或更新 TimeRCD 权重。

在 `dataset_gen/src/` 新增独立配置/生成器扩展，至少覆盖长周期、分段状态（季节/幅值/噪声）、重复或聚集异常；每条样本写入已知生成参数、seed、生成器版本。训练/开发/测试按**生成种子和参数组合**拆分，不能把同一随机样本切片到不同分区。合成标签只用于效用基线、机制分析或明确声明的辅助目标，不代替主教师监督。

#### 特征 v1

优先做可测、缓存、便于审计的特征：`log(T)`、周期及其可用性/强度、SuSS 统计尺度、变点率/分段统计、必要的自相似性摘要，以及每个候选的 `W/P, T/W, W*r_cp, W>=T`。长度上限、降采样方式、缺失值编码和单位换算进入配置。先做 `log(T)` 和 `log(T)+周期` 两个弱基线，再逐步加特征；catch22、PELT、FLUSS、矩阵剖面属于可选扩展，必须计时并实测其复杂度，不作为 v1 阻塞条件。

所有特征仅依赖序列输入及长度，不允许读 labels、teacher 分数、重建误差、窗口效用或测试集统计量。拟合任何标准化/缺失填充值都只在训练 split 上完成。

#### 学生 v1 与损失测试

- 先实现小型 PyTorch 条件打分器/分类器，直接优化候选组 partial-label loss，作为正确性基准；每个序列的 softmax 只在有效候选上归一化。
- 对 LightGBM 自定义目标，输入必须按序列分组，在组内共同 softmax、返回候选 logit 梯度。先用有限差分测试梯度；真实 Hessian 对角可能为负，若实现采用正曲率替代，文档和代码中应明确叫“替代曲率”，不能写成精确 Hessian。全候选弃权组的候选集损失/梯度为零并跳过。
- 以来源/数据集分组做交叉验证和超参选择；用相同拆分拟合长度回归、长度+周期模型。报告学生预测分布，检查是否塌缩到众数候选。
- GBDT v1 先不加生成器多任务头。1D-CNN、辅助生成参数目标、硬标签交叉熵、`log(W)` 回归均作为后续消融，不和主版本混在一起。

### P5：关卡 3 与最终确认

比较固定 5k、开发集选定 `W_fix`、长度回归、长度+周期、周期/统计尺度启发式、合成效用标签学生、教师区间学生；同一候选集、同一权重、同一分数评估代码。报告：VUS-PR 等检测指标、学生 top-1 命中教师区间（仅教师有效样本）、全样本覆盖/弃权、同候选 oracle 遗憾、端到端延迟和显存。

只有关卡 1/2 在开发阶段通过、代码与超参锁定、来源/数据清单审计完成后，才运行最终 Eva 确认。最终标签结果出来后若调整规则或重训学生，该轮改称探索性结果；需要新的独立来源/测试集才能再次作确认性声明。

## 6. 评估、成本与结果报告约定

### 6.1 统一指标与失败状态

- 主指标 VUS-PR；辅助指标 AUC-PR、Affiliation-F、F1-T、Standard-F1，与方案一致。
- 汇总同时出逐序列与逐数据集均值，并附每组样本数、缺失数、失败数。指标异常/无定义保存为 NA + reason，不悄悄转换为 0。之前项目日志中已出现无真实事件导致 Affiliation 失败的情况，评估脚本应可识别和披露。
- 所有方法复用同一标签切片、序列长度、指标函数版本和后处理，不为某一方法单独调阈值。

### 6.2 统计

以数据集/来源为 cluster 做配对 bootstrap；同一 cluster 内保留配对方法结果。另报告逐序列配对 Wilcoxon 作为补充。数据集数量不适合可靠做 Friedman/Nemenyi 时，报告效果量和区间而不强行给显著性结论。将 bootstrap seed、重复次数、CI 方法纳入 manifest。

### 6.3 速度与成本

分开测量：学生特征、学生打分、单次 TimeRCD 分块前向、完整自适应端到端；固定 5k、学生窗口、K 窗口集成和离线教师分别列出。固定输入长度、通道数、硬件、batch size、预热次数及计时边界，CUDA 计时同步，重复多次报告中位数/分位数；记录峰值显存和底层块数。

不把“一个窗口配置”写成“网络只调用一次”：长序列仍由多个 block 前向组成。不要从论文局部速度表线性外推本地教师成本。

## 7. 必须覆盖的测试清单

### 单元/性质测试

- 候选截断去重：有效候选排序、反向映射一致，单候选可识别。
- patch/mask：patch 边界和末端 padding 正确；目标位置在候选间完全一致；重建 loss 分母是有效被遮挡点数；不同候选都在相同目标上计分。
- 归一化：同一序列所有候选使用一组全局参数；裁剪探针也不单独重算统计量。
- 分块 offset：offset=0 与官方推理逐点一致；非零 offset 输出映射回原始索引无重复、无遗漏、长度匹配。
- 比较有效区：不同窗口和扰动只在共同坐标集上计算；空/过短交集显式弃权；常数分数/Spearman 无定义不能判成一致。
- 教师规则：`W_lo <= W_hi`、区间非空；平坦/失败探针输出全有效候选；并列按固定规则处理。
- partial-label loss：概率和为 1；全候选区间 loss/梯度为 0；有限差分和自动微分吻合；padding/重复候选不重复计权。
- 推理公平性：冻结权重无梯度；选出的窗口正是实际送给 TimeRCD 的窗口；最终输出每个输入时刻恰有一个分数。

### 端到端 smoke test

用极小数据清单跑完整流程到结果 CSV：每个序列应有固定基线、候选分数、教师状态、学生预测、分数长度和错误原因。随机抽样重新运行缓存条目，验证数值和 checkpoint/hash 匹配。另用独立进程重载学生与配置，证明部署推理不依赖训练进程状态。

## 8. 建议执行命令与开发节奏

建议 CLI 明确拆分阶段，以下是**拟实现的接口示例，不是现有命令**：

```bash
# 1) 检查权重/数据来源并固化拆分
python -m adaptive_window.cli prepare --config configs/aw_uni_v1.yaml

# 2) 生成官方候选和密集窗口异常分数（先开发集，后锁定评测集）
python -m adaptive_window.cli score --split dev --windows 1000,2000,4000,8000,16000
python -m adaptive_window.cli score --split dev --windows 1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000

# 3) 探针与教师诊断，再计算完整教师表
python -m adaptive_window.cli teacher --split dev --diagnostic-sample N
python -m adaptive_window.cli teacher --split train --config configs/aw_uni_v1.yaml

# 4) 只从训练语料生成学生特征、训练与来源分组交叉验证
python -m adaptive_window.cli features --split train
python -m adaptive_window.cli fit --config configs/aw_uni_v1.yaml

# 5) 锁定版本后做确认性评估
python -m adaptive_window.cli evaluate --split eva --checkpoint runs/<run_id>/student.pkl
```

先实现 P0 + P1 的 5–10 条 smoke test；接着开发集窗口效用扫描，通过关卡 1 后再做教师探针；关卡 2 通过后才扩展训练语料与学生。GPU 时间预算以小规模实测外推，不从方案中的粗略估计承诺工期。

## 9. 每阶段的 go/no-go 决策

| 阶段 | 继续条件 | 未通过时的处理 |
|---|---|---|
| P0 | 权重、推理、patch/mask、归一化和对齐全部可复现 | 修正接口/协议；若重建头或掩码无法核实，暂停主教师，不静默替换方法。 |
| 关卡 1 | W5 同候选逐序列 oracle 有预先定义的实际收益空间；简单长度/统计尺度基线不能解释全部空间 | 停止自适应提升主张或重新定义研究问题。 |
| 关卡 2 | 教师稳定、有信息覆盖，区间内部质量和单窗遗憾优于锁定固定基线 | 诊断探针/一致性规则假设；失败版本归档，不能先大规模训练学生。 |
| 关卡 3 | 学生在同等标签权限下对最强可部署固定/启发式基线有实际收益，CI 足以支持结论，成本满足目标 | 结论为失败或不确定；按方案区分教师噪声、特征不可预测、学生优化和域偏移。 |
| 最终测试 | 协议、规则、代码和阈值已锁定；数据来源与缓存完全审计 | Eva 被查看后再改动只作探索性分析，另找独立确认集。 |

## 10. 最终报告最少表格

1. 数据来源/分割/样本量与排除清单；模型 checkpoint 哈希和配置。
2. 候选窗口及短序列有效类别数；每窗口效用曲线与 W5/W_eval oracle。
3. 关卡 1 逐序列/逐数据集余量、固定基线和长度/周期基线。
4. 教师有效率/弃权率/稳定性、区间命中、区间内近优比例、遗憾及教师消融。
5. 学生/基线主结果，逐序列与逐数据集双汇总、配对置信区间、失败指标数。
6. 特征/损失/语料/候选数消融，合成机制控制实验。
7. 端到端时间、特征成本、学生成本、TimeRCD 前向、显存、块数及 K 窗口集成成本。
8. 所有偏离预注册方案的事项与探索性分析标记。

## 11. 当前已知注意事项

- 方案中默认的 `q_norm=0.5, rho=0.15, M=4, d_min=W1/2`、`q_top`、`ε`/`κ` 初值和最小共同有效位置数都是待标定/待核实，不是已经证实的最佳参数。
- 论文敏感性表与当前 2026.04 官方权重不是同一权重版本；窗口偏好和余量须用当前 checkpoint 重新测量。换权重后重建教师标签并重训学生。
- 密集窗口分数缓存可共用于 oracle 诊断，但主教师标签/学生候选不得混入 W_eval 的额外候选。
- 学生“只凭原始序列”是指输入不包含 TimeRCD 输出；候选窗口相关项 `psi(x,W)` 仍由候选 W 与原序列长度/统计尺度计算。
- 当前阶段不纳入 multi；多变量下 token/算力、跨通道聚合、patch 重建目标和窗口收益都需独立设计与验收。
