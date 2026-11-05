# TimeRCD 诊断实验

本页保留已有实验的实现说明和复现入口。研究入口为[问题定义](../TimeRCD_问题定义与研究边界.md)和[实验记录](../EXPERIMENT_LOG.md)，当前不指定下一项实验。

`protocols/`和各实验包中的协议保留执行前原文，供追溯当时设计、验收和停止条件；其中的候选方向与未来时态不作为当前指令。报告保留已执行设计、数值、失败和解释限制，已清除后续路线建议。`runs/`中的历史快照和原始产物不改动。

## 目录与职责

```text
experiments/
├── configs/                # 纳入版本管理的实验配置
├── common/
│   ├── config.py           # 配置校验、路径解析
│   ├── data.py             # 合成样本、TSB 数据读取、长度分层
│   ├── metrics.py          # 阈值无关指标和数值差异
│   └── artifacts.py        # 版本、指纹、源码快照、结果写入
├── stage1/
│   ├── inference.py        # 两条推理路径和尾窗补齐探针
│   ├── effects.py          # 尾窗回接、完整序列指标
│   ├── select.py           # 分层冻结开发诊断集
│   ├── analyze.py          # 成对效应、窗口范围和序列 bootstrap
│   ├── audit_precision.py  # 单例 float32/float64 数值复核
│   └── run.py              # 阶段一入口
├── selections/             # 冻结的选样清单、协议和配置
├── reports/                # 已执行实验的结论与产物索引
├── tests/                  # 无官方权重依赖的小型 CPU 回归测试
└── runs/                   # 每次执行独立目录，Git 忽略
```

配置和代码进入Git；运行结果保存在`runs/`。下列命令是既有实验的复现说明，不表示已授权重新执行。

## 执行

从仓库根目录执行：

```bash
# 边界长度 smoke：7 条短合成序列 × 3 个窗口，CPU
conda run -n RCD_yyd python -m experiments.stage1.run \
  --config experiments/configs/stage1_smoke.json

# 指定的一条真实序列 × 4 个窗口
conda run -n RCD_yyd python -m experiments.stage1.run \
  --config experiments/configs/stage1_tsb_sample.json

# 无权重的小型回归测试
conda run -n RCD_yyd python -m unittest discover -s experiments/tests -v
```

配置中的相对路径均以**仓库根目录**为基准，`--config` 路径以命令工作目录为基准。必须提供本地 checkpoint；不会自动下载。首版固定单变量，实际 Patch 长度沿用推理 API 的 16。

每个正式实验复制配置并使用可辨识的 `name`。TSB 配置必须显式列出 `files`，不会自动扫描全量数据；`split: test` 按文件名 `_tr_<长度>_` 切分，`full` 使用整条序列。数据必须为单值列加末尾 `Label`，标签为 0/1；遇到 NaN、非法标签、空测试段直接失败，不删除数据行。

CPU 配置会在导入 PyTorch 前隐藏 GPU，以避开官方 loader 先自动选择 CUDA 的行为。GPU 实验将 `device` 改为 `cuda`，并在启动前用 `CUDA_VISIBLE_DEVICES` 选择一张卡，例如 `CUDA_VISIBLE_DEVICES=1 conda run ...`。配置中的 `threads` 控制 PyTorch CPU 线程数。种子、确定性算法和 TF32 设置会被记录；跨硬件／软件版本不承诺逐位一致。

## 实验比较内容

### 1. 推理后端一致性

公共 `TimeRCDDetector.predict` 与 `models/TimeRCD.py` 的实际 `zero_shot` 路径使用同一个已加载、处于 eval 模式的模型。比较有效点上的概率和 logit 差，并核对旧路径补齐后的原始输出长度。

旧路径每条序列使用新的 tester 外壳，匹配 `main.py` 每次创建 tester 的生命周期，避免短序列修改 `win_size` 后影响下一条序列。该比较覆盖标准化、窗口、collate、forward 和有效输出裁剪；**不验证旧 checkpoint loader、`model_wrapper.py` 的共享配置行为或整套 `main.py` 指标流程**。

### 2. 尾窗补齐探针

先对整条推理输入按官方实现标准化，三个条件共用完全相同的有效数据和位置：

| 条件 | 外层输入 | 有效掩码 |
|---|---|---|
| `current` | 当前实现：末值重复补到实际窗口长度 | 只有原始点有效 |
| `zero` | 无效位置改成 0，保持相同输入长度 | 同上 |
| `compact` | 仅传入尾窗有效点，编码器自行补到 Patch 边界 | 输入点均有效 |

只比较有效点。`compact` 与 `zero` 的差异检查全无效 Patch 对有效输出的影响；`current` 与另外两者的差异观察部分有效 Patch 内补齐值的影响。额外比较单独运行 `current` 尾窗与整条序列推理对应片段，检查批量及输出对齐。

短于配置窗口的序列默认使用自身长度，三个外层条件相同。配置 `force_short_padding: true` 时，额外比较 `short_forced_zero` 和 `short_forced_edge`：保持标准化及有效掩码不变，补到配置窗口长度。该开关默认关闭。

`zero`、`compact` 尾窗输出接回完整序列后，重新计算 AP 和 ROC-AUC；无外层补齐时两种干预为严格 no-op。`tail_replay` 用重新运行的当前尾窗接回序列，作为纯执行数值误差对照。当前探针不覆盖移动窗口起点、重叠窗口或所有中间窗口的诊断。

### 3. 指标与判定

- `metrics.csv` 记录公共 API 的 ROC-AUC 与 **Average Precision（AP）**。AP 不是仓库原指标函数的 PR 曲线梯形面积，不能直接混用名称或比较数字。
- 单一类别样本标记 `single_class`，指标为空；计算异常直接记录失败。
- 原始长度、实际输入长度、窗口数、有效 Patch 数、末窗有效比例及 Patch 对齐情况逐例记录。
- 数值检查包含：两后端一致、`compact`/`zero` 一致、单独尾窗与批量推理一致。以配置 `atol`、`rtol` 判定；非有限输出或形状错误直接失败。
- `current` 相对 `zero` 的差异是观察量，不自动判为代码错误，也不说明检测指标一定受损。数值检查失败退出码为 2，运行异常退出并保存 traceback。

这些结果用于定位实现因素。当前脚本不包含阈值优化、重型 VUS 指标、跨训练种子区间估计或正式基准结论。`analyze.py` 的区间按序列重采样；每个序列的多个窗口不作为独立样本。

## 每次运行的回溯材料

启动时打印独立目录 `runs/<UTC时间>_<名称>_<随机后缀>/`，同名配置重复执行也不会覆盖已有目录。

| 文件 | 内容 |
|---|---|
| `config.input.json` / `config.resolved.json` | 原始与解析后配置 |
| `manifest.json` | 状态、命令、环境、设备、Git commit/status、权重 SHA-256 |
| `environment.packages.json` | 当前 Python 环境的完整已安装包名与版本 |
| `working_tree.patch` | 相对 HEAD 的已跟踪文件差异 |
| `source/` / `source.sha256.json` | 源码与配置快照，含未提交的实验代码 |
| `data_manifest.json` | 数据来源、原 CSV 指纹、划分偏移、输入快照指纹 |
| `arrays/input_s*.npz` | 实际输入值和标签，可恢复输入坐标 |
| `arrays/s*_w*.npz` | 有效点分数、logits 及三个尾窗探针输出 |
| `backend_parity.csv` | 两条推理路径的逐例差异 |
| `padding.csv` | 三种补齐条件的尾窗差异 |
| `tail_alignment.csv` | 独立尾窗与原推理尾部的差异 |
| `metrics.csv` | 基线指标与长度／Patch 分层字段 |
| `condition_metrics.csv` | 各补齐条件的整序列指标、相对基线变化、尾窗异常覆盖 |
| `short_padding.csv` | 可选的短序列等长补齐检查 |
| `selection.json` | 使用冻结诊断集时，复制其选择协议和原数据 SHA-256 |
| `summary.json` | 数值检查通过／失败计数 |
| `run.log` / `error.txt` | 推理日志／异常堆栈 |

源码快照包含已跟踪的 Python、JSON、TOML、Markdown 和本实验目录的代码、配置；权重只记录路径与 SHA-256，不复制。恢复运行时需使用匹配的权重及环境。`manifest.status` 只有 `completed` 才表示本轮数值检查全部通过；中断后留下的 `running` 目录不能当作已完成结果。

## 已冻结的真实基准序列诊断集

`selections/stage1_real_v1/` 固定 24 条开发诊断序列、来源 9 类，低／中／高异常占比各 8 条。候选来自已有 700 条评测元数据，只读取文件名、长度、训练段长度三列；模型指标不参与选择。实际读取数据后筛去单类标签，最终有 531 条符合 256–20000 测试点长度范围的候选。

按 5000 点窗口分层：短序列 8 条、完整多窗 1 条、末窗有效比例不超过 20% 的 7 条、其余末窗 8 条。优先均衡来源、异常占比、Patch 对齐情况，最后以 seed 与文件名的 SHA-256 排序。所有入选文件的 SHA-256 会在实验启动时校验。

```bash
# 重新生成清单时使用一个新的目录名，已有目录不会被覆盖
conda run -n RCD_yyd python -m experiments.stage1.select \
  --output-dir experiments/selections/stage1_real_v2 --device cuda

# 运行已冻结的 v1；CUDA_VISIBLE_DEVICES 选择实际 GPU
CUDA_VISIBLE_DEVICES=2 conda run -n RCD_yyd python -m experiments.stage1.run \
  --config experiments/selections/stage1_real_v1/config.json

# 仅分析已保存结果，不重新推理；analysis/ 已存在时拒绝覆盖
conda run -n RCD_yyd python -m experiments.stage1.analyze <run_directory>

# 对短序列单例做 CPU 精度复核，独立保存 float32/float64 输出
conda run -n RCD_yyd python -m experiments.stage1.audit_precision \
  <run_directory> --case s0003_w5000 --device cpu
```

首次执行的结果和数值复核见 [阶段一真实数据诊断记录](reports/stage1_real_v1.md)。精度复核的 `completed` 只表示复核执行完成，各精度的判定见 `precision.json`；它不改变原实验的 `checks_failed` 标记。

## 阶段二：全层标签辅助硬屏蔽（历史名Oracle）

代码集中在 `stage2/`：`masking.py` 管理标签到 Patch 的映射及注意力干预，`synthetic.py` 生成受控数据，`inference.py` 执行窗口对照，`run.py` 记录运行，`analyze.py` 做成对分析。配置为 `configs/stage2_oracle_v1.json`。

- 沿用阶段一冻结的 24 条开发序列、官方单变量权重和四个窗口，5000 点为主窗口。
- 每组包含原模型、全层 Oracle 和五个预定随机屏蔽种子。随机组在每个窗口保留与 Oracle 相同数量的有效 Patch；汇总先分别算指标再平均，不做分数集成。
- Oracle 仅允许所有有效点均正常的 Patch 作为 key/value。混合 Patch 一并屏蔽；每层检查实际收到的掩码，查询、残差、位置和输出不变。
- 无干净参照的窗口中，Oracle 没有定义；各组统一保留原模型分数，并在 `windows.csv` 标记。`full_fallback` 为这种显式回退下的整序列指标，`eligible` 仅评估各组共同可干预的点。单类标签的指标留空。
- 合成诊断为长度 5000、每个事件 5 点、固定 +4 偏移，异常点数 5/500/2000（0.1%/10%/40%），3 个生成种子 × 前部/后部/打散三种布局，共 27 条。事件之间至少 5 个正常点，各密度使用嵌套事件集。
- 合成组使用对应干净生成信号的固定均值/标准差，防止异常点数改变预处理统计量；真实组仍用官方整序列标准化。两种设置分别汇总。

```bash
# 先确认所选 GPU 空闲
CUDA_VISIBLE_DEVICES=2 conda run -n RCD_yyd python -m experiments.stage2.run \
  --config experiments/configs/stage2_oracle_v1.json

conda run -n RCD_yyd python -m experiments.stage2.analyze <run_directory>
```

运行目录保存：`metrics.csv`（逐条件 AP/ROC-AUC）、`windows.csv`（混合 Patch、干净参照和回退覆盖）、`checks.csv`（无干预及公共 API 对齐）、`arrays/*_masks.npz`（每个窗口每个条件的实际掩码）、完整分数、输入和源码快照。每次创建独立运行目录，分析目录也拒绝覆盖。

解释时以 Oracle 相对原模型及随机组的**成对改善**为准。真实组按序列 bootstrap；合成组先平均同一生成种子的三个布局，再按生成种子 bootstrap，只有三个生成种子，区间是探索性描述。Oracle 需要标签、随机组也用标签决定屏蔽数量，均不是部署算法；随机组未匹配空间布局，因此只能诊断当前干预机制。

首轮结果见 [阶段二 Oracle 实验报告](reports/stage2_oracle_v1.md)：合成中高密度组恢复，但基准序列的 5000 点主窗口明显退化。该结果只约束当前硬屏蔽干预，不是干净参照的性能上界。

## 阶段二后续：分数路径与归一化交叉对照

`stage2_followup/` 读取上一轮冻结输入和分数，原目录不被覆盖：

- `diagnostics.py`：正常/异常点的配对分数变化；ROC-AUC 的窗内/跨窗正负样本对精确分解，平分并列分数。AP 不作这种加性分解。
- `offline.py`：分析原 123 组结果，保存逐序列分数变化、排序变化、逐窗口指标，以及读取的父运行文件指纹。
- `run.py`：在原 27 条合成输入上，将固定干净统计量切换为官方整序列标准化，保留权重、标签、窗口和全部随机/标签掩码；逐一验证实际掩码相同及公共 API 一致性。
- `analyze.py`：按生成种子汇总两个预处理条件下的屏蔽效应及其差值；保存分析脚本快照。

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n RCD_yyd python -m experiments.stage2_followup.run \
  --config experiments/configs/stage2_followup_v1.json
conda run -n RCD_yyd python -m experiments.stage2_followup.analyze <run_directory>
```

结果见 [阶段二后续诊断](reports/stage2_followup_v1.md)。类别平均分数变化不等于固定阈值下的误报/漏报变化；窗内/跨窗 AUC 的贡献同时受对应样本对数量影响，不能把贡献占比直接解释成因果归因。

## 注意力连接路径消融

执行前协议：[stage2_attention_paths_v1.md](protocols/stage2_attention_paths_v1.md)；配置：`configs/stage2_paths_v1.json`。代码位于 `stage2_paths/`，分别负责连接适配、冻结输入、推理检查、运行记录和预定成对分析。

固定5000点窗口，使用原24条基准序列和官方整序列标准化的27条合成序列。A保留原连接；B删除选中Patch自身连接；C保留自身但禁止其他查询读取选中Patch；D删除选中Patch整列。标签选择和五个父运行随机集合分别用于B/C/D，每组分最后一层和所有层两种范围，共37个条件/序列。

`attention.py`仅在上下文管理器内替换单个模型实例的attention forward，保留原Q/K/V投影与RoPE，统一使用显式`[1,1,query,key]`掩码。退出时恢复原方法，不修改生产推理代码或全局torch函数。

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n RCD_yyd python -m experiments.stage2_paths.run \
  --config experiments/configs/stage2_paths_v1.json
conda run -n RCD_yyd python -m experiments.stage2_paths.analyze <run_directory>
```

运行先在预定边界样本（索引0/8/9/24，非效果选样）进行验收；通过后才运行完整51条。验收比较原API、父运行A/D、旧整列掩码、last层输入/Q/K/V、C/D非选中位置不变性，以及标准化后选中Patch输入扰动的传播。完整运行继续检查A/D对齐和输出不变性，前后核对模型状态指纹。关键数值失败会记录差值及失败数组并停止；若出现父运行中没有的空参照窗口，停止排查，不生成单臂特殊回退的机制结论。

`gates/`保存验收产物；`layer_masks.csv`保存每次forward每层实际连接矩阵指纹和有效查询连接数；`arrays/*_selections.npz`保存从父运行复制的实际Patch集合，足以结合四条件规则重建矩阵；`checks.csv`、`score_shifts.csv`、`ranking.csv`、`metrics.csv`记录检查和所有分析指标。`parents.sha256.json`、`protocol.md`及源码/权重指纹用于回溯。已有输出不覆盖。

主比较为基准all条件的C−D，以及该差值的标签版本减随机均值；同步报告C−A。随机种子在序列内部平均，基准按序列重采样，合成先按生成种子平均布局再重采样；保存逐来源和剔除单来源敏感性结果。恢复自身连接同时改变注意力分母和参照数量，不能将差值直接解释成独立的自身语义贡献。

首轮执行结果见[注意力连接路径消融报告](reports/stage2_paths_v1.md)：基准组保留自身连接缓解部分退化，但未恢复到原模型；合成中高密度组全层保留自身连接反而减少收益。执行前协议保留原版本，实际运行状态以报告及运行manifest为准。

## 混合／全异常Patch参照消融

[执行前协议](protocols/stage2_patch_types_v1.md)、`configs/stage2_patch_types_v1.json`和`stage2_patch_types/`记录本轮设计及实现。沿用前轮51条输入、5000窗口、官方标准化及冻结权重，将含异常Patch拆为M（混合有效点）、P（全异常有效点），U为并集。每类分别做全层C（保留自身）和D（整列删除），配五个数量匹配随机集合；加A共37条件/序列。

随机U复用父运行，M/P按固定SHA256种子互斥分割，数量与标签集合对应。`selections.py`负责集合，`data.py`在模型调用前冻结输入及覆盖清单，`inference.py`负责推理与逐点检查，`run.py`管理验收和运行，`analyze.py`完成预定统计。空集合复用A并记录；不把无操作样本静默排除。

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n RCD_yyd python -m experiments.stage2_patch_types.run \
  --config experiments/configs/stage2_patch_types_v1.json
conda run -n RCD_yyd python -m experiments.stage2_patch_types.analyze <run_directory>
```

运行前先确认所选GPU空闲。预先验收样本由标签覆盖选择：首条无P基准、有P基准、partial基准及合成序列，去重。A对齐公共推理和父A，U_C/U_D对齐父全层结果，各集合C/D非选中输出一致；验收另查旧整列实现及选中输入扰动的传播。保留`preflight.json`、`coverage.csv`、选择与输入快照、`noops.csv`、每层mask指纹、全部分数及模型状态指纹。

主比较为基准M_C−A及其减随机均值，附带P/M覆盖子集、C−D和U−M−P+A交互、来源敏感性及合成密度分组。随机指标先在序列内平均；合成按生成种子平均布局后重采样。M/P删除量不同，不能把两者效果直接解释成单位Patch价值；混合Patch屏蔽同时改变正常和异常参照信息，仍不能单独识别正常信息的因果作用。

执行结果见[混合／全异常Patch参照消融报告](reports/stage2_patch_types_v1.md)：仅限制混合Patch的主比较未显示改善；仅限制全异常Patch、保留自身连接仍有退化，因而混合Patch正常点参照损失不能解释全部损失。51×37条预测完成，7948项运行检查通过；合成P为空、M=U，属于无操作及父结果回归验证。

## P_C离线点组与排序损失定位

当前研究状态统一见根目录[实验记录](../EXPERIMENT_LOG.md)。本轮[协议](protocols/stage2_pc_localization_v1.md)固定读取E05的24条基准输出，分析四点组及四类异常—正常排序对，再拆窗内/跨窗；完整序列AP单独报告。13条含P、11条无P预定分组，五随机先在序列内平均指标，序列配对bootstrap及来源敏感性均保存。

`stage2_pc_localization/`中，`data.py`核对父数据/类型/选择/覆盖；`diagnostics.py`实现无模型依赖的点组与排序分解；`statistics.py`负责身份配对、区间和来源分析；`analyze.py`创建独立离线运行并保存源码和父产物指纹。

```bash
conda run -n RCD_yyd python -m experiments.stage2_pc_localization.analyze \
  --config experiments/configs/stage2_pc_localization_v1.json
```

[结果报告](reports/stage2_pc_localization_v1.md)：含P集合AP差−0.230379，P异常—N正常对贡献最大的平均AUC损失，但该类点对权重约77%，且贡献区间跨零。多来源存在AP损失，不能据AUC分解直接解释AP退化。48项测试、5493项运行检查通过，零新增推理。

## 最后一层干净反事实参照替换

[协议](protocols/stage2_clean_reference_v1.md)、[结果报告](reports/stage2_clean_reference_v1.md)及`configs/stage2_clean_reference_v1.json`记录E07。代码位于`stage2_clean_reference/`：数据生成与共享标准化、query-dependent非对角K/V替换、推理验收、种子级配对统计、运行溯源分别拆分。

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n RCD_yyd python -m experiments.stage2_clean_reference.run \
  --config experiments/configs/stage2_clean_reference_v1.json
```

8种子×3形态×3事件长度×2对齐共144条，A/C/V/KV/C_all共720条正式预测；12例gate、27条旧合成回归另记。55项测试及4812项运行检查通过。长事件V−A的P—N条件AUC +0.020107但AP −0.052355，KV−V的AP +0.041559；干净替换并非普遍改善。层范围、形态和对齐条件有异质性。

## 背景配对与共同P位置控制

E08的[协议](protocols/stage2_alignment_controls_v1.md)、[报告](reports/stage2_alignment_controls_v1.md)和`configs/stage2_alignment_controls_v1.json`固定384条配对输入、A及full/common×last/all共1920条预测。offset8为整条信号和标签共同平移；common是两对齐的绝对P位置交集。`stage2_alignment_controls/`分离生成验收、推理、配对交互统计、运行与独立离线恢复。

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n RCD_yyd python -m experiments.stage2_alignment_controls.run \
  --config experiments/configs/stage2_alignment_controls_v1.json
```

原执行推理后在统计阶段外部超时，保留`interrupted`；独立离线恢复`completed`，没有重复推理。61项测试、28728项原数值检查、8448项恢复检查通过。AP交互I_full=−0.360421，I_common=+0.026556；集合控制显著改变结果但不是纯数量因果识别。最终统计路径详见报告。

## 同输入、同数量的P位置选择

E09的[协议](protocols/stage2_p_positions_v1.md)、[报告](reports/stage2_p_positions_v1.md)及`configs/stage2_p_positions_v1.json`固定复用E08 offset0的192条输入，每事件保留首、末或五固定随机之一，屏蔽其余P并保留自身。A、full与匹配选择共17条件、3264条正式预测，另有24例gate。

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n RCD_yyd python -m experiments.stage2_p_positions.run \
  --config experiments/configs/stage2_p_positions_v1.json
```

67项测试、58416项运行检查、24240项零推理输出验收通过。全层保留末块相对首块AP差−0.024854；层范围交互−0.001082（区间跨零），但32/128点与正/负偏移分层方向相反。全层随机指标均值AP0.501962，首/末0.570465/0.545612；同数量具体选择仍影响检测。

`stage2_p_positions/`分离父输入/选择验收、推理、种子配对统计和运行；`validate.py`将保存分数重算、有限性及指标身份核对另存到`output_validation/`。主运行在统计前保存`inference_summary.json`，给运行足够超时；本轮约223秒，无中断。

## 高分区排序与保留/屏蔽P定位

E10的[协议](protocols/stage2_rank_localization_v1.md)、[报告](reports/stage2_rank_localization_v1.md)及`configs/stage2_rank_localization_v1.json`固定复用E09全部保存输出。`stage2_rank_localization/`分离父产物校验、固定/角色分组、同分完整阈值与平均排名、配对种子统计和离线入口。

```bash
conda run -n RCD_yyd python -m experiments.stage2_rank_localization.analyze \
  --config experiments/configs/stage2_rank_localization_v1.json
```

192条×17条件×两分数，共6528行、6个预定召回目标；74项测试、132089项检查通过，零新增推理。50%召回下全层末块相对首块precision差−0.029138；固定末块排名后移，首块在短/长事件方向相反。10%召回precision差为正、75%以后更负，均值与单一阈值不足以解释完整排序。

## 固定首末掩码的累积屏蔽深度

E11的[协议](protocols/stage2_depth_controls_v1.md)、[报告](reports/stage2_depth_controls_v1.md)与`configs/stage2_depth_controls_v1.json`固定192条父输入及首末选择，比较最后1/2/4/8层C屏蔽，加A共9条件、1728条正式预测。

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n RCD_yyd python -m experiments.stage2_depth_controls.run \
  --config experiments/configs/stage2_depth_controls_v1.json
```

79项测试、127824项运行检查通过。位置AP差的相邻深度变化总体区间均跨零，但d1→d2在短/长事件为+0.020740/−0.018746；首/末总体AP在d4为0.619991/0.591161，d8下降，深度效果非单调。

`stage2_depth_controls/`分离独立深度适配器、父数据、推理/读出、统计及运行。d1/d8逐点对齐E09、全部端点诊断对齐E10，干预起点及此前QKV固定，模型和父产物不变。运行约406秒，统计前保存`inference_summary.json`，应给足超时。

## 最后两层2×2屏蔽开关

E12的[协议](protocols/stage2_layer_factorial_v1.md)、[报告](reports/stage2_layer_factorial_v1.md)与`configs/stage2_layer_factorial_v1.json`固定首末掩码，对第7/8层分别开关C屏蔽，两选择共享A，共7条件、1344条正式预测。

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n RCD_yyd python -m experiments.stage2_layer_factorial.run \
  --config experiments/configs/stage2_layer_factorial_v1.json
```

85项测试、116160项运行检查通过。第7层单独屏蔽位置AP差Δ10=+0.004924，与两层交互首末差K=−0.003927相互抵消，合计复现E11的+0.000997；K在短/长事件为+0.003129/−0.010983。

`stage2_layer_factorial/`分离显式层适配器、父数据核验、推理、统计和运行；00/01/11逐点及全部读出对齐E11，10/11第7与第8层input/QKV完全相同，模型及父产物不变。运行约327秒，推理验收先落盘再统计。

## logit加性替代离线读出

E13的[协议](protocols/stage2_logit_additivity_v1.md)、[报告](reports/stage2_logit_additivity_v1.md)与`configs/stage2_logit_additivity_v1.json`复用E12输出，float64构造l_add=l10+l01−l00；用logits AP将总交互写成加性替代交互J_add与真实11相对替代的差R。

```bash
conda run -n RCD_yyd python -m experiments.stage2_logit_additivity.analyze \
  --config experiments/configs/stage2_logit_additivity_v1.json
```

3456行读出、384行残差统计，91项测试、78799项检查通过，零新增推理。总体K_add=−0.007350、K_R=+0.003423相互抵消；负AP交互首末差已可在加性logits下出现，不能直接归于网络输出负交互。

`stage2_logit_additivity/`分离加性构造、父核验、读出、统计与运行。父原条件全部读出及J/K回归、派生数组落盘复核、父产物和权重哈希通过；运行约226秒，读出验收先落盘再统计。

## 固定点组残差回放

E14的[协议](protocols/stage2_residual_replay_v1.md)、[报告](reports/stage2_residual_replay_v1.md)与`configs/stage2_residual_replay_v1.json`固定E13 logit基线，按F/L/I/N的16个子集回放残差，两选择共6144行读出及12288行上下文边际。

```bash
conda run -n RCD_yyd python -m experiments.stage2_residual_replay.analyze \
  --config experiments/configs/stage2_residual_replay_v1.json
```

98项测试、143380项检查通过，零新增推理。总体末−首平均AP边际D_N=+0.006622、D_L=−0.005465；D_L在短/长事件方向反转。该归因不证明正常参照普遍可靠。

E14于07:49误重复执行一次，保留第二次运行但不计独立证据；两次均零推理、143380检查通过且父产物不变。E15引用第二次运行并独立核验E13全链。

## 正常点残差阈值集合转移

E15的[协议](protocols/stage2_normal_replay_transitions_v1.md)、[报告](reports/stage2_normal_replay_transitions_v1.md)与`configs/stage2_normal_replay_transitions_v1.json`比较add、N-only（mask8）和full（mask15）的六召回点集，共6912条阈值记录、32256条分组转移。

```bash
conda run -n RCD_yyd python -m experiments.stage2_normal_replay_transitions.analyze \
  --config experiments/configs/stage2_normal_replay_transitions_v1.json
```

106项测试、514369项检查通过；保存membership后另逐行重计39168组。零模型加载、零新增推理，父链及权重不变。N-only净误报变化随召回变号：首块10%/90%为+15.188/−26.349，末块10%/100%为−18.135/+7.479。N-only异常点和阈值严格不变，full阈值则全部改变，不能把full差归于N单组。

`stage2_normal_replay_transitions/`分开阈值/转移核、父链和数组核验、读出、种子配对统计与运行。报告包含完整24格。

## 正常分数与正例阈值来源交叉

E16的[协议](protocols/stage2_score_threshold_cross_v1.md)、[报告](reports/stage2_score_threshold_cross_v1.md)与`configs/stage2_score_threshold_cross_v1.json`固定四格00/10/01/11为正常/正例来源add或full，对齐E14 mask0/8/7/15。

```bash
conda run -n RCD_yyd python -m experiments.stage2_score_threshold_cross.analyze \
  --config experiments/configs/stage2_score_threshold_cross_v1.json
```

9216四格读出、11520条N转移、2304点交互计数、18432比较；116项测试、423937运行检查通过，保存点集后独立重计41472项。零模型加载/新增推理，父链及权重不变。

首块10%召回FP分解S0+T0+I=+15.188−46.021−2.474=−33.307；末块100%为+7.479−282.984−5.495=−281.000。阈值来源差能解释部分方向反转，但不同召回和因素格并不一致；交互是指定输出与阈值读出的性质，不是网络层内因果结论。

`stage2_score_threshold_cross/`分开交叉核、父产物验证、读出、种子统计与运行。报告包含全部六召回及完整24格交互，全部八比较的FP/FPR、绝对读出及点转移均有完整分层表。

## R与S及实现排查

E17/R的[执行前协议](stage2_cluster_probe/protocols/real_similarity_v1.md)和[报告](stage2_cluster_probe/reports/real_similarity_v1.md)记录真实开发序列的零推理相似度分析。E18/S的[执行前协议](stage2_isolated/protocols/isolated_v1.md)和[报告](stage2_isolated/reports/isolated_v1.md)记录孤立异常、随机删除数量、微型掩码、Patch边界及独立产物验收。具体数值和运行目录见报告及根目录实验记录；本轮H未实现、未执行，损失机制仍未解释。
