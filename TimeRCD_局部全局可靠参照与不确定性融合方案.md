# TimeRCD 局部上下文、窗口外可靠参照与不确定性融合方案

## 0. 方案状态与核心定义

本文档记录一个条件性的候选架构，不表示其模块已经通过实验验证。问题定义见[TimeRCD 问题定义与研究边界](TimeRCD_问题定义与研究边界.md)，实验顺序以[诊断优先实验路线](TimeRCD_诊断优先实验路线.md)为准：在实现新增模块前，先完成窗口实现诊断、Oracle 参照污染实验和窗口敏感性归因。CrossAD 已提出跨窗口全局原型上下文，因此本方案的窗口外 Patch 读取只是待比较的候选机制，不是“首次跨窗建模”。多尺度不属于本方案成立的前提，是否加入须由窗口敏感性诊断决定。

若前置诊断确认 TimeRCD 存在不能由实现问题解释的上下文失效，本候选架构才考虑改进其上下文构建方式：以较短窗口进行局部编码，再研究目标条件的窗口外 Patch 读取。CrossAD 的查询路由和全局原型记忆只作为可能的设计参考；是否借鉴其中具体机制，须根据 TimeRCD 的零样本设定和诊断结果另行判断，不将其预设为基线或必做分支。只有参照污染经 Oracle 对照支持后，才研究候选可信度估计与局部、全局证据融合。

在当前逐 Patch 检索候选中，窗口是局部编码和计算的分块单位，跨窗联系由窗口外参照分支建立；这只是待比较的一种实现，并非已选定的结构。首版候选范围为单变量、双向离线异常检测。多变量扩展保留独立的通道轴，后续加入跨通道建模。

需要检验的核心假设是：**局部上下文与可信的窗口外参照能够提供互补证据；逐 Patch 融合可以在两路证据一致时增强判断，在两路冲突或参照不足时正确提高不确定性。**这是一项待实验验证的假设，不是已有结论。

## 1. 研究问题与设计边界

### 1.1 诊断通过后的候选研究问题

> 若前置诊断确认固定窗口造成可复现的性能损失，且 Oracle 对照支持参照污染假设，能否在保留 TimeRCD 局部相对上下文建模能力的基础上，为目标 Patch 构建有用且可信的窗口外上下文，并检验其相对原始 TimeRCD 和验证集选出的强固定窗口设置的收益？

### 1.2 待区分的机制假设与候选响应

| 待检验假设 | 可观察表现 | 若获支持，可比较的响应 |
| --- | --- | --- |
| 上下文可访问范围受限 | 固定窗口之外可能存在有用信息；窗口长度或边界变化引起可复现性能差异 | 以强固定窗口设置为主要对照，研究目标条件的窗口外 Patch 读取；CrossAD 的机制仅作后续设计参考，多尺度仅在证据支持时考虑 |
| 参照受到异常或不适配片段影响 | 标签 Oracle 屏蔽参照优于基线及等量随机屏蔽，且效应在高异常密度下更明显 | 再研究软参照可信度估计；相似度和正常性分别评估 |

检索到相似 Patch 不等于找到了正常 Patch。重复异常也可能彼此相似。因此，只有参照污染得到实验支持时，候选全局分支才需要额外估计参照可信度，不能预先把相似度解释为正常性。

局部与全局判断不一致可能表示证据冲突或上下文不确定，不能直接等同于异常。若后续保留融合模块，需分别验证异常分数和不确定性输出的有效性。

### 1.3 首版范围

- 单变量时序，通道数为 $C=1$。
- 双向离线检测，允许目标 Patch 访问时间上位于其前后的窗口外参照。
- 局部窗口长度 $W_s$ 以原始时间点计，候选实验值为 $128$、$256$、$512$；窗口长度的敏感性作为分析，不是本方案的核心贡献。
- 沿用 TimeRCD 的 Patch 长度 $P$ 和预训练权重。若 $P=16$，每个局部窗口分别包含 $8$、$16$、$32$ 个 Patch。
- 每条序列独立建立候选集合；不同序列和 padding 位置都不能作为互相参照。
- 不在推理时使用目标序列异常标签，不在线更新正常原型库。

## 2. 与已有工作的关系

### 2.1 TimeRCD：局部相对上下文建模

TimeRCD 使用目标片段与上下文之间的关系构造异常证据，并以合成的上下文相关异常提供训练监督。已核对的公开推理代码快照按 `win_size` 将长序列切成互不交互的窗口，窗口内部由 Transformer 建模；代码细节和有效长度边界见[问题定义](TimeRCD_问题定义与研究边界.md)。TimeRCD 已正式发表于 ICML 2026（PMLR 306:62494–62529）。正式论文报告固定 5000 点主实验窗口和 1000 至 13000 点的窗口敏感性分析，并在局限性中提出大窗口异常可能影响局部参照；这些结果和讨论仍不能替代针对当前代码与数据的独立因果检验。本目录保存的出版版论文为[TimeRCD 论文 PDF](<Lan 等 - 2026 - Towards Foundation Models for Zero-Shot Time Series Anomaly Detection Leveraging Synthetic Data and.pdf>)。arXiv v5 页面另标有该版本撤回状态，该状态仅针对 arXiv 版本。

本方案保留 TimeRCD 的局部编码器、Patch 表示和点级任务头作为初始化，新增窗口外参照读取、参照可信度估计和证据融合。改变注意力范围会改变表示分布，因此新增模块需要训练或微调；加载原有 checkpoint 本身不会自动获得检索能力。

来源：[TimeRCD 代码快照 `b59e86e`](https://github.com/Adomyyd/TimeRCD/tree/b59e86e7e60b16ed696fff3fa8ced5e502fe6ee5)、[ICML 2026 / PMLR 正式论文页](https://proceedings.mlr.press/v306/lan26b.html)、本目录的[TimeRCD 出版版论文](<Lan 等 - 2026 - Towards Foundation Models for Zero-Shot Time Series Anomaly Detection Leveraging Synthetic Data and.pdf>)；[arXiv v5 版本状态](https://arxiv.org/abs/2509.21190v5)。

### 2.2 CrossAD：跨窗口全局原型上下文

CrossAD 已直接研究多尺度关联与跨窗口上下文共享。其周期感知路由器根据当前窗口的频域特征选择可学习的子序列查询；查询从当前窗口的多尺度编码中提取表示，再分配给最近的全局原型并通过 EMA 更新。解码器读取全局原型上下文，并以多尺度重建误差产生异常分数。

这一机制是本方案必须考虑的近邻工作：CrossAD 已覆盖“窗口之间共享全局信息”，因此不能把全局跨窗建模本身作为本方案的增量。它与本方案的候选差异在于，CrossAD 使用窗口级查询和训练期原型记忆；本方案拟研究目标 Patch 对窗口外实际 Patch 的读取及其参照可信度。CrossAD 没有显式的候选正常性可信度或局部、全局分歧不确定性输出。其按数据集训练的重建设定也不同于 TimeRCD 零样本推理，不能只比较论文中的分数来断言哪种机制更好。

来源：[CrossAD, arXiv:2510.12489v1，NeurIPS 2025](https://arxiv.org/abs/2510.12489v1)。

### 2.3 VETime：Patch 级动态融合

VETime 将视觉、时间和异常增强表征对齐到时间 Patch，并使用 Patch 级任务自适应权重融合不同来源。其结果支持动态门控相较于直接拼接或相加的设计动机。

本方案借鉴“在 Patch 位置依据当前证据动态分配来源权重”，不借用视觉分支，也不把 VETime 的对齐机制描述成窗口外时间检索。窗口外候选检索和参照可信度估计是本方案需要单独验证的部分。

来源：[VETime arXiv v3](https://arxiv.org/abs/2602.16681v3)、[曾核对的代码版本](https://github.com/yyyangcoder/VETime/tree/7b81a7efbae4760eafc438ffe32fa19df8de7465)。

### 2.4 正常记忆与近邻参照

[PatchCore](https://arxiv.org/abs/2106.08265) 和 [MemAE](https://doi.org/10.1109/ICCV.2019.00179) 使用正常样本表征构成记忆，再按输入查询相关记忆项；[MEMTO](https://doi.org/10.52202/075280-2525) 将记忆机制引入多变量时间序列异常检测。这些工作支持“正常模式原型可为参照可信度提供证据”，但并不直接验证当前序列中的窗口外 Patch 是否可靠。

### 2.5 关联差异与高异常密度风险

[Anomaly Transformer](https://arxiv.org/abs/2110.02642) 以关联分布差异形成异常证据，并利用异常较稀少这一假设。该思路说明关系结构可提供异常线索，也提醒我们：依赖正常/异常关联模式差异的机制可能在异常密度升高或异常重复时失效。

### 2.6 新近相关工作

[LEAP](https://arxiv.org/abs/2610.35190) 探索潜在表征新颖性、局部密度校准、局部时间对比及多路异常证据；[GSLAD](https://arxiv.org/abs/2609.15483) 探索多个正常结构原型、结构不确定性归一化和预测偏差融合。这些工作可作为多证据互补与原型置信度设计的参考；它们是新近预印本，且均未直接验证本文的窗口外可靠 Patch 检索结构。

## 3. 符号与维度

| 符号 | 含义 | 首版约束或形状 |
| --- | --- | --- |
| $B$ | 批次中的序列数 | 每条序列独立处理 |
| $T$ | 时间长度 | 原始时间点数 |
| $C$ | 通道数 | 首版 $C=1$ |
| $W_s$ | 局部处理窗口长度 | 以时间点计 |
| $P$ | Patch 长度 | 沿用 TimeRCD 配置 |
| $J$ | 序列中的局部窗口数 | 约为 $\lceil T/W_s\rceil$ |
| $N_w$ | 每窗 Patch 数 | $W_s/P$；末窗可用掩码处理 |
| $N$ | 序列内 Patch 总数 | $J N_w$，忽略 padding token |
| $d$ | Patch 隐藏维度 | 沿用预训练模型 |
| $A$ | 注意力头数 | 沿用 TimeRCD 或新增模块配置 |
| $d_h$ | 单头维度 | $d/A$ |
| $d_{\mathrm{proj}}$ | 逐时间点表示维度 | 沿用 TimeRCD 输出头 |
| $d_a$ | Patch 地址向量维度 | 地址投影输出维度 |
| $K_0$ | 初筛远程候选数 | 先取较大的候选池 |
| $K$ | 最终参与全局读取的候选预算 | $K\le K_0$ |
| $M$ | 正常模式原型数 | 由训练数据和验证实验选择 |
| $\mathcal{M}_N$ | 正常模式原型集合 | $\mathbb{R}^{M\times d}$ |
| $r_j$ | 候选 Patch $j$ 的正常性可信度 | $[0,1]$ |
| $Q_i$ | 目标 Patch $i$ 的全局参照质量 | $[0,1]$ |
| $u_j^L$ | 候选 Patch $j$ 的局部预测不确定性 | 局部异常概率的归一化熵 |

## 4. 模型结构

### 4.1 输入、分窗与 Patch 编码

输入为 $X\in\mathbb{R}^{B\times T\times C}$。首版 $C=1$。序列按时间切为长度 $W_s$ 的处理窗口，尾部 padding 由有效位置掩码排除：

$$
X_{\mathrm{win}}\in\mathbb{R}^{B\times J\times W_s\times 1}.
$$

每个窗口再按 Patch 长度 $P$ 切分并编码：

$$
X_{\mathrm{patch}}\in\mathbb{R}^{B\times J\times N_w\times P},\qquad
H^0\in\mathbb{R}^{B\times J\times N_w\times d}.
$$

Patch embedding、TimeRCD Transformer、RMSNorm、RoPE、残差、FFN 及预训练参数作为局部分支起点。局部 Transformer 对各窗口独立运行，输出：

$$
H^L\in\mathbb{R}^{B\times J\times N_w\times d}.
$$

### 4.2 窗口外候选检索

将 $H^L$ 展平为 $B\times N\times d$。目标 Patch $i$ 的地址查询由局部表示产生：

$$
a_i=\operatorname{norm}(W_a h_i^L).
$$

候选集合排除目标 Patch 所在的整个窗口：

$$
\mathcal{C}_i=\{j:\operatorname{window}(j)\ne\operatorname{window}(i),\ j\text{ 有效}\}.
$$

先按地址相似度从 $\mathcal{C}_i$ 中取 $K_0$ 个候选，作为待评估集合。首轮机制实验使用精确检索，以便测量参照机制本身；只有确认检索有效后，再考虑近似索引加速。若一条序列只有一个窗口，则没有窗口外候选，全局分支关闭。

地址表示用于搜索位置，候选读取值来自候选 Patch 的当前序列编码表示。地址相似度不能直接解释为候选正常，也不能把检索出的离散 Patch 重新编号成连续时间序列。全局读取应保留候选真实时间索引或时间间隔特征。

### 4.3 参照可信度估计

候选 $j$ 的正常性可信度 $r_j\in[0,1]$ 由可学习估计器 $g_r$ 产生。候选与目标是否相关由上一节的地址相似度负责；可信度估计主要回答候选本身是否像一个正常参照。首版输入至少包括：

- 候选在其局部窗口中的 TimeRCD 异常概率；
- 候选上下文表征到正常原型集合 $\mathcal{M}_N=\{\mu_m\}_{m=1}^{M}$ 的距离或密度；
- 候选 Patch 的有效性与局部上下文质量。

可写作：

$$
r_j=\sigma\!\left(g_r\left(p_{j,\mathrm{anom}}^L,
d(h_j^L,\mathcal{M}_N),u_j^L\right)\right).
$$

正常原型由训练阶段已知的正常合成 Patch 表征构造，涵盖多种正常状态；原型只作为可信度证据，不直接替代当前序列中的窗口外上下文。首版推理时冻结原型，不使用预测为正常的测试 Patch 在线更新，避免异常逐步污染记忆。

式中 $d(h_j^L,\mathcal{M}_N)$ 表示候选表征到原型集合的距离，首版可取经验证集尺度归一化后的最近原型距离。$p_{j,\mathrm{anom}}^L$ 和 $u_j^L$ 分别是候选 Patch 覆盖时间点的局部异常概率均值和归一化熵均值。若原型距离大或局部分支对候选判断不确定，通常应降低其参照可信度；该映射由合成训练数据监督并通过高异常密度实验检验。

可信度是软权重，不是正确性保证。候选相似度与可信度分别承担“是否与目标相关”和“候选是否像正常参照”两种职责。若局部异常分数和原型距离互相冲突，可信度估计器应降低 $r_j$；若因此使目标的总体参照质量 $Q_i$ 下降，目标不确定性 $U_i$ 随之升高。

### 4.4 可信窗口外上下文

对初筛候选按“地址相关性 + 正常性可信度”联合排序，最多保留 $K$ 个进入 $\mathcal{G}_i$；同时限制候选过度集中于同一来源窗口。全局注意力权重同时结合检索相关性和候选可信度：

$$
w_{ij}=\operatorname{softmax}_{j\in\mathcal{G}_i}\left(
\frac{q_i^\top k_j}{\sqrt{d_h}}+\log(r_j+\epsilon)+b(\Delta t_{ij})\right),
$$

$$
c_i^G=\sum_{j\in\mathcal{G}_i}w_{ij}v_j,\qquad
Q_i=\sum_{j\in\mathcal{G}_i}w_{ij}r_j.
$$

其中 $\mathcal{G}_i$ 是最多 $K$ 个候选；$b(\Delta t_{ij})$ 是可选的时间间隔偏置；$Q_i\in[0,1]$ 表示该目标获得的全局参照质量。候选来自过少的窗口或集中于同一异常片段时，应限制其总贡献，避免同一来源被重复计权。

候选 key/value 先由对应 Patch 的局部表示产生：$k_j=W_k h_j^L$、$v_j=W_v h_j^L$。目标查询为 $q_i=W_q h_i^L$。多头输出拼接并投影回 $d$ 维，得到 $c_i^G\in\mathbb{R}^d$。第一版可在最终局部编码层后增加一层全局交叉注意力，控制对预训练结构的改动。

### 4.5 全局增强表示与点级异常头

使用参照质量调节全局残差：

$$
h_i^G=\operatorname{LayerNorm}\left(h_i^L+Q_i W_o c_i^G\right).
$$

局部表示 $h_i^L$ 和全局增强表示 $h_i^G$ 分别通过 TimeRCD 原有 Patch 输出投影，将每个 Patch 映射回 $P$ 个时间点表示。批次级逐点张量形状为：

$$
Z^L,Z^G\in\mathbb{R}^{B\times T'\times 1\times d_{\mathrm{proj}}},
$$

其中 $T'$ 为补齐后的长度。共用或初始化自同一组 TimeRCD 权重的异常头产生：

$$
p^L,p^G\in[0,1]^{B\times T'\times1\times2}.
$$

两路分数须使用同一异常类别定义，并在合成验证集上校准到可比较的尺度。有效点掩码用于排除 padding 的损失和输出。

### 4.6 Patch 级门控与分歧不确定性

对目标 Patch $i$ 计算局部门控提议 $g_i\in[0,1]$，它根据 $h_i^L$、$h_i^G$、两路异常证据和 $Q_i$ 动态调整。为了使不可靠的全局参照不能压过局部证据，定义：

$$
\alpha_i^G=Q_i g_i,\qquad
\alpha_i^L=1-\alpha_i^G.
$$

门控对同一 Patch 覆盖的 $P$ 个时间点共享，并广播到点级概率：

$$
p_{i,r}^F=\alpha_i^L p_{i,r}^L+\alpha_i^G p_{i,r}^G,
\qquad r=1,\ldots,P.
$$

最终异常分数取 $p_{i,r}^F$ 的异常类别概率。上下文不确定性单独计算，不默认加到异常分数中。首版可按以下三项构成：

$$
U_i=\lambda_d D_i+\lambda_h H_i^F+\lambda_q(1-Q_i),
$$

其中 $D_i$ 是 Patch 内局部与全局点级异常概率的平均差异，$H_i^F$ 是 Patch 覆盖时间点的融合概率熵均值并归一化到 $[0,1]$，$1-Q_i$ 表示可信参照不足，且 $\lambda_d+\lambda_h+\lambda_q=1$。系数在合成验证集上校准。$U_i$ 是上下文不确定性指标，不应直接称为已校准的概率；需用错误识别能力和风险-覆盖率实验评估。高 $U_i$ 表示上下文判断不稳，不自动表示异常。

该输出规则对应四种证据组合：

| 局部证据 | 全局证据 | 解释 |
| --- | --- | --- |
| 正常 | 正常 | 若全局参照质量高且两路置信充分，提供较强正常证据 |
| 正常 | 异常 | 两路冲突，提高不确定性；可能是异常、状态变化或远程参照不适配 |
| 异常 | 异常 | 两路一致支持异常；仍需验证两路是否共享同一偏差 |
| 异常 | 正常 | 两路冲突，提高不确定性；不能仅凭一条分支覆盖另一条分支 |

## 5. 训练流程

### 5.1 初始化与训练样本

使用 TimeRCD 预训练 checkpoint 初始化 Patch embedding、局部 Transformer、逐点输出投影和异常头。新增地址投影、参照可信度估计器、全局交叉注意力和融合模块需要训练或微调。

从 TimeRCD 合成数据或兼容的合成训练流程构造完整序列，再切成局部窗口。训练查询 Patch 时，只能从其他窗口抽取全局候选。训练情形应覆盖：

- 同一正常状态的正常参照；
- 不同正常状态或状态转换附近的参照；
- 含异常的参照窗口；
- 重复出现的异常模式；
- 可信候选不足或完全无候选。

正常原型使用已知正常的训练 Patch，从多个正常状态分别形成原型，减少将一种正常模式误当成唯一参照标准的风险。

### 5.2 训练目标

总目标由点级异常检测、两路分支辅助监督、参照可信度监督和可用的重建辅助目标组成：

$$
\mathcal{L}=\mathcal{L}_{F}
+\lambda_L\mathcal{L}_{L}
+\lambda_G\mathcal{L}_{G}
+\lambda_r\mathcal{L}_{r}
+\lambda_{rec}\mathcal{L}_{rec}.
$$

- $\mathcal{L}_{F}$：融合后逐点异常分类损失。
- $\mathcal{L}_{L}$、$\mathcal{L}_{G}$：分别监督局部与全局分支，避免门控早期完全关闭某一路。
- $\mathcal{L}_{r}$：参照可信度损失；目标可以是候选 Patch 中正常点的比例，或由合成生成过程定义的参照适用标签。
- $\mathcal{L}_{rec}$：仅在原预训练重建头和对应训练目标可用时保留，用作辅助表示学习，不作为推理异常分数。

不强制两路门控权重平均使用；若某些样本主要由局部证据解决，门控应允许局部分支占主导。可用分支 dropout 或分阶段训练避免全局分支早期失效，但须通过消融验证。

不确定性系数和校准器只能在合成验证集或合法开发集上确定，不能查看最终测试标签后再调整。

## 6. 推理流程

1. 对输入序列做与 TimeRCD 一致的标准化和有效点掩码处理。
2. 按配置的 $W_s$ 分窗、按 $P$ 切 Patch，运行局部 TimeRCD 编码器。
3. 计算每个 Patch 的局部点级异常分数，并作为窗口外候选的可信度输入之一。
4. 对每个目标 Patch，在其他窗口中检索前 $K_0$ 个相关候选。
5. 根据候选局部分数和正常原型距离估计可信度，形成最多 $K$ 个软加权参照。
6. 使用全局交叉注意力读取可信窗口外上下文，得到全局增强表示与全局异常分数。
7. 逐 Patch 计算融合门控，输出逐点异常分数和上下文不确定性。
8. 移除 padding，并映射回原序列坐标。

推理仍是双向离线检测。若要在线使用，须另行限制只能访问历史窗口，并重新训练、设计候选缓存与更新策略。

## 7. 候选模块实验与证伪条件（通过前置诊断后）

本节只适用于前置诊断通过后的模块评估。诊断阶段与停止条件见[实验路线](TimeRCD_诊断优先实验路线.md)。

### 7.1 必要对照

| 对照 | 检验问题 |
| --- | --- |
| 原始 TimeRCD 大窗口 | 复现基线 |
| 小窗口局部分支 | 判断缩短局部范围本身的影响 |
| 小窗口 + 相似度检索，不做可信度 | 判断增加窗口外候选是否有效，以及污染风险 |
| 小窗口 + 可信度加权参照 | 判断候选可信度估计是否提供增益 |
| 固定权重融合 vs. Patch 级门控 | 判断动态门控是否优于静态融合 |
| 门控融合但不输出 $U_i$ vs. 同时输出 $U_i$ | 判断不确定性是否有效反映证据冲突和参照不足 |
| 人工提供干净正常参照的 oracle | 测量参照质量的理论上限，不作为可部署基线 |

CrossAD 不列为当前必要基线。若前置诊断通过后决定借鉴其查询路由或原型记忆，再将具体适配机制作为可选消融或设计对照；需在相同数据和评估设定下比较，不能直接拿 CrossAD 论文中的结果与 TimeRCD 零样本结果作基线比较。

### 7.2 关键测试情形

除常规基准外，需特别分析：重复异常、异常密度变化、状态切换、窗口边界异常、短时点异常和依赖长程模式的上下文异常。测试标签只用于评估，不参与推理检索或参照可信度判定。

异常检测报告 VUS-PR、Affiliation-F1 等与 TimeRCD 对齐的指标；不确定性报告其区分错误预测与正确预测的能力、风险-覆盖率曲线，以及在不同异常密度和参照质量下的校准情况。

### 7.3 证伪条件

- 相似度检索不优于局部基线，且在受控远程依赖任务上也无帮助：远程访问假设不成立或检索表征无效。
- 可信度加权不优于未筛选候选，或高异常密度时更易受重复异常污染：当前可信度定义无效。
- Patch 级门控不优于固定融合，且没有稳定的样本级互补性：动态融合没有必要。
- $U_i$ 无法识别分支冲突、低参照质量或预测错误：当前不确定性构造不能作为可信信号。
- oracle 正常参照明显有效，而实际可信度估计持续失败：研究瓶颈在参照可靠性估计，不能通过更复杂的门控掩盖。

## 8. 复杂度与后续实现

精确地对每个 Patch 与所有其他窗口 Patch 比较，时间复杂度约为 $O(BN^2d_a)$，适合作为机制验证基线，不应预先宣称加速。报告检索时间、候选召回、可靠候选比例、检测时延和显存。只有在全局参照机制有效后，再评估分层索引、近似近邻或按窗口摘要检索。

不按本方案的模块章节顺序直接开工。先完成[诊断优先实验路线](TimeRCD_诊断优先实验路线.md)第 1–3 阶段；只有异常参照污染得到支持，才实现参照可信度与窗口外读取；只有窗口敏感性可归因且能在新数据集上预测，才实现多尺度。任何模块需先单独验证，再决定是否合并。检索、融合与不确定性的复杂度评估均属于后续阶段。

## 9. 多变量扩展原则

首版令 $C=1$，但张量和模块边界按多变量扩展设计。后续表示保留为 $H\in\mathbb{R}^{B\times J\times N_w\times C\times d}$：局部和窗口外时间参照首先沿时间轴建立；通道间交互作为单独的分支或注意力模块加入。检索键应保留通道身份，先验证同通道时间参照，再单独研究跨通道参照。不得把通道编号直接当成时间位置拼接进同一条检索轴。

## 10. 主要风险与边界

- **重复异常互为参照：** 相似度和邻居数量不能保证正常；需要正常原型、候选自身局部证据和异常密度压力测试。
- **正常状态多模态：** 单一正常原型会错拒合法状态；原型库需覆盖多个正常模式，可信度采用软权重。
- **预训练域偏移：** 合成正常原型可能不覆盖目标域；原型距离只能作为软证据，参照不足时提高不确定性。
- **证据相关：** 局部和全局分支共享编码器与异常头，不能把两路一致简单解释成统计独立的双重证明。
- **分数校准：** 两分支分数必须在相同标签定义和独立验证数据上校准；门控权重可解释为模型路由权重，不直接解释成真实概率。
- **训练需求：** 原 checkpoint 可初始化结构，但新增离散检索、可信度估计、跨注意力和融合行为都需训练或微调。

## 11. 参考文献

1. Lan, T., Le, H. D., Li, J., He, W., Wang, M., Liu, C., & Zhang, C. (2026). *Towards Foundation Models for Zero-Shot Time Series Anomaly Detection: Leveraging Synthetic Data and Relative Context Discrepancy*. Proceedings of the 43rd International Conference on Machine Learning, PMLR 306, 62494–62529. https://proceedings.mlr.press/v306/lan26b.html. 本目录出版版原文：[TimeRCD 论文 PDF](<Lan 等 - 2026 - Towards Foundation Models for Zero-Shot Time Series Anomaly Detection Leveraging Synthetic Data and.pdf>)。arXiv v5 的撤回状态仅指该 arXiv 版本：https://arxiv.org/abs/2509.21190v5
2. Li, B. et al. *CrossAD: Time Series Anomaly Detection with Cross-scale Associations and Cross-window Modeling*. NeurIPS 2025; arXiv:2510.12489v1. https://arxiv.org/abs/2510.12489v1
3. Yang, Y. et al. *VETime: Vision Enhanced Zero-Shot Time Series Anomaly Detection*. arXiv:2602.16681, v3. https://arxiv.org/abs/2602.16681v3
4. Roth, K. et al. *Towards Total Recall in Industrial Anomaly Detection*. CVPR 2022. https://arxiv.org/abs/2106.08265
5. Gong, D. et al. *Memorizing Normality to Detect Anomaly: Memory-Augmented Deep Autoencoder for Unsupervised Anomaly Detection*. ICCV 2019. https://doi.org/10.1109/ICCV.2019.00179
6. Song, J. et al. *MEMTO: Memory-guided Transformer for Multivariate Time Series Anomaly Detection*. NeurIPS 2023. https://doi.org/10.52202/075280-2525
7. Xu, J. et al. *Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy*. ICLR 2022. https://arxiv.org/abs/2110.02642
8. Moya-Mota, C. et al. *LEAP: a Latent Embedding Anomaly Patch-Transformer for time-series anomaly detection*. arXiv:2610.35190, v1 preprint. https://arxiv.org/abs/2610.35190
9. Zhang, Z. et al. *GSLAD: Prototype-Regularized Graph Structure Learning for Multivariate Time Series Anomaly Detection*. arXiv:2609.15483, v1 preprint. https://arxiv.org/abs/2609.15483
