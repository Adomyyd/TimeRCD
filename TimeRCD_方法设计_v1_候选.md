# TimeRCD 共享异常属性表示：方法设计 v1.1（候选）

版本：`design-v1.1`，日期 2026-11-07。状态：**候选设计，未实现，无实验支持**。本次修订依据：明确区分检测推理与正常反事实/异常重构推理路径。

本文把 [TimeRCD_问题定义与研究边界.md](TimeRCD_问题定义与研究边界.md) 第 2、3、7 节展开成可审核的公式、模块、损失、数据与训练协议。**研究定义仍以该文件为准**；本文是"第 7 节第一项交付"的落地版本，与 `EXPERIMENT_LOG.md` 一起构成"定义 + 设计 + 执行状态"三件套，本文不承担定义职责，也不记录执行结果。

证据标签：**【核实】**（本轮在代码或原文中可直接核验，含位置）/**【推断】**（有推导链）/**【共识】**（领域常识，具体出处待核）/**【需核实】**（文献或外部事实未核实）。

---

## 0. 本轮代码核实结论（取代问题定义 §5 的"本轮未复核"）

以下均在 2080Ti 的 `/diskA/YYD/TimeRCD` 实读得到，行号对应当前工作区版本。本轮**未修改任何官方代码**。

### 0.1 H 的张量层级、时间粒度与跨通道结构

- `H = local_embeddings ∈ R^{B×T×C×d_proj}`，`d_proj=256`（`time_rcd_config.py:21`），训练时 `patch_size=16`（`training.py:785`），`d_model=512`、8 层、8 头（`time_rcd_config.py:20,23,25`）。**【核实】**
- token = (通道, patch) 对，序列长度 `L = C·⌈T/W⌉`；`feature_id = arange(C).repeat_interleave(P)`（`ts_encoder_bi_bias.py:334`），注意力全双向，且仅当 `num_features>1` 时叠加 `BinaryAttentionBias`（同变/异变二值偏置，逐头一组，`ts_encoder_bi_bias.py:38-55`）。⇒ 跨通道信息通过"同一序列内全注意力"传递，**没有独立的跨通道模块**。**【核实】**
- RoPE 频率按扁平化 token 下标生成（`rope_embedder(total_length)`，`ts_encoder_bi_bias.py:349`），故不同通道的同一 patch 位置获得不同旋转角；跨通道"同相位"对齐不被显式提供。**【核实】**
- 输出投影 `d_model → W·d_proj` 后 reshape 到 `(T,C,d_proj)`（`ts_encoder_bi_bias.py:371-374`）。⇒ **同一 patch 内的 W 个时间步共享同一 token 表示，只经固定线性切片变化**。两个直接后果：**(a)** 上下文比较的实际时间粒度是 `W=16`；**(b)** 亚 patch 脉冲（1–2 点尖峰）在 token 表示里被 patch 聚合稀释，E 的点级定位精度受 W 限制。这是**设计约束**，不是实现 bug。**【核实】**

### 0.2 原训练目标与两处已确认缺陷

- 训练损失 `L = L_mask_recon + L_anomaly_CE`，系数均为 1（`training.py:352`；原文 Appendix C.4 同为 1）。`L_mask_recon` 只对被 mask 位置算 MSE（`training.py:133-157`）。**【核实】**
- 原文明确把重构定位为**训练期辅助脚手架**：推理时"Reconstruction Head is discarded completely, and no masking is applied"，分数只来自 anomaly head（Appendix C.3–C.4；推理代码 `_inference.py:107-116` 亦只调 `anomaly_head`）。**【核实】** ⇒ 本方案的 $D_N/D_A$ 是**新增部件**，不能靠"复用推理期已有的重构头"实现；这是 §3、§6 必须重新设计模块的原因。
- 遮蔽输入用 `N(0,0.1²)` 噪声替换被遮位置（`training.py:218`），且传入编码器的是 `attention_mask & ~mask`（`training.py:407`）。⇒ **被遮蔽位置的 key 被全局移除**，其 H 仅由未遮蔽上下文决定。这是本设计 §3 可以复用的现成算子。**【核实】**
- `normal_time_series` 被读入、参与标准化与 padding（`training.py:74,244-256`），但**未进入任何损失**。**【核实】**
- **缺陷 A（标签）**：`labels` 以 -1 填充（`training.py:262`），随后 `(labels > 0.5).long()`（`training.py:172`）把 -1 变成 0，且 `valid_mask = labels != -1`（`training.py:175`）恒为真 ⇒ padding 被当作"正常"计入交叉熵。**【核实】**
- **缺陷 B（通道池化）**：异常头逐 `(t,c)` 出 2 logits 后 `mean(dim=-2)` 压掉通道（`training.py:165`，推理同 `_inference.py:111`）⇒ 单通道异常被 `1/C` 稀释，与"任一时段异常即为异常"的标签语义不一致。**【核实】**

### 0.3 预处理与数据划分缺陷

- 训练侧用**整个 batch 拼接后的跨样本均值/方差**标准化（`training.py:240-247`）；推理侧用**单条序列**自身统计量（`_inference.py:21-23`）。⇒ 训练/推理统计口径不一致，且训练侧统计依赖 batch 组成。**【核实】**
- 划分按**样本下标随机打乱**（`time_rcd/_core/dataset.py:22-32`），不按基底/背景分组 ⇒ 同一基底的正常与干预副本可跨 train/test。**【核实】**

### 0.4 数据生成器的监督结构

- 干预是**加性**的：`x_obs = x_N + Δ`，Δ 定义在 `[t_s,t_e) × S` 上；原文 Table 7 给出各 family 的 Δ 显式表达式（尖峰为三角脉冲、阶跃为 `A·σ(κ(t−t_0))`、shake 为正弦等）。**【核实】**
- 样本 dict 为 `{time_series, normal_time_series, labels, attribute}`（`dataset_gen/src/generate_dataset.py:65-96`）；`attribute` 内含逐通道 local 事件记录（`type, position_start, position_end, amplitude, detail`）以及 seasonal/trend/frequency/noise/overall_amplitude/statistics。**【核实】**
- 多变量用滞后 DAG 生成，带 exogenous/endogenous 标记与边表，标签为逐时刻跨通道 OR 合并（`labels = labels | labels_i`，`ts_multi_generator.py:695`）。⇒ **时间标签语义 = OR over channels**。**【核实】**
- 幅度乘子在 generator 内以"相对基底尺度"生成（`base_scale × multiplier`，`ts_generator.py:1187-1213`）。⇒ "偏离程度"必须相对上下文定义，与目标一致。**【核实】**
- 仓库内**没有故障类型标签数据**：`datasets/` 仅 TSB-AD-U 870 条、TSB-AD-M 200 条，均只有时间标签。**【核实】**

---

## 1. 形式化问题

### 1.1 记号

窗口 $x\in\mathbb{R}^{T\times C}$，有效掩码 $m\in\{0,1\}^T$，通道标签 $y\in\{-1,0,1\}^{T\times C}$（$-1$ 表示 padding/未知），时间标签 $y_t=\max_c y_{t,c}$。

$$\begin{aligned}
H &= F_\theta(x,m)\in\mathbb{R}^{T\times C\times D}, & Z &= E(H)=\{(\pi_k,a_k)\}_{k=1}^{K},\\
s &= G(Z), & \hat x_N &= \textstyle\frac{1}{|R(t)|}\sum_{r} D_N(H^{(r)}),\\
\hat x_A &= D_A(\hat x_N, Z).
\end{aligned}$$

合成配对样本：$(x_N,\,x_{\text{obs}},\,\Delta,\,\mathcal{A})$，其中 $\Delta=x_{\text{obs}}-x_N$，$\mathcal{A}=\{(f_i,[s_i,e_i),S_i,A_i,\theta_i)\}_i$ 为事件记录（family、区间、通道集、幅度乘子、其余参数）。生成过程按 0.4 写作

$$x_{\text{obs}}=x_N+\sum_i \Delta_{f_i}(\cdot;A_i,\theta_i)\odot\mathbb{1}_{[s_i,e_i)\times S_i}. \tag{1}$$

注意 $A_i$ 是**相对基底尺度**的乘子而非绝对幅度，因此"偏离程度"的监督必须相对化（见 §7.7）。

### 1.2 三类样本的监督可用矩阵

| 样本 | $x$ | $x_N$ 真值 | $\Delta$ / 事件记录 | $y$ | 允许的损失 |
|---|---|---|---|---|---|
| 真实正常段 | 可观测 | 取 $x_N:=x_{\text{obs}}$（此段即正常） | 无 | 0 | $L_{\det}$, $L_N$ |
| 配对合成异常 | 可观测 | **有**（同基底反事实） | **有**（完整 $\mathcal{A}$） | 逐点 + 逐通道 | 全部 |
| 真实异常 | 可观测 | **无，禁止伪造** | 无 | 通常仅逐时刻 | $L_{\det}$（+ 标注正常段的 $L_N$） |

推论（必须写进协议）：$L_N$ 与 $L_A$ 的监督强度在真实数据上不可得，因此"重构类证据"只能在合成域建立、在真实域**只做零样本迁移与诊断**，不能拿真实域的重构误差当有效性证据。

---

## 2. 总览与三条防旁路规则

```
x ──► F ──► H_full ──► E ──► Z ──► G ──► s_t
x ──► F(遮蔽) ──► H^(r) ──► D_N ──► x̂_N
(x̂_N, Z) ──► D_A ──► x̂_A
```

三条规则是**结构性的**（写进接口，不靠损失惩罚）：

- **R1**：$G$ 的输入只有 $Z$，不含 $H$、不含 $x$。
- **R2**：$D_A$ 的输入只有 $(\hat x_N, Z)$，**不含 $x_{\text{obs}}$** ⇒ 观测中的异常信息只能经 $Z$ 进入异常重构。
- **R3**：$D_N$ 的输入是"该位置所在 patch 被完全遮蔽"的 $H$，不含该 patch 的观测。

R2 给出一个**可证伪的因果命题**（而非解释性声明）：若零化或置换 $Z$ 后 $L_A$ 基本不变，则要么实现违反了 R2，要么 $Z$ 不是异常通道。这正是问题定义 §4.2 与 §5"最强反驳"要求的检验形式。

### 2.1 训练与推理路径

- **训练**：按各阶段目标计算所需模块。启用 $L_N$ 或 $L_A$ 时运行 $D_N$，并按 §3.1 的遮蔽策略生成正常估计；检测损失本身使用完整输入的 $Z$。
- **检测推理（默认）**：只运行 $H_{\text{full}}\to E\to Z\to G\to s_t$。不运行 $D_N$、$D_A$，因此不需要多次遮蔽前向。
- **可选重构/诊断推理**：仅当调用方还要求输出 $\hat x_N$ 或 $\hat x_A$ 时才运行 $D_N$（以及 $D_A$）。可从一次遮蔽开始；$4\sim8$ 次只作为待验证的方差降低选项，不是检测方法运行的必要条件。

---

## 3. $D_N$：正常反事实的估计

### 3.1 定义（遮蔽 bagging 的留一期望）

令 patch 数 $P=\lceil T/W\rceil$。抽 $R$ 个独立遮蔽集合 $\mathcal{M}_r\subseteq[P]$，$|\mathcal{M}_r|=\rho P$，$\rho$ 默认 0.5，**遮蔽与任何标签无关**（否则泄漏）。掩码 $M^{(r)}$ 是不带新参数的 attention mask，$H^{(r)}=F_\theta(x\odot(\mathbb{1}-M^{(r)}),\,m\odot(\mathbb{1}-M^{(r)}))$。则

$$\hat x_N(t,c)=\frac{1}{|R(t)|}\sum_{r\in R(t)} D_N\big(H^{(r)}(t,c)\big),\qquad R(t)=\{r:\ \text{patch}(t)\in\mathcal{M}_r\}, \tag{2}$$

$$\hat\sigma_N^2(t,c)=\widehat{\mathrm{Var}}_{r\in R(t)}\big[D_N(H^{(r)}(t,c))\big].$$

训练时取 $R_{\text{tr}}=2$，用于估计 $L_N$（具体训练采样策略仍待实现验证）。推理时的 $R_{\text{inf}}$ **不是检测所需的前向次数**：默认检测路径只计算完整输入上的 $H_{\text{full}}\to E\to Z\to G\to s_t$，跳过 $D_N$ 与 $D_A$；仅当调用方还要求输出正常反事实 $\hat x_N$ 或异常重构 $\hat x_A$ 时，才启用 $D_N$。此时可用一次遮蔽估计（$R_{\text{inf}}=1$），或把 $4\sim8$ 次作为降低估计方差的可选设置；额外前向开销仅属于该重构/诊断路径，需单独验证（见 §12 T2）。

$D_N$ 结构：点级 MLP（复用原 `reconstruction_head` 的形状：$D\to4D\to4D\to1$，`training.py:100-110`），不做跨时间卷积（正常预期所需的上下文已在 $H$ 中）。

### 3.2 为什么这样能防泄漏

- 由 0.2，被遮蔽位置的 key 被全局移除，故 $H^{(r)}(t)$ 是"该 patch 完全不可见时"由上下文推断的表示；$D_N$ 在结构上无法复制 $x(t)$ 的局部值。**【核实基础 + 推断】**
- 留出单位是 **patch**（$W=16$）而非单点：单点留出会被同 patch 邻居"插值"泄漏；patch 留出以"定位精度 ≥ W"换取"不可泄漏"。这是有意的取舍，不是疏漏。
- **代价（必须测量）**：若异常是持续型（`sudden increase`、level shift 跨多个 patch），其他 patch 也含该异常信息，$D_N$ 会把"异常水平"当正常预期 ⇒ $L_N$ 上产生系统偏置。这不是设计缺陷，而是**问题本身的不可辨识部分**；协议要求按 family 分组（点型 / 持续型）分别报告。

### 3.3 泄漏度量与对照

$$Lk=\frac{\big\|(\hat x_N-x_N)\odot\mathbb 1_{\mathcal A}\big\|_2}{\|\Delta\odot\mathbb 1_{\mathcal A}\|_2},\qquad
\rho_{\text{leak}}=\operatorname{corr}\big(\hat x_N-x_N,\ \Delta\big)_{\mathcal A}.$$

- **N1（对照臂）**：$\hat x_N=D_N(H_{\text{full}})$，$R=1$，不遮蔽。N1 是消融对照，不是默认臂。
- 判据：N2 的 $Lk$ 应显著低于 N1；若两者都高，问题在 $D_N$ 容量或输入，不在 bagging。

### 3.4 方差的用途（不进主分数）

$\hat\sigma_N$ 只用于 (i) $L_N/L_A$ 的难度加权（$1/\hat\sigma_N^2$，作为消融）；(ii) 诊断（正常区段归一化残差分布）。**不进入 $G$**，以免把主分数退化成残差分数（问题定义 §2 的约束）。

---

## 4. $E$：异常方向分析器

### 4.1 单点证据三元组

对 $(t,c)$，令 $h=H_{\text{full}}(t,c)\in\mathbb R^{D}$。

- 共享读出 $\varphi_\psi:\mathbb R^{D}\to\mathbb R^{d_z}$（1×1 MLP，逐点共享）；
- 专家原型 $U=[u_1,\dots,u_K]\in\mathbb R^{d_z\times K}$，列单位化 $\|u_k\|_2=1$；
- 路由（软分配）与 top-m 稀疏化：
$$\hat\pi_k(t,c)=\operatorname{softmax}_k\!\Big(\frac{\langle\varphi_\psi(h),u_k\rangle}{\tau_\pi}\Big),\qquad
\pi(t,c)=\operatorname{topm}\big(\hat\pi(t,c),\,m\big),\quad m\in\{1,2\}; \tag{3}$$
- 带符号幅度：
$$a_k(t,c)=\langle\psi_a(h),v_k\rangle,\qquad v_k\in\mathbb R^{d_a},\ d_a=1\ \text{(默认)}. \tag{4}$$

输出 $Z(t,c)=\{(\pi_k,a_k)\}_{k=1}^{K}$。语义分工：$\pi_k$ 承载**方向归属**，$a_k$ 承载**带符号的偏离程度**，二者相乘 $\pi_k a_k$ 是该点在专家 $k$ 上的**符号化偏移量**（可直接读出贡献，见 §5）。

（可选）形状码 $e_k(t,c)=\varphi_\psi(h)\odot u_k\in\mathbb R^{d_z}$，仅供下游映射用，不进 $G$。

### 4.2 该形式的理由与代价

- 与问题定义 §2 的"方向 + 偏离程度"逐项对应；$\pi/a$ 的分解使 $G$ 的加性聚合与定位诊断成立。
- 代价：表达力弱于"自由 MLP 出 K 维码"——跨专家的乘性交互无法直接表示。用 **G-MLP 对照臂**（§5 末）检验这一限制是否起作用。
- $K_{\max}$、$d_z$、$m$ 是**待消融旋钮**，不预设"专家 = 某属性"。

### 4.3 可辨识性的边界（必须写在方法正文，不能只放附录）

**命题 1（仅靠检测损失不可辨识方向）**：若 $s=G(Z)$ 且损失只依赖 $(s,y)$，则对任意可逆 $T$，只要 $G$ 所在函数类对与 $T^{-1}$ 复合封闭（MLP、§5 的加性证据模型都满足），$Z'=T(Z)$ 与原 $Z$ 就能达到相同的最优 $s$ 与相同的风险，因而**$Z$ 的逐分量解读不可区分**。因此"第 $k$ 个专家编码 family $f_k$"这一结论**不可能**由检测损失推出。

*证明要点*：损失只经由 $s$ 依赖 $Z$；取 $G'=G\circ T^{-1}$ 即可完全补偿 $T$，风险与最优 $s$ 不变。∎

**推论**：方向语义只能来自额外的结构性监督（配对干预、跨背景不变性、组合关系），且即使如此也只到"在给定模型类与数据分布下可复现、可复用、可定位"，**不到"真实生成因子"**。这是本方案对自身主张等级的限定。

**文献定位（已核实，含移植前提核查）**

- 非线性 ICA 在无辅助变量时**一般不可辨识**：Hyvärinen & Morioka, *Unsupervised Feature Extraction by Time-Contrastive Learning and Nonlinear ICA*, NIPS 2016 ([arXiv:1605.06336](https://arxiv.org/abs/1605.06336)) 明确指出此前非线性 ICA "not identifiable"，其时间对比方案给出首个严格可辨识结果。**【核实】**
- 引入**辅助观测变量**后可在给定生成模型类内辨识到简单变换：Khemakhem, Kingma, Monti, Hyvärinen, *Variational Autoencoders and Nonlinear ICA: A Unifying Framework* (iVAE), AISTATS 2020, PMLR 108:2207–2217 ([arXiv:1907.04809](https://arxiv.org/abs/1907.04809))。其结论依赖"因子化指数族先验 + 以辅助变量为条件 + 可逆混合"等假设。**【核实】**

因此本方案把**配对干预记录**（同一基底、不同 $\Delta$；不同基底、同一 $\Delta$）当作辅助变量，这是让"方向"具备可辨识性的标准路径。但必须逐条核前提：

| iVAE 侧假设 | 本方案是否满足 | 后果 |
|---|---|---|
| 观测由潜变量经可逆混合生成 | **不满足**：$Z$ 只是 $H$ 的一种读出，不是生成模型 | 不能主张潜变量唯一性 |
| 辅助变量只影响先验分布 | **部分满足**：我们的"辅助变量"是配对关系，不是逐样本标签 | 只能主张"关系一致"，不能主张"因子被找到" |
| 先验为指数族因子化 | **不满足** | 无唯一性保证 |

⇒ 只借用"**必须引入辅助结构**"这一必要性结论，**不借用唯一性保证**。这是本方案最容易被人用来过度声张的地方，故在正文显式限定。若审稿人问"你凭什么说这是属性"，唯一站得住的回答是 §10 的 M1/M5/M7 三个可证伪检验，不是可辨识性定理。

**自我反驳**：即便有配对，若某些 family 只出现在特定基底上（生成器采样相关），$L_{\text{attr}}$ 会学到"基底捷径"而非方向。必须以跨背景配对（R1）与 held-out family 检验，否则不成立。

### 4.4 退化模式清单（每条对应一项监督）

| 编号 | 退化 | 对应约束 |
|---|---|---|
| D1 | 全部路由到单一专家（坍塌） | $L_{\text{bal}}$（与均匀分布的 KL）+ top-m |
| D2 | 路由均匀、无专化 | $L_{\text{attr}}$ 的同 family 正样本 |
| D3 | $a_k$ 只编码"总偏离强度"，各专家同向 | $L_{\text{attr}}$ 的**幅度匹配硬负样本**（§7.6） |
| D4 | 跨种子换位/旋转 | 稳定性度量（§10 M4），不以色号作结论 |
| D5 | 只编码生成器模板外观，而非相对上下文变化 | 同 family 跨基底正样本 + 相对幅度（乘子）而非绝对幅度 |

MoE 坍塌与路由负载不均是已知失效模式：Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*, ICLR 2017 ([arXiv:1701.06538](https://arxiv.org/abs/1701.06538)) 引入负载均衡损失；Chi et al., *On the Representation Collapse of Sparse Mixture of Experts*, NeurIPS 2022 ([arXiv:2204.09179](https://arxiv.org/html/2204.09179v1)) 说明路由使 token 向少数专家中心聚类即导致表示坍塌。**【核实】** 但"专家专化"绝不等于"专家 = 语义属性"，本方案不主张后者。

### 4.5 加性解码器与组合外推：为什么 §6.1 敢用加性

Lachapelle, Mahajan, Mitliagkas, Lacoste-Julien, *Additive Decoders for Latent Variables Identification and Cartesian-Product Extrapolation*, NeurIPS 2023：当解码器对潜变量**按块加性**时，潜变量可辨识到**块内可逆变换 + 块置换**，并且可外推到训练中未见的块组合（Cartesian-product extrapolation）。**【核实，来源见 §11.1】**

对本方案的三点用途：

1. 给 §6.1 的加性 $D_A$ 一个**独立于"生成器恰好是加性"的理由**：加性结构本身就是可辨识性与外推的杠杆，而不是省事的选择。
2. 把"组合可分解"从愿望变成**有理论预期的检验**：M5（组合描述子一致性、未见组合外推）正是该定理预测应成功的场景；若它失败，说明 $Z$ 的块结构与真实潜变量块不对应。
3. **保证不迁移**：该结论针对"已知加性解码器 + 真实潜变量 + 满足其生成假设"；本方案的 $Z$ 是学出来的、$K$ 是过参数化的、真实时序不满足其生成假设。故只能借设计动机与检验方向，**不能主张定理成立**。

### 4.6 $E$ 怎么训练：梯度来源、干扰与冻结策略

$E$ 的参数集合为 $\theta_E=\{\psi,\psi_a,U,V\}$（读出 MLP、幅度 MLP、路由原型、幅度方向），数量级 $O(d_z D + K d_z)$，相对编码器可忽略（$d_z{=}16,K{=}16,D{=}256$ 时约 1 万参数）。

| 损失 | 梯度经过的路径 | 主要塑形的参数 | 意图 |
|---|---|---|---|
| $L_{\det}$ | $G\to E$ | $\psi_a,V$（以及经 $\pi$ 的 $U$） | 让证据对异常**敏感** |
| $L_A$ | $D_A\to E$ | $\psi_a,V,\psi$ | 让 $Z$ 承载**可重构**的偏离信息（R2 下这是唯一通路） |
| $L_\pi,L_{\text{bal}}$ | 直接作用于 $\pi$ | $U$ | 支撑稀疏与不坍塌 |
| $L_{\text{attr}}$ | 描述子 (9) | $\psi,U$（$q_i$ 已幅度归一，故对 $V$ 的梯度很弱） | 让方向码**跨背景一致、对 family 有选择性** |
| $L_{\text{mag}}$ | $\sum\pi|a|$ | $V$ | 让程度**单调**于相对幅度 |

**已知的梯度干扰（必须显式处理，不能靠调参碰运气）**：$L_{\det}$ 只要求 $s$ 可分，会把 $Z$ 推向"信息最省的单向强度轴"（即 D3 退化）；$L_{\text{attr}}$ 要求方向码在幅度变化下不变。两者方向相反。处理办法：

1. **解耦幅度与路由**：$\pi$ 只由 $\psi$（读出）决定，$a$ 只由 $\psi_a$ 决定；$\psi_a$ 不参与 $L_{\text{attr}}$（描述子用归一化 $q_i$，实现上可对 $a$ 做 `detach`）。这样"敏感"与"不变"分别落在两组参数上，冲突被结构隔离。
2. **梯度比例诊断**：每个 epoch 记录 $\|\nabla_{\theta_E}L_{\det}\|$ 与 $\|\nabla_{\theta_E}L_{\text{attr}}\|$ 的比值，作为 $\lambda_{\text{attr}}$ 的调节依据（记入 manifest，不靠肉眼）。
3. **分阶段**：S1 只开 $L_{\det},L_N,L_\pi,L_{\text{bal}}$（此时 $\theta_E$ 学到"敏感的证据轴"）；S3 才加 $L_{\text{attr}},L_{\text{mag}}$。若 S3 使 S1 的检测指标显著下降，则判为干扰不可接受，须回到 T1 缩小预算或改用 detach 方案。
4. **优化器分组**：新模块（$E,G,D_N,D_A$）用比编码器更高的学习率（如编码器 $1\times10^{-5}$、新模块 $5\times10^{-4}$），因为编码器只需适配、新模块需从零学。

**初始化**：$U$ 随机正交初始化（仅作起点，不主张正交语义——正交约束本身不产生语义，见 §4.3 的命题 1）；$V$ 零初始化使初始 $a\equiv0$，即模型从"证据为零"出发，避免一开始就用随机幅度污染 $L_A$。

---

## 5. $G$：证据到主分数

$$s(t,c)=b_0+\sum_{k=1}^{K}\pi_k(t,c)\Big[\lambda_k^{+}\,(a_k)_{+}+\lambda_k^{-}(-a_k)_{+}\Big],\qquad \lambda_k^{\pm}=\operatorname{softplus}(\cdot)>0. \tag{5}$$

即**加性证据模型**：分段线性、对每个方向的贡献单调、逐方向可读出 $s$ 的分解。

**通道池化（默认：noisy-OR）**

$$s_t=c_0+\sum_{c=1}^{C}\operatorname{softplus}\big(s(t,c)\big). \tag{6}$$

*推导*：若给定证据时各通道异常事件条件独立，则
$-\log P(\text{时刻 }t\text{ 无任何通道异常})=\sum_c\log\big(1+e^{s(t,c)}\big)=\sum_c \operatorname{softplus}(s(t,c))$，即 (6) 是"任一通道异常"的精确负对数似然。**【推断】** 这与 0.4 核实的标签语义 $y_t=\mathrm{OR}_c\,y_{t,c}$ 一致，同时修掉 0.2 缺陷 B（`mean` 把单通道异常稀释为 $1/C$）。

**对照池化**：(a) `mean`（现状）；(b) $\log\frac1C\sum_c e^{s(t,c)}$（max 型平滑 OR）。当生成器的 DAG 使通道强耦合时，(6) 的独立性假设失效，须实测三者差异。

$s_t$ 是 logit，直接配 `BCEWithLogits`，不强行压到 $[0,1]$。

**G-MLP 对照臂**：$s=\mathrm{MLP}(Z)$，仍满足 R1。用于检验加性聚合是否造成欠拟合。

---

## 6. $D_A$：异常重构

### 6.1 主臂（局部条件残差）

证据图 $\mathbf E\in\mathbb R^{T\times C\times K}$，$\mathbf E_{t,c,k}=\pi_k(t,c)\,a_k(t,c)$。

$\hat\Delta=\operatorname{Conv1d}_\theta\big([\hat x_N;\mathbf E]\big),\quad \text{kernel}=2W+1,\ \text{逐通道（可选跨通道混合）};\qquad \hat x_A=\hat x_N+\hat\Delta. \tag{7}$

该重构路径用于训练和可选诊断；纯检测推理不需要调用 $D_N/D_A$。

理由三条：**(i)** 生成器的 $\Delta$ 在原文 Table 7 中**本身就是加性**的，主臂与数据生成过程一致（0.4 核实）；**(ii)** 卷积核 $2W+1$ 使标量残差能与邻域一致，可表达尖峰/斜坡/阶跃的形状；**(iii)** 输入不含 $x_{\text{obs}}$，满足 R2。

### 6.2 对照臂（非加性、窗口级条件解码）

以证据序列为 query，对 $\hat x_N$ 及其上下文嵌入做一层 cross-attention，直接产出整窗输出（不做残差相加）。可表达相位/周期重排等非加性变化，参数与训练成本更高。

**两臂对照的意义**：若加性臂在 held-out 的周期/相位类 family 上明显落后，则"异常可加性分解"这一隐含假设被证伪（对应问题定义 §2 对"没有确定必须相加"的警告）。

### 6.3 为什么不能复制输入

R2 保证 $D_A$ 看不到 $x_{\text{obs}}$，因此 $\|\hat x_A-x_{\text{obs}}\|$ 的下降只能来自 $Z$。配套检查：若零化/置换 $Z$ 后该量基本不变，则实现违反 R2，或 $\hat x_N$ 已把异常带进来（用 $Lk$ 检出）。

---

## 7. 完整损失

$$\mathcal L=\lambda_d L_{\det}+\lambda_N L_N+\lambda_A L_A+\lambda_\pi L_\pi+\lambda_{\text{bal}}L_{\text{bal}}+\lambda_{\text{attr}}L_{\text{attr}}+\lambda_{\text{mag}}L_{\text{mag}}. \tag{8}$$

默认 $\lambda_d=\lambda_N=\lambda_A=1$（沿用原实现与原文 C.4 的口径），其余在 **dev 背景组**上选，test 组只看一次。

### 7.1 检测 $L_{\det}$

$$L_{\det}=-\frac{1}{\sum_t m_t}\sum_{t:\,m_t=1}\Big[w_+ y_t\log\sigma(s_t)+w_-\,(1-y_t)\log\big(1-\sigma(s_t)\big)\Big],\qquad w_+=\tfrac{\#\text{neg}}{\#\text{pos}}.$$

padding 由 $m$ 排除，$y=-1$ 的位置不参与（修掉缺陷 A 的语义：**不再把 padding 当正常**）。若通道标签可用（**仅合成**），附加 $\lambda_{dc}\sum_{t,c:\,y_{t,c}\neq-1}\mathrm{BCE}\big(s(t,c),y_{t,c}\big)$。**不允许**把只有时间标签的真实数据当成逐通道真值。

### 7.2 正常反事实 $L_N$

$$L_N=\frac{\sum_{t,c}m_t\,v(t,c)\,\big(\hat x_N-x_N\big)^2}{\sum_{t,c}m_t\,v(t,c)},\qquad v\equiv1\ (\text{默认}),\quad v=1+\beta_N\mathbb 1_{\mathcal A}\ (\text{消融}).$$

### 7.3 异常重构 $L_A$

$$L_A=\frac{\sum_{t,c}m_t\,\big(\hat x_A-x_{\text{obs}}\big)^2}{\sum_{t,c}m_t}\quad(\text{仅含事件的窗口}).$$

$1/\hat\sigma_N^2$ 加权作为消融。退化检查：若 $L_A\approx L_N$ 的水平，说明 $D_A$ 学会了"什么都不改"。

### 7.4 支撑稀疏 $L_\pi$（监督式，不是无差别 $L_1$）

$$L_\pi=\frac{1}{|\Omega_0|\,C}\sum_{(t,c)\in\Omega_0}\sum_k\pi_k(t,c),\qquad \Omega_0=\{y=0\text{ 且不在任何事件支撑内}\}.$$

理由：无差别 $L_1$ 会把异常区的证据也压小，与检测目标直接冲突；只在"应当无证据"的位置施加稀疏。同步报告"全体 $\pi$ 的 $L_1$"作为对照，用以检验这条设计选择是否必要。

**与稀疏学习文献的关系**：$\pi$ 的 top-m 竞争对应"特征互斥竞争支撑"的思路（exclusive lasso，Zhou, Jin, Hoi, AISTATS 2010, PMLR 9:988–995）；把"时间×通道支撑"整体稀疏化则接近分组稀疏（group lasso，Yuan & Lin, JRSS-B 2006，卷页需二次核对）。**【核实（exclusive）/需核实（group lasso 卷页）】** 区别要写明：上述工作稀疏的是**线性权重**，本方案稀疏的是**路由概率** $(\pi)$ 与**支撑**，不是同一对象；借用的是"竞争抑制共线支撑"的直觉，不是其估计性质。

### 7.5 负载均衡 $L_{\text{bal}}$

$$L_{\text{bal}}=\mathrm{KL}\Big(\frac{1}{|\Omega|}\sum_{(t,c)\in\Omega}\pi(t,c)\ \Big\|\ \mathrm{Uniform}(K)\Big).$$

防 D1。系数必须小：过大会强迫罕见 family 也被均匀分配，反而毁掉专化。

### 7.6 集合级关系监督 $L_{\text{attr}}$（本方案的核心监督）

**事件描述子**。对事件 $i$（区间 $[s_i,e_i)$、通道集 $S_i$）：

$$p_i(k)=\frac{\sum_{(t,c)\in S_i}\pi_k(t,c)}{\sum_{k'}\sum_{(t,c)\in S_i}\pi_{k'}(t,c)},\qquad
q_i(k)=\frac{\sum_{(t,c)\in S_i}a_k(t,c)}{\big\|\sum_{(t,c)\in S_i}a_\cdot(t,c)\big\|_2+\epsilon},\qquad
e_i=\frac{[p_i;q_i]}{\|[p_i;q_i]\|_2}\in\mathbb R^{2K}. \tag{9}$$

$q_i$ 做了幅度归一，使 $e_i$ 只承载**方向**；程度由 $L_{\text{mag}}$ 单独管。

**正样本关系（三种）**
- **R1** 同 family、**不同基底**（不同背景组）；
- **R2** 同一次事件出现在单事件窗口 vs 组合窗口（支撑互不重叠）；
- **R3** 同 family、幅度乘子比落在 $[1/1.5,\,1.5]$（排除"只靠幅度相近"的伪正样本）。

**硬负样本**：同基底、**不同 family**、且**幅度分箱相同**（排除幅度捷径，专打 D3）。

$$L_{\text{attr}}=-\frac{1}{|\mathcal P|}\sum_{(i,j)\in\mathcal P}\log\frac{\exp\big(\operatorname{sim}(e_i,e_j)/\tau\big)}{\sum_{n\in N(i)}\exp\big(\operatorname{sim}(e_i,e_n)/\tau\big)}. \tag{10}$$

形式上属于 Supervised Contrastive Learning 一类（Khosla et al., *Supervised Contrastive Learning*, NeurIPS 2020，[链接](https://neurips.cc/virtual/2020/poster/17228)）。**【核实】** 但作用对象是**事件的支撑池化描述子**，且负样本按幅度匹配挑选。

**必须记录的限制**：R1/R2/R3 使用了生成器 family 标签作为**训练期辅助信号**。因此"学到开放属性"只能由 **held-out family** 与真实数据上的证据支持，**不能**由训练集上的可分性支持（问题定义 §3.3 的同一条纪律）。若用户不接受训练期使用 family 标签，见 §12 T3。

### 7.7 程度校准 $L_{\text{mag}}$（相对、同基底）

对同基底、同 family、不同幅度乘子 $A_i>A_j$ 的事件对：

$$L_{\text{mag}}=\frac{1}{|\mathcal P_{\text{mag}}|}\sum_{(i,j)}\operatorname{ReLU}\Big(\delta-\big(\hat a^{\Sigma}_i-\hat a^{\Sigma}_j\big)\Big),\qquad
\hat a^{\Sigma}=\sum_{(t,c)\in S}\sum_k\pi_k\,|a_k|. \tag{11}$$

只在**同一基底内**比较，故不主张跨基底的绝对幅度可比（0.4：幅度本身是相对量）。防 D3 的"$a$ 与幅度无关"。

### 7.8 明确留空（v1 不做）

跨通道传播（exogenous→endogenous 的时序先后）、专家正交/去相关、以语义命名专家。理由：缺少独立证据，且可能编码生成器伪影。列入 §10 的后续单独消融，不预先堆进损失。

---

## 8. 训练数据构造

### 8.1 背景分组（修复 0.3 的划分缺陷）

把生成器的**背景因子**——seasonal 类型 × trend 类型 × frequency × noise 等级 × 通道数 $C$ × DAG 拓扑/边数 × 周期区间——作为分组键，**按组划分**：

| 组 | 用途 |
|---|---|
| train 背景组 | 训练 |
| dev 背景组 | 选 $\lambda$、早停、结构旋钮 |
| test 背景组 | 只看一次 |

同一基底的 $x_N$ 与**全部**干预副本必须同组。

### 8.2 family 划分

$F_{\text{train}}$ / $F_{\text{dev}}$ / $F_{\text{unseen}}$。$F_{\text{unseen}}$ 专门用于"未见属性族外推"，**不参与任何超参选择**。候选（登记协议时冻结清单）：seasonal 侧 `pulse width modulation`、`waveform inversion`；local 侧 `continuous downward spike`、`rapid decline followed by slow rise`。

### 8.3 episode 结构（配对来源）

对每个基底 $b$ 生成一个 episode：

1. 1 个正常窗口 $x_N$（$y=0$）；
2. 单事件窗口 $\{x_{\text{obs}}^{(i)}\}$：$f\in F_{\text{train}}$、幅度乘子 $A\in$ 分箱、位置 $p$、通道集 $S$ 随机；
3. 组合窗口：2 个 family，支撑互不重叠；
4. 硬负样本对：同基底、不同 family、同位置、同幅度分箱；
5. 跨背景正样本对：同 family、同幅度分箱、$b'\neq b$ 且背景组不同。

⇒ **一个 episode 同时提供 R1/R2/R3 所需的三类关系**，天然配对且可追溯。每条落盘：基底 id、背景组、DAG、事件记录、manifest 版本、随机种子。

### 8.4 真实数据接入

TSB-AD-U/M（870/200 条，仅时间标签）：只用于 $L_{\det}$ 与"标注正常段的 $L_N$"。
**禁止**：为真实异常伪造 $x_N$；把时间标签当逐通道真值；用真实域重构误差作为有效性证据。
真实异常上的 $Z$ 只用于下游映射与推理诊断。

**数据缺口（需用户裁定）**：少样本故障类型映射需要带**故障类型标签**的真实数据，仓库内没有。候选来源须由用户指定，本设计不擅自引入（见 §12 T4）。

### 8.5 归一化

统一改为**逐窗口实例标准化**，三臂消融：

| 臂 | 说明 |
|---|---|
| P1 | per-series z（现推理口径，`_inference.py:21-23`） |
| P2 | per-window z（窗口自身 mean/std） |
| P3 | per-window robust（median/IQR） |

必须按 family 分组报告对**持续型**（`sudden increase/decrease`、level shift）的影响：per-window 标准化会抹平整窗整体抬升，per-series 会按整条序列压缩异常幅度。**禁止**跨样本 batch 统计（修 0.3）。

### 8.6 标签完整性（修 0.2 缺陷 A）

新管道产出 $y\in\{-1,0,1\}^{T\times C}$，$-1$ = padding/未知；所有损失以 $y\neq-1$ 与 $m$ 双重排除。单元测试必须覆盖：全 padding batch、全正常 batch、单点异常、通道标签缺失。**不修改官方代码**，新管道独立实现，旧权重只读。

---

## 9. 训练流程与阶段门

| 阶段 | 内容 | 新增监督 | 门禁（不通过则停） |
|---|---|---|---|
| S0 | 编码器适配：$F_{\text{train}}$ 背景合成数据，masked recon + 逐点/逐通道分类（与原目标同构） | 原目标 | 与官方权重的零样本水平差距可解释 |
| S1 | 接入 $D_N$(N2)、$E$、$G$ | $L_{\det},L_N,L_\pi,L_{\text{bal}}$ | **G1**：$s_t$ ≥ 同数据直接检测基线；否则先修数据/归一化，不进 S2 |
| S2 | 接入 $D_A$ | $L_A$ | **G2**：零化/置换 $Z$ 明显抬高 $\|\hat x_A-x_{\text{obs}}\|$；$Lk$ 低于 N1 |
| S3 | 方向语义 | $L_{\text{attr}},L_{\text{mag}}$ | **G3**：held-out 背景上 family 检索 R@1 显著高于随机；跨种子匹配率达标 |
| S4 | 少样本映射：冻结 $F,E$，只训 $Z\to$ 类型的小头 | 下游任务 | **G4**：优于参数匹配的"直接分类头"基线 |

初始化两种臂都做：官方权重 / 从头训练。**每阶段只新增一个损失族**，使每个约束的边际作用可归因（问题定义 §6 的要求）。

---

## 10. 机制验证与消融（含证伪条件）

| 编号 | 检验 | 指标 | 证伪条件 |
|---|---|---|---|
| M1 | 零化 $Z$ / 跨样本置换 $Z$ / 只置换 $\pi$ / 只置换 $a$ | $L_A$、$\|\hat\Delta\|$、属性检索 | **F1**：零化 $Z$ 后 $L_A$ 几乎不变 ⇒ $Z$ 不是异常通道 |
| M2 | $D_N$ 泄漏对照 N1 vs N2 | $Lk$、$\rho_{\text{leak}}$ | **F2**：N1 与 N2 的 $Lk$ 都高 ⇒ $D_N$ 容量/输入有问题 |
| M3 | 旁路容量诊断：$D_A$ 只吃 $\hat x_N$（无 $Z$）；把 $d_z$ 加倍 | $L_A$、检测 | 加宽 $Z$ 后检测几乎不变 ⇒ 瓶颈不是限制因素（"参数更多"的质疑成立） |
| M4 | 专家稳定性：3–5 种子，按 family 激活矩阵匈牙利匹配 | 匹配一致性、旋转后残差 | **F3**：匹配率低 ⇒ 专家是种子伪影 |
| M5 | 组合性：组合数增加时的描述子偏差与性能衰减曲线 | 逐事件描述子偏差 | 组合后描述子不可解释 ⇒ "可组合"主张不成立 |
| M6 | 检测对照（参数/计算匹配）：原 TimeRCD；同数据直接检测器（无 $Z$ 瓶颈、无重构）；冻结编码器 + 仅检测头；本方案 | AUC-PR/ROC、VUS、F1 | **F4**：本方案不优于参数匹配直接检测器 ⇒ 结构无增量 |
| M7 | 属性有效性：held-out 背景、held-out family、held-out 幅度分箱上的 family 检索与少样本分类；对比"仅检测头隐层特征" | R@1、N-way k-shot | 无显著高于对照 ⇒ 不得声称学到共享属性 |
| M8 | 归一化（P1/P2/P3）与标签修复的单独效应 | 按 family 分组 | — |

所有 M 项在 dev 背景组上定超参，test 背景组只看一次。E 的定位精度另需单列指标：$W=16$ 下能否定位 1–2 点尖峰（预期不能，见 0.1），此点直接影响"方向可定位"的主张边界。

---

## 11. 与现有工作的位置、增量与最强质疑

### 11.1 已有路线（本轮逐项核实到来源）

| 路线 | 代表性来源 | 与本方案的关系 |
|---|---|---|
| 关系式 / 上下文相对正常性（RCD） | Lan, Le, Li, He, Wang, Liu, Zhang, *Towards Foundation Models for Zero-Shot Time Series Anomaly Detection: Leveraging Synthetic Data and Relative Context Discrepancy*, ICML 2026, PMLR 306:62494–62925；[arXiv:2509.21190](https://arxiv.org/abs/2509.21190)；仓库内 PDF | 本方案的 $F$ 与"正常性由上下文定义"直接沿用；其重构头是**训练期辅助项、推理时丢弃**，主目标是 RCD 关系判别。本方案把输出从"逐点二分类概率"改为**可分解的方向证据 → 加性分数**，并显式引入正常反事实与配对干预。**【核实】** |
| 非线性 ICA 可辨识性 | Hyvärinen & Morioka, NIPS 2016, [arXiv:1605.06336](https://arxiv.org/abs/1605.06336)；Khemakhem et al., iVAE, AISTATS 2020, PMLR 108:2207–2217 | 见 §4.3：只借"必须引入辅助结构"的必要性结论。**【核实】** |
| 加性解码器可辨识与组合外推 | Lachapelle, Mahajan, Mitliagkas, Lacoste-Julien, *Additive Decoders for Latent Variables Identification and Cartesian-Product Extrapolation*, NeurIPS 2023 | 见 §4.5：给加性 $D_A$ 与 M5 组合检验提供理论依据；保证不迁移。**【核实】** |
| 稀疏 / 专家竞争 | Shazeer et al., ICLR 2017, arXiv:1701.06538；Chi et al., NeurIPS 2022, arXiv:2204.09179；exclusive lasso: Zhou/Jin/Hoi, AISTATS 2010, PMLR 9:988–995 | 见 §4.4、§7.4：借用竞争抑制与负载均衡的直觉，不借其估计性质。**【核实】** |
| 合成异常注入作为监督 | DRAEM (Zavrtanik et al., ICCV 2021)；CutPaste (Li et al., CVPR 2021)；SimpleNet (Liu et al., CVPR 2023)；时序侧 CARLA (Darban et al., arXiv:2308.09296；Pattern Recognition 2024) 明确"injects various types of anomalies as negative samples" | 借用"用受控合成异常提供监督"的一般思路。区别：本方案把**干预参数**（family/区间/通道/相对幅度）显式保留并用于关系监督，而非只当负样本。**【核实】** |
| 偏差分数（deviation-based） | Pang, Shen, van den Hengel, *Deep Anomaly Detection with Deviation Networks*, KDD **2019** | 本方案的 $s$ 也是"偏差量→分数"的参数化，区别是偏差被显式**分解到 K 个方向**并带支撑用于定位。**【核实】** |
| 异常分数多分量加权 / 定位 | Anomaly Transformer (Xu, Wu, Wang, Long, ICLR **2022** Spotlight, arXiv:2110.02642)；DCdetector (Yang et al., KDD 2023) | Anomaly Transformer 的分数确为多分量加权（`softmax(-series_loss-prior_loss)*loss`），但其 `loss` 对特征维取均值 ⇒ **无通道级定位**；本方案把"逐通道证据 + OR 池化"写进结构。检索未发现"同时具备多分量可分解分数与通道级定位"的时序工作，故此处只能声称**缺口**，不能声称"首次"。**【核实】** |
| 事件级弱监督 / MIL | Sultani, Chen, Shah, CVPR 2018（bag=视频、instance=片段、MIL ranking）；Ilse, Tomczak, Welling, ICML 2018（attention pooling） | 支撑"对支撑区池化得到事件描述子"与"OR 型聚合"。**【核实】** |
| 掩码自监督 | PatchTST (Nie et al., ICLR 2023, arXiv:2211.14730) 含掩码自监督预训练 | 支撑 §3 的 $D_N$ 算子（从上下文预测被遮蔽 patch）。**【核实】** |
| 条件解码 | CVAE (Sohn, Lee, Yan, NIPS 2015)；CFG (Ho & Salimans, arXiv:2207.12598，正式版为 NeurIPS 2021 workshop 短文) | $D_A$ 属条件生成/条件重构；v1 不引入扩散，理由为 11GB 显存与训练成本。**【核实】** |
| 反事实不变性 | Veitch, D'Amour, Yadlowsky, Eisenstein, NeurIPS 2021, [arXiv:2106.00545](https://arxiv.org/abs/2106.00545) | 形式化"背景改变但属性不变"这一要求（§7.6 的 R1 就是它的一个具体实例）。**【核实】** |
| 评测基准 | TSB-AD (Liu & Paparrizos, *The Elephant in the Room*, NeurIPS 2024 D&B Track) | 仓库 `datasets/` 即该基准子集。**【核实】** |

**最近邻工作（必须正面处理）**：Lan et al., *Anlu: Enabling In-Context Time Series Anomaly Detection in Foundation Models via Counterfactual Supervision*, [arXiv:2610.06180](https://arxiv.org/abs/2610.06180) 用"一条 query + 两条不同 normal-rule 的 reference"做反事实配对监督。**【核实（据摘要）；全文尚未精读】** 这在**思路**上与 §8.3 的配对 episode 重叠。因此本方案的差异化不能停在"用了配对监督"，只能落在：(i) 方向证据被显式分解且带支撑（可用于定位与逐方向读出）；(ii) 加性正常/异常双重构与旁路可检验性（R1–R3、M1）；(iii) 少样本类型映射这一独立评测。**若精读后发现其已覆盖 (i) 或 (ii)，本方案的增量须相应降级**，这一点在接手时必须先做。

### 11.2 增量形态与判据

形态 = **问题形式化 + 机制改进**（不是范式创新，不声称从 0 到 1）。

- *形式化*：命题 1 说明检测损失对 $Z$ 的重参数化不变，从而把"属性有效性"从一个不可裁决的声明，变成必须由配对/不变性监督与 held-out 检验裁决的问题（可证伪）。
- *机制*：R1–R3 把"$Z$ 是否为异常信息的必要通道"变成可直接干预检验的因果命题（M1），而不是解释性叙事。
- *场景迁移*：跨背景/跨设备复用是目标场景，其隐含假设（全局常态）在零样本下失效，已有 RCD 论证；本方案附加更强也更可检验的主张——**同一异常属性在不同背景下应得到同一方向码**。

### 11.3 最强质疑（正面写进文档）

> "这不就是把 MoE + 对比学习 + 重构接到 TimeRCD 上吗？"

诚实回答：**如果 M4/M5/M6/M7 都只能达到参数匹配基线的水平，那么是的，且应判为负结果。** 本方案的可证伪核心只有三条：

1. $Z$ 是异常信息的必要通道（M1）；
2. 方向码跨背景可复用（M7）；
3. 组合可分解（M5）；

任一条不成立而其余成立，结论降级为"多了一个可解释中间层"，**不得**声称学到共享异常属性。

### 11.4 自我反驳（最可能的失败点）

最可能错在：**"方向"在时序异常里根本不可加性分解**。周期/相位变化、波形族替换并非对正常信号的局部平移，加性 $D_A$ 与 $K$ 个方向专家会系统性拟合不足，而非加性臂又缺乏可靠的可解释读出。若在 held-out 的 seasonal 家族上两臂都失败，应退回"事件描述子 + 少样本映射"这一较弱但可验证的形态。

---

## 12. 待用户裁定的取舍（价值判断，不替用户拍板）

| 编号 | 取舍 | 选项与影响 |
|---|---|---|
| T1 | 结构预算：$K_{\max}\in\{8,16\}$、$d_z\in\{16,32\}$、$m\in\{1,2\}$ | 预算大 ⇒ 表达力强，但逐专家可读性与可归因性下降 |
| T2 | 是否把正常反事实作为训练目标及可选推理输出（$D_N$ + bagging） | 训练中用于 $L_N/L_A$；检测推理默认跳过。若还需输出反事实/重构，再评估一次或多次遮蔽的收益与成本 |
| T3 | 是否接受训练期使用 family 标签做关系监督 | 接受 ⇒ 主张必须靠 held-out family 支撑；不接受 ⇒ 只剩无监督结构约束，识别力显著下降 |
| T4 | 少样本故障类型数据的来源 | 当前缺口，须用户指定；不擅自引入公开数据集 |
| T5 | 是否改动编码器（$W$、层数）以换取定位精度 | 改 $W$ ⇒ 亚 patch 稀释缓解，但与官方权重不兼容、需重训 |

---

## 13. 下一步最小验证（不训练大模型）

| 编号 | 动作 | 判据 / 目的 |
|---|---|---|
| V1 | 现有 generator 产 200 条 episode，实现 §8.6 标签完整性测试 + §8.5 三种归一化 | 半天；量出"持续型 vs 点型"的标准化敏感性 ⇒ 决定 $D_N$ 能否用 bagging |
| V2 | 单背景组跑 S1（无 $L_A/L_{\text{attr}}$） | 1 天；$s_t$ 是否达到直接检测基线（G1） |
| V3 | 开 $L_A$，测 $Lk$ 与 F1/F2 | 1–2 天；**只有 V3 通过才值得投 $L_{\text{attr}}$** |

任何一步不通过就停，**不改口径继续调参**（问题定义 §6、AGENTS.md §持续实验记录）。

---

维护规则：本文是 `design-v1.1` 的冻结候选。修改须升版本并说明依据；被否决的方案写进 `EXPERIMENT_LOG.md`，不在本文末尾叠加相互覆盖的"当前方案"。执行结果一律记入 `EXPERIMENT_LOG.md`，不写进本文。
