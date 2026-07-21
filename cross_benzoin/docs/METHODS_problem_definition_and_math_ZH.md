# 问题定义与数学表达 —— benzoin ΔG 预测

整理自 GitHub 侧一次 Codex 会话的方法论讨论(`thought.md`,2026-07-14),
供以后写论文/技术报告复用。数学公式采用可直接用于论文的 LaTeX 风格记号。
对应英文版:`METHODS_problem_definition_and_math_EN.md`。

## 1. "benzoin ΔG" 究竟指什么

"从醛的 SMILES 预测 benzoin ΔG"这句话本身是欠定义的,必须先明确指的是
下面两个不同量中的哪一个,以及在什么条件下:

- **反应自由能** $\Delta G_{\mathrm{rxn}} = G_{\mathrm{product}} -
  G_{\mathrm{donor}} - G_{\mathrm{acceptor}}$ —— 一个热力学量,也是本项目
  实际预测的目标。
- **活化自由能** $\Delta G^{\ddagger}$ —— 需要 NHC 催化剂、碱、溶剂和具体
  反应机理,不能仅由两个醛的 SMILES 决定,不在本项目范围内。

要让 $\Delta G_{\mathrm{rxn}}$ 成为 `(donor SMILES, acceptor SMILES)` 这
一输入对上定义良好的函数,必须固定并记录以下内容,否则同一个 SMILES 对
可能对应多个不同的"正确答案":

- donor/acceptor **方向**(哪个醛被 NHC 活化为亲核的酰基负离子等价体)——
  cross 情形下每个无序对对应两个定向反应,见第 6 节;
- 由此得到的产物**区域异构体**(哪个羰基变成酮,哪个变成醇/carbinol);
- 温度(本项目:$T = 298.15\,\mathrm{K}$);
- 溶剂/隐式溶剂化模型(本项目:DMSO;GFN2 用 ALPB,g-xTB 用 COSMO,DFT
  用 CPCM);
- 标准态(1 mol/L 溶液);
- 电荷、质子化状态与互变异构体处理规则;
- 构象处理方式(最低能构象 vs. Boltzmann 系综,见第 4 节);
- 具体的电子结构方法及其版本(GFN2-xTB / g-xTB / DFT 级别、泛函、基组或
  复合方法——本项目的 DFT 标签级别是 r2SCAN-3c,见
  [[dft-labels-r2scan-not-pbe0]])。

## 2. 多精度能量分解(Δ-learning)

当几何结构和热力学校正来自便宜的级别(GFN2-xTB `--ohess`),而更贵的级别
(g-xTB 或 DFT)只在**同一个已优化几何结构上做单点计算**时,某个物种
$M$ 在高精度级别下的自由能可以通过复用低精度级别的热校正来近似:

$$
G_M^{\mathrm{high}} \;\approx\; E_M^{\mathrm{high,SP}} +
\left( G_M^{\mathrm{GFN2}} - E_M^{\mathrm{GFN2,SP}} \right)
$$

因此对于一个涉及 $\{A, B, P\}$(donor、acceptor、产物)三个物种的反应:

$$
\Delta G_{\mathrm{high}} \;\approx\; \Delta G_{\mathrm{GFN2}} +
\Delta\Delta E_{\mathrm{SP}}, \qquad
\Delta\Delta E_{\mathrm{SP}} = \delta E_P - \delta E_A - \delta E_B, \qquad
\delta E_M = E_M^{\mathrm{high,SP}} - E_M^{\mathrm{GFN2,SP}}
$$

这正是 `cross_benzoin/cb_featurize.py` 中 `_g_gxtb()` 在 g-xTB 级别上实际
执行的计算(每个物种做一次 g-xTB 单点,复用 GFN2 `ohess` 的热校正),也是
本项目 BDE/BDFE 描述符增益(见 [[bde-descriptor-idea]])以及 g-xTB→DFT
校正体系(见 [[gxtb-dft-correction-champion]])背后共同的恒等式。

## 3. 推荐的 ML 目标:按物种预测校正量,再按化学计量组合

不建议直接学习 $(\mathrm{SMILES}_A, \mathrm{SMILES}_B) \to \Delta
G_{\mathrm{DFT}}$,而应学习**按物种**的校正量,再按反应化学计量组合:

$$
f(M) = E_M^{\mathrm{high}} - E_M^{\mathrm{GFN2}}, \qquad
\widehat{\Delta G} = \Delta G_{\mathrm{GFN2}} + f(P) - f(A) - f(B)
$$

相比直接学习"反应对 → ΔG"的整体回归器,这种架构有以下优点:

- **热力学加和关系是结构性保证的**,而不是靠模型近似学出来的;
- 一个物种(某个醛,或在多个定向反应中共用的产物)的校正只需计算一次,
  就能在它参与的所有反应中复用/摊销;
- 会**自然地、无需改动架构**地退化为 homo 情形 $A = B$:
  $$
  \widehat{\Delta G}_{\mathrm{homo}} = \Delta G_{\mathrm{GFN2}} + f(P) - 2f(A)
  $$
- 在联合训练时,可以显式加入 **homo↔cross 一致性正则项**:
  $$
  \mathcal{L}_{\mathrm{consistency}} =
  \big| f_{\mathrm{cross}}(A, A) - f_{\mathrm{homo}}(A) \big|
  $$

## 4. 构象系综自由能

对于一个有多个相关构象 $i$、自由能为 $G_i$ 的物种,物理上正确的自由能是
Boltzmann 加权的系综平均,而不是单一最低能构象:

$$
G_{\mathrm{ensemble}} = -RT \ln \sum_i \exp(-G_i / RT)
$$

本项目的 `funnel_v3` 构象搜索目前报告的是**单一最低能构象**的自由能
(确定性 + RMSD + 拓扑守卫漏斗,见 [[crest-conformer-search]] /
[[conformer-search-noise]]),这是一个合理的实用近似,但并不等同于上面的
$G_{\mathrm{ensemble}}$——这是一个已被记录在案的近似,不是错误,也是一个
可能的改进方向(`boltz_relabel_worker.py` / `collect_boltz_relabel.py` 探针
已经在测试用 $K=5$ 的 Boltzmann 校正重新打标签与之对比)。

## 5. 角色感知的描述符构造

对于连续型/量化描述符 $\phi(\cdot)$,采用与第 3 节能量分解一致的化学计量
差分:

$$
\phi_{\mathrm{rxn}} = \phi(P) - \phi(A) - \phi(B)
$$

同时保留各物种的原始描述符块,使模型既能看到组合信号,也能看到每个物种
各自的上下文:

$$
\text{输入} = \big[\, \phi(A),\ \phi(B),\ \phi(P),\ \phi_{\mathrm{rxn}} \,\big]
$$

指纹类特征(如 Morgan/ECFP)采用拼接或计数差/异或的方式组合,而不是直接
相减(相减只对连续、近似可加的量才有意义)。每个描述符都带有**角色前缀**
——`donor_*`、`acceptor_*`、`product_*`、`interaction_*`——因为即使用同一个
原始描述符函数计算,donor 和 acceptor 在化学上也扮演着不同的角色(亲核体
vs. 亲电体);完整的分块表(donor 的 HOMO/Fukui⁻、acceptor 的 LUMO/Fukui⁺
等)以及交互项(donor HOMO − acceptor LUMO 能隙、Fukui 配对乘积、位阻/电荷
失配)见 `DESCRIPTOR_POLICY_CROSS.md`。

## 6. 模型输入构造(共享物种编码器)

对于 GNN 或任意可学习的按物种编码器 $\mathrm{Encoder}(\cdot)$:

$$
h_A = \mathrm{Encoder}(A), \quad h_B = \mathrm{Encoder}(B), \quad
h_P = \mathrm{Encoder}(P)
$$
$$
h_{\mathrm{rxn}} = \big[\, h_A,\ h_B,\ h_P,\ h_P - h_A - h_B,\
|h_A - h_B|,\ h_A \odot h_B \,\big]
$$

编码器**不能**对 donor/acceptor 这一对做成置换不变的:交换角色会改变
哪个羰基变成酮、哪个变成 carbinol,因此会同时改变产物结构和 $\Delta G$
(第 7 节给出了来自 `cross_pilot_v1` 的实证确认)。应改用显式的
donor/acceptor/product 角色 embedding。

## 7. 无序对 vs. 定向产物 —— 为什么要做这个区分

一个**无序对** $\{A, B\}$ 只是指定了两个醛的候选组合,并不声明谁扮演
哪个角色。但 NHC 催化的 benzoin 机理在一般(cross,$A \neq B$)情形下
**并不对称**:恰好有一个醛被催化剂去质子化/活化为亲核的酰基负离子等价体
(即 **donor**),它会进攻另一个未被活化的醛(即 **acceptor**)的羰基碳。
交换哪个分子扮演哪个角色,会改变哪个碳最终变成新的酮碳、哪个碳变成新的
carbinol(醇)碳——这是**两个化学上不同的区域异构体产物**,通常对应两个
不同的 $\Delta G$ 值。因此每一个 $A \neq B$ 的无序对 $\{A, B\}$ 都必须
展开成**两条定向反应**:$(A{\to}\text{donor}, B{\to}\text{acceptor})$ 和
$(B{\to}\text{donor}, A{\to}\text{acceptor})$,候选/标签空间才算化学上
完整;homo($A = B$)是两个方向重合的退化情形(在没有其他手性中心时)。

这不只是理论上的担忧:本项目第一次真实的 cross-benzoin 试点运行
(`data/cross_benzoin/cross_pilot_v1/`,job 24607515)已经给出了实证确认
——例如某个无序对,一个方向的 $\Delta G_{\mathrm{xtb}} =
-5.96\,\mathrm{kcal/mol}$,另一个方向是 $-11.35\,\mathrm{kcal/mol}$;
仅仅因为哪个分子被标记为"donor"、哪个是"acceptor",就产生了超过
5 kcal/mol 的差异,证实两条定向记录确实是不同的化学反应,而不是元数据
互换的重复项。

## 8. 评估方案 —— 四种泛化情形

单一的总体 MAE 会把"内插"和真正的"外推"混在一起。应分别报告以下四种
情形的指标:

1. **Homo 对角线**($A = B$)——本项目目前训练最充分的情形。
2. **训练内新组合**——$A$ 和 $B$ 各自都在训练集别处出现过,但这个具体
   配对没有出现过(检验组合泛化能力)。
3. **单侧外推**——$A, B$ 中恰好有一个在训练集中未出现过。
4. **双侧外推**——$A, B$ 都未出现过,理想情况下两者的 Bemis–Murcko 骨架
   也都未出现过——这是唯一真正检验向新化学空间外推能力的情形。

每种情形都应报告 MAE、RMSE、中位数 AE、P90/P95 AE、最大 AE、$R^2$、
Spearman $\rho$,以及**有符号平均误差**(系统性偏差),而不只是一个 MAE。
同一个无序对的两条定向记录必须始终留在同一个 split 里——否则模型可能
通过训练集中见过的反方向数据,"泄漏"出另一个方向的答案。

## 9. 推荐的数据划分层级

- **随机配对划分**——内插能力的上限(偏乐观的估计)。
- **分子隔离划分**——对已知单体的新组合的泛化能力(本项目在 cross-benzoin
  之外的既有默认做法,见 [[data-split-721]])。
- **双侧骨架隔离划分**(donor 和 acceptor 都做 Bemis–Murcko 骨架隔离)——
  最严格的测试,真正向新化学空间外推;截至 2026-07-14,尚未应用于任何
  cross-benzoin 结果。
