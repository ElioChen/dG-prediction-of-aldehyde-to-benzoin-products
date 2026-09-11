# 苯偶姻反应 ΔG 预测 —— 项目计划

> **活文档。** 项目每个组件的多层地图，各标 ✅ 完成 · 🔄 进行中 · ❌ 缺失/未启动 ·
> 🅿️ 搁置（试过，降优先级）。每项写：**原理**（为什么是对的一步 + 背后的化学/统计/ML
> 道理）、**状态**、**还缺什么**。
>
> 2026-09-10 按用户要求创建。English: `PROJECT_PLAN.md`（保持同步）。伴生文档：
> `LAB_JOURNAL_ZH.md`（每步操作的有理有据叙述）、`RUN_LOG_20260903.md`（电报体日志）、
> `PROJECT_SUMMARY_20260907.md`（汇报底稿）、`HANDOFF_<日期>.md`（会话交接）、
> `CHEMICAL_SPACE_ZH.md`（flying dataset 规范）、`CLUSTER_SHARING_ZH.md`（集群规则）。
>
> 用户要的顶层结构：**1. dG 模拟工作流 · 2. 预测优化工作流**
> （3. 催化剂空间不在本项目范围 —— 见 §3；+ 4. 横切基础设施）。

---

## 0. 目标与科学框架

**反应。** 苯偶姻缩合：两个醛（donor R¹CHO + acceptor R²CHO，可同分子）偶联成
α-羟基酮（"苯偶姻"）产物 R¹C(=O)–CH(OH)R²。实际由 NHC（N-杂环卡宾）或氰化物经酰基
负离子（"Breslow"）中间体催化。donor = acceptor 是 **homo**（2 A → AA）；不同是
**cross**（A + B → AB）。

**预测量。** 反应自由能 **ΔG = G(产物) − G(donor) − G(acceptor)**，kcal/mol，
DMSO 298 K，r2SCAN-3c / CPCM(DMSO) 水平 + GFN2-xTB RRHO 热校正。这是偶联的**热力学**
—— 告诉我们哪些配对是下坡（favorable 产物）、下坡多少。**不是**动力学势垒、**不**依赖
催化剂（催化剂侧是独立项目，§3，不在本项目范围）。

**为什么预测而不是逐个算。** 有用的设计问题是"~10⁶ 个可能的醛对里，哪些给 favorable
苯偶姻？"逐对 DFT ΔG 要几小时 CPU；训好的模型只要毫秒。所以本项目是**代理模型驱动的
筛选/排序**问题，DFT 只用来生成训练标签、仲裁模型最不确定的判断。

**"做完"是什么样：**
- （Goal 1）可信的 ΔG 数据集 + 能按需给任意新对打标签的模拟工作流。
- （Goal 2）骨架不相交 holdout 误差贴近标签噪声地板的模型，交付成可部署工具
  （`predict_dg.py`），带校准的不确定性 + favorable/unfavorable + 排序决策层。
- （Goal 3）部署：给大候选库打分，交给化学家一份排序清单，把模型的高风险判断路由到 DFT。

**当前一句话（2026-09-10）。** cross **参照模型** = r1-10 blend，骨架不相交 holdout
MAE **2.215 kcal/mol**（g-xTB 基线 5.037），贴在 ≈ 2.9 kcal/mol 的单构象标签噪声地板上。
Goal 3 工具已交付。**项目正在按用户 2026-09-10 的输入重新奠基**：cross 化学空间之前定义
错了（`candidates_v3` ≠ 真实的 220,859² 对空间），所以 cross AL 会在一个正确的
**"flying dataset"**（§2.11）上重做；过去的 AL 轮次和 r1-10 冠军保留为参照，不是前进主线。
活跃计算：(a) **B97-3c 便宜基线杠杆**（Tier B 重标注，可能突破 2.9 地板）、(b) 一个
**从零的全库 homo 模型**（重算 DFT 标签）。催化剂空间（§3）**不在本项目范围** —— 本项目
只管底物轴。

---

## 1. dG 模拟工作流 —— 真值 ΔG 标签

*整个工作流的原理：**一个方法，全部保存。** 每个物种（每个醛、每个产物）用完全一样的
构象搜索 + 几何 + 热校正 + 单点配方处理，所有几何、能量、描述符都持久化并按稳定整数
`id` 交叉链接。这让 ΔG 成为一致计算的自由能之差，也让后续任何步骤（重标注、特征、审计）
复用中间产物。*

### 1.1 反应与热力学目标 —— ✅
- **原理。** A + B → AB 的 ΔG。用自由能不用电子能，因为熵（偶联时失去一个分子的平动/
  转动自由度）是一阶效应（~+10 kcal/mol 不利）且随底物大小变化。溶剂（DMSO）重要，
  因为产物那个会氢键的 α-羟基酮和两个醛的稳定化不同。
- **状态。** 定义冻结。`dG = (G_prod − G_don − G_acc) · 627.509`。
- **缺。** 无。

### 1.2 醛结构库 —— ✅（这是真值源；另见 §2.11）
- **原理。** 单体的宇宙。`data/library/aldehydes_clean_v6.csv` = **220,859 个醛** ——
  一个大枚举过滤到可合成单醛，分类（`cho_class` ∈ 脂肪 / 芳香碳环 / 芳香杂环）以便
  类别均衡采样，带 `xtb_risk` 标志。**v6 库刻意包含算不出来的分子**（张力大、超价、
  巨大、构象面病态 —— GFN2 优化或 DFT 会失败）。那部分损耗（流程中 ~5–15%）是**库的
  属性**，不是流程 bug —— "这个算不了"是有效的、记录在案的结果。
- **状态。** 下游缓存：`homo_v6/aldehydes_all.csv` 209,526 行带 QM 描述符；~22 万带
  Mordred / BDE；homo 产物库 `products_all.csv` 184,199 行有有效结构（到 22 万的缺口
  ≈ 算不出来的比例 + 构建失败）。
- **缺 / 风险。** 恢复/重建的 CSV 有缺陷（2026-09-10 浮现）：(a) `"2.0"` 式 float id，
  不做 `qc.norm_id` 规范化会 string-join 到 ~0 行；(b) `G_xtb` 列对 **~8% 的行错位**
  （thermal −84…+87 Ha，或细微的如 19 原子醛上 1.23 Ha；home 备份里也是这些坏值 →
  不是 purge 损坏，是 featurize bug）。**规则：用任何存储的 xTB 能量/热校正前先界检**
  （Gibbs 热校正的每原子带 0.001–0.020 Eh/atom）。

### 1.3 构象生成 —— `conf_funnel_v3` —— ✅
- **原理。** 分子的自由能取决于你评估哪个 3D 构象；错构象是标签噪声的头号单一来源。
  `funnel_v3` 是分级搜索：RDKit ETKDG 集合 → 便宜的 GFN-FF / GFN2 筛 → 留低能漏斗 →
  一个拓扑守卫拒绝改变了成键的 embed。"v3" 修了一个坏 embed 静默污染标签的 bug。
- **状态。** round 7 起每个物种的生产方法。取代了 CREST 和更早的 funnel 变体（有记录：
  更慢 / 无精度增益）。
- **缺。** 主线无缺。多构象 **Boltzmann 平均**标签作为降噪杠杆试过 —— **2× 确认 null**
  （§1.7），所以保持单个最低-G 构象。

### 1.4 半经验几何 + 热校正 —— GFN2-xTB `--ohess tight --alpb dmso` —— ✅
- **原理。** 对 ~20 万物种做 DFT 几何优化 + Hessian 负担不起。GFN2-xTB 是紧束缚方法，
  便宜 ~1000×，给出足够好做 DFT 单点的几何，加一个全数值 Hessian → RRHO（刚性转子/
  谐振子）**热校正** `G_xtb − E_el_xtb`（ZPE + 热焓 − TΔS）。我们把这个 xTB 热校正叠在
  DFT 电子能上：`G_lvl = E_lvl(DFT SP) + (G_xtb − E_el_xtb)`。这是标准"复合"热化学 ——
  热校正比电子能对方法敏感度低得多，所以便宜的 Hessian 可接受。
- **状态。** 生产。几何 + xTB G/E 已持久化；2026-09 归档
  （`bde_homo_product_featurize_20260902/chunk_*/geom.tar.zst`）存了 ~18 万。
- **缺。** 2026-09 几何归档对 **chunk ~1400–1830 不全**（~2.5% 的对无可用几何）→ 那些
  要重新 GFN2 opt。坏的存储热校正（§1.2）→ 在归档几何上重跑 `--ohess` 重算
  （2026-09-10 已接进 `homo_sp_from_geom_worker.py`）。

### 1.5 DFT 单点 —— 🔄
- **原理。** 标签的精度层级。三个层级在用：
  - **r2SCAN-3c / CPCM(DMSO)** —— 项目**标签**。一个 "3c" 复合方法：r2SCAN meta-GGA
    泛函 + 专门的三-ζ 基组（def2-mTZVPP）+ D4 色散 + 几何 counterpoise 修正。选它是
    20 万规模战役下最好的精度/成本比；对更高泛函 benchmark 过（wB97X-3c 解析器打不通
    → 搁置）。
  - **B97-3c / CPCM(DMSO)** —— 候选 **Δ-learning 基线**（"便宜基线杠杆"，Rec-1 / D 工作
    集）。一个 GGA 复合方法，比 r2SCAN-3c 标签便宜 ~5–20×。pilot（128 对，3 个 SP 同一
    几何）：残差 `dG_r2scan − dG_b973c` = 近常数 −5.24 kcal 偏置（Δ-模型平凡吸收）+
    **std 1.11**，vs g-xTB 的 std 4.32。std 比 0.26 → 换基线后 Δ-模型理论误差地板降 ~4×。
    几个月来第一个真正的精度杠杆。
  - **g-xTB / COSMO(DMSO)** —— 当前最便宜的 Δ-learning 基线（半经验）。留作诊断 + 兜底。
- **状态。**
  - r2SCAN-3c 标签：✅ cross 1–10 轮（35,528 对）；❌ 全 homo 库（purge 中物理丢失 ——
    ~21.9 万里剩 3 万；任何地方都没备份）。
  - **cross Tier B 战役** 🔄：把全部 35,528 个 cross 对自洽重算（新几何 + 在这一个几何上
    同时算 r2SCAN-3c 和 B97-3c 和 g-xTB）。2026-09-10 ~72%，drain ~09-11/12。drain 后 →
    用 `BASELINE_COL = dG_b973c_kcal` 重训冠军。
  - **homo 全库 relabel** 🔄：在归档几何上做 SP（r2SCAN-3c 标签 + B97-3c 基线），
    179,431 对；+ 4,621 对的 regen 轨补几何缺口。2026-09-10 修了三个污染 bug 后干净重启。
- **缺。** Tier B merge + 冠军重训 + 判决（B97-3c 在全量规模上能否突破 2.9 地板？）。
  homo 全库标签。wB97X-3c / 更高层仲裁搁置，不需要。

### 1.6 复合自由能与 ΔG 组装 —— ✅
- **原理。** `G_lvl(物种) = E_lvl(SP) + thermal_xtb(物种)`；
  `dG_lvl = G_lvl(prod) − G_lvl(don) − G_lvl(acc)`（cross）或
  `G_lvl(prod) − 2·G_lvl(ald)`（homo）。逐对组进训练表，和特征放一起。
- **状态。** `assemble_cross_training_table*.py`、`homo_sp_from_geom_worker.py`。能用。
- **缺。** `assemble_homo_standalone_table.py` 的 `--full-library` 模式（列可用性已核；
  homo 标签落地后写）。

### 1.7 标签噪声表征 —— ✅（已结案）
- **原理。** 回归器练不到低于标签噪声。这里标签是一个 xТБ 构象上的一个 r2SCAN-3c 能量；
  构象选择注入噪声。
- **结果（3 个独立探针，2026-09-04 汇合）。**
  1. 构象噪声探针（32 产物 × 5 构象）：每产物 `dG_std` mean **2.975** / median 2.884
     kcal/mol；单构象 vs Boltzmann-均值标签只差 |0.15| kcal → **Boltzmann 重标注是 null**
     （homo 已 2× 独立确认，这是第 3 次）。
  2. round10 AL "修复"消融：在 1,370 个 AL 难例上训练，MAE 动了 **−0.03**（噪声内）。
  3. 三种 3D-GNN 架构 blend MAE 全在 2.18 → 真实 3D 几何相对 2D + attentive 无清晰增益。
- **状态。** **单构象 r2SCAN-3c 标签噪声地板 ≈ 2.9 kcal/mol。** 冠军（2.215）贴着它。
  无新证据不要重开定向重标注辩论。B97-3c 杠杆（§1.5）是被认可的尝试突破地板的方式，因为
  它改的是*基线*，不是标签几何。
- **缺。** B97-3c 杠杆在全量规模上是否真突破地板（= Tier B 判决）。

### 1.8 数据溯源与恢复 —— 🔄
- **原理。** gitignore 的大产物 + 无 home 备份 = 一次存储策略变更就永久丢。2026-07
  Snellius scratch purge 就是这样。
- **状态。** 2026-09-02 起系统性恢复：能从 `~/benzoin_backups` bit-exact 重建的就重建
  （醛 Mordred、产物 Mordred 2026-09-09 恢复、BDE）；否则从零重算（cross r8/9 标签、
  ~4.15 万 SP）。rounds 1-7 复现验证 CV MAE 1.877 vs 历史 1.883。纪律：结果 `git add -f`、
  `submit_backup_recovery_artifacts.sh`。
- **缺。** 全 homo DFT 标签（~18.9 万）确认不可恢复 → 正在重算。FILE_MAP.md 长期未提交
  的改动。

---

## 2. 预测优化工作流 —— 模型

### 2.1 学习框架 —— Δ-learning —— ✅
- **原理。** 不从零 `模型 → ΔG`，而是预测对一个便宜物理估计的**修正**：
  `ΔG_pred = ΔG_baseline + f_ML(特征)`，`f_ML` 学 `ΔG_DFT − ΔG_baseline`。基线（g-xTB，
  MAE 5.037）已抓住大部分电子结构物理 + 全部平凡的大小/熵标度，所以 ML 目标是个小的、
  光滑的、约零均值的残差 → 比直接回归数据效率高一个量级，且退化优雅（最坏 ≈ 基线）。
- **状态。** 冻结。`BASELINE_COL` 可 env 覆盖（`CB_BASELINE_COL`），Tier B 重训一个 flag
  就能 g-xTB → B97-3c。
- **缺。** 结构上无缺；开放问题是*哪个*基线（§1.5）。

### 2.2 特征化 —— 260 → 257 特征 schema —— ✅
- **原理。** 一个对由以下描述：(a) 每个醛反应原子上的**局部 QM 描述符** —— xTB 前线
  轨道、Fukui 指数、Mulliken/ADCH 电荷、Wiberg 键级、QTAIM 键临界点性质、buried volume /
  Sterimol 位阻 —— 每醛算一次复用；(b) **Mordred** 2D/3D 分子描述符（精选 slim 子集，
  ~schema 的 46%）；(c) **RDKit 2D** 描述符（~19%）；(d) 产物侧 QM + BDE 特征；(e) 手工
  `interaction_*` 对项。局部 QM 的想法：偶联化学发生在 formyl 碳，所以*那里*的电荷/
  亲电性/位阻应该携带大部分信号。
- **状态。** round 7 起冻结 260 特征 schema。2026-09-08 审计 → 3 个 `n_CHO` 特征退化
  （>99% 一个值）；删它们（schema v2 = 257）A/B 无害，post-Tier-B 重训生效。
  `interaction_*` 项：反复试，**一个都没进最终列表**（GNN 那半隐式学到了）。
- **缺。** 重尾 `wbo_CC_new` / `mulliken_*` 特征的 winsorize（推迟到 Tier B 重训）。
  ~21k 行下无证据更大特征集有帮助（学习曲线平）。

### 2.3 数据切分 —— Bemis–Murcko 骨架不相交 —— ✅（一个来之不易的勘误）
- **原理。** 随机或分子级（InChIKey 不相交）测试集会泄漏：测试分子的*骨架*已在训练里，
  模型在内插，报告的误差美化了真实（新骨架）泛化。诚实的测试是骨架不相交：训练和测试
  之间不共享 Bemis–Murcko 骨架。
- **结果（2026-07-17）。** 旧分子级切分里 93% 的冻结 holdout 骨架已在训练。干净的骨架
  不相交 80/10/10 切分（测试 n≈448，round 7 起冻结）揭示了**真实、可复现的 +0.221 MAE
  （~9.8%）泛化差距**。历史上每个"冻结 holdout MAE"都把真实泛化高估了 0.2–0.5。
  BDE 子课题独立踩到同一个坑（+43%）。homo 自己的泄漏溢价 +15%。
- **状态。** 只引用骨架不相交评估。任何"冻结 holdout MAE"先查骨架重叠。
- **缺。** 无。

### 2.4 表格模型 —— XGBoost、MLP+XGB ensemble —— ✅
- **原理。** 梯度提升树（XGB，depth-3，300 树，lr 0.05）是表格化学描述符的强基线 ——
  处理混合尺度、准单调关系、缺失值，不需要缩放。**MLP+XGB ensemble**（scaler + MLP
  回归器 + 2 个 XGB seed，平均）加了一个光滑函数学习器，抓树漏掉的信号；holdout 上比
  单-XGB 好 ~0.2 MAE。
- **状态。** `train_scaffold_disjoint.py` —— 单-XGB 冠军 + ensemble，按 `pair_key`
  group-K-fold CV。Holdout：单-XGB 2.548，ensemble 2.326。
- **缺。** 在 Tier B / schema-v2 表上重训。

### 2.5 图模型 —— 三编码器 attentive-pooling GNN —— ✅
- **原理。** 描述符丢掉了模型本可利用的成键拓扑；消息传递 GNN 直接读分子图。这个有
  **三个编码器**（donor、acceptor、产物图）、attentive pooling（学习加权原子成分子
  向量）、一个也吃 257 QM 描述符的头（`x_d` 通道）—— 所以是图 + 描述符，不是纯图。
- **结果。** round 1-7 数据上 GNN blend 是 **null**；round 1-8 起（attentive pooling）
  稳健显著（P=0.992），其 blend 权重此后**每轮都涨**（0.40 → 0.50）。教训：架构对比
  必须在目标数据规模上做，小数据 null 不能外推。
- **状态。** `train_cross_gnn_arch_sweep.py`（arch=attentive、h128、l4、lr3e-4）。CPU 训
  （r1-10 冠军 GNN 也是 CPU 训的）。GNN-only holdout 2.313。
- **缺。** post-Tier-B 重训成 **4-seed ensemble**（2026-09-08：4 seed 平均把 blend
  holdout MAE 2.215 → 2.137，w_gnn → 0.75；~0.5 bootstrap-SE 还不显著，但折进重训）。

### 2.6 Blend 与 seed 集成 —— ✅
- **原理。** 两个误差去相关的模型族 → 加权平均胜过任一。`w_gnn` 在 validation 切分上选
  （当前 0.50；4-seed GNN 下 ~0.75）。seed 集成降 GNN 的 ~±0.06 单 seed 方差。
- **状态。** `predict_cross_champion.py`（冠军加载器）、`blend_gnn_seed_ensemble_r10.py`。
- **缺。** Tier B + 4-seed 重训后重扫 `w_gnn`。

### 2.7 主动学习 —— 1–10 轮做完，但 🅿️ 暂停 / 待重做
- **原理。** DFT 贵；有用的问题是接下来标注哪些对。每轮：用 **pair-grouped bootstrap
  ensemble** 给未标注池打分，按预测 std 排序（query-by-committee epistemic 代理），
  DFT 标注 top batch，重训。
- **做了什么（2026-07-14 → 09-04）。** 在 `candidates_v3` 池（~124 万对）上 10 个闭环，
  产出 35,528 个标签集和 r1-10 blend 冠军（MAE 2.215）。round10 消融把*诊断*和*修复*
  分开：AL 确实找到盲点（r1-9 在选出的对上 MAE 2.78 vs holdout 2.17），但**在 1,370
  个上训练把 MAE 动了 −0.03**（噪声内）→ ~2k 规模的轮次不推动。
- **状态（用户，2026-09-10）：过去的 AL 暂时不采用，cross AL 会重做。** 原因：
  `candidates_v3` 池**不是真实化学空间** —— cross 建库错了（§2.11）。一个全新的 AL
  战役会在正确定义的池（220,859² 空间上的 flying dataset）上跑，等它和更好的标签
  （Tier B / B97-3c）就位。
- **保留什么。** 35,528 个 DFT 标签仍是有效数据。r1-10 blend 保留为**参照模型**，不是
  前进主线。
- **缺。** 正确定范围的候选池（§2.11）；新的采集策略决定（不确定性 vs 多目标 vs 覆盖），
  由修正后的空间告知；重做的战役。

### 2.8 评估与不确定性 —— ✅
- **原理。** (a) 点误差：骨架不相交 holdout MAE / R²。(b) **校准的不确定性**：
  split-conformal 90% 预测区间（分布无关的边际覆盖 ≥ 90%；±5.24 kcal，宽是因为点估计
  贴地板 + 残差重尾）。(c) **风险旗标**：`baseline_risk`（g-xTB 基线失败子结构 ——
  超价 P、磺酰基、亚砜、硝基、N-氧化物、Se、三氟甲磺酸酯；flagged 行 MAE 2.86 vs 2.10 →
  路由到 DFT）、`dg_high_sigma`（OOD 守卫）。(d) **重表述**（见 2.9）：把同一回归器输出
  阈值化成 favorable/unfavorable 判别或排序 → AUC 0.92–0.93，top-decile 精度 0.64 ——
  这些指标**不**受 2.9 kcal 标签地板约束，因为二元/序数判断在构造上对它鲁棒。
- **状态。** `build_predict_dg_calibration.py` → `predict_dg_calibration.json`（n=929）；
  `eval_reformulation_classification_ranking.py`。均 2026-09-07 交付。
- **缺。** 任何冠军重训后重校准。

### 2.9 部署 —— `predict_dg.py` —— ✅
- **原理。** 单一端到端入口：featurize → assemble → 裁剪到 schema → blend 推理 →
  给任意新醛对发点估计 + 区间 + 决策列。
- **状态。** 端到端可用（2026-09-07）。输出列：`dG_pred_kcal`、`ens_member_sigma`、
  `dg_favorable`/`dg_below_train_median`/`dg_rank_pct`、`dG_pi_lo_90`/`dG_pi_hi_90`、
  `baseline_risk`/`baseline_risk_motifs`、`dg_high_sigma`。慢几何情况的 SLURM 路径已接。
- **缺。** 对真正新对的一次新鲜精度检查（历史的"5 个已知对 MAE 1.65"早于一个已修的
  回归）。陷阱：任何改共享 `homo_v6/*_all.csv` schema 的工作，改完必跑 `predict_dg.py`
  20-pair smoke test（2026-09-06 一个共享文件 bug 静默让它对所有新对失败）。

### 2.10 子工作流

#### 2.10.a BDE 预测（键解离能） —— ✅ 冠军已训
- **原理。** 直接预测对偶联重要的两个键的 BDE —— 醛 formyl **C–H** 和产物中心
  **ketC–carbC** —— 作为独立目标（机理抓手），标签来自项目自己的 g-xTB 计算。
- **状态。** 冠军 **B6 = D-MPNN + 局部 3D 描述符经 `x_d` 通道融合**；2026-09-06 全量
  220k 骨架不相交重训：**醛 MAE 1.851 / 产物 2.826**（旧的 42k 局部库 1.579/3.060 作废
  —— 数据规模不同）。唯一稳健的架构结论：`x_d` 融合胜过纯 2D 图（35–47 pts）；更细的
  改动是骨架泄漏假象。
- **缺。** 无活跃项。Phase-3 3D BDE 模型：查证后**不建** —— 它其实预测 cross-dG 目标，
  且该架构族在那上面已确认 null。入口 `pipeline/bde/STATUS.md`。

#### 2.10.b homo-only ΔG 模型，从零 —— 🔄（活跃，用户 2026-09-10 指示）
- **原理。** 在**全**库上、用重算的 DFT 标签重建 homo（A+A）模型，用**单个 XGBoost** +
  **单个 attentive GNN**，各自独立报 —— 刻意*不用*迭代出来的 cross 冠军 blend，得到一个
  干净、理解清楚的全数据基线。
- **状态。** DFT 标签在重算（§1.5）。manifest 拆 179,431 archived-geom + 4,621 regen。
  2026-09-10 找到并修了三个污染 bug（orca_sp/ 目录撞车；`_extract` 跨 chunk 缓存撞车 →
  27% 垃圾 dG；坏存储热校正 → worker 内 `--ohess` 重算）+ 第 4 个（每原子热校正界）。
  战役干净重启。旧的 purge 前 homo 冠军（`ENSEMBLE72`，随机切分 MAE 1.503）仍在线作参照。
- **缺。** 干净标签战役 drain（~2–3 天）→ merge + QC 判决 → `--full-library` assembler →
  两个模型 → 结果进 `data/cross_benzoin/homo_standalone/README.md`。

#### 2.10.c homo + cross 统一（Rec-2） —— 🟡 AMBER-GREEN，准备中
- **原理。** 若对等条件下 homo 和 cross ΔG 是同一物理（B 工作集表明如此 —— 表观差距是
  切分口径 + 数据规模，不是任务难度），那把两者池化成一个带 `is_homo` 标志的模型应该
  经由更多数据 / 更广化学覆盖帮上忙。
- **状态。** `naive_merge`（homo:cross ≈ 1:1，72-feat）在两个模型类上都可复现地
  **−0.11 kcal**；`finetune` 是 null。产物 Mordred 已恢复（2026-09-09）。assembler
  `--full-library` 模式待写。
- **缺。** 等从零 homo 标签（2.10.b）→ 组一张统一 260-feat 表 → 带 `is_homo`/
  `sample_weight` 重训冠军 + GNN → −0.11 在 260-feat + GNN 规模下存活吗？存活的话再加
  GNN homo-pretrain→finetune 路径（当前冠军 GNN 是纯 cross，purge 后从没重建过 homo
  预训练路径）。

#### 2.10.d 便宜基线杠杆 —— cross Tier B 重标注（Rec-1 / D 工作集） —— 🔄
- **原理。** §1.5：把 Δ-learning 基线 g-xTB → B97-3c。pilot（残差 std 4.32 → 1.11）说
  Δ-模型地板可能降 ~4×。
- **状态。** 全部 35,528 个 cross 对的自洽重标注在跑（2026-09-10 ~72%）。drain 后：
  `merge_rec1_b973c_tierB.py` → QC 判决 → `patch_train_table_tierB_b973c.py`（schema v2）→
  `CB_BASELINE_COL=dG_b973c_kcal` 重训 → GNN 4-seed → blend 重扫。Runbook：
  `data/cross_benzoin/rec1_b973c_tierB/DRAIN_RUNBOOK.md`。
- **缺。** drain + 重训 + 判决（holdout MAE 降到 ~1.0–1.5 = 项目级突破，还是杠杆在规模上
  被冲淡？）。部分数据预览显示 holdout ens MAE 2.53 → 0.69（对照隔离）—— 强信号，还不是
  headline。

### 2.11 化学空间定义与 "flying dataset" —— 🔄 构建中（用户，2026-09-10）
- **原理。** 一个筛选模型只有它筛的空间有意义才有意义。用户两个修正：
  1. **homo 空间** = 220,859 个 v6 醛的自配对。有些设计上算不出来（§1.2）—— 那是记录
     在案的结果，不是缺口。
  2. **cross 空间** —— 旧 `candidates_v3` 池（~124 万对）是一个**构造错的子集**，不是
     真实空间。真实 cross 空间 = **每个 v6 醛的有向对：220,859² ≈ 4.88 × 10¹⁰**
     （~2.44 × 10¹⁰ 无向；donor/acceptor 角色化学上不同所以有向是诚实计数）。
     `candidates_v3` 应退役为"那个池"。（220,859 = 2026-09-11 修正，CSV 220,860 行是含
     表头的行数，之前少减了 1。）
- **"flying dataset" —— 设计。** 你不能也不应落地 488 亿个产物 SMILES。而是一个
  **惰性 / 虚拟**数据集：
  - **基础索引。** v6 醛库是唯一真值源，冻结一个 canonical 整数索引 `0 … 220,859`
    （稳定，永不是即时 enumerate 索引；= 库的 `index` 列 / InChIKey）。
  - **一个对就是一个地址**：`(donor_idx, acceptor_idx)`。磁盘上没有对表。
  - **每醛缓存**（算一次、到处复用）：几何、xTB 能量 + 热校正、局部 QM 描述符、Mordred、
    BDE —— 按基础索引 key。这些大部分已在 `homo_v6/` 下。
  - **按需生成**：给一个对地址，生成器产出产物 SMILES（反应模板）、从两个醛缓存 +
    产物侧特征 + 对/交互项组出 260 特征行、（可选）g-xTB / B97-3c 基线 —— 全部惰性、
    流式、最多短暂缓存。
  - **交付物**：一份规范文档 + 一个薄读取 API（`pair(i, j) -> {smiles, features,
    baseline, split}`），让未来任何步骤（模拟、标注、预测、AL 采集）统一读这个空间，
    不用一张巨表。split 分配（骨架不相交）由两个醛的 scaffold 现算。
- **状态。** `CHEMICAL_SPACE_ZH.md`（规范，§8 构建顺序）2026-09-10 写完。构建顺序
  第 1-2 步 2026-09-11 完成：`data/chemical_space/aldehyde_index.parquet` 已冻结
  （220,859 行，`ald_idx` + canonical SMILES + 复用 `candidates_v3` 醛侧 scaffold，
  逐位校验过匹配）；`homo_v6/aldehydes_all.csv` 核实早已用 `ald_idx` 正确 key
  （209,526/209,526，0 孤儿）。
- **缺。** 第 3-6 步：`chemical_space.py` 的 `pair(i, j)` 读取 API（含第3步对 ~20
  个已知对的校验，不能跳）；标签/split/基线；迁移现有 35,528 个标签；`candidates_v3`
  退役。（详见 `CHEMICAL_SPACE_ZH.md`。）

---

## 3. 催化剂空间 —— 本项目**不管**（2026-09-10，用户）

**本项目只是底物轴。** 它预测一个醛对是否给热力学 favorable 的苯偶姻（无催化剂 ΔG）。
它**不**建模 NHC 催化剂、动力学势垒、对映选择性。

- **边界。** 把模型的 favorable / 排序输出读作"值得细看"，不是"能行" —— 一个**热力学
  预筛**。`predict_dg.py` 文档应明说（一个小 ❌ 待加）。
- **催化剂侧是独立、成熟的工作**，在姊妹仓库里（`ElioChen/nhc-benzoin-pipeline`、
  `nhc-benzoin-active-learning`、`nhc-active-learning`、`stereo-catalyst-engine`、
  `nhc-pkah-predictor`）：每非对映体的 TS_CC / TS_CN → 微观动力学 → rate（Kozuch–Shaik
  能量跨度）+ |ee|，多目标池式 AL 覆盖 ~1300 万立体异构体的 NHC 库。**这里不管。**
  集群上的 `nhc-gsp-*` / `sourceB-kinetics` / `qm-benzoin` 作业属于它 —— 本项目会话
  不碰。
- **任何底物 × 催化剂的联合是用户 / NHC 项目的事，不是本计划的事。**

---

## 4. 横切基础设施

| 组件 | 原理 | 状态 | 缺 |
|---|---|---|---|
| **计算** | Snellius SLURM：rome（521 节点，大共用池）、genoa、fat_rome/fat_genoa（4.3× 内存，贵 50%，无 QOS 上限）、gpu_h100/a100。rome+genoa 上 QOS `MaxJobsPU=128`。 | ✅ 能用 | 必须**和 NHC 项目分享节点**（同账号，用户指示 2026-09-10）—— `%N` 留余量，别占满一个分区。规则见 `CLUSTER_SHARING_ZH.md`。 |
| **环境** | `venv/nhc-workflow`（pandas/xgb/sklearn/rdkit/mordred）、`venv/nequip`（GNN CPU）、`/home/schen3/xtb`（g-xTB 能力）、`/home/schen3/orca`（`ORCA_SCF=default`）。 | ✅ | `envs/gnn` 共享环境损坏（空 stdlib）—— 没用，忽略。 |
| **数据存储** | 活仓库 `benzoin-dg-restored`；大产物 gitignore → 必须 `git add -f` + 归档。 | 🔄 | 2026-07 purge 恢复进行中（§1.8）。 |
| **备份** | `~/benzoin_backups` 挺过了 purge；`submit_backup_recovery_artifacts.sh` 归档新结果。 | ✅ 纪律就位 | homo 全 DFT 标签从没备份 → 丢了。 |
| **监控** | 会话级 `Monitor` 任务（guardian：janitor + inode；每阵列一个 campaign monitor）。**不跨会话存活** —— 每次交接必须重建。 | ✅ 模式 | 固有脆弱；交接 §3 总列重建命令。 |
| **交接套路** | 每次会话结束：commit/push、写 `HANDOFF_<日期>.md`（固定 8 节结构 + 自主推进条款）、收尾 `RUN_LOG`、更新 memory、产出 resume prompt。 | ✅ | —— |
| **Journal + Plan**（本文件 + `LAB_JOURNAL_ZH.md`） | 用户指示 2026-09-10：有理有据的每日 journal + 这份活计划；理解每一步的原理。中英各一份。 | 🔄 刚开始 | 保持两份同步、都跟进。 |

---

## 5. 已定的决定 —— 无新证据不再讨论

- cross **参照模型** = r1-10 blend，MAE 2.215（已暂停的 AL 1–10 轮产出）。是要超越的
  基线，**不是**冻结的前进主线 —— cross AL 在正确定义的空间上重做（§2.7、§2.11）。
- **`candidates_v3` 池是错的。** 真实 cross 空间 = 220,859² 有向醛对；建 flying dataset
  （§2.11），退役 `candidates_v3`。
- **v6 醛库设计上包含算不出来的分子** —— "算不了"是记录在案的结果，不是 bug。
- **标签噪声地板 ≈ 2.9 kcal/mol**（单构象 r2SCAN-3c）。参照模型贴着它。
- **DFT 标签无几何方法偏差**（三物种 ΔΔG 消融；产物侧 g-xTB vs GFN2 偏差在 ΔG 差值里
  抵消）。标签质量调查**结案**。
- **round10 规模的 AL 对精度是 null。** 停 ~2k 规模的 AL 轮。
- **多构象 Boltzmann 重标注是 null**（3× 确认）。
- **3D-GNN 相对 2D + attentive pooling 无清晰增益**（cross）。Phase-3 3D BDE 模型不建。
- **对等切分口径 + 规模下 homo 不比 cross 难。**
- **B97-3c 基线杠杆**是被认可的尝试突破地板的方式（Tier B）。
- **骨架不相交是唯一诚实的评估。** 任何冻结 holdout 数字在查骨架重叠前都可疑。
- **不追速度；理解每一步**（2026-09-10）。
- **不碰** NHC / kinetics / mace / orcasp / qm-benzoin 作业。

---

## 6. 立即行动（2026-09-10）

**在飞的计算（都已降并发，和 NHC 分享集群）：**
1. **cross Tier B** 🔄 —— B97-3c 重标注，~72%。drain 时（monitor `bpdz6bks8`）：
   `DRAIN_RUNBOOK.md` step 1-6 → 冠军 + GNN(4-seed) 重训，`CB_BASELINE_COL=dG_b973c_kcal`，
   schema v2。判决：holdout MAE 降到 ~1.0–1.5 吗？*（这个重训是 r1-10 数据上的"更好标签"
   实验 —— 影响参照模型，不是重做的 AL。）*
2. **homo 从零标签** 🔄 —— 三个污染 bug + 第 4 个（每原子热校正界）后干净战役。
   archived 轨（`submit_homo_sp.sh`，179,431）+ regen 轨（`submit_homo_regen.sh`，4,621）。
   drain 后：`merge_homo_sp.py` → QC → `--full-library` assembler → 单 XGB + 单 GNN。

**设计 / 结构工作（无计算，慢慢做 —— 用户：理解每一步）：**
3. **Flying dataset 构建**（§2.11）—— 🔄 第 1-2 步 2026-09-11 完成（索引已冻结、
   `aldehydes_all.csv` 核实已用 `ald_idx` key）。下一步：第 3 步，写
   `chemical_space.py` 的 `pair(i, j)` 特征路径，对 ~20 个已知对校验后才能信它。
   重做 cross AL 和 Goal-3 筛选的前提。
4. **Rec-2 统一** 🟡 —— homo 标签落地后：统一表 → 重训 → −0.11 在 260-feat + GNN 下存活吗？
5. **催化剂空间不在范围** —— 本项目只管底物。任何联合是用户 / NHC 项目的事。
6. **重做 cross AL** —— (3) 和更好的标签之后：在 flying dataset 上定采集策略，跑新战役。

保持 `LAB_JOURNAL_ZH.md` + `LAB_JOURNAL.md` 跟进；每晚收尾。
