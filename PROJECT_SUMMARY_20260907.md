# benzoin-dg 项目总结（2026-09-07 版）—— 主动学习驱动的 cross-benzoin ΔG 预测

> **取代 `PROJECT_SUMMARY_20260904.md` 作为对外汇报底稿。** 09-04 版仍保留作历史
> 快照；本版重新围绕**主动学习（AL）主线 + ΔG 预测**组织，把 BDE / homo dG 两个
> 姊妹课题压缩成速览（§7），并纳入 09-07 收尾的四项"开放线复盘"工作（§6）。
>
> 英文版：`PROJECT_SUMMARY_20260907_EN.md`（两份改动时保持同步）。
> 逐日细节看 `RUN_LOG_20260903.md`；技术参考看
> `ALDEHYDE_LIBRARY_AND_DG_WORKFLOW_20260907.md`。

---

## 0. 一句话 + 关键数字

**目标**：给任意一对醛（donor + acceptor，可同分子）预测苯偶姻缩合反应的 ΔG
（kcal/mol）；用主动学习挑选最有信息量的下一批 DFT 计算，把"哪些反应对值得做
DFT"从盲选变成模型驱动的排序问题。

| 里程碑 | 数字 |
|---|---|
| AL 轮次 | **10 轮完整闭环**（round1→round10），2026-07-14 启动，2026-09-04 round10 落地 |
| 当前 champion | **r1-10 blend**（MLP+XGB tabular ensemble ⊕ triple-encoder attentive GNN，w_gnn=0.50） |
| champion 诚实 MAE（骨架不相交 holdout, n=448） | **2.215 kcal/mol**（g-xTB 物理基线 5.037，相对基线 **−56%**） |
| 训练数据规模 | 35,528 行 / clean-train 22,771 行（10 轮 AL 累计 DFT 标签） |
| **标签质量天花板** | 单构象 r2SCAN-3c DFT 标签噪声 std ≈ **2.9 kcal/mol** —— champion MAE 已贴地板 |
| **round10 AL 消融** | AL 诊断出真实盲点，但训练收益 **≈0**（−0.03，噪声内） |
| **几何方法偏差消融** | 三物种 ΔΔG 消融：hetero 中位 −0.47 vs ctrl −0.25 kcal，均 < 1 kcal 阈值 → **无偏差，标签质量调查结案** |
| **重表述为分类/排序** | 同一 champion 做"favorable/unfavorable"判别 **AUC 0.92-0.93**，top-10% 精度 **64%**（g-xTB 基线 31%，随机 10%）—— **不受标签噪声天花板限制**，09-07 起是 `predict_dg.py` 的默认输出列 |
| **homo/cross 差异（09-07 新）** | 对等切分口径+数据规模下 cross ≈ homo（甚至略好）；homo 1.503 的"优势"= 随机切分 + 12× 训练数据，**不是任务难度或标签质量** |
| 姊妹课题：BDE 预测 | champion B6 5-seed deep ensemble，骨架不相交全量 220k：醛 MAE **1.851** / 产物 **2.826** |
| 姊妹课题：homo dG（A+A，已上线） | 219,364 DFT 标签全覆盖，champion 测试 MAE 1.503（随机切分口径） |

---

## 1. 问题设定

### 1.1 化学 + 学习框架

- **反应**：donor 醛 + acceptor 醛（可同分子）→ 苯偶姻类产物。
- **预测量**：`dG_orca_kcal`，r2SCAN-3c / CPCM(DMSO) 单点，
  `dG = G(prod) − G(donor) − G(acc)`，几何来自 funnel_v3 构象搜索。
- **Δ-learning**：`dG_pred = dG_gxtb 物理基线 + ML 修正量`。ML 只学修正量、不从零
  预测，数据效率高一个量级。g-xTB 基线本身 MAE 5.037，ML 修正到 2.215。
- **特征**：260 维冻结 schema（局部 QM 描述符 + 精选 mordred + RDKit 2D），从
  round7 起未变。mordred 占 46%、RDKit 2D 19%、局部 QM ~35%。手工构造的
  `interaction_*` pair 特征试过，一个都没进最终列表（GNN 那半隐式学到了）。
- **模型**：单 XGB → MLP+XGB ensemble → **+ attentive-pooling triple-GNN 的 50/50
  blend**（三代架构演进，见 §3）。
- **评估**：Bemis-Murcko **骨架不相交** held-out 集，2026-07-17 后固定 n≈448。

### 1.2 主动学习为什么是主线

- 候选池 ~124 万个未标注醛对；DFT 单点昂贵。核心问题不是"能不能预测 dG"，而是
  "**下一批 DFT 算力花在哪些对上信息量最大**"。
- round1 用类别多样性铺开覆盖面；round2 起全部是 **bootstrap ensemble 不确定性
  AL**——用当前 champion 对候选池打分，按预测方差降序选下一批做 DFT。
- 十轮下来的核心问题演化成：**AL 到底在多大程度上真的推动了模型**（§3 收获 3）。

---

## 2. 十轮 AL 闭环（时间线）

| 版本 | 时间 | clean-train | ensemble-only MAE | GNN-only MAE | **blend MAE** | 备注 |
|---|---|---:|---:|---:|---:|---|
| R1-3 | 07-14~15 | ~2,472 | — | — | 2.966(CV)† | 类别多样性采样 |
| R1-5 | 07-16 | — | — | — | 2.633† | 首次三编码器 GNN，P=0.987(n=29) |
| R1-6 | 07-16 | — | — | — | 2.582† | GNN 未复现，P=0.456(n=29) |
| R1-7（旧切分） | 07-17前 | ~27,583前身 | — | — | 1.883(CV)† | screen10k 候选池耗尽 |
| **R1-7（骨架不相交重估）** | 07-17 | 19,687 | 2.256 | — | **2.215** | 泄漏勘误后的诚实数字 |
| R1-8 | 07-20 | 27,583 | 2.201 | 2.324 | **2.106** | attentive-pooling GNN 引入，P=0.9923 |
| R1-9（purge 前） | 07-21 | 43,367 | 2.163 | 2.162 | **2.074** | 学习曲线支持继续，P=0.988 |
| **R1-9（purge 后重算）** | 09-04 | ~32,630 | 2.254 | 2.313 | **2.167** | 标签从零重算，P=0.9954 |
| **R1-10（当前 champion）** | 09-04 | 22,771 | 2.326 | 2.313 | **2.215** | +round10 AL 批次，P=0.99855 |

（† = 2026-07-17 骨架泄漏勘误之前的旧切分数字，**不要引用为真实精度**。purge 前后的
r1-9/r1-10 由于底层 DFT 标签重算 + 骨架切分重建，数字不直接可比。）

**四个关键转折**：

1. **2026-07-17 骨架泄漏勘误**：早期"分子级别（InChIKey 不相交）"切分让 93% 的旧冻结
   留出集在骨架层面早已出现在训练集里。独立骨架不相交 5 折 CV 证实一个真实、可复现
   的 **+0.221 MAE（~9.8%）泛化差距**。历史上每一个"冻结留出集 MAE"都把真实泛化能力
   高估了 0.2-0.5。重建真骨架不相交 80/10/10 切分（干净测试集扩到 n≈450），所有数字
   重估——更诚实但更差。BDE 子课题独立踩到同一个坑（+43~47%），说明这不是偶然。

2. **2026-07-20 起 GNN 随规模翻盘**：round1-7 GNN blend 是 null；round1-8 引入
   attentive-pooling 后稳健显著（P=0.9923）；一路到 round1-10，GNN 权重还在涨
   （0.40→0.50）。**"小数据上架构对比得出的 null 结论"不能外推到更大规模。**

3. **2026-07~29 Snellius scratch 全量 purge**：round8/9 DFT 标签、醛侧 r2SCAN-3c SP
   缓存、所有 GNN 权重、BDE 描述符库全部丢失（gitignore + 无 home 备份）。2026-09-02
   起系统性恢复：能从 home 备份 bit-exact 重拼的重拼，其余从零重算（~41.5k 单点能）。
   round1-7 复现验证 CV MAE 1.877 vs 历史 1.883（噪声内），恢复可信。

4. **2026-09-04 round10 落地 + AL 消融**：round10 三腿 DFT 完成（1,969 对新标签），
   r1-10 champion 确认。**round10 AL 双层评估**见 §3 收获 3。

---

## 3. 方法论核心收获（十轮下来最重要的五条）

1. **诚实的评估比好看的数字更重要。** 骨架泄漏一次性让所有历史数字回撤 0.2-0.5 MAE，
   两个子课题独立踩坑。任何"冻结留出集"数字都要先问一句"测试骨架有没有在训练里出现
   过"。

2. **GNN blend 的收益随规模显现。** round1-7 null → round1-8 起稳健 → round1-10 权重
   还在涨。架构对比必须在目标数据规模上做，小数据结论不可外推。

3. **AL 能诊断盲点，不保证能修复盲点。** round10 是本项目第一次把这两件事分开测量：
   - *诊断层*：r1-9 blend 在 round10 那 1,969 个 AL 选出的对上 MAE = **2.780**（vs
     自己冻结集 2.167，差 28%）；不确定性 ⟂ 真实误差 Pearson r=0.40（p=9e-77），按
     不确定性四分位分组 MAE 从 Q1 2.245 单调升到 Q4 3.798。**AL 确实挑出了真实盲点。**
   - *修复层*：把 round10 的 30%（589 行）冻成硬测试集，用"含/不含另外 70%（1,370
     行）round10"训同一个 ensemble → ΔMAE = **−0.033**（噪声内）。**训练那 1,370 个
     AL 难例几乎不改变精度。**
   - 可能原因：+6.6% 的边际数据量；AL 难例是"不可约噪声难"而非"覆盖难"；~21k 对规模
     下这套 260 维特征已接近平台。**推荐：不要再做 ~2k 规模的 AL 轮。**

4. **标签噪声地板是硬约束。** 三条独立证据链在 2026-09-04 汇合：
   - 构象噪声探针（32 产物 × K=5 构象）：`dG_std` mean **2.975** / median 2.884；
     单构象 vs Boltzmann 平均标签差 |mean| 仅 0.15 kcal → **多构象重标签是 null**
     （homo 项目已 2× 独立确认，这是第 3 次）。
   - round10 AL 修复层 null（收获 3）。
   - 2026-07-21 三种 3D-GNN 架构（attentive3d / distattn / …）blend MAE 全挤在
     **2.177-2.188** → 真实 3D 几何相对 2D+attentive 无清晰增益。
   - 当 MAE 逼近标签本身的噪声 std 时，加同类数据 / 加模型容量都边际收益归零。

5. **重表述目标可以绕开标签噪声天花板。** 见 §5。

---

## 4. 标签质量调查全过程 + 结案

**动机**：champion MAE 2.215 逼近 ~2.9 的标签噪声地板 → 要么标签本身升级，要么换
问题形态。系统排查了三个"标签升级"杠杆：

| 杠杆 | 探针 | 结果 |
|---|---|---|
| 多构象 Boltzmann 重标签 | `confnoise_cross`（32 产物 × K=5） | ❌ null（§3 收获 4） |
| 换几何方法（GFN2 → g-xTB 精修） | `geom_bias`（12 杂原子重产物，单侧能量） | 产物侧结构化偏差：超价 P(=O) −9.2、多磺酰基 −5.0、硼 −4.2 kcal —— 看似有杠杆 |
| 换泛函层级（r2SCAN-3c → wB97X-3c） | `wb97x_shift` | ❌ 解析器打不通 range-separated 输出，判定优先级最低，取消 |

**决定性消融 `dg_geom_method`（2026-09-04 收官）**：`geom_bias` 只看了产物单侧。
但 `dG = G(prod) − G(donor) − G(acc)` 是差值——如果三物种几何偏差同向，donor/acceptor
侧会**抵消**掉产物侧的偏差。60 对（35 杂原子重 + 25 对照），每对三物种都做
GFN2-opt 和 g-xTB-opt（**g-xTB 从 GFN2 极小点精修，不独立重搜**——v1 用独立重搜结果
是构象噪声在冒充几何偏差，已修）→ r2SCAN-3c SP → `ddG = dG_gxtb − dG_gfn2`。

| 组 | n | 中位 ddG (kcal) | 范围 |
|---|---:|---:|---|
| hetero（杂原子重） | 34 | **−0.47** | −5.22 ~ +2.84 |
| control | 25 | **−0.25** | −4.43 ~ +1.96 |

`rmsd_prod_med = 0.20 Å`（<0.3 健全性阈值，确认不是构象噪声）。两组中位数都远低于
1 kcal 判定阈值、同号、分布高度重叠 → **产物侧几何偏差进入 ΔG 后被 donor/acceptor
抵消，无显著标签偏差。**

**结案**：不启动任何定向 g-xTB/r2SCAN 重标签战役，champion 维持 r1-10，资源全部
转向 Goal 3。**除非有新证据，不要再提定向重标签。**

---

## 5. Goal 3 —— 部署 + 重表述为分类/排序

### 5.1 重表述验证为正（2026-09-04）

**不重新训练任何模型**，直接拿冻结的 r1-10 champion blend 在它自己 n=448 holdout 上
的连续预测值做事后重表述：

| 指标 | ML (champion blend) | g-xTB 物理基线 | 随机 |
|---|---:|---:|---:|
| Spearman rho（预测 vs 真值排序） | **0.884** | — | 0 |
| T=0 二分类 AUC / acc / F1 | **0.934 / 0.911 / 0.697** | 0.781 / 0.752 / 0.448 | 0.5 |
| T=训练中位数 二分类 AUC / acc / F1 | **0.924 / 0.848 / 0.835** | 0.726 / 0.632 / 0.689 | 0.5 |
| 真实最优 10% 中模型 top-10% 命中率 | **0.644** | 0.311 | ~0.10 |
| 真实最优 20% 中模型 top-20% 命中率 | **0.756** | 0.511 | ~0.20 |

尽管连续值 MAE 卡在地板上，champion 的**排序/分类质量非常好**（AUC 0.92-0.93），且
每个切面都**远超 g-xTB 基线**。这是不受标签噪声天花板束缚、立刻可交付的产出。

### 5.2 部署工具 `predict_dg.py`（2026-09-07 起为标准输出）

给任意新分子对打分，端到端（featurize → assemble → prune → blend 推理）。输出列：
- `dG_pred_kcal`（点估计）、`ens_member_sigma`（廉价方向性不确定性）
- **重表述三列**：`dg_favorable`（dG<0）、`dg_below_train_median`、`dg_rank_pct`
  （批内百分位排名，用于给候选批排序）
- **部署健壮性三列（09-07 新增，见 §6 A）**：`dG_pi_lo_90`/`dG_pi_hi_90`
  （split-conformal 90% 区间）、`baseline_risk`/`baseline_risk_motifs`（g-xTB 基线
  失败子结构旗标）、`dg_high_sigma`（OOD 守卫）

**跨子项目 bug（09-06 引入，09-07 修复，commit `c520acf`）**：09-06 BDE 库全量重建
选的覆盖率哨兵字段（`G_gxtb`）恰好是 BDE 不需要、cross-dG 需要的量，静默让
`predict_dg.py` 对任何新分子对 100% 失败。修复：换成覆盖率 99.9%+ 的
`donor_xtb_HOMO` 做哨兵；`G_gxtb` 缺就走中位数填补。教训：**任何改共享库文件
schema 的工作，改完必跑 `predict_dg.py` 的 20-pair smoke test。** G_gxtb 缺口本身
09-07 也补齐（array `26432805`，覆盖率 1.3% → 99.07%）。

---

## 6. 2026-09-07 开放线复盘：A / B / C / D

用户对"项目静止"提出质疑，点名开放线（homo/cross 差异、S/P 处理、g-xTB 价值、
Δ-learning、GNN 架构）。全项目通读后，把开放线分为"已 resolved"和"仍 open"，推进
四项：

### A —— `predict_dg.py` 部署健壮性 ✅ 已交付（commit `0a2d5f7`）

`build_predict_dg_calibration.py` → `predict_dg_calibration.json`（骨架不相交
test+val 残差，n=929）。新增：

- **`dG_pi_lo_90` / `dG_pi_hi_90`**：split-conformal 90% 预测区间，±5.24 kcal，
  分布无关的边际覆盖 ≥ 90%（实测 0.94 test / 0.87 val）。区间宽是因为点估计贴在标签
  地板上、残差重尾。**σ-归一化版没更窄** → `ens_member_sigma` 条件信息不足，发全局版。
- **`baseline_risk` / `baseline_risk_motifs`**：超价 P / 磺酰基 / 亚砜 / 硝基 /
  N-氧化物 / Se / 三氟甲磺酸酯 SMARTS。holdout 验证：flagged 行 blend MAE **2.86 vs
  2.10**，|g-xTB 基线误差| **6.35 vs 4.81**（印证 homo hard-tail 的 corr(残差, 基线
  误差)=0.888）。`baseline_risk=True` = 建议直接上 DFT。
- **`dg_high_sigma`**：`ens_member_sigma` > calib p99（2.81 kcal）→ OOD，点估计和区间
  都别信。

冠军本体和 MAE 2.215 不变。

### B —— homo/cross ΔG 差异定量分解 ✅ 完成

`homo 1.503`（随机切分、全 219k）和 `cross 2.215`（骨架不相交、23k）并排引用，但**不是
同一把尺子**。用 `homo_unify_v1` 30k（purge 后唯一带 `dG_orca` + 骨架切分的 homo
数据），同 72-feat Δ 配方：

| 条件 | n_train | MAE | g-xTB 基线 MAE |
|---|--:|--:|--:|
| homo，**随机**切分 | 18,197 | **2.265** | 4.09 |
| homo，**骨架不相交**切分 | 19,030 | **2.608** | 4.43 |
| *cross ensemble-only（骨架不相交，22,771）* | 22,771 | *2.326* | *5.04* |

**homo 1.503 → cross 2.215（+0.712 kcal）分解：**

| 步 | ΔMAE | 是什么 |
|---|--:|---|
| homo 1.503 → homo 随机 @18k 2.265 | **+0.762** | 纯**数据规模**（154k→18k），同随机切分口径 |
| → homo 骨架不相交 @19k 2.608 | **+0.343（+15%）** | homo **自身的泄漏溢价**（内插→外推） |
| → cross ensemble-only @23k 2.326 | **−0.282** | 对等切分口径+规模+架构下，**cross 任务反而更容易** |
| → cross blend 2.215 | **−0.111** | homo 没有的 GNN blend |

**结论：差异 = 切分口径 + 数据规模，不是任务难度或标签质量。** 对等条件下 cross ≈
homo，甚至略好。homo 的泄漏溢价 +15%，和 cross 自己的 +9.8%、BDE 的 +43% 同一族——
"分子级切分 ≠ 骨架泛化"是三方独立确认的、与任务无关的效应。homo 骨架不相交 MAE 2.61
远在 homo 自己 ~2.1 地板之上 → 对等条件下 homo 是**数据/泛化受限**，而 cross blend
贴着 ~2.9 地板 → cross 更接近"做完了"。
详见 `data/analysis/homo_cross_gap/FINDING.md`。**不要再重提"cross 任务更难"。**

### C —— homo+cross 联合 tabular 🟡 AMBER-GREEN

"开放问题1"（homo+cross 联合训练）上次只在 ≤17k cross 行规模试过、搁置为收益萎缩。
现在 cross 23k + homo 26k（比例 ~1:1），72-feat 共享空间，eval cross 骨架不相交
holdout：

| | 单-XGB | MLP+XGB ensemble |
|---|--:|--:|
| cross_only | 2.812 | 2.716 |
| naive_merge（+homo，+is_homo 旗标） | **2.688（−0.124）** | **2.610（−0.106）** |
| finetune（homo → cross continue-train） | 2.849（null） | — |

`naive_merge` 在**两个模型类上都稳定提升 ~0.11 kcal** → 可复现，不是"弱模型爱数据"。
`finetune` = null。和 purge 前 BDE 侧结论相反，因为这里 homo:cross ≈ 1:1（无稀释）；
全量 6:1 规模下稀释会回来，除非 homo 降权。

**判决：值得一个有界的下一步**——给 30k homo_unify 做完整 260-schema featurize
（mordred + assemble，~1-2 天）+ homo 降权到 1:1 重训 cross ensemble/GNN，看 −0.1 能否
传导到 260-feat + GNN 的冠军；能的话再投 GNN homo-pretrain（**当前冠军 GNN 是纯
cross，purge 后从没重建过 homo 预训练路径**）。不是空白支票。

### D —— 更好的便宜 Δ-learning 基线 pilot 🏃 进行中（array `26441171`）

DFT 仲裁发现 g-xTB↔r2SCAN-3c 的差距主导来自**单点方法层级**（|Δ_SP| ~16 vs
|Δ_geom| ~5）→ 唯一没试过的精度杠杆是"换一个更好但仍便宜的单点方法当 Δ-learning
基线"。g-xTB（半经验）→ **B97-3c**（GGA 复合泛函，比 r2SCAN-3c 标签便宜 ~5-20×）。

128 对（64 杂原子 hard-tail + 64 对照），每物种一个新 GFN2 几何，同一几何上做
g-xTB / B97-3c / r2SCAN-3c 三个单点 → `resid_gxtb` vs `resid_b973c` 无构象噪声。

**前 97 对预览**：

| 分组 | g-xTB 残差 std | B97-3c 残差 std |
|---|--:|--:|
| 杂原子 hard-tail | ~5.1 | **~1.2** |
| 对照 | ~3.4 | **~0.8** |
| 全部 | ~4.6 | **~1.1** |

B97-3c 残差是个 ~−5.6 kcal 的**近常数偏置** + std ≈ 1.1。Δ-learning 模型平凡吸收
常数偏置 → **能达到的地板由 std 决定**：g-xTB ~4.6 → B97-3c **~1.1**。且在最难的
杂原子 hard-tail（g-xTB std 5.1）上 B97-3c 仍保持 std 1.2。若全量成立，**B97-3c 基线
可能把 Δ-learning 地板压到亚 1-kcal**——这会是几个月来第一个真正的精度杠杆。

⚠️ **陷阱**：一次性 ETKDG/GFN2 几何比生产 funnel_v3 标签偏 ~18 kcal，绝对 dG 不代表
生产设定。但 B97-3c↔r2SCAN-3c 的近常数偏置关系是 level-of-theory 属性、大概率
几何鲁棒。**判决前必须**：拿 ~30 对在生产 funnel_v3 几何上复核低-scatter 性质。

`merge_cheap_baseline_pilot.py` 在 array 跑完后给 GREEN/AMBER/RED 判决。

---

## 7. 姊妹课题速览

### 7.1 BDE 预测（键解离能）

直接预测反应中间体的键解离能本身（不是当 cross 项目的输入特征），两个目标键：醛
formyl **C–H**、产物中心 **ketC–carbC**。标签均为项目自己的 g-xTB 计算。

- **champion**：B6 = D-MPNN + 局部 3D 描述符经 `x_d` 通道融合。2026-09-06 全量 220k
  骨架不相交重训：**B6 5-seed deep ensemble 醛 MAE 1.851 / 产物 2.826**（旧的 42k
  局部库数字 1.579/3.060 已作废，数据规模不同不可比）。
- **唯一稳健的架构结论**：`x_d` 融合相对纯 2D 图（B4/B5）的优势（35-47 pts）。
  细粒度改动（attentive pooling、MAB）在骨架不相交下是泄漏假象。
- **Phase-3 3D BDE 模型：查证后不建**——查脚本发现它其实预测的是 `dG_orca−dG_gxtb`
  （cross-dG 目标），且这个架构族已在 dG 任务上验证过 null（§3 收获 4）。
- g-xTB 标签系统误差主导来自 single-point 层级（Δ_SP −21 vs Δ_geom 1.6）；构象噪声
  下限 ~2.1-2.7 kcal——和 cross 的 ~2.9 是同一类物理限制。

### 7.2 homo dG（A+A 同源，已上线）

- 219,364 DFT 标签覆盖全部 220,859 同源醛库。champion `ENSEMBLE72`，随机切分测试
  MAE **1.503**。推理入口 `benzoin_dG.predict_dG_champion()`。
- 与 cross 是两条独立模型线。homo 主动学习做过：pool-AL 不适用（库已全标注）；
  难尾巴 Boltzmann 重标签是 **2× 确认的 null**。难尾巴（P/磺酰基/亚胺/酰胺）
  残差 = **g-xTB 基线失败**，corr(残差, 基线误差)=0.888——推理时应按子结构
  route_to_dft（09-07 在 cross 侧实现为 `baseline_risk` 旗标，见 §6 A）。
- 09-07 的 B 分解表明：对等切分口径下 homo 并不比 cross 容易。

---

## 8. 基础设施事故（贯穿全程）

| 事故 | 时间 | 影响 | 应对 |
|---|---|---|---|
| git 对象库损坏 | 2026-07-13 | 历史重开，此前提交记录丢失（代码现状保留） | 无法挽回，历史从 `25f5400` 起计 |
| **Snellius scratch 全量 purge** | 2026-07-20~29 | B6 checkpoint、round8/9 DFT 标签、所有 GNN 权重、BDE 描述符库丢失 | 2026-09-02 起系统性恢复：home 备份 bit-exact 重拼 + 从零重算 + 建立"结果一律 `git add -f` + `submit_backup_recovery_artifacts.sh` 归档"纪律 |
| scratch1 inode 硬顶 133% | 2026-09-03 深夜 | 账号级 EDQUOT，写操作全面受阻 | 节流两个 BDE array + 孤儿清理器 + 几何改增量 tar.zst 压缩归档；01:05 回落硬顶下 |

**共同教训**：gitignore + 无 home 备份 = 一次存储策略变更就永久丢失；共享有限资源
（inode、GPU fairshare）上的自动化流水线必须内建监控 + 节流；会话内 Monitor 任务不
跨会话存活，交接文档需显式"随会话失效必须重建"提醒。

**2026-09-07 首次 push 到 GitHub**（`agent/recovery-20260902` 分支，此前 92+ commit
一直只在本地）。

---

## 9. 当前状态快照（2026-09-07 傍晚）

| 线 | 状态 |
|---|---|
| **cross-benzoin ΔG champion** | r1-10 blend，MAE 2.215，`CHAMPION.md` 冻结，未变 |
| **标签质量调查** | ✅ 结案，无几何方法偏差，标签噪声地板 ~2.9 |
| **Goal 3 部署** | ✅ `predict_dg.py` 端到端可用，默认输出重表述三列 + 部署健壮性三列 |
| **A 部署健壮性** | ✅ 已交付并 push |
| **B homo/cross 分解** | ✅ 完成——差异 = 口径+规模 |
| **C homo+cross 联合** | 🟡 AMBER-GREEN，有界下一步待用户拍板 |
| **D 便宜基线 pilot** | 🏃 array `26441171` 跑中（~97/128），前 97 对预览强正（B97-3c 残差 std ~1.1 vs g-xTB ~4.6）；判决前需生产几何复核 |
| **BDE champion** | ✅ 全量重训完成，醛 1.851 / 产物 2.826 |
| **homo dG** | 稳定在线，本轮无新动作 |
| **git** | 已 push 到 GitHub，HEAD 见 `git log`，除长期存在的 FILE_MAP.md / round9 模型改动外 clean |
| **在跑的本项目作业** | 只有 D（`26441171`）+ 两个 janitor 服务 |
| **集群** | genoa 09-07 白天 drain 过，傍晚恢复；新作业前先 `sinfo -s` |

**不要重新讨论的事**：
- champion = r1-10（blend MAE 2.215）。
- 几何方法偏差 = 无偏差，标签质量调查结案。
- round10 AL（2k 规模）= null，多构象重标签 = null，标签噪声地板 ≈ 2.9。
- 3D-GNN（cross）无清晰增益；Phase-3 3D BDE 模型不跑。
- homo 不比 cross 容易（对等切分口径下）。

---

## 10. 文档地图

| 文档 | 作用 |
|---|---|
| **本文件** | 对外汇报底稿，围绕 AL 主线 + dG 预测 |
| `PROJECT_SUMMARY_20260907_EN.md` | 本文件英文版 |
| `PROJECT_SUMMARY_20260904.md` | 上一版（历史快照，含更详细的 BDE 叙事） |
| `RUN_LOG_20260903.md` | 逐日细节（jobid / 决策 / bug，living doc） |
| `HANDOFF_20260907.md` | 会话交接（§0b = A/B/C/D 状态） |
| `cross_benzoin/CHAMPION.md` | champion 加载方式、数字、lineage、输出列 |
| `ALDEHYDE_LIBRARY_AND_DG_WORKFLOW_20260907.md`（+`_EN`） | 醛库构建史 + dG 工作流 + 描述符工程技术参考 |
| `data/analysis/homo_cross_gap/FINDING.md` | §6 B/C 的完整数字 + 分解 |
| `data/cross_benzoin/reformulation_classification_ranking_eval.json` | §5.1 分类/排序重表述完整数字 |
| `pipeline/bde/STATUS.md` | BDE 子课题权威入口 |
| memory `~/.claude/.../memory/*.md` | 见 `MEMORY.md` 索引 |

---

*本版 2026-09-07 撰写，围绕主动学习主线 + ΔG 预测重新组织。09-04 版保留作历史。
后续进展更新 `RUN_LOG_20260903.md`；本文件在有明确汇报用途时做增量编辑。*
