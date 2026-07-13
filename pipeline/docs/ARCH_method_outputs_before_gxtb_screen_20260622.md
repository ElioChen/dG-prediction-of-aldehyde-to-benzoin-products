# 现状架构 / 方法 / 输出 + 产物描述符 (2026-06-22)

在把 g-xTB 基线铺到 220k 全库筛选之前，记录当前模拟的架构、理论层级、输出，以及加入苯偶姻**产物**描述符的方案。

## 0. 反应与 ΔG 定义
- Homo-benzoin（220k 主筛选）：**2 R-CHO → R-CH(OH)-C(=O)-R**，ΔG = G(product) − 2·G(aldehyde)
- Cross-benzoin：D-CHO + A-CHO → D-C(=O)-CH(OH)-A，ΔG = G(prod) − G(donor) − G(acceptor)
- ΔG<0 = 热力学有利。目标是全程准确预测 ΔG（不按量级裁剪）。

## 1. 架构（三层层级）
| 层 | 用途 | 几何 | 能量 | DFT? | 成本 |
|---|---|---|---|---|---|
| **T1 Δ-model** | 训练/精确 | funnel_v3 | 半经验基线 + ML 修正 → DFT 级 ΔG | 训练时是 | 高 |
| **T2 Screen** | 220k 全库 | funnel_v3(rank-0) | 仅半经验 ΔG + QM 描述符，喂给 T1 模型 predict | 否 | 中 |
| **T0 Surrogate-2D** | 最快/兜底 | 无 | 纯 SMILES 2D → DFT 级 ΔG | 否 | 极低 |

数据流：filter → 构象搜索 → 半经验基线 ΔG + 描述符 →（训练加 DFT 标签）→ Δ-model →
对 220k screen 输出做 `model.predict` 得 DFT 级 ΔG + AD 置信。

## 2. 各阶段方法 / 理论层级
| 阶段 | 方法 | 溶剂 | 备注 |
|---|---|---|---|
| 过滤 | filter_v6（芳香醛，碳+杂芳环；剔脂肪/烯醛/炔醛/α-二羰基/MW>500；标 xtb_risk） | — | ~222k |
| 构象搜索 | funnel_v3 = ETKDG(密集,确定性) → MMFF → GFN-FF opt → GFN2 SP 筛 → GFN2 opt + **拓扑守卫**（丢断键构象） | ALPB-DMSO | 确定性，主误差杠杆 |
| 热力学几何 | GFN2 `--ohess tight` 在 rank-0 构象 | ALPB-DMSO | 给 RRHO 热校正 |
| **半经验基线** | **现已切 g-xTB `--gxtb --sp --cosmo dmso`**（在 GFN2-ohess 几何上），G = E_gxtb + GFN2-RRHO 热校正 | **COSMO-DMSO** | 旧=GFN2 ALPB；g-xTB vs DFT MAE 3.0 (旧 13) |
| DFT 标签（仅训练） | r2SCAN-3c SP（同几何）+ GFN2-RRHO 热校正 | CPCM-DMSO | 1% 验证 + 训练标签 |
| 描述符 | 醛侧 QM（见 §3） | — | Multiwfn L3 仅子集 |
| Δ-model | XGBoost，目标 = dG_orca − baseline，预测 = baseline + 修正 | — | CV MAE 2.00 |

## 3. 输出（能量 + 描述符）
**能量列**：`dG_xtb_kcal`（半经验基线 ΔG，现 g-xTB）、`dG_orca_kcal`（DFT 级 ΔG，仅训练）、分项
`G_ald_xtb_Eh`/`G_bz_xtb_Eh`、`xtb_energy`。几何：`xyz_file`。

**描述符（醛侧锚定在唯一 CHO 碳，screen schema 37 列 / 模型 63 特征）**：
- 全局 xTB 电子：HOMO/LUMO/gap/IP/EA/μ/η/ω/dipole、α0/C6
- CHO 位点电荷：Mulliken、Gasteiger、ADCH（C 和 O）
- 亲电性：Fukui⁺/⁻/0、dual descriptor、ADCH-Fukui
- 键：WBO(C=O)、QTAIM BCP(rho/lap/ell) 于 C=O
- 质子亲和：pa_CHO_O
- 立体/色散：%Vbur、Sterimol(L/B1/B5)、SASA、P_int
- RDKit-2D：MW/ExactMW/LogP/MolMR/TPSA/HBD/HBA/RotBonds/环系/BertzCT/Chi/Kappa…
- 基线：`dG_xtb_kcal` 本身也是一个特征

**模型输出**：`dG_pred`（DFT 级）、`correction`、`ad_flag`/`ad_distance`、`favorable`、`confidence`、`cho_class`。

⚠️ 当前 220k screen 的 ADCH/QTAIM(L3) 为空（Multiwfn ~分钟/分子，全库不可行），仅子集开启。

## 4. 加入产物描述符（待决策的扩展）
**现状**：模型只用**醛侧（反应物）**描述符。但 xTB–DFT 误差主要在**产物侧**（EWG 苯偶姻电子描述失败，见 hbond 分析）。给模型产物侧特征，理论上能让 ML 修正更有的放矢。

**已有工具**：`featurize_product.py`（cross_benzoin）已实现，把描述符从单一 CHO 重锚到产物的
**两个羰基衍生位点 + 分子内氢键**：
- 位点：ketC（酮碳）、ketO、carbC（连醇碳）、hydO、hydH
- 键：CO_ket(ketC=ketO)、CC_new(ketC–carbC)、CO_carb(carbC–hydO)
- 氢键：hydH···ketO（α-羟基酮普遍的五元环 H-bond）
- xTB：全局电子 + ketC/carbC 的 Mulliken & Fukui + 3 键 WBO + ketO 的 PA
- morfeus：ketC/carbC 的 %Vbur、CC_new 的 Sterimol、SASA、P_int、H-bond 几何(距离/角)+核心二面角
- Multiwfn：ADCH、ADCH-Fukui、QTAIM(CO_ket/CC_new + H-bond BCP) [慢，仅子集]
- ΔG：xTB ohess，dG=G(prod)−G(donor)−G(acceptor)，每个醛的 G 缓存复用

**集成路径**：训练 featurize 追加产物描述符 → 重训 Δ-model（反应物+产物特征）→ 重发；screen 侧
也要产物描述符（成本翻倍）。**成本**：产物 ~2× 大小，QM 描述符约翻倍；L3 Multiwfn 全库仍不可行
→ 产物 adch_/qtaim_ 在 220k 同样为空（与醛侧 gap 一致），只能在子集补。

## 5. 铺 g-xTB 到 220k 前的决策点
- **A. 最小**：screen 仅把基线 GFN2→g-xTB（在 GFN2-ohess 几何上加 `--gxtb --sp --cosmo dmso`），
  描述符保持醛侧。最便宜，直接拿到 4× 更准的基线 + 现成模型可用。
- **B. 加产物描述符**：A + 训练/screen 都加 featurize_product 的产物特征，重训反应物+产物模型。
  更可能提升精度（对症 EWG 产物误差），但描述符成本 ~2×、需重训、L3 仍只能子集。
- **C. 范围**：220k 是 homo-benzoin（donor==acceptor），用 featurize_product 的 homo 路径即可。
