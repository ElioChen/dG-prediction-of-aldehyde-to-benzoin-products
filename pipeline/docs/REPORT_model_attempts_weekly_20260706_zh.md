# 本周预测模型尝试与结果一览（2026-07-06）

**模型谱系（核心指标 = g-xTB→DFT 校正的 test MAE, kcal/mol）：**

| 尝试 | 特征/方法 | test MAE | 结论 | 报告位置 |
|---|---|---|---|---|
| 起点（基线） | ENSEMBLE72（72特征） | 1.61 | 沿用中 | 06-26 已有 |
| Mordred 目标子集 | +438维 mordred（弥散/尺寸/形状） | **1.517** | 真实提升 | `REPORT_MORDRED510_FINAL_20260703.md` + PNG 70-79系列 |
| SHAP 剪枝 | 271维（72+199，半量） | **1.525** | 同精度减半特征，曾经的首选 | `REPORT_MORDREDSLIM271_FINAL_20260703.md` + PNG 90-99系列 |
| ADCH/QTAIM、morfeus-9、RDKit-434 | 三个正交描述子族 | 持平或更差 | **空结果**，排除 | 无独立报告，console记录 |
| SELFIES/ECFP/序列GRU 代理模型 | 直接从结构预测ΔG | 3.0-3.4（跑输2.92基线） | **空结果**，排除 | `pipeline/train_selfies_surrogate.py`、`train_reaction_repr_compare.py`（无md，仅console表格） |
| GFN2级 BDE/BDFE | 键解离能/自由能 | BDE+0.024边缘，BDFE空 | BDE勉强算真，BDFE排除 | 见记忆 `bde-descriptor-idea.md` |
| g-xTB一致 BDE/BDFE（quick-check） | +4特征，2-XGB快检 | 1.612→**1.563** | 真实（超噪声带4倍） | `REPORT_bdfe_gxtb_full_augment_20260706.md` |
| **★ 完整生产管线（最终版）** | 275特征，MLP+3×XGB | 1.525→**1.503** | **真实但温和，新任冠军** | `REPORT_MORDREDSLIM271_BDEGXTB_FINAL_20260706.md` + PNG **100-109系列** |

**全部报告/图片统一路径：**
`/gpfs/scratch1/shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/viz_gxtb_20260625/`

**当前生产冠军模型 bundle：**
`pipeline/models/gxtb_dft_correction_MORDREDSLIM271_BDEGXTB_20260706.joblib`（含MLP+XGB_d8+XGB_d10 ensemble、不确定性路由、275特征）

**本周净效果：** MAE 从上上周的 1.61 → 本周末的 **1.503**，累计降幅 ~7%；主要贡献来自"有针对性"的 mordred 子集（而非盲目整包）和 g-xTB 口径一致的 BDE 特征；SELFIES/ECFP/序列模型、纯 GFN2-BDFE 等方向被明确排除。
