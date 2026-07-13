# MORDREDSLIM271_BDEGXTB SHAP 重要性 + 成本感知精简（20260707）

Bundle：`gxtb_dft_correction_MORDREDSLIM271_BDEGXTB_20260706.joblib`（测试集 MAE 1.503）。
在 4000 行测试子样本上做 SHAP 分析（XGB_d8）。

## BDE 与 BDFE：这个昂贵的描述符值不值它的成本？

| 特征 | SHAP 排名（共275） | mean\|SHAP\| | 获取成本 |
|---|---|---|---|
| prod_bdfe_gxtb_kcal | 15 | 0.1910 | 昂贵（`--ohess` Hessian+RRHO） |
| ald_bdfe_gxtb_kcal | 38 | 0.0991 | 昂贵（`--ohess` Hessian+RRHO） |
| prod_bde_gxtb_kcal | 6 | 0.4835 | 便宜（SP/opt） |
| ald_bde_gxtb_kcal | 4 | 0.5868 | 便宜（SP/opt） |

重要性求和：BDE（便宜，仅 SP/opt）= 1.0703；BDFE（昂贵，完整 `--ohess` Hessian+RRHO）=
0.2902；BDFE/BDE 比值 = 0.27。

**解读**：既然 BDFE 的重要性相对 BDE 较小，一个丢弃这 2 个昂贵 BDFE 列（只保留 BDE）的
成本感知变体值得训练并与完整 275 特征冠军模型对比——对于筛选新分子的前瞻性场景，BDE
（无需 Hessian）远比 BDFE 便宜得多。

## Mordred 重新精简

在现有 199 特征 SHAP 精简版 mordred 集之上，本轮重要性+相关性筛选保留了**199/199**
（本轮新丢弃 0 个冗余特征，即与更高排名保留特征 \|corr\|>0.9 的特征——注意：本轮重要性
排名反映的是*联合* 275 特征模型，而非该精简方案最初所依据的孤立 mordred510 模型，所以部分
此前保留的特征现在的有用程度可能有所变化）。

**成本感知候选方案**：273 特征（丢弃 2 个昂贵的 BDFE 列，保留 BDE）——需要独立重训练以确认
其精度保持不变；见 `mordredslim271_bdegxtb_slim_selection_20260707.json`
（`cost_aware_feats_total`）。

## 全局重要性 Top-25

| 排名 | 特征 | mean\|SHAP\| | 家族 |
|---|---|---|---|
| 1 | P_int | 1.8489 | QM(72) |
| 2 | ald_wbo_CO | 0.8527 | QM(72) |
| 3 | ald_P_int | 0.6099 | QM(72) |
| 4 | ald_bde_gxtb_kcal | 0.5868 | g-xTB BDE/BDFE |
| 5 | ald_xtb_dipole | 0.5723 | QM(72) |
| 6 | prod_bde_gxtb_kcal | 0.4835 | g-xTB BDE/BDFE |
| 7 | ald_pa_CHO_O | 0.4522 | QM(72) |
| 8 | xtb_dipole | 0.3884 | QM(72) |
| 9 | wbo_CC_new | 0.3311 | QM(72) |
| 10 | vbur_ketC | 0.3039 | QM(72) |
| 11 | mulliken_carbC | 0.2452 | QM(72) |
| 12 | ald_mulliken_CHO_C | 0.2339 | QM(72) |
| 13 | ald_mordred_Mor02se | 0.2279 | mordred |
| 14 | fukui_plus_ketC | 0.2114 | QM(72) |
| 15 | prod_bdfe_gxtb_kcal | 0.1910 | g-xTB BDE/BDFE |
| 16 | mordred_TopoPSA(NO) | 0.1892 | mordred |
| 17 | ald_mordred_PNSA3 | 0.1799 | mordred |
| 18 | mordred_PNSA3 | 0.1742 | mordred |
| 19 | mordred_Mor03v | 0.1707 | mordred |
| 20 | mordred_GeomRadius | 0.1687 | mordred |
| 21 | ald_SASA_total | 0.1668 | QM(72) |
| 22 | ald_mulliken_CHO_O | 0.1662 | QM(72) |
| 23 | ald_mordred_GeomDiameter | 0.1506 | mordred |
| 24 | ald_mordred_TPSA | 0.1490 | mordred |
| 25 | ald_mordred_WNSA3 | 0.1455 | mordred |
