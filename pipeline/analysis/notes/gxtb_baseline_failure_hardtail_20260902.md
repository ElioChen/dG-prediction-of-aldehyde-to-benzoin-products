# homo dG 难尾巴 = g-xTB 基线失败(2026-09-02)

**问题**:champion(MLP+XGB Δ-learning,`dG = dG_gxtb + ML修正`)在 P/磺酰基/亚胺/酰胺
多官能团分子上误差 ~10 kcal。是模型/特征不够,还是 g-xTB 基线本身崩了?照搬
`pipeline/bde/notes/nitro_delta_sp_bimodal_20260902.md` 的思路查。

**数据**:`boltz_relabel_hardtail_sample_20260710.csv` 的 150 个分子(当前 champion 测试集里
|error| 最大、且带 sulfonyl/P/imine/amide 标记的子集)。诊断表落盘为
`data/cross_benzoin/homo_v6/viz_gxtb_20260625/hardtail_gxtb_baseline_diagnosis_20260902.csv`。

## 结论:是 g-xTB 基线失败,不是模型问题

| 量(kcal/mol) | mean | mean\|·\| | p95\|·\| | max\|·\| |
|---|---:|---:|---:|---:|
| **g-xTB 基线误差** `dG_gxtb − dG_orca` | −9.21 | **13.89** | 33.1 | 42.0 |
| champion 残差 `dG_pred − dG_orca` | −0.90 | 10.01 | 15.3 | 20.6 |
| ML 实际施加的修正 `dG_pred − dG_gxtb` | +8.31 | 9.00 | 21.6 | 27.3 |

- **`corr(champion 残差, g-xTB 基线误差) = 0.888`** —— 模型的残差几乎线性地跟着 g-xTB
  基线误差走。**模型恰好错在 g-xTB 错的地方**。
- 全库 g-xTB 基线误差 ~4.26 kcal;这个子集 mean\|·\| **13.9,是 3.3×**。
- **44%**(66/150)的分子 |基线误差| > 15 kcal,**13%** > 25 kcal —— benzoin 缩合 ΔG
  典型只有 0–10 kcal,这些基线值物理上不合理。
- ML 修正确实在努力(施加 ~9 kcal 修正),但 Δ-learning 假设"基线大致对、学残差",
  一个 ~14 kcal 的破基线,平滑的 QM-特征回归补不回来。

### 按官能团(基线误差 mean\|·\|)
| 标签 | n | g-xTB 基线误差 mean\|·\| | champion 残差 mean\|·\| |
|---|---:|---:|---:|
| P(含 phosphine oxide) | 11 | **16.24** | 10.80 |
| P(裸) | 7 | **17.25** | 13.21 |
| sulfonyl | 37 | 15.74 | 10.10 |
| amide | 51 | 14.40 | 9.93 |
| imine | 11 | 10.00 | 8.97 |
| amide+sulfonyl | 11 | 9.23 | 9.30 |

**磷(尤其 P=O / 膦氧 / 鏻盐)是最严重的 g-xTB 电子结构失败**,磺酰基、重酰基取代次之
—— 与 cross-dG 侧"含磷是最大误差驱动(MAE 4.14 ≈ 基线 2×)"、以及 nitro BDE 侧的
g-xTB 失败,是**同一个根因**:g-xTB 在高价 P/S、强极性多官能团上不可靠。

## 可行动方向(便宜 → 贵)

1. **推理时硬路由**:分子若命中 `[#15]=O` / `[P+]` / 多磺酰基 / 多酰胺,不管 ensemble
   不确定性多低,**直接 route_to_dft**。当前路由器只按 ensemble std 挑 top 15%,
   这些 baseline-failure 分子不一定 std 大(模型可能"自信地错")。加一个 substructure
   veto 规则即可,零训练成本。
2. **在 hard-set 里单列一类** `gxtb_baseline_failure`(见 `build_hard_set.py`),
   和"特征不够"、"构象噪声"、"scaffold 新"分开,别混着当"模型难例"。
3. 换基线不值得:GFN2 对产物过稳定 ~15 kcal,也不行;没有更好的便宜半经验方法。

## 与"主动学习能否帮 homo"的关系
不能。这批分子的误差既不是数据量问题(见 [[homo-active-relabel-null-result]]),
也不是标签噪声问题(Boltzmann 重标 2× null),而是**基线方法在特定子结构上的系统失败**
——AL 治不了,路由 + 标注这类为独立类别才对。
