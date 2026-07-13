# 构象/几何 与 xTB–DFT ΔG 偏差 — 综合结论与建议 (2026-06-22)

综合 funnel_v3 构象研究、1% DFT-SP 验证、g-xTB pilot/同构象/全 1% 几何-DFT-SP，
对"构象优化不足 → xTB ΔG 不准"这个问题给出收敛性结论。**核心更正**：先前
"EWG 误差是 g-xTB 能量模型、残差 ~35 kcal、g-xTB 未闭合差距"的判断，建立在 13 个
DFT **不收敛**的硬例上，参考是假的。全 1%（n=2191，DFT-SP 在 g-xTB 几何上 2191/2192
收敛）给出的图景相反。

来源：`gxtb_test/gxtbgeom_dftsp_1pct_all_20260621.csv`、`dft_sp_r2scan3c/analysis/`、
[[delta-mae-noise-floor]]、[[conformer-search-noise]]、[[crest-conformer-search]]、
[[hbond-not-product-error-driver]]、[[gxtb-install]]。

---

## 1. 三个干净的对照（口径已对齐：同相/同热校正）

g-xTB opt+hess 与 DFT-SP **都是气相**；下面用同相、同热校正的列做对照，避免溶剂污染。

| 检验 | 控制了什么 | 结果 (n=2191) |
|---|---|---|
| **几何质量**：DFT 电子能在 g-xTB 几何上是否更低（逐物种） | 同法同相，只变几何 | g-xTB 几何更低仅 **醛 1.1% / 苯偶姻 3.3%** → GFN2/funnel 几何 97–99% 更优 |
| **几何对反应能的影响**：DFT 电子 ΔE(gxtb几何) − DFT ΔE(xtb几何) | 同法同相，只变几何 | 中位 −2.6、MAD **7.84**、**66% >5 kcal** |
| **能量模型**：g-xTB ΔG vs r2SCAN-3c ΔG，同 g-xTB 几何+热校正+气相 | 只变电子能量方法 | **MAE 3.30、bias −2.12、RMSE 4.46、r 0.966**；58% 在 3 kcal 内，仅 3.3% >10 |

对比基线：GFN2 vs DFT（1% DMSO）**MAE 15.04、bias −13.64、r 0.574**。

## 2. 由此得到的三条结论

1. **主误差来源是构象搜索，不是半经验能量方法、也不是几何优化器。**
   - funnel_v3（更好的构象搜索）单独把 Δ-模型 MAE 2.68→2.19/2.23（[[delta-mae-noise-floor]]）。
   - 即使在 DFT 级别，仅换几何/构象就让反应 ΔE 摆动 ~8 kcal MAD（66% >5）。
   - g-xTB 几何逐物种 97–99% 比 GFN2 几何能量更高 → **g-xTB 不是更好的几何引擎**；
     "g-xTB 几何低 40 kcal" 只是个别构象跳变。同构象检验里 GFN2 几何 14/17 胜。

2. **g-xTB 的能量模型其实很好（MAE 3.30、r 0.966），比 GFN2（15.0）好一个数量级。**
   定向削减了 GFN2 的系统性过度放热。先前"未闭合差距、残差 35 kcal"是
   DFT 不收敛参考的假象。残留 bias −2.1（轻度残余过度放热），在 DFT-吸热/EWG 尾部
   放大到 ~−3.5，但已不是数量级问题。

3. **DFT 在 EWG/大分子硬例上反复不收敛，是拿到干净参考的真正瓶颈。**
   但注意：**把 g-xTB 几何喂给 DFT-SP，2191/2192 收敛了** —— 即"g-xTB opt → DFT-SP"
   这条路本身就基本解决了收敛问题（DFT-opt 直接做这些例子 0/13）。

## 3. 建议（按性价比排序）

- **A. 别再用更好的优化器修几何（g-xTB-opt / DFT-opt），把算力投到构象搜索。**
  真正移动数字的杠杆是采样够多构象 + 选到真正最低点（funnel_v3，CREST 作交叉验证）。
  对柔性大的苯偶姻产物加密构象，比换优化器更值。

- **B.【最高价值且最便宜】把 g-xTB 当作生产用的能量方法，跑在 funnel_v3 几何上。**
  组合 "funnel_v3 构象搜索(GFN-FF/GFN2) → g-xTB 单点能量" 预计把 ΔG 的 MAE 从 15 压到
  ~3，纯半经验成本、零 DFT。即用 g-xTB 替换 Δ-learning 基线里的 "xTB"，留给 ML
  的修正量大幅变小。**值得 A/B：在 g-xTB 基线上重训 Δ-模型 vs 现 GFN2 基线。**
  注意口径：g-xTB 溶剂化不稳，需固定一致的溶剂模型（现 screen 是 DMSO），−2.1 bias 可能
  部分来自气相 vs DMSO。

- **C. 残余 EWG 吸热尾部（−3.5 bias + 最差 ±15–38 kcal 离群）按 [[no-dG-extreme-filtering]]
  路由到 DFT，不要删。** 触发条件：(i) 超价 S/EWG 基序；(ii) g-xTB 与 GFN2 强烈分歧。

- **D. 修 DFT 收敛**（更稳 SCF/SOSCF/level-shift、用 g-xTB 密度做初猜、或 PBE0/r2SCAN 对照），
  才能定量 EWG 尾部还差多少。**已验证的捷径：先 g-xTB opt 得几何，再 DFT-SP——收敛率
  从 0 升到 ~100%。**

- **E. H-bond 几何半边问题**（[[hbond-not-product-error-driver]] 仍开放）现可**降级**：
  既然 g-xTB 几何不降低 DFT 能量、同构象 GFN2 几何更优，没有证据表明 DFT 弛豫会系统性
  重构改变 ΔG 的 H-bond。低优先。
