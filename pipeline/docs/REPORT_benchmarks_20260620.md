# Benchmark 后续分析 — g-xTB / DFT-opt / 构象方法 (2026-06-20)

四个补充 benchmark(均 COMPLETED 17:39–17:40)的结果分析。核心结论:**g-xTB 方向正确但未闭合差距,而三个 benchmark 都被 DFT 不收敛严重拖累。**

来源:`gxtb_test/gxtb_compare_results.csv`、`gxtb_test/geom_out/mol_*.json`、
`dft_sp_r2scan3c/dftopt_bench/`、`dft_sp_r2scan3c/conf_method_compare/`。

---

## 1. g-xTB vs GFN2 vs DFT(反应能,13 个硬例)

| 类别 | n | GFN2 ΔE | g-xTB ΔE | g-xTB 修正量 | DFT ΔG(参考) |
|---|---|---|---|---|---|
| sulfonyl | 6 | −39.7 | −20.0 | **+19.7** | **+16.5** |
| nitro | 3 | −34.7 | −18.2 | **+16.5** | **+13.2** |
| benign | 4 | −32.7 | −23.8 | +8.9 | −9.1 |

**判读:**
- **g-xTB 系统性削减 GFN2 的过度放热,且对 EWG 削得更多**(sulfonyl/nitro +17~20 kcal vs benign +9)——正好针对 [[nonewg-outlier-drivers]] / EWG 这个 xTB 最大失效类,方向完全正确。
- **但没闭合**:EWG 的 DFT ΔG **9/9 为吸热(+13~+16)**,而 g-xTB **9/9 仍放热**(−18~−20),残差均值 **~35 kcal**。g-xTB 把符号对的方向推了一半,却还在错的符号一侧。
- 注意口径:g-xTB/GFN2 是**电子 ΔE**,DFT 是**含热含溶剂的 ΔG**,~35 kcal 残差里有一部分是热力学/溶剂项;但**符号矛盾(放热 vs 吸热)是稳健的**。

图:`gxtb_test/fig_gxtb_vs_gfn2_vs_dft_byclass_20260620.png`

---

## 2. g-xTB 几何(10 分子,键长)

- **DFT 几何全程未收敛**:`d_dft == d_gfn2` 对 **95/95 键**精确成立 → JSON 里的 "d_dft" 是 GFN2 回退值,**不是真 DFT 几何**(`dft_conv=False`)。
- 因此只能比 **g-xTB vs GFN2** 键长差异(无真 DFT 基准、无法判优劣):
  - 总体 g-xTB 偏离 GFN2 **8.2 mÅ**;**超价硫键改动最大**:S-Cl **25.7**、S-C **16.5** mÅ;S-O 6.7、C=O 3.0 mÅ。
  - 即 g-xTB 主要在**超价 S 键**上重排几何——与它修正 EWG 能量一致,但缺真 DFT 几何无法验证是否更准。

---

## 3. DFT-opt benchmark(2 分子)

- idx 3、38 **两个 DFT 优化都没收敛**(`ald_conv=bz_conv=False`,所有 geom-term/能量列空)。
- 仅有 hbond:idx3 bz_hbond 2.10 Å、idx38 3.66 Å,xtb 与 dft 几何下 hbond_delta=0(因 DFT 几何未生成)。
- **未能回答"DFT-opt 是否改变 ΔG 的几何项"**——这正是 [[hbond-not-product-error-driver]] 里仍开放的几何半边,**仍被 DFT 不收敛挡住**。

## 4. 构象方法对比(1 分子)

- idx 38:**funnelv3 比 legacy 的 ΔG_xTB +9.6 kcal**(−23.5 → −13.9),`broken=False` 两法都未坏拓扑。DFT 侧两列均空(未收敛)。单分子,仅指示性。

---

## 总结与下一步

1. **贯穿性问题:这些硬 EWG/大分子上 r2SCAN-3c(及其优化)反复不收敛** —— dftopt 2/2、g-xTB 能量 0/13、g-xTB 几何 0/10。**所有打算以 DFT 为基准的 benchmark 都被这一点废掉了**。这与 [[delta-mae-noise-floor]]、主全量 SP 里 ~2% 静默缺失同源。
2. **g-xTB 值得继续**:它定向削减 EWG 过度放热(对症),但需要**先让 DFT 参考收敛**(更稳的 SCF/初猜、或换泛函/求解器)才能定量"还差多少"。
3. 建议:对这批 EWG 硬例做一轮**收敛性增强的 DFT 单点/优化**(tighter SCF、SOSCF/level-shift、或 r2SCAN-3c→PBE0 对照),拿到真 DFT 基准后再复跑这四个 benchmark。
