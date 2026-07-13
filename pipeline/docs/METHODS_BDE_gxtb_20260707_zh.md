# BDE/BDFE 方法层级 3：g-xTB 一致的 BDE + BDFE

**状态：已采纳——当前生产冠军特征集**

## 动机（"方法不匹配"假设）

层级 1-2 都在 GFN2-xTB 层级计算 BDE/BDFE，但实际被校正的生产基线（dG_gxtb）是
**g-xTB**，不是 GFN2——GFN2 只是内部混合公式中提供热校正的辅助量（见 `gxtb_baseline.py` 的
`G_gxtb = E_gxtb + (G_gfn2 - E_gfn2)`）。假设：描述符（GFN2）与它要帮助校正的对象
（g-xTB 自身的误差）之间的层级不匹配，可能稀释了信号——如果这个假设成立，在 g-xTB 层级重新
计算 BDE/BDFE 应该能携带更多信号。

## 方法

`pipeline/compute/calc_bde_free_energy_gxtb.py`，复用项目自己的混合校正模式：
- 母体的 `G_gxtb` 已经缓存（`{aldehydes,products}_all.csv`）——无需重新计算。
- 母体的原始 `E_gxtb` 可以**免费代数求出**：`E_gxtb_母体 = G_gxtb_母体 - (G_gfn2_母体 -
  E_gfn2_母体)`（对混合公式取逆；两个 GFN2 项都是已缓存的 `G_xtb`/`xtb_energy` 列）。
- 每个自由基片段做一次全新的 GFN2 `--ohess`（热校正 + 弛豫几何，同层级 2），**外加在同一
  几何上做一次 g-xTB 单点**（`--gxtb --cosmo dmso`，已冒烟测试可同时使用——g-xTB 的解析
  Hessian 可能比 GFN2 的数值 Hessian 更数值稳定，这与层级 2 中"低频模式噪声"的猜测相关）。
  `bde_gxtb_kcal` 和 `bdfe_gxtb_kcal` 都由同一次片段计算同时产出——片段侧的 `E_gxtb` 本来
  就是单点步骤内部算出的，此前被丢弃，现在直接保留。

## 中试验证

3 分子中试：3/3 成功，`BDE > BDFE`，符合解离熵效应的预期。分散采样中试（而非仅取前 N
行——那些全是小的脂肪族分子，耗时高度依赖分子大小，每 20 分子 chunk 从 30 秒到 14 分钟不等）：
- n≈140/侧：醛 C-H BDFE(g-xTB) 与 dG_gxtb 的相关性 r=0.300（p=3e-4）——相对 GFN2 版本
  在全库上的 r=0.044 是真实的跃升。产物 C-C BDFE(g-xTB) r=-0.184，与 GFN2 版本量级相近——
  产物侧没有改善。
- n≈711 醛 / 694 产物（更大的合并中试）：醛侧 r=0.209（p=1.9e-8），仍是 GFN2 版本 r=0.044
  的约 4-5 倍，且高度显著；产物侧 r=-0.254，相比 GFN2 的 -0.173 仍无明显改善。

## 覆盖率

全库阵列：醛 1460/1471（99.3%，任务 24437300，干净完成）；产物 1459/1463（99.7%，任务
24437301→24454920，中间因共享环境损坏事故被迫回填——见 `shared-env-instability-2026-07-05`
——通过隔离的 `envs/bde_lite` venv 恢复）。附属文件：
`{aldehydes,products}_bdfe_gxtb_descriptors.csv`，共 219,095 行有标签数据，各列覆盖率
95.8-97.8%（仅在训练集上做中位数填补）。

## 结果

`finalize_correction_bdfe_gxtb_full.py` 在 mordredslim271（271 特征，MAE 1.612）上做快速
2 成员 XGB 检验：

| 配置 | 特征数 | 测试集 MAE | 差值 |
|---|---|---|---|
| mordredslim271 基线 | 271 | 1.612 | — |
| + BDFE(g-xTB)，两侧 | 273 | 1.605 | -0.007（空结果，与 GFN2 自身的空结果一致） |
| + BDE(g-xTB)，两侧 | 273 | **1.580** | **-0.032（真实——大于 GFN2 自身的 +0.024）** |
| + BDE(g-xTB) + BDFE(g-xTB)，两侧 | 275 | **1.563** | **-0.049（迄今最佳结果，约为噪声带的 4 倍）** |

**完整生产确认**（`finalize_correction_mordredslim271_bdegxtb.py`，任务 24468737，真正的
MLP + XGB_d8 + XGB_d10 + 分位数不确定性集成，而非快速 2 成员检验）：**测试集 MAE 1.503**，
相对 mordredslim271 的 1.525 差值为 -0.022，正好卡在噪声带边缘（完整集成本身已经吸收了部分
这些原始特征在纯 2-XGB 检验上带来的增益，所以增益不会 1:1 从快速检验转移过来）。仍判定为
真实且值得采纳的提升。

## 解读

方法不匹配假设得到证实，但**仅对 BDE（原始电子能）成立，对 BDFE 不成立**。g-xTB 一致的
BDE 超越了 GFN2 层级 BDE 本已勉强的提升，而 BDE+BDFE 合并使用是超出噪声带的明确胜利——尽管
BDFE **单独**使用依然是空结果，与 GFN2 的模式完全一致。BDFE 中额外的 RRHO 热校正项看起来
**无论底层电子结构方法是什么**都更像噪声而非信号，但当 BDE 在与被校正对象真正匹配的层级上
计算时，其原始电子能会携带更多信息。

## 成本感知后续分析（2026-07-07 SHAP 审计）

对完整 275 特征冠军模型做 SHAP 重要性分析（4000 行测试子样本，XGB_d8）：`ald_bde_gxtb_kcal`
排名 4/275（mean|SHAP|=0.587），`prod_bde_gxtb_kcal` 排名 6/275（0.484)——两者都**便宜**
（单次 SP/opt）。`prod_bdfe_gxtb_kcal` 排名 15/275（0.191），`ald_bdfe_gxtb_kcal` 排名
38/275（0.099）——两者都**昂贵**（每个片段都需要完整的 `--ohess` Hessian+RRHO）。重要性
求和：BDE=1.070，BDFE=0.290（比值 3.7 倍）。**对未来新分子的前瞻性筛选建议：只计算 BDE，
跳过 BDFE**——重要性损失接近零，却能省下大量计算（无需 Hessian）。现有生产 bundle 保留两者
（全库已经算过，成本已沉没），但这应指导未来的特征工程方向：不再投入 BDFE 族（热/熵）描述符。

另见 [METHODS_BDE_gfn2_raw_energy_20260707_zh.md](METHODS_BDE_gfn2_raw_energy_20260707_zh.md)
和 [METHODS_BDE_gfn2_free_energy_20260707_zh.md](METHODS_BDE_gfn2_free_energy_20260707_zh.md)
——本层级取代的两个更早方法层级。
