# screen_v6 建模汇总 — 半经验 ΔG 的可仿真性天花板 (2026-06-29)

整合 screen_v6（第一次全库，GFN2-xTB）上做过的所有建模/预测工作。此前结果散在
`findings_2026-06-17_screen_v6_linear_model.md` + `findings_2026-06-17_screen_v6_analysis.md`
两个文档和 `data/raw/screen_v6/analysis/` 的图里，这里给一个统一汇总。

---

## 0. 前提：这些模型在预测什么（务必先读）

**screen_v6 上所有模型的预测目标是 `dG_xtb_kcal` 本身** —— GFN2-xTB `--ohess`
(ALPB-DMSO) 算出的苯偶姻自缩合 ΔG（`2 RCHO → R-CO-CH(OH)-R`）。所以这一批是
**「半经验 ΔG 的代理/仿真模型」**：用描述符或 2D 结构去拟合 xTB 自己算出来的 ΔG，
回答「不跑 xTB 能多准复现 xTB ΔG」。

⚠️ 它**不是** Δ-learning (DFT−g-xTB) 修正模型 —— 那是 homo_v6 / g-xTB 阶段的工作
（gine_hybrid 2.13、dual_qm 2.02 等，见 `REPORT_MASTER_gxtb_dft_correction_20260625.md`），
目标和数据完全不同，不要混。screen_v6 建模没有用来「填预测」（全库 xTB ΔG 已直接算出），
它的价值是**确立表征瓶颈**，论证必须引入 3D/QM 信息 + DFT 标签 → 直接导向第二次全库 (g-xTB)。

- 数据：220,859 分子，xTB 优化成功 99.94%，物理窗 [−40,20] 内 220,138；|ΔG|>40 失败 239 (0.11%)
- 特征：24 个 xTB/morféus 电子+空间描述符；后补 7 个 Multiwfn ADCH/QTAIM → 31 个
- 数据质量、描述符↔ΔG 相关性：见 `findings_2026-06-17_screen_v6_analysis.md`

## 1. 模型阶梯（随机划分，held-out）

| 模型 | 脚本 | aromatic R² | aromatic MAE | all R² | all MAE |
|---|---|---|---|---|---|
| 线性 OLS (24 feat) | `linear_model.py` | 0.55 | 2.35 | 0.44 | 2.63 |
| MLP (128,64) | `nn_model.py` | 0.66 | 2.02 | 0.60 | 2.21 |
| MLP 调参 (256,128) | `gbt_mlp_tune.py` | 0.65 | 2.05 | 0.57 | 2.31 |
| **GBT (24 feat)** | `gbt_mlp_tune.py` | **0.66** | **2.01** | **0.585** | **2.27** |
| GBT +ADCH/QTAIM (31) | `mwf_feature_test.py` | 0.64 (+0.014) | 2.06 | 0.597 (+0.009) | 2.21 |

- **线性是「单描述符故事」**：Mulliken q(O) 单特征就 R²=0.50，其余 23 个只再 +0.05 → 强非线性。
- **非线性天花板 ≈ 0.66 (aromatic) / 0.58 (all)**：MLP 加深加宽、GBT 都顶到这，**容量不是瓶颈**。
- **ADCH/QTAIM 只买 ~1% 方差** (+0.01)，值得留但非关键。剩 ~35% 不是这套标量描述符的函数。
- **分体系建模更优**：aromatic（电子主导）0.66 > pooled > aliphatic（空间主导，最差）。

## 2. 诚实泛化检验：scaffold split

`scaffold_split_eval.py`，Bemis–Murcko 整骨架不泄漏，full aromatic n≈146,741：

| 划分 | test R² | MAE |
|---|---|---|
| random-molecule (泄漏参照) | 0.690 | 1.93 |
| scaffold-random (代表性新骨架) | 0.680 | 2.13 |
| scaffold-rare (最难，罕见骨架) | 0.542 | 2.58 |

随机划分只乐观 −0.01 → 学的是物理（xTB 电子/空间），迁移性好；真正新奇骨架掉到 0.54，
是特征空间外推，需谨慎。

## 3. GINE GNN — 2D 图能否突破标量天花板？（不能）

`gnn_dG.py`（GINE，A100，scaffold-split，aromatic）。06-18 首跑因 nequip 环境缺 rdkit
崩溃，06-20 用 `envs/gnn` 重跑成功。同一 scaffold 划分对照 GBT：

| 模型 | scaffold-split TEST R² | RMSE | MAE |
|---|---|---|---|
| GINE GNN (纯 2D 结构) | 0.611 | 3.10 | 2.18 |
| **GBT (31 描述符)** | **0.631** | 3.02 | 2.14 |

（数字读自 fig22 / fig23 标题；成功 run 的 stdout 未留存，logs_gnn 里只有 06-18 的崩溃日志。）

**结论：纯 2D-图 GNN 没能超过标量描述符天花板，scaffold 划分下反而略低于 GBT。**
瓶颈是**信息量不是模型容量**：2D 图看不到驱动 ΔG 的 3D/电子态。突破靠更丰富表征
（3D / QM 描述符在 readout 注入）—— 即后来 homo_v6 的 gine_hybrid(图+34QM)→2.13、
dual_qm(+56QM)→2.02。

### GINE GNN 是什么

- **GINE** = Graph Isomorphism Network **with Edge features**。GIN 是表达力最强的一类 MPNN
  （逼近 Weisfeiler–Lehman 上界）；GINE 让**键特征也进消息传递**，对分子图关键。
- **分子=图**：原子=节点，键=边。
  - 节点特征 (`atom_feat`)：元素 one-hot、连接度、形式电荷、杂化、氢数、芳香、在环。
  - 边特征 (`bond_feat`)：键级 one-hot(单/双/三/芳香) + 共轭 + 在环。
- **单层更新**：`h_i' = MLP((1+ε)·h_i + Σ_j ReLU(h_j + e_ij))`。邻居消息加上边特征 e_ij（=「E」），
  **求和聚合**（保留度数/计数信息，GIN 表达力来源），外层 MLP 保单射。
- **本仓库配置** (`gnn_dG.py:94-109`)：h=128、4 层 GINEConv（内部 Linear→ReLU→Linear）、
  节点/边各先线性投影到 128 维、残差 `x+=ReLU(BN(conv))`、读出 = mean-pool ⊕ add-pool 拼接
  → Linear→ReLU→Dropout(0.1)→Linear。训练 AdamW(lr1e-3,wd1e-5)、target z-score、
  ReduceLROnPlateau、按 val R² 早停(patience15/最多120ep)。
- **为何没赢**：只吃 2D 拓扑，看不到 3D 几何与 xTB 电子态（q(O)/WBO/LUMO/ω），所以输给吃 31 个
  QM 标量的 GBT。

## 4. 一句话总结

screen_v6 建模本质是一次**「半经验 ΔG 可仿真性」天花板研究**：线性 0.55 → 非线性/GNN 都顶在
~0.66 (aromatic)，GNN 不胜 GBT。确立了**表征瓶颈**，论证必须上 3D/QM + DFT 标签 →
直接导向第二次全库 (g-xTB) + Δ-learning 路线。
