# 数据划分协议 — screen_v6 vs homo_v6

**日期:** 2026-06-30 · 两个数据集用**不同的划分哲学**,因为建模目标不同。

## 总览

| | **screen_v6** | **homo_v6** |
|---|---|---|
| 数据 | 220,859 个醛的筛选库(芳香子集 n≈146,741) | homo-苯偶姻产物 ~220k 对 |
| 任务 | 2D-SMILES/标量描述符 → **xTB ΔG** 代理模型 | **g-xTB→DFT 修正**(预测 `DFT_r2SCAN-3c − g-xTB` 的 delta) |
| 标签 | xTB ΔG(无 DFT) | DFT r2SCAN-3c(CPCM-DMSO) |
| **主划分** | **Bemis–Murcko scaffold split**(整骨架不泄漏,~10% test) | **随机 70/20/10 hold-out**(seed 42) |
| 辅助划分 | scaffold-rare(最难档)、random(泄漏参照) | scaffold-disjoint(诚实外推)、RepeatedKFold(快速 CV) |
| 报告口径 | scaffold split | 随机 7:2:1 |

## screen_v6 — 以 scaffold 划分为主

训练/评测脚本 `scaffold_split_eval.py`;结果见 `REPORT_screen_v6_models_20260629.md` §2。
Bemis–Murcko **整骨架不泄漏**(同一骨架的所有分子只进 train 或只进 test),test ~10%:

| 划分 | test R² | MAE (kcal/mol) | 含义 |
|---|---|---|---|
| random-molecule | 0.690 | 1.93 | **泄漏参照**(同骨架近邻可同时入训练+测试 → 偏乐观) |
| scaffold-random | 0.680 | 2.13 | 代表性**新骨架** |
| scaffold-rare | 0.542 | 2.58 | **最难**:罕见骨架全划到 test |

为什么用 scaffold:22 万库骨架高度重复,随机划分会泄漏 → 高估泛化;screen_v6 关心**对新骨架的外推**。GINE-GNN 与 GBT 在**同一 scaffold 划分**下对比(GNN 2.18 ≥ GBT 2.14,2D 图未超过标量描述符天花板)。

## homo_v6 — 以随机 70/20/10 为主

生产脚本 `train_cb_721.py` / `finalize_correction.py` / `scope_arch_compare.py`,固定 seed=42:

```python
tr, tmp = train_test_split(idx, test_size=0.30, random_state=42)  # 70% train
va, te  = train_test_split(tmp, test_size=1/3,  random_state=42)  # 20% val / 10% test
```

协议:**VAL(20%)调超参 → TRAIN+VAL 重拟合 → 未碰过的 TEST(10%)报最终指标**(holdout,非 K-fold)。
最终模型 = MLP+XGB 集成,uncertainty = 成员 std / 分位数 PI 宽度,最不确定的 15% 标 `route_to_dft`。

**辅助划分**:
- `exp_scaffold_split.py` — Murcko 整骨架,~10% 骨架进 test,作**诚实外推**检验,报告 random→scaffold 的泛化 gap;
- `delta_core.cv_evaluate` — RepeatedKFold(n_splits×n_repeats,固定 seed),用于快速模型对比。

## 一致原则

两者都支持"整骨架不泄漏"的诚实检验;区别在 **headline 口径**:screen_v6 报 scaffold split(防大库骨架泄漏),homo_v6 报随机 7:2:1(标准产线 holdout,符合用户既定协议 [[data-split-721]])。
