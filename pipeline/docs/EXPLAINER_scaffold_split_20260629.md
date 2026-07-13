# Scaffold split — honest generalization to novel cores
# 骨架划分 —— 对新骨架的诚实泛化检验

_Companion figure 配图: `figs/scaffold_split_explainer_20260629.png`
(script 脚本 `pipeline/plot_scaffold_split_explainer.py`).
Implementation 实现: `pipeline/analysis/exp_scaffold_split.py`.
Results source 数据来源: `pipeline/docs/REPORT_screen_v6_models_20260629.md` §2–3._

---

## 1. What is a Bemis–Murcko scaffold? / 什么是 Bemis–Murcko 骨架？

**EN.** A Bemis–Murcko scaffold is the molecule reduced to its **ring systems plus the
linker atoms that connect them**, with all terminal side chains removed. Two molecules
that differ only in substituents (e.g. `4-Cl-benzaldehyde` vs `4-NO2-benzaldehyde`)
collapse to the **same** scaffold (the benzaldehyde core). In code it is one call:
`rdkit.Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmiles(...)`.

**中文.** Bemis–Murcko 骨架是把分子**剥掉所有末端侧链、只保留环系 + 连接它们的连接子**
后得到的核心结构。只在取代基上不同的两个分子（如 `4-Cl-苯甲醛` 与 `4-NO2-苯甲醛`）会归为
**同一个**骨架（苯甲醛核）。代码里就一行：
`rdkit.Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmiles(...)`。

---

## 2. Random split vs scaffold split / 随机划分 vs 骨架划分

**EN.**
- **Random split** shuffles all rows and takes ~10% as test. Because substituent
  analogues share a core, the **same scaffold appears in both train and test** — the
  model can "recognize" a test molecule from a near-twin it already saw. This **leaks**
  information and **over-states** real-world accuracy.
- **Scaffold split** first groups molecules by scaffold, then assigns **whole scaffold
  groups** to test (~10%). A test scaffold is therefore **never seen during training**,
  so the score measures genuine **extrapolation to new chemical cores**.

**中文.**
- **随机划分**把所有行打乱后取 ~10% 当测试。由于取代基类似物共享同一个核，
  **同一骨架会同时出现在训练集和测试集**——模型可以靠"见过的近亲双胞胎"来"认出"测试分子。
  这造成**信息泄漏**，会**高估**真实精度。
- **骨架划分**先按骨架把分子分组，再把**整组骨架**整体划入测试集（~10%）。
  这样一个测试骨架在训练中**从未出现过**，得分衡量的才是对**全新化学核心的外推**能力。

The figure's left panel shows leakage (test colors A/C/E/B all reappear in train → ✗);
the right panel shows whole cores E/F held out (train only has A–D → ✓).
配图左侧展示泄漏（测试颜色 A/C/E/B 在训练里都重现 → ✗）；右侧把整核 E/F 留作测试
（训练只有 A–D → ✓）。

---

## 3. How it is implemented / 具体实现

From `exp_scaffold_split.py` (key lines):
```python
df["scaf"] = [MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s))
              for s in df.smiles]
groups = defaultdict(list)
[groups[s].append(k) for k, s in enumerate(df.scaf)]   # rows per scaffold

order = list(groups); rng.shuffle(order); te = []
for s in order:                       # add WHOLE scaffolds until test ≈ 10%
    if len(te) < 0.1*len(df): te += groups[s]
    else: break
tr = [k for k in range(len(df)) if k not in set(te)]    # rest = train
```
The same 72-feature MLP+XGB ensemble is then trained on `tr` and scored on `te`, once
for the random split and once for the scaffold split, so the only thing that changes is
**how the test set is carved**. 之后用同一个 72 特征的 MLP+XGB 集成分别在随机划分和骨架划分
下训练/评估——唯一变化的只是**测试集如何切分**。

---

## 4. Results / 结果 (aromatic n ≈ 146,741)

| split / 划分 | test MAE (kcal/mol) | test R² |
|---|---|---|
| random / 随机 | **2.13** | 0.680 |
| scaffold-random / 代表性新骨架 | **2.18** | 0.611 |
| scaffold-rare (hardest) / 最难的罕见骨架 | **2.58** | 0.542 |

**EN.** Moving from random to scaffold split costs only ~0.05 kcal MAE on typical new
cores (2.13 → 2.18) but ~0.45 kcal on rare/unusual cores (→ 2.58). The model
generalizes **honestly and gracefully** — there is no catastrophic collapse, but the
**rare-scaffold gap is the real headroom**.

**中文.** 从随机划分换到骨架划分，对常见新骨架只损失约 0.05 kcal MAE（2.13 → 2.18），
但对罕见/少见骨架损失约 0.45 kcal（→ 2.58）。模型的泛化是**诚实且平稳的**——没有灾难性崩溃，
而**罕见骨架那道差距才是真正的提升空间**。

---

## 5. Why we use it / 为什么要用它

**EN.** A benzoin ΔG model is only useful if it predicts aldehydes the chemist has
**not** made yet. Random-split scores flatter the model by testing on near-duplicates;
scaffold split is the honest number we report and the split under which we judged the
2D-graph GNN (it does **not** beat GBT even here — see
`figs/screen_v6_gnn_vs_gbt_20260629.png`).

**中文.** 一个苯偶姻 ΔG 模型只有在能预测化学家**还没合成过**的醛时才有用。随机划分用近重复样本
测试，会美化模型；骨架划分才是我们上报的诚实数字，也是我们评判 2D 图 GNN 的划分方式
（即便在这个划分下它**也没**赢过 GBT——见 `figs/screen_v6_gnn_vs_gbt_20260629.png`）。
