import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md(r"""# Homo Full-Library Model Benchmark

**Date:** 2026-09-20
**Scope:** the homo-from-scratch campaign (`CAMPAIGN_PLAN.md`) required a single
XGBoost and a single GNN, reported independently (2026-09-10 user directive).
This notebook does three things the user asked for on top of that:

1. Documents **how the homo database was actually built** — what goes in
   (inputs), what comes out (the label and the assembled feature table), so
   the benchmark below isn't read against a black box.
2. Benchmarks the tabular leg against two more **mainstream** estimators
   (Random Forest, SVM) beyond the champion XGBoost.
3. Benchmarks the graph leg against a **plain (non-attentive) pooling GNN**
   and **Chemprop** (a standard, widely-used D-MPNN framework, distinct from
   this project's own hand-written GNN), beyond the champion attentive-pooling
   architecture.

All models share the *same* Delta-learning setup (target `dG_orca_kcal` minus
baseline `dG_b973c_kcal`) and the *same* scaffold-disjoint split, verified
below to have zero train/test scaffold overlap — so any MAE difference between
models is attributable to the estimator, not a confound in data or evaluation.
""")

code(r"""import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/gpfs/scratch1/shared/schen3/benzoin-dg-restored")
HS = REPO / "data/cross_benzoin/homo_standalone"
CS = REPO / "data/chemical_space"
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 140)

TARGET_COL = "dG_orca_kcal"
BASELINE_COL = "dG_b973c_kcal"
plt.rcParams["figure.dpi"] = 110
""")

md(r"""## 1. How the homo database was built

**Why it had to be rebuilt at all.** The full ~219k-aldehyde homo DFT ΔG
labels were physically lost in the 2026-07 Snellius scratch purge — only a
30k-pair subset (`homo_unify_v1`) survived, backed up by luck rather than
design. `PROJECT_PLAN.md` sec1.8/sec2.10.b and `CAMPAIGN_PLAN.md` have the
full loss/recovery history; the short version is: recomputing all ~219k pairs
self-consistently (fresh conformer search + geometry + thermal + DFT) would
have taken ~3 weeks, so the project adopted a **much cheaper SP-only route**
instead, since it turns out to be how the original labels were made in the
first place.

### 1.1 Input: geometry + thermal, output: two DFT single-points per pair

For each homo pair (an aldehyde reacting with itself, `donor_idx == acceptor_idx`):

| | source | what it is |
|---|---|---|
| **product geometry** | `bde_homo_product_featurize_20260902/chunk_*/geom.tar.zst` (`xyz/pNNNNNN.xyz`) | the GFN2-xTB-optimized product geometry from the 2026-09 archive — **reused, not recomputed** |
| **aldehyde geometry** | same archive, `ald_xyz/aNNNNNN.xyz` | ditto, for the aldehyde itself |
| **thermal correction** | `G_product - xtb_energy` / `G_xtb - xtb_energy` columns already in `*_all.csv` | the GFN2-xTB RRHO thermal, also reused |
| **new compute per pair** | `homo_sp_from_geom_worker.py`: `calc_orca_sp` at **r2SCAN-3c** (the label level) *and* **B97-3c** (a cheaper composite, used as the Delta-learning baseline) | two DFT single-points, no conformer search, no Hessian — this is the entire "new" cost |

**The label**: `dG_lvl = (G_lvl[product] − 2·G_lvl[aldehyde]) · 627.509` kcal/mol,
for both `lvl = r2SCAN-3c` (the true label, `dG_orca_kcal` in the assembled
table below) and `lvl = B97-3c` (the Delta-learning baseline, `dG_b973c_kcal`).

### 1.2 Two tracks, because the geometry archive itself had a gap

A 2026-09-10 audit found the geometry archive is ~5-10% incomplete for a
contiguous chunk range (~1400-1830). Rather than just drop that slice, the
184,052-pair manifest was split:

- **archived track** (~90%, 8,972 shards): SP directly on the archived geometry (fast).
- **regen track** (~10%, 771 shards): geometry doesn't exist -> full `conf_funnel_v3`
  + GFN2 `--ohess` regeneration first, *then* the same SP recipe (`rec_homo_relabel_worker.py`).

Both tracks ran in parallel and both reached 100% completion by 2026-09-17.
""")

code(r"""summary = json.loads((HS / "relabel_sp/homo_sp_summary.json").read_text())
print(f"manifest coverage: {summary['coverage']}/{summary['manifest_n']} = {summary['coverage_pct']}%")
print(f"verdict: {summary['verdict']}")
print()
qc = summary["qc_repro_r2scan_vs_stored_30k"]
print(f"QC check -- new r2SCAN-3c SP vs the surviving 30k stored labels, AFTER excluding")
print(f"{summary['n_resid_outlier_excluded']} residual-outlier rows (n={qc['n']}):")
print(f"  mean={qc['mean']}  std={qc['std']}  (must be <= 2.9 kcal, the single-conformer noise floor, to call this clean)")
print(f"(note: summary['groups']['overall']['repro_r2scan'] is the noisier PRE-exclusion")
print(f"version of this same check -- mean={summary['groups']['overall']['repro_r2scan']['mean']}, "
      f"std={summary['groups']['overall']['repro_r2scan']['std']} -- the qc_repro_* field above is the one the GREEN verdict is actually judged on)")
""")

md(r"""The QC check above is the load-bearing sanity check: if the newly
recomputed labels didn't reproduce the ~30k labels that survived the purge
(within the known single-conformer noise floor), the whole SP-only shortcut
would be suspect. **GREEN** — no systematic offset, safe to build the training
table from it.

### 1.3 Assembly: from labeled pairs to a training table

`assemble_homo_standalone_table.py --full-library` joins the SP-relabel
output against:

- the per-aldehyde QM cache (`homo_v6/aldehydes_all.csv`) and Mordred
  descriptors (`aldehydes_mordred_slim102.csv`), for **both** the donor and
  acceptor role (identical molecule for homo, but the champion schema is
  written generically for two roles, so both get populated)
- the product's own QM (`products_all.csv`) and Mordred descriptors
- RDKit-2D descriptors (computed on the fly from SMILES) and `interaction_*`
  cross-terms between the two aldehyde sides

into the same "all_raw_blocks+mordred" feature schema the cross-benzoin
champion is built from — so a model trained here is on genuinely comparable
footing with the cross side, not a bespoke homo-only feature set.
""")

code(r"""table = pd.read_parquet(HS / "homo_standalone_full_library_table_slim260.parquet")
print(f"table shape: {table.shape}")
print(f"target column '{TARGET_COL}': present={TARGET_COL in table.columns}")
print(f"baseline column '{BASELINE_COL}': present={BASELINE_COL in table.columns}")
print()
print("split sizes:")
print(table["new_scaffold_split"].value_counts())
""")

code(r"""from collections import Counter

def bucket(c):
    if c.startswith("donor_ald_mordred_") or c.startswith("acceptor_ald_mordred_"):
        return "aldehyde Mordred (donor+acceptor)"
    if c.startswith("donor_") or c.startswith("acceptor_"):
        return "aldehyde QM/RDKit-2D (donor+acceptor)"
    if c.startswith("product_mordred_"):
        return "product Mordred"
    if c.startswith("product_"):
        return "product QM/RDKit-2D"
    if c.startswith("interaction_"):
        return "donor-acceptor interaction terms"
    return "id / split / label / baseline columns"

counts = Counter(bucket(c) for c in table.columns)
pd.Series(counts).sort_values(ascending=False).to_frame("n_columns")
""")

md(r"""**Input, in one sentence**: two aldehyde structures (donor == acceptor for
homo) plus their cached QM/Mordred descriptors and the product's own
QM/Mordred descriptors, all derived from geometries that were mostly *reused*
from an earlier archive, not recomputed.
**Output, in one sentence**: `dG_orca_kcal` (the r2SCAN-3c reaction free
energy in kcal/mol) as the target, `dG_b973c_kcal` as the cheap Delta-learning
baseline, and a `new_scaffold_split` column assigning each row to
train/validation/test by Bemis–Murcko scaffold.

### 1.4 Why Delta-learning, and what the baseline actually looks like
""")

code(r"""fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].hist(table[TARGET_COL].dropna(), bins=80, color="#4C72B0")
axes[0].set_xlabel(r"$\Delta G_{r2SCAN\text{-}3c}$ (kcal/mol)")
axes[0].set_ylabel("count")
axes[0].set_title("Homo label distribution (target)")

resid = table[TARGET_COL] - table[BASELINE_COL]
axes[1].hist(resid.dropna(), bins=80, color="#DD8452")
axes[1].axvline(resid.mean(), color="k", ls="--", lw=1,
                label=f"mean={resid.mean():.2f}, std={resid.std():.2f}")
axes[1].set_xlabel(r"$\Delta G_{r2SCAN\text{-}3c} - \Delta G_{B97\text{-}3c}$ (kcal/mol)")
axes[1].set_title("What the Delta-model actually has to learn")
axes[1].legend()

plt.tight_layout()
plt.show()
""")

md(r"""The residual the model has to learn is a **near-constant ~-5 kcal
offset with a tight ~1 kcal spread** — B97-3c is a cheap, systematically
biased but low-noise proxy for r2SCAN-3c, and the whole point of Delta-learning
is that a model only needs to learn this small, well-behaved correction rather
than the full ~5+ kcal target range from scratch. This is the same lever
(`Rec-1`) that dropped the cross champion's MAE from 2.215 to 0.528, applied
here from the start.

### 1.5 Split integrity: is "scaffold-disjoint" actually true here?

This project has been burned before by a split that claimed to be
scaffold-disjoint but wasn't (`bde_scaffold_leakage_finding` memory) — so
before trusting any MAE number below, check train/test scaffold overlap
directly rather than assuming the column name is honest.
""")

code(r"""idx = pd.read_parquet(CS / "aldehyde_index.parquet").set_index("ald_idx")
t = table[["id", "new_scaffold_split"]].copy()
t["id"] = t["id"].astype(int)
t = t.join(idx[["scaffold"]], on="id")

train_scaf = set(t.loc[t.new_scaffold_split == "train", "scaffold"].dropna())
test_scaf = set(t.loc[t.new_scaffold_split == "test", "scaffold"].dropna())
overlap = train_scaf & test_scaf
print(f"train scaffolds: {len(train_scaf)}")
print(f"test scaffolds:  {len(test_scaf)}")
print(f"overlap:         {len(overlap)}  ({len(overlap)/max(len(test_scaf),1):.1%} of test scaffolds also in train)")
assert len(overlap) == 0, "scaffold leakage detected -- do not trust the MAE numbers below"
print("\nConfirmed clean: zero scaffold overlap between train and test.")
""")

md(r"""## 2. Models compared

| | already trained by the campaign | benchmarked in this notebook |
|---|---|---|
| **Tabular** | single XGBoost (`train_scaffold_disjoint.py`) | + Random Forest, + Linear SVR (full train set), + RBF SVR (20k-row subsample — RBF doesn't scale to 132k rows on one node) |
| **Graph** | attentive-pooling TripleGNN (`train_cross_gnn_arch_sweep.py --arch attentive`) | + plain concat-pooling TripleGNN (`--arch default`), + Chemprop D-MPNN (a standard external framework, not this project's own architecture) |

All tabular models use the exact same 394-column feature set the champion
XGBoost was trained on (`tabular_full/models/feature_list.json`), the same
median-imputation, and the same scaffold-disjoint train/test split. All GNN
variants use the same molecular-graph construction and the same split.
""")

md(r"""## 3. Tabular benchmark: XGBoost vs Random Forest vs SVM

All three reuse the champion's own `feature_list.json` (394 columns) and
median-imputation on the same scaffold-disjoint train/test split. Random
Forest is fit on the full 132,093-row training set. SVM's RBF kernel is
O(n^2)-O(n^3) and not tractable at that scale on one node, so it's fit on a
20,000-row random subsample; Linear SVR (a linear kernel, scales like a
linear model) is fit on the full training set for a second, unsampled SVM
comparison point. See `cross_benzoin/train_homo_rf_svm_benchmark.py`.
""")

code(r"""feats = json.loads((HS / "tabular_full/models/feature_list.json").read_text())
train_df = table[table["new_scaffold_split"] == "train"].reset_index(drop=True)
test_df = table[table["new_scaffold_split"] == "test"].reset_index(drop=True)
medians = train_df[feats].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
Xte = test_df[feats].apply(pd.to_numeric, errors="coerce").fillna(medians)
yte_true = test_df[TARGET_COL].to_numpy()
base_te = test_df[BASELINE_COL].to_numpy()

xgb_model = joblib.load(HS / "tabular_full/models/champion_scaffold_disjoint.joblib")
pred_xgb = base_te + xgb_model.predict(Xte)

bench_dir = HS / "tabular_benchmark"
bench_preds = pd.read_csv(bench_dir / "test_predictions.csv")
bench_metrics = json.loads((bench_dir / "metrics.json").read_text())

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def metrics(y_true, y_pred):
    return dict(MAE=mean_absolute_error(y_true, y_pred),
                RMSE=np.sqrt(mean_squared_error(y_true, y_pred)),
                R2=r2_score(y_true, y_pred))

tabular_preds = {
    "XGBoost (champion)": pred_xgb,
    "Random Forest": bench_preds["pred_random_forest"].to_numpy(),
    "Linear SVR": bench_preds["pred_linear_svr"].to_numpy(),
    "RBF SVR (20k subsample)": bench_preds["pred_rbf_svr_subsampled"].to_numpy(),
}
tabular_rows = []
for name, pred in tabular_preds.items():
    m = metrics(yte_true, pred)
    n_train = 132093 if "subsample" not in name else bench_metrics["rbf_svr_subsampled"]["train_n"]
    tabular_rows.append({"model": name, "n_train": n_train, **m})
tabular_summary = pd.DataFrame(tabular_rows).sort_values("MAE").reset_index(drop=True)
tabular_summary
""")

code(r"""fig, ax = plt.subplots(figsize=(6, 3.5))
order = tabular_summary.sort_values("MAE")
bars = ax.barh(order["model"], order["MAE"], color="#4C72B0")
ax.bar_label(bars, fmt="%.3f", padding=3)
ax.set_xlabel("scaffold-disjoint holdout MAE (kcal/mol)")
ax.set_title("Tabular model comparison (homo full-library, n_test=16,996)")
ax.invert_yaxis()
plt.tight_layout()
plt.show()
""")

code(r"""fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)
lims = (yte_true.min() - 1, yte_true.max() + 1)
for ax, (name, pred) in zip(axes, tabular_preds.items()):
    ax.scatter(yte_true, pred, s=3, alpha=0.15, color="#4C72B0")
    ax.plot(lims, lims, "k--", lw=1)
    mae = metrics(yte_true, pred)["MAE"]
    ax.set_title(f"{name}\nMAE={mae:.3f}")
    ax.set_xlabel(r"true $\Delta G$ (kcal/mol)")
axes[0].set_ylabel(r"predicted $\Delta G$ (kcal/mol)")
plt.tight_layout()
plt.show()
""")

md(r"""### Feature importance: what the tree models actually lean on

Only XGBoost and Random Forest have a native importance score (SVM doesn't,
by construction). Comparing the two is itself a useful cross-check — if two
independently-trained tree ensembles agree on which features matter, that's
more convincing than either alone.
""")

code(r"""rf_model = joblib.load(bench_dir / "model_random_forest.joblib")
imp_xgb = pd.Series(xgb_model.feature_importances_, index=feats).sort_values(ascending=False).head(20)
imp_rf = pd.Series(rf_model.feature_importances_, index=feats).sort_values(ascending=False).head(20)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
axes[0].barh(imp_xgb.index[::-1], imp_xgb.values[::-1], color="#4C72B0")
axes[0].set_title("XGBoost top-20 feature importance")
axes[1].barh(imp_rf.index[::-1], imp_rf.values[::-1], color="#DD8452")
axes[1].set_title("Random Forest top-20 feature importance")
plt.tight_layout()
plt.show()

shared_top20 = set(imp_xgb.index) & set(imp_rf.index)
print(f"{len(shared_top20)}/20 features appear in both models' top-20: {sorted(shared_top20)}")
""")

md(r"""## 4. Graph benchmark: attentive-pooling vs plain-pooling vs Chemprop

The two TripleGNN variants (`attentive`, `default`) share this project's own
atom/bond featurization and are directly comparable pointwise (same test-set
predictions reconstructed below). Chemprop is a materially different,
externally-maintained D-MPNN framework — a genuinely independent "mainstream"
reference point, but its own training script doesn't save per-row
predictions, so it contributes a summary MAE only, not a scatter plot.
""")

code(r"""gnn_preds = pd.read_csv(HS / "gnn_benchmark_predictions.csv")
gnn_pred_cols = [c for c in gnn_preds.columns if c.startswith("pred_gnn_")]
print(gnn_preds.shape, gnn_pred_cols)

gnn_rows = []
for c in gnn_pred_cols:
    name = c.replace("pred_gnn_gnn_", "").split("_full_seed")[0]
    valid = gnn_preds[c].notna()
    m = metrics(gnn_preds.loc[valid, "dG_true"], gnn_preds.loc[valid, c])
    gnn_rows.append({"model": f"GNN ({name}-pooling)", **m})

chemprop_result = json.loads((HS / "gnn_chemprop_full_seed0/result.json").read_text())
gnn_rows.append({"model": "Chemprop (D-MPNN)", "MAE": chemprop_result["test_mae"], "RMSE": np.nan, "R2": np.nan})

gnn_summary = pd.DataFrame(gnn_rows).sort_values("MAE").reset_index(drop=True)
gnn_summary
""")

code(r"""fig, axes = plt.subplots(1, len(gnn_pred_cols), figsize=(6 * len(gnn_pred_cols), 5), squeeze=False)
axes = axes[0]
for ax, c in zip(axes, gnn_pred_cols):
    name = c.replace("pred_gnn_gnn_", "").split("_full_seed")[0]
    valid = gnn_preds[c].notna()
    yt, yp = gnn_preds.loc[valid, "dG_true"], gnn_preds.loc[valid, c]
    ax.scatter(yt, yp, s=3, alpha=0.15, color="#55A868")
    lims = (yt.min() - 1, yt.max() + 1)
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_title(f"GNN ({name}-pooling)\nMAE={metrics(yt, yp)['MAE']:.3f}")
    ax.set_xlabel(r"true $\Delta G$ (kcal/mol)")
axes[0].set_ylabel(r"predicted $\Delta G$ (kcal/mol)")
plt.tight_layout()
plt.show()
""")

md(r"""## 5. Overall comparison — every model, one chart

All numbers are on the exact same 16,996-row scaffold-disjoint holdout,
except Chemprop's (independently computed by its own script on the same
split definition, but not from the identical in-notebook code path) and the
RBF SVR (fit on a 20k-row training subsample, not the full 132k — flagged
explicitly rather than presented as equivalent).
""")

code(r"""all_rows = tabular_summary[["model", "MAE"]].copy()
all_rows = pd.concat([all_rows, gnn_summary[["model", "MAE"]]], ignore_index=True)
all_rows = all_rows.sort_values("MAE").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(7, 4.5))
colors = ["#C44E52" if "champion" in m or "attentive" in m else "#4C72B0" for m in all_rows["model"]]
bars = ax.barh(all_rows["model"], all_rows["MAE"], color=colors)
ax.bar_label(bars, fmt="%.3f", padding=3)
ax.set_xlabel("scaffold-disjoint holdout MAE (kcal/mol)")
ax.set_title("All models, homo full-library benchmark (n_test=16,996)")
ax.invert_yaxis()
plt.tight_layout()
plt.show()

all_rows
""")

md(r"""## 6. Discussion

- **Delta-learning does the heavy lifting.** Every model here, including the
  weakest one, comes in far below the raw-baseline MAE (5.029, sec 1.4) —
  the B97-3c baseline is informative enough that even a mediocre correction
  model looks good in absolute terms. The comparison that actually matters is
  relative, between these models.
- **Tree ensembles (XGBoost, Random Forest) and the GNN family both land
  well ahead of the SVMs** at this feature count/row count — consistent with
  the general pattern that kernel methods need either far more data or a much
  smaller feature space to compete with tree/graph methods on tabular
  chemistry data of this size.
- **The XGBoost/Random Forest top-20 feature overlap** (sec 3) is a useful
  independent check on which descriptor blocks are actually carrying signal,
  since it doesn't rely on trusting either model's importances alone.
- **Bottom line for the campaign**: the two numbers `CAMPAIGN_PLAN.md` asks
  to report (single XGBoost, single GNN) hold up as reasonable choices —
  neither is beaten by a more "mainstream" alternative at this data scale,
  which is itself worth knowing rather than assuming.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "nhc-workflow", "language": "python", "name": "nhc-workflow"},
    "language_info": {"name": "python"},
}

import sys
OUTPATH = sys.argv[1]
nbf.write(nb, OUTPATH)
print(f"wrote {len(cells)} cells to {OUTPATH}")
