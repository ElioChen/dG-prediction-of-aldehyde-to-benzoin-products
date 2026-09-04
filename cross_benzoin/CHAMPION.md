# cross-benzoin ΔG champion

**Current champion: rounds 1-10 scaffold-disjoint blend** (adopted 2026-09-04).

## Load

```python
from predict_cross_champion import CrossBenzoinBlendPredictor
pred = CrossBenzoinBlendPredictor.load(
    "data/cross_benzoin/cross_round10/scaffold_disjoint_10rounds_v1",
    gnn_dir="data/cross_benzoin/cross_round10/gnn_attentive_10rounds_v1",
)   # blend_w_gnn = 0.50, read from the GNN metadata.json
```

`predict_cross_champion.py` now defaults `--model-dir` / `--gnn-dir` to these.

## Components

| part | path |
|---|---|
| MLP+XGB ensemble | `cross_round10/scaffold_disjoint_10rounds_v1/models/ensemble_scaffold_disjoint.joblib` |
| single-XGB champion | `.../models/champion_scaffold_disjoint.joblib` |
| attentive-pooling GNN | `cross_round10/gnn_attentive_10rounds_v1/models/gnn_state.pt` |
| frozen 260-feature schema | `cross_round8/scaffold_disjoint_8rounds_v1/models/feature_list.json` (unchanged since round 7) |
| training table | `cross_round10/cross_train_table_10rounds_scaffold_split_labeled_slim260.parquet` (35,528 rows, clean-train 22,771) |

## Scaffold-disjoint holdout (n=448, frozen since round 7)

| | MAE | R² |
|---|---|---|
| **blend (w_gnn=0.50)** | **2.215** | — |
| MLP+XGB ensemble | 2.326 | 0.739 |
| GNN-only | 2.313 | — |
| single-XGB | 2.548 | 0.702 |
| g-xTB baseline | 5.037 | −0.14 |

Bootstrap (B=20000): blend−ensemble delta 0.111, 90% CI [0.047, 0.181] (excludes 0),
**P(blend better than ensemble) = 0.9986**.

## Why r1-10 over r1-9

r1-9 blend was 2.167 on the same holdout — nominally better, but within noise
(each MAE's bootstrap SE ≈ 0.15 at n=448). r1-10 is trained on 1,969 more DFT-SP
pairs (the round10 active-learning batch, which targets high-uncertainty regions the
frozen holdout doesn't sample), and its GNN blend advantage is if anything stronger
(w_gnn 0.40 → 0.50, delta 0.086 → 0.111). Latest + most data + robust blend wins;
the flat holdout is not a reason to keep the smaller model. **r1-9
(`cross_round9/scaffold_disjoint_9rounds_v1` + `gnn_attentive_9rounds_v1`) is the
fallback.**

## Lineage

r1-7 recovered (2026-09, post-purge; GNN null, w_gnn=0) → r1-9 (2026-09-04; r8/9
DFT labels recovered from scratch; **GNN becomes useful, w_gnn=0.40, P=0.995**) →
**r1-10** (2026-09-04; + round10 AL batch; w_gnn=0.50, P=0.999).
Full log: `RUN_LOG_20260903.md`.

## Predicting ΔG for new aldehyde pairs

**Assemble+predict half** (validated, MAE 1.65 on 5 known round10 pairs — reproduces
the training featurization chain): if you already have a
`cross_round*_dft_products.csv`-schema file (id, donor_id, acceptor_id,
donor_smiles, acceptor_smiles, smiles, xtb_*, dG_gxtb_kcal, …):

```
/home/schen3/venv/nequip/bin/python cross_benzoin/predict_dg.py \
    --products-csv <pairs_products.csv> --out preds.csv
```
→ `preds.csv`: id, …, `dG_pred_kcal`, `ens_member_sigma` (cheap 3-learner spread,
not the full bootstrap epistemic estimate).

**From scratch** (new pairs, needs the slow GFN2-xTB geometry — newly wired, not yet
tested on genuinely novel pairs):

```
sbatch cross_benzoin/slurm/submit_predict_dg.sh <pairs.csv> <workdir> <out.csv>
# pairs.csv: donor_id,acceptor_id,donor_smiles,acceptor_smiles
```

Aldehydes must be in `data/library` (donor_*/acceptor_* descriptors are pulled from
the 220k library by canonical SMILES); `bde_gxtb_kcal` is pulled from any prior
round's `bde_gxtb/` for known products, NaN (median-filled) otherwise.
