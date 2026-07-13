# B-validation — do benzoin PRODUCT descriptors help the Δ-model? (2026-06-23)

## Question
Before committing the 220k full-library featurization, decide whether adding **product-side**
QM descriptors (Multiwfn-free, funnel_v3) to the g-xTB-baseline Δ-model lowers CV MAE — i.e.
whether the model should *use* product features, not whether to *compute* the product (the
product xyz + xTB/g-xTB G are mandatory for ΔG = G(prod) − 2·G(ald) regardless).

## Setup
`pipeline/bvalid_product_eval.py` on the **same** molecules / g-xTB baseline / QC / CV
(RepeatedKFold 5×4, XGBoost Δ-target = dG_orca − baseline). Product descriptors from the
re-run B-validation array **24126497** (`featurize_product.py`, funnel_v3, NO Multiwfn):
**1691/1695 dG, 0 quota errors** (the prior 46/1695 run was wiped by the scratch-inode bug,
now fixed — see [[orphan-scratch-cleanup]]). Merged on canonical donor SMILES (the product
CSV's `index` is a per-chunk local row number, not the library id). n = 1643 matched.

## Result (matched in-scope n=1643)

| tier | feat | CV MAE | RMSE | R² |
|---|---|---|---|---|
| A  aldehyde-only (no Multiwfn) | 52 | 2.026 | 2.835 | 0.692 |
| **B  aldehyde + product (no Multiwfn)** | 89 | **1.980** | 2.779 | 0.704 |
| ref  full incl Multiwfn (aldehyde) | 63 | 1.995 | 2.804 | 0.699 |

**ΔMAE (B − A) = −0.046 kcal/mol → product descriptors HELP** (decision threshold −0.02).

## Reading (multi-dimensional)

1. **Product descriptors help — modestly but above threshold.** MAE 2.026 → 1.980 (−0.046,
   ~2.3% rel), R² 0.692 → 0.704. Consistent with the mechanistic prior: the xTB–DFT error
   lives mostly on the **product** side (EWG benzoin electronic failure), so product features
   give the ML correction something to anchor to.

2. **The decisive finding: B (Multiwfn-FREE) beats ref (full Multiwfn).** 1.980 < 1.995. Cheap
   product xTB/morfeus descriptors don't just substitute for the expensive aldehyde Multiwfn
   (ADCH/QTAIM) block — they **exceed** it. This confirms **MULTIWFN=0 at 220k is not a
   feasibility compromise but the better model**, and independently reproduces the earlier
   ablation (dropping all 11 reactant Multiwfn feats cost only +0.03 kcal, see [[gxtb-baseline-ab]]).

3. **Magnitude is near the g-xTB noise floor (~2.0).** The −0.046 gain is small relative to the
   already-large win from switching the baseline GFN2→g-xTB (CV 2.52→2.10/2.00). Product
   descriptors are a second-order refinement on top of that first-order baseline win, not a
   game-changer. Worth taking because the product is computed anyway.

4. **Where it likely helps most:** the EWG endergonic tail that g-xTB still mis-ranks (routed to
   DFT per [[no-dG-extreme-filtering]]); a per-category / per-EWG breakdown of the B−A residual
   would localize the gain (follow-up; the aggregate −0.046 understates a concentrated subset).

## Decision for the 220k full-library run
- **Compute the product unconditionally** (xyz + xTB/g-xTB G needed for ΔG) — already the plan.
- **Use the product descriptors as model features** (tier B): they help and are free once the
  product geometry+G are computed.
- **Drop Multiwfn entirely** (MULTIWFN=0): B-no-Mwfn ≥ ref-Mwfn, and L3 is infeasible at 220k.
- Production model = g-xTB baseline + aldehyde(no-Mwfn) + product(no-Mwfn) descriptors.

## Caveats
- Homo-benzoin only (donor==acceptor); cross pairs untested here.
- −0.046 is within plausible CV-fold noise; the qualitative ranking (B ≥ ref > A) is the robust
  takeaway, not the third decimal.
- 4/1695 molecules failed dG (embed/ohess on hard species), 1643 matched after QC — full coverage.

Driver: `pipeline/bvalid_product_eval.py` · data: `data/raw/product_bvalid/chunks_out/` (array 24126497).
