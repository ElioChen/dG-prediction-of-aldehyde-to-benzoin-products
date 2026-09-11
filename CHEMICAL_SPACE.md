# Chemical Space & the Flying Dataset — specification

> Status: **spec written 2026-09-10; build order steps 1-3 done 2026-09-11**
> (`aldehyde_index.parquet` frozen, `chemical_space.py`'s `pair(i,j)` feature
> path written + verified, see sec8). Written 2026-09-10 on the user's direction
> ("cross 的建库之前根本不对，真正的化学空间应该是 220k 的平方 … 需要知道 flying
> dataset，方便以后读取以及模拟，预测"). See `PROJECT_PLAN.md` §2.11.
>
> Purpose: define, once and unambiguously, **what the benzoin cross ΔG chemical
> space is** and **how any future step reads it** — labeling, simulation,
> prediction, active-learning acquisition, Goal-3 screening — without ever
> materializing a table of all pairs.

---

## 1. Why the old pool was wrong

The cross model was trained/screened against `candidates_v3` (~1.24 M directed
pairs). That set was a **constructed subset** — it came from a bounded generator
(MaxMin diversity picks, category strata, a fixed reservoir) and was never the
real space. Every "screen the whole library" statement made against it is
therefore scoped to an arbitrary sample, and the round-by-round AL acquisition
was choosing from that arbitrary sample.

**The real space** is: pick any aldehyde as the donor, any aldehyde as the
acceptor. Donor and acceptor play chemically distinct roles (the donor becomes
the acyl anion / Breslow carbon; the acceptor is electrophile), so **ordered**
pairs are the honest count.

---

## 2. The space, precisely

| | |
|---|---|
| **Base library** | `data/library/aldehydes_clean_v6.csv` — **220,859** rows (the CSV has 220,860 lines incl. header -- corrected 2026-09-11, was off by one) (mono-aldehydes, filtered, `cho_class` + `xtb_risk` tagged). Deliberately includes molecules that fail xTB/DFT (§ `PROJECT_PLAN.md` 1.2). |
| **homo space** | the 220,859 diagonal pairs `(i, i)`. |
| **cross space** | all ordered `(i, j)`, `i ≠ j` optional-included: **220,859² = 4.878 × 10¹⁰** ordered pairs (48.8 billion). Unordered ≈ 2.44 × 10¹⁰. |
| **materialized** | ~35,528 pairs have a DFT label today (the parked AL rounds 1–10). Everything else is virtual. |

No file will ever hold all pairs. A pair is an **address**, computed on demand.

---

## 3. Canonical aldehyde index (must be frozen first)

- **`ald_idx`** ∈ `[0, 220859]` — a stable integer, assigned once, in the row
  order of a **frozen** copy of `aldehydes_clean_v6.csv`. Never re-derived,
  never an `enumerate()` over a filtered view.
- Carried alongside: canonical SMILES (RDKit canonical, single fixed protocol),
  `InChIKey` (the human-readable secondary key), `cho_class`, `xtb_risk`,
  Bemis–Murcko scaffold SMILES, and a `computable` flag (set false once a
  molecule is confirmed to fail geometry/DFT — so downstream can skip it without
  re-attempting).
- Artifact: `data/chemical_space/aldehyde_index.parquet` — **frozen 2026-09-11**
  (`cross_benzoin/build_aldehyde_index.py`). This file is the single source of
  truth; `homo_v6/*` caches must be re-keyed to it (build-order step 2, not yet
  done).

---

## 4. Per-aldehyde caches (compute once, reuse for ~10⁶ pairs each)

Keyed by `ald_idx`. Most already exist under `data/cross_benzoin/homo_v6/`,
just not indexed consistently:

| cache | content | source | status |
|---|---|---|---|
| geometry | GFN2 `--ohess` xyz + `xtbopt` | funnel_v3 | ✅ ~180k (partial archive) |
| xТБ energetics | `E_el`, `G` (Gibbs), thermal `G−E_el` | xtb | ⚠️ present but `G_xtb` id-misaligned for ~8% — **bounds-check per §1.2 rule** |
| local QM descriptors | frontier orbitals, Fukui, Mulliken/ADCH, WBO, QTAIM, Vbur, Sterimol (the "local QM" ~35% of the 260 schema) | `ald_descriptors_qm.py` | ✅ `aldehydes_all.csv` 209,526 |
| Mordred (slim) | 2D/3D descriptors, curated subset | mordred | ✅ `aldehydes_mordred_slim102.csv` |
| BDE | formyl C–H BDE (ALFABET + g-xTB) | `pipeline/bde` | ✅ ~220k |
| g-xTB / B97-3c single points | for the Δ-learning baseline | this project | 🔄 (per-pair, see §6) |

**Principle:** the aldehyde is the unit of expensive compute. A pair's features
are (donor cache) ⊕ (acceptor cache) ⊕ (a few product-side + interaction terms).
So the cost of the whole 4.9 × 10¹⁰ space is ~220k aldehyde featurizations +
cheap per-pair assembly, not 4.9 × 10¹⁰ of anything.

---

## 5. The read API (the "flying dataset")

A thin module — `chemical_space.py` — exposing a lazy view:

```
space = FlyingDataset(index="data/chemical_space/aldehyde_index.parquet",
                      caches="data/cross_benzoin/homo_v6/",
                      labels="data/chemical_space/dft_labels.parquet")

space.n_aldehydes                      # 220859
space.pair(i, j)                       # -> dict:
    {  donor_idx, acceptor_idx,
       donor_smiles, acceptor_smiles, product_smiles,   # product from the reaction template
       reaction_type,                                   # from the two cho_class
       features: np.ndarray[260],                       # assembled from caches, lazily
       baseline_gxtb, baseline_b973c,                   # None if not computed
       scaffold_split,                                  # scaffold-disjoint, computed from the 2 scaffolds
       label_dG,                                        # None unless in dft_labels
       computable }                                     # False -> caller skips

space.iter_pairs(donor=None, acceptor=None, where=...)  # generator, never a list
space.sample(n, strategy=..., seed=...)                 # for AL / screening batches
space.features_batch(pairs)                             # vectorized assembly for a batch
```

- **product SMILES**: generated by the reaction template
  (`featurize_product.build_product` / `FP.build_product`, the same one the
  pipeline uses) — deterministic from (donor, acceptor).
- **feature assembly**: reuse `assemble_cross_training_table*`'s column logic,
  refactored to operate on one pair from cache dicts instead of a merged table.
- **correction found while implementing step 3 (2026-09-11):** only part of the
  260-feature schema is actually lazy. `donor_*`/`acceptor_*` local QM + BDE
  (aldehyde-index cache) and all three RDKit-2D blocks and `interaction_*`
  terms are genuinely computable from just the two aldehydes -- **but**
  `product_*` QM (mulliken/wbo/fukui/vbur/sterimol/hb_*/dih_core) and
  `product_mordred_*` (checked: `ignore_3D=False`, several families are
  inherently 3D) both come from xTB/DFT run on the product's own optimized
  geometry, same as `baseline_gxtb`/`baseline_b973c`/`label_dG` -- **none of
  those four groups are lazy.** `pair(i, j)` therefore returns a `lazy_features`
  dict (always available) plus a `computed_full` flag + `known_row` (only
  populated if the address has already been through the DFT/xTB pipeline, read
  back verbatim rather than recomputed). This is a correction to the "features:
  np.ndarray[260], assembled from caches, lazily" line above, not a redesign of
  the address/cache architecture.
- **scaffold_split**: a pair is `train` iff *neither* aldehyde's scaffold is in
  the held-out scaffold set; `test`/`validation` iff *both* are; else `mixed`
  (excluded). Held-out scaffold sets are frozen in the index file.
- **no disk pair table.** `iter_pairs` / `sample` yield addresses; assembly is
  on demand and only transiently cached.

---

## 6. DFT-label storage

- `data/chemical_space/dft_labels.parquet` — one row per computed pair:
  `donor_idx, acceptor_idx, dG_r2scan_kcal, dG_b973c_kcal, dG_gxtb_kcal,
   geom_hash, method_tag, campaign, date`.
- Keyed by the `(donor_idx, acceptor_idx)` address — **not** an InChIKey pair
  string, **not** a `candidates_v3` row id.
- **Migration:** map the existing 35,528 labeled pairs (rounds 1–10 + Tier B)
  onto `ald_idx` addresses via canonical SMILES → `ald_idx` lookup; keep them
  (they are valid data), tag `campaign="al_r1_10"`.

---

## 7. What using this looks like

- **Redone cross AL** (`PROJECT_PLAN.md` §2.7): the acquisition function scores
  `space.sample(n_candidates, strategy="stratified")` batches, ranks, picks the
  next DFT batch as addresses, `run` labels them, `dft_labels.parquet` grows.
  No `candidates_v3`.
- **Goal-3 screening**: `space.iter_pairs(where="dg_favorable & ~baseline_risk")`
  streamed through `predict_dg`, top-k by conformal upper bound → shortlist.
- **Simulation** (any future DFT campaign): takes a list of addresses, pulls
  geometries from the caches, computes, writes back to `dft_labels`.

---

## 8. Build order (each step small, understood, verified)

1. ✅ **Done 2026-09-11.** Froze `aldehyde_index.parquet` from
   `aldehydes_clean_v6.csv` (220,859 rows; `ald_idx`, canonical SMILES,
   `InChIKey`, `cho_class`, `xtb_risk`, scaffold, `computable` left unknown).
   Reused rather than recomputed the Bemis-Murcko scaffold from
   `candidates_v3/aldehydes_with_scaffold_split.parquet` — its `id` column
   verified positionally == `ald_idx` on the full 220,524-row overlap (0
   mismatches, exact raw-SMILES match), so this was a safe row-order merge.
   335 rows (absent from that parquet) have `scaffold=None`; 0 rows failed
   RDKit canonicalization; 0 canonical-SMILES duplicates. Script:
   `cross_benzoin/build_aldehyde_index.py`.
2. ✅ **Done 2026-09-11 — turned out to be verification, not a rebuild.**
   Checked `homo_v6/aldehydes_all.csv`'s existing `id` column (after
   `qc.norm_id`) against `ald_idx` for the **full population**, not a sample:
   209,526/209,526 rows matched 1:1, 0 orphans on either side — `aldehydes_all.csv`
   was already correctly keyed by `ald_idx`, just not documented as such. One
   real finding: its `smiles` column is **not reliably canonical** — 2,604/209,526
   (1.24%) differ from `aldehyde_index.parquet`'s freshly-computed
   `smiles_canonical` by representation only (e.g. Kekulized upper-case aromatic
   vs RDKit's lower-case canonical form — same molecule, checked by hand on 8
   samples). **So: join future caches to `ald_idx`, not to `smiles` string
   equality** — this is exactly the drift the frozen index exists to remove.
3. ✅ **Done 2026-09-11.** `cross_benzoin/chemical_space.py` `FlyingDataset.pair(i,j)`
   -- see sec5's correction note for what "the feature path" turned out to mean
   (a lazy tier + a cache-hit tier, not one flat 260-vector). Verification
   (`cross_benzoin/verify_chemical_space_pair.py`, 20 random pairs sampled from
   the round-10 champion table): the two deterministic lazy tiers matched
   **bit-exact** -- RDKit-2D 600/600, `interaction_*` all matched, `product_smiles`
   20/20. Donor/acceptor QM matched 995/1380 (72%) within 0.2% relative
   tolerance; the rest differ by up to a few % -- **not a bug**, this is the
   already-characterized `aldehyde-recompute-fidelity` effect (the round-10
   table's QM snapshot slightly predates the current, since-regenerated
   `aldehydes_all.csv`; judged by relative deviation, not equality, per that
   memory). One real bug caught and fixed along the way: the first verification
   draft searched for a matching `pair_key` to recover a row's (donor,
   acceptor) address, which silently picked the wrong row when a `pair_key`
   paired with both role orderings in the table -- fixed by resolving each
   row's own donor_smiles/acceptor_smiles directly instead of round-tripping
   through the address lookup.
4. Add labels + split + baselines.
5. Migrate the 35,528 labels; retire `candidates_v3` (move, don't delete).
6. Only then: point the redone AL / screening at it.

**Do not skip the verification in step 3** — a silent feature-assembly drift
would poison every downstream model.
