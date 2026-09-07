# Aldehyde library construction history + cross-benzoin dG workflow / descriptor engineering (2026-09-07)

> English version of `ALDEHYDE_LIBRARY_AND_DG_WORKFLOW_20260907.md`. Keep the two in sync
> when either changes.
>
> Separate from `PROJECT_SUMMARY_20260904.md` (that one is the chronological, AL-centric
> narrative). This document answers two lower-level questions: **where the aldehyde
> structure library came from and what its backup status is**, and **what the dG
> prediction workflow actually looks like step by step, with the descriptor-engineering
> detail for each stage**. It is a technical reference, not a timeline — when it goes
> stale, edit it in place; no need to rewrite the whole thing as a "snapshot".

---

## 1. The aldehyde structure library (`aldehydes_clean_v6`)

### 1.1 Source and filtering pipeline

**Raw pool**: ~450k molecules (`name/SMILES/MW/CID/...`), from a PubChem-style candidate
pool that predates this repo. The path in the script comments
(`/scratch-shared/schen3/aldehydes.csv`) no longer exists and is older than the start of
the current `agent/recovery-20260902` git history, so the original download/screening
session cannot be reconstructed — **this is the only untraceable link in the chain**;
every step after it has a script and an output file on record.

**Filter script**: `pipeline/filter_smiles_v6.py` (iterated v1→v6; v6 is current, and the
v1–v5 scripts and outputs are still in the repo at `data/library/aldehydes_clean_v{1..6}.csv`
so the effect of each rule change can be traced). What v6 changed over v5 (verbatim from
the script docstring):
1. New reject rule `malformed_boron` — bare boron atoms with incomplete valence (e.g.
   `[B]-C`). This kind of dirty PubChem data makes GFN2-xTB return physically nonsensical
   energies (a −237 kcal/mol value was seen in a pilot).
2. The `xtb_risk` tag was extended to boron / phosphorus / selenium — these three
   elements were ~11–12× enriched among v5's ΔG outliers; QM is not fully reliable for
   them but they are kept (not rejected — tagged for down-weighting / extra review),
   handled the same way as the existing nitro / azide / N-oxide tags.

**Reject taxonomy (first-failing-rule-wins, in script order)**:
`invalid_parse` (RDKit parse failure) → `multi_component` (salt / mixture / counter-ion)
→ `net_charged` (non-zero net charge) → `mw_too_high` (>500 Da) → `disallowed_element`
(element not in {H,C,N,O,F,S,Cl,Br,I,B,Si,P,Se}) → `isotope` (isotope labels) →
`malformed_boron` (new in v6) → `zwitterion_or_ylide` (zwitterions not explained by an
allowed nitro / azide / N-oxide / amine-oxide motif) → `not_single_aldehyde` /
`multi_aldehyde` (0 or >1 true aldehyde groups) → `enal` / `ynal` (α,β-unsaturated /
ynals — routed to NHC Stetter / homoenolate, not standard benzoin substrates) →
`alpha_dicarbonyl` → `reactive_group` (nitroso / azo / SF5 / hypervalent S–F / ketene /
isocyanide) → `vinyl_conj` → `aliphatic_too_large` (no aromatic ring and >12 carbons).

**Final output**: `data/library/aldehydes_clean_v6.csv` = **220,859 aldehydes** (this is
the "220k" / "220,859" quoted everywhere in the project). `aldehydes_rejected_v6.csv` =
~728k rejected rows (with reject reason), for audit/review.

### 1.2 Backup / version-control status

**Bottom line: this file is safe and needs no extra care.**

```
git log --oneline -- data/library/aldehydes_clean_v6.csv
25f5400 Fresh history: initial commit after old .git object database was found corrupted (2026-07-13)
```

It has been **in git from the very start** (`25f5400`, the first commit of the current
history, right after the 2026-07 git-database corruption "fresh history" event), was
never gitignored, and is small (~41 MB). Git itself (plus the pushed GitHub remote) is
its backup — no separate home-directory archive needed. The `v1`–`v5` historical
versions and `aldehydes_rejected_v{1..6}.csv` are likewise all in git, tracked with the
whole `data/library/` directory.

**This is in sharp contrast to the local/global descriptor libraries in the next section
(`aldehydes_all.csv` etc.)** — those are larger and grow faster, were excluded by
gitignore, were all lost in the purge, and were the main content of the project's 2026-09
recovery work.

---

## 2. Local/global descriptor libraries: construction and recovery history

The structure library (§1) is just SMILES + reject tags. What actually feeds the model
is a second layer of files built on top of it — geometry optimisation + QM/descriptor
computation for every aldehyde — all under `data/cross_benzoin/homo_v6/`. This layer is
entirely gitignored (`*_all.csv` and similar), and **this is exactly the layer the
2026-07 full scratch purge destroyed**.

### 2.1 File inventory and current status (2026-09-07)

| file | content | rows | current status |
|---|---|---:|---|
| `aldehydes_bdfe_gxtb_descriptors.csv` | aldehyde BDE (bond dissociation energy) labels, g-xTB single point | 220,522 | ✅ bit-exact re-stitched from the home backup (`homo_v6_scratch_archive/bde/bdfe_gxtb`, chunk by chunk) |
| `aldehydes_mordred_slim102.csv` | aldehyde-side mordred descriptors (102-column curated subset) | ~220k | ✅ bit-exact re-stitched from the home backup |
| `aldehydes_all.csv` | aldehyde local 3D electronic-structure descriptors (xtb / morfeus / multiwfn) | **209,526** | ✅ full rebuild 09-06 (the earlier purge recovery had only filled a 42,336-row local subset); **the one gap**: `G_gxtb` (whole-molecule g-xTB free energy) was not computed in the 09-06 rebuild — see §6 (**resolved 2026-09-07**, coverage now 99.07%) |
| `products_bdfe_gxtb_descriptors.csv` | product BDE labels | 218,966 | ✅ re-stitched from the home backup 09-02 |
| `products_all.csv` | product local 3D electronic-structure descriptors | **184,199** | ✅ 09-06 — **the first full version ever** (no full version existed before, not even pre-purge — only a partial) |
| `aldehydes_scaffold_split_from_dG.csv` / `products_scaffold_split.csv` | Bemis-Murcko scaffold-disjoint train/val/test split | 220,859 full coverage | ✅ always in git, unaffected by the purge |
| B6 BDE model checkpoints (`.pt`) | — | — | ❌ original weights permanently lost in the purge; the 09-06 full retrain produced new ones (see `PROJECT_SUMMARY §4.4`), now `git add -f`'d |

### 2.2 Scope of the 2026-07 purge loss + recovery paths

- **Why it was lost**: this layer is large (hundreds of MB to GB), iterates fast, and by
  early-project convention was `gitignore`d, existing only on Snellius `scratch` (path at
  the time `/scratch-shared/schen3/benzoin-dg`). Around 2026-07-20~29 scratch was wiped;
  every file without a gitignore exemption and without a systematic home backup was gone.
- **Recovery was a mix of luck and discipline**: `aldehydes_bdfe_gxtb_descriptors.csv` /
  `aldehydes_mordred_slim102.csv` could be recovered bit-exact only because **an earlier
  session, out of personal habit, had tar'd them into `/home/schen3/benzoin_backups/`**
  (not a project policy — pure chance). `products_all.csv` had no such accidental backup
  and could only be recomputed in full from 2026-09-02 (~6× the aldehyde-side cost,
  because product molecules are larger and more complex).
- **Systematic remediation from 2026-09-02**: `cross_benzoin/slurm/submit_backup_recovery_artifacts.sh`
  was created — it systematically tars this class of "gitignored + a training output, so
  losing it means losing it for real" files into
  `/gpfs/home4/schen3/benzoin_backups/recovery_20260902/` (home directory, not subject to
  scratch lifecycle policy). It is idempotent, re-runnable any time, and its manifest was
  extended again on 09-07 (adding `products_all.csv` itself and this cycle's model-sweep
  result directory).

### 2.3 Backup-location summary (2026-09-07 state)

| backup directory | content |
|---|---|
| `/home/schen3/benzoin_backups/bde_homo_rebuild_20260902/` | `aldehydes_all.partial42k_20260902.csv.tar.gz` (the 09-02 partial rebuild, kept for historical comparison) + `aldehydes_all_full_20260906.csv.tar.gz` (the 09-06 full rebuild) |
| `/gpfs/home4/schen3/benzoin_backups/recovery_20260902/` | the systematic archive managed by `submit_backup_recovery_artifacts.sh`: the aldehyde-library trio, DFT-SP labels, round8/9/10 intermediate tables, champion model dirs, the 09-06 BDE model-sweep results (`runs/logs/scaffold_disjoint_bde/`) |
| git (`data/library/`, `*_scaffold_split*.csv`, champion model `.joblib`/`.pt`, all result `.json`/`_pred.csv`) | anything of manageable size is `git add -f`'d straight into the repo; pushed to the GitHub remote on 2026-09-07 — the most reliable backup layer |

**The one gap that was still open at the time of writing**: the `G_gxtb` column in
`aldehydes_all.csv` (§6). Not "lost" — never computed in the 09-06 rebuild.
**Closed 2026-09-07**: recompute array `26432805` finished, `merge_aldehyde_gxtb.py`
brought coverage from 1.3% to 99.07% (207,580 / 209,526); the residual ~0.93% is the
25 known-unrecoverable featurize chunks and is accepted.

---

## 3. dG prediction workflow overview (end to end)

Given a pair of aldehydes (donor + acceptor, may be the same molecule), predict the
benzoin condensation `dG_orca_kcal` (r2SCAN-3c DFT level). Full flow:

```
(1) aldehyde structure library (§1) -> (2) geometry + local descriptors (cb_featurize.py,
funnel_v3 search) -> (3) mordred global descriptors (free on the aldehyde side + cheap
on-the-fly on the product side) -> (4) DFT single-point labels (training set only) ->
(5) assemble table (assemble_cross_training_table_v3.py) -> (6) prune to the champion's
260 features (prune_table_to_champion_features.py) -> (7) model inference
(predict_cross_champion.py: MLP+XGB tabular ensemble (+) triple-encoder attentive GNN,
blend w_gnn=0.5) -> (8) deployment wrapper (predict_dg.py, see PROJECT_SUMMARY §3.10/§4.5)
```

### 3.1 Geometry search method: funnel v3

`pipeline/compute/conf_funnel_v3.py` (adds a topology guard on top of v2):

- **v1→v2 motivation**: an early K3→K10→K20→K30 convergence study found the DFT ΔG
  labels carried ~1.4 kcal of conformer-search noise plus ~4% catastrophic failures
  (6–15 kcal). The root cause: v1 used `numThreads=0` for RDKit conformer embedding —
  multi-threaded embedding is non-deterministic even with a fixed random seed, so
  independent runs would miss the true global minimum. v2 fix: `numThreads=1 +
  randomSeed=42` (fully reproducible) + RMSD pruning (retained conformers are distinct
  wells, not near-duplicates) + doubled sampling density.
- **v2→v3 motivation**: a CREST A/B test found funnel's only *catastrophic* failure mode
  is the lowest-energy conformer having **broken connectivity** (GFN-FF relaxed a
  conformer into an isomerised / bond-broken structure whose spurious low energy then
  dominated the Boltzmann average — one measured case was 12 kcal off). CREST's built-in
  topology check avoids this, but its GFN-FF metadynamics doesn't sample large benzoin
  molecules densely enough. So rather than switch engines, funnel gets the one layer it
  was missing: **each conformer gets a cheap RDKit bond-connectivity check, and any
  conformer whose connectivity disagrees with the input graph is discarded**. The rest of
  the flow (dense ETKDG sampling → GFN-FF pre-screen → keep top-L → GFN2 optimisation →
  Boltzmann ranking) is unchanged.
- **Standard flow** (unchanged since v1): dense ETKDG sampling → GFN-FF fast pre-screen →
  keep top-L (default L=10) → GFN2 `--opt` refinement → rank by GFN2 energy, take the
  lowest as the representative conformer.

### 3.2 Label sources

- **True label `dG_orca_kcal`**: ORCA r2SCAN-3c single point on the funnel_v3 geometry.
  Only molecule pairs that were AL-selected for DFT have it (used for training/eval).
- **Physical baseline `dG_gxtb_kcal`**: g-xTB single point, also on the funnel_v3
  geometry, orders of magnitude faster than DFT, present for all 220k pairs. It is both a
  model input feature and the control group for "how much does the ML improve over the
  physical baseline".
- **Aldehyde-side whole-molecule free energy `G_gxtb`** (distinct from the pair-level
  `dG_gxtb_kcal` above): a hybrid-correction recipe from `gxtb_baseline.py` — GFN2
  `--ohess` (optimisation + Hessian) gives the thermodynamic correction `(G_gfn2 −
  E_gfn2)`, g-xTB only does the single-point electronic energy `E_gxtb` (g-xTB has no
  production-grade thermodynamic-correction pipeline in this project), so `G_gxtb =
  E_gxtb + (G_gfn2 − E_gfn2)`. This is the quantity the 09-06 BDE rebuild skipped and
  that was backfilled on 09-07 — see §6.

---

## 4. Descriptor-engineering detail

### 4.1 Local electronic-structure descriptors (xtb / morfeus / multiwfn)

The aldehyde side (`ALDEHYDE_FEATS`, 27 raw quantities, see
`cross_benzoin/assemble_cross_training_table.py`) and the product side (`PRODUCT_FEATS`,
38) are computed with the **same method** and the fields are largely symmetric, so the
model can learn a consistent cross-species representation:

| layer | quantities | notes |
|---|---|---|
| xtb single point / derived | `xtb_energy`; `xtb_HOMO`/`xtb_LUMO`/`xtb_gap`; `xtb_IP`/`xtb_EA` (vertical, `--vip`/`--vea`); `xtb_mu`/`xtb_eta`/`xtb_omega` (chemical potential / hardness / electrophilicity, derived from IP/EA); `xtb_dipole` | standard reactivity descriptors |
| Mulliken charges | key atoms (aldehyde C/O; product side ketC/ketO/carbC/hydO/hydH) | site-specific, not whole-molecule |
| WBO (Wiberg bond order) | aldehyde C=O; on the product side the newly formed C–C bond + the existing carbonyl bond | bond strength / reactive-centre indicator |
| Fukui indices | `fukui_plus`/`fukui_minus`/`fukui_0`/`dual` (finite difference, `_fukui_finite_diff`) | nucleophilic/electrophilic site ID, taken at key atoms |
| proton affinity `pa_*` | add an H to the aldehyde oxygen, take the energy difference | proxy for basicity / H-bond-acceptor strength |
| morfeus: buried volume `vbur_*` | steric crowding around the key carbon | |
| morfeus: Sterimol `sterimol_L/B1/B5` | length/width steric descriptors | |
| morfeus: `SASA_total`, `P_int` | accessible surface area, polar integral | |
| Multiwfn: ADCH charges, QTAIM (Laplacian / ellipticity at bond critical points) | only in `aldehydes_all.csv` (added this cycle — the `adch_*`/`qtaim_*` columns in the §2.1 table) | one of the B6 BDE model's `x_d` fusion features; not used by the dG project currently |
| product-side only: H-bond geometry `hb_dist`/`hb_angle`/`dih_core` | the newly formed intramolecular H-bond (classic β-hydroxy-ketone conformation) + core dihedral | product-specific, no aldehyde-side analogue |
| aldehyde-side only: whole-molecule free energy `G_xtb` (GFN2) / `G_gxtb` (g-xTB) | | see §3.2; `G_gxtb` was the backfill target |

### 4.2 Global molecular descriptors (mordred + RDKit 2D)

- **mordred** (targeted families set by `pipeline/analysis/finalize_correction_mordred_slim.py`:
  MoRSE / CPSA / Polarizability / GeometricalIndex / MomentOfInertia / PBF /
  McGowanVolume / VdwVolumeABC / Weight / TopoPSA — not the full mordred 1800+, a curated
  subset):
  - **Free on the aldehyde side**: `aldehydes_mordred_slim102.csv` (102 columns) comes
    from the existing full 220k-library mordred computation; donor/acceptor each looked
    up and concatenated — zero new compute.
  - **A real new computation on the product side, but cheap**
    (`add_mordred_cross_products.py`, ~1 s/molecule): reuses the saved product geometry
    (`xyz_file`), no re-optimisation, pure post-processing.
- **RDKit 2D** (`RDKIT_FEATS`, 16: `MW/LogP/TPSA/HBD/HBA/RotBonds/ArRings/ArHetRings/
  AlRings/Rings/Heteroatoms/FractionCSP3/BertzCT/Kappa2/NumStereocenters/n_CHO`): one set
  each for donor/acceptor/product, zero cost (pure topology, no geometry needed).

### 4.3 Pair-level interaction features (tried, not kept in the champion)

`assemble_cross_training_table_v3.py` generates `interaction_gap_HOMOd_LUMOa` (donor HOMO
− acceptor LUMO, frontier-orbital match), `interaction_fukui_match` (donor nucleophilic
Fukui × acceptor electrophilic Fukui), and `MISMATCH_PAIRS` (absolute donor−acceptor
differences of `xtb_gap`/`dipole`/`sterimol_L/B1/B5`/`SASA_total`/`MW`/`TPSA` — a measure
of the electronic/steric complementarity of the pair). **These were designed specifically
on the hypothesis that they carry "cross-specific information that homo pretraining can't
learn" — but the champion's frozen 260-feature list contains not a single
`interaction_*`** (verification in the next section). They didn't survive
feature-selection/pruning into the final model — the GNN half (the GNN blend component)
has most likely already learned equivalent or stronger interaction information
implicitly, making these hand-built explicit interaction terms redundant. A negative
result worth recording, to avoid re-"inventing" the same features later.

### 4.4 What the champion's 260 features actually are (`feature_list.json` measured counts)

| group | count | notes |
|---|---:|---|
| `donor_*` | 71 | 43 local QM raw quantities (§4.1) + 28 mordred |
| `acceptor_*` | 82 | 43 local QM raw quantities + 39 mordred |
| `product_*` | 69 | 16 RDKit 2D + 53 mordred |
| product-side local QM (no prefix: `xtb_*`/`fukui_*`/`mulliken_*`/`wbo_*`/`sterimol_*`/`dual_*`/`vbur_*`/`hb_*`/`pa_*`/`SASA_total`/`P_int`/`dih_core`) | 35 | the "product-side only" batch in the §4.1 table |
| `bde_gxtb_kcal` | 1 | pair-level g-xTB BDE baseline (from `cross_round*/bde_gxtb/`; a different quantity from the aldehyde-side `G_gxtb` in §3.2 — don't confuse them) |
| `interaction_*` | **0** | see §4.3 — tried, not kept |
| **total** | **260** | |

mordred is 120/260 (46%), the single largest source; RDKit 2D 50/260 (19%); local QM raw
quantities (including `donor_G_gxtb`/`acceptor_G_gxtb`, see §6) ~90/260.

---

## 5. Model architecture (`predict_cross_champion.py`)

**Blend = tabular ensemble (+) triple-encoder GNN**, the two halves trained
independently and linearly mixed at inference (`blend_w_gnn` read from metadata.json;
current champion r1-10 is 0.50):

- **Tabular half**: `MLPXGBEnsemble` (`train_cross_ensemble.py`) — a simple average of
  one MLP + two XGBoost models ("XGB-a"/"XGB-b", most likely bagging variants with
  different seeds/hyperparameters), fed the pruned 260 features.
- **GNN half**: `TripleGNN` / `TripleGNNAttn` (`train_cross_gnn.py` +
  `gnn_architectures.py`) — **three independent graph encoders**, one each for the donor
  molecule graph + acceptor molecule graph + product molecule graph. The current champion
  uses the attentive-pooling variant (`TripleGNNAttn`, introduced 07-20, with a robust
  advantage over default pooling on the scaffold-disjoint split — see `PROJECT_SUMMARY
  §3.11`).
- **Inference environment**: a single `envs/gnn_lite` (torch + torch_geometric + sklearn
  + xgboost + rdkit together), no cross-environment subprocess bridging needed (the homo
  project's ENSEMBLE72 model needs bridging because of an environment conflict; the cross
  project deliberately avoids the same problem here).
- **Uncertainty proxy**: `predict_dg.py` uses the standard deviation across the tabular
  half's 3 base learners (MLP/XGB-a/XGB-b) — a **cheap, direction-only** proxy, not the
  full pair-grouped bootstrap epistemic estimate (which is more expensive — see
  `score_round_active_learning.py --n-boot`).

---

## 6. Known gap (now closed): `G_gxtb` (aldehyde-side whole-molecule g-xTB free energy)

**Status as written (2026-09-07)**: the 09-06 full BDE descriptor-library rebuild only
computed the local bond descriptors BDE itself needs, and never recomputed the
whole-molecule `G_gxtb` defined in §3.2 — the source of `donor_G_gxtb`/`acceptor_G_gxtb`
(2 of the 260 champion features). After the rebuild only 2,718 of the 209,526 aldehydes
(backfilled from the old 42k local library) had a value; the other ~207k were empty.

This gap **once caused `predict_dg.py` to fail 100% on every new molecule pair** (a
coverage-sentinel field was chosen wrong and misjudged "didn't match the library" when
the match had in fact succeeded) — fixed; details in `PROJECT_SUMMARY_20260904.md §4.5`.
After the fix these two features are simply median-imputed as normal, and no longer make
the whole row evaporate.

**Backfill** (not a full geometry re-search — just two cheap single points on the
existing optimised geometry; see `pipeline/bde/recompute_aldehyde_gxtb.py` and the method
note in `PROJECT_SUMMARY §4.5`): **completed 2026-09-07.** Array `26432805` finished
(2,175/2,200 chunks; 25 known-unrecoverable), `pipeline/bde/merge_aldehyde_gxtb.py`
merged the results back into `aldehydes_all.csv` (filling only the missing `G_gxtb`,
leaving the existing 2,718 rows untouched) — coverage 1.3% → **99.07%** (207,580 /
209,526). The residual ~0.93% is from the 25 known-unrecoverable featurize chunks and is
accepted; not pursued further. A `predict_dg.py` 20-pair smoke test after the merge was
clean.

---

## 7. Key script / file index

| script/file | purpose |
|---|---|
| `pipeline/filter_smiles_v6.py` | aldehyde structure-library filter (§1) |
| `data/library/aldehydes_clean_v6.csv` / `aldehydes_rejected_v6.csv` | structure-library output |
| `cross_benzoin/cb_featurize.py` | end-to-end featurize entry point for a single molecule pair (geometry + local descriptors) |
| `pipeline/compute/conf_funnel_v3.py` (+ `v2`) | geometry search method (§3.1) |
| `pipeline/compute/gxtb_baseline.py` | the original hybrid-correction recipe for `G_gxtb` (§3.2) |
| `cross_benzoin/add_mordred_cross_products.py` | product-side mordred (§4.2) |
| `cross_benzoin/assemble_cross_training_table_v3.py` + `assemble_cross_training_table_combined.py` | table assembly (§4.3–4.4); a bug was fixed here on 09-07 (referenced in §6) |
| `cross_benzoin/prune_table_to_champion_features.py` | prune to the frozen feature list |
| `cross_benzoin/train_cross_ensemble.py` / `train_cross_gnn.py` / `gnn_architectures.py` | model training (§5) |
| `cross_benzoin/predict_cross_champion.py` | inference wrapper (model layer) |
| `cross_benzoin/predict_dg.py` | end-to-end deployment tool (user layer, see `PROJECT_SUMMARY §3.10`) |
| `pipeline/bde/recompute_aldehyde_gxtb.py` / `merge_aldehyde_gxtb.py` | `G_gxtb` backfill (§6) |
| `cross_benzoin/slurm/submit_backup_recovery_artifacts.sh` | home-backup manifest for the descriptor-library family (§2.3) |
| `data/cross_benzoin/cross_round8/scaffold_disjoint_8rounds_v1/models/feature_list.json` | the frozen 260-feature list (source of the §4.4 counts) |

---

*First version 2026-09-07. Technical reference, not a timeline snapshot — edit in place
when it goes stale, no need for a new version. For overall project progress / timeline
see `PROJECT_SUMMARY_20260904.md` and `RUN_LOG_20260903.md`.*
