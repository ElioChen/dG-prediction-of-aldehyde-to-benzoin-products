# Conformer generation/optimisation — assessment (2026-06-19)

Scope: evaluate the conformer search/opt used across the project (legacy
`_rank_conformers`, robust funnel v2/v3, CREST) **and** its impact on the in-flight
DFT single-point validation. TL;DR: the method choice for *training labels* is already
settled (funnel_v3), but the **DFT-SP validation + the running full-library DFT job
still use the legacy unguarded `_rank_conformers`, which poisons ~8.5% of geometries
and inflates the apparent xTB–DFT gap**.

## 1. The three approaches (what each is, established verdict)

| method | sampling | topology guard | cost | role |
|---|---|---|---|---|
| **legacy `_rank_conformers`** (thermo_orca, `--conformer default`) | ETKDG `_auto_nconfs`: **1 conf for ≤3 rotbonds**, modest otherwise; GFN-FF/MMFF→GFN2 opt, pick lowest | **none** | cheap | what the 220k screen + DFT-SP run use |
| funnel v2 | deterministic ETKDG (seed42, 1 thread) + RMSD prune + ~2× denser GFN-FF prescreen | none | ~2–4 min/mol | intermediate; broken-prone |
| **funnel v3** = v2 + topology guard | dense deterministic + drop any conformer whose heavy-atom graph ≠ input | **yes (RDKit perception)** | v2 + 1 perception | **production winner for labels** |
| CREST 3.0.2 (`conf_crest.py`, gfn2//gfnff) | GFN-FF metadynamics + internal topology check | yes (native) | 8–11 min/mol | robust **cross-check**, under-samples big polycyclics |

**Settled conclusion (findings_2026-06-14):** don't switch engines — *guard the funnel*.
- v2→v3 guard is **surgical**: MAD 0.31 kcal, fires on ~2.7% broken-topology catastrophes.
- legacy→funnel_v3 is a **major overhaul** (MAD 3.86, 76% labels move >1 kcal, 7% >10).
- Re-labelling training set with funnel_v3 dropped the Δ-model noise floor **2.68 → ~2.19
  (−18%)**, RMSE falling more than MAE (tail/catastrophe gains); ridge≈xgb≈gpr after =
  residual is near irreducible label noise.
- CREST under-samples large fused polycyclics where dense ETKDG finds 3.8–4.2 kcal lower
  minima → kept as second-opinion, not production engine.
- **Open production gap:** funnel_v3 has no size-aware conformer cap → a 66-heavy-atom
  product grinds ~8 h. Add `n_confs` down-scaling above ~55 heavy atoms before rollout.

## 2. NEW finding — the DFT validation inherits the legacy unguarded search

The 1% DFT-SP validation and the **running full-library DFT job (23992496)** both submit
`--conformer default` (= legacy `_rank_conformers`, **no topology guard**). Running the
funnel_v3 topology fingerprint over all 2192 validation benzoin geometries:

- **187 / 2192 = 8.5%** of benzoin product geometries are **broken-topology**
  (bond formed/broken/fragmented during GFN-FF/GFN2 opt) — exactly the failure mode
  funnel_v3 was built to remove.
- Broken rate is **2× higher in EWG molecules** (15.3% vs 7.8% benign).
- Broken geometries carry **mean |residual| 28.4 vs 13.8 kcal** for intact — broken
  topology ~doubles the apparent xTB–DFT disagreement.
- **11/20 of the worst outliers are broken-topology**; 42% of |resid|>30 are broken vs
  only 5% of |resid|<10.

**=> A large share of the "xTB fails catastrophically on EWG" outliers (+85–120 kcal)
are conformer-SEARCH artifacts, not genuine xTB electronic-structure failures.** This is
a confound on the EWG/decomp/H-bond analyses, which used the as-published labels.

## 3. Decomposing the discrepancy by confound (filtering cascade)

| filter | n | MAE | bias | Pearson r |
|---|---|---|---|---|
| FULL (as-published) | 2192 | 15.04 | −13.64 | 0.574 |
| drop broken-topology | 2005 | 13.80 | −12.60 | 0.651 |
| drop EWG | 1977 | 12.96 | −11.73 | 0.779 |
| **drop EWG + broken (genuine electronic)** | 1823 | **12.03** | **−11.02** | **0.851** |

Two distinct error sources, both real:
1. **Conformer-search noise** (broken topology): removable with funnel_v3 → r 0.574→0.651.
2. **xTB electronic failure on hypervalent-S/EWG**: removable only by excluding/those
   motifs → r →0.779.
3. After both, the **genuine** xTB↔DFT relationship is **r=0.851**, but still bias −11 /
   MAE 12 — a real, systematic over-exergonicity of GFN2-xTB even on clean benign cases.

## 4. Recommendations

1. **Filter, don't trust raw outliers.** Use `benzoin_topology_flags_20260619.csv`
   (broken flag per molecule) to exclude/flag the 8.5% broken benzoins from every
   downstream xTB–DFT comparison; re-issue the EWG/decomp/H-bond conclusions on the
   intact set. The "genuine" gap is r=0.851 / bias −11, not r=0.574 / bias −13.6.
2. **Decide the conformer engine for the full-library DFT run (23992496, ~25% done):**
   - *If its purpose is to validate the screen as-is* → keep legacy (apples-to-apples with
     the 220k screen, which also has this 8.5%). But then label every output with the
     topology flag so broken cases don't enter any fit.
   - *If its purpose is gold-standard DFT ΔG for training/correction* → switch to
     **funnel_v3** conformers (+ size cap). This is a real fork worth an explicit call;
     8.5% poisoned labels at full scale = ~18.7k molecules.
3. **Production inference / any new labelling:** funnel_v3 + size-aware `n_confs` cap;
   CREST as periodic cross-check on suspicious (high-σ) predictions.
4. **dft_opt_bench (②) caveat:** it starts from `_rank_conformers`, so it tests local
   DFT relaxation, NOT wrong-basin/broken-topology recovery. Pair it with the topology
   flag: a broken xTB start won't be "fixed" by a local DFT opt.

Artifacts: `analysis/benzoin_topology_flags_20260619.csv`. Relates to
[[crest-conformer-search]], [[conformer-search-noise]], [[dft-sp-workflow-handoff]],
[[hbond-not-product-error-driver]].
