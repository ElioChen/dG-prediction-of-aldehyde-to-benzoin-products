> **CORRECTION (same day, later pass):** §6/§7 below cited the project's own prior claim that
> multi-conformer Boltzmann relabeling would drop the floor "~1.57→~1.38 kcal/mol." That number
> does not exist anywhere in the repo's actual data. The one completed pilot
> (`boltz_relabel_summary_20260626_1559.md`, n=95, K=5 conformers) found the OPPOSITE — relabeling
> made the frozen model's measured MAE worse (0.970→1.284), i.e. no evidence label noise was
> inflating the floor. Tier 3 as originally scoped in §7 should NOT be launched on that premise;
> see `descriptor-search-exhausted` memory's 2026-07-10 correction for the full trace. Tiers 1–2
> below are unaffected by this correction.

# Review of external diagnosis doc `benzoin_dG_project_context.md` (2026-07-10)

Reviewer: this session, working directly from repo code/data/job logs (not from the doc's
own claims). Every verdict below is grounded in a file path, report, or job-log timestamp
found in `/scratch-shared/schen3/benzoin-dg`. Champion model referenced throughout:
`gxtb_dft_correction_MORDREDSLIM271_BDEGXTB_20260706.joblib` — **test MAE 1.503, RMSE 2.257,
R² 0.875**, 275 features, uncertainty-routed 85%/15% split (confident MAE 1.252, routed MAE
2.923, ROC AUC 0.796).

## 0. Headline verdict

The doc's diagnosis of *where* the model hurts (sulfonyl/phosphorus subset) is correct and
independently confirmed. Its explanation of *why* (D4 dispersion polarizability extrapolation
feeding a toxic whole-molecule `P_int`) is a reasonable hypothesis, but **the specific fix it
recommends as highest-ROI (Action A, atom-local carbonyl-site `P_int`) was already tried on
2026-07-01 and came back null** — this isn't a fresh idea to prioritize, it's a completed
negative result the doc's author didn't know about. Several other "action items" are likewise
either already shipped (uncertainty routing, Action G) or already tested null (geometry
proxies, dispersion consistency). The genuinely open items are narrower than the doc implies,
and the single biggest quantified lever for the ~1.5 kcal/mol floor lies **outside** the doc's
scope entirely (multi-conformer relabeling of the hard tail).

## 1. Claims in the doc that are correct

| Claim | Verification |
|---|---|
| Sulfone/phosphorus subset has the largest error | Confirmed and quantified: `ald_sulfonyl`/`prod_sulfonyl` **11.24×** enriched, `has_P` **9.48×** enriched in worst-15% routed set (`REPORT_deep_error_analysis_champion275_20260707.md`) |
| `P_int` is the most important SHAP feature | Confirmed: rank 1 of 275, mean\|SHAP\|=1.849 (`REPORT_shap_mordredslim271_bdegxtb_20260707.md`) |
| Labels (r²SCAN-3c) and g-xTB baseline both carry D4 dispersion → no baseline/label dispersion mismatch | Confirmed correct, and actually **stronger** than stated: GFN2-xTB (the geometry-optimization stage) is itself a Grimme-group method with **native D4 dispersion built into its Hamiltonian**, and g-xTB is the same lineage. So all three pipeline stages — geometry opt, xTB baseline SP, DFT label — share D4, not just the two the doc checked. |

## 2. Claims / recommendations already falsified by completed experiments

### Action A — atom-local carbonyl-site `P_int` ("highest-value, do first")

**Already implemented and tested, 2026-07-01.** `cross_benzoin/analysis/add_morfeus_descriptors.py`
computes exactly what the doc proposes — `morfeus.Dispersion.atom_p_int` restricted to
ketC/carbC/CHO_C (the reaction site), plus per-atom pyramidalization — and
`pipeline/analysis/finalize_correction_morfeus.py` added these 9 features on top of the
champion-72 baseline, same 70:20:10 split, same ensemble:

```
baseline_72:        MAE 1.574
72_plus_morfeus9:    MAE 1.576   (atom-local disp_p_int + pyramidalization at ketC/carbC/CHO_C)
```

Delta (+0.002) is inside the measured noise band (1.571 ± 0.013 over 5 reshuffled seeds,
`REPORT_robustness_baseline72_20260702.md`). **No detectable effect, in either direction.**
(`REPORT_morfeus_augment_20260701.md`, jobs 24356687/24357492.)

This doesn't kill the underlying mechanism story, but it does mean the doc's #1 recommended
next step is not a next step — it's a null result the doc's author should have found and
didn't. Re-running it as originally scoped would waste a cycle.

### "Secondary issue: geometry" (GFN2 poorly describes S=O/P geometry → error in disguise)

Two independent pieces of existing evidence point the other way — toward an **electronic**
description problem, not a geometry problem:

1. **`hbond-not-product-error-driver` finding (2026-06-19):** at a *fixed* xTB geometry, the
   large product-side error on EWG/hypervalent-S benzoins does **not** track H-bond geometry
   (r = 0.06–0.15, R² < 0.025). The large-error molecules are non-H-bonded, and the paper
   trail explicitly concludes "xTB's electronic description of the hypervalent-S-substituted
   benzoin, independent of the H-bond" — i.e. the error survives with geometry held constant.
2. **The one completed row of the geometry-QC array** (`dftopt_36hard_v2/row_001.csv`, a
   sulfonyl-fluoride case, `S(=O)(=O)F`): re-optimizing fully at r²SCAN-3c instead of using
   the GFN2 geometry changes the DFT single-point energy by **−0.061 kcal/mol** (RMSD 0.335 Å
   heavy-atom displacement). That's noise-level, not a multi-kcal correction. n=1, so not
   conclusive on its own — see §3 for why it's still only 1 of 36.

Net: the existing (partial) evidence base leans toward "xTB/g-xTB gets the *electronics* of
hypervalent S/P wrong, geometry is a minor contributor," which is a **narrower and better-
supported** diagnosis than the doc's dispersion-and-geometry framing, but it isn't the doc's
"P_int/D4" story specifically — see the synthesis in §4.

### Action G — uncertainty / conformal prediction / route-to-DFT

Already in production, not a TODO. The champion bundle ships quantile-XGB (5%/95%) prediction
intervals, a route-to-DFT flag at the worst 15% by PI width, and a measured ROC AUC of 0.796
for whether that uncertainty actually tracks true error (confident-85% MAE 1.252 vs
routed-15% MAE 2.923). The doc frames this as future work; it should be struck from the list.

## 3. Genuinely open items (real gaps, not yet resolved)

### Action C — geometry QC on the S/P subset: infra exists, but **stalled at 4/36**

`pipeline/compute/dft_opt_bench.py` + the `dftopt_36hard_v2` SLURM array (job 24300065,
serial nprocs=1 after an earlier MPI-startup artifact was fixed, see `dftopt-36hard-fair-
rerun` memory) is the right tool and is already built. But of 36 hard S/P/B/Si/nitro
molecules submitted, **only 4 (`row_001/013/015/021.csv`) actually completed**; the other 32
log files are empty (never produced output — likely died on wall-time or were never
resubmitted after the fix). No summary report was ever generated
(`summarize_dftopt_36hard.py` exists but was never run to completion on v2). This is the one
item in the doc's action plan that is legitimately unfinished infrastructure, not a settled
question — worth finishing before drawing conclusions either way about geometry.

### Action B — raw per-atom D4 parameters (C6, static polarizability, CN, EEQ charge) as explicit features

Never literally tried. Two *adjacent* proxies came back null — atom-local `P_int`
(above, SASA-integrated dispersion potential) and Multiwfn ADCH/QTAIM per-atom charges
(2.44→2.45 MAE on a 2.5k subset, per `descriptor-search-exhausted` memory) — which lowers the
prior, but neither is the same physical quantity as raw dftd4 C6/α/CN/EEQ output. Also note
the ADCH/QTAIM null was only tested on a 2.5k-molecule subset, not the full 220k library
(`multiwfn-env-and-screen-gap` memory: the full-library screen never actually ran `--multiwfn`,
so this data doesn't exist at scale). Low-to-moderate expected value; cheap enough to be worth
a single targeted test restricted to S/P atoms specifically (not a full re-dump), but not a
priority given the adjacent nulls.

### Action D — explicit hypervalent-class flags as *model* features

The SMARTS-based tags the doc wants (`ald_sulfonyl`, `ald_has_P`, `ald_imine`, `ald_amide`,
etc.) **already exist** — they're computed in `deep_error_analysis_champion275.py` for the
enrichment-diagnosis table — but they were never added to the champion's `FEATS` list in
`finalize_correction_mordredslim271_bdegxtb.py` (that list only has coarse `g_has_S`/`g_has_P`
whole-molecule booleans, no S=O/P=O counts or oxidation-state proxies). This is real,
untested, and cheap: the tagging code and the training harness both already exist, this is a
merge + retrain, not new engineering.

### Action E — subset-only SHAP + interaction values on S/P

Not done. What exists is global SHAP (top-25 over the whole 220k) and a separate
tag-enrichment analysis (which subset of *molecules* is hard) — but nobody has recomputed SHAP
*restricted to* the sulfonyl/P/imine slice, or pulled SHAP interaction values to see whether
the model is trying (and failing) to isolate the hypervalent subclass, as the doc proposes.
Cheap — pure re-analysis of the existing model and existing test predictions, no retraining.

### Action F — ΔE(electronic) / ΔG(thermal) decomposition

Not done. Circumstantial support exists: **aliphatic MAE 1.870 vs aromatic MAE 1.327**
(`REPORT_deep_error_analysis_champion275_20260707.md`) — more rotatable bonds/flexibility
correlating with worse error is consistent with RRHO/low-frequency-mode entropy error being a
real contributor. But this is mechanistically the **same phenomenon** the project's own
already-planned (not-yet-started) multi-conformer Boltzmann relabeling targets — see §5. Don't
treat F as a separate workstream; it's the diagnostic justification for the relabeling work,
not an alternative to it.

## 4. Synthesis: a better mechanistic picture than the doc's

The doc's causal chain is: *D4 polarizability manifold → distorted `P_int` → error on S/P*.
The atom-local `P_int` test directly falsifies the middle link — restricting the SASA
integration to the reaction site didn't clean up the signal, so `P_int`'s *specific* dispersion
mechanism is not obviously the toxic pathway. Combined with the H-bond memory's independent
finding (fixed-geometry electronic-structure error, not an NCI/dispersion effect) and the
single geometry-QC data point (relaxing geometry barely moves the energy), the more
defensible read of the evidence in hand is:

> xTB/g-xTB's **electronic-structure Hamiltonian** (not specifically its D4 dispersion term,
> and not primarily geometry) is parameterized against normal-valence references and
> mis-describes hypervalent S(VI)/P bonding generally — charge distribution, orbital energies,
> WBOs — and `P_int` merely *correlates* with this because sulfonyl/phosphorus groups also
> happen to be large, polarizable, SASA-heavy substituents. Fixing the D4/`P_int` pathway
> specifically (Action A, B) is unlikely to help much further; the error is closer to
> "xTB-family method error on hypervalent main group," which is a data/label problem (need
> more/better reference calculations on that subclass) more than a feature-engineering one.

This also matches the project's own standing conclusion in `descriptor-search-exhausted`
(three orthogonal descriptor families tried, floor didn't move) that the ~1.5–1.57 kcal/mol
floor is **data-limited, not feature-limited**.

## 5. Additional finding not in the doc: GNN+tabular stacking is dangling, unresolved

Not mentioned in the reviewed doc, found while checking the "don't pursue GNN further"
claim. On 2026-07-07 a GNN+tabular stacking test (`gnn_dual_qm_champion275_ensemble.py`)
crashed twice (pickle/cache bugs) before completing on a third attempt, producing
`ensemble_stack_champion275_20260707.csv`: best blend at w_gnn=0.35–0.4 gives MAE 1.425 vs
tabular-only 1.476 (**−0.051**, i.e. real-looking on its face). **But** the GNN test split and
the tabular test split only overlap on 6,601 of 21,911 molecules (~30%) — the comparison is on
a partial, non-representative subset, not the full held-out test set. The proper fix
(`gnn_reeval_matched`, job 24494009, full ID-matched re-evaluation) was **cancelled by SIGTERM
mid-run and never resubmitted**. This is a real, cheap-to-close loose end: one clean GPU job
would confirm or kill the −0.05 stacking gain properly.

## 6. Is further *simulation* (new DFT/xTB compute) actually needed?

Two things, and only two things, in this whole review justify new physical simulation rather
than re-analysis of data already on disk:

1. **Finish the stalled `dftopt_36hard_v2` array** (32/36 molecules never completed) — cheap,
   infrastructure already built, resolves the geometry-vs-electronics question on n=36 hard
   S/P/B/Si cases instead of the current n=1 anecdote. This is the legitimate part of the
   doc's Action C.
2. **Multi-conformer Boltzmann relabeling of the hard tail** — already quantified in prior
   work (`descriptor-search-exhausted`/`gxtb-dft-session-handoff` memories) as dropping the
   floor from ~1.57 to ~1.38 kcal/mol in quadrature, at ~5× DFT cost **on the tail only**, not
   the full 220k library. This is the single largest lever identified anywhere in this
   project, is independent of and larger than anything in the reviewed doc, and directly
   subsumes the doc's Action F (entropy) hypothesis — it's not been started, "needs user
   go-ahead" per its own memory note.

Everything else actionable (Action D hypervalent flags, Action E subset SHAP, the GNN-stacking
clean rerun, a narrow S/P-only Action B test) is feature-engineering / re-analysis on data
**already in hand** — zero new DFT/xTB compute. Recommend doing all of those first (cheap,
days not weeks) before committing to the ~5× DFT-cost relabeling campaign, since they're
nearly free and may shift what the relabeling campaign should even target.

## 7. Recommended order of operations

**Tier 1 (no new compute, days):**
1. Subset-only SHAP + interaction values on sulfonyl/P/imine/amide slice (Action E).
2. Add the existing SMARTS hypervalent tags as literal model features, retrain, check if MAE
   moves on the S/P subset specifically (Action D).
3. Clean, ID-matched rerun of GNN+tabular stacking to settle the −0.05 question (§5).

**Tier 2 (small new compute, decide after Tier 1):**
4. Finish the `dftopt_36hard_v2` array (32 remaining molecules) — settles geometry vs.
   electronics with real n.
5. Narrow Action B test: raw D4 C6/polarizability/CN/EEQ *only* on S/P atoms, if Tier 1's
   subset SHAP still points at dispersion-adjacent features on that slice.

**Tier 3 (the real lever, needs an explicit go-ahead on cost):**
6. Multi-conformer Boltzmann relabeling of the hard tail.

## 8. On the GitHub upload

Noted as a follow-on task, not started here. Once Tier 1 (and possibly Tier 2/3) lands, the
natural contents for a new repo are: `src/benzoin_dG/`, `pipeline/` (compute + analysis +
models, trimmed of `__pycache__`/scratch), the champion `.joblib` bundle(s), the key
`REPORT_*.md`/`REPORT_*_zh.md` pairs, and `FILE_MAP.md`. Will need your input on repo name,
public/private, and whether to include the full 220k-row data tables (likely too large for a
normal GitHub repo — would want Git LFS or an external data host, with only sample/schema
data committed directly) before actually pushing anything.
