# Gap analysis: codex's cross-benzoin session vs. the actual repo state (2026-07-14)

Companion file: `REPORT_codex_gap_analysis_20260714_ZH.md` (中文版).

## What codex actually did

A separate Codex CLI session (Windows, `ElioChen/dG-prediction-of-aldehyde-to-benzoin-products`
fork) worked purely at the **GitHub/data-generation** layer, with no access to this
HPC environment's xTB/g-xTB/ORCA binaries or SLURM. Across four turns it:

1. Generated an unlabeled directed cross-benzoin candidate table three times
   (v1: 120k aromatic-only pairs → v2: 1M pairs / all chemistry → v3: 2M pairs
   / 4M directed rows), each time correcting course on user feedback (don't
   pre-filter aliphatic; scale is too small; need full aldehyde-space coverage).
2. Gave a genuinely useful, well-sourced ML-methodology writeup: energy
   decomposition for multi-fidelity Δ-learning (`G_high ≈ E_high,SP + (G_GFN2 −
   E_GFN2,SP)`), species-additive `δE_P − δE_A − δE_B` correction architecture,
   homo-as-diagonal-of-cross framing, four cross-validation scenarios (homo
   diagonal / new combination / single-side extrapolation / double-side
   extrapolation), and a case for OOF stacking + uncertainty-aware gating over
   a hand-tuned 40/60 blend.
3. Opened PR #1 with the v3 dataset (Git LFS, ~216 MB), a streaming pair-chunker
   (`prepare_pair_chunks.py`), a product-enumeration/QC tool
   (`prepare_product_manifest.py`), a role-aware descriptor policy
   (`DESCRIPTOR_POLICY_CROSS.md`), and an execution roadmap (`NEXT_STEPS.md`).
4. Correctly flagged that the repo's `benzoin-dg` CLI is still homo-only and
   avoided overclaiming cross-inference capability in the README rewrite.

The methodology is sound and mostly *not* new to this project — Δ-learning,
uncertainty routing, and (as of the previous session) GNN+tabular stacking
were already built and validated here before codex ever looked at the repo.
Codex's real value-add is the **candidate-space generation and cross-specific
planning docs**, which this project's HPC-side pipeline had not produced yet.

## Gaps found and fixed this session

1. **The PR branch would have deleted the just-confirmed GNN-stacking
   promotion if merged.** Codex's branch (`agent/add-cross-benzoin-v2-data`)
   was cut before `main`'s `cf870d7` commit ("fix+promote GNN+tabular stacking
   to full-library scale", job 24578348). A naive merge/rebase would have
   silently dropped `pipeline/analysis/promote_gnn_stacking_full_library.py`,
   the ChemBERTa/GINE pure-SMILES baselines, and the 218k-row
   `products_dG_corrected_GNNSTACK_w40_20260714.csv` — the actual current
   production-scoring artifact. Fixed by merging `main` into the PR branch
   (clean, no conflicts — the two branches touched disjoint files) and
   re-pushing. PR #1 is now additive-only relative to `main`.
2. **README/status text was stale relative to the confirmed result.** It
   still called the 1.427 GNN+tabular blend an unverified candidate and named
   the 1.503 tabular model "current best," even though `cf870d7` had already
   confirmed the blend on a held-out test set. Updated, and — importantly —
   made explicit that **the confirmed 1.427 blend is not yet in the
   installable inference path**: `src/benzoin_dG/models/` only contains the
   tabular joblib, no GNN checkpoint, so `predict_dG()` still returns the
   1.503 model. That packaging gap is real and unresolved.
3. **g-xTB failures were silent.** `cb_featurize.py` set an explicit `error`
   string for GFN2-side failures (`dG_failed`) but left `dG_gxtb_kcal` simply
   blank on g-xTB failure with no tag distinguishing "product SP failed" from
   "a cached reactant's g-xTB value was missing." Added `gxtb_sp_failed` /
   `gxtb_dG_failed_reactant` error tags (codex had flagged this as a
   correctness suggestion during its read-only review but never had the
   binaries to exercise the code path).
4. **The v3 4M dataset is real but still zero-labeled**, and it lives behind
   Git LFS, which is not installed on this HPC filesystem. Rather than stand
   up LFS here, this session sampled directly from the same source library
   (`data/library/aldehydes_clean_v6.csv`) that v3 was built from — see below.

## The actual gap this session closes: real cross-benzoin product labels

Before this session, **every cross (donor ≠ acceptor) product ΔG in this
project was zero** — codex's 4M-row release is candidates, not labels, and
all prior compute here (`homo_v6`, the GNN/tabular models, the stacking
result) was diagonal-only (`A, A`). Codex's own roadmap (`NEXT_STEPS.md`
Phase 2) calls for exactly this: "reuse the existing aldehyde GFN2/g-xTB
table by canonical SMILES; calculate only new products."

`cross_benzoin/select_cross_pilot_sample.py` draws a **300 unordered / 600
directed** pilot, stratified evenly across all 6 unordered category
combinations of `{aromatic_carbo, aromatic_hetero, aliphatic}` (Cartesian
donor/acceptor swap kept as two separate directed rows each), restricted to
molecules whose aldehyde-side GFN2 and g-xTB free energies are already cached
in `data/cross_benzoin/homo_v6/aldehydes_all.csv` (220,526/220,859 molecules,
99.9% of the library). `submit_cb_featurize_array.sh` gained
`ALD_CACHE`/`REQUIRE_CACHE_COMPLETE` passthrough so the run pays only for the
genuinely new part — product conformer search, GFN2 optimization/frequency,
and g-xTB single point — with zero aldehyde recompute (verified: the array
did not raise the hard-fail-on-cache-miss check).

**COMPLETE: SLURM job 24607515 (array 0–5, genoa, 6 nodes) finished in ~48
minutes, 600/600 rows, zero errors.** This gives the first-ever real
cross-benzoin ΔG values in this project (previously exactly zero cross,
donor≠acceptor, rows had ever been computed — all prior compute here was
homo/diagonal-only):

- `dG_xtb_kcal`: n=600, mean −10.62, sd 3.69, range [−22.16, +7.13] kcal/mol.
- `dG_gxtb_kcal`: n=600, mean +2.74, sd 4.37, range [−13.43, +29.19] kcal/mol.
- Structural integrity: 300/300 unordered pairs have exactly 2 directed rows
  (no duplicates, no missing directions).
- **AB/BA orientation check, full sample**: `|dG_xtb(A→B) − dG_xtb(B→A)|`
  has mean 2.64, median 2.16, max 9.37 kcal/mol across all 300 pairs; only
  1.7% of pairs (5/300) have a near-zero (<0.05 kcal/mol) direction delta.
  This is a strong, quantitative confirmation (not just spot-checked
  examples) that donor/acceptor direction is a first-order effect on ΔG,
  not metadata — see the new methods doc §7 for the mechanistic reason
  (different carbon becomes ketone vs. carbinol depending on direction).
- By category-pair (`dG_xtb_kcal`, n=100 each): aliphatic/aliphatic −11.96
  (sd 3.29), aliphatic/aromatic_carbo −10.81 (2.54), aliphatic/
  aromatic_hetero −11.96 (4.08), aromatic_carbo/aromatic_carbo −8.82 (2.88),
  aromatic_carbo/aromatic_hetero −9.79 (3.63), aromatic_hetero/
  aromatic_hetero −10.36 (4.33). Aliphatic-involving pairs are more
  exergonic than aromatic-only pairs by ~2–3 kcal/mol on average — plausible
  and consistent with known steric/electronic trends, not an artifact (no
  category shows a degenerate/constant distribution).

While watching this run, a genuine (if here non-triggered) bug was found and
fixed in `submit_cb_featurize_array.sh`: the array's resume check only
tested "is `products.csv` non-empty," rather than "does it have a row for
every pair in the chunk." `cb_featurize.py` flushes rows incrementally as
each product finishes, so any task that *does* hit its time limit or gets
preempted/requeued before finishing its chunk — plausible at larger scale or
with slower molecules (bigger/more flexible substrates, Multiwfn enabled,
node contention) — would leave a partial, non-empty `products.csv` that the
old check would have silently treated as fully done on any resubmit,
permanently losing the un-run pairs with no error or warning. Fixed: the
resume check now compares completed rows against the chunk's true expected
count and only skips on an exact match; a partial chunk is flagged and
reprocessed in full (row-level resume inside `cb_featurize.py` itself
doesn't exist yet, so a resumed chunk redoes its product-side compute, but
not the aldehyde side, which stays cache-hit).

Re-run the validation with:

```bash
cd /scratch-shared/schen3/benzoin-dg && python3 -c "
import csv, glob
rows = []
for f in sorted(glob.glob('data/cross_benzoin/cross_pilot_v1/chunk_*/products.csv')):
    rows.extend(csv.DictReader(open(f, encoding='utf-8')))
print(len(rows), 'rows,', sum(1 for r in rows if r['error']), 'errors')
"
```

## What's still genuinely missing (priority order)

1. **Real cross labels beyond this ~600-row pilot.** Once the pilot validates
   (non-trivial ΔG spread, correct AB≠BA regiochemistry, low error rate),
   scale to the diversity/uncertainty-driven few-thousand-row set
   `NEXT_STEPS.md` Phase 2 calls for — not all 4M, per both codex's and this
   project's prior "don't compute everything, use active learning" conclusion
   ([[delta-mae-noise-floor]]).
2. **Package the confirmed 1.427 GNN+tabular blend into `predict_dG()`.** The
   research artifact exists (full-library CSV); the installable path does not.
3. **Role-aware (donor/acceptor) descriptor computation.** The *policy* is
   written (`DESCRIPTOR_POLICY_CROSS.md`); no cross-specific descriptor table
   has actually been assembled yet, because there was no labeled cross data to
   assemble it from until this session's pilot.
4. **Scaffold-/molecule-disjoint cross test splits**, per codex's Phase 4 and
   this project's own established practice ([[data-split-721]]) — not
   meaningful until (1) produces enough rows to split.
5. **PR #1 body is stale** (still describes the pre-merge, pre-pilot state).
   This session could not update it: `gh` is not authenticated in this HPC
   environment (only SSH git push works, under the `ElioChen` identity). The
   branch and its commits are current; only the PR *description* text needs a
   manual refresh or a `gh auth login` in an interactive session.
