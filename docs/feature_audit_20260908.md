# 260-feature champion schema — distribution audit (2026-09-08)

Triggered by: `n_CHO` is near-constant because `aldehydes_clean_v6` is filtered to
exactly one formyl group. Audited all 260 frozen champion features
(`cross_round10/scaffold_disjoint_10rounds_v1/models/feature_list.json`) on the
r1-10 training table (35,528 rows). Raw per-feature stats:
`docs/feature_audit_20260908.csv`.

## Verdict

**Only the 3 `n_CHO` features are degenerate. Everything else is healthy.**
No zero-variance features, no >20% missingness, no infinities, nothing else
near-constant even at a 95% threshold.

### Drop (schema v2, 260 → 257)

| feature | n_unique | top value | top-value frac | std |
|---|--:|--:|--:|--:|
| `donor_n_CHO` | 3 (1,2,3) | 1 | **99.71%** | 0.055 |
| `acceptor_n_CHO` | 3 (1,2,3) | 1 | **99.70%** | 0.056 |
| `product_n_CHO` | 3 (0,1,2) | 0 | **99.41%** | 0.079 |

The ~0.3% non-1 donor/acceptor rows are pairs whose RDKit formyl SMARTS count
disagrees with the library's single-aldehyde filter (di-aldehyde edge cases); the
~0.6% non-0 `product_n_CHO` rows are benzoin products retaining an unreacted CHO
from a di-aldehyde donor. In all cases the feature carries essentially no variance
and trees never split on it.

**Empirical A/B** (single-XGB, `{n_estimators:300, max_depth:3, lr:0.05}`,
scaffold-disjoint holdout n=448, 5 seeds):

| schema | holdout MAE | sd |
|---|--:|--:|
| 260 | 2.525 | 0.027 |
| 257 (no `n_CHO`) | 2.532 | 0.040 |

Δ = 0.006 kcal, far inside seed noise. **Removing `n_CHO` is verified harmless.**

## Not degenerate — keep (documented so they are not re-flagged)

**≥50% exactly zero** — genuine structural counts, real signal (many benzoin
substrates are flat polyaromatics):
`donor/acceptor_AlRings` (66% zero), `donor/acceptor_NumStereocenters` (63%),
`donor/acceptor_HBD` (60%), `donor/acceptor_ArHetRings` (51%).

**Heavy right tail / row-level outliers** (a handful of rows each, <1%; a data-
quality issue in specific geometries, not a bad feature — the Tier B geometry
regen may clean these):
- `wbo_CC_new` — new C–C Wiberg bond order, median 0.95, **max 1.63** (a single
  bond can't have WBO 1.6; ~a dozen products with a pathological funnel_v3 geom).
- `mulliken_carbC` — carbinol-C Mulliken charge, median 0.095, max 0.363.
- `acceptor_mulliken_CHO_C` — carbonyl-C charge, median 0.207, min −0.027.
- `product_mordred_RPCS` — CPSA-family (relative positive charge surface),
  median 16, max 911. Expected heavy tail for this mordred descriptor, not a bug.

Recommendation for the post-Tier-B rebuild: after regenerating geometries, re-run
this audit; if `wbo_CC_new` / `mulliken_*` tails persist for the same pairs,
winsorise at p1/p99 rather than dropping.

## Applying schema v2

- `cross_benzoin/assemble_cross_training_table.py`: `n_CHO` removed from
  `RDKIT_FEATS` (2026-09-08) → future assemblies emit 257, not 260.
- Staged 257-list: `data/cross_benzoin/feature_list_257_no_nCHO_v2.json`.
- The **shipped r1-10 champion is unchanged** — it loads its own frozen 260-list
  from its model dir. Schema v2 takes effect at the post-Tier-B champion retrain
  (which changes labels + baseline anyway, so it is the right moment).
