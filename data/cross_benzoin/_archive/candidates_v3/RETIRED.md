# candidates_v3 — retired 2026-09-15

`candidates_v3` was a **constructed subset** of the cross space: ~2M unordered
(~4M directed) donor/acceptor pairs, picked by a bounded generator (MaxMin
diversity + category strata + a fixed reservoir) over the same 220,859-aldehyde
`aldehydes_clean_v6.csv` library this project still uses today. It was never
the real space — see `CHEMICAL_SPACE.md` §1 for the full critique (user,
2026-09-10) that led to the "flying dataset" (`cross_benzoin/chemical_space.py`)
replacing it as the canonical way to address the cross space.

**Retired here, not deleted** (PROJECT_PLAN.md §2.11 build order step 5):
`README.md`, `manifest.json`, `cross_benzoin_dG_candidates_v3_QA.xlsx`,
`representativeness_check/`, and the two `.csv.gz` files. Note: those two
`.csv.gz` files (`cross_benzoin_dG_candidates_v3.csv.gz`,
`cross_benzoin_aldehydes_v3.csv.gz`) are **not valid gzip** — 133-134 bytes
each, almost certainly a lost-in-the-2026-07-purge artifact (git-lfs pointer
or truncated stub) that was never restored, consistent with the project's
other purge-recovery gaps (`benzoin_dg_repo_moved_and_data_loss` memory).
`candidates_v3_pairs_with_scaffold_split.parquet` and
`inchikey_split_map.parquet` (referenced by several training/sampling
scripts, see below) are similarly **absent from disk** — also unrecovered
purge losses, not files this move touched.

**What actually moved, not just archived**: `aldehydes_with_scaffold_split.parquet`
(the one substantial, still-live asset in this directory — 220,524-aldehyde
Bemis-Murcko scaffold assignment, still v6-keyed and still actively used) is
now at `data/library/aldehydes_with_scaffold_split.parquet`, next to
`aldehydes_clean_v6.csv` — it was never conceptually part of the retired pair
pool, just filed alongside it; keeping it under a "candidates_v3" name/path
was the exact confusion a user flagged 2026-09-15. Active readers
(`cross_benzoin/build_aldehyde_index.py`, `pipeline/bde/build_scaffold_splits.py`)
and its writer (`cross_benzoin/rebuild_aldehydes_with_scaffold_split.py`) were
updated to the new path.

**Scripts referencing the now-archived (or already-lost) pair-pool files** —
paths updated to point here so they stay reproducible/greppable, but note most
were already unrunnable before this move (the files they need never survived
the purge): `sample_round{3,8,9,10}_from_candidates_v3.py`,
`sample_from_candidates_v3.py` (one-shot AL-round candidate selection scripts;
rounds 1-10 are done, cross AL is PARKED per PROJECT_PLAN §2.7 pending the
flying-dataset redo), `train_cross_delta.py`/`train_cross_ensemble.py`
(`SPLIT_MAP`, a legacy frozen-holdout eval path superseded by
`train_scaffold_disjoint.py`'s scaffold-disjoint split — not part of the
current champion retrain chain, see `DRAIN_RUNBOOK.md`), `train_cross_gnn.py`,
`score_round_active_learning.py`, `analysis/candidates_v3_representativeness.py`,
`rebuild_scaffold_disjoint_split.py`/`_v2.py`.

Going forward, address the cross space via `cross_benzoin/chemical_space.py`'s
`FlyingDataset` (donor/acceptor `ald_idx` pairs into the full 220,859² space),
not this directory.
