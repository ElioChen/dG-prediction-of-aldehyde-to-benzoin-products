# Lab Journal — Benzoin ΔG Prediction

> Reasoned, narrative account of every operation: **what was attempted, why it
> was the right step (the principle), what broke and its root cause, the
> result.** Figures inline where they clarify. Updated continuously; each day's
> section is closed in the evening.
>
> Started 2026-09-10 at the user's request. The terse dated log is
> `RUN_LOG_20260903.md`; the component map is `PROJECT_PLAN.md`.
> Earlier history (before 2026-09-10) is in `RUN_LOG_20260903.md` and the
> `HANDOFF_*.md` / `PROJECT_SUMMARY_*.md` files — not backfilled here in detail.

---

## 2026-09-10

### Context at start of day
Two campaigns running in parallel:
- **cross Tier B** (self-consistent B97-3c relabel of all 35,528 cross pairs) —
  the "cheap-baseline lever": if swapping the Δ-learning baseline g-xTB → B97-3c
  lowers the residual scatter (pilot said std 4.32 → 1.11), the champion MAE
  could drop below the 2.9 kcal label-noise floor for the first time. ~68–72%
  done through the day.
- **homo from-scratch relabel** — user directive: rebuild the homo (A+A) model
  on the *full* library with recomputed DFT labels (the ~189k labels lost in the
  2026-07 purge), using a single XGBoost + a single GNN.

### Attempt 1 — homo relabel via SP-on-archived-geometry ("fast route")
**Principle.** The full homo DFT labels are physically gone (verified: not in
git, not in old or restored scratch, not in any of 87 home-backup tarballs).
But the **geometries** (GFN2-opt, 2026-09 BDE-featurize archive) and the **xTB
thermal corrections** (stored in `products_all.csv` / `aldehydes_all.csv`)
survived. So we don't need the expensive full pipeline (conformer search + xTB
Hessian + DFT) — just a **DFT single point on the archived geometry**, reusing
the stored thermal:
`G_lvl(species) = E_lvl(SP on archived geom) + (G_xtb − E_el_xtb)_stored`.
This is exactly how the original 2026-06 homo labels were made (the whole 219k
library in ~18 h). ~10–20× cheaper than a self-consistent regen, and
geometry-consistent with the descriptor library.

Built: `build_homo_sp_manifest.py` (184,052 pairs), `homo_sp_from_geom_worker.py`
(extract prod + ald xyz from the `geom.tar.zst`, run r2SCAN-3c + B97-3c per
species, `dG = (G_prod − 2·G_ald)·627.509`), `submit_homo_sp.sh` (CHUNK 20 →
9,203 array tasks). Launched 2 arms (fat_genoa + genoa).

### Problem 1 — smoke test: all 3 pairs `sp_fail`, ORCA returned None
**Root cause.** `thermo_orca.calc_orca_sp` writes its scratch as `orca_sp/`
*next to the input xyz*. The worker ran several SPs concurrently on the **same**
extracted geometry file (product r2SCAN-3c and product B97-3c both point at
`geoms/xyz_p000062.xyz`), so they shared one `orca_sp/` dir and clobbered each
other's `input.inp` / output → both fail.
**Fix.** `_sp` now copies the geometry into its own `tempfile.mkdtemp()` per
call — the pattern the validated `dft_sp_from_geom.py` already used. A direct
`calc_orca_sp` on one archived geom then ran clean (E = −1575.285 Eh, 6.7 min).
Committed. Smoke #2 passed: 3 QC pairs, repro vs stored 30k label +0.06 / +8e-6
/ +2.9 kcal — no systematic offset.

### Problem 2 — `merge_homo_sp.py` on 498 partial shards: verdict RED, 27 % of dG garbage
Ran the merge/QC on the ~4,200 pairs done so far.
- `repro_r2scan` **median 1.3 kcal** (excellent — most pairs reproduce the
  surviving DFT labels), but **std 6 × 10⁵**, and **27 % of |dG| > 200 kcal/mol**
  (up to ±10⁶).
- The median being fine means most extractions are correct; a large tail is
  catastrophically wrong.

**Root cause (diagnosed by pulling the worst rows).** `_extract` cached each
extracted member under `dst / member.replace("/","_")` — i.e. by the **member
basename** `xyz/p000062.xyz`. But `pNNNNNN` is a **per-chunk local index**, not
a global id: chunk_1523's `p000062` and chunk_1900's `p000062` are different
molecules. A task processes 20 *shuffled* pairs from random chunks; the first
pair to want local index 62 populates the cache, and every later pair wanting
`p000062` from a *different* chunk gets served that first molecule's geometry.
Result: E computed for the wrong molecule → energies off by ~10× → dG blows up.
The atom-count ratio E_prod/E_ald stayed ≈ 2.0 even for garbage rows because
both wrong molecules were roughly the right *size* class — so a size check alone
wouldn't have caught it.

**Fix.**
1. Cache key now includes the chunk directory: `<chunk_NNNN>__<member>`.
2. After extraction, **verify the xyz atom count == the SMILES heavy+H count**
   (`_n_heavy_h`); reject on mismatch. Defense-in-depth — also catches archive
   gaps and any future member confusion.
Cancelled both arms, **wiped all ~430 contaminated shards**, restarting.
~7 h of compute discarded — but the merge QC caught it before any model
training, which is the point of running the QC on partial data.

### Problem 3 — smoke #3 (fixed extract): still one row with dG = 28,293 kcal/mol
20-pair smoke of the chunk-keyed + atom-checked worker. 19 rows clean (dG 1–7
kcal, right molecules), **1 row garbage**. E_prod/E_ald ratio was 2.0 and the
atom count matched — so geometry was *correct* this time.

**Root cause.** `dG = (E_prod + thermal_prod) − 2·(E_ald + thermal_ald)`. With
E_prod ≈ 2·E_ald, the garbage must come from **thermal**. The manifest's
`thermal_ald = G_xtb − xtb_energy` from `aldehydes_all.csv`. Inspecting that
column: `thermal_ald` ranges **−84 Ha to +87 Ha** for ~8 % of rows (physical
thermal is +0.05 to +0.9 Ha), and some rows share an identical `G_xtb` value
across two chemically different aldehydes. So `aldehydes_all.csv`'s `G_xtb`
column is **id-misaligned for ~8 % of rows** — a defect baked in by the
2026-09-06 BDE featurize (the same bad values are in the home backup, so it is
not a purge corruption). `xtb_energy` itself looks sane.

No clean source for the missing thermal (`products_all.csv`'s `G_donor` was
populated from the same broken column).

**Fix.** The worker now **bounds-checks the stored thermal** (`0 ≤ thermal ≤
1.6 Ha`); if it fails, it **recomputes the thermal from the archived geometry**
with a fresh `xtb --ohess` (`_thermal_job`, folded into the same job pool). A
new `th_src` column records `stored` vs `recomp` per species. This is cheap
(~5–15 min per small aldehyde) and self-consistent (same GFN2 level as the
stored values). Smoke #4 (`26554004`) validating.

### Two-track split
`split_homo_sp_manifest.py` scans every archive and partitions the 184,052
pairs:
- **archived track — 179,431 (97.5 %)**: geometry present → `homo_sp_from_geom_worker`
  (SP on archived geom; bad thermal self-heals via `_thermal_job`).
- **regen track — 4,621 (2.5 %)**: geometry missing (chunks ~1400–1830 were only
  partially archived) → `submit_homo_regen.sh` runs the full
  `rec_homo_relabel_worker` (conf_funnel_v3 + `--ohess` fresh thermal +
  r2SCAN-3c + B97-3c + g-xTB per species). Slower per pair but only 4.6k of them.
At merge, check the regen subset's label distribution against the archived
subset for a systematic offset (the regen route's earlier 2-pair smoke showed
~−5.5 kcal vs stored labels — may be genuine conformer improvement or a
protocol difference; if real, carry an `is_regen` flag).

### Cluster sharing (user directive)
User: "don't occupy all nodes — share with the other project (NHC)." The NHC
kinetics campaign runs on the same account / QOS. Actions: Tier B fat_genoa arms
throttled to %8 (running tasks scancelled — resume-safe, ~1 pair lost each,
freed ~250 fat_genoa nodes), rome arm to %48; homo relabel restarts at modest
concurrency and scales up only as Tier B vacates. Saved as memory
`share-cluster-nodes`. NHC's `nhc-gsp-fine-*` confirmed running normally again.

### Deliverables today
- `PROJECT_PLAN.md` — this project's full component map (§1 dG Simulation /
  §2 Prediction Optimization / §3 Catalyst Space / §4 infra), done/missing per
  component, with the principle behind each. Created.
- `LAB_JOURNAL.md` — this file. Created.
- Memory `working-style-journal-plan-understand` — the new standing requirements
  (journal + plan + understand every step, stop chasing speed).

### Open threads / to close this evening
- Smoke #4 (`26554004`) result → if clean, relaunch archived + regen tracks.
- Rebuild the homo-SP campaign monitor with the new job ids.
- cross Tier B still ~72 %; deprioritized, trickling.

### Principle notes worth keeping
- **Why the QC on partial data matters.** Running `merge_homo_sp.py` on ~2.5 %
  of the campaign is what caught the 27 % contamination after ~7 h instead of
  after ~3 days + a trained model. Cheap partial QC on any long labelling run.
- **Why "E_prod ≈ 2·E_ald" is a good but insufficient check.** For a homo pair
  the product is two aldehyde units minus water-equivalent, so its electronic
  energy is close to twice the aldehyde's. It catches wildly wrong pairings but
  not a wrong-molecule-of-the-right-size, nor a bad thermal. Need the atom-count
  check *and* the thermal bounds check *and* the repro-vs-stored-label QC.
- **Recovered-CSV data quality is the recurring hazard.** Three of today's
  problems trace to defects in files rebuilt after the purge (float ids earlier,
  now cross-chunk index reuse in geom archives, and id-misaligned `G_xtb`).
  Treat every stored value from `homo_v6/*_all.csv` as suspect until
  bounds-checked.
