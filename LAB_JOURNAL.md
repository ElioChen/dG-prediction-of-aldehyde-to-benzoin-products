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

### 2026-09-10 (evening cont.) — 4th homo-SP contamination source + project re-framing

**Problem 4 — smoke #3 (thermal recompute in place): still 27 %→ small % garbage.**
The bounds check `0 ≤ thermal ≤ 1.6 Ha` was too loose. A concrete miss: aldehyde
`O=Cc1c(O)ccc(C(F)(F)F)c1Br` (18 atoms) had a *stored* thermal of **1.229 Ha**
(should be ~0.13). 1.229 < 1.6 so it passed, and it enters ΔG as `−2·thermal_ald`
→ a ~1500 kcal/mol error.
**Root cause.** The Gibbs thermal correction scales ~linearly with atom count
(~0.003–0.012 Ha/atom for these organics: 13-atom aldehyde 0.072, 26-atom 0.136).
An *absolute* bound cannot separate a legitimate large-molecule thermal from a
misaligned small-molecule one; a **per-atom** bound can.
**Fix (commit `b4706df`).** `_th_ok(v, n_atoms)` requires `0.001 ≤ v/N ≤ 0.020`
(N = heavy+H from the SMILES). A recomputed thermal that is itself out of band is
rejected. Plus a hard gate: any `|dG_r2scan| > 200 kcal/mol` → `error=dG_implausible`
at the worker, so garbage never reaches a shard. Verified the bound rejects the
1.229 Ha value (0.068/atom) and passes the legit 0.13 (0.0072/atom). Smoke v4
(`26554455`) running.
*Note:* a residual ~1–3 % of pairs (e.g. a di-boronic-acid benzoin with a huge
conformer surface) will still have noisy single-conformer labels — those are the
"library includes uncomputable/hard examples by design" cases the user flagged;
the merge `repro_r2scan` tail + the assembly-time |dG| clip handle them, we do
not chase them.

**Principle note (recovered-CSV data quality).** Four distinct defects in
post-purge / post-featurize `homo_v6/*_all.csv` files have now bitten:
float ids, cross-chunk geom-index reuse, gross `G_xtb` misalignment, and subtle
`G_xtb` misalignment. **Every stored xTB quantity is now bounds-checked before
use** (per-atom thermal band, |dG| plausibility gate). This is cheaper than
trusting the file and finding out at model-training time.

**Project re-framing (user input, acted on today).**
1. *homo* library = `aldehydes_clean_v6.csv` (220,860), filtered — and it
   **deliberately contains molecules that cannot be optimized/DFT'd**. That
   attrition is a library property, not a bug; a "can't compute" is a valid
   recorded outcome.
2. *cross* library was **built wrong**. `candidates_v3` (~1.24 M pairs) is an
   arbitrary constructed subset, not the space. The **real cross space is
   220,860² ≈ 4.88 × 10¹⁰ ordered aldehyde pairs**.
3. Do not materialize the SMILES. Build a **"flying dataset"** — a lazy,
   pair-addressed virtual dataset: freeze a canonical `ald_idx` for the 220,860
   aldehydes, keep per-aldehyde caches (geometry / QM / Mordred / BDE / xTB
   energetics), and generate a pair's product SMILES + 260 features + baseline +
   split *on demand* from the two aldehyde caches. Spec written:
   **`CHEMICAL_SPACE.md`** (frozen index → re-keyed caches → `pair(i,j)` read
   API → labels/splits → migrate the 35,528 existing labels → retire
   `candidates_v3`). Build order is deliberately small-step with a bit-level
   feature-reproduction check before anything downstream trusts it.
4. **Past AL (rounds 1–10) is parked; cross AL will be redone** over the flying
   dataset. The 35,528 DFT labels are retained as data; the r1-10 blend
   (MAE 2.215) is now a *reference* model, not the frozen forward line.
5. **Catalyst Space** filled in from the sibling GitHub repos
   (`nhc-benzoin-pipeline` = TS_CC/TS_CN + microkinetic ee(t);
   `nhc-benzoin-active-learning` = multi-objective pool AL over ~13 M
   stereoisomers, objectives = Kozuch–Shaik energy span + |ee|, both censored;
   `stereo-catalyst-engine` = Bayesian-opt catalyst design; `nhc-pkah-predictor`
   = azolium pKaH gate). This project = the *substrate* axis; those = the
   *catalyst* axis; the **substrate × catalyst integration is the open gap**
   (`PROJECT_PLAN.md` §3.3), and whether this project takes it on needs user
   scoping.

**Deliverables today (cont.).** `PROJECT_PLAN.md` updated (v6-library caveat,
§2.11 flying dataset, §2.7 AL parked, §3 rewritten). `CHEMICAL_SPACE.md` created.
Memory `working-style-journal-plan-understand` (journal + plan + understand every
step) and this journal + plan themselves.

### 2026-09-10 (evening cont. 2) — QOS corrected, homo SP launched, day close

**QOS understanding corrected (user).** `MaxJobsPU=128` is **per partition** ->
rome 128 + genoa 128 = a 256 combined ceiling. Rule: never run to it -- target
<= ~100 combined (both projects) per partition, <= ~200 across both, own
steady-state <= ~90; fat partitions <= ~75% occupancy. `CLUSTER_SHARING.md` +
`_ZH` rewritten. **Found my Tier B sitting at 128 on rome** (old arm 26467946
throttled to %50 but not converged, 16 h walltime) -> `scancel --state=RUNNING`,
re-throttled both Tier B rome arms to %30 + %25 (~55 combined). Could not
identify the NHC session among the 4 current peers to send the proposal; it is
in the repo doc, complying on my side.

**smoke v4 (per-atom thermal bound + dG gate).** 9 rows, 8 usable, 0 with
|dG| > 200 -- the +/-1e6 garbage is gone. **resid_b973c median -5.21 +/- MAD
0.61** -- exactly the pilot's -5.24 +/- ~1.1 level-of-theory scatter -> the
r2SCAN-3c and B97-3c SPs are internally consistent on every row. The residual
-26 / -49 kcal `repro_r2scan` rows have *clean* resid_b973c -> not a silent SP
failure, a genuine archived-geometry-vs-2026-06-label conformer mismatch on a
hard/flexible molecule (di-boronic-acid benzoin). Those are the v6 library's
by-design hard cases; `merge_homo_sp.py` excludes them from `_labels.csv` via
two label-free / QC gates: resid_b973c 6-MAD-sigma band + |repro_r2scan|>15 on
the QC rows. Not chased further -- the model sees the residual noise, the
conformal PI widens, `dg_high_sigma` flags it at inference.

**homo SP campaign LAUNCHED** on **genoa** (not fat -- per CLUSTER_SHARING):
`26555994` archived (%60, 179,431 pairs) + `26555995` regen (%12, 4,621 pairs).
Monitor `bmiabhrw6`. Worker now robust vs 5 contamination classes; merge catches
the 6th.

**Day close.** Docs delivered: `PROJECT_PLAN`(+ZH), `CHEMICAL_SPACE`(+ZH),
`CLUSTER_SHARING`(+ZH), `LAB_JOURNAL`(+ZH). Memories
`working-style-journal-plan-understand`, `share-cluster-nodes`. Compute: cross
Tier B ~72% on rome (deprioritized, ~55 concurrent); homo SP relabel launched
clean on genoa after 4 worker fixes + 2 merge QC gates. Open threads: send
cluster rules to NHC when identifiable; on homo SP drain -> merge + assemble +
single-XGB/GNN; on Tier B drain -> DRAIN_RUNBOOK; the flying-dataset build
(`CHEMICAL_SPACE.md` s8) is the next design task.


### 2026-09-11 — session resume, real-throughput ETA, flying dataset steps 1-2

**Resumed an interrupted morning session.** Working repo is
`benzoin-dg-restored`, not the stale `benzoin-dg` shell the environment
defaults to. Found uncommitted work from before the interruption --
`assemble_homo_standalone_table.py --full-library` (the mode HANDOFF_20260910
sec1.3 marked "to write") and `merge_homo_sp.py`'s two-directory glob -- smoke
tested both against the partial shard output already on disk (11,362 rows x
200 features; merge verdict GREEN, no offset), committed (`a0cdc17`).
Rebuilt the three session monitors (guardian, Tier B drain, homo SP drain) --
they do not survive a session end, this is the third time writing them from
the handoff's rebuild recipe.

**"Should we use fat nodes to go faster?"** Checked `sinfo -s`: both
`fat_rome` and `fat_genoa` were at 0 idle nodes cluster-wide -- fat wouldn't
even queue faster right now, on top of already costing 50% more (the reason
we moved off it last night). The real lever was idle `rome` capacity (121
idle nodes, nothing of ours or NHC's contending there) -- raised Tier B's two
rome arm throttles 29+25 -> 50+40 (uses the `ArrayTaskThrottle` update that
looks like a permission error but works, per the recurring cluster quirk).

**homo SP real ETA came in much worse than promised.** Measured actual
throughput instead of trusting the 09-10 "~2 days" estimate: archived track
~45 shard/h, regen ~4.1 shard/h -- both extrapolate to **~7-7.5 days**, not 2,
because last night's QOS-fairness correction cut concurrency from a planned
~250 down to a genoa-only 72. Diagnosed *why* it's stuck there: our genoa
usage (homo_sp 60 + homo_regen 12) plus NHC's own `nhc-gsp-*` jobs (~54) are
already sitting on the shared 128-per-partition QOS cap -- genoa has no more
room, and fat_genoa is also full. Same fix as Tier B: rome has real slack and
essentially zero NHC presence, so launched two top-up arms there
(`26573745` archived 3000-8971%25, `26573746` regen 300-770%6, ranges chosen
to avoid re-doing genoa's already-dispatched low indices). Projected ETA after
the boost: ~5.4d archived / ~4.8d regen (~09-16/17). Next lever (not yet
pulled, waiting for the drain notification so it doesn't collide): once Tier B
drains, redirect most of its ~90 rome slots into homo_sp archived -- could pull
the ETA back to drain +2-3d.

**Flying dataset (CHEMICAL_SPACE.md sec8), steps 1-2, while both campaigns
wait on compute.** This is exactly the "no compute, understood carefully"
work item 3 of PROJECT_PLAN sec6 flagged as next. Step 1: froze
`data/chemical_space/aldehyde_index.parquet` (220,859 rows -- corrected an
off-by-one that had propagated through both plan docs, the CSV has 220,860
*lines* including the header, not 220,860 aldehydes). Found and reused,
rather than recomputed, the Bemis-Murcko scaffold already computed for this
exact library in `candidates_v3/aldehydes_with_scaffold_split.parquet` (the
same file `pipeline/bde/build_scaffold_splits.py` reused for BDE) -- verified
its `id` column is positionally == `ald_idx` on the *full* 220,524-row
overlap (0 mismatches, exact raw-SMILES match), not a sample, before trusting
the merge. Step 2 turned into a verification rather than a rebuild:
`homo_v6/aldehydes_all.csv`'s existing `id` (after `qc.norm_id`) already
equals `ald_idx` 1:1 across its full 209,526 rows, 0 orphans either side --
it was already correctly keyed, just undocumented as such. Real finding along
the way: that file's `smiles` column is not reliably canonical (1.24% differ
from the freshly-computed canonical form by representation only, e.g.
Kekulized vs lower-case aromatic) -- future cache joins must key on `ald_idx`,
not SMILES string equality, which is exactly the kind of drift the frozen
index exists to remove. `git add -f`'d the parquet (gitignored by default,
same discipline as everything else that must not repeat the purge loss).

--- Snapshot (2026-09-11): both compute campaigns still in flight (Tier B
~78%, homo SP slow but boosted); flying dataset build order at step 2/6;
PROJECT_PLAN + CHEMICAL_SPACE (+ZH) updated to match. Next design step: 3 --
`chemical_space.py`'s `pair(i,j)` feature path, with the mandatory ~20-pair
verification against the current champion table. ---

**Same evening, continued (not journaled at the time):** step 3 done --
`FlyingDataset.pair(i,j)`, verified bit-exact on the two deterministic lazy
tiers (RDKit-2D, `interaction_*`, `product_smiles`) against 20 random pairs
from the round-10 table; donor/acceptor QM matched 72% within 0.2% (the rest
is the already-characterized aldehyde-recompute-fidelity effect, not a bug).
One real bug caught: an early verification draft resolved a row's
(donor, acceptor) address via `pair_key` lookup, which silently picked the
wrong row when a `pair_key` paired with both role orderings -- fixed by
resolving from the row's own donor_smiles/acceptor_smiles directly. Then an
ablation answering the user's "why 260 features, on which objects" question:
dropping the 91 product-side QM/mordred columns (product geometry isn't
lazy, CHEMICAL_SPACE.md sec5) costs **+1.00 kcal MAE** (2.544 -> 3.548,
+39.5%, 5-seed pooled sd ~0.02 -- real, well-powered) on the round-10
champion table. Still well above the g-xTB baseline (5.037), so a legitimate
cheap first-pass filter, but not a substitute for the full pipeline -- the
flying dataset cannot be made fully lazy without a real accuracy cost.
Reframes "redesign the descriptors" from feature selection to a compute-cost
question.

### 2026-09-12 / 09-13 — no session

No commits either day.

### 2026-09-14 — Tier B drains: b973c champion breakthrough, deployed (reconstructed 2026-09-15)

*Not journaled same-day; reconstructed from commit messages (`50d37b7`,
`3a894bc`, `273d74f`, `e798755`) and CHAMPION.md content while resuming
2026-09-15 -- flagged as reconstructed rather than presented as a live
account.*

**Homo aldehyde chemotype clustering diagnostic**, requested to inform the
flying-dataset split design: MiniBatchKMeans (k=150) on ECFP4 over the full
220,859-aldehyde library, joined against the in-flight homo SP labels (43.0%
coverage at the time), 3 split regimes x 4 methods. Finding: the R²~0.03
cluster/scaffold group-mean signal seen under random CV is a **leakage
artifact** -- collapses to ~0 under both cluster-disjoint and
scaffold-disjoint GroupKFold, while raw fingerprint structure keeps a small
real signal (R²~0.04-0.05) across all three. Confirms the flying-dataset
split must be structurally grouped, and that 2D chemotype alone explains
little of the homo target (consistent with the homo-active-relabel-null-
result redirect). Mid-session infra fix: a PCA-based structure bracket hung
because this venv's numpy runs BLAS matmuls unvectorized on this node's CPU
generation (76000x2048 benchmarked at 14s vs <1s expected) -- replaced with
pure-numpy ECFP4 bit-folding (2048->128, data-independent, no leakage risk),
finished in minutes. Delivered as a genuinely-executed notebook
(`notebooks/cross_benzoin/homo_aldehyde_cluster_dG_analysis.ipynb`), the
first under the 2026-09-14 user preference for notebook-format analysis.

**Tier B (Rec-1 cheap-baseline lever) drained** at 98.9% coverage (35,136/
35,528), GREEN QC (B97-3c residual std 0.916 vs g-xTB's 3.785, ratio 0.242 --
matches the pilot's 0.26 at full scale). `DRAIN_RUNBOOK.md` steps 1-6:
retrained the champion pipeline on the B97-3c-baseline Δ-target (schema v2,
257 features). Result, same frozen n=448 holdout: single-XGB 0.632, MLP+XGB
ensemble 0.603, GNN 4-seed average 0.531, **blend (w_gnn=0.85) 0.528** --
vs the g-xTB champion's 2.215, a **-76%** drop. Far past the "~1.0-1.5 =
breakthrough" bar PROJECT_PLAN had guessed at on 2026-09-10 -- the real gain
was much larger. Mechanism: the Δ-model only has to learn `r2SCAN-3c -
B97-3c`, already tight (std 0.916) before any ML, because B97-3c starts much
closer to the label than semiempirical g-xTB does.

**Wired into `predict_dg.py`** the same session (`cb_featurize.py
--with-b973c` computes the new baseline for genuinely new pairs). Testing
this end-to-end surfaced two bugs predating this session, blocking BOTH the
g-xTB and b973c from-scratch paths: (1) `submit_predict_dg.sh` read a
`features.csv` `cb_featurize.py` has never written (writes `products.csv`);
(2) the 2026-09-08 n_CHO feature-audit drop had silently broken assembling a
table for ANY new pair against the still-260-feature deployed schema since
that patch landed (`calc_rdkit` still computes n_CHO; only the assembler's
column *selection* had dropped it) -- fixed via `include_ncho=True`.

**Real-pair precision check** (fresh GFN2-xTB + ORCA from scratch, not a
training-table replay, through the actual `submit_predict_dg.sh` ->
`predict_dg.py` pipeline): on 2 clean test-split pairs, b973c MAE 0.34 vs
g-xTB MAE 2.50 -- both landing right on their respective holdout MAE, an
independent confirmation the b973c gain isn't a training-table artifact. A
3rd pair (zwitterion-prone amino-acid-like donor) failed **both** models
similarly -- a shared upstream feature/geometry issue, not b973c-specific;
the g-xTB path's `dg_high_sigma` flag caught it correctly. **CHAMPION.md
flipped**: r1-10-b973c is now the champion + deployed default, r1-10 (g-xTB)
demoted to documented fallback. Known gap flagged at session end: no
calibration artifact built for b973c yet.

### 2026-09-15 — resume after the gap, b973c calibration + a real seed-averaging deployment bug

**Resumed cold** (no HANDOFF or journal entry past 09-11 tail). Reconstructed
09-12/13 (no session) and 09-14 (dense, unjournaled) from `git log` + commit
bodies before doing anything else -- see the two entries just above. Homo SP
campaign confirmed still healthy: archived 6268/8972 shards (69.9%), regen
654/771 (84.8%), combined throughput ~50 shard/h (archived is the
bottleneck) -> **ETA ~2.3 days (~09-17/18)**, close to the 09-11 projection.

**Picked up CHAMPION.md's flagged gap**: built
`cross_benzoin/predict_dg_calibration_b973c.json` by generalizing
`build_predict_dg_calibration.py` (was hardcoded to the g-xTB champion's
table/model paths; now takes `--table --model-dir --gnn-dir --blend-w-gnn
--baseline-col --label-col`, verified byte-for-byte equivalent on the
original g-xTB config modulo 1e-13 float noise before trusting the refactor
on anything new).

**Real bug found while doing it, not cosmetic**: computing the b973c
holdout MAE via the documented "Load" recipe
(`CrossBenzoinBlendPredictor.load(..., gnn_dir=seed4)`) gave **0.544**, not
the champion's documented **0.528**. Root cause: 0.528 is the 4-seed-GNN-
averaged number (`gnn_seed_ensemble_r10_b973c_result.json`'s sweep,
`w_gnn=0.85`), but the 2026-09-14 session's `predict_dg.py` wiring only ever
loaded a single seed dir (seed4 alone, `w_gnn=0.70` from that seed's own
`metadata.json`) -- CHAMPION.md's own "Load" code sample was internally
inconsistent (claimed w_gnn=0.85 "from the GNN metadata.json" while pointing
at a single dir whose metadata actually says 0.70). Same class of gap as the
09-08 g-xTB seed-ensemble finding (2.215->2.137) -- except there the
09-08 session correctly judged the gain not-yet-significant and did NOT
adopt it (left as documented history, untouched here); here the 09-14
session HAD already decided to adopt the 4-seed number as champion, just
never wired the code to match.

**Fixed properly, not patched around**: `CrossBenzoinBlendPredictor.load()`
now accepts a *list* of `gnn_dir`s and averages each member's own
de-normalized prediction (verified per-seed `ym`/`ysd` genuinely differ
slightly, ~0.003-0.009, so this has to be per-member, not shared stats);
`blend_w_gnn` is *required* explicitly when passing a list (each seed's own
metadata.json only tunes itself alone -- guessing which sweep row applies
would silently ship the wrong weight, same "explicit not silently wrong"
discipline as the rest of this file). Re-verified: 4-seed load reproduces
MAE 0.5284, matching the sweep to 5 decimals. `predict_dg.py`'s `--gnn-dir`
now accepts a comma-separated list + `--blend-w-gnn`; `build_predict_dg_
calibration.py` mirrors it. Rebuilt the calibration artifact against the
*correct* 4-seed champion (first build, against the wrong single-seed
config, was caught and redone before it could ship): split-conformal 90%
interval half-width **±1.21 kcal** (vs g-xTB's ±5.24 -- 4.3x tighter,
tracks the MAE gap), coverage 0.908 test / 0.894 validation.

**One more real finding, not just plumbing**: the same 7 baseline-risk
SMARTS motifs that clearly flag g-xTB baseline failures (flagged rows'
|baseline error| 6.35 vs 4.81 not-flagged) do NOT carry the same meaning for
b973c -- flagged rows there have higher blend MAE (0.80 vs 0.50) but FLAT
|baseline error| (4.94 vs 4.98). B97-3c itself isn't failing on those
structures; something else about them is harder for the Δ-model. Documented
in both CHAMPION.md and predict_dg.py's own printed summary so a caller
doesn't over-read `baseline_risk=True` as "route to DFT" for the b973c path
the way it correctly means for g-xTB.

`predict_dg.py`/`predict_cross_champion.py`/`build_predict_dg_calibration.py`
/`CHAMPION.md`/`PROJECT_PLAN.md` all updated together; full pipeline
re-verified in-process (load 4-seed champion -> predict -> attach
calibration columns -> motif flags) on 20 real holdout rows before treating
this as done, not just "imports cleanly."

**Same day, continued: flying dataset build order step 4.** With homo SP the
only thing left waiting on compute (~2.3d), picked up the queued design task
(PROJECT_PLAN §6 item 2). `FlyingDataset.pair()` now returns first-class
`label` / `label_col` / `split` / `baseline_gxtb_kcal` / `baseline_b973c_kcal`
for the cache-hit tier (honest `None` for any address not in `known_pairs`).
The real design question was column-name resolution: the g-xTB-era champion
table's label is `dG_orca_kcal` and it has no `dG_b973c_kcal` column at all,
while the current champion (the b973c Tier B table) uses `dG_r2scan_kcal` as
its true label and carries both baselines -- one fixed column name would
silently break on whichever table wasn't tested. `LABEL_COL_CANDIDATES`/
`SPLIT_COL_CANDIDATES`/`BASELINE_COLS` try a name list in order per field,
so the module works unmodified against either table generation. Extended
`verify_chemical_space_pair.py` with a `step4` tier (resolved independently
in the test, not just re-calling the module's own logic -- an order bug
would otherwise self-confirm) and a `--table` flag, then ran it against
*both* tables (not just the default): 20/20 pairs, 80/80 step4 fields exact
on each run, `label_col` correctly reported `dG_orca_kcal` on one table and
`dG_r2scan_kcal` on the other. That cross-table run is what actually
exercises the resolution logic -- testing only the default table would have
missed a same-column-name-on-both-tables bug entirely.

**Same day, continued: step 5, user chose "宽做" (the broad option).** Asked
the user what still needed a decision; offered a narrow vs. broad scope for
step 5 (just wire the read API vs. actually retire candidates_v3's files and
its 13 dependent scripts). User picked broad -- but first asked a sharper
question that reframed the whole task: "candidates_v3 跟你有什么关系？你用的
不是v6版本的醛数据库吗" (what does candidates_v3 have to do with you, aren't
you using the v6 aldehyde library?). Answering it properly required actually
checking what's IN that directory rather than trusting the "retire
candidates_v3" phrasing at face value -- and the check found the user's
instinct was right to probe: `candidates_v3/aldehydes_with_scaffold_split.parquet`
(10MB, still actively read by `build_aldehyde_index.py` AND
`pipeline/bde/build_scaffold_splits.py`) is a v6-aldehyde-level asset that
was never part of the deprecated pair pool -- it just happened to be filed
under a "candidates_v3" path, which is exactly the confusion the question
surfaced. A blind "move the whole directory" would have broken both an
active BDE-pipeline read and the flying dataset's own index-builder.

**Full dependency audit before touching anything**: grepped all 26 files
referencing "candidates_v3" and sorted them into three buckets -- (1) active
readers of the scaffold parquet (2 files, real risk), (2) the actual
"~1.24M-pair arbitrary subset" this project's critique (CHEMICAL_SPACE.md
sec1) is about, referenced by 13 sampling/training/analysis scripts, and
(3) prose-only historical comments (no code risk). Checking bucket 2 found
something unexpected: the pair-pool files it names
(`candidates_v3_pairs_with_scaffold_split.parquet`, `inchikey_split_map.parquet`)
don't exist on disk at all, and the two `.csv.gz` files that are present
(`cross_benzoin_dG_candidates_v3.csv.gz`, `cross_benzoin_aldehydes_v3.csv.gz`)
are 133-134 bytes and not valid gzip -- this data was already lost in the
2026-07 purge and never restored (consistent with cross AL being PARKED).
So bucket-2 scripts were already unrunnable before this move touched
anything; retiring them was cleanup and path-hygiene, not breaking a live
dependency. Confirmed `train_scaffold_disjoint.py` (the actually-deployed
retrain script, DRAIN_RUNBOOK.md) only imports two env-overridable constants
from `train_cross_delta.py` at module level -- no candidates_v3 file is read
at import time or by that script's own logic, which uses the training
table's own `new_scaffold_split` column instead (that's precisely *why*
`train_scaffold_disjoint.py` exists -- it replaced the leaky
candidates_v3-split-based training).

**Executed**: `aldehydes_with_scaffold_split.parquet` moved to
`data/library/` (next to `aldehydes_clean_v6.csv`, where it actually
belongs); its 2 active readers + 1 writer repointed there, verified by
resolving the path constants and re-running `verify_chemical_space_pair.py`
(still 10/10, 15/15 across two runs). Everything else in the directory
(README, manifest, QA xlsx, representativeness_check/, the two dead
`.csv.gz` stubs) archived to `data/cross_benzoin/_archive/candidates_v3/`
with a `RETIRED.md` explaining what moved, what was already lost, and why.
13 dependent scripts' paths updated to the archive location (so they stay
reproducible/greppable even though most were already non-functional); the 5
one-shot AL-round sampling scripts + the representativeness analysis got a
prepended retirement notice pointing at the flying dataset as the
replacement. `train_cross_delta.py`'s `SPLIT_MAP` got an inline comment
explaining it's legacy (superseded by scaffold-disjoint splitting, not part
of the current champion chain) rather than silently repointing a path with
no explanation.

**Also did the other half of step 5** ("migrate the labels", not just
"retire candidates_v3" -- both named in the same build-order line): wrote
`build_labeled_pairs.py`, which freezes
`data/chemical_space/labeled_pairs.parquet` -- the 35,136 usable-labeled
pairs (b973c Tier B table, the current superset), joined to
`aldehyde_index.parquet` by **InChIKey** (not the SMILES-canonicalization
route `chemical_space.py`'s own lookup uses for an arbitrary table -- this
script controls its own source columns, so it can use the exact key and
skip that whole class of ambiguity): 35,136/35,136 resolved, 0 duplicate
addresses. Gave `FlyingDataset._build_known_lookup` a fast path: a
`known_pairs` table carrying `donor_ald_idx`/`acceptor_ald_idx` columns
directly (this new canonical table does) skips the SMILES round-trip
entirely. Verified end to end: 100/100 random rows from the new table
exact-match on label/split/both baselines via the new path; the old
SMILES-based path re-run clean against both champion tables afterward (no
regression). `LABEL_COL_CANDIDATES`/`SPLIT_COL_CANDIDATES`/`BASELINE_COLS`
extended so the canonical table's own column names (`label`, `split`, ...)
resolve first, ahead of the older per-table-generation names.

--- Snapshot (2026-09-15, end of day): homo SP the only compute in flight,
ETA ~09-17/18, nothing else to do there but wait + monitor. Cross-benzoin
deployment thread has no known open gaps. Flying dataset build order now
5/6: steps 1-5 done, step 6 (point redone cross AL / Goal-3 screening at it)
is gated on the acquisition-strategy redesign, a design choice for the user,
not more flying-dataset engineering. Open decisions
surfaced to the user this session and still awaiting their call: Catalyst
Space / NHC-repo integration scope, whether Rec-2's -0.11 finding is still
worth re-testing now that the b973c floor changed, and the redone-AL
acquisition strategy itself. ---

## 2026-09-16

**Status-check only, no code changes.** Resumed to a user request to report
progress; everything actionable right now is gated on either the homo SP
compute drain or a user decision already surfaced 09-15 (Catalyst Space
scope, Rec-2 retest, AL redesign), so this was a health check, not new work
-- writing filler tasks to look busy would be worse than reporting "still
waiting."

Progress since the 09-15 09:41 snapshot: archived track 6268/8972 ->
**7350/8972 shards (81.9%)**; regen track 654/771 -> **771/771 (complete,
766/771 `.done`-marked, remaining 5 mid-finalize)**. Combined throughput
matches the ~50 shard/h archived-side bottleneck already diagnosed
09-10/09-11 -- no new throttling needed. `sacct` across all 6 job-array IDs
(genoa + rome top-up + fat_rome x2, archived and regen) shows 20,825
COMPLETED / 159 RUNNING / 27 PENDING / only 3 CANCELLED+ (negligible,
consistent with ordinary requeues) -- no failure pattern to chase. Quota
checked given the standing [[scratch-disk-quota-risk]] concern: home
61.8%/67.9% (GiB/inodes), scratch1 16.2%/31.6% -- healthy, no risk of a
silent final-write loss right now. Remaining archived shards (1622) at
~50/h implies **~32h more, i.e. still tracking the 09-17/18 ETA**, no
revision needed.

Not yet run: the drain sequence (`merge_homo_sp.py` -> QC -> `--full-library`
assembler -> single XGB + single GNN, PROJECT_PLAN §6 item 1) -- correctly
gated on the archived track actually reaching 100%, not just close. Will
fire it without waiting for another prompt once shards complete, per the
standing autonomous-advance authority ([[handoff-routine-and-autonomy]]).

**Same day, continued: user asked to try chemprop and a condensed reaction
graph (CRG) for the dG GNN leg** (both new comparison points against the
deployed TripleGNN, 0.556 single-seed / 0.531 4-seed / 0.528 blend champion
numbers), plus pushed a Rec-2 early-look with the partial homo table already
on disk.

**Rec-2 early look (provisional, homo SP still draining)**: re-ran
`assemble_homo_standalone_table.py --full-library` against the 136,874-row
partial `merge_homo_sp.py` output -- found and fixed a real gap first: the
script never joined aldehyde-side mordred (`donor_ald_mordred_*`/
`acceptor_ald_mordred_*`, 67 of the 257 champion columns), silently training
on a reduced schema on both the 30k and full-library paths since it was
written; its own docstring's "champion has none" claim was already wrong at
round8. Fixed (merge `aldehydes_mordred_slim102.csv`, add to both
`ald_lookup` and `feat_cols`). Wrote `homo_cross_joint_tabular_v2.py` (Task
C's methodology at 257-feat + b973c scale, `+naive_merge_weighted` condition
to directly test FINDING.md's dilution-at-6:1-scale warning). Result on the
116,740-row provisional table (homo:cross 5.18:1): naive_merge (unweighted)
beats cross_only by 0.046 (0.617->0.572), AMBER
(likely <1 bootstrap SE at n=448) -- and *unweighted* beat the
down-weighted variant, the opposite of what the dilution warning predicted.
Provisional only; full table needed before treating this as a real answer.

**chemprop**: found the project's shared chemprop envs (`envs/gnn`,
`envs/bde_gnn` under `/gpfs/scratch1/shared/schen3/envs/`) are ALL corrupted
post-purge (empty `bin/`), but `/home/schen3/venv/bde_gnn` (home, rebuilt
~09-02) works (torch 2.13+cu130, chemprop 2.2.0; installed missing pyarrow).
Wrote `train_cross_gnn_chemprop.py` using chemprop v2's native
`MulticomponentMessagePassing`/`MulticomponentMPNN` (3 separate
`BondMessagePassing` blocks over product/donor/acceptor, `shared=False` to
match TripleGNN's 3-separate-encoder design) + the 257-feat schema as `x_d`
(chemprop convention: x_d/y read from the first component's dataset only,
`collate_multicomponent` -- attached to the product component). CPU smoke
test first, then one real GPU run (gpu_a100, 21 min): single-seed test
MAE **0.600** (n=448) -- ties MLP+XGB ensemble (0.603), beats single-XGB
(0.632), behind TripleGNN's single-seed (0.556). No seed-ensembling
attempted (chemprop was the secondary ask; CRG got the follow-through).

**Real bug found while building the chemprop script, not fixed there**:
`train_cross_gnn.py`'s own split (`train_cross_delta.pair_split_labels()`)
reads candidates_v3's `SPLIT_MAP`, retired 2026-09-15 -- the file does not
exist, so the function returns `None` unconditionally now, which the
consuming code turns into "every row -> train_extra" (empty val/test). Every
existing champion GNN checkpoint predates the retirement so is unaffected;
this would only bite the *next* re-run. Fixed later in the day (see below).

**CRG (condensed reaction graph) -- the deeper build.** Key insight that
avoids needing an external reaction atom-mapper: benzoin coupling (2 RCHO ->
R-CO-CH(OH)-R') is atom-economical, so the product SMILES already physically
contains every donor+acceptor atom as intact substituent trees -- a CRG here
doesn't need to union three mol objects, just (a) locate the 5-atom reactive
core (ketC/ketO/carbC/hydO, no separate hydH node since this project's
graphs are implicit-H) via one SMARTS (`[CX3](=O)[CX4][OX2H1]`, validated
97.5-97.6% unique-match across two independent samples of 500/2000 rows),
and (b) BFS-split every other atom by which side of the new ketC-carbC bond
it sits on -- both purely from the product graph's own connectivity, no
donor_smiles/acceptor_smiles cross-referencing needed. Formal net atom
mapping derived and documented in `crg_builder.py`'s docstring (donor
CHO_C/O -> ketC/ketO unchanged; acceptor CHO_C/O -> carbC/hydO, C=O order
2->1; donor's CHO_H migrates to become the new hydroxyl's implicit H) --
mass-balance-consistent, not a mechanistic claim about the actual NHC
Umpolung intermediates. `crg_builder.py` is pure RDKit, unit-tested against
a toy molecule (side/core tags and the new-bond edge flag all verified
exactly right) before touching the real table.

`train_cross_gnn_crg.py`: single `Enc()` (same GINEConv block as
TripleGNN's per-branch encoder, so this isolates the connected-vs-disconnected
representation effect, not a feature-set change) over the one condensed
product graph + the 257-feat x_d channel. Same `new_scaffold_split`-based
split as the chemprop script (same reason: avoids the just-found
pair_split_labels landmine). CPU smoke test, then real GPU runs.

**First single run: 0.559 (n=432, ~2.5% of rows dropped for no/ambiguous
core-SMARTS match)** -- ran a fairness check before getting excited (the
champion blend's own MAE on that *same* 432-row subset is 0.5295, vs 0.5284
on the full 448, so the subset isn't secretly easier -- the comparison is
valid). Then genuinely over-interpreted a "seed 1/2/3" follow-up: `SEED=$s
sbatch ...` does not propagate through this cluster's sbatch (site default
`--export` doesn't inherit the shell var the way `SEED=$s sbatch` implies),
so three "different seeds" silently all ran seed 0 again -- caught because
the *reported* seed0 number kept changing between checks (CUDA isn't fully
deterministic even at a fixed seed). User pushed back twice on the resulting
n=4 read ("seed是否太少" / "还是数据太少") -- rightly: those 4 numbers
(0.559/0.564/0.604/0.593) looked like a wide, worrying spread, but n=4 can't
tell noise from a real bimodal failure mode.

**Properly powered version**: fixed the sbatch bug (`--export=ALL,SEED=$s`),
ran 30 genuinely independent seeds. Mean 0.572 +/- 0.012 (min 0.554, max
0.605) -- the earlier "0.610 outlier" was itself a small-sample illusion,
not a real bimodal tail. Added true prediction-level ensembling (average
per-row y_pred across seeds, not average-the-MAE-values -- matching how the
champion's own 4-seed GNN number is built) via a `test_predictions.csv` a
script edit added: 4-seed ensemble 0.543, plateaus there (8/16/30-seed:
0.543/0.542/0.543) -- same "plateaus around 4" pattern the champion's own
seed-ensemble lever showed.

**Hyperparameter sweep** (user-requested next: "进行超参搜索"): CRG had used
TripleGNN's un-tuned defaults throughout. 20 random configs (hidden/layers/
lr/dropout/weight_decay) x 2 seeds = 39/40 jobs (1 lost to a transient SLURM
"compute budget" error), selected on **validation** MAE only (test never
touched for selection). Best: hidden=128 layers=4 lr=3e-3 dropout=0
wd=1e-4 -- close to the defaults, mainly a 3x higher LR and no dropout.
Re-ran that config x 12 seeds: 12-seed ensemble test MAE **0.535** (n=432),
essentially tying the champion's 0.5295 on the same subset.

**User pushed back a third time** ("n=432 是否太少了 需要更多的数据") --
right again: extended evaluation to the pooled test+validation set (n=898,
same convention `build_predict_dg_calibration.py` already uses), which
needed CRG inference on the validation split too (wrote a small script
reusing the training script's own `build_crg`/`CRGGNN`/`make_loader`,
reconstructing the *same* leakage-safe x_d standardization from the
deterministic train-row set rather than needing to have saved it). Pooled
result: champion 0.5556, CRG (tuned, 12-seed) 0.5635. Bootstrap (20000
resamples): CRG-champion delta mean +0.0079, 90% CI **[-0.0029, 0.0189]**
(P(CRG worse)=88.6%, up from 75% at n=432) -- more data sharpened the signal
from "can't tell" toward "champion probably still a bit better," but the
90% CI still just barely contains zero. Corrected framing from the earlier
"basically tied" to "close, probably slightly behind, not a large gap."
Also found (not yet explained): validation-split MAE is worse than
test-split MAE for BOTH models (champion 0.58 vs 0.53, CRG 0.59 vs 0.53),
and CRG's relative gap to champion widens slightly on validation --
flagged, not chased further.

**Tried blending CRG with the tabular ensemble** (champion's own recipe:
`(1-w)*ens_delta + w*gnn_delta`, w selected on validation only). Result:
w=0.76, test MAE 0.5352 -- statistically the same as CRG alone (0.5348), no
blending gain. Makes sense in hindsight: CRG already fuses the *same*
257-feat schema as its own x_d channel, so a tabular model trained on only
those same features adds little a GNN that already sees them doesn't already
have -- unlike TripleGNN, which apparently benefits more from the tabular
stack (w_gnn=0.85, not 1.0). A real negative result, not a bug.

**Fixed the `train_cross_gnn.py` landmine found above**: when the table
carries `new_scaffold_split` (every current-era table does), use it directly
(`mixed` -> `train_extra`, preserving the script's existing train/train_extra/
validation/test bucket semantics) instead of the broken
`pair_split_labels()`; falls back to the old candidates_v3 path only if a
table lacks the column. Verified against the real b973c table: reproduces
the exact known split sizes (22529/11678/481/448).

**Where this leaves things**: CRG is a real, validated, working architecture
-- close to but (with reasonably strong statistical power now) probably
still a bit behind the mature, tuned TripleGNN, not a clear win. Not
concluded as "adopt" or "close the lever" -- open threads if continued:
(1) architecture refinements to CRG itself (the order-changed carbC-hydO
edge flag noted but not built in `crg_builder.py`'s docstring; blending CRG
with TripleGNN itself rather than the tabular ensemble); (2) same CRG
treatment for BDE (raised by the user alongside the dG ask, "BDE等工作也可以
尝试" -- not started this session, single-molecule + explicit
target-bond-marking is the natural analogue but needs its own build).

**Same day, continued after an unlogged interruption: the BDE-CRG follow-through,
picked up cold on resume.** The session that wrote the paragraph above did go on to
build the BDE analogue (`pipeline/bde/train_gnn_hybrid_bde_crg.py`, products/Task B
only -- Task A's formyl C-H has no graph edge to mark under this project's
implicit-H representation, see the file's own docstring) and launched a
10-job (5 seed x marked/unmarked) full-scale ablation on gpu_a100, but the
session ended before anyone looked at the results or wrote them down.

**Resume finding: the full-scale ablation is broken, not just negative.**
All 10 completed runs (job array ending ~16792943-950, full 173k-row
scaffold-disjoint products split) show MAE swinging 9-239 kcal/mol and R^2
at/below zero (down to -7.29) -- including the `--no-mark` control, which
should be architecturally a near-no-op (extra_bond_fdim=1 channel always
zeroed) and therefore should track the champion's known-stable single-seed
number (MAE 2.07-3.24, confirmed from the untouched `b6_ensemble_26418250_*`
logs on the exact same task/split/hyperparams). A second, identical 10-job
batch (26799798-809) was still running at resume, silently reproducing the
same broken experiment.

**Diagnostic (n=5000, 15-epoch CPU smoke, 3-way: baseline / CRG --no-mark /
CRG marked, same seed/split/hyperparams).** All three trained fine at this
scale -- R^2 0.80-0.81, MAE 4.2-4.8, spearman 0.89-0.91, no divergence in
any of the three. This clears the `extra_bond_fdim`/`E_f` mechanism itself
(feature-array shape and per-bond indexing checked directly against
chemprop's `SimpleMoleculeMolGraphFeaturizer.__call__` and `make_mol` source
-- both use plain `Chem.MolFromSmiles` with matching default flags, so the
bond order `target_bond_features()` computes independently does line up with
chemprop's internal graph; no ALFABET-style misalignment). **Conclusion: this
is a full-scale-only instability**, most likely `train_one`'s
`enable_checkpointing=False` + early-stopping-without-restore-best-weights
(shared with the champion script, `train_gnn_hybrid_bde.py`) combined with
the larger row count and up to 120 epochs giving the (also larger, d_h=500)
model enough steps to wander far from its best point before patience=20
triggers -- something a 15-epoch/5k-row smoke test can't hit. Not proven by
direct evidence yet (no per-epoch val_loss was logged in either batch,
`enable_progress_bar=False, logger=False`), just the best-supported
hypothesis given what's ruled out. Tried to `scancel` the redundant
26799798-809 batch (guaranteed, on this evidence, to reproduce the same
uninformative numbers) -- blocked by the auto-mode workload-interference
guard; flagged to the user rather than worked around, left running.
**Open, not closed**: the real fix (add `ModelCheckpoint` + restore best
weights to `train_one`, or empirically cap `max_epochs`/`message_hidden` and
re-verify) has not been attempted -- touches the shared champion training
path, out of scope to change unilaterally mid-resume.

**Backup/sync audit (user-requested, "是否有效备份和同步关键数据").** Checked
three layers:
1. **git/GitHub**: 7 local commits (dating back through the 09-16 CRG work)
   had never been pushed, last push 09-14 -- pushed now (`7e14f16`).
2. **The account's only nominally-relevant automated backup**,
   `/home/schen3/bin/backup_benzoin_scratch_weekly.sh`, turns out to target
   `/scratch-shared/schen3/workfow` (the **NHC catalyst project**'s tree, not
   `benzoin-dg-restored` despite the script's name) and has been stuck since
   **2026-09-08**: a stale `.backup.lock` (no live rsync/backup process holds
   it) makes every retry print "backup already running; exiting" and exit --
   something has been re-triggering it every ~20-30s for over a week (no
   crontab or systemd timer found; source not identified), spraying ~15k tiny
   log files into `logs/`. Its own `.last_success_epoch` is frozen at 09-01.
   **Net effect: this project's scratch1 data has had no automated backup
   coverage for >2 weeks**, and the "backup" that nominally exists is scoped
   to a different, unrelated project entirely. Not fixed -- it's not this
   project's script and the fix (clear the lock? redirect it? find and kill
   the retry loop?) needs the NHC side's context before touching it; flagged
   to the user instead of acted on.
3. **BDE's own documented policy** (`pipeline/bde/STATUS.md` sec 7: force-add
   to git or list in `submit_backup_recovery_artifacts.sh`, specifically
   naming `aldehydes_all.csv`, `products_all.csv`, `b6_*.pt`) turned out to
   not actually be followed for the two CSVs: both are matched by
   `.gitignore`'s `/data/cross_benzoin/*/*_all.csv` rule, neither was ever
   force-added, and neither had a home-directory copy -- 100MB
   (`aldehydes_all.csv`) + 153MB (`products_all.csv`) of expensive-to-recompute
   local-3D descriptor data (the products file alone took the ~10-15h
   `bde_homoprod` genoa array to produce) existed in exactly one place, on
   scratch1, the same failure mode that caused the 2026-07 purge loss. The
   `.pt` checkpoints, by contrast, *were* correctly tracked (verified via
   `git ls-files`) -- so the policy was followed for the small artifacts and
   silently missed for the big ones. **Fixed**: both files rsynced to
   `/home/schen3/benzoin_backups/bde_critical_data_20260916/`;
   `aldehydes_all.csv` (under GitHub's 100MB hard limit, at 95.5MB) was also
   `git add -f`'d and pushed (`e8d3656`). `products_all.csv` (153MB) is over
   the GitHub limit -- home-backup-only for now; git-lfs would be the way to
   also cover it on GitHub, not set up.

## 2026-09-17

Resumed from `HANDOFF_20260916.md`. First priority per its §1.2: checked the
`train_one` checkpoint-restore fix's full-scale verification jobs.

**Result**: `26807944` (marked, CRG target-bond flagged) COMPLETED cleanly in
3:48:14 -- MAE 3.140, RMSE 5.338, R² 0.886, spearman 0.945 on the 170,996-row
scaffold-disjoint products set (138268/15363/17365 train/val/test). This lands
squarely in the expected 2-4 kcal/mol / R² 0.85-0.9 band (cf. B6 single-seed
2.09/3.19 in STATUS.md sec 2.1b) -- **confirms the checkpoint-restore fix
resolved the training instability**, not a data or architecture issue as
originally suspected.

`26807945` (unmarked `--no-mark` control) hit its 4h wall-time TIMEOUT without
finishing -- notably, it hadn't converged early the way the marked run did,
suggesting CRG target-bond marking may speed convergence (needs the unmarked
result to confirm properly, not a conclusion yet). Resubmitted with
`--time=08:00:00` and the correct `sbatch --export=ALL,SEED=0,MARK=0 ...`
form (job `26829622`) -- confirmed queued correctly this time, no repeat of
the 09-16 `SEED=0 sbatch` env-var-not-passed trap. Backgrounded a wait-loop to
report when it lands. `pipeline/bde/STATUS.md` sec 9 now has the full table;
sec 8's heading updated to point at it.

homo SP archived track was at 8488/8972 (94.6%) on resume, up from 7841/8972
(87.4%) at the 09-16 handoff -- regen track already complete at 771/771.
Backgrounded a poll loop to flag when archived hits 8972/8972 so the Phase 2
merge/assemble + Phase 3 single-XGB/single-GNN steps (CAMPAIGN_PLAN.md) can
start without the session needing to sit and watch it.

Noted but did not touch: the 6 `cross_round9` file diffs flagged unexplained
in the 09-16 handoff sec 4 are confirmed to be a **real retrain**, not a stat
artifact -- `gnn_attentive_9rounds_v1/models/metadata.json` shows
n_train 39030→18728, n_val 4820→2565 (roughly halved) with MAE 2.16→2.31,
best_blend_w_gnn 0.55→0.40. Something re-ran training on round9 with a
different (smaller) split. Source still unidentified; round9 is superseded by
round10 so low stakes, but per the handoff's own guidance ("不确定就先别动")
leaving it uncommitted and untouched pending whoever remembers running it.

**User asked to try other reaction-aware GNN architectures** ("尝试其他的基于
反应的GNN"), scoped after a quick clarifying question to cross-benzoin dG
only. Added two comparison points alongside the existing champion TripleGNN
(concat, 3 separate encoders) and plain CRG (graph merge, 1 encoder):

1. **CGR-delta** (`crg_builder.build_crg_delta()` + `train_cross_gnn_cgr_delta.py`):
   the refinement plain CRG's own docstring flagged but didn't build --
   proper "dynamic bond" before/after bond-order encoding instead of a single
   is_new_bond flag. Two edges actually change order across the reaction
   (not just one): ketC-carbC (new, 0->1) AND carbC-hydO (order-changed,
   acceptor's CHO C=O -> product's C-OH, 2->1) -- the second one was
   previously invisible to the model. Edge features grow 7->12 dims.
2. **WLDN-style difference network** (`train_cross_gnn_wldn.py`): a SHARED
   single encoder applied separately to product/donor/acceptor graphs (all
   three land in a common embedding space, unlike TripleGNN's independently-
   weighted per-role encoders), combined via h_P - (h_D + h_A) -- the
   Weisfeiler-Lehman Difference Network reaction vector -- instead of
   concatenation (TripleGNN) or graph merging (CRG).

Both reuse the champion's exact atom/bond featurization (`af()`/`bf()`/
`graph()` from `train_cross_gnn.py`) so any MAE delta is attributable to
combination strategy, not a feature-set change. Both smoke-tested cleanly on
CPU (n=300) before launching GPU single-seed full runs (jobs 26832100
CGR-delta, 26832101 WLDN; `gpu_a100`, `/home/schen3/venv/nequip`, same
scaffold-disjoint b973c-baseline setup as plain CRG for a clean 4-way
comparison against 0.528 champion blend / 0.535 tuned-CRG). Results pending.
