# Subset expansion + parallel ORCA labeling

**Date:** 2026-06-12 · How the training set is grown (incremental MaxMin) and how
descriptors + ΔG labels are computed at scale (per-molecule SLURM arrays).

## 1. Incremental MaxMin expansion ([expand_subset.py](../expand_subset.py))

Replaces the original PCA + KMeans + silhouette selection, which was weak for
binary fingerprints:

- PCA-Euclidean misrepresents chemical (Tanimoto) similarity on sparse binary FP.
- Molecular libraries have **no natural clusters** → silhouette ≈ 0.05, so the
  "optimal k" it picked was illusory (it just took the grid minimum).

MaxMin picks a **maximally diverse** set directly on Tanimoto distance. The count
is a *budget*, not a derived optimum; coverage is judged by nearest-representative
distance (below), not silhouette. **Incremental:** the existing subset is passed as
`firstPicks`, so MaxMin only ADDS the next-most-diverse molecules — already-computed
descriptors/ΔG are never wasted.

```bash
python pipeline/expand_subset.py --n-add 300      # 200 -> 500, writes subset_expansion.csv
```

### Coverage is intrinsically flat — more points barely help

Nearest-representative distance (1 − Tanimoto to closest rep), 20k-sample:

| N_total | mean | median | p95 | max |
|--------:|-----:|-------:|----:|----:|
| 200  | 0.656 | 0.667 | 0.781 | 0.912 |
| 400  | 0.652 | 0.667 | 0.771 | 0.819 |
| 800  | 0.644 | 0.660 | 0.759 | 0.798 |
| 1600 | 0.632 | 0.651 | 0.744 | 0.777 |
| 3200 | 0.612 | 0.637 | 0.722 | 0.753 |

Even 16× more representatives (200→3200) moves the median only 0.667→0.637. The
218k-aldehyde space is too diffuse to "cover" with hundreds–thousands of ORCA
points — this is intrinsic, **not** a budget limit. MaxMin only meaningfully
improves the *max* (worst-covered edges: 0.912→0.753). Implication: the model is a
local/screening tool; per-prediction **applicability-domain** flagging (distance to
nearest training molecule) is the honest way to handle out-of-domain inputs.

So the +300 expansion is run as an **experiment** (does diverse data improve the
model's CV / learning curve?), not as a coverage fix.

### Data-quality filters added in expansion

MaxMin preferentially selects fingerprint-rare molecules, which are often chemical
artifacts rather than useful diversity. Two filters were added:

| filter | why | effect |
|--------|-----|--------|
| **exclude isotopes** (`GetIsotope()!=0`) | ¹⁴C/²H/¹⁷O are electronically identical to unlabeled forms → redundant for ΔG/descriptors; also break the benzoin-SMILES builder | MaxMin had over-picked them ~7× (2.0% of picks vs 0.28% of library) |
| **neutral carbonyl** `[CX3H1;+0](=[O;+0])[#6]` | the loose `=O` SMARTS also matched charged carbonyls — carbonyl oxides (`C=[O+][O-]`), ketenes (`C=C=O`) — which aren't benzoin substrates | removes them while keeping legit charged groups elsewhere (e.g. nitroaromatics) |

(Both should also be folded into `filter_smiles.py` on the next full library re-filter.)

## 2. Per-molecule parallel ORCA labeling ([submit_labels_array.sh](../slurm/submit_labels_array.sh))

The original whole-subset job ran serial ORCA (`nprocs=1`) with molecule-level
workers — fine for small molecules, but the **large benzoin products timed out at
the 7200 s single-point limit**. Fix = SLURM **array, one molecule per task**, each
task running PARALLEL ORCA on its own allocation. SLURM packs ~8 tasks/node and
spreads across the cluster, so every big single-point gets dedicated cores and a
failed/slow molecule is isolated (resubmit alone).

```bash
pipeline/slurm/run_labels_array.sh INPUT.csv OUTDIR MAXCONCURRENT   # auto-sizes the array
```

### The real bottleneck was xtb threading, not ORCA

A first 4-molecules-per-node supplement crawled (37 min, 0 done). Diagnosis: the
script exported `OMP_NUM_THREADS=1`, so xtb's **numerical Hessian (`ohess`
frequencies)** ran single-threaded — 15/16 cores idle during the slow ohess phase,
which dominates wall time for ~70-atom benzoin products. The ORCA single-point was
*not* the main cost. Fix in the array script:

```
ORCA_NPROCS=16        # parallel ORCA single-point
XTB_THREADS=16        # OMP_NUM_THREADS — multi-thread the xtb Hessian
cpus-per-task=16      # both phases (sequential) use the full allocation
```

## 3. Per-chunk parallel descriptors ([submit_descriptors_array.sh](../slurm/submit_descriptors_array.sh))

`ald_descriptors.py` has no internal parallelism, so descriptors are parallelized by
a SLURM array that slices the input into chunks (descriptors are cheap → per-chunk,
not per-molecule), each task multi-threaded xtb + Multiwfn.

```bash
pipeline/slurm/run_descriptors_array.sh INPUT.csv OUTDIR CHUNK_SIZE
```

## 4. Merge → one training table ([merge_labels.py](../merge_labels.py))

Combines reconstructed main run + supplement + array tasks, preferring rows that
actually carry a DFT ΔG, into `data/labels/chunk_000/delta_G.csv` (the pipeline's
label glob). Descriptor chunks merge by concatenation.

## Current run (2026-06-12)

| job | what | layout |
|-----|------|--------|
| 23697101 | supplement: original-200's 8 timed-out molecules | array, 16-core parallel ORCA |
| 23697353 | +300 expansion descriptors | array, 10 chunks × 30 |
| 23697354 | +300 expansion ΔG labels | array, 300 tasks, %50, 16-core ORCA |

`166865` (deuterated octanal) is dropped — `_make_benzoin_smiles` builds an
invalid valence-5 carbon for it (a real builder bug, not a compute failure).

After all three finish: merge → **500-molecule dataset** → run the **200-vs-500
diversity experiment** (re-measure CV + learning curve to test whether diverse data
helps where random subsampling didn't), then train the production model.
