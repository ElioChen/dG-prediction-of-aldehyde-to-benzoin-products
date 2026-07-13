# screen_v6 xTB screen — multi-dimensional analysis (2026-06-17)

## Why
`screen_v6` is the xTB-only library screen (no DFT) over the filtered aldehyde
library (`screen-v6-pipeline`). All **1105 chunks / 220,859 molecules** finished.
This note goes past a single ΔG histogram: it quantifies data quality, the ΔG
landscape per class, **which electronic/steric descriptors actually drive ΔG**,
the substructure-risk conditioning, and the outlier pathology — and what each
means for catalyst/substrate selection.

**What ΔG is here.** `dG_xtb_kcal = G(benzoin) − 2·G(aldehyde)`, GFN2-xTB `--ohess`
free energies in ALPB(DMSO), single best conformer each side
(`featurize_screen.py:114`, `thermo_orca._dG_kcal`). It is the **reaction free
energy of the homo (self-)benzoin condensation** `2 RCHO → R-CO-CH(OH)-R`. More
negative ⇒ benzoin formation more thermodynamically favourable.

Aggregate table, figures and CSVs live in
`data/raw/screen_v6/analysis/` (regenerate with `deep_analyze.py`).

## Data quality (read this before trusting the numbers)
| metric | value |
|---|---|
| molecules total | 220,859 |
| xTB optimized OK | 220,733 (99.94%) |
| valid numeric ΔG | 220,520 (99.8%) |
| inside physical window [−40, +20] | 220,138 |
| `|ΔG| > 40` outliers (xTB failures) | **239 (0.11%)** |
| ADCH / QTAIM (Multiwfn) columns | **0% populated — entirely empty** |
| all xTB + morféus descriptors | ~99.9% populated |

Two things to act on:

1. **The 7 Multiwfn columns are empty for the whole screen** (`adch_CHO_C/O`,
   `adch_fukui_*`, `qtaim_lap_CO`, `qtaim_ell_CO`). This confirms
   `multiwfn-env-and-screen-gap`: the screen submit omitted `--multiwfn`. Any
   model trained on the screen set must drop these columns or backfill them on a
   sampled subset — they carry **zero information** as shipped.
2. **239 hard outliers** (ΔG down to −185). These are not chemistry; they are xTB
   blow-ups on pathological valences — the SMILES are littered with hypofluorites
   (`OF`), carbenes (`[CH]`), `O[Si]`, polysulfonic anhydrides, etc. Only **17.6%**
   of them carry an `xtb_risk` tag, so the current tag set does **not** catch most
   of them. Filter on `|ΔG| > 40` (or tighten `filter-v6` for `O–F`/carbene)
   before any modelling. They are excluded from everything below.

## The ΔG landscape
> **"aromatic in-scope"** throughout = `cho_class ∈ {aromatic_carbo,
> aromatic_hetero}` (the project's target set), with **aliphatic excluded** per
> `aromatic-only-scope`. Analyses below are reported for **three groupings**: all
> aldehydes, aromatic in-scope, and non-aromatic (aliphatic) — because, as the
> next section shows, the drivers differ between them.

Clean set (n=220,138): mean **−9.46**, median **−9.64**, σ **5.0**; 5–95% spans
**−17.2 … −1.0**. **96% of all substrates are exergonic** (ΔG<0). See
[fig01](../../data/raw/screen_v6/analysis/fig01_dG_dist_grouped.png) (distribution,
all / aromatic / non-aromatic) and
[fig02](../../data/raw/screen_v6/analysis/fig02_dG_ecdf_aromatic.png)
(ECDF, aromatic in-scope).

**Implication:** at the *sign* level ΔG barely discriminates — almost everything is
downhill. The useful signal is the **spread** (a ~16 kcal window), and given the
~3 kcal xTB noise floor (`delta-mae-noise-floor`), ΔG should be used as a coarse
thermodynamic pre-filter / one feature, **not** a fine ranking on its own.

### By carbonyl class ([fig03](../../data/raw/screen_v6/analysis/fig03_dG_violin_by_class.png))
| class | n | median ΔG | mean | σ | 5–95% |
|---|---|---|---|---|---|
| aliphatic | 73,397 | **−10.91** | −10.43 | 5.0 | −17.6 … −1.8 |
| aromatic_hetero | 58,902 | −9.70 | −9.66 | 5.6 | −18.6 … −0.3 |
| aromatic_carbo | 87,839 | **−8.79** | −8.51 | 4.4 | −15.1 … −1.1 |

Clear, chemically sensible ordering: **carbo-aromatic aldehydes are the *least*
exergonic**. Ring conjugation stabilises the aldehyde reactant the most, shrinking
the driving force; hetero-aromatics sit between; aliphatics (no conjugative
stabilisation) are most exergonic. This is exactly why the **aromatic-only scope**
(`aromatic-only-scope`) is the harder, more interesting design space — and note
**aliphatic (73k) is in the library but out of scope** and is excluded from the
in-scope correlations below.

## What drives ΔG — descriptor correlations
Ranked bars per group:
[all](../../data/raw/screen_v6/analysis/fig04_corr_all.png) ·
[aromatic](../../data/raw/screen_v6/analysis/fig04_corr_aromatic.png) ·
[non-aromatic](../../data/raw/screen_v6/analysis/fig04_corr_nonaromatic.png).
Full matrices:
[all](../../data/raw/screen_v6/analysis/fig11_heatmap_all.png) ·
[aromatic](../../data/raw/screen_v6/analysis/fig11_heatmap_aromatic.png) ·
[non-aromatic](../../data/raw/screen_v6/analysis/fig11_heatmap_nonaromatic.png).

Top drivers on the **aromatic in-scope set (n=146,741)** (Pearson r / Spearman ρ):

| descriptor | r | ρ | reading (more negative ΔG = better) |
|---|---|---|---|
| Mulliken q(O) of C=O | **−0.71** | −0.72 | more **negative** carbonyl-O charge ⇒ more favourable |
| WBO C=O bond order | **−0.66** | −0.67 | higher C=O bond order ⇒ more favourable |
| xTB LUMO | +0.55 | +0.58 | **lower LUMO** (electrophilic C=O) ⇒ more favourable |
| chem. potential μ | +0.54 | +0.56 | lower μ ⇒ more favourable |
| IP | −0.50 | −0.52 | higher IP (electron-poor π) ⇒ more favourable |
| electrophilicity ω | −0.46 | −0.48 | **higher ω** ⇒ more favourable |
| HOMO | +0.45 | +0.47 | lower HOMO ⇒ more favourable |
| proton affinity (O) | +0.45 | +0.50 | lower carbonyl basicity ⇒ more favourable |
| EA | −0.40 | −0.40 | higher EA ⇒ more favourable |
| Sterimol B5 | +0.25 | +0.25 | sterics: weak |
| MW | +0.22 | +0.22 | weak |
| %Vbur (carbonyl C) | +0.09 | +0.07 | negligible |
| **Fukui f⁺ (carbonyl C)** | **−0.00** | −0.03 | **no correlation** |

Two interpretive headlines:

- **Benzoin ΔG is an electronically-driven, carbonyl-centric quantity.** Every
  top driver points the same way: **electron-poor, electrophilic, EWG-substituted
  carbonyls are the most exergonic** (low LUMO, high ω, high IP/EA, low μ/HOMO,
  low O basicity). Sterics (Sterimol, %Vbur, MW) are weak second-order effects.
  See the ω and gap maps:
  [fig05](../../data/raw/screen_v6/analysis/fig05_dG_vs_omega.png),
  [fig06](../../data/raw/screen_v6/analysis/fig06_dG_vs_gap.png),
  [fig07](../../data/raw/screen_v6/analysis/fig07_dG_vs_pa.png),
  [fig09](../../data/raw/screen_v6/analysis/fig09_dG_vs_sterics.png),
  [fig10](../../data/raw/screen_v6/analysis/fig10_dG_vs_MW.png).
- **Local electrophilic Fukui f⁺ at the carbonyl C is essentially uncorrelated
  (r≈0)** even though *global* electrophilicity ω is a strong driver
  ([fig08](../../data/raw/screen_v6/analysis/fig08_dG_vs_fukui.png)). The
  thermodynamics is set by whole-molecule electronics, not the local softness at
  the reacting carbon. Useful for feature selection: keep ω/LUMO/q(O)/WBO, the
  local Fukui f⁺ adds little for the *thermodynamic* target (it may still matter
  for kinetics — different question).

### Aromatic vs non-aromatic: the drivers are NOT the same
Pearson r(ΔG, descriptor), same descriptor set, three groupings
(`corr_by_group.json`; compare
[fig04 aromatic](../../data/raw/screen_v6/analysis/fig04_corr_aromatic.png) vs
[fig04 non-aromatic](../../data/raw/screen_v6/analysis/fig04_corr_nonaromatic.png)):

| descriptor | all | aromatic | non-aromatic |
|---|---|---|---|
| Mulliken q(O) | −0.61 | **−0.71** | −0.40 |
| C=O bond order (WBO) | −0.51 | **−0.66** | −0.32 |
| electrophilicity ω | −0.23 | **−0.46** | −0.11 |
| xTB EA | −0.12 | **−0.40** | −0.02 |
| xTB LUMO | +0.32 | **+0.55** | +0.26 |
| chem. potential μ | +0.33 | +0.54 | +0.25 |
| proton affinity (O) | +0.34 | +0.45 | +0.19 |
| **Sterimol B1** | +0.20 | +0.16 | **+0.35** |
| **%Vbur (carbonyl C)** | +0.12 | +0.09 | **+0.34** |
| **Mulliken q(C)** | −0.16 | −0.32 | **+0.19** (sign flip) |

Three things this says:

- **Aromatics are an electronic regime; aliphatics are a steric regime.** Every
  *global electronic* descriptor (ω, LUMO, EA, μ) is ~2× stronger for aromatics —
  ring conjugation couples substituent electronics into the carbonyl, so EWG
  substitution tunes ΔG strongly. For aliphatics those same descriptors go weak
  (ω −0.11, EA −0.02), and **sterics rise to the top instead** (Sterimol B1 +0.35,
  %Vbur +0.34): aliphatic ΔG is governed more by branching/strain near the
  carbonyl than by π-electronics. Mulliken q(C) even **flips sign** between the two.
- **Pooling the two regimes dilutes everything.** The "all" column is uniformly
  weaker than aromatic (ω −0.23 vs −0.46, q(O) −0.61 vs −0.71): mixing two
  different structure–property laws washes out the signal. This is concrete
  support for **modelling aromatic separately** (`aromatic-only-scope`) rather than
  one pooled model — and, if aliphatics are ever revisited, giving them their own
  steric-weighted feature set.
- The carbonyl's *own* descriptors (q(O), WBO C=O) stay the #1–2 drivers in **both**
  regimes — they are the robust, transferable core features.

## Substructure risk does **not** mean "bad ΔG"
([fig12](../../data/raw/screen_v6/analysis/fig12_dG_by_risk.png), all classes)

| tag | n | median ΔG |
|---|---|---|
| clean | 208,603 | −9.57 |
| nitro | 8,484 | **−11.28** |
| selenium | 279 | −11.25 |
| phosphorus | 1,363 | −10.64 |
| n_oxide | 125 | −10.23 |
| boron | 1,284 | −9.39 |

The `xtb_risk`-tagged substrates (the "keep-but-tag" set from `filter-v3-relaxed`)
sit at the **more-exergonic end**, not the outlier end — nitro especially (strong
EWG ⇒ low LUMO ⇒ favourable, fully consistent with the correlation story). So they
should **not** be dropped on thermodynamic grounds; the tags are about xTB
reliability, and that is better handled by the `|ΔG|>40` failure filter.

## Candidate substrates
`analysis/top_exergonic_aromatic.csv` lists the 50 most-exergonic in-scope
aromatics (ΔG down to **−38.3**). They are dominated by **strongly EWG-substituted
benzaldehydes** (fluorosulfonates `OS(=O)(=O)F`, sulfonyl fluorides, nitro,
polyfluoro) — exactly what the descriptor analysis predicts. **Caveat:** a few of
the very top hits carry chemically dubious groups (`SF`, `OF`) that survived the
filter; sanity-check the head of that list before promoting any to DFT, and treat
it as "electronic regime of interest" rather than a literal shopping list.

## Actions
1. **Backfill the 7 empty Multiwfn columns** (ADCH/QTAIM) onto screen_v6 via
   `submit_backfill_mwf_array.sh` (reuses saved geometry, no re-opt). A single-chunk
   validation (chunk_0000, job 23964377) is running; scale (full 1105 vs stratified
   sample) is decided after it confirms the columns populate correctly
   (`multiwfn-env-and-screen-gap`).
2. Apply a `|ΔG| > 40` (≈239 rows) cut and consider tightening `filter-v6` for
   `O–F` / carbene / hypervalent valences the tag set misses.
3. Use xTB ΔG as a coarse pre-filter + one feature, not a fine ranking (noise
   floor ≈ 3 kcal).
4. Keep electronic descriptors (q(O), WBO C=O, LUMO, ω, IP, μ, PA) as the primary
   feature block; carbonyl-local Fukui f⁺ is uninformative for this target.
5. Carry `xtb_risk` substrates forward — they are favourable, not anomalous.

Figures are **one standalone file per dimension** in `data/raw/screen_v6/analysis/`
(`fig01`…`fig12`).
