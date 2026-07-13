# screen_v6 — xtb_risk coverage gap (follow-up viz, 2026-06-19)

Follow-up to the functional-group analysis (`REPORT_screen_v6_funcgroups_*.md`) and the
DFT-SP validation (`hbond-not-product-error-driver` / `screen-v6-funcgroup-analysis`).
Reuses the cached per-molecule group flags — no SMARTS recompute. Quantifies a single
actionable claim: **the pipeline `xtb_risk` tag misses the sulfonyl / hypervalent-S
family, which the DFT-SP validation proved is xTB's worst electronic failure and which
is the most enriched group in the screen's exergonic tail.**

## Headline numbers (n = 220,520 valid-ΔG)
- EWG xTB-unreliable molecules (sulfonyl + sulfonyl-F + triflate + nitro): **16,666 (7.6%)**.
- Current `xtb_risk` (nitro/B/P/Se/N-oxide) flags only **51.2%** of them → **8,137 missed**.
- The entire miss is the sulfonyl family (`has_sulf` = 8,483): adding it to the tag →
  **100% coverage, 0 missed**.
- Most-exergonic 5% tail (ΔG ≤ −17.2, n=11,026): **10.2% (1,123 molecules) are
  xTB-unreliable yet UNflagged** by the current tag, vs 8.4% (922) caught — i.e. more
  of the tail's suspect molecules are missed than are caught.

## Figures (standalone, one per file)
- `fig_risk_coverage_gap_*.png` — coverage of the EWG xTB-unreliable set: current tag
  (51.2%, 8,137 missed) vs proposed +sulfonyl (100%).
- `fig_topK_suspect_fraction_*.png` — suspect fraction vs top-K most-exergonic
  candidates. The selection use-case: as you take the K most-favorable screen hits, the
  share carrying an xTB-unreliable group sits **2–5× above the 7.6% library baseline**
  across most of the practical K range (a second hump of ~24% around K≈500–2,000), and
  the missed-by-current-tag curve tracks it. Picking the screen's best hits draws
  disproportionately from molecules where xTB ΔG is least trustworthy.
- `fig_tail_composition_*.png` — tail partition: 81.5% no-unreliable-group, 8.4% caught,
  **10.2% unreliable-but-missed**.

## Action (unchanged, now quantified)
Add `sulfonyl` / `sulfonyl-fluoride` / `triflate` SMARTS to the pipeline `xtb_risk`
tagger (`filter-v6`). It closes 100% of the EWG-unreliable coverage gap (8,137 molecules)
and removes the 10.2% silently-suspect fraction from the exergonic tail used for
candidate ranking. Pair with the `|ΔG| > 40` hard-outlier cut (≈239 rows) before any
modelling or DFT promotion.
