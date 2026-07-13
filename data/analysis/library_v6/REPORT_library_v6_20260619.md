# aldehydes_clean_v6 / rejected_v6 — library composition (2026-06-19)

Structural characterization of the filter_v6 production library (no ΔG; from
`data/library/aldehydes_clean_v6.csv` + `aldehydes_rejected_v6.csv`).
Script: `pipeline/analysis/library_v6_viz.py` → `data/analysis/library_v6/`
(4 standalone figs + stats JSON, one figure per file).

## Headline
- **Kept 220,859 · rejected 229,328 → keep rate 49.1%** of the deduplicated PubChem
  aldehyde pool. Source is uniformly PubChem.
- cho_class: aromatic_carbo 87,987 (39.8%), aromatic_hetero 58,981 (26.7%),
  aliphatic 73,891 (33.5%). **Aromatic in-scope = 146,968 (66.5%)**; aliphatic is in
  the library but out of scope ([[aromatic-only-scope]]).
- MW: median 275.6, mean 290.2, capped at the filter_v6 MW=500 ceiling. Aromatic-carbo
  peaks sharpest (~260); aliphatic is the flattest/heaviest-tailed toward the 500 cap.
- xtb_risk: **94.8% clean**; tagged = nitro 8,492 (3.84%), P 1,396, B 1,311, Se 295,
  N-oxide 132; only 39 carry >1 tag. **sulfonyl/hypervalent-S is NOT in the tag set** —
  the known gap quantified in [[screen-v6-funcgroup-analysis]] (current tag covers only
  51% of the DFT-proven xTB-unreliable set).

## Rejection funnel (why 229,328 were dropped)
multi_component 63,500 (27.7%) · mw_too_high 63,179 (27.5%) · not_single_aldehyde
53,842 (23.5%) · enal 22,342 (9.7%) · aliphatic_too_large 8,695 (3.8%) · multi_aldehyde
6,149 · net_charged 3,826 · alpha_dicarbonyl 2,442 · reactive_group 1,462 · vinyl_conj
930 · isotope 891 · ynal 732 · zwitterion_or_ylide 590 · malformed_boron 424 ·
disallowed_element 324. The top-3 (salts/mixtures, oversize, non-mono-aldehyde) account
for ~79% of all rejections; the chemistry-scope filters (enal/ynal/α-dicarbonyl/
vinyl_conj) remove the xTB/mechanism-problematic carbonyls per [[filter-v3-relaxed]].

## Figures
- `fig_lib_v6_MW_by_class_*.png` — MW distribution by carbonyl class (MW=500 cap marked)
- `fig_lib_v6_cho_class_*.png` — composition; aromatic in-scope vs aliphatic out
- `fig_lib_v6_xtb_risk_*.png` — xtb_risk tag breakdown (log; sulfonyl-gap note)
- `fig_lib_v6_reject_funnel_*.png` — filter_v6 rejection reasons
