# screen_v6 functional-group annotated analysis — 20260619_033149

Source: `/scratch-shared/schen3/benzoin-dg/data/raw/screen_v6/analysis/screen_v6_features_all.csv`  (valid-ΔG n = **220,520**; cho_class: aromatic_carbo 87,900, aliphatic 73,666, aromatic_hetero 58,954).

## Functional groups annotated

| group | SMARTS | category | n | % lib | median ΔG | tail share | enrichment |
|---|---|---|---|---|---|---|---|
| sulfonyl  S(=O)(=O) | `[#16X4](=[OX1])(=[OX1])` | EWG xTB-unreliable | 8,483 | 3.85% | -11.3 | 11.0% | 2.9× |
| sulfonyl fluoride S(=O)2F | `[#16X4](=[OX1])(=[OX1])[F]` | EWG xTB-unreliable | 238 | 0.11% | -14.6 | 0.8% | 7.6× |
| triflate OS(=O)2CF3 | `[OX2][#16X4](=[OX1])(=[OX1])[CX4]([F])([F])[F]` | EWG xTB-unreliable | 618 | 0.28% | -13.6 | 1.4% | 4.9× |
| nitro [N+](=O)[O-] | `[$([NX3](=O)=O),$([NX3+](=O)[O-])]` | EWG xTB-unreliable | 8,492 | 3.85% | -11.3 | 8.3% | 2.1× |
| N-oxide | `[#7+][#8X1-]` | risk-tagged | 8,621 | 3.91% | -11.3 | 8.4% | 2.2× |
| boron | `[#5]` | risk-tagged | 1,311 | 0.59% | -9.4 | 1.0% | 1.7× |
| phosphorus | `[#15]` | risk-tagged | 1,393 | 0.63% | -10.7 | 1.6% | 2.6× |
| selenium | `[#34]` | risk-tagged | 295 | 0.13% | -11.6 | 0.4% | 3.3× |
| silicon | `[#14]` | risk-tagged | 2,746 | 1.25% | -8.5 | 0.5% | 0.4× |

Baseline (no annotated group) median ΔG = **-9.5** kcal/mol.
Exergonic-tail cut (most-favorable 5%) = ΔG ≤ **-17.2** kcal/mol.

## Filtering context
- FILTERED OUT pre-screen (filter_v6/v4, not in data): α,β-unsaturated enal, ynal, α-dicarbonyl, MW>500. KEPT & TAGGED here.
- The pipeline `xtb_risk` tag covers nitro/B/P/Se/N-oxide but **does NOT tag sulfonyl/hypervalent-S** — the group the DFT-SP validation proved is xTB's worst failure (EWG-only set: Pearson r = −0.32, MAE 34). Recommend adding sulfonyl/sulfonyl-fluoride/triflate to `xtb_risk`.

## Key reading
- **16,666 molecules (7.6%)** carry an EWG xTB-unreliable group.
- These groups are **enriched in the exergonic tail** (see enrichment column / fig_tail_enrichment): the screen's most-favorable predictions are disproportionately molecules where xTB ΔG is least trustworthy → tail ranking is suspect.
- Combined with the conformer broken-topology rate (~1.9% full library), the screen's extreme-exergonic end mixes genuine signal with xTB-electronic + conformer artifacts.

## Figures (standalone)
- `fig_dG_dist_EWG_annotated_20260619_033149.png` — ΔG distribution, EWG-unreliable highlighted + group annotation
- `fig_dG_by_group_box_20260619_033149.png` — ΔG by functional group (box)
- `fig_tail_enrichment_20260619_033149.png` — exergonic-tail enrichment per group
- `fig_unreliable_fraction_vs_dG_20260619_033149.png` — % xTB-unreliable vs ΔG bin