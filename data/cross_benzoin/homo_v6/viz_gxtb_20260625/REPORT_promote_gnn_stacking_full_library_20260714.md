# GNN+tabular stacking promoted to full-library scoring (20260714)

Follows the CONFIRMED aligned-v2 result (job 24578348): full held-out test MAE
tabular-only 1.485 -> blend (w_gnn=0.40) 1.427 (delta -0.058).

This run applies that same w_gnn=0.40 blend to the WHOLE library using the saved
GNN checkpoint (`gnn_dual_champion275_ALIGNEDV2_state_20260713.pt`) and the
existing tabular champion's full-library predictions
(`products_dG_corrected_MORDREDSLIM271_BDEGXTB_20260706.csv`).

- Full library n = 218,227
- GNN cache coverage: n = 218,105 (99.9%) -- the
  ~122 without a cached graph pair keep the
  tabular-only prediction (`blend_source = tabular_only_no_gnn_cache`).
- Output: `products_dG_corrected_GNNSTACK_w40_20260714.csv` (`dG_blend_final` is the new best-estimate column;
  `dG_gxtb_corrected_final` from the tabular champion kept alongside for reference).
- Sanity check on the full labeled overlap (n=218,105, note this INCLUDES
  the GNN's own training rows so it is optimistic, not a held-out number -- the
  trustworthy number is the aligned-v2 test-set MAE above): tabular 1.055,
  GNN 1.350, blend 1.095.

**Recommendation:** use `products_dG_corrected_GNNSTACK_w40_20260714.csv`'s `dG_blend_final` as the new
production prediction column going forward.
