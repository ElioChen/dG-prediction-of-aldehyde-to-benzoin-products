# Hypervalent-tag feature augmentation (20260710)

Tier-1b of the 2026-07-10 external-diagnosis review (Action D): adds explicit SMARTS-based hypervalent/functional-group tags (sulfonyl, sulfonyl-F, nitro, nitrile, imine, has_B/Si/P, soft-S-thioether, halogen, ester, amide -- boolean + match-count, both aldehyde and product side, 24 new columns) to the champion 275-feat set. Same 70:20:10 split (seed 42), same ensemble as the production champion.

| variant | n_feat | test MAE | RMSE | R2 | hard-subset MAE (n) | background MAE |
|---|---|---|---|---|---|---|
| champion275 | 275 | 1.496 | 2.251 | 0.876 | 2.226 (4685) | 1.297 |
| champion275_plus_hypertags | 323 | 1.486 | 2.242 | 0.877 | 2.220 (4685) | 1.287 |

**Delta overall MAE: -0.009** (noise band on the 72-feat baseline is +/-0.013 over reshuffled seeds -- treat |delta|<0.02-0.03 as noise, per descriptor-search-exhausted memory). **Delta hard-subset MAE: -0.006**.

If the hard-subset delta is negative well beyond noise while the overall delta is flat, the explicit tags are a real, localized win worth keeping even though they don't move the global number (which is dominated by the 85%+ easy majority). If both deltas are flat, Action D is also a null result, same conclusion pattern as Action A (atom-local P_int) and ADCH/QTAIM.
