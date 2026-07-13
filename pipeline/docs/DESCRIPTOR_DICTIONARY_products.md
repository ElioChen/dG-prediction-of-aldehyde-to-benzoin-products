# Product descriptor dictionary — `products_all.csv` (homo_v6, g-xTB/GFN2 featurization)

The benzoin product is an **α-hydroxyketone**  R–C(=O)–CH(OH)–R′  formed by C–C coupling of
two aldehyde carbons. Descriptors are evaluated at the mechanistically important atoms:

| key atom | meaning in the product |
|---|---|
| **ketC / ketO** | the **ketone** carbonyl C and O (the carbon that stays C=O) — the electrophilic acyl center |
| **carbC** | the **carbinol** carbon C–OH (the former acceptor carbonyl C, now sp³ bearing OH) |
| **hydO / hydH** | the hydroxyl O and its H (the new –OH) |
| **CC_new** | the **newly formed C–C bond** joining the two former carbonyl carbons |
| **CO_ket / CO_carb** | the ketone C=O bond / the carbinol C–O bond |

---

## 1. Global electronic (whole-molecule, from xTB)
| column | meaning | why it matters |
|---|---|---|
| `xtb_HOMO`,`xtb_LUMO` | frontier orbital energies (eV) | electron-donating/-accepting ability; sets reactivity |
| `xtb_gap` | HOMO–LUMO gap | kinetic stability / hardness; small gap = reactive/polarizable |
| `xtb_IP`,`xtb_EA` | ionization potential / electron affinity | how easily the molecule loses/gains an electron |
| `xtb_mu` | electronic chemical potential μ = −(IP+EA)/2 | escaping tendency of electrons; drives charge transfer |
| `xtb_eta` | chemical hardness η = IP−EA | resistance to charge redistribution |
| `xtb_omega` | global electrophilicity ω = μ²/2η | overall electrophilic power; high ω ↔ strong EWG character |
| `xtb_dipole` | molecular dipole (Debye) | polarity; couples to solvation (DMSO) |

## 2. Atomic charges (Mulliken; ADCH variants are EMPTY here — MULTIWFN=0)
| column | meaning |
|---|---|
| `mulliken_ketC/ketO` | partial charge on ketone C / O — electrophilicity of the acyl carbon, polarization of C=O |
| `mulliken_carbC` | charge on the carbinol carbon |
| `mulliken_hydO/hydH` | charge on hydroxyl O / H — H-bond donor strength of the OH |
| `adch_*` | ADCH (atomic-dipole-corrected Hirshfeld) charges + Fukui — **all NaN** (needs Multiwfn; back-fill on a subset) |

## 3. Bond orders (Wiberg, WBO)
| column | meaning |
|---|---|
| `wbo_CO_ket` | ketone C=O bond order (~2 = clean double bond; lowered by conjugation/EWG) |
| `wbo_CO_carb` | carbinol C–O bond order (~1) |
| `wbo_CC_new` | order of the new C–C bond — integrity/strength of the coupling that was just formed |

## 4. Local reactivity (Fukui functions / dual descriptor)
Condensed Fukui = how much electron density at an atom changes on adding/removing an electron.
| column | meaning |
|---|---|
| `fukui_plus_*` (f⁺) | susceptibility to **nucleophilic** attack (electrophilic site strength) |
| `fukui_minus_*` (f⁻) | susceptibility to **electrophilic** attack (nucleophilic site strength) |
| `fukui_0_*` | radical Fukui (average) |
| `dual_*` = f⁺−f⁻ | **dual descriptor**: >0 electrophilic site, <0 nucleophilic site |
| `*_ketC` / `*_carbC` | evaluated at the ketone carbon / carbinol carbon (the two reacting carbons) |

`dual_ketC > 0` confirms the ketone carbon is the electrophilic center — the handle for retro-reactivity.

## 5. Steric / shape
| column | meaning |
|---|---|
| `vbur_ketC`,`vbur_carbC` | **% buried volume** around ketone / carbinol C — local crowding (Tolman-style); high = hindered |
| `sterimol_L` | Sterimol length (along the substituent axis) |
| `sterimol_B1`,`sterimol_B5` | Sterimol minimum / maximum width — anisotropic sterics |
| `SASA_total` | solvent-accessible surface area — overall molecular size; solvation proxy |
| `P_int` | interaction/polarizability surface descriptor (xTB) — dispersion + electrostatic surface character |
| `pa_ketO` | **proton affinity at the ketone O** — basicity of the carbonyl, key for acid/base & H-bonding |

## 6. Intramolecular H-bond geometry (product α-hydroxyketone)
The OH can fold back onto the ketone O (O–H···O=C 5-membered motif).
| column | meaning |
|---|---|
| `hb_dist` | H···O(=C) distance (Å) — shorter = stronger intramolecular H-bond |
| `hb_angle` | O–H···O angle (deg) — closer to 180° = more linear/stronger H-bond |
| `dih_core` | core dihedral about the new C–C bond — conformation of the two halves |

> Note ([hbond-not-product-error-driver]): this intramolecular H-bond is **not** the main
> product-side xTB↔DFT error driver at fixed geometry; EWG electronics are.

## 7. Energies / reaction ΔG
| column | meaning |
|---|---|
| `G_donor`,`G_acceptor`,`G_xtb` | GFN2-xTB free energies of the two aldehydes and the product |
| `dG_xtb_kcal` | GFN2-xTB reaction ΔG = G(product) − 2·G(aldehyde) (homo case) |
| `G_*_gxtb`,`dG_gxtb_kcal` | same from **g-xTB** (Grimme g-xTB) |

**Calibration finding (2026-06-25):** vs DFT r2SCAN-3c, **g-xTB MAE≈4.3** vs **GFN2 MAE≈15.5** — g-xTB
is the correct Δ-baseline; GFN2 over-stabilizes the product by ~15 kcal. See
`REPORT_gxtb_to_dft_calibration_20260625.md`.
