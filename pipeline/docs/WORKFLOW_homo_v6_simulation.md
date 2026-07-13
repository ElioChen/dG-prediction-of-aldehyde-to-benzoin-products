# homo_v6 Simulation Workflow — with a focus on geometry optimization

_Dataset: `data/cross_benzoin/homo_v6/` (the self-coupling benzoin **product** set).
Entry point: `cross_benzoin/cb_featurize.py`. Geometry engine: `conf_funnel_v3.py`.
Energies / thermochemistry: `thermo_orca.py`._

---

## 0. What this dataset is

A **benzoin** is the α-hydroxyketone formed when two aldehyde molecules are coupled
carbon-to-carbon (an NHC-catalysed reaction). This dataset covers the **self-coupling
("homo") case**, where the two aldehydes are identical, run over the **full aldehyde
library (~220,000 molecules)**:

- `aldehydes_all.csv` — **220,525** rows (one per unique aldehyde)
- `products_all.csv`  — **220,724** rows (one per self-coupled product)

Everything is computed at a **semi-empirical quantum-chemistry level** (the fast GFN2-xTB
method, plus a single-point with the newer g-xTB method); no DFT is run at full-library
scale. All calculations are done **in DMSO solvent** using an implicit solvation model.

**Guiding principle — one method, everything saved.** Aldehydes and products are built
with the *same* conformer-search procedure and the *same* property calculators, and every
optimized geometry, energy, and descriptor is stored so the two tables can be linked.

---

## 1. Data flow at a glance

```
aldehyde library CSV (id, SMILES, ...)
      │  self-coupling: pair each aldehyde with itself
      ▼
cb_featurize.py   (conformer search → optimized geometry → free energy → descriptors)
      ├─ per unique aldehyde → one aldehydes.csv row + its optimized 3D structure
      └─ per product         → one products.csv row  + its optimized 3D structure
                                ΔG(reaction) = G(product) − G(aldehyde 1) − G(aldehyde 2)
```

The aldehyde free energies are computed once and **reused** on the product side, so the
expensive geometry work is never repeated.

---

## 2. Geometry optimization (the core of the workflow)

Finding the right 3D structure is the hard part: a flexible molecule has many possible
shapes (conformers), and the reported energy is only meaningful if we locate the true
lowest-energy shape. Doing an expensive optimization on *every* candidate shape would be
prohibitive at 220k molecules, so the workflow uses a **funnel**: generate many candidate
shapes, filter them with progressively more accurate (and more expensive) methods, and
only fully optimize the handful that survive. The **same procedure is applied to every
aldehyde and every product.**

### Stage 1 — Generate candidate conformers (RDKit)
- Candidate 3D structures are built with RDKit's distance-geometry embedding (ETKDG),
  then cleaned up with a classical force field (MMFF).
- Generation is made **fully reproducible** by running single-threaded with a fixed random
  seed, so the same molecule always yields the same conformers.
- Near-duplicate shapes are **pruned by RMSD** at generation time, so the kept conformers
  represent genuinely *distinct* shapes rather than many copies of the same one.
- The **number of conformers scales with molecular flexibility**, capped for large
  molecules to keep the per-molecule runtime bounded:

  | rotatable bonds | conformers generated |
  |---|---|
  | ≤ 7  | 100 |
  | 8–12 | 400 |
  | ≥ 13 | 600 |

  (capped to 200 above ~55 heavy atoms, and to 100 above ~70 heavy atoms.)

### Stage 2 — Cheap force-field pre-optimization (all conformers)
Every candidate is relaxed with the very fast **GFN-FF** force field
(`--opt loose`). This settles each conformer into its nearby local minimum at negligible
cost, so the ranking in the next stage is meaningful.

### Stage 3 — Fast quantum single-point, keep the best 10
Each pre-optimized shape gets a **GFN2-xTB single-point energy** (`--sp`). The conformers
are ranked by this energy and **only the 10 lowest are kept**. This is where the funnel
narrows from hundreds of candidates to ten.

### Stage 4 — Accurate optimization on the survivors only
The 10 survivors are fully optimized with **GFN2-xTB** (`--opt tight`). This is the only
expensive quantum optimization step, and it runs on 10 structures instead of hundreds —
the whole point of the funnel.

### Stage 5 — Connectivity check (safety guard)
The single most dangerous failure mode is a conformer that has **silently changed its
bonding** — the force field occasionally relaxes a structure into a rearranged or
fragmented form whose artificially low energy would then corrupt the result. To prevent
this, each optimized structure's perceived heavy-atom **bonding pattern is compared to the
original molecule**, and any structure whose connectivity has changed is discarded. If (due
to a perception edge case) *every* structure is flagged, the workflow falls back to the
unfiltered ranking rather than returning nothing.

### Stage 6 — Pick the representative geometry
The surviving structures are ranked by their GFN2-xTB optimized energy, and the lowest one
is taken as the molecule's **representative geometry** and saved.

> **In one line:** generate many reproducible, de-duplicated conformers → cheap force-field
> relax all → fast quantum single-point, keep the 10 lowest → accurate quantum optimization
> of those 10 → discard any that changed their bonding → keep the lowest-energy structure.

---

## 3. Free energy and reaction ΔG

**Gibbs free energy (GFN2-xTB).** On the representative geometry, a frequency + thermochemistry
calculation (`--ohess tight`, in DMSO) gives the **Gibbs free energy**
`G = electronic energy + thermal (vibrational/rotational/translational) correction`.
This is stored as `G_xtb`.

**g-xTB single-point (better baseline).** On the *same* geometry, one single-point with the
newer **g-xTB** method (in DMSO) provides a more accurate electronic energy. It is combined
with the GFN2 thermal correction to give `G_gxtb`. Because it reuses the existing geometry,
this adds only ~10–20% cost instead of doubling it.

**Reaction free energy.**
```
ΔG (kcal/mol) = [ G(product) − G(aldehyde 1) − G(aldehyde 2) ] × 627.51
```
For the self-coupling case the two aldehyde terms are identical. The result is stored as
`dG_xtb_kcal` (GFN2 level) and `dG_gxtb_kcal` (g-xTB level).

> Against DFT reference calculations, **g-xTB is substantially more accurate than GFN2**
> for these reaction energies (roughly 4 vs 15 kcal/mol mean error), which is why g-xTB is
> used as the baseline for any downstream correction model.

---

## 4. Descriptors (what is computed and why)

Descriptors are numerical features describing each molecule's electronics, charge
distribution, bonding, reactivity, sterics, and internal hydrogen bonding. For the
**product**, they are evaluated at the mechanistically important atoms of the
α-hydroxyketone core, R–C(=O)–CH(OH)–R′:

| atom label | what it is in the product |
|---|---|
| **ketC / ketO** | the **ketone** carbon and oxygen (the C that stays C=O) — the electrophilic acyl center |
| **carbC** | the **carbinol** carbon (C–OH), now sp³ |
| **hydO / hydH** | the hydroxyl oxygen and its hydrogen (the new –OH) |
| **CC_new** | the **newly formed C–C bond** joining the two former carbonyl carbons |
| **CO_ket / CO_carb** | the ketone C=O bond and the carbinol C–O bond |

### 4.1 Global electronic properties (whole molecule)
| column | meaning | why it matters |
|---|---|---|
| `xtb_HOMO`, `xtb_LUMO` | frontier orbital energies (eV) | electron-donating / -accepting ability |
| `xtb_gap` | HOMO–LUMO gap | stability vs reactivity; a small gap means reactive/polarizable |
| `xtb_IP`, `xtb_EA` | ionization potential / electron affinity | how easily an electron is lost / gained |
| `xtb_mu` | chemical potential μ = −(IP+EA)/2 | electrons' escaping tendency; drives charge transfer |
| `xtb_eta` | chemical hardness η = IP−EA | resistance to charge redistribution |
| `xtb_omega` | electrophilicity ω = μ²/2η | overall electrophilic power (high ↔ strong electron-withdrawing character) |
| `xtb_dipole` | molecular dipole (Debye) | polarity; couples to DMSO solvation |

### 4.2 Atomic charges
| column | meaning |
|---|---|
| `mulliken_ketC / ketO` | partial charge on the ketone C / O — electrophilicity and polarization of the C=O |
| `mulliken_carbC` | charge on the carbinol carbon |
| `mulliken_hydO / hydH` | charge on the hydroxyl O / H — hydrogen-bond-donating strength of the OH |

_(A more refined charge scheme, ADCH from Multiwfn, is supported but was left off in this
run; those columns are empty and can be back-filled on a subset if needed.)_

### 4.3 Bond orders (Wiberg)
| column | meaning |
|---|---|
| `wbo_CO_ket` | ketone C=O bond order (~2 for a clean double bond; lowered by conjugation) |
| `wbo_CO_carb` | carbinol C–O bond order (~1) |
| `wbo_CC_new` | order of the newly formed C–C bond — integrity of the coupling |

### 4.4 Local reactivity (Fukui functions)
A Fukui function measures how much the electron density at a given atom changes when an
electron is added or removed — i.e. where the molecule is most reactive.
| column | meaning |
|---|---|
| `fukui_plus_*` (f⁺) | susceptibility to **nucleophilic** attack (electrophilic-site strength) |
| `fukui_minus_*` (f⁻) | susceptibility to **electrophilic** attack (nucleophilic-site strength) |
| `fukui_0_*` | radical (average) Fukui |
| `dual_*` = f⁺ − f⁻ | **dual descriptor**: > 0 marks an electrophilic site, < 0 a nucleophilic site |

Evaluated mainly at the two reacting carbons (`*_ketC`, `*_carbC`). A positive `dual_ketC`
confirms the ketone carbon is the electrophilic handle.

### 4.5 Steric / shape
| column | meaning |
|---|---|
| `vbur_ketC`, `vbur_carbC` | **% buried volume** around the ketone / carbinol carbon — local crowding (high = hindered) |
| `sterimol_L` | Sterimol length along the substituent axis |
| `sterimol_B1`, `sterimol_B5` | Sterimol minimum / maximum width — directional (anisotropic) size |
| `SASA_total` | solvent-accessible surface area — overall size; solvation proxy |
| `P_int` | polarizability / interaction-surface descriptor — dispersion + electrostatic surface character |
| `pa_ketO` | **proton affinity at the ketone oxygen** — carbonyl basicity, relevant to acid/base and H-bonding |

### 4.6 Intramolecular hydrogen bond
In the product the hydroxyl can fold back onto the ketone oxygen (an O–H···O=C motif).
| column | meaning |
|---|---|
| `hb_dist` | H···O(=C) distance (Å) — shorter is a stronger internal H-bond |
| `hb_angle` | O–H···O angle (degrees) — closer to 180° is stronger |
| `dih_core` | dihedral about the new C–C bond — the relative orientation of the two halves |

### 4.7 Energies / reaction ΔG (summary columns)
| column | meaning |
|---|---|
| `G_donor`, `G_acceptor`, `G_xtb` | GFN2-xTB free energies of the two aldehydes and the product |
| `dG_xtb_kcal` | GFN2-xTB reaction free energy |
| `G_*_gxtb`, `dG_gxtb_kcal` | the same, computed with g-xTB (the more accurate baseline) |

---

## Appendix: where each step lives in the code

| step | file : function |
|---|---|
| orchestration | `cross_benzoin/cb_featurize.py` : `_featurize_aldehyde` / `_featurize_product` |
| conformer generation | `pipeline/compute/conf_funnel_v2.py` : `_embed_conformers_robust` |
| conformer count / pruning | same file : `_frust_nconfs_v2` / `_rmsd_prune_thresh` |
| force-field & fast-quantum stages | `pipeline/compute/conf_funnel.py` : `_xtb_gfnff_opt` / `_xtb_gfn2_sp` |
| accurate quantum optimization | `pipeline/compute/thermo_orca.py` : `_xtb_opt_energy` |
| connectivity guard | `pipeline/compute/conf_funnel_v3.py` : `rank_conformers_funnel_v3` |
| free energy (frequencies) | `pipeline/compute/thermo_orca.py` : `run_ohess` / `parse_xtb_G` |
| g-xTB single-point | `cross_benzoin/cb_featurize.py` : `_gxtb_sp` / `_g_gxtb` |
