# Problem definition and mathematical formulation — benzoin ΔG prediction

Extracted and cleaned up from a GitHub-side Codex session's methodology
discussion (`thought.md`, 2026-07-14) for reuse in future write-ups/papers.
Math is written in paper-ready LaTeX-style notation. Companion file:
`METHODS_problem_definition_and_math_ZH.md`.

## 1. What "benzoin ΔG" means, precisely

"Predict benzoin ΔG from aldehyde SMILES" is underspecified until one fixes
which of two different quantities is meant, and under which conditions:

- **Reaction free energy** $\Delta G_{\mathrm{rxn}} = G_{\mathrm{product}} -
  G_{\mathrm{donor}} - G_{\mathrm{acceptor}}$ — a thermodynamic quantity,
  and what this project actually predicts.
- **Activation free energy** $\Delta G^{\ddagger}$ — requires the NHC
  catalyst, base, solvent, and explicit mechanism; it is *not* determined by
  the two aldehyde SMILES alone and is out of scope here.

For $\Delta G_{\mathrm{rxn}}$ to be a well-defined function of a
`(donor SMILES, acceptor SMILES)` pair, the following must be fixed and
recorded, or the same SMILES pair can map to multiple different "correct"
answers:

- donor/acceptor **direction** (which aldehyde is the NHC-activated acyl
  anion equivalent) — the cross case has two directed reactions per
  unordered pair, see §6;
- the resulting **regioisomer** of the product (which carbonyl becomes the
  ketone, which becomes the carbinol);
- temperature (this project: $T = 298.15\,\mathrm{K}$);
- solvent / implicit solvation model (this project: DMSO, ALPB at GFN2,
  COSMO at g-xTB, CPCM at DFT);
- standard state (1 mol/L solution);
- charge, protonation state, and tautomer handling;
- conformer treatment (lowest-energy conformer vs. Boltzmann ensemble, §4);
- the exact electronic-structure method and its version
  (GFN2-xTB / g-xTB / DFT level, functional, basis / composite method —
  this project's DFT label level is r2SCAN-3c, see
  [[dft-labels-r2scan-not-pbe0]]).

## 2. Multi-fidelity energy decomposition (Δ-learning)

When geometry and thermal corrections come from a cheap level (GFN2-xTB
`--ohess`) and a more expensive level (g-xTB or DFT) is evaluated only as a
**single point on that same optimized geometry**, the higher-level free
energy of a species $M$ can be approximated by reusing the low-level
thermal correction:

$$
G_M^{\mathrm{high}} \;\approx\; E_M^{\mathrm{high,SP}} +
\left( G_M^{\mathrm{GFN2}} - E_M^{\mathrm{GFN2,SP}} \right)
$$

so for a reaction with species $\{A, B, P\}$ (donor, acceptor, product):

$$
\Delta G_{\mathrm{high}} \;\approx\; \Delta G_{\mathrm{GFN2}} +
\Delta\Delta E_{\mathrm{SP}}, \qquad
\Delta\Delta E_{\mathrm{SP}} = \delta E_P - \delta E_A - \delta E_B, \qquad
\delta E_M = E_M^{\mathrm{high,SP}} - E_M^{\mathrm{GFN2,SP}}
$$

This is exactly what `cross_benzoin/cb_featurize.py`'s `_g_gxtb()` computes
operationally for the g-xTB level (one g-xTB SP per species, reusing the
GFN2 `ohess` thermal correction), and it is the same identity underlying
this project's BDE/BDFE descriptor gain (see [[bde-descriptor-idea]]) and
the g-xTB→DFT correction hierarchy (see [[gxtb-dft-correction-champion]]).

## 3. Recommended ML target: per-species correction, stoichiometric recombination

Rather than learning $(\mathrm{SMILES}_A, \mathrm{SMILES}_B) \to \Delta
G_{\mathrm{DFT}}$ directly, learn a **per-species** correction and recombine
it according to reaction stoichiometry:

$$
f(M) = E_M^{\mathrm{high}} - E_M^{\mathrm{GFN2}}, \qquad
\widehat{\Delta G} = \Delta G_{\mathrm{GFN2}} + f(P) - f(A) - f(B)
$$

Advantages over a monolithic pair→ΔG regressor:

- **Thermodynamic additivity is enforced by construction**, not learned
  approximately;
- a species (an aldehyde, or a product shared across several directed
  reactions) has its correction computed once and reused/amortized across
  every reaction it participates in;
- it degrades gracefully and *automatically* to the homo case
  $A = B$ with no architecture change:
  $$
  \widehat{\Delta G}_{\mathrm{homo}} = \Delta G_{\mathrm{GFN2}} + f(P) - 2f(A)
  $$
- it licenses an explicit **homo↔cross consistency regularizer** when
  training a joint model:
  $$
  \mathcal{L}_{\mathrm{consistency}} =
  \big| f_{\mathrm{cross}}(A, A) - f_{\mathrm{homo}}(A) \big|
  $$

## 4. Conformer ensemble free energy

For a species with multiple relevant conformers $i$ with free energies
$G_i$, the physically correct free energy is the Boltzmann-weighted
ensemble average, not the single lowest conformer:

$$
G_{\mathrm{ensemble}} = -RT \ln \sum_i \exp(-G_i / RT)
$$

This project's `funnel_v3` conformer search currently reports the
**single lowest-energy conformer's** free energy (deterministic + RMSD +
topology-guard funnel, see [[crest-conformer-search]] /
[[conformer-search-noise]]), which is a reasonable practical approximation
but not the same quantity as $G_{\mathrm{ensemble}}$ above — a documented
approximation, not an error, and a candidate refinement direction (the
`boltz_relabel_worker.py` / `collect_boltz_relabel.py` probe already tests a
$K=5$ Boltzmann-corrected relabeling against this).

## 5. Role-aware descriptor construction

For continuous/QM descriptors $\phi(\cdot)$, use the stoichiometric
difference, matching the energy decomposition in §3:

$$
\phi_{\mathrm{rxn}} = \phi(P) - \phi(A) - \phi(B)
$$

while retaining the raw per-species blocks so the model can see both the
compositional signal and the individual-species context:

$$
\text{input} = \big[\, \phi(A),\ \phi(B),\ \phi(P),\ \phi_{\mathrm{rxn}} \,\big]
$$

Fingerprints (e.g. Morgan/ECFP) are concatenated or combined via
count-difference/XOR rather than subtracted directly (subtraction is only
meaningful for continuous, roughly additive quantities). Every descriptor
is assigned a **role prefix** — `donor_*`, `acceptor_*`, `product_*`,
`interaction_*` — because donor and acceptor play chemically different
roles (nucleophile vs. electrophile) even when computed with the same raw
descriptor function; see `DESCRIPTOR_POLICY_CROSS.md` for the full block
table (HOMO/Fukui⁻ for donor, LUMO/Fukui⁺ for acceptor, etc.) and
interaction terms (donor HOMO − acceptor LUMO gap, Fukui-pair product,
steric/charge mismatch).

## 6. Model input construction (shared species encoder)

For a GNN or any learned per-species encoder $\mathrm{Encoder}(\cdot)$:

$$
h_A = \mathrm{Encoder}(A), \quad h_B = \mathrm{Encoder}(B), \quad
h_P = \mathrm{Encoder}(P)
$$
$$
h_{\mathrm{rxn}} = \big[\, h_A,\ h_B,\ h_P,\ h_P - h_A - h_B,\
|h_A - h_B|,\ h_A \odot h_B \,\big]
$$

The encoder must **not** be made permutation-invariant over the
donor/acceptor pair: swapping roles changes which carbonyl becomes ketone
vs. carbinol, hence changes both the product structure and $\Delta G$ (see
§7 empirical confirmation from `cross_pilot_v1`). Use explicit
donor/acceptor/product role embeddings instead.

## 7. Directed vs. unordered pairs — why the distinction matters

An **unordered pair** $\{A, B\}$ names a candidate combination of two
aldehydes with no claim about which plays which role. But the NHC-catalyzed
benzoin mechanism is **not symmetric** in the general (cross, $A \neq B$)
case: exactly one aldehyde is deprotonated/activated by the catalyst into
the nucleophilic acyl-anion equivalent (the **donor**), and it attacks the
carbonyl carbon of the other, unactivated aldehyde (the **acceptor**).
Swapping which molecule plays which role changes which carbon ends up as
the new ketone carbon and which as the new carbinol (alcohol) carbon —
**two chemically distinct regioisomeric products**, generally with two
different $\Delta G$ values. So every unordered pair $\{A, B\}$ with
$A \neq B$ must be expanded into **two directed reactions**,
$(A{\to}\text{donor}, B{\to}\text{acceptor})$ and
$(B{\to}\text{donor}, A{\to}\text{acceptor})$, for the candidate/label space
to be chemically complete; homo ($A = B$) is the degenerate case where the
two directions coincide (absent other stereocenters).

This is not just a theoretical concern: the first real cross-benzoin pilot
run in this project (`data/cross_benzoin/cross_pilot_v1/`, job 24607515)
empirically confirms it — e.g. one unordered pair gave $\Delta
G_{\mathrm{xtb}} = -5.96\,\mathrm{kcal/mol}$ in one direction and
$-11.35\,\mathrm{kcal/mol}$ in the other; a >5 kcal/mol swing purely from
which molecule is nominally "donor" vs. "acceptor," confirming the two
directed rows are genuinely different chemistry, not a metadata-swapped
duplicate.

## 8. Evaluation protocol — four generalization regimes

A single aggregate MAE conflates interpolation with true extrapolation.
Report metrics separately for:

1. **Homo diagonal** ($A = B$) — the regime this project has trained on
   most extensively so far.
2. **New combination** — $A$ and $B$ individually appear elsewhere in
   training, but this specific pair does not (tests combinatorial
   generalization).
3. **Single-side extrapolation** — exactly one of $A, B$ is unseen in
   training.
4. **Double-side extrapolation** — both $A, B$ unseen, ideally with both
   Bemis–Murcko scaffolds unseen — the only regime that tests genuine
   structural generalization into new chemical space.

For each regime, report MAE, RMSE, median AE, P90/P95 AE, max AE, $R^2$,
Spearman $\rho$, and **mean signed error** (systematic bias), not MAE alone.
The two directed rows of one unordered pair must always stay in the same
split — otherwise the model can leak the answer for one direction from
having seen the reverse direction in training.

## 9. Recommended split hierarchy

- **Random pair split** — interpolation ceiling (optimistic upper bound).
- **Molecule-disjoint split** — generalization to unseen combinations of
  otherwise-known monomers (this project's established default outside
  cross-benzoin, see [[data-split-721]]).
- **Double-scaffold-disjoint split** (Bemis–Murcko on both donor and
  acceptor) — the strictest test, genuine extrapolation into new chemical
  space; not yet applied to any cross-benzoin result as of 2026-07-14.
