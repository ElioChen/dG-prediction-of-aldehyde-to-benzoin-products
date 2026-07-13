# Role-aware descriptor policy for cross-benzoin ΔG

## Decision

Cross-benzoin needs descriptors for both aldehydes **and** the directed product. The
same aldehyde descriptor calculation can be reused, but its columns must be assigned a
role at table-assembly time. `donor_foo` and `acceptor_foo` are not interchangeable:
the donor carbonyl becomes the ketone site and the acceptor carbonyl becomes the
carbinol site. Homo-benzoin is the constrained case `donor == acceptor`.

## Feature blocks

| Block | Keep or add | Chemical responsibility |
|---|---|---|
| Donor | HOMO, IP, nucleophilicity/Fukui-minus at CHO, O/C charges, C=O WBO, local sterics, global size/shape | Ability of the acyl-anion equivalent to donate and form the new C–C bond |
| Acceptor | LUMO, EA/electrophilicity, Fukui-plus at CHO, carbonyl-C charge, C=O WBO, local sterics, global size/shape | Susceptibility of the electrophilic aldehyde and approach geometry |
| Product | GFN2/g-xTB energy and thermal terms; ketC/ketO/carbC/hydO/hydH charges; new C–C/ketone/carbinol WBO; buried volume, Sterimol, SASA, dispersion, H-bond geometry; conformer/QC fields | Stability and geometry of the actual directed regioisomer |
| Interaction | donor HOMO − acceptor LUMO gap; donor Fukui-minus × acceptor Fukui-plus; charge and steric mismatch; shared-feature absolute differences/products; `P − D − A` stoichiometric differences | Pair complementarity that homo-only data cannot teach |

Use role prefixes even when donor and acceptor share the same raw descriptor schema.
For the GNN, use explicit donor/acceptor/product role embeddings; do not make the pair
encoder permutation-invariant because swapping roles can change the product and ΔG.

## Pruning and leakage rules

Remove identifiers, raw provenance strings, target-derived fields and any DFT-level
descriptor unavailable at inference. Prune exact duplicates, constants, near-zero
variance columns and features with excessive missingness. For highly correlated groups,
retain the chemically interpretable representative unless ablation shows complementary
value. Do not silently median-impute failure: include calculation-status/missingness flags
and route severe failures out of domain.

ADCH/QTAIM are optional high-cost blocks. Existing homo ablations indicate they should
not be back-filled at full scale unless a cross-specific ablation improves molecule-
disjoint validation. The initial cross baseline should prioritize stable xTB, local
reaction-center, steric, conformer and 2D descriptors.

## Required ablations

Report tabular and fusion results for: 2D only; aldehydes only; product only;
donor+acceptor; all raw role blocks; all plus interaction terms; and all minus each
expensive descriptor family. Fit feature selection and imputers on training folds only.
