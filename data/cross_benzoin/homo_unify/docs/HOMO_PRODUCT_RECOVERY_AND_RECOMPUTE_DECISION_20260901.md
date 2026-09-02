# Homo-product recovery / recomputation decision

Date: 2026-09-01

## Decision

Do not recompute all homo-products now.

The restored repository already contains a complete numeric homo-product table:

- `data/cross_benzoin/homo_unify/homo_unify_v1_products.csv`
- rows: 30000
- non-empty errors: 0
- `dG_xtb_kcal`: 30000 / 30000 present
- `dG_gxtb_kcal`: 30000 / 30000 present
- `G_xtb`: 30000 / 30000 present
- `G_gxtb`: 30000 / 30000 present
- `smiles`: 30000 / 30000 present

A gzip backup has been written:

- `data/cross_benzoin/homo_unify/homo_unify_v1_products.csv.gz`

## Missing / degraded part

The `xyz_file` column is populated but points to the old damaged working tree, e.g.

- `/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/chunk_2204/xyz_prod/prod_220545.xyz`

A 200-row sample found 0 existing XYZ paths. Therefore the numeric product table is recovered, but the corresponding product XYZ geometries are not currently available in the restored repository.

## Practical implication

For tabular dG prediction and cross/homo training-table assembly, the full homo-product recomputation is unnecessary because the features and dG values are already present.

For workflows that require product geometries directly, such as some 3D-GNN experiments, geometry audits, or BDE recomputation from saved structures, first search for a compressed `homo_v6` / `xyz_prod` backup. If no backup exists, recompute XYZ only for the subset needed by that downstream task.

## Recommended next steps

1. Preserve the existing CSV and gzip backup.
2. Do not launch a full homo-product featurization campaign unless a downstream task explicitly needs all product XYZ files.
3. For Round10 cross-benzoin featurization, prioritize building/reusing an aldehyde cache from new stage outputs, because Round10 currently lacks broad aldehyde cache coverage.
4. If geometry recovery is needed, restore archived `xyz_prod` files into a clearly named directory and rewrite `xyz_file` paths in a derived copy, not in the original CSV.
