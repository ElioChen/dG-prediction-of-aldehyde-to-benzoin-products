# Benzoin Aldehyde SMILES Filters

Deep SMILES filtering applied to the raw candidate library
`aldehyde/aldehydes_benzoin.csv` to keep only valid substrates for the benzoin
condensation that are also clean enough for the automated xTB/DFT descriptor +
ΔG pipeline.

Implemented in [`ml/filter_smiles.py`](filter_smiles.py).

```
python ml/filter_smiles.py \
    --input  aldehyde/aldehydes_benzoin.csv \
    --output ml/data/aldehydes_clean.csv \
    --rejects ml/data/aldehydes_rejected.csv
```

## Why filter

The raw library was assembled by a loose substructure match (`[CX3H1]=O`), which
also captures **formamides, formic acid/esters, acyl halides, salts, charged
species, and exotic-element compounds** — none of which are valid mono-aldehyde
benzoin substrates. Feeding these to the pipeline produced blank descriptor
values (xTB non-convergence, failed aldehyde-atom detection, QTAIM BCP misses)
and meaningless ΔG labels. Filtering upstream removes that noise once, so every
downstream stage (fingerprint selection → descriptors → ΔG → ML) works on clean
data.

## Filter rules

A molecule is **kept** only if it passes all of the following (evaluated in order;
first failure is the recorded reason):

| # | Rule | Reject reason | Rationale |
|---|------|---------------|-----------|
| 1 | RDKit can parse the SMILES | `invalid_parse` | Unparseable / empty input |
| 2 | No `.` in the SMILES | `multi_component` | Salts / mixtures / counter-ions are not single substrates; the "CHO" is often a separate formate/acid-chloride ion |
| 3 | Net formal charge == 0 | `charged` | Free ions (e.g. oxonium `[OH2+]`, phosphonium) are not neutral substrates. *Net* charge — internally charge-separated groups like nitro `[N+](=O)[O-]` (net 0) are kept |
| 4 | All atoms ∈ {H, C, N, O, F, S, Cl, Br, I} | `disallowed_element` | Common organic set for benzoin aldehydes; excludes P, B, Se, Si, metals, etc. that xTB/DFT handle poorly or that aren't relevant substrates |
| 5 | Exactly one **true** aldehyde `[CX3H1](=O)[#6]` | `not_single_aldehyde` | R–CHO with **R = carbon**. Excludes formic acid/ester/amide, acyl halide, formaldehyde (0 true aldehydes) and di-aldehydes (≥2, ambiguous for benzoin) |

Key SMARTS:

- True aldehyde: `[CX3H1](=O)[#6]` — carbonyl carbon with exactly one H, a double
  bond to O, **and a carbon neighbour**. This `[#6]` requirement is what separates
  a real aldehyde from a formamide (`N–CHO`), formate (`O–CHO`), formic acid, or
  acyl halide (`X–CHO`).

Allowed elements (atomic numbers): `{1, 6, 7, 8, 9, 16, 17, 35, 53}`.

## Results on `aldehydes_benzoin.csv` (2026-06-11)

Input: **291,145** rows.

| Category | Count | % |
|----------|------:|----:|
| **keep** (clean true aldehydes) | **217,975** | 74.87 |
| not_single_aldehyde | 49,650 | 17.05 |
| multi_component | 14,249 | 4.89 |
| disallowed_element | 7,408 | 2.54 |
| charged | 1,863 | 0.64 |
| invalid_parse | 0 | 0.00 |

**Removed 73,170 (25.1%) noise.**

### Verification of the largest rejected class

All **49,650** `not_single_aldehyde` rejects are **0-true-aldehyde** cases
(formamides, formic acid, formate esters); **zero** were di-aldehydes — i.e. no
valid mono-aldehyde substrate was lost. Representative rejects:

- `CN(C)C=O` (DMF — formamide)
- `C(=O)O` (formic acid)
- `CCOC=O` (ethyl formate)
- `C(=O)N` (formamide)

Other categories (examples):
- multi_component: `...[P+](...)...F.C(=O)[O-]` (phosphonium + formate salt)
- disallowed_element: `CCC1=CC=C([Se]1)C=O` (selenophene-2-carbaldehyde)
- charged: `...C=O)[OH2+]` (oxonium)

## Outputs

- `ml/data/aldehydes_clean.csv` — 217,975 kept rows (original columns preserved).
- `ml/data/aldehydes_rejected.csv` — 73,170 rows + a `reject_reason` column.

This clean library is the input to fingerprint-based representative selection
([`ml/select_subset.py`](select_subset.py)); the descriptor and ΔG pipelines then
run only on the selected representatives.
