#!/usr/bin/env python
"""Tier-3 (corrected) of the 2026-07-10 external-diagnosis review: build a PROPERLY targeted
sample for the multi-conformer Boltzmann relabeling probe.

The 2026-06-26 pilot (boltz_relabel_sample.csv, n=120) was a RANDOM draw from the test split,
not the hard tail, used only K=5 conformers, and compared against a STALE June model's
predictions (products_dG_corrected_FINAL_20260626.csv). It found relabeling made the frozen
model's apparent MAE WORSE (0.97->1.28) -- see descriptor-search-exhausted memory's 2026-07-10
correction. That result says little about whether label noise matters specifically for the
molecules that actually set the error floor.

This builds a new sample: the top-N highest-|error| molecules from the CURRENT champion's test
set (test_predictions_MORDREDSLIM271_BDEGXTB_20260706.csv) that also carry a sulfonyl/P/imine/
amide tag (the confirmed 11.2x/9.5x/3.6x/3.4x-enriched hard subset) -- i.e. exactly the
molecules whose error the champion model has NOT been able to explain. Compared against the
CURRENT champion's own dG_pred (not a stale June model).
"""
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

R = "/scratch-shared/schen3/benzoin-dg"; H = f"{R}/data/cross_benzoin/homo_v6"
OUT = f"{H}/viz_gxtb_20260625"
N_SAMPLE = 150

SMARTS = {
    "sulfonyl": "[#16X4](=[OX1])(=[OX1])",
    "has_P": "[#15]",
    "imine": "[CX3]=[NX2]",
    "amide": "[CX3](=O)[NX3]",
}
PATS = {k: Chem.MolFromSmarts(v) for k, v in SMARTS.items()}


def is_hard(ald_smi, prod_smi):
    ma, mp = Chem.MolFromSmiles(str(ald_smi)), Chem.MolFromSmiles(str(prod_smi))
    for m in (ma, mp):
        if m is None: continue
        for p in PATS.values():
            if m.HasSubstructMatch(p): return True
    return False


def main():
    te = pd.read_csv(f"{OUT}/test_predictions_MORDREDSLIM271_BDEGXTB_20260706.csv")
    te["hard"] = [is_hard(a, s) for a, s in zip(te["ald_smiles"], te["smiles"])]
    hard = te[te["hard"]].copy().sort_values("error", ascending=False)
    print(f"hard-tail candidates: {len(hard):,}; taking top {N_SAMPLE} by |error|", flush=True)
    samp = hard.head(N_SAMPLE)[["id", "smiles", "ald_smiles", "dG_orca_kcal", "dG_gxtb_kcal", "dG_pred", "error"]].rename(
        columns={"smiles": "prod_smiles"})
    samp.to_csv(f"{OUT}/boltz_relabel_hardtail_sample_20260710.csv", index=False)
    print(f"wrote boltz_relabel_hardtail_sample_20260710.csv "
          f"(error range {samp['error'].min():.2f}-{samp['error'].max():.2f} kcal/mol)", flush=True)


if __name__ == "__main__":
    main()
