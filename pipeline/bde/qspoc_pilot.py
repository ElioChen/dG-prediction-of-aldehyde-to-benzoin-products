#!/usr/bin/env python
"""Q-SPOC pilot (2026-07-15, user-suggested https://github.com/DeepSynthesis/Q-SPOC): does
targeting the formyl C-H bond specifically (atom_lists/bond_lists -> Q-SPOC's 'atom'/'bond'
descriptor categories: CM5 charge, Wiberg bond order, bond dipole) plus its whole-molecule
CDFT/ESP surface descriptors (not in this project's own data) beat H-SPOC's R2=0.651?

Atom indices: reuses calc_bde_alfabet.target_bond_aldehyde's canonicalize-then-SMARTS logic
(same fix as the two ALFABET bugs earlier this session) to get the formyl C/H atom indices
on the SAME canonical-SMILES parse Q-SPOC will use internally (RDKit's AddHs is
deterministic -- new explicit H atoms always appended after heavy atoms in their original
order -- so these indices should line up with whatever 3D structure Q-SPOC generates from
that canonical string, assuming it uses the standard RDKit ETKDG+AddHs path like everything
else in this project's pipeline).

Usage:
  ENV=/gpfs/scratch1/shared/schen3/envs/qspoc
  $ENV/bin/python qspoc_pilot.py --n 300 --out /tmp/qspoc_pilot_result.json
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "compute"))
from qc import qc_filter  # noqa: E402
from splits import molecule_cold_split  # noqa: E402

RDLogger.DisableLog("rdApp.*")
H = Path("/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6")
FORMYL_CH = Chem.MolFromSmarts("[CX3H1](=O)")


def formyl_atom_indices(smiles):
    """Return (canonical_smiles, c_idx, h_idx) on the canonical-order AddHs mol, or None."""
    mol0 = Chem.MolFromSmiles(smiles)
    if mol0 is None:
        return None
    csmi = Chem.MolToSmiles(mol0)
    mol = Chem.MolFromSmiles(csmi)
    match = mol.GetSubstructMatch(FORMYL_CH)
    if not match:
        return None
    c_idx = match[0]
    molH = Chem.AddHs(mol)
    h_idx = next((nbr.GetIdx() for nbr in molH.GetAtomWithIdx(c_idx).GetNeighbors()
                  if nbr.GetSymbol() == "H"), None)
    if h_idx is None:
        return None
    return csmi, c_idx, h_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work-dir", default="/scratch-local/qspoc_pilot")
    ap.add_argument("--multiwfn-path", default="/home/schen3/mutiwfn/Multiwfn_noGUI")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    labels = pd.read_csv(H / "aldehydes_bdfe_gxtb_descriptors.csv", dtype={"id": str}) \
        .dropna(subset=["bde_gxtb_kcal"])
    labels = labels[qc_filter(labels["bde_gxtb_kcal"])]
    mol = pd.read_csv(H / "aldehydes_all.csv", usecols=["id", "smiles", "error"], dtype=str,
                       keep_default_na=False)
    mol = mol[(mol["error"] == "") & (mol["smiles"] != "")]
    df = labels.merge(mol, on="id", how="inner")
    df = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)

    rows = []
    for _, r in df.iterrows():
        hit = formyl_atom_indices(r["smiles"])
        if hit is None:
            continue
        csmi, c_idx, h_idx = hit
        rows.append({"id": r["id"], "bde_gxtb_kcal": r["bde_gxtb_kcal"],
                     "smiles_list": csmi, "name_list": r["id"],
                     "atom_lists": str([c_idx, h_idx]),
                     "bond_lists": str([[c_idx, h_idx]])})
    work = pd.DataFrame(rows)
    print(f"{len(work)}/{len(df)} matched formyl C-H", flush=True)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    input_csv = work_dir / "input.csv"
    work[["smiles_list", "name_list", "atom_lists", "bond_lists"]].to_csv(input_csv, index=False)

    settings_src = Path("/home/schen3/mutiwfn/settings.ini")
    (work_dir / "settings.ini").write_text(settings_src.read_text())

    qspoc_bin = str(Path(sys.executable).parent / "qspoc")
    cmd = [qspoc_bin, "desc", str(input_csv), "--save-dir", str(work_dir / "results"),
           "--precision", "xtb", "--version", "all", "--include", "cdft,esp,atom,bond",
           "--multiwfn-path", args.multiwfn_path]
    print("running:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(work_dir), capture_output=True, text=True, timeout=7200)
    print(r.stdout[-3000:], flush=True)
    print(r.stderr[-3000:], flush=True)
    if r.returncode != 0:
        print(f"qspoc failed with exit code {r.returncode}", flush=True)
        sys.exit(1)

    desc = pd.read_csv(work_dir / "results" / "multiwfn_descriptors.csv")
    desc = desc.rename(columns={"name": "id"})
    desc["id"] = desc["id"].astype(str)
    merged = work.merge(desc, on="id", how="inner")
    print(f"{len(merged)} rows with descriptors computed", flush=True)

    feat_cols = [c for c in desc.columns if c != "id"]
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
    from xgboost import XGBRegressor

    X = merged[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y = merged["bde_gxtb_kcal"].to_numpy(dtype=float)
    split = molecule_cold_split(merged["id"], test_frac=0.2, seed=args.seed)
    tr, te = (split == "train").to_numpy(), (split == "test").to_numpy()
    print(f"train={tr.sum()} test={te.sum()}", flush=True)

    model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                          random_state=args.seed, n_jobs=-1)
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])

    result = {
        "n": len(merged), "n_features": len(feat_cols),
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "MAE": float(mean_absolute_error(y[te], pred)),
        "RMSE": float(root_mean_squared_error(y[te], pred)),
        "R2": float(r2_score(y[te], pred)),
        "spearman_rho": float(spearmanr(y[te], pred).correlation),
        "feature_importance": dict(sorted(zip(feat_cols, model.feature_importances_.tolist()),
                                           key=lambda kv: -kv[1])),
    }
    print(json.dumps(result, indent=2))
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
