#!/usr/bin/env python
"""Does restricting to AROMATIC aldehydes improve g-xTB→DFT correction accuracy?
Diagnostic only — aromatic-only was DROPPED for production scope (aliphatic also reacts).

Compares Ridge(linear) / MLP / GBT across scopes {all, aromatic, aliphatic}, Δ-learning
(DFT − g-xTB) on 34 QM descriptors, random 70/20/10 seed 42, test MAE on absolute ΔG.
Also the FAIR specialization test: model trained on ALL vs aromatic-specialist, both
evaluated on the SAME aromatic test molecules.
"""
import glob
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
ROOT = "/scratch-shared/schen3/benzoin-dg"
PROD = f"{ROOT}/data/cross_benzoin/homo_v6/products_all.csv"
DFTDIR = f"{ROOT}/data/raw/dft_sp_funnelv3"
CLS = f"{ROOT}/data/cross_benzoin/homo_v6/aldehyde_class.parquet"
OUT = Path(f"{ROOT}/data/cross_benzoin/homo_v6/viz_gxtb_20260625")
QM = ["xtb_HOMO", "xtb_LUMO", "xtb_gap", "xtb_IP", "xtb_EA", "xtb_mu", "xtb_eta", "xtb_omega",
      "xtb_dipole", "mulliken_ketC", "mulliken_ketO", "mulliken_carbC", "mulliken_hydO",
      "mulliken_hydH", "wbo_CO_ket", "wbo_CC_new", "wbo_CO_carb", "fukui_plus_ketC",
      "fukui_minus_ketC", "dual_ketC", "fukui_plus_carbC", "fukui_minus_carbC", "dual_carbC",
      "vbur_ketC", "vbur_carbC", "sterimol_L", "sterimol_B1", "sterimol_B5", "SASA_total",
      "P_int", "pa_ketO", "hb_dist", "hb_angle", "dih_core"]


def split(n, seed=42):
    idx = np.random.default_rng(seed).permutation(n)
    a, b = int(.7 * n), int(.9 * n)
    return idx[:a], idx[a:b], idx[b:]


def models():
    return {
        "Ridge(linear)": lambda: Ridge(alpha=5.0),
        "MLP": lambda: MLPRegressor(hidden_layer_sizes=(256, 128), alpha=1e-4,
                                    max_iter=120, early_stopping=True, n_iter_no_change=8),
        "GBT": lambda: XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, n_jobs=8,
                                    early_stopping_rounds=40, eval_metric="mae"),
    }


def fit_eval(name, mk, Xtr, dtr, Xva, dva, Xte, gte, yte):
    m = mk()
    if name == "GBT":
        m.fit(Xtr, dtr, eval_set=[(Xva, dva)], verbose=False)
    else:
        m.fit(Xtr, dtr)
    yhat = gte + m.predict(Xte)        # absolute ΔG = g-xTB + predicted Δ
    return float(np.abs(yhat - yte).mean()), m


def main():
    fs = sorted(glob.glob(f"{DFTDIR}/chunk_*.csv"))
    dft = pd.concat([pd.read_csv(f, usecols=["id", "dG_orca_kcal"]) for f in fs], ignore_index=True)
    dft = dft.dropna(subset=["dG_orca_kcal"]).drop_duplicates("id")
    p = pd.read_csv(PROD, usecols=["id", "dG_gxtb_kcal"] + QM, low_memory=False)
    cls = pd.read_parquet(CLS)
    df = p.merge(dft, on="id").merge(cls, on="id").dropna(subset=["dG_gxtb_kcal", "dG_orca_kcal"] + QM)
    df = df[df["dG_orca_kcal"].abs() < 60].reset_index(drop=True)
    df["delta"] = df["dG_orca_kcal"] - df["dG_gxtb_kcal"]
    print("labeled+classified:", len(df), df["cls"].value_counts().to_dict(), flush=True)

    rows = []
    scope_sets = {"all": df, "aromatic": df[df.cls == "aromatic"], "aliphatic": df[df.cls == "aliphatic"]}
    for scope, d in scope_sets.items():
        d = d.reset_index(drop=True)
        tr, va, te = split(len(d))
        sc = StandardScaler().fit(d[QM].values[tr])
        Xtr, Xva, Xte = sc.transform(d[QM].values[tr]), sc.transform(d[QM].values[va]), sc.transform(d[QM].values[te])
        dtr, dva = d.delta.values[tr], d.delta.values[va]
        gte, yte = d.dG_gxtb_kcal.values[te], d.dG_orca_kcal.values[te]
        raw_mae = float(np.abs(gte - yte).mean())
        rec = {"scope": scope, "n": len(d), "raw_gxtb": raw_mae}
        for name, mk in models().items():
            mae, _ = fit_eval(name, mk, Xtr, dtr, Xva, dva, Xte, gte, yte)
            rec[name] = mae
            print(f"  {scope:9s} {name:14s} MAE={mae:.3f}", flush=True)
        rows.append(rec)

    # ---- fair specialization test: ALL-trained vs aromatic-specialist on SAME aromatic test ----
    dar = df[df.cls == "aromatic"].reset_index(drop=True)
    tr_a, va_a, te_a = split(len(dar))
    sc_a = StandardScaler().fit(dar[QM].values[tr_a])
    Xte_a = sc_a.transform(dar[QM].values[te_a]); gte_a = dar.dG_gxtb_kcal.values[te_a]; yte_a = dar.dG_orca_kcal.values[te_a]
    # specialist
    spec = XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, n_jobs=8, early_stopping_rounds=40, eval_metric="mae")
    spec.fit(sc_a.transform(dar[QM].values[tr_a]), dar.delta.values[tr_a],
             eval_set=[(sc_a.transform(dar[QM].values[va_a]), dar.delta.values[va_a])], verbose=False)
    mae_spec = float(np.abs(gte_a + spec.predict(Xte_a) - yte_a).mean())
    # generalist trained on ALL (exclude the aromatic test ids), eval same aromatic test
    test_ids = set(dar.id.values[te_a])
    dall = df[~df.id.isin(test_ids)].reset_index(drop=True)
    tr_g, va_g, _ = split(len(dall))
    sc_g = StandardScaler().fit(dall[QM].values[tr_g])
    gen = XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.05, subsample=0.8,
                       colsample_bytree=0.8, n_jobs=8, early_stopping_rounds=40, eval_metric="mae")
    gen.fit(sc_g.transform(dall[QM].values[tr_g]), dall.delta.values[tr_g],
            eval_set=[(sc_g.transform(dall[QM].values[va_g]), dall.delta.values[va_g])], verbose=False)
    mae_gen = float(np.abs(gte_a + gen.predict(sc_g.transform(dar[QM].values[te_a])) - yte_a).mean())
    print(f"\nSPECIALIZATION (aromatic test): specialist MAE={mae_spec:.3f} vs all-trained MAE={mae_gen:.3f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "scope_arch_results.csv", index=False)
    print("\n=== SCOPE × ARCH test MAE (kcal/mol) ===\n" + res.to_string(index=False), flush=True)

    # figure: grouped bars
    fig, ax = plt.subplots(figsize=(8, 4.5))
    archs = ["raw_gxtb", "Ridge(linear)", "MLP", "GBT"]
    x = np.arange(len(archs)); w = 0.26
    for i, scope in enumerate(["all", "aromatic", "aliphatic"]):
        r = res[res.scope == scope].iloc[0]
        ax.bar(x + (i - 1) * w, [r[a] for a in archs], w, label=f"{scope} (n={r['n']:,})")
    ax.set_xticks(x); ax.set_xticklabels(archs, rotation=15)
    ax.axhline(2.46, color="k", ls=":", lw=1, label="GBT all (full-feat) 2.46")
    ax.set_ylabel("test MAE vs DFT (kcal/mol)")
    ax.set_title("Architecture × scope: does aromatic-only help?")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "28_scope_arch_compare.png", dpi=150)
    print("wrote 28_scope_arch_compare.png", flush=True)

    with open(OUT / "scope_arch_summary.txt", "w") as fh:
        fh.write(res.to_string(index=False) + "\n")
        fh.write(f"\nSpecialization (aromatic test): specialist={mae_spec:.3f}  all-trained={mae_gen:.3f}\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
