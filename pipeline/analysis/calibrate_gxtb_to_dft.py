#!/usr/bin/env python
"""Calibrate full-library g-xTB ΔG to DFT (r2SCAN-3c) — correction hierarchy + MAE.

KEY CONTEXT (reframes the earlier g-xTB-vs-GFN2 report): joined to the in-progress DFT-SP
funnel_v3 labels, g-xTB is ALREADY close to DFT (MAE~4.3) while GFN2 is far (MAE~15.5).
So DFT is the right correction target and g-xTB the right Δ-baseline. We fit, on a 70/20/10
split, a hierarchy of corrections to g-xTB and report held-out TEST error:
  L0 raw g-xTB | L1 +constant bias | L2 affine(a*g-xTB+b) | L3 ridge Δ(descriptors)
  | L4 GBT Δ(descriptors)   where Δ = DFT − g-xTB (Δ-learning).
g-xTB-only features (deployment: no need to also run GFN2). One PNG per figure, dated dir.
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "savefig.bbox": "tight"})

ROOT = Path("/scratch-shared/schen3/benzoin-dg")
PROD = ROOT / "data/cross_benzoin/homo_v6/products_all.csv"
DFTDIR = ROOT / "data/raw/dft_sp_funnelv3"
STAMP = "20260625"
OUT = ROOT / "data/cross_benzoin/homo_v6" / f"viz_gxtb_{STAMP}"
OUT.mkdir(exist_ok=True)
REPORT = OUT / f"REPORT_gxtb_to_dft_calibration_{STAMP}.md"

FEATS = ["xtb_HOMO", "xtb_LUMO", "xtb_gap", "xtb_IP", "xtb_EA", "xtb_mu", "xtb_eta",
         "xtb_omega", "xtb_dipole",
         "mulliken_ketC", "mulliken_ketO", "mulliken_carbC", "mulliken_hydO", "mulliken_hydH",
         "wbo_CO_ket", "wbo_CC_new", "wbo_CO_carb",
         "fukui_plus_ketC", "fukui_minus_ketC", "dual_ketC",
         "fukui_plus_carbC", "fukui_minus_carbC", "dual_carbC",
         "vbur_ketC", "vbur_carbC", "sterimol_L", "sterimol_B1", "sterimol_B5",
         "SASA_total", "P_int", "pa_ketO", "hb_dist", "hb_angle", "dih_core"]


def metrics(y, yhat):
    e = yhat - y
    return dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()),
                bias=e.mean(), R2=1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum())


def save(fig, name):
    fig.savefig(OUT / name); plt.close(fig); print("  wrote", name)


def parity(y, yhat, title, name, lim=(-25, 30)):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.hexbin(y, yhat, gridsize=60, bins="log", cmap="viridis", mincnt=1, extent=(*lim, *lim))
    ax.plot(lim, lim, "w--", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
    m = metrics(y, yhat)
    ax.set_xlabel("DFT r2SCAN-3c ΔG (kcal/mol)"); ax.set_ylabel("predicted ΔG (kcal/mol)")
    ax.set_title(f"{title}\nMAE={m['MAE']:.2f} RMSE={m['RMSE']:.2f} bias={m['bias']:+.2f} R²={m['R2']:.2f}")
    save(fig, name)


def main():
    import glob
    print("loading DFT labels + products")
    fs = sorted(glob.glob(str(DFTDIR / "chunk_*.csv")))
    dft = pd.concat([pd.read_csv(f, usecols=["id", "dG_orca_kcal"]) for f in fs], ignore_index=True)
    dft = dft.dropna(subset=["dG_orca_kcal"]).drop_duplicates("id")
    p = pd.read_csv(PROD, low_memory=False)
    df = p.merge(dft, on="id", how="inner")
    df = df.dropna(subset=["dG_gxtb_kcal", "dG_orca_kcal"] + FEATS)
    # clamp pathological labels (broken SCF) out of fit; report count
    n_all = len(df)
    df = df[df["dG_orca_kcal"].abs() < 60]
    print(f"matched {n_all:,}; usable after |DFT|<60 + full-feature: {len(df):,}")

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(df))
    n = len(df); ntr, nva = int(.7 * n), int(.2 * n)
    tr, va, te = idx[:ntr], idx[ntr:ntr + nva], idx[ntr + nva:]
    D = df.reset_index(drop=True)
    y = D["dG_orca_kcal"].values
    g = D["dG_gxtb_kcal"].values
    X = D[FEATS].values
    delta = (y - g)  # Δ-learning target

    rows = []

    def ev(name, yhat_te):
        m = metrics(y[te], yhat_te); m["model"] = name; rows.append(m)
        print(f"  {name:28s} MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f} bias={m['bias']:+.3f} R2={m['R2']:.3f}")

    # L0 raw g-xTB
    ev("L0 raw g-xTB", g[te])
    # also raw GFN2 for contrast
    if "dG_xtb_kcal" in D:
        ev("(ref) raw GFN2-xTB", D["dG_xtb_kcal"].values[te])
    # L1 constant bias
    b = delta[tr].mean()
    ev("L1 +constant bias", g[te] + b)
    # L2 affine DFT ~ a*g + b
    lr = LinearRegression().fit(g[tr].reshape(-1, 1), y[tr])
    ev("L2 affine(a*g-xTB+b)", lr.predict(g[te].reshape(-1, 1)))
    # L3 ridge Δ(descriptors)
    sc = StandardScaler().fit(X[tr])
    rg = Ridge(alpha=5.0).fit(sc.transform(X[tr]), delta[tr])
    ev("L3 ridge Δ(descriptors)", g[te] + rg.predict(sc.transform(X[te])))
    # L4 GBT Δ(descriptors), early-stopped on val
    gb = XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, n_jobs=8,
                      early_stopping_rounds=40, eval_metric="mae")
    gb.fit(X[tr], delta[tr], eval_set=[(X[va], delta[va])], verbose=False)
    yhat_gbt = g[te] + gb.predict(X[te])
    ev("L4 GBT Δ(descriptors)", yhat_gbt)

    res = pd.DataFrame(rows).set_index("model")

    # ---- figures ----
    parity(y[te], g[te], "L0 raw g-xTB vs DFT", "20_parity_L0_raw_gxtb.png")
    parity(y[te], yhat_gbt, "L4 GBT-corrected g-xTB vs DFT", "21_parity_L4_gbt_corrected.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    order = [m for m in res.index if not m.startswith("(ref)")]
    ax.bar(range(len(order)), res.loc[order, "MAE"],
           color=["#cb181d", "#fb6a4a", "#fdae6b", "#74c476", "#238b45"][:len(order)])
    ax.axhline(3.2, color="k", ls=":", label="~3.2 noise floor")
    for i, mname in enumerate(order):
        ax.text(i, res.loc[mname, "MAE"] + .1, f"{res.loc[mname,'MAE']:.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([m.split(" ", 1)[0] for m in order])
    ax.set_ylabel("test MAE vs DFT (kcal/mol)")
    ax.set_title("g-xTB → DFT correction hierarchy")
    ax.legend()
    save(fig, "22_correction_hierarchy_MAE.png")

    # feature importance (GBT gain)
    imp = pd.Series(gb.feature_importances_, index=FEATS).sort_values()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(imp.index[-20:], imp.values[-20:], color="#238b45")
    ax.set_xlabel("XGBoost gain importance")
    ax.set_title("Δ-correction drivers (g-xTB→DFT, GBT)")
    save(fig, "23_gbt_feature_importance.png")

    # error vs molecular size (SASA) before/after
    fig, ax = plt.subplots(figsize=(7, 4.5))
    s = D["SASA_total"].values[te]
    ax.scatter(s, np.abs(g[te] - y[te]), s=3, alpha=.15, color="#cb181d", label="L0 raw |err|")
    ax.scatter(s, np.abs(yhat_gbt - y[te]), s=3, alpha=.15, color="#238b45", label="L4 GBT |err|")
    ax.set_xlabel("SASA_total (molecular size proxy)"); ax.set_ylabel("|error vs DFT| (kcal/mol)")
    ax.set_ylim(0, 25); ax.set_title("Error vs size, before/after correction")
    ax.legend()
    save(fig, "24_error_vs_size.png")

    # ---- report ----
    L = [f"# g-xTB → DFT calibration ({STAMP})\n",
         f"Matched products∩DFT-SP(in-progress): **{n_all:,}**; usable (|DFT|<60, full features): **{len(df):,}**.",
         "Split 70/20/10 (train/val/test), seed 42. Target = DFT r2SCAN-3c ΔG. Δ-learning on g-xTB.\n",
         "## Held-out TEST metrics (kcal/mol)\n",
         "| model | MAE | RMSE | bias | R² |", "|---|---|---|---|---|"]
    for mname, r in res.iterrows():
        L.append(f"| {mname} | {r.MAE:.2f} | {r.RMSE:.2f} | {r.bias:+.2f} | {r.R2:.2f} |")
    raw = res.loc["L0 raw g-xTB", "MAE"]; best = res.loc["L4 GBT Δ(descriptors)", "MAE"]
    L += ["",
          "## Findings\n",
          f"- **g-xTB raw MAE = {raw:.2f}** vs DFT — already far better than GFN2 "
          f"({res.loc['(ref) raw GFN2-xTB','MAE']:.1f}). g-xTB is the correct Δ-baseline.",
          f"- A **constant bias removal** alone -> MAE {res.loc['L1 +constant bias','MAE']:.2f} "
          f"(bias {res.loc['L0 raw g-xTB','bias']:+.2f} -> ~0): most of the raw error is a fixed offset.",
          f"- **Descriptor Δ-correction (GBT) -> MAE {best:.2f}, R²={res.loc['L4 GBT Δ(descriptors)','R2']:.2f}**, "
          f"a {100*(raw-best)/raw:.0f}% cut from raw, approaching the ~3.2 kcal noise floor.",
          "- Top correction drivers: see fig 23 (electronic IP/HOMO/ω + carbonyl charge/Fukui), "
          "consistent with where xTB-family methods misjudge EWG electronics.",
          "",
          "## Figures",
          "- `20_parity_L0_raw_gxtb.png`, `21_parity_L4_gbt_corrected.png`",
          "- `22_correction_hierarchy_MAE.png`, `23_gbt_feature_importance.png`, `24_error_vs_size.png`"]
    REPORT.write_text("\n".join(L))
    res.to_csv(OUT / "gxtb_to_dft_metrics.csv")
    print("report ->", REPORT)


if __name__ == "__main__":
    main()
