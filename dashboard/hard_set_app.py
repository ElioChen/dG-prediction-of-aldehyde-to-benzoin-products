"""Streamlit dashboard for the project-wide "hard set" -- the molecules our models predict
worst, across every task (homo BDE aldehyde / homo BDE product / homo dG / cross dG), both
reactant and product side, tagged by *why* they are hard.

Run:
    /home/schen3/venv/nhc-workflow/bin/streamlit run dashboard/hard_set_app.py

Data comes from `data/analysis/hard_set/hard_set.parquet`, (re)built by
`pipeline/analysis/build_hard_set.py`. Re-run that after any model retrain and refresh the
page -- the set is meant to shrink as we overcome each class of difficulty. The Progress
tab reads `hard_set_history/` snapshots to show what has dropped out.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

REPO = Path(__file__).resolve().parents[1]
HS_DIR = REPO / "data/analysis/hard_set"
HARD_SET = HS_DIR / "hard_set.parquet"
HISTORY = HS_DIR / "hard_set_history"

CAUSE_GLOSSARY = {
    "gxtb_baseline_failure": "|g-xTB baseline − DFT| ≥ 15 kcal/mol. Δ-learning assumes the "
        "baseline is roughly right; here it is not, so a smooth ML correction cannot "
        "recover it. See pipeline/analysis/notes/gxtb_baseline_failure_hardtail_20260902.md "
        "and pipeline/bde/notes/nitro_delta_sp_bimodal_20260902.md.",
    "flexible": "≥12 rotatable bonds or MW ≥ 500. The single-conformer DFT label is noisier "
        "for these (conformer spread ~2 kcal); also under-represented in training.",
    "aliphatic": "No aromatic ring. Aliphatic aldehydes/products are systematically harder "
        "than aromatic across both projects (homo dG 2.03 vs 1.40).",
    "heavy_heteroatom": "Contains Se / Te / As / Br / I.",
    "high_uncertainty": "Model's own uncertainty (PI width) is in the top decile -- these "
        "are the ones route_to_dft already flags.",
    "implausible_label": "DFT label outside the task's physical window -- a data-quality "
        "outlier, not model difficulty.",
    "implausible_baseline": "g-xTB baseline outside the task's physical window.",
    "unexplained": "No structural/data reason found yet. The frontier -- these are where we "
        "still do not know why the model fails.",
}
FG_NOTE = ("fg:* = carries a known-hard substructure. Phosphorus (esp. P=O / phosphonium), "
           "sulfonyl and heavy amide substitution are where g-xTB's electronic structure is "
           "least reliable.")

TASK_LABELS = {
    "homo_dg": "homo ΔG (self-condensation)",
    "cross_dg": "cross ΔG (A + B, A≠B)",
    "bde_aldehyde": "homo BDE — aldehyde formyl C–H",
    "bde_product": "homo BDE — product ketC–carbC",
}


@st.cache_data(show_spinner=False)
def load_hard_set() -> pd.DataFrame:
    if not HARD_SET.exists():
        return pd.DataFrame()
    df = pd.read_parquet(HARD_SET)
    df["causes"] = df["causes"].map(lambda c: list(c) if c is not None else [])
    return df


@st.cache_data(show_spinner=False)
def load_history() -> pd.DataFrame:
    rows = []
    if HISTORY.exists():
        for d in sorted(HISTORY.glob("*/hard_set_summary.json")):
            try:
                s = json.loads(d.read_text())
            except json.JSONDecodeError:
                continue
            rows.append({"snapshot": d.parent.name, "generated_at": s.get("generated_at"),
                         "n_total": s.get("n_total"), **{f"cause:{k}": v
                         for k, v in s.get("by_cause", {}).items()},
                         **{f"task:{k}": v for k, v in s.get("by_task", {}).items()}})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def mol_svg(smiles: str, w: int = 340, h: int = 240) -> str | None:
    if not isinstance(smiles, str) or not smiles:
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    d = rdMolDraw2D.MolDraw2DSVG(w, h)
    d.drawOptions().addStereoAnnotation = True
    rdMolDraw2D.PrepareAndDrawMolecule(d, m)
    d.FinishDrawing()
    return d.GetDrawingText()


def svg_html(svg: str | None) -> str:
    return svg if svg else "<div style='color:#999'>— no structure —</div>"


st.set_page_config(page_title="benzoin-dg hard set", layout="wide")
hs = load_hard_set()

if hs.empty:
    st.error("`data/analysis/hard_set/hard_set.parquet` not found. Build it first:\n\n"
             "`/home/schen3/venv/nhc-workflow/bin/python pipeline/analysis/build_hard_set.py`")
    st.stop()

st.title("benzoin-dg — hard set")
st.caption(f"molecules our models predict worst, tagged by cause · generated "
           f"{hs['generated_at'].iloc[0]} · {len(hs):,} rows · rebuild with "
           f"`pipeline/analysis/build_hard_set.py`")

# ---------------- sidebar filters ----------------
sb = st.sidebar
sb.header("filters")
tasks = sorted(hs["task"].unique())
pick_tasks = sb.multiselect("task", tasks, default=tasks,
                            format_func=lambda t: TASK_LABELS.get(t, t))
all_causes = sorted({c for cs in hs["causes"] for c in cs})
pick_causes = sb.multiselect("cause (any of)", all_causes, default=[])
cause_mode = sb.radio("cause match", ["any selected", "all selected", "only these"],
                      horizontal=False, disabled=not pick_causes)
res_lo, res_hi = float(hs["abs_residual"].min()), float(hs["abs_residual"].max())
res_min = sb.slider("min |residual| (kcal/mol)", res_lo, min(res_hi, 40.0),
                    value=res_lo, step=0.5)
smi_q = sb.text_input("SMILES contains", "")
sb.divider()
sb.markdown(f"**cause glossary**\n\n" +
            "\n\n".join(f"**`{k}`** — {v}" for k, v in CAUSE_GLOSSARY.items()) +
            f"\n\n{FG_NOTE}")

f = hs[hs["task"].isin(pick_tasks) & (hs["abs_residual"] >= res_min)].copy()
if pick_causes:
    ps = set(pick_causes)
    if cause_mode == "any selected":
        f = f[f["causes"].map(lambda c: bool(ps & set(c)))]
    elif cause_mode == "all selected":
        f = f[f["causes"].map(lambda c: ps <= set(c))]
    else:
        f = f[f["causes"].map(lambda c: set(c) <= ps and bool(c))]
if smi_q:
    q = smi_q.strip()
    f = f[f["smiles"].str.contains(q, na=False, regex=False) |
          f.get("ald_smiles", pd.Series(index=f.index)).str.contains(q, na=False, regex=False)]

tab_over, tab_browse, tab_prog = st.tabs(["overview", "browse molecules", "progress"])

# ---------------- overview ----------------
with tab_over:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("hard rows (filtered)", f"{len(f):,}", f"{len(f) - len(hs):,} vs all")
    c2.metric("mean |residual|", f"{f['abs_residual'].mean():.2f}" if len(f) else "–")
    c3.metric("worst |residual|", f"{f['abs_residual'].max():.1f}" if len(f) else "–")
    c4.metric("tasks in view", len(f["task"].unique()))

    if len(f):
        ex = f.explode("causes")
        by_cause = (ex.groupby(["causes", "task"]).size().reset_index(name="n")
                    .sort_values("n", ascending=False))
        st.plotly_chart(px.bar(by_cause, x="n", y="causes", color="task", orientation="h",
                               title="hard rows by cause (stacked by task)",
                               color_discrete_map={t: px.colors.qualitative.Safe[i]
                                                   for i, t in enumerate(tasks)}),
                        use_container_width=True)
        colA, colB = st.columns(2)
        with colA:
            tm = (f.groupby("task")["abs_residual"].agg(["count", "mean", "max"])
                  .rename(columns={"count": "n", "mean": "MAE_hard", "max": "worst"}).round(2))
            tm.index = tm.index.map(lambda t: TASK_LABELS.get(t, t))
            st.dataframe(tm, use_container_width=True)
        with colB:
            piv = (ex.pivot_table(index="causes", columns="task", values="id",
                                  aggfunc="count", fill_value=0))
            st.plotly_chart(px.imshow(piv, text_auto=True, aspect="auto",
                                      title="cause × task"), use_container_width=True)

# ---------------- browse ----------------
with tab_browse:
    st.caption(f"{len(f):,} molecules match. Sorted worst-first. Structures rendered on demand.")
    sort_col = st.selectbox("sort by", ["abs_residual", "residual", "baseline_err",
                                        "uncertainty", "mw", "rotb"], index=0)
    page_size = st.select_slider("per page", [12, 24, 48, 96], value=24)
    fs = f.sort_values(sort_col, ascending=False, key=lambda s: s.abs()
                       if sort_col in ("residual", "baseline_err") else s)
    n_pages = max(1, (len(fs) + page_size - 1) // page_size)
    pg = st.number_input("page", 1, n_pages, 1) - 1
    view = fs.iloc[pg * page_size:(pg + 1) * page_size]

    for _, r in view.iterrows():
        with st.container(border=True):
            left, right = st.columns([1, 1.1])
            with left:
                st.markdown(f"**{TASK_LABELS.get(r['task'], r['task'])}** · id `{r['id']}`"
                            + (f" · {r['cls']}" if pd.notna(r.get('cls')) else ""))
                cc = st.columns(2)
                cc[0].markdown("reactant / donor" if r["task"] != "homo_dg" else "product")
                cc[0].markdown(svg_html(mol_svg(r["smiles"])), unsafe_allow_html=True)
                if isinstance(r.get("ald_smiles"), str) and r["ald_smiles"]:
                    cc[1].markdown("acceptor / aldehyde")
                    cc[1].markdown(svg_html(mol_svg(r["ald_smiles"])), unsafe_allow_html=True)
            with right:
                st.markdown(
                    f"| | value |\n|--|--|\n"
                    f"| DFT label | {r['label']:.2f} |\n"
                    f"| model pred | {r['pred']:.2f} |\n"
                    f"| **residual** | **{r['residual']:+.2f}** (|{r['abs_residual']:.2f}|, "
                    f"pctile {r['pctile_abs_res']:.2f}) |\n"
                    + (f"| g-xTB baseline | {r['baseline']:.2f} (err {r['baseline_err']:+.2f}) |\n"
                       if pd.notna(r.get("baseline")) else "")
                    + (f"| uncertainty (PI width) | {r['uncertainty']:.2f} |\n"
                       if pd.notna(r.get("uncertainty")) else "")
                    + f"| MW / rot-bonds | {r['mw']:.0f} / {int(r['rotb']) if pd.notna(r['rotb']) else '–'} |\n"
                )
                st.markdown("**causes:** " + " ".join(f"`{c}`" for c in r["causes"]))
                st.code(r["smiles"], language=None)

# ---------------- progress ----------------
with tab_prog:
    hist = load_history()
    if hist.empty or len(hist) < 2:
        st.info("Need ≥2 snapshots in `hard_set_history/` to show progress. "
                "Each `build_hard_set.py` run (without --no-history) writes one.")
    else:
        hist = hist.sort_values("generated_at")
        st.plotly_chart(px.line(hist, x="generated_at", y="n_total", markers=True,
                                title="total hard-set size over time"),
                        use_container_width=True)
        cause_cols = [c for c in hist.columns if c.startswith("cause:")]
        long = hist.melt(id_vars=["generated_at"], value_vars=cause_cols,
                         var_name="cause", value_name="n")
        long["cause"] = long["cause"].str.replace("cause:", "", regex=False)
        st.plotly_chart(px.line(long, x="generated_at", y="n", color="cause", markers=True,
                                title="hard-set size by cause over time"),
                        use_container_width=True)
        st.caption("A cause line trending down = that class of difficulty is being overcome. "
                   "Flat/rising 'unexplained' = the frontier we still can't attribute.")
