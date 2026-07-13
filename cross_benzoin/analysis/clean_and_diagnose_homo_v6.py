#!/usr/bin/env python
"""Clean products_all.csv and diagnose which aldehydes failed and why.

Outputs (into data/cross_benzoin/homo_v6/):
  products_clean.csv     — rows with a real donor_id and no error (the usable set).
  failed_aldehydes.csv   — one row per aldehyde that has NO clean product, with its
                           SMILES, the failure category, and the raw error message.

Failure categories (earliest failing stage wins; a molecule that succeeded on any
attempt is NOT listed as failed):
  disk_quota_rerunnable  — Errno 122 disk-quota hit mid-run. INFRASTRUCTURE, not a
                           property of the molecule; simply re-runnable.
  embed_failed_no_3d     — product SMILES built but RDKit could not embed a 3D structure
                           (cannot generate the product geometry).
  core_not_perceived     — 3D built but the benzoin alpha-hydroxyketone core could not be
                           located, so descriptor sites cannot be mapped.
  energy_dG_failed       — geometry fine but the xTB free energy / reaction dG could not
                           be formed (energy/descriptor computation failed).
"""
import re
from pathlib import Path
import pandas as pd

ROOT = Path("data/cross_benzoin/homo_v6")
A = ROOT / "aldehydes_all.csv"
P = ROOT / "products_all.csv"

CAT = {  # raw-error prefix -> (category, human buckets asked for by the user)
    "exception": "disk_quota_rerunnable",
    "prod_embed_failed": "embed_failed_no_3d",
    "core_not_found": "core_not_perceived",
    "dG_failed": "energy_dG_failed",
}
# Priority: report a genuine chemistry failure over an infra failure if a molecule hit both.
PRIORITY = ["embed_failed_no_3d", "core_not_perceived", "energy_dG_failed",
            "disk_quota_rerunnable"]


def to_id(x):
    """Normalise a possibly float-formatted id ('2.0') to a clean int string ('2')."""
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)) or str(x).strip() == "":
            return None
        return str(int(float(x)))
    except (ValueError, TypeError):
        return None


def categorise(err):
    if not isinstance(err, str) or not err:
        return None
    key = err.split(":", 1)[0]
    return CAT.get(key, "other")


def main():
    prod = pd.read_csv(P, dtype=str, keep_default_na=False, low_memory=False)
    ald = pd.read_csv(A, dtype=str, keep_default_na=False, low_memory=False)

    has_donor = prod["donor_id"].str.strip() != ""
    has_error = prod["error"].str.strip() != ""

    # 1) clean products
    clean = prod[has_donor & ~has_error].copy()
    clean.to_csv(ROOT / "products_clean.csv", index=False)

    # set of aldehyde ids that DID yield a clean product (normalised)
    clean_ids = set(to_id(v) for v in clean["donor_id"]) - {None}

    # 2) gather every failed attempt, recovering the aldehyde id (+ any SMILES on the row)
    prod_re = re.compile(r"prod_(\d+)")
    fails = {}  # ald_id -> list of (category, raw_error, donor_smiles)
    for donor, err, dsmi in zip(prod.loc[has_error, "donor_id"],
                                prod.loc[has_error, "error"],
                                prod.loc[has_error, "donor_smiles"]):
        aid = to_id(donor)
        if aid is None:                        # placeholder row: recover from prod_<N>
            m = prod_re.search(err or "")
            aid = m.group(1) if m else None
        if aid is None:
            continue
        fails.setdefault(aid, []).append((categorise(err), err, dsmi))

    # 3) a molecule is "failed" only if it has NO clean product
    ald_smiles = dict(zip((to_id(v) for v in ald["id"]), ald["smiles"]))
    ald_ids = set(ald_smiles) - {None}
    rows = []
    for aid, attempts in fails.items():
        if aid in clean_ids:
            continue                            # succeeded on another attempt
        cats = [c for c, _, _ in attempts]
        cat = next((c for c in PRIORITY if c in cats), cats[0])
        raw = next((e for c, e, _ in attempts if c == cat), attempts[0][1])
        # SMILES: prefer the aldehyde table, else the product row's donor_smiles
        smi = ald_smiles.get(aid) or next((s for _, _, s in attempts if s), "")
        rows.append({"ald_id": aid, "smiles": smi,
                     "aldehyde_in_table": "Y" if aid in ald_ids else "N",
                     "category": cat, "n_attempts": len(attempts),
                     "error_raw": raw})

    failed = pd.DataFrame(rows).sort_values(["category", "ald_id"])
    failed.to_csv(ROOT / "failed_aldehydes.csv", index=False)

    # 4) summary
    n_ald = ald["id"].apply(to_id).nunique()
    print(f"aldehydes (unique):            {n_ald}")
    print(f"products_all rows:             {len(prod)}")
    print(f"  clean products written:      {len(clean)}  -> products_clean.csv")
    print(f"  aldehydes WITH clean product:{len(clean_ids)}")
    print(f"failed aldehydes (no clean product): {len(failed)}  -> failed_aldehydes.csv")
    print("\nby category:")
    for c in PRIORITY + ["other"]:
        n = int((failed["category"] == c).sum())
        if n:
            print(f"  {c:22s} {n}")
    genuine = failed[failed["category"] != "disk_quota_rerunnable"]
    print(f"\ngenuine (non-infra) un-computable aldehydes: {len(genuine)}")
    print(f"disk-quota (re-runnable, molecule is fine):  {len(failed) - len(genuine)}")
    print("\ncannot generate product 3D structure  (embed_failed_no_3d):",
          int((failed['category'] == 'embed_failed_no_3d').sum()))
    print("cannot compute descriptors (core_not_perceived + energy_dG_failed):",
          int(failed['category'].isin(['core_not_perceived', 'energy_dG_failed']).sum()))


if __name__ == "__main__":
    main()
