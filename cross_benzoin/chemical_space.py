#!/usr/bin/env python
"""Flying-dataset read API, build-order step 3 (CHEMICAL_SPACE.md sec8.3):
`pair(i, j)`, the feature path only.

IMPORTANT CORRECTION vs the original spec (found while implementing this step,
documented rather than silently papered over -- see LAB_JOURNAL.md 2026-09-11):
**not all 260 champion features are lazily derivable from the two aldehyde
caches.** The spec's sec4 per-aldehyde-cache table already only covers the
donor/acceptor side; checking what `product_*` actually is in the champion
table exposed why the product side needs its own compute, not a cache lookup:

  - `PRODUCT_FEATS` (mulliken/wbo/fukui/vbur/sterimol/hb_*/dih_core/bde on the
    product) come from xTB/QM run on the product's OWN optimized 3D geometry
    (cb_featurize.py) -- there is no shortcut from the two aldehyde geometries.
  - `product_mordred_*` (checked `add_mordred_cross_products.py`) uses
    `Calculator(..., ignore_3D=False)` on `mol_from_xyz(product geometry)` --
    families like MoRSE/CPSA/MomentOfInertia/PBF are inherently 3D. Also needs
    the optimized product geometry, not just the product SMILES.
  - `baseline_gxtb` / `baseline_b973c` / `label_dG` are DFT/xTB single points --
    obviously not lazy.

So `pair(i, j)` returns two honest tiers:
  - **lazy tier** (always available, no compute): donor/acceptor local QM +
    BDE (aldehyde_index-keyed caches), donor/acceptor/product RDKit-2D
    (SMILES-only, `ald_descriptors.calc_rdkit`), the `interaction_*` block
    (derived from the lazy donor/acceptor QM), product SMILES + reaction_type
    (from the reaction template + cho_class pair).
  - **cache-hit tier** (only for a pair that has already been through the full
    DFT/xTB featurize pipeline): if the pair is present in a supplied "known
    pairs" table (e.g. the round-10 champion training table), its full stored
    row -- product QM, product mordred, baselines, label -- is read back
    verbatim, not recomputed. `pair.computed_full` says which case happened.

This is what makes the sec8.3 verification meaningful: for a *known* pair, the
lazy-tier columns this module computes must equal the corresponding columns
already sitting in the champion table (bit-level for the deterministic ones);
the cache-hit tier is that same table's row, so it matches by construction.
For a genuinely new (never-computed) pair, the cache-hit tier is honestly
None/NaN with `computed_full=False` -- the caller (AL acquisition, screening)
must know that before treating the row as if the model had product QM to
train on.

Usage
    from chemical_space import FlyingDataset
    space = FlyingDataset()
    p = space.pair(0, 3)
    p["lazy_features"]      # dict of the always-available columns
    p["product_smiles"]     # str | None
    p["reaction_type"]      # e.g. "aliph-carbo"
    p["computed_full"]      # False unless (0,3) is in the known-pairs table
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(REPO / "pipeline" / "compute"))
sys.path.insert(0, str(REPO / "pipeline" / "bde"))
from ald_descriptors import calc_rdkit  # noqa: E402
from featurize_product import build_product  # noqa: E402
from qc import norm_id  # noqa: E402

ALDEHYDE_INDEX = REPO / "data/chemical_space/aldehyde_index.parquet"
ALDEHYDE_QM = REPO / "data/cross_benzoin/homo_v6/aldehydes_all.csv"
ALDEHYDE_BDE = REPO / "data/cross_benzoin/homo_v6/aldehydes_bdfe_gxtb_descriptors.csv"

# Same list as assemble_cross_training_table.ALDEHYDE_FEATS (kept independent
# here on purpose -- this module must not import that pilot-era assembler,
# per sec8.3 "the feature path only"; duplication is the honest cost of not
# reusing a module with a much wider, product-DFT-dependent scope).
ALDEHYDE_FEATS = [
    "G_xtb", "G_gxtb", "xtb_energy", "xtb_HOMO", "xtb_LUMO", "xtb_gap",
    "xtb_IP", "xtb_EA", "xtb_mu", "xtb_eta", "xtb_omega", "xtb_dipole",
    "mulliken_CHO_C", "mulliken_CHO_O",
    "fukui_plus_CHO_C", "fukui_minus_CHO_C", "fukui_0_CHO_C", "dual_descriptor_CHO_C",
    "wbo_CO", "pa_CHO_O", "vbur_CHO_C",
    "sterimol_L", "sterimol_B1", "sterimol_B5", "SASA_total", "P_int",
    "bde_gxtb_kcal",
]
RDKIT_FEATS = ["MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds", "ArRings",
               "ArHetRings", "AlRings", "Rings", "Heteroatoms", "FractionCSP3",
               "BertzCT", "Kappa2", "NumStereocenters"]
MISMATCH_PAIRS = ["xtb_gap", "xtb_dipole", "sterimol_L", "sterimol_B1",
                   "sterimol_B5", "SASA_total", "MW", "TPSA"]
_SHORT_CHO = {"aliphatic": "aliph", "aromatic_carbo": "carbo", "aromatic_hetero": "hetero"}


@lru_cache(maxsize=1)
def _aldehyde_index() -> pd.DataFrame:
    return pd.read_parquet(ALDEHYDE_INDEX).set_index("ald_idx")


@lru_cache(maxsize=1)
def _aldehyde_qm_cache() -> pd.DataFrame:
    """ald_idx -> ALDEHYDE_FEATS (minus bde_gxtb_kcal, joined separately)."""
    ald = pd.read_csv(ALDEHYDE_QM, usecols=["id"] + [f for f in ALDEHYDE_FEATS if f != "bde_gxtb_kcal"],
                       low_memory=False)
    ald["id"] = norm_id(ald["id"]).astype(int)
    bde = pd.read_csv(ALDEHYDE_BDE, usecols=["id", "bde_gxtb_kcal"])
    bde["id"] = norm_id(bde["id"]).astype(int)
    bde.loc[bde["bde_gxtb_kcal"].abs() > 200, "bde_gxtb_kcal"] = None
    ald = ald.merge(bde, on="id", how="left")
    return ald.drop_duplicates("id").set_index("id")


def _rdkit_2d(smiles: str | None) -> dict:
    if not smiles:
        return {f: None for f in RDKIT_FEATS}
    d = calc_rdkit(smiles)
    return {f: d.get(f) for f in RDKIT_FEATS}


def _interaction_block(donor_qm: dict, acceptor_qm: dict) -> dict:
    out = {}
    if donor_qm.get("xtb_HOMO") is not None and acceptor_qm.get("xtb_LUMO") is not None:
        out["interaction_gap_HOMOd_LUMOa"] = donor_qm["xtb_HOMO"] - acceptor_qm["xtb_LUMO"]
    if donor_qm.get("fukui_minus_CHO_C") is not None and acceptor_qm.get("fukui_plus_CHO_C") is not None:
        out["interaction_fukui_match"] = donor_qm["fukui_minus_CHO_C"] * acceptor_qm["fukui_plus_CHO_C"]
    for feat in MISMATCH_PAIRS:
        dv, av = donor_qm.get(feat), acceptor_qm.get(feat)
        if dv is not None and av is not None:
            out[f"interaction_absdiff_{feat}"] = abs(dv - av)
    return out


class FlyingDataset:
    """Lazy view over the 220,859^2 benzoin cross-ΔG space. See module docstring
    for exactly which fields are cache-derived (`lazy_features`) vs. genuinely
    require the DFT/xTB featurize pipeline (only present via `known_pairs`)."""

    def __init__(self, known_pairs: pd.DataFrame | None = None):
        """`known_pairs`: optional DataFrame indexed/keyed the same way as the
        round-N champion training table (must carry donor_smiles/acceptor_smiles
        so pairs can be located by ald_idx via canonical-SMILES lookup). If not
        given, `computed_full` is always False -- honest default, not a bug."""
        self._idx = _aldehyde_index()
        self._qm = _aldehyde_qm_cache()
        self._known = known_pairs
        self._known_by_pair: dict[tuple[int, int], int] | None = None
        if known_pairs is not None:
            self._build_known_lookup()

    def _build_known_lookup(self) -> None:
        idx = self._idx
        smi_to_ald_idx = {row.smiles_canonical: i for i, row in idx.iterrows() if row.smiles_canonical}

        def canon(s):
            m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
            return Chem.MolToSmiles(m) if m is not None else None

        d_idx = self._known["donor_smiles"].map(canon).map(smi_to_ald_idx.get)
        a_idx = self._known["acceptor_smiles"].map(canon).map(smi_to_ald_idx.get)
        lut = {}
        for pos, (d, a) in enumerate(zip(d_idx, a_idx)):
            if d is not None and a is not None:
                lut[(int(d), int(a))] = pos
        self._known_by_pair = lut

    @property
    def n_aldehydes(self) -> int:
        return len(self._idx)

    def _ald_view(self, ald_idx: int, prefix: str) -> dict:
        if ald_idx not in self._idx.index:
            return {"idx": ald_idx, "found": False}
        row = self._idx.loc[ald_idx]
        qm = self._qm.loc[ald_idx].to_dict() if ald_idx in self._qm.index else {f: None for f in ALDEHYDE_FEATS}
        out = {
            "idx": ald_idx, "found": True,
            "smiles": row["smiles_canonical"], "cho_class": row["cho_class"],
            "scaffold": row["scaffold"], "computable": row["computable"],
        }
        out.update({f"{prefix}_{k}": v for k, v in qm.items()})
        out.update({f"{prefix}_{k}": v for k, v in _rdkit_2d(row["smiles_canonical"]).items()})
        return out, qm

    def pair(self, donor_idx: int, acceptor_idx: int) -> dict:
        d_view, d_qm = self._ald_view(donor_idx, "donor")
        a_view, a_qm = self._ald_view(acceptor_idx, "acceptor")

        result: dict = {
            "donor_idx": donor_idx, "acceptor_idx": acceptor_idx,
            "donor_smiles": d_view.get("smiles"), "acceptor_smiles": a_view.get("smiles"),
            "computable": (d_view.get("computable"), a_view.get("computable")),
        }

        if not (d_view.get("found") and a_view.get("found")):
            result["error"] = "donor_idx or acceptor_idx out of range"
            result["computed_full"] = False
            return result

        product_smiles = build_product(d_view["smiles"], a_view["smiles"])
        result["product_smiles"] = product_smiles
        result["reaction_type"] = (
            "-".join(sorted([_SHORT_CHO.get(d_view["cho_class"], "?"),
                              _SHORT_CHO.get(a_view["cho_class"], "?")]))
            if d_view["cho_class"] and a_view["cho_class"] else None
        )

        lazy = {k: v for k, v in d_view.items() if k not in ("idx", "found", "smiles", "cho_class", "scaffold", "computable")}
        lazy.update({k: v for k, v in a_view.items() if k not in ("idx", "found", "smiles", "cho_class", "scaffold", "computable")})
        lazy.update({f"product_{k}": v for k, v in _rdkit_2d(product_smiles).items()})
        lazy.update(_interaction_block(d_qm, a_qm))
        result["lazy_features"] = lazy

        # cache-hit tier: only if this exact address has already been through
        # the full DFT/xTB pipeline (see class docstring).
        pos = self._known_by_pair.get((donor_idx, acceptor_idx)) if self._known_by_pair else None
        result["computed_full"] = pos is not None
        if pos is not None:
            result["known_row"] = self._known.iloc[pos].to_dict()
        return result

    def iter_pairs(self, donor_idx=None, acceptor_idx=None):
        """Generator over addresses -- never materializes a pair table.
        Restricting to one side (donor_idx or acceptor_idx fixed) is the only
        mode implemented for now; a full unrestricted sweep is 4.9e10 addresses
        and is deliberately not offered here (sample() below is the entry
        point for that, once build-order step 4-5 lands acquisition support)."""
        aldehydes = list(self._idx.index)
        if donor_idx is not None:
            for a in aldehydes:
                yield (donor_idx, a)
        elif acceptor_idx is not None:
            for d in aldehydes:
                yield (d, acceptor_idx)
        else:
            raise NotImplementedError(
                "unrestricted iter_pairs over the full 220,859^2 space needs an "
                "explicit strategy (build-order step 4+); pass donor_idx or acceptor_idx")
