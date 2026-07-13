from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "cross_benzoin" / "prepare_product_manifest.py"
    spec = importlib.util.spec_from_file_location("prepare_product_manifest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_directed_pair_builds_distinct_regioisomers():
    module = _load_module()
    benzaldehyde = "O=Cc1ccccc1"
    acetaldehyde = "CC=O"
    ab, ab_error = module.build_product(benzaldehyde, acetaldehyde)
    ba, ba_error = module.build_product(acetaldehyde, benzaldehyde)
    assert not ab_error and not ba_error
    assert ab and ba and ab != ba


def test_invalid_smiles_is_rejected_before_qm():
    module = _load_module()
    product, error = module.build_product("not-a-smiles", "O=Cc1ccccc1")
    assert product == ""
    assert error == "invalid_donor_smiles"
