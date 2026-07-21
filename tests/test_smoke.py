"""Smoke tests that don't need xTB/Multiwfn or a trained model."""
import numpy as np
import pytest


def test_imports():
    import benzoin_dG
    assert benzoin_dG.__version__
    assert callable(benzoin_dG.predict_dG)


def test_config_discovery_returns_str_or_none():
    from benzoin_dG import config
    for fn in (config.find_xtb, config.find_multiwfn):
        v = fn()
        assert v is None or isinstance(v, str)


def test_build_vector_orders_and_imputes():
    from benzoin_dG import features
    feats = ["MW", "LogP", "dG_xtb_kcal"]
    medians = {"MW": 100.0, "LogP": 1.0, "dG_xtb_kcal": -10.0}
    # LogP missing -> median; order follows `feats`
    X = features.build_vector({"MW": 123.0, "LogP": None}, dG_xtb=-8.0,
                              feats=feats, medians=medians)
    assert X.shape == (1, 3)
    np.testing.assert_allclose(X[0], [123.0, 1.0, -8.0])


def test_predict_without_model_raises_clearly(tmp_path):
    from benzoin_dG.predict import predict_dG
    with pytest.raises(FileNotFoundError):
        predict_dG("O=Cc1ccccc1", models_dir=str(tmp_path))

def test_package_model_metadata_matches_feature_list():
    import json
    from pathlib import Path

    models = Path(__file__).resolve().parents[1] / "src" / "benzoin_dG" / "models"
    feats = json.loads((models / "feature_list.json").read_text())
    meta = json.loads((models / "metadata.json").read_text())
    assert meta["n_features"] == len(feats)
    assert meta["baseline"] == "gxtb_cosmo_dmso"


def test_out_of_scope_short_circuits_without_model(tmp_path):
    from benzoin_dG.predict import predict_dG

    p = predict_dG("O=CCC", models_dir=str(tmp_path))
    assert p.benzoin_relevant is False
    assert p.cho_class == "aliphatic"
    assert p.error.startswith("out_of_scope:")


def test_cli_rejects_conflicting_tiers(capsys):
    from benzoin_dG.cli import main

    assert main(["O=Cc1ccccc1", "--fast", "--champion"]) == 2
    assert "choose only one" in capsys.readouterr().err

