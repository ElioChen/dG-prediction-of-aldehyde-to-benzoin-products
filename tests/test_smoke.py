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
