import numpy as np
from forecast.data import generate
from forecast.model import add_features, backtest, summary, wape


def test_generator_is_deterministic_and_complete():
    a, b = generate(seed=1), generate(seed=1)
    assert a.equals(b) and a["region"].nunique() == 5 and (a["demand"] >= 0).all()


def test_features_do_not_leak_the_target():
    df = add_features(generate())
    one = df[df["region"] == "DE"].reset_index(drop=True)
    assert np.isclose(one.loc[10, "lag_1"], one.loc[9, "demand"])
    assert np.isclose(one.loc[20, "roll_4"], one.loc[16:19, "demand"].mean())


def test_wape():
    assert wape(np.array([100, 100]), np.array([90, 110])) == 0.1


def test_model_beats_seasonal_naive():
    s = summary(backtest(generate(), folds=3))
    assert s["accuracy_model"] > s["accuracy_naive"] and s["accuracy_model"] > 0.9
