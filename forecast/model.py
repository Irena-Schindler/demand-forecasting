"""Feature engineering, gradient boosting model, seasonal-naive baseline and rolling-origin backtest."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

FEATURES = ["lag_1", "lag_2", "lag_52", "roll_4", "roll_13", "week_of_year", "promo",
            "installed_base", "region_code", "holiday"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["region", "week"]).copy()
    g = df.groupby("region")["demand"]
    df["lag_1"], df["lag_2"], df["lag_52"] = g.shift(1), g.shift(2), g.shift(52)
    df["roll_4"] = g.transform(lambda s: s.shift(1).rolling(4).mean())
    df["roll_13"] = g.transform(lambda s: s.shift(1).rolling(13).mean())
    df["region_code"] = df["region"].astype("category").cat.codes
    df["holiday"] = df["week_of_year"].isin([52, 1, 31, 32, 33]).astype(int)
    return df


def wape(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.abs(y - p).sum() / np.abs(y).sum())


def backtest(df: pd.DataFrame, folds: int = 6, horizon: int = 4) -> pd.DataFrame:
    """Rolling origin: train on everything before the cut, predict the next `horizon` weeks (one step ahead)."""
    df = add_features(df).dropna(subset=FEATURES)
    weeks = np.sort(df["week"].unique())
    out = []
    for f in range(folds, 0, -1):
        cut = weeks[-f * horizon]
        test_weeks = weeks[(weeks >= cut)][:horizon]
        train, test = df[df["week"] < cut], df[df["week"].isin(test_weeks)]
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
        m.fit(train[FEATURES], train["demand"])
        out.append({"cut": pd.Timestamp(cut).date(),
                    "wape_model": wape(test["demand"].to_numpy(), m.predict(test[FEATURES])),
                    "wape_seasonal_naive": wape(test["demand"].to_numpy(), test["lag_52"].to_numpy())})
    return pd.DataFrame(out)


def summary(bt: pd.DataFrame) -> dict:
    model, naive = bt["wape_model"].mean(), bt["wape_seasonal_naive"].mean()
    return {"accuracy_model": round(float(1 - model), 3), "accuracy_naive": round(float(1 - naive), 3),
            "error_reduction_vs_naive": round(float(1 - model / naive), 3)}
