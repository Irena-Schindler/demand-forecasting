"""Synthetic weekly demand for printer consumables across regions.

Synthetic on purpose: real sales data is confidential. The generator reproduces the drivers that matter in
practice: an installed base of devices, yearly seasonality, holiday dips, promotions and noise.
"""
import numpy as np
import pandas as pd

REGIONS = ["DE", "AT", "CH", "NL", "FR"]


def generate(weeks: int = 208, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=weeks, freq="W-MON")
    rows = []
    for i, region in enumerate(REGIONS):
        base = 400 + 150 * i
        installed = base * (1 + 0.004 * np.arange(weeks)) * rng.uniform(0.95, 1.05)
        woy = dates.isocalendar().week.to_numpy().astype(int)
        season = 1 + 0.15 * np.sin(2 * np.pi * (woy - 10) / 52)
        holiday = np.where(np.isin(woy, [52, 1, 31, 32, 33]), 0.7, 1.0)
        promo = (rng.random(weeks) < 0.08).astype(int)
        demand = installed * 0.9 * season * holiday * (1 + 0.25 * promo) * rng.normal(1, 0.06, weeks)
        for d, w, p, inst, y in zip(dates, woy, promo, installed, demand):
            rows.append({"week": d, "region": region, "week_of_year": w, "promo": p,
                         "installed_base": round(inst), "demand": max(0, round(y))})
    return pd.DataFrame(rows)
