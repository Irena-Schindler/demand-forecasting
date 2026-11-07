# demand-forecasting

Weekly demand forecasting for consumables (e.g. printer cartridges) across five regions, with an honest
**rolling-origin backtest against a seasonal-naive baseline**.

## Result (synthetic data, 6 folds × 4 weeks)

| Model | Accuracy (1 − WAPE) |
|---|---|
| Seasonal naive (same week last year) | 84.9 % |
| Gradient boosting (lags, rolling means, seasonality, promotions, installed base) | **93.0 %** |

→ 53 % lower forecast error than the baseline.

## Why it is built this way

- **Installed base as driver:** consumable demand follows the number of active devices, not just past sales.
- **No leakage:** every lag and rolling feature is shifted; a test enforces it.
- **Rolling-origin backtest:** the model is always evaluated on weeks it has never seen, like in production.
- **Baseline first:** a forecast only earns its keep if it beats "same week last year".
- **WAPE instead of MAPE:** robust for low-volume regions and what supply planning actually feels.

The data is synthetic (real sales data is confidential) but mirrors the drivers seen in industry:
installed base growth, yearly seasonality, holiday dips and promotions. In a previous industry role a
comparable approach forecast cartridge sales with 93 % accuracy on real data.

## Run

```bash
pip install -r requirements.txt
pytest -q
python -m forecast        # prints backtest per fold and summary
```
