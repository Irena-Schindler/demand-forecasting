from .data import generate
from .model import backtest, summary

bt = backtest(generate())
print(bt.round(3).to_string(index=False))
print(summary(bt))
