# AutoTrader
Automated trading program using qtpylib

Uses Geometric Brownian Motion as the base process in a Monte Carlo simulation, to predict a security's price in a given timedelta (i.e. in 1 hour).

Main strategy (gbm_strategy.py) uses support/resistance levels and Fibonacci levels to set take profit/stop loss levels.
Strategy also monitors current position and modifies it depending on changing predictions and to ensure maximum profitability.

Next phase modifications:
    - Value at Risk given confidence level
    - Determining whether an order is worth making (depending on VaR and chances of profitability)
    - Best selection(s) from pool of securities
    - Optimal strategy selection from master strategy
