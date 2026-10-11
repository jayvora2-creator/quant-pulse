
import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def calculate_performance(daily_returns):
    """Calculate performance and risk metrics from daily returns."""

    returns = pd.Series(daily_returns).dropna().astype(float)

    if returns.empty:
        raise ValueError("No valid daily returns are available.")

    equity_curve = (1 + returns).cumprod()

    total_return = float(equity_curve.iloc[-1] - 1)

    if len(returns) > 0 and equity_curve.iloc[-1] > 0:
        annualized_return = float(
            equity_curve.iloc[-1]
            ** (TRADING_DAYS_PER_YEAR / len(returns))
            - 1
        )
    else:
        annualized_return = np.nan

    annualized_volatility = float(
        returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    ) if len(returns) > 1 else 0.0

    sharpe_ratio = (
        annualized_return / annualized_volatility
        if annualized_volatility > 1e-12
        else 0.0
    )

    running_peak = equity_curve.cummax()
    drawdowns = equity_curve / running_peak - 1
    maximum_drawdown = float(drawdowns.min())

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": float(sharpe_ratio),
        "maximum_drawdown": maximum_drawdown,
        "equity_curve": equity_curve,
        "drawdowns": drawdowns,
    }


def backtest_strategy(
    prices,
    strategy="sma_crossover",
    short_window=20,
    long_window=50,
    momentum_window=20,
    transaction_cost_bps=10,
):
    """
    Backtest a long-only strategy using historical closing prices.

    Supported strategies:
        sma_crossover
        momentum
        buy_and_hold

    Signals are shifted by one trading day to avoid using today's
    closing-price signal to earn today's return.

    Transaction costs are charged on position changes.
    10 basis points = 0.10% per transaction.
    """

    prices = pd.Series(prices).copy()

    prices = pd.to_numeric(prices, errors="coerce")
    prices = prices.dropna().sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]

    if len(prices) < 3:
        raise ValueError(
            "At least three valid closing prices are required."
        )

    if (prices <= 0).any():
        raise ValueError("All closing prices must be positive.")

    if transaction_cost_bps < 0:
        raise ValueError("Transaction costs cannot be negative.")

    if strategy == "sma_crossover":
        if short_window < 1 or long_window < 2:
            raise ValueError("Moving-average windows must be positive.")

        if short_window >= long_window:
            raise ValueError(
                "The short moving-average window must be smaller "
                "than the long window."
            )

        short_ma = prices.rolling(
            window=short_window,
            min_periods=short_window,
        ).mean()

        long_ma = prices.rolling(
            window=long_window,
            min_periods=long_window,
        ).mean()

        # 1 means invested; 0 means in cash.
        signal = (short_ma > long_ma).astype(float)

        # Do not trade before both averages are available.
        signal[long_ma.isna()] = 0.0

    elif strategy == "momentum":
        if momentum_window < 1:
            raise ValueError(
                "The momentum window must be at least one day."
            )

        trailing_return = prices.pct_change(
            periods=momentum_window
        )

        signal = (trailing_return > 0).astype(float)
        signal[trailing_return.isna()] = 0.0

    elif strategy == "buy_and_hold":
        signal = pd.Series(1.0, index=prices.index)

    else:
        raise ValueError(
            "Unsupported strategy. Choose 'sma_crossover', "
            "'momentum', or 'buy_and_hold'."
        )

    market_returns = prices.pct_change().fillna(0.0)

    # A signal observed at one close can only affect the next day's return.
    positions = signal.shift(1).fillna(0.0)

    # Buy-and-hold starts with a position at the beginning.
    if strategy == "buy_and_hold":
        positions.iloc[0] = 1.0

    turnover = positions.diff().abs()
    turnover.iloc[0] = abs(positions.iloc[0])

    transaction_cost_rate = transaction_cost_bps / 10000.0

    costs = turnover * transaction_cost_rate

    strategy_returns = (
        positions * market_returns - costs
    )

    # Remove the initial zero-return observation from reported metrics.
    strategy_returns = strategy_returns.iloc[1:]
    benchmark_returns = market_returns.iloc[1:].copy()

    # Apply the initial purchase cost to buy-and-hold as well.
    if strategy == "buy_and_hold" and not benchmark_returns.empty:
        benchmark_returns.iloc[0] -= transaction_cost_rate

    strategy_metrics = calculate_performance(strategy_returns)
    benchmark_metrics = calculate_performance(benchmark_returns)

    results = pd.DataFrame({
        "Closing Price": prices,
        "Market Return": market_returns,
        "Position": positions,
        "Turnover": turnover,
        "Transaction Cost": costs,
        "Strategy Return": positions * market_returns - costs,
    })

    results = results.iloc[1:].copy()

    results["Strategy Equity"] = (
        1 + results["Strategy Return"]
    ).cumprod()

    results["Buy and Hold Equity"] = (
        1 + benchmark_returns
    ).cumprod()

    results["Strategy Drawdown"] = (
        results["Strategy Equity"]
        / results["Strategy Equity"].cummax()
        - 1
    )

    results["Buy and Hold Drawdown"] = (
        results["Buy and Hold Equity"]
        / results["Buy and Hold Equity"].cummax()
        - 1
    )

    # Count each entry or exit as a position change.
    number_of_trades = int((turnover.iloc[1:] > 0).sum())

    return {
        "strategy": strategy,
        "metrics": strategy_metrics,
        "benchmark_metrics": benchmark_metrics,
        "number_of_trades": number_of_trades,
        "results": results,
        "transaction_cost_bps": transaction_cost_bps,
    }
