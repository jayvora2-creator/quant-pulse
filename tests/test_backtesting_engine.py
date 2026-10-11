
import pandas as pd

from app.data_provider import get_stock_data
from app.backtesting_engine import backtest_strategy


def main():
    ticker = "NVDA"
    period = "2y"

    print(f"Downloading historical data for {ticker}...")

    data = get_stock_data(ticker, period)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    column = "Adj Close" if "Adj Close" in data.columns else "Close"
    prices = data[column]

    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]

    strategies = [
        "sma_crossover",
        "momentum",
        "buy_and_hold",
    ]

    for strategy in strategies:
        result = backtest_strategy(
            prices,
            strategy=strategy,
            short_window=20,
            long_window=50,
            momentum_window=20,
            transaction_cost_bps=10,
        )

        metrics = result["metrics"]
        benchmark = result["benchmark_metrics"]

        print("\n" + "=" * 45)
        print(f"STRATEGY: {strategy.upper()}")
        print("=" * 45)

        print(f"Total return: {metrics['total_return']:.2%}")
        print(
            f"Annualized return: "
            f"{metrics['annualized_return']:.2%}"
        )
        print(
            f"Annualized volatility: "
            f"{metrics['annualized_volatility']:.2%}"
        )
        print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        print(
            f"Maximum drawdown: "
            f"{metrics['maximum_drawdown']:.2%}"
        )
        print(f"Position changes: {result['number_of_trades']}")
        print(
            f"Buy-and-hold total return: "
            f"{benchmark['total_return']:.2%}"
        )


if __name__ == "__main__":
    main()
