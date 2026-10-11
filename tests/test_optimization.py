
from app.data_provider import get_stock_data
from app.portfolio_engine import optimize_portfolio


TICKERS = ["NVDA", "MSFT", "RELIANCE.NS"]
PERIOD = "1y"


def main():
    price_series = {}

    for ticker in TICKERS:
        data = get_stock_data(ticker, period=PERIOD)
        close_column = "Adj Close" if "Adj Close" in data.columns else "Close"
        prices = data[close_column]

        if getattr(prices, "ndim", 1) == 2:
            prices = prices.iloc[:, 0]

        price_series[ticker] = prices

    price_data = __import__("pandas").concat(
        price_series, axis=1
    ).dropna()

    result = optimize_portfolio(price_data)

    for strategy in ("minimum_volatility", "maximum_sharpe"):
        item = result[strategy]
        print(f"\n{strategy.upper().replace('_', ' ')}")
        print("Weights:", {
            ticker: f"{weight:.1%}"
            for ticker, weight in item["weights"].items()
        })
        print(f"Annualized return: {item['annualized_return']:.2%}")
        print(f"Annualized volatility: {item['annualized_volatility']:.2%}")
        print(f"Sharpe ratio: {item['sharpe_ratio']:.3f}")

    print("\nEfficient frontier points:", len(result["efficient_frontier"]))


if __name__ == "__main__":
    main()
