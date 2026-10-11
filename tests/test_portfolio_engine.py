
import pandas as pd

from app.data_provider import get_stock_data
from app.portfolio_engine import analyze_portfolio


# Editable test inputs. The engine supports other tickers too.
weights = {
    "RELIANCE.NS": 0.40,
    "TCS.NS": 0.30,
    "AAPL": 0.30,
}

period = "1y"

print("\nDownloading actual market data...")

price_series = {}

for ticker in weights:
    data = get_stock_data(ticker, period=period)

    price_column = (
        "Adj Close" if "Adj Close" in data.columns else "Close"
    )
    prices = data[price_column]

    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]

    price_series[ticker] = prices.rename(ticker)

# Align dates and use only dates with prices for every selected stock.
prices = pd.concat(price_series.values(), axis=1, sort=False)
prices = prices.dropna().sort_index()

if len(prices) < 2:
    raise ValueError("Not enough overlapping price data.")

results = analyze_portfolio(prices, weights)

print("\n========== INDIVIDUAL STOCK ANALYSIS ==========")

for ticker, metrics in results["individual_results"].items():
    print(f"\n{ticker} (Weight: {metrics['weight']:.0%})")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(
        "Annualized Volatility: "
        f"{metrics['annualized_volatility']:.2%}"
    )

print("\n========== PORTFOLIO PERFORMANCE & RISK ==========")
print(f"Annualized Return: {results['annualized_return']:.2%}")
print(
    "Annualized Volatility: "
    f"{results['annualized_volatility']:.2%}"
)
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Sortino Ratio: {results['sortino_ratio']:.2f}")
print(f"Maximum Drawdown: {results['maximum_drawdown']:.2%}")
print(f"95% Historical Daily VaR: {results['historical_var_95']:.2%}")

print("\n========== RISK CONTRIBUTIONS ==========")

for ticker, risk in results["risk_contributions"].items():
    print(f"\n{ticker}")
    print(
        "Annualized volatility contribution: "
        f"{risk['annualized_volatility_contribution']:.2%}"
    )
    print(
        "Share of portfolio risk: "
        f"{risk['percentage_of_portfolio_risk']:.2%}"
    )

print("\n========== STOCK RETURN CORRELATION ==========")
print(results["correlation_matrix"].round(2).to_string())

print("\n==============================================")
print("Source: Yahoo Finance via yfinance")
print(f"Requested historical period: {period}")
print("Historical estimates only; not investment advice.")