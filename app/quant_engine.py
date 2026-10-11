
import numpy as np
import pandas as pd


TRADING_DAYS = 252
MIN_BETA_OBSERVATIONS = 30


def _clean_prices(data):
    """Return a clean, positive closing-price Series."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Price data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("Price data is empty.")

    if "Close" not in data.columns:
        raise ValueError("Price data must contain a 'Close' column.")

    close = data["Close"]

    # Some yfinance responses return a DataFrame for Close
    # when the columns contain multiple ticker symbols.
    if isinstance(close, pd.DataFrame):
        if close.shape[1] != 1:
            raise ValueError(
                "Expected closing prices for one ticker at a time."
            )
        close = close.iloc[:, 0]

    close = pd.to_numeric(close, errors="coerce")
    close = close.replace([np.inf, -np.inf], np.nan).dropna()
    close = close[close > 0]
    close = close.sort_index()

    if isinstance(close.index, pd.DatetimeIndex):
        close = close[~close.index.duplicated(keep="last")]

    if len(close) < 2:
        raise ValueError(
            "At least two valid closing prices are required."
        )

    return close.astype(float)


def calculate_returns(data):
    """Calculate simple daily returns from closing prices."""
    prices = _clean_prices(data)
    returns = prices.pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    return returns


def annualized_return(returns):
    """Annualize compounded returns using 252 trading days."""
    returns = pd.Series(returns, dtype=float)
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    if returns.empty:
        return 0.0

    if (returns < -1).any():
        raise ValueError("A return cannot be less than -100%.")

    growth = float((1 + returns).prod())

    if not np.isfinite(growth):
        raise ValueError("Could not calculate compounded returns.")

    if growth <= 0:
        return -1.0

    periods = len(returns) / TRADING_DAYS
    return float(growth ** (1 / periods) - 1)


def annualized_volatility(returns):
    """Calculate annualized volatility from daily returns."""
    returns = pd.Series(returns, dtype=float)
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    if len(returns) < 2:
        return 0.0

    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Calculate annualized Sharpe ratio.

    risk_free_rate is an annual decimal rate:
    0.05 means 5% per year.
    """
    returns = pd.Series(returns, dtype=float)
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    if len(returns) < 2:
        return 0.0

    daily_risk_free = (1 + risk_free_rate) ** (
        1 / TRADING_DAYS
    ) - 1
    excess_returns = returns - daily_risk_free
    daily_volatility = returns.std(ddof=1)

    if not np.isfinite(daily_volatility) or daily_volatility == 0:
        return 0.0

    result = (
        excess_returns.mean()
        / daily_volatility
        * np.sqrt(TRADING_DAYS)
    )

    return float(result) if np.isfinite(result) else 0.0


def sortino_ratio(returns, risk_free_rate=0.0):
    """
    Calculate annualized Sortino ratio using daily risk-free return
    as the minimum acceptable return.

    risk_free_rate is an annual decimal rate:
    0.05 means 5% per year.
    """
    returns = pd.Series(returns, dtype=float)
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    if returns.empty:
        return 0.0

    daily_risk_free = (1 + risk_free_rate) ** (
        1 / TRADING_DAYS
    ) - 1
    excess_returns = returns - daily_risk_free

    downside_returns = np.minimum(excess_returns.to_numpy(), 0.0)
    downside_deviation = float(
        np.sqrt(np.mean(np.square(downside_returns)))
    )

    if not np.isfinite(downside_deviation) or downside_deviation == 0:
        return 0.0

    result = (
        excess_returns.mean()
        / downside_deviation
        * np.sqrt(TRADING_DAYS)
    )

    return float(result) if np.isfinite(result) else 0.0


def maximum_drawdown(returns):
    """Calculate the worst peak-to-trough drawdown from returns."""
    returns = pd.Series(returns, dtype=float)
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    if returns.empty:
        return 0.0

    wealth = (1 + returns).cumprod()
    running_peak = wealth.cummax()
    drawdowns = wealth / running_peak - 1

    return float(drawdowns.min())


def _normalise_return_dates(returns):
    """Normalize return dates so stock and benchmark dates can match."""
    returns = pd.Series(returns, dtype=float).copy()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError(
            "Stock and benchmark returns must have DatetimeIndex dates."
        )

    index = returns.index

    if index.tz is not None:
        index = index.tz_localize(None)

    returns.index = index.normalize()
    returns = returns.groupby(level=0).last()
    return returns


def _align_returns(stock_returns, benchmark_returns):
    """Match stock and benchmark returns on the same calendar dates."""
    stock_returns = _normalise_return_dates(stock_returns)
    benchmark_returns = _normalise_return_dates(benchmark_returns)

    aligned = pd.concat(
        [
            stock_returns.rename("stock"),
            benchmark_returns.rename("benchmark"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    return aligned


def calculate_beta(stock_returns, benchmark_returns):
    
    
    """
    Calculate historical beta using daily stock and benchmark returns
    matched by date.

    Beta is the sample covariance of stock returns and benchmark
    returns divided by the sample variance of benchmark returns.

    Returns None if there are too few matched observations or the
    benchmark has zero variance. Results may differ from published
    beta values because of differences in data windows, return
    frequency, benchmark, and methodology.
    """

    
    try:
        aligned = _align_returns(stock_returns, benchmark_returns)

        if len(aligned) < MIN_BETA_OBSERVATIONS:
            return None

        benchmark_variance = aligned["benchmark"].var(ddof=1)

        if (
            not np.isfinite(benchmark_variance)
            or benchmark_variance <= 0
        ):
            return None

        covariance = aligned["stock"].cov(aligned["benchmark"])
        beta = covariance / benchmark_variance

        if not np.isfinite(beta):
            return None

        return float(beta)

    except (TypeError, ValueError, KeyError):
        return None


def historical_var(returns, confidence=0.95):
    """
    Return the positive historical one-period VaR loss threshold.

    For example, 0.02 means a 2% loss threshold at the selected
    confidence level. This is a historical estimate, not a guarantee
    or the maximum possible loss.
    """
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1.")

    returns = pd.Series(returns, dtype=float)
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    if returns.empty:
        return 0.0

    lower_quantile = float(returns.quantile(1 - confidence))

    if not np.isfinite(lower_quantile):
        return 0.0

    return float(max(0.0, -lower_quantile))


def analyze_stock(stock_data, benchmark_data, risk_free_rate=0.0):
    """Calculate stock metrics and beta against a benchmark."""
    stock_returns = calculate_returns(stock_data)
    benchmark_returns = calculate_returns(benchmark_data)

    return {
        "annualized_return": annualized_return(stock_returns),
        "annualized_volatility": annualized_volatility(stock_returns),
        "sharpe_ratio": sharpe_ratio(stock_returns, risk_free_rate),
        "sortino_ratio": sortino_ratio(stock_returns, risk_free_rate),
        "maximum_drawdown": maximum_drawdown(stock_returns),
        "historical_var_95": historical_var(
            stock_returns, confidence=0.95
        ),
        "beta": calculate_beta(stock_returns, benchmark_returns),
    }

def compare_stock_with_benchmark(stock_data, benchmark_data):
    """
    Compare stock and benchmark price returns over the same dates.

    Uses price returns only; dividends are not included.
    Returns None when there is insufficient overlapping data.
    """
    try:
        stock_prices = _clean_prices(stock_data)
        benchmark_prices = _clean_prices(benchmark_data)

        # Normalize dates before matching the price observations.
        stock_prices.index = pd.DatetimeIndex(stock_prices.index)
        benchmark_prices.index = pd.DatetimeIndex(benchmark_prices.index)

        if stock_prices.index.tz is not None:
            stock_prices.index = stock_prices.index.tz_localize(None)
        if benchmark_prices.index.tz is not None:
            benchmark_prices.index = benchmark_prices.index.tz_localize(None)

        stock_prices.index = stock_prices.index.normalize()
        benchmark_prices.index = benchmark_prices.index.normalize()

        stock_prices = stock_prices.groupby(level=0).last()
        benchmark_prices = benchmark_prices.groupby(level=0).last()

        aligned = pd.concat(
            [
                stock_prices.rename("stock"),
                benchmark_prices.rename("benchmark"),
            ],
            axis=1,
            join="inner",
        ).dropna().sort_index()

        if len(aligned) < 2:
            return None

        first_stock = float(aligned["stock"].iloc[0])
        last_stock = float(aligned["stock"].iloc[-1])
        first_benchmark = float(aligned["benchmark"].iloc[0])
        last_benchmark = float(aligned["benchmark"].iloc[-1])

        stock_total_return = last_stock / first_stock - 1
        benchmark_total_return = last_benchmark / first_benchmark - 1

        # Use the same elapsed calendar period for both annualized returns.
        elapsed_days = (
            aligned.index[-1] - aligned.index[0]
        ).total_seconds() / 86400

        if elapsed_days <= 0:
            return None

        years = elapsed_days / 365.25

        stock_annualized = (
            (1 + stock_total_return) ** (1 / years) - 1
            if stock_total_return > -1 else -1.0
        )
        benchmark_annualized = (
            (1 + benchmark_total_return) ** (1 / years) - 1
            if benchmark_total_return > -1 else -1.0
        )

        return {
            "start_date": aligned.index[0].date().isoformat(),
            "end_date": aligned.index[-1].date().isoformat(),
            "overlapping_price_observations": len(aligned),
            "stock_total_price_return": float(stock_total_return),
            "benchmark_total_price_return": float(benchmark_total_return),
            "stock_annualized_return": float(stock_annualized),
            "benchmark_annualized_return": float(benchmark_annualized),
            "annualized_return_difference": float(
                stock_annualized - benchmark_annualized
            ),
            "comparison_basis": "Same overlapping dates; price returns exclude dividends.",
        }

    except (TypeError, ValueError, KeyError, IndexError):
        return None
