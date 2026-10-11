
import pandas as pd
import numpy as np


TRADING_DAYS = 252


def _validate_weights(weights: dict, available_columns) -> dict:
    """Validate portfolio weights and selected stock columns."""

    if not weights:
        raise ValueError("Add at least one stock to the portfolio.")

    clean_weights = {}

    for ticker, weight in weights.items():
        weight = float(weight)

        if not np.isfinite(weight):
            raise ValueError(f"Invalid weight for {ticker}.")

        if weight < 0:
            raise ValueError(
                "Negative weights are not supported in this version."
            )

        if ticker not in available_columns:
            raise ValueError(
                f"{ticker} not found in price data."
            )

        clean_weights[ticker] = weight

    total_weight = sum(clean_weights.values())

    if not np.isclose(total_weight, 1.0):
        raise ValueError(
            f"Portfolio weights must add up to 100%. "
            f"Current total: {total_weight:.2%}"
        )

    return clean_weights


def calculate_portfolio_returns(
    price_data: pd.DataFrame,
    weights: dict
) -> pd.Series:
    """Calculate daily portfolio returns from stock prices."""

    weights = _validate_weights(weights, price_data.columns)

    prices = price_data[list(weights)].copy()
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.replace([np.inf, -np.inf], np.nan).dropna()

    if len(prices) < 2:
        raise ValueError(
            "Not enough overlapping price data to calculate returns."
        )

    if (prices <= 0).any().any():
        raise ValueError("Stock prices must be positive.")

    stock_returns = prices.pct_change().dropna()

    weight_series = pd.Series(weights, dtype=float)

    portfolio_returns = stock_returns.mul(
        weight_series, axis="columns"
    ).sum(axis=1)

    return portfolio_returns


def portfolio_return(
    portfolio_returns: pd.Series
) -> float:
    """Calculate annualized compounded portfolio return."""

    returns = portfolio_returns.dropna()

    if len(returns) == 0:
        return 0.0

    if (returns <= -1).any():
        raise ValueError(
            "A daily return of -100% or lower prevents this calculation."
        )

    total_return = (1 + returns).prod()
    years = len(returns) / TRADING_DAYS

    return float(total_return ** (1 / years) - 1)


def portfolio_volatility(
    portfolio_returns: pd.Series
) -> float:
    """Calculate annualized portfolio volatility."""

    returns = portfolio_returns.dropna()

    if len(returns) < 2:
        return 0.0

    return float(returns.std() * np.sqrt(TRADING_DAYS))


def portfolio_sharpe_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.0
) -> float:
    """Calculate annualized Sharpe ratio."""

    annual_return = portfolio_return(portfolio_returns)
    volatility = portfolio_volatility(portfolio_returns)

    if volatility == 0:
        return 0.0

    return float((annual_return - risk_free_rate) / volatility)


def portfolio_sortino_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.0
) -> float:
    """Calculate annualized Sortino ratio using downside deviation."""

    returns = portfolio_returns.dropna()

    if len(returns) == 0:
        return 0.0

    annual_return = portfolio_return(returns)

    # Downside deviation includes all observations,
    # treating positive returns as zero downside.
    downside = np.minimum(returns.to_numpy(), 0.0)
    downside_deviation = np.sqrt(np.mean(downside ** 2)) * np.sqrt(
        TRADING_DAYS
    )

    if downside_deviation == 0:
        return 0.0

    return float(
        (annual_return - risk_free_rate) / downside_deviation
    )


def portfolio_maximum_drawdown(
    portfolio_returns: pd.Series
) -> float:
    """Calculate the largest peak-to-trough portfolio loss."""

    returns = portfolio_returns.dropna()

    if len(returns) == 0:
        return 0.0

    wealth = (1 + returns).cumprod()

    # Include initial portfolio value as the starting peak.
    wealth = pd.concat([
        pd.Series([1.0], index=[returns.index[0] - pd.Timedelta(days=1)]),
        wealth
    ])

    running_peak = wealth.cummax()
    drawdowns = wealth / running_peak - 1

    return float(drawdowns.min())


def portfolio_historical_var(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """Calculate historical daily VaR as a positive percentage."""

    returns = portfolio_returns.dropna()

    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be between 0 and 1."
        )

    if len(returns) == 0:
        return 0.0

    percentile = 1 - confidence_level
    var = -returns.quantile(percentile)

    return max(float(var), 0.0)


def correlation_matrix(
    price_data: pd.DataFrame
) -> pd.DataFrame:
    """Calculate correlations between selected stocks' daily returns."""

    returns = price_data.pct_change().dropna()

    return returns.corr()


def calculate_risk_contributions(
    price_data: pd.DataFrame,
    weights: dict
) -> dict:
    """
    Estimate each stock's contribution to portfolio volatility.

    Contributions use the covariance matrix of daily returns.
    Short positions are not supported by the weight validator.
    """

    weights = _validate_weights(weights, price_data.columns)

    prices = price_data[list(weights)].copy()
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.replace([np.inf, -np.inf], np.nan).dropna()

    returns = prices.pct_change().dropna()

    if len(returns) < 2:
        raise ValueError("Not enough data to estimate risk contributions.")

    weight_vector = np.array(
        [weights[ticker] for ticker in returns.columns],
        dtype=float
    )

    covariance = returns.cov().to_numpy()
    portfolio_variance = float(
        weight_vector @ covariance @ weight_vector
    )

    if portfolio_variance <= 0:
        return {
            ticker: {
                "annualized_volatility_contribution": 0.0,
                "percentage_of_portfolio_risk": 0.0
            }
            for ticker in returns.columns
        }

    daily_volatility = np.sqrt(portfolio_variance)

    # Euler contribution to volatility: w_i * (Cov @ w)_i / sigma_p
    contributions = (
        weight_vector * (covariance @ weight_vector)
    ) / daily_volatility

    annualized_contributions = contributions * np.sqrt(TRADING_DAYS)
    annualized_portfolio_volatility = (
        daily_volatility * np.sqrt(TRADING_DAYS)
    )

    return {
        ticker: {
            "annualized_volatility_contribution": float(
                annualized_contributions[i]
            ),
            "percentage_of_portfolio_risk": float(
                annualized_contributions[i]
                / annualized_portfolio_volatility
            )
        }
        for i, ticker in enumerate(returns.columns)
    }


def analyze_portfolio(
    price_data: pd.DataFrame,
    weights: dict,
    risk_free_rate: float = 0.0
) -> dict:
    """Calculate performance and risk metrics for a portfolio."""

    weights = _validate_weights(weights, price_data.columns)

    portfolio_returns = calculate_portfolio_returns(
        price_data,
        weights
    )

    risk_contributions = calculate_risk_contributions(
        price_data,
        weights
    )

    individual_results = {}

    prices = price_data[list(weights)].copy()
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.replace([np.inf, -np.inf], np.nan).dropna()

    stock_returns = prices.pct_change().dropna()

    for ticker in weights:
        returns = stock_returns[ticker]

        individual_results[ticker] = {
            "weight": weights[ticker],
            "annualized_return": portfolio_return(returns),
            "annualized_volatility": portfolio_volatility(returns)
        }

    return {
        "annualized_return": portfolio_return(portfolio_returns),
        "annualized_volatility": portfolio_volatility(portfolio_returns),
        "sharpe_ratio": portfolio_sharpe_ratio(
            portfolio_returns, risk_free_rate
        ),
        "sortino_ratio": portfolio_sortino_ratio(
            portfolio_returns, risk_free_rate
        ),
        "maximum_drawdown": portfolio_maximum_drawdown(
            portfolio_returns
        ),
        "historical_var_95": portfolio_historical_var(
            portfolio_returns, 0.95
        ),
        "individual_results": individual_results,
        "risk_contributions": risk_contributions,
        "correlation_matrix": correlation_matrix(prices)
    }

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def optimize_portfolio(price_data, risk_free_rate=0.0):
    """
    Optimize a long-only portfolio using historical daily prices.

    Returns:
        Minimum-volatility portfolio
        Maximum-Sharpe-ratio portfolio
        Efficient frontier data
    """
    if price_data is None or price_data.empty:
        raise ValueError("Price data is empty.")

    prices = price_data.select_dtypes(include=[np.number]).dropna()

    if prices.shape[1] < 2:
        raise ValueError("At least two stocks with valid price data are required.")

    returns = prices.pct_change().dropna()

    if len(returns) < 30:
        raise ValueError("At least 30 daily return observations are required.")

    mean_returns = returns.mean().to_numpy() * 252
    covariance = returns.cov().to_numpy() * 252
    tickers = list(returns.columns)
    number_of_assets = len(tickers)

    if not np.isfinite(mean_returns).all() or not np.isfinite(covariance).all():
        raise ValueError("The price data contains invalid values.")

    covariance = (covariance + covariance.T) / 2
    bounds = tuple((0.0, 1.0) for _ in range(number_of_assets))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    initial_weights = np.repeat(1 / number_of_assets, number_of_assets)

    def portfolio_metrics(weights):
        expected_return = float(weights @ mean_returns)
        variance = float(weights @ covariance @ weights)
        volatility = float(np.sqrt(max(variance, 0.0)))
        sharpe = (
            (expected_return - risk_free_rate) / volatility
            if volatility > 1e-12 else 0.0
        )
        return expected_return, volatility, sharpe

    min_vol_result = minimize(
        lambda w: portfolio_metrics(w)[1],
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    max_sharpe_result = minimize(
        lambda w: -portfolio_metrics(w)[2],
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not min_vol_result.success or not max_sharpe_result.success:
        raise RuntimeError(
            "Optimization did not converge. Try another period or ticker combination."
        )

    min_weights = np.clip(min_vol_result.x, 0, 1)
    min_weights /= min_weights.sum()

    sharpe_weights = np.clip(max_sharpe_result.x, 0, 1)
    sharpe_weights /= sharpe_weights.sum()

    min_return, min_volatility, min_sharpe = portfolio_metrics(min_weights)
    max_return, max_volatility, max_sharpe = portfolio_metrics(sharpe_weights)

    # Generate portfolios targeting a range of annualized returns.
    frontier = []
    target_returns = np.linspace(
        float(np.min(mean_returns)),
        float(np.max(mean_returns)),
        25,
    )

    for target in target_returns:
        target_constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {
                "type": "eq",
                "fun": lambda w, t=target: float(w @ mean_returns) - t,
            },
        ]

        result = minimize(
            lambda w: float(np.sqrt(max(w @ covariance @ w, 0.0))),
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=target_constraints,
            options={"maxiter": 1000, "ftol": 1e-10},
        )

        if result.success:
            weights = np.clip(result.x, 0, 1)
            weights /= weights.sum()
            expected_return, volatility, sharpe = portfolio_metrics(weights)
            frontier.append({
                "Annualized Return (%)": expected_return * 100,
                "Annualized Volatility (%)": volatility * 100,
                "Sharpe Ratio": sharpe,
            })

    return {
        "tickers": tickers,
        "minimum_volatility": {
            "weights": dict(zip(tickers, min_weights.tolist())),
            "annualized_return": min_return,
            "annualized_volatility": min_volatility,
            "sharpe_ratio": min_sharpe,
        },
        "maximum_sharpe": {
            "weights": dict(zip(tickers, sharpe_weights.tolist())),
            "annualized_return": max_return,
            "annualized_volatility": max_volatility,
            "sharpe_ratio": max_sharpe,
        },
        "efficient_frontier": pd.DataFrame(frontier),
    }
