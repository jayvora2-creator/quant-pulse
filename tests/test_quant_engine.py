
import numpy as np
import pandas as pd
import pytest

from app.quant_engine import (
    _align_returns,
    annualized_return,
    annualized_volatility,
    calculate_beta,
    calculate_returns,
    historical_var,
    maximum_drawdown,
    sharpe_ratio,
    sortino_ratio,
)


def test_calculate_returns_from_closing_prices():
    prices = pd.DataFrame(
        {"Close": [100, 110, 99]},
        index=pd.date_range("2026-01-01", periods=3, freq="D"),
    )

    returns = calculate_returns(prices)

    assert len(returns) == 2
    assert returns.iloc[0] == pytest.approx(0.10)
    assert returns.iloc[1] == pytest.approx(-0.10)


def test_annualized_return_compounds_returns():
    returns = pd.Series([0.01] * 252)

    result = annualized_return(returns)

    assert result == pytest.approx((1.01 ** 252) - 1)


def test_annualized_volatility_is_nonnegative():
    returns = pd.Series([0.01, -0.02, 0.015, -0.005])

    result = annualized_volatility(returns)

    assert np.isfinite(result)
    assert result > 0


def test_maximum_drawdown_is_negative_after_a_loss():
    returns = pd.Series([0.10, -0.20])

    result = maximum_drawdown(returns)

    assert result == pytest.approx(-0.20)


def test_historical_var_reports_positive_loss_threshold():
    returns = pd.Series([-0.10] * 3 + [0.0] * 7)

    result = historical_var(returns, confidence=0.90)

    assert result == pytest.approx(0.10)


def test_historical_var_rejects_invalid_confidence():
    with pytest.raises(ValueError):
        historical_var(pd.Series([0.01, -0.01]), confidence=1.0)


def test_sharpe_ratio_is_finite_for_variable_returns():
    returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])

    result = sharpe_ratio(returns)

    assert np.isfinite(result)


def test_sortino_ratio_is_finite_for_returns_with_downside():
    returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02])

    result = sortino_ratio(returns)

    assert np.isfinite(result)


def test_beta_uses_only_matching_dates():
    dates = pd.date_range("2026-01-01", periods=40, freq="D")
    benchmark = pd.Series(
        np.linspace(-0.02, 0.02, 40),
        index=dates,
    )
    stock = pd.Series(
        2 * benchmark.to_numpy() + 0.001,
        index=dates + pd.Timedelta(days=1),
    )

    aligned = _align_returns(stock, benchmark)

    assert len(aligned) == 39


def test_beta_is_calculated_from_matched_returns():
    dates = pd.date_range("2026-01-01", periods=40, freq="D")
    benchmark = pd.Series(
        np.linspace(-0.02, 0.02, 40),
        index=dates,
    )
    stock = pd.Series(
        2 * benchmark.to_numpy() + 0.001,
        index=dates,
    )

    result = calculate_beta(stock, benchmark)

    assert result == pytest.approx(2.0)


def test_beta_returns_none_when_too_few_dates_match():
    dates = pd.date_range("2026-01-01", periods=20, freq="D")
    benchmark = pd.Series(np.linspace(-0.02, 0.02, 20), index=dates)
    stock = pd.Series(2 * benchmark.to_numpy(), index=dates)

    assert calculate_beta(stock, benchmark) is None
