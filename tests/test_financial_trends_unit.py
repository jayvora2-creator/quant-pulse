
import pandas as pd
import pytest

from app.financial_trends import (
    _calculate_yoy_growth,
    _extract_annual_series,
    build_financial_trends,
    summarize_financial_trends,
)


def test_extract_annual_series_preserves_reporting_dates():
    statement = pd.DataFrame(
        [[100, 120]],
        index=["Total Revenue"],
        columns=pd.to_datetime(["2024-03-31", "2025-03-31"]),
    )

    result = _extract_annual_series(
        statement, ["Total Revenue", "Revenue"]
    )

    assert list(result.index) == list(statement.columns)
    assert result.tolist() == [100.0, 120.0]


def test_yoy_growth_calculates_consecutive_years():
    dates = pd.DatetimeIndex(["2024-03-31", "2025-03-31"])
    values = pd.Series([100.0, 125.0], index=dates)

    growth = _calculate_yoy_growth(values, list(dates))

    assert pd.isna(growth.iloc[0])
    assert growth.iloc[1] == pytest.approx(0.25)


def test_yoy_growth_skips_missing_reporting_year():
    dates = pd.DatetimeIndex(["2023-03-31", "2025-03-31"])
    values = pd.Series([100.0, 150.0], index=dates)

    growth = _calculate_yoy_growth(values, list(dates))

    assert pd.isna(growth.iloc[1])


def test_yoy_growth_skips_zero_or_negative_previous_value():
    dates = pd.DatetimeIndex(
        ["2024-03-31", "2025-03-31", "2026-03-31"]
    )
    values = pd.Series([0.0, -10.0, 20.0], index=dates)

    growth = _calculate_yoy_growth(values, list(dates))

    assert pd.isna(growth.iloc[1])
    assert pd.isna(growth.iloc[2])


def test_missing_statement_returns_empty_trends():
    trends = build_financial_trends({})

    assert trends.empty


def test_summary_does_not_call_negative_to_positive_a_growth_rate():
    trends = pd.DataFrame(
        {
            "Revenue": [100.0, 120.0],
            "Net Income": [-10.0, 5.0],
            "EPS": [1.0, 1.5],
            "Free Cash Flow": [10.0, 12.0],
        },
        index=pd.to_datetime(["2024-03-31", "2025-03-31"]),
    )

    observations = summarize_financial_trends(trends)

    net_income_observation = next(
        item for item in observations if item.startswith("Net Income:")
    )

    assert "negative" in net_income_observation
    assert "Conventional cumulative percentage growth is not reported" in (
        net_income_observation
    )
