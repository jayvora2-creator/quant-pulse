
import numpy as np
import pandas as pd


METRICS = ["Revenue", "Net Income", "EPS", "Free Cash Flow"]


def _find_row(statement, possible_names):
    """Prefer exact row names before considering unambiguous partial matches."""
    if statement is None or statement.empty:
        return None

    normalized = {
        str(index).strip().lower(): index
        for index in statement.index
    }

    for name in possible_names:
        match = normalized.get(name.strip().lower())
        if match is not None:
            return match

    matches = []
    for index in statement.index:
        label = str(index).strip().lower()
        if any(name.strip().lower() in label for name in possible_names):
            matches.append(index)

    # Do not guess if partial matching identifies multiple rows.
    return matches[0] if len(matches) == 1 else None


def _extract_annual_series(statement, possible_names):
    """Extract numeric values indexed by the provider's reporting dates."""
    metric_name = possible_names[0]

    if statement is None or statement.empty:
        return pd.Series(dtype=float, name=metric_name)

    row = _find_row(statement, possible_names)
    if row is None:
        return pd.Series(dtype=float, name=metric_name)

    raw_values = statement.loc[row]
    if isinstance(raw_values, pd.DataFrame):
        if len(raw_values) != 1:
            return pd.Series(dtype=float, name=metric_name)
        raw_values = raw_values.iloc[0]

    values = pd.to_numeric(raw_values, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()

    dates = pd.to_datetime(values.index, errors="coerce")
    valid = ~dates.isna()
    values = values.loc[valid]
    dates = dates[valid]

    if values.empty:
        return pd.Series(dtype=float, name=metric_name)

    result = pd.Series(
        values.to_numpy(dtype=float),
        index=pd.DatetimeIndex(dates),
        name=metric_name,
    )
    return result.groupby(level=0).last().sort_index()


def _calculate_yoy_growth(series, reporting_dates):
    """Calculate YoY growth only for consecutive annual reporting dates."""
    growth = pd.Series(np.nan, index=series.index, dtype=float)

    if series.empty:
        return growth

    for position in range(1, len(series)):
        current_date = pd.Timestamp(reporting_dates[position])
        previous_date = pd.Timestamp(reporting_dates[position - 1])

        current_value = series.iloc[position]
        previous_value = series.iloc[position - 1]

        # Annual observations must be about one year apart.
        if current_date.year - previous_date.year != 1:
            continue

        # Exclude unusual gaps between reporting anniversaries.
        if abs((current_date - previous_date).days - 365) > 3:
            continue

        if pd.isna(current_value) or pd.isna(previous_value):
            continue
        if previous_value <= 0:
            continue

        growth.iloc[position] = current_value / previous_value - 1

    return growth


def _reporting_label(date):
    """Format a date without guessing the fiscal year for non-March reports."""
    date = pd.Timestamp(date)

    if date.month == 3 and date.day in (30, 31):
        return f"{date:%Y-%m-%d} (FY {date.year - 1}-{str(date.year)[-2:]})"

    return f"{date:%Y-%m-%d} (reporting date; fiscal year not inferred)"


def build_financial_trends(fundamentals):
    """Build historical trends without fabricating unavailable data."""
    fundamentals = fundamentals or {}

    income = fundamentals.get("income_statement", pd.DataFrame())
    cash_flow = fundamentals.get("cash_flow", pd.DataFrame())

    revenue = _extract_annual_series(
        income, ["Total Revenue", "Operating Revenue", "Revenue"]
    )
    net_income = _extract_annual_series(
        income,
        [
            "Net Income",
            "Net Income Common Stockholders",
            "Net Income From Continuing Operation Net Minority Interest",
        ],
    )
    eps = _extract_annual_series(income, ["Diluted EPS", "Basic EPS"])
    free_cash_flow = _extract_annual_series(cash_flow, ["Free Cash Flow"])

    trends = pd.concat(
        [
            revenue.rename("Revenue"),
            net_income.rename("Net Income"),
            eps.rename("EPS"),
            free_cash_flow.rename("Free Cash Flow"),
        ],
        axis=1,
    ).sort_index()

    if trends.empty:
        return trends

    reporting_dates = list(trends.index)

    for metric in METRICS:
        trends[f"{metric} YoY Growth"] = _calculate_yoy_growth(
            trends[metric], reporting_dates
        )

    return trends


def summarize_financial_trends(trends):
    """Describe cumulative changes and latest valid YoY growth separately."""
    if trends is None or trends.empty:
        return ["No usable historical financial data is available."]

    observations = []

    for metric in METRICS:
        if metric not in trends.columns:
            continue

        values = trends[metric].dropna()
        growth_column = f"{metric} YoY Growth"

        if len(values) < 2:
            observations.append(
                f"{metric}: insufficient historical observations "
                "to assess cumulative change."
            )
        else:
            first_date = pd.Timestamp(values.index[0])
            last_date = pd.Timestamp(values.index[-1])
            first = float(values.iloc[0])
            last = float(values.iloc[-1])

            elapsed_days = (last_date - first_date).days
            elapsed_years = elapsed_days / 365.2425

            if first <= 0:
                observations.append(
                    f"{metric}: changed from {first:,.4f} on "
                    f"{first_date:%Y-%m-%d} to {last:,.4f} on "
                    f"{last_date:%Y-%m-%d}. Conventional cumulative "
                    "percentage growth is not reported because the "
                    "starting value was zero or negative."
                )
            else:
                change = last / first - 1
                observations.append(
                    f"{metric}: cumulative change was {change:+.1%} "
                    f"from {_reporting_label(first_date)} to "
                    f"{_reporting_label(last_date)} "
                    f"({elapsed_years:.2f} elapsed years; "
                    f"starting value {first:,.4f}, ending value "
                    f"{last:,.4f}). This is not annual growth."
                )

        if growth_column in trends.columns:
            valid_growth = trends[growth_column].dropna()

            if valid_growth.empty:
                observations.append(
                    f"{metric}: no valid YoY growth rate is available "
                    "from consecutive annual observations with a "
                    "positive prior-year value."
                )
            else:
                growth_date = pd.Timestamp(valid_growth.index[-1])
                growth_value = float(valid_growth.iloc[-1])
                observations.append(
                    f"{metric}: latest valid YoY growth was "
                    f"{growth_value:+.1%} for reporting date "
                    f"{_reporting_label(growth_date)}."
                )

    return observations
