
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


# Fields are optional because availability differs by company and market.
FUNDAMENTAL_FIELDS = {
    "marketCap": "Market Capitalization",
    "trailingPE": "Trailing P/E",
    "forwardPE": "Forward P/E",
    "priceToBook": "Price-to-Book",
    "priceToSalesTrailing12Months": "Price-to-Sales",
    "enterpriseToEbitda": "EV/EBITDA",
    "enterpriseToRevenue": "EV/Revenue",
    "dividendYield": "Dividend Yield",
    "beta": "Beta",
    "trailingEps": "Trailing EPS",
    "forwardEps": "Forward EPS",
    "totalRevenue": "Revenue",
    "netIncomeToCommon": "Net Income",
    "grossMargins": "Gross Margin",
    "operatingMargins": "Operating Margin",
    "profitMargins": "Net Profit Margin",
    "returnOnEquity": "Return on Equity",
    "returnOnAssets": "Return on Assets",
    "debtToEquity": "Debt-to-Equity",
    "currentRatio": "Current Ratio",
    "freeCashflow": "Free Cash Flow",
}


def _get_statement(ticker_object, statement_name):
    """Return a financial statement, or an empty DataFrame if unavailable."""

    try:
        statement = getattr(ticker_object, statement_name)
        if isinstance(statement, pd.DataFrame):
            return statement
    except Exception:
        pass

    return pd.DataFrame()


def get_company_fundamentals(ticker: str) -> dict:
    """
    Retrieve available company information and financial statements.

    Data is retrieved from Yahoo Finance through yfinance.
    Missing fields are left unavailable rather than estimated.
    """

    ticker = ticker.strip()

    if not ticker:
        raise ValueError("Please enter a stock ticker.")

    company = yf.Ticker(ticker)

    try:
        info = company.info or {}
    except Exception as exc:
        raise RuntimeError(
            f"Could not retrieve company information for {ticker}. "
            "Check the ticker or try again later."
        ) from exc

    income_statement = _get_statement(company, "income_stmt")
    balance_sheet = _get_statement(company, "balance_sheet")
    cash_flow = _get_statement(company, "cashflow")

    if not info and income_statement.empty and balance_sheet.empty:
        raise RuntimeError(
            f"No fundamental data was returned for {ticker}. "
            "The ticker may be unsupported or the provider may be "
            "temporarily unavailable."
        )

    company_profile = {
        "ticker": ticker,
        "company_name": info.get("longName")
        or info.get("shortName")
        or ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "country": info.get("country"),
        "currency": info.get("currency"),
        "exchange": info.get("exchange"),
        "website": info.get("website"),
        "business_summary": info.get("longBusinessSummary"),
    }

    valuation = {
        label: info.get(field)
        for field, label in FUNDAMENTAL_FIELDS.items()
    }

    return {
        "profile": company_profile,
        "valuation": valuation,
        "income_statement": income_statement,
        "balance_sheet": balance_sheet,
        "cash_flow": cash_flow,
        "source": "Yahoo Finance via yfinance",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
