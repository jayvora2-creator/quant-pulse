
import yfinance as yf
import pandas as pd


def get_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Download historical stock market data from Yahoo Finance.

    Examples:
        RELIANCE.NS -> Reliance Industries
        TCS.NS      -> Tata Consultancy Services
        AAPL        -> Apple
        NVDA        -> NVIDIA
    """
    ticker = ticker.strip()

    if not ticker:
        raise ValueError("Please enter a valid stock ticker.")

    try:
        data = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if data.empty:
            raise ValueError(
                f"No market data found for {ticker}. "
                "Check the ticker symbol or try again later."
            )

        # Handle multi-level columns returned by yfinance.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.loc[:, ~data.columns.duplicated()]
        data = data.dropna(how="all")

        if data.empty:
            raise ValueError(f"No usable price data found for {ticker}.")

        return data

    except Exception as e:
        raise RuntimeError(
            f"Unable to retrieve market data for {ticker}: {e}"
        ) from e


def search_stock_tickers(query: str, max_results: int = 10) -> list:
    """
    Search Yahoo Finance for stocks by company name or ticker.
    """
    query = query.strip()

    if not query:
        return []

    try:
        search = yf.Search(
            query,
            max_results=max_results,
            news_count=0,
        )

        quotes = search.quotes or []
        results = []

        for item in quotes:
            if item.get("quoteType") != "EQUITY":
                continue

            symbol = item.get("symbol")

            if not symbol:
                continue

            results.append({
                "symbol": symbol,
                "name": (
                    item.get("longname")
                    or item.get("shortname")
                    or symbol
                ),
                "exchange": item.get("exchange", ""),
            })

        return results

    except Exception as e:
        raise RuntimeError(
            f"Stock search failed for '{query}': {e}"
        ) from e
