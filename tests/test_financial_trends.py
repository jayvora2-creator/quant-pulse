
from app.fundamentals_engine import get_company_fundamentals
from app.financial_trends import (
    build_financial_trends,
    summarize_financial_trends,
)


def main():
    for ticker in ["NVDA", "RELIANCE.NS"]:
        print("\n" + "=" * 55)
        print(f"FINANCIAL TREND TEST: {ticker}")
        print("=" * 55)

        try:
            fundamentals = get_company_fundamentals(ticker)
            trends = build_financial_trends(fundamentals)

            if trends.empty:
                print("No historical financial trends available.")
                continue

            print("\nHistorical financial data:")
            print(trends.to_string())

            print("\nTrend observations:")
            for observation in summarize_financial_trends(trends):
                print("-", observation)

        except Exception as exc:
            print(f"FAILED for {ticker}: {exc}")


if __name__ == "__main__":
    main()
