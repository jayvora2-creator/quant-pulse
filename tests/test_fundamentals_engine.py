
from app.fundamentals_engine import get_company_fundamentals


def main():
    # Test one US company and one Indian company.
    tickers = ["NVDA", "RELIANCE.NS"]

    for ticker in tickers:
        print("\n" + "=" * 55)
        print(f"FUNDAMENTALS TEST: {ticker}")
        print("=" * 55)

        try:
            result = get_company_fundamentals(ticker)

            profile = result["profile"]
            print(f"Company: {profile['company_name']}")
            print(f"Sector: {profile['sector']}")
            print(f"Industry: {profile['industry']}")
            print(f"Currency: {profile['currency']}")
            print(f"Source: {result['source']}")
            print(f"Retrieved at (UTC): {result['retrieved_at_utc']}")

            print("\nVALUATION AND FUNDAMENTALS")

            for name, value in result["valuation"].items():
                if value is not None:
                    print(f"{name}: {value}")

            print("\nFINANCIAL STATEMENT AVAILABILITY")

            for statement_name in [
                "income_statement",
                "balance_sheet",
                "cash_flow",
            ]:
                statement = result[statement_name]

                if statement.empty:
                    print(f"{statement_name}: Not available")
                else:
                    print(f"\n{statement_name}:")
                    print(statement.iloc[:8, :3])

        except Exception as exc:
            print(f"FAILED for {ticker}: {exc}")


if __name__ == "__main__":
    main()
