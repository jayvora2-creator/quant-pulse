
from app.ai_research_agent import research_stock


def main():
    ticker = "RELIANCE.NS"

    question = (
        "Analyze this company's historical risk and return metrics, "
        "valuation indicators, and financial trends. Explain the main "
        "strengths, risks, and limitations. Use the supplied figures "
        "and do not make a buy or sell recommendation."
    )

    try:
        print(f"\nAI STOCK RESEARCH TEST: {ticker}")
        print("=" * 60)

        answer = research_stock(
            ticker=ticker,
            question=question,
            period="1y",
        )

        print(answer)

    except Exception as exc:
        print(f"\nTEST FAILED: {exc}")


if __name__ == "__main__":
    main()
