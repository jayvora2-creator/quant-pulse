
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from app.data_provider import get_stock_data
from app.quant_engine import analyze_stock
from app.fundamentals_engine import get_company_fundamentals
from app.financial_trends import (
    build_financial_trends,
    summarize_financial_trends,
)


load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-20b"


SYSTEM_PROMPT = """
You are an AI financial research assistant producing evidence-based
stock research reports.

STRICT EVIDENCE AND REPORTING RULES:

1. Never claim a stock outperformed or underperformed the market unless
   both the stock and benchmark returns for the same dates are supplied.

2. Never invent the reason for a difference between two beta values.
   State that the reason is unverified unless the underlying periods,
   benchmarks, and calculation methods are provided.

3. Map fiscal years from reporting dates correctly:
   - 31 March 2024 = FY 2023-24
   - 31 March 2025 = FY 2024-25
   - 31 March 2026 = FY 2025-26
   If uncertain, display the reporting date instead.

4. Calculate cumulative percentage change using the first and last
   supplied values. State the actual elapsed period between those dates.
   Do not call a three-year interval a four-year period merely because
   four annual observations are present.

5. Never claim margins are stable without historical margin observations.
   Never call leverage manageable or moderate without adequate evidence.

6. Never describe valuation as cheap, expensive, moderate, attractive,
   or a premium without relevant comparison data.

7. Do not compare stock volatility with index volatility unless both
   values are supplied for the same dates and calculated comparably.

8. Do not invent data dates, provider methodologies, peer comparisons,
   liquidity metrics, or explanations.

9. Before writing the final report, check every factual interpretation
   against the supplied data. If evidence is missing, explicitly say so.

10. Treat supplied financial figures as provider-reported data, not
    independently audited figures. Clearly disclose missing or uncertain
    units and definitions.


FISCAL-YEAR AND FINANCIAL-TREND RULES:
- Use the reporting dates supplied in the financial data.
- For Indian companies, a reporting date of 31 March 2026 normally
  corresponds to FY 2025-26, not FY 2026-27.
- Do not infer fiscal-year labels that conflict with reporting dates.
- If the reporting period or fiscal-year convention is uncertain,
  show the reporting date instead of guessing.
- Distinguish year-over-year growth from cumulative growth over
  multiple years.
- Never describe a multi-year percentage change as annual growth.
- Use the supplied growth-rate columns when available.
- Do not invent or recalculate financial figures that conflict
  with the supplied data.


STRICT ACCURACY RULES:
1. Use only the supplied evidence for company-specific claims.
2. Never invent prices, financial metrics, dates, sources, or figures.
3. Clearly distinguish facts, interpretations, and missing information.
4. Never compare a company with industry averages, peers, or a market
   benchmark unless relevant comparison data is explicitly supplied.
5. Never claim a stock outperformed or underperformed a benchmark unless
   both returns are supplied for comparable periods.
6. Do not describe annualized return as total return.
7. Historical returns and risk metrics are not forecasts.
8. Explain important uncertainties and limitations.
9. Do not promise returns or give personalized financial advice.
10. If data is missing or contradictory, disclose the limitation.
11. If calculated beta differs from provider-reported beta, report both
    separately. Explain that methodologies may differ.
12. Do not call a stock cheap, expensive, fairly valued, moderately
    valued, or attractive without relevant comparison evidence.
13. If peer or industry valuation data is unavailable, explicitly state
    that relative valuation cannot be assessed.
14. Never claim that valuation multiples are within a typical range
    without supporting comparison data.
15. Do not describe leverage as moderate or manageable without a
    relevant benchmark.
16. Do not describe margins as stable unless comparable historical
    margin data supports that conclusion.
17. Do not infer improved liquidity from free cash flow alone.
18. If quantitative metrics are supplied, report them accurately.
19. Do not claim a metric is unavailable when its calculated value
    is supplied in the evidence.
20. Do not infer causation from historical correlations or trends.
21. Do not describe a metric as high, low, healthy, or poor without
    sufficient context or a clearly stated basis.
22. Do not invent explanations for dividend yield or other unusual
    provider-reported values.

FINANCIAL METRIC RULES:
- Use the units explicitly stated in the evidence.
- Do not convert percentage values a second time.
- A decimal ratio such as 0.75 represents 75% only when the source
  defines that metric as a decimal fraction.
- Do not assume every provider-reported ratio uses the same scale.
- Price-to-earnings, price-to-book, and EV/EBITDA are valuation
  multiples, not percentages.
- Debt-to-equity may use a different scale from profitability ratios.
- Distinguish provider-reported figures from application calculations.
- Historical VaR at 95% confidence is a historical loss-threshold
  estimate, not a guaranteed maximum loss. Losses can exceed it.
- Do not claim that a negative Sharpe ratio means returns were below
  the risk-free rate unless the calculation explicitly uses a specified
  risk-free rate and that assumption is supplied.
- A negative Sharpe or Sortino ratio must be interpreted in the
  context of the calculation method and supplied evidence.
- Beta describes estimated sensitivity to a benchmark, not total risk.
  Do not use beta alone to label a stock low-risk or high-risk.
- Do not describe historical volatility as a forecast of future risk.
- If the risk-free-rate assumption or a metric's calculation method is
  unknown, state the limitation rather than guessing.
- Treat provider-reported dividend yield cautiously when its units
  or calculation convention are not verified.

REPORT STRUCTURE:
1. Executive Summary
2. Quantitative Performance
3. Fundamentals and Financial Trends
4. Key Risks
5. Data Limitations
6. Balanced Conclusion

Use concise, professional language. Explain what supplied metrics
mean without overstating what they prove. Do not make a buy or sell
recommendation unless explicitly requested. Never present unsupported
conclusions as facts.
"""


def _get_client():
    """Create the Groq client using settings from the project .env file."""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Check your project-root .env file."
        )

    model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)
    return Groq(api_key=api_key, timeout=60.0), model


def generate_research_answer(
    question: str,
    research_context: str = "",
) -> str:
    """Generate an AI answer using the supplied financial evidence."""

    question = question.strip()
    research_context = research_context.strip()

    if not question:
        raise ValueError("Please enter a research question.")

    client, model = _get_client()

    prompt = f"""
USER QUESTION:
{question}

AVAILABLE FINANCIAL EVIDENCE:
{research_context if research_context else
 "No company-specific financial evidence was supplied."}

Instructions:
- Follow the system accuracy rules.
- Use the six required report sections for stock research.
- Do not invent missing data or unsupported valuation comparisons.
- Explain the limitations of historical metrics.
- Answer the user's question directly.
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_completion_tokens=3000,
            reasoning_effort="low",
            include_reasoning=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Groq request failed: {type(exc).__name__}: {exc}"
        ) from exc

    if not response.choices:
        raise RuntimeError("Groq returned no response choices.")

    choice = response.choices[0]
    answer = choice.message.content

    if not answer or not answer.strip():
        raise RuntimeError(
            "Groq returned no report text. "
            f"Finish reason: {choice.finish_reason}. "
            "Try again or check the selected model and API status."
        )

    return answer.strip()


def format_fundamental_metric(name: str, value) -> str:
    """Format fundamental values without silently changing their units."""

    if value is None:
        return "unavailable"

    try:
        if pd.isna(value):
            return "unavailable"
    except (TypeError, ValueError):
        pass

    metric_name = name.strip().lower()

    # This provider field was observed in percentage-point form for the
    # project's current data source. Do not multiply it by 100 again.
    # If the provider changes its convention, verify the source first.
    if metric_name == "dividend yield":
        return (
            f"{float(value):.2f}% "
            "(provider-reported value; displayed without conversion)"
        )

    # These fields are expected to be decimal fractions from the
    # fundamentals engine, for example 0.25 meaning 25%.
    percentage_fraction_metrics = {
        "gross margin",
        "operating margin",
        "net profit margin",
        "return on equity",
        "return on assets",
    }

    if metric_name in percentage_fraction_metrics:
        return (
            f"{float(value) * 100:.2f}% "
            "(converted from provider decimal fraction)"
        )

    # Preserve the provider's scale. Do not assume the number is a
    # decimal fraction or convert it without checking the source.
    if metric_name == "debt-to-equity":
        return (
            f"{float(value):.2f} "
            "(provider-reported scale; not converted)"
        )

    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):,.4f}".rstrip("0").rstrip(".")

    return str(value)


def _format_quantitative_metric(name: str, value) -> str:
    """Format metrics calculated by the quantitative engine."""

    if value is None:
        return "unavailable"

    try:
        if not pd.notna(value):
            return "unavailable"
    except (TypeError, ValueError):
        return "unavailable"

    percentage_metrics = {
        "annualized_return",
        "annualized_volatility",
        "maximum_drawdown",
        "historical_var_95",
    }

    numeric_value = float(value)

    if name in percentage_metrics:
        formatted = f"{numeric_value * 100:.2f}%"

        if name == "historical_var_95":
            return (
                f"{formatted} "
                "(historical 95% VaR estimate; losses can exceed this)"
            )

        return formatted

    return f"{numeric_value:.4f}"

def build_stock_research_context(
    ticker: str,
    period: str = "1y",
) -> str:
    """
    Retrieve market and fundamental data and calculate metrics using
    the project's existing data and analysis engines.
    """

    ticker = ticker.strip().upper()

    if not ticker:
        raise ValueError("Please enter a stock ticker.")

    # Select a broad index proxy based on the ticker format.
    benchmark_ticker = "^NSEI" if ticker.endswith(".NS") else "^GSPC"

    stock_data = get_stock_data(ticker, period=period)

    if stock_data is None or stock_data.empty:
        raise ValueError(
            f"No market data was returned for {ticker}. "
            "Check the ticker symbol and try again."
        )

    if "Close" not in stock_data.columns:
        raise ValueError(
            f"Closing prices are unavailable for {ticker}."
        )

    latest_close = pd.to_numeric(
        stock_data["Close"], errors="coerce"
    ).dropna()

    if latest_close.empty:
        raise ValueError(
            f"No usable closing prices were found for {ticker}."
        )

    # Benchmark data is useful but optional.
    benchmark_data = None
    benchmark_status = "unavailable"

    try:
        benchmark_data = get_stock_data(
            benchmark_ticker,
            period=period,
        )

        if benchmark_data is not None and not benchmark_data.empty:
            benchmark_status = "retrieved"
        else:
            benchmark_data = None

    except Exception as exc:
        benchmark_data = None
        benchmark_status = (
            f"retrieval failed ({type(exc).__name__})"
        )

    metrics = analyze_stock(stock_data, benchmark_data)

    # Calculate the benchmark's total price return over the retrieved
    # observations. This does not include dividends.
    benchmark_return = None

    if benchmark_data is not None and not benchmark_data.empty:
        try:
            benchmark_close = pd.to_numeric(
                benchmark_data["Close"], errors="coerce"
            ).dropna()

            if len(benchmark_close) >= 2:
                first_close = float(benchmark_close.iloc[0])
                last_close = float(benchmark_close.iloc[-1])

                if first_close > 0:
                    benchmark_return = (
                        last_close / first_close
                    ) - 1.0

        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            benchmark_return = None

    context_lines = [
        f"Requested ticker: {ticker}",
        "Market data source: Yahoo Finance via yfinance",
        (
            "Data retrieval time (UTC): "
            f"{datetime.now(timezone.utc).isoformat()}"
        ),
        f"Requested history: {period}",
        (
            "First available stock observation: "
            f"{stock_data.index.min()}"
        ),
        (
            "Last available stock observation: "
            f"{stock_data.index.max()}"
        ),
        (
            "Latest available closing price: "
            f"{latest_close.iloc[-1]}"
        ),
        f"Benchmark used: {benchmark_ticker}",
        f"Benchmark data status: {benchmark_status}",
        (
            "Benchmark total price return over retrieved observations: "
            f"{benchmark_return * 100:.2f}%"
            if benchmark_return is not None
            else "Benchmark total price return: unavailable"
        ),
        "Risk-free-rate assumption used for Sharpe and Sortino: 0% annually",
        "Benchmark return is a total price return, not an annualized return.",
        (
            "Do not claim market outperformance or underperformance unless "
            "stock and benchmark returns use comparable dates and periods."
        ),
        "",
        "QUANTITATIVE METRICS",
        (
            "These are application-calculated historical metrics unless "
            "otherwise identified. They are not forecasts."
        ),
    ]

    for name, value in metrics.items():
        context_lines.append(
            f"{name}: {_format_quantitative_metric(name, value)}"
        )

    # Fundamental data is optional.
    try:
        fundamentals = get_company_fundamentals(ticker)

        if not isinstance(fundamentals, dict):
            raise ValueError(
                "The fundamentals engine returned an unexpected format."
            )

        profile = fundamentals.get("profile") or {}
        valuation = fundamentals.get("valuation") or {}

        context_lines.extend([
            "",
            "COMPANY PROFILE",
            f"Company name: {profile.get('company_name', 'unavailable')}",
            f"Sector: {profile.get('sector', 'unavailable')}",
            f"Industry: {profile.get('industry', 'unavailable')}",
            f"Reporting currency: {profile.get('currency', 'unavailable')}",
            "",
            "FUNDAMENTAL METRICS (provider-reported values)",
        ])

        for name, value in valuation.items():
            if name.lower() == "beta":
                label = (
                    "Provider-reported beta "
                    "(Yahoo Finance; methodology may differ)"
                )
            else:
                label = name

            formatted_value = format_fundamental_metric(name, value)
            context_lines.append(f"{label}: {formatted_value}")

        trends = build_financial_trends(fundamentals)

        if trends is not None and not trends.empty:
            context_lines.extend([
                "",
                "HISTORICAL FINANCIAL TRENDS",
                trends.to_string(),
                "",
                "TREND OBSERVATIONS",
            ])

            observations = summarize_financial_trends(trends)

            if observations:
                context_lines.extend(
                    str(item) for item in observations
                )
            else:
                context_lines.append(
                    "No additional trend observations were produced."
                )

        else:
            context_lines.extend([
                "",
                "HISTORICAL FINANCIAL TRENDS",
                "Comparable historical trend data is unavailable.",
            ])

        context_lines.extend([
            "",
            "Fundamental data source: "
            f"{fundamentals.get('source', 'not specified')}",
        ])

    except Exception as exc:
        context_lines.extend([
            "",
            "FUNDAMENTAL DATA STATUS",
            "Fundamental data could not be fully retrieved or processed.",
            f"Reason: {type(exc).__name__}: {exc}",
            "Do not infer missing fundamental values.",
        ])

    context_lines.extend([
        "",
        "DATA LIMITATIONS",
        (
            "Market data represents the historical observations returned "
            "by the provider and is not necessarily a real-time quote."
        ),
        "Annualized statistics are historical calculations, not forecasts.",
        (
            "The benchmark is a broad index proxy and may not be the most "
            "appropriate benchmark for every company."
        ),
        (
            "Historical VaR is an estimate based on historical data, "
            "not a guaranteed maximum loss."
        ),
        (
            "Do not infer a risk-free rate, peer valuation, or industry "
            "benchmark unless the relevant evidence is supplied."
        ),
    ])

    return "\n".join(context_lines)


def research_stock(
    ticker: str,
    question: str,
    period: str = "1y",
) -> str:
    """Build the evidence context and generate an AI research report."""

    context = build_stock_research_context(
        ticker=ticker,
        period=period,
    )

    return generate_research_answer(
        question=question,
        research_context=context,
    )
