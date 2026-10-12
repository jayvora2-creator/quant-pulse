import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from app.data_provider import get_stock_data
from app.quant_engine import (
    analyze_stock,
    compare_stock_with_benchmark,
)
from app.portfolio_engine import (
    analyze_portfolio,
    optimize_portfolio,
)
from app.backtesting_engine import backtest_strategy
from app.fundamentals_engine import get_company_fundamentals
from app.financial_trends import (
    build_financial_trends,
    summarize_financial_trends,
)
from app.ai_research_agent import research_stock

# --------------------------------------------------
# QUANT//PULSE — PAGE CONFIGURATION & VISUAL THEME
# --------------------------------------------------

st.set_page_config(
    page_title="QUANT | Investment Research",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
:root {
    --qp-bg: #07140d;
    --qp-surface: #0d2116;
    --qp-surface-raised: #132b1d;
    --qp-border: #244432;
    --qp-text: #edf5ee;
    --qp-muted: #a3b7a8;
    --qp-green: #46c58a;
    --qp-green-soft: #a9efc9;
    --qp-green-deep: #0d3020;
    --qp-cream: #e9f3e8;
    --qp-red: #e8877b;
}

html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", sans-serif;
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--qp-bg);
    color: var(--qp-text);
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    width: 100%;
    max-width: 1600px;
    padding: 1rem clamp(.75rem, 2.5vw, 2rem) 2rem;
}

h1, h2, h3, h4 {
    color: var(--qp-text);
    letter-spacing: -.025em;
}

p, label, [data-testid="stMarkdownContainer"] {
    color: #d7e4d9;
}

[data-testid="stSidebar"] {
    background: #091b11;
    border-right: 1px solid var(--qp-border);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
    color: var(--qp-green-soft);
    letter-spacing: -.04em;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: var(--qp-muted);
    letter-spacing: .05em;
}

[data-testid="stMetric"] {
    background: var(--qp-surface);
    border: 1px solid var(--qp-border);
    border-radius: 13px;
    padding: 14px 16px;
    min-height: 96px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, .08);
}

[data-testid="stMetricLabel"] {
    color: var(--qp-muted) !important;
    font-size: .78rem !important;
}

[data-testid="stMetricValue"] {
    color: var(--qp-text) !important;
    font-weight: 700 !important;
    overflow-wrap: anywhere;
}

[data-testid="stMetricDelta"] {
    font-weight: 600;
}

.stButton > button,
.stFormSubmitButton > button {
    background: var(--qp-green);
    color: #062013 !important;
    border: 1px solid var(--qp-green);
    border-radius: 9px;
    font-weight: 700;
    transition: background .15s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #70dca7;
    border-color: #70dca7;
    color: #062013 !important;
}

.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
[data-baseweb="select"] > div {
    background: #0a1c12 !important;
    color: var(--qp-text) !important;
    border-color: var(--qp-border) !important;
    border-radius: 8px !important;
}

[data-testid="stExpander"] {
    background: var(--qp-surface);
    border: 1px solid var(--qp-border);
    border-radius: 10px;
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid var(--qp-border);
    border-radius: 9px;
    overflow: hidden;
}

hr {
    border-color: var(--qp-border) !important;
}

[data-testid="stAlert"] {
    border-radius: 10px;
}

[data-testid="stCaptionContainer"] {
    color: var(--qp-muted);
}

[data-testid="stMarkdownContainer"] a {
    color: var(--qp-green-soft);
}

/* Hero banner */
.qp-hero {
    background: linear-gradient(115deg, #43c58b, #54d09a);
    border: 1px solid rgba(169, 239, 201, .35);
    border-radius: 17px;
    padding: clamp(18px, 3vw, 30px);
    margin-bottom: 1.25rem;
    color: #082b1a;
}

.qp-hero * {
    color: #082b1a !important;
}

.qp-eyebrow {
    font-size: .72rem;
    letter-spacing: .14em;
    font-weight: 800;
    text-transform: uppercase;
    opacity: .78;
}

.qp-hero-title {
    font-size: clamp(1.45rem, 3vw, 2.15rem);
    line-height: 1.16;
    font-weight: 800;
    letter-spacing: -.045em;
    margin: .55rem 0;
}

.qp-hero-copy {
    max-width: 760px;
    font-size: .95rem;
    line-height: 1.55;
    margin: 0;
}

/* Research workspace cards */
.qp-module {
    background: linear-gradient(145deg, #10281a, #0c1d13);
    border: 1px solid #244432;
    border-radius: 13px;
    padding: 17px;
    min-height: 142px;
    margin-bottom: 8px;
    overflow-wrap: anywhere;
    transition: border-color .15s ease;
}

.qp-module:hover {
    border-color: #46c58a;
}

.qp-module-number {
    color: #80d9a8;
    font-size: .7rem;
    font-weight: 800;
    letter-spacing: .1em;
}

.qp-module-title {
    color: #edf5ee;
    font-size: 1rem;
    font-weight: 700;
    margin: .55rem 0 .35rem;
}

.qp-module-copy {
    color: #a9bdad;
    font-size: .82rem;
    line-height: 1.45;
    margin: 0;
}

[data-testid="stPlotlyChart"] {
    width: 100%;
    min-width: 0;
}

/* Tablet and mobile adjustments */
@media (max-width: 900px) {
    .block-container {
        padding-left: .8rem;
        padding-right: .8rem;
    }

    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: .7rem !important;
    }

    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        min-width: min(100%, 240px) !important;
        flex: 1 1 240px !important;
    }

    .qp-module {
        min-height: auto;
    }
}

@media (max-width: 480px) {
    .block-container {
        padding-top: .5rem;
        padding-left: .65rem;
        padding-right: .65rem;
    }

    [data-testid="stMetric"] {
        padding: 11px;
        min-height: auto;
    }

    [data-testid="stMetricLabel"] {
        font-size: .7rem !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.15rem !important;
    }

    .qp-hero {
        border-radius: 13px;
    }
}
</style>
""", unsafe_allow_html=True)


# Terminal header
st.markdown("""
<div style="
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:16px;
    flex-wrap:wrap;
    padding:8px 0 18px 0;
    border-bottom:1px solid #30384a;
    margin-bottom:20px;
">
  <div>
    <div style="
        font-size:27px;
        font-weight:900;
        letter-spacing:-1.5px;
        color:#f7f8ff;
    ">
      QUANT<span style="color:#ff9d32;">//</span>PULSE
    </div>
    <div style="
        font-size:10px;
        letter-spacing:3px;
        color:#929db1;
        margin-top:3px;
    ">
      AI INVESTMENT RESEARCH TERMINAL
    </div>
  </div>
  <div style="text-align:right;">
    <span style="color:#35e6a0;font-size:11px;">● DATA WORKSPACE</span>
    <div style="font-size:11px;color:#929db1;margin-top:5px;">
      INDIA / US · HISTORICAL ANALYTICS
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Market research • Quantitative risk • Portfolio construction • AI-assisted analysis"
)

st.info(
    "Research and educational use only. Historical data availability "
    "depends on the provider; this application does not guarantee "
    "institutional real-time market feeds."
)

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def search_stock_tickers(query, max_results=10):
    """Search for supported stocks by company name or ticker."""

    query = query.strip()

    if not query:
        return []

    search = yf.Search(
        query,
        max_results=max_results,
        news_count=0,
    )

    results = []

    for item in search.quotes or []:
        if item.get("quoteType") != "EQUITY":
            continue

        ticker = item.get("symbol")

        if not ticker:
            continue

        results.append({
            "symbol": ticker,
            "name": (
                item.get("longname")
                or item.get("shortname")
                or ticker
            ),
            "exchange": item.get("exchange", ""),
        })

    return results


def get_close_prices(data):
    """Extract a clean closing-price series from downloaded data."""

    if data is None or data.empty:
        raise ValueError("No market data was returned.")

    if isinstance(data.columns, pd.MultiIndex):
        data = data.copy()
        data.columns = data.columns.get_level_values(0)

    column = "Adj Close" if "Adj Close" in data.columns else "Close"

    if column not in data.columns:
        raise ValueError("The downloaded data has no closing-price column.")

    prices = data[column]

    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]

    prices = pd.to_numeric(prices, errors="coerce")
    prices = prices.dropna().sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]

    if prices.empty:
        raise ValueError("No valid closing prices were found.")

    return prices


def download_price_history(ticker, period):
    """Download historical prices for one selected ticker."""

    data = get_stock_data(ticker, period)
    return get_close_prices(data)


def format_percentage(value):
    return f"{value:.2%}"




# --------------------------------------------------
# QUANT — SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.markdown("## QUANT")
st.sidebar.caption("INVESTMENT RESEARCH TERMINAL")
def go_to_workspace(title):
    st.session_state["workspace_navigation"] = title
selected_page = st.sidebar.radio(
    "WORKSPACE",
    [
        "Overview",
        "Stock Research",
        "Portfolio Analysis",
        "Portfolio Optimization",
        "Backtesting",
        "Fundamentals & Valuation",
        "AI Research Agent",
    ],
    key="workspace_navigation",
)


st.sidebar.divider()
st.sidebar.caption("INDIA • US MARKETS")
st.sidebar.caption("Quantitative Research")

if selected_page == "Overview":
    st.markdown(
        """
        <div style="
            background:linear-gradient(120deg,#171c28,#251638,#102b35);
            border:1px solid #343b4d;
            border-radius:18px;
            padding:28px;
        ">
            <div style="color:#39dfff;letter-spacing:3px;font-size:12px;">
                QUANTITATIVE RESEARCH TERMINAL
            </div>
            <h1 style="color:#ffffff;font-size:38px;">
                QUANT<span style="color:#ff9d32;">//</span>PULSE
            </h1>
            <p style="color:#b5c0d4;">
                Research markets. Measure risk. Build better portfolios.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### About QUANT//PULSE")

    st.write(
        "QUANT//PULSE is an AI-powered quantitative investment "
        "research platform designed to help users analyse financial "
        "markets, evaluate investment risk, and explore portfolio "
        "strategies using market data and financial analytics."
    )

    st.markdown("#### Key Capabilities")

    st.markdown(
        """
        - **Stock Research:** Historical returns, volatility, Sharpe "
        "and Sortino ratios, beta, and maximum drawdown.
        - **Portfolio Analysis:** Performance, risk exposure, "
        "diversification, and correlation.
        - **Portfolio Optimization:** Minimum-volatility and "
        "maximum-Sharpe portfolio allocations.
        - **Strategy Backtesting:** Test strategies against "
        "historical market data.
        - **Fundamentals & Valuation:** Financial statements, "
        "valuation ratios, profitability, and financial trends.
        - **AI Research Agent:** Structured, AI-assisted research reports.
        """
    )

    st.markdown("#### Data & Methodology")

    st.write(
        "The application retrieves market data through Yahoo Finance "
        "using yfinance. Data availability and freshness depend on "
        "the provider."
    )

    st.caption(
        "Independent research project for educational and analytical "
        "purposes. Not personalized investment advice."
    )


    st.markdown("### MARKET RADAR")
    st.caption("Latest available daily market data; not guaranteed real-time.")

    cols = st.columns(4)

    indices = [
        ("NIFTY 50", "^NSEI"),
        ("SENSEX", "^BSESN"),
        ("S&P 500", "^GSPC"),
        ("NASDAQ", "^IXIC"),
    ]

    for col, (name, symbol) in zip(cols, indices):
        with col:
            try:
                history = yf.Ticker(symbol).history(period="5d")

                if history.empty:
                    st.metric(name, "Unavailable")
                else:
                    latest = float(history["Close"].iloc[-1])

                    if len(history) >= 2:
                        previous = float(history["Close"].iloc[-2])
                        change = (latest / previous - 1) * 100
                        st.metric(name, f"{latest:,.2f}", f"{change:+.2f}%")
                    else:
                        st.metric(name, f"{latest:,.2f}")

            except Exception:
                st.metric(name, "Unavailable")

    st.divider()
    
    st.markdown("### RESEARCH WORKSPACE")

    cards = [
        ("01", "Stock Research", "Returns, volatility, beta and drawdown."),
        ("02", "Portfolio Analysis", "Performance, risk and diversification."),
        ("03", "Portfolio Optimization", "Explore optimized allocations."),
        ("04", "Backtesting", "Test historical investment strategies."),
        ("05", "Fundamentals & Valuation", "Financials and valuation ratios."),
        ("06", "AI Research Agent", "AI-assisted investment research."),
    ]

    for start in range(0, len(cards), 3):
        row = st.columns(3)

        for col, card in zip(row, cards[start:start + 3]):
            number, title, description = card

            with col:
                st.markdown(
                    f"""
                    <div style="
                        background:linear-gradient(145deg,#171c28,#10131b);
                        border:1px solid #30384a;
                        border-radius:14px;
                        padding:18px;
                        min-height:135px;
                        margin-bottom:8px;
                    ">
                        <div style="color:#ff9d32;font-size:12px;">
                            MODULE {number}
                        </div>
                        <h4 style="color:#f4f7ff;margin:10px 0;">
                            {title}
                        </h4>
                        <p style="color:#a7b1c5;font-size:13px;">
                            {description}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.button(
                   f"Open {title}",
                   key=f"open_{number}",
                   use_container_width=True,
                   on_click=go_to_workspace,
                   args=(title,),
            )
# ==================================================
# TAB 1: SINGLE STOCK RESEARCH
# ==================================================

if selected_page == "Stock Research":

    st.subheader("Single Stock Research")

    st.write(
        "Enter a supported Yahoo Finance ticker to analyze a company."
    )

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        stock_ticker = st.text_input(
            "Stock ticker",
            value="RELIANCE.NS",
            key="single_stock_ticker",
            help="Examples: TATAMOTORS.NS, INFY.NS, NVDA, MSFT, AAPL",
        ).strip().upper()

    with col2:
        benchmark_ticker = st.text_input(
            "Benchmark ticker",
            value="^NSEI",
            key="single_benchmark_ticker",
            help="Examples: ^NSEI for NIFTY 50, SPY for a US-market benchmark.",
        ).strip().upper()

    with col3:
        stock_period = st.selectbox(
            "Historical period",
            ["6mo", "1y", "2y", "5y"],
            index=1,
            key="single_stock_period",
        )

    if st.button("Analyze Stock", type="primary", key="analyze_single"):

        try:
            if not stock_ticker or not benchmark_ticker:
                raise ValueError(
                    "Enter both a stock ticker and a benchmark ticker."
                )

            with st.spinner("Downloading market data and calculating metrics..."):

                stock_data = get_stock_data(
                    stock_ticker,
                    stock_period,
                )

                benchmark_data = get_stock_data(
                    benchmark_ticker,
                    stock_period,
                )

                stock_prices = get_close_prices(stock_data)

                if len(stock_prices) < 2:
                    raise ValueError(
                        "At least two daily prices are needed."
                    )

                results = analyze_stock(
                    stock_data,
                    benchmark_data,
                )
            comparison = compare_stock_with_benchmark(
                      stock_data,
                      benchmark_data,
                )
            latest_price = float(stock_prices.iloc[-1])
            previous_price = float(stock_prices.iloc[-2])

            daily_change = (
                latest_price / previous_price - 1
            ) * 100

            st.divider()
            st.subheader(f"Research results: {stock_ticker}")

            st.caption(
                f"Latest available closing date: "
                f"{stock_prices.index[-1].date()}"
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Latest available close",
                f"{latest_price:,.2f}",
                f"{daily_change:+.2f}%",
            )

            c2.metric(
                "Annualized return",
                format_percentage(results["annualized_return"]),
            )

            c3.metric(
                "Annualized volatility",
                format_percentage(results["annualized_volatility"]),
            )

            c4, c5, c6, c7 = st.columns(4)

            c4.metric(
                "Sharpe ratio",
                f"{results['sharpe_ratio']:.2f}",
            )

            c5.metric(
                "Sortino ratio",
                f"{results['sortino_ratio']:.2f}",
            )

            c6.metric(
                "Maximum drawdown",
                format_percentage(results["maximum_drawdown"]),
            )

            c7.metric(
                "Beta vs benchmark",
                f"{results['beta']:.2f}",
            )

            st.metric(
                "95% historical daily VaR",
                format_percentage(results["historical_var_95"]),
            )
            
            st.divider()
            st.subheader("Stock vs Benchmark Comparison")

            if comparison is None:
                st.warning(
                    "A comparable return analysis could not be calculated. "
                    "Check whether both instruments have sufficient "
                    "overlapping historical price data."
                )
            else:
                st.caption(
                    f"Stock: {stock_ticker} | "
                    f"Benchmark: {benchmark_ticker}"
                )

                comp1, comp2, comp3 = st.columns(3)

                comp1.metric(
                    "Stock annualized price return",
                    format_percentage(
                        comparison["stock_annualized_return"]
                    ),
                )

                comp2.metric(
                    "Benchmark annualized price return",
                    format_percentage(
                        comparison["benchmark_annualized_return"]
                    ),
                )

                comp3.metric(
                    "Annualized return difference",
                    f"{comparison['annualized_return_difference'] * 100:+.2f} pp",
                )

                comp4, comp5 = st.columns(2)

                comp4.metric(
                    "Stock total price return",
                    format_percentage(
                        comparison["stock_total_price_return"]
                    ),
                )

                comp5.metric(
                    "Benchmark total price return",
                    format_percentage(
                        comparison["benchmark_total_price_return"]
                    ),
                )

                st.caption(
                    f"Comparison period: {comparison['start_date']} "
                    f"to {comparison['end_date']} | "
                    f"Overlapping price observations: "
                    f"{comparison['overlapping_price_observations']}"
                )

                st.info(
                    "Both instruments are compared over the same dates. "
                    "Returns are based on prices and exclude dividends. "
                    "The annualized return difference is expressed in "
                    "percentage points (pp). Historical performance "
                    "does not predict future returns."
                )

            st.subheader("Historical closing price")

            chart_data = stock_prices.rename("Closing Price").reset_index()
            chart_data.columns = ["Date", "Closing Price"]

            fig = px.line(
                chart_data,
                x="Date",
                y="Closing Price",
                title=f"{stock_ticker} historical closing prices",
            )

            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                f"Source: Yahoo Finance via yfinance | "
                f"Requested period: {stock_period}"
            )

        except Exception as error:
            st.error(f"Stock analysis failed: {error}")

            st.caption(
                "Check the ticker, benchmark, internet connection, "
                "and availability of historical data."
            )


# ==================================================
# TAB 2: PORTFOLIO ANALYSIS
# ==================================================

if selected_page == "Portfolio Analysis":

    st.subheader("Build Your Own Portfolio")

    st.write(
        "Search for companies, select their tickers, "
        "and assign your own portfolio weights."
    )

    st.caption(
        "Indian examples: TATAMOTORS.NS, RELIANCE.NS, INFY.NS. "
        "US examples: NVDA, MSFT, AAPL. "
        "The search feature retrieves available results; "
        "not every company or exchange is guaranteed to be supported."
    )

    # ----------------------------------------------
    # INITIAL PORTFOLIO
    # ----------------------------------------------

    if "portfolio_rows" not in st.session_state:

        st.session_state.portfolio_rows = [
            {
                "id": 1,
                "ticker": "RELIANCE.NS",
                "weight": 40.0,
            },
            {
                "id": 2,
                "ticker": "TCS.NS",
                "weight": 30.0,
            },
            {
                "id": 3,
                "ticker": "AAPL",
                "weight": 30.0,
            },
        ]

        st.session_state.next_portfolio_id = 4

    if "next_portfolio_id" not in st.session_state:
        st.session_state.next_portfolio_id = 4

    rows = st.session_state.portfolio_rows

    st.markdown("### Portfolio holdings")

    # ----------------------------------------------
    # COMPANY SEARCH AND EDITABLE WEIGHTS
    # ----------------------------------------------

    for row in rows:

        row_id = row["id"]

        st.markdown("---")

        company_col, weight_col, remove_col = st.columns(
            [4, 2, 1]
        )

        with company_col:

            st.markdown(f"**Stock {row_id}**")

            query = st.text_input(
                "Search company name or ticker",
                value=row["ticker"],
                key=f"portfolio_query_{row_id}",
                placeholder="Try Tata, NVIDIA, Microsoft...",
            )

            search_col, manual_col = st.columns(2)

            with search_col:

                if st.button(
                    "Search companies",
                    key=f"search_button_{row_id}",
                    use_container_width=True,
                ):

                    try:
                        with st.spinner("Searching available stocks..."):

                            matches = search_stock_tickers(query)

                        st.session_state[
                            f"portfolio_matches_{row_id}"
                        ] = matches

                        if not matches:
                            st.warning(
                                "No matching stocks found. "
                                "Try another search term."
                            )

                    except Exception as error:

                        st.error(
                            f"Stock search failed: {error}"
                        )

            with manual_col:

                if st.button(
                    "Use ticker entered",
                    key=f"manual_button_{row_id}",
                    use_container_width=True,
                ):

                    if not query.strip():

                        st.warning("Enter a ticker first.")

                    else:

                        row["ticker"] = query.strip().upper()

                        st.session_state.pop(
                            f"portfolio_matches_{row_id}",
                            None,
                        )

                        st.rerun()

            matches = st.session_state.get(
                f"portfolio_matches_{row_id}",
                [],
            )

            if matches:

                choices = [
                    (
                        f"{item['symbol']} | "
                        f"{item['name']} | "
                        f"{item['exchange']}"
                    )
                    for item in matches
                ]

                selected_choice = st.selectbox(
                    "Matching companies",
                    choices,
                    key=f"portfolio_selection_{row_id}",
                )

                selected_index = choices.index(selected_choice)

                selected_ticker = matches[selected_index]["symbol"]

                if st.button(
                    "Select this company",
                    key=f"select_button_{row_id}",
                    type="primary",
                ):

                    row["ticker"] = selected_ticker

                    st.session_state.pop(
                        f"portfolio_matches_{row_id}",
                        None,
                    )

                    st.rerun()

            st.caption(f"Current ticker: {row['ticker'] or 'Not selected'}")

        with weight_col:

            row["weight"] = st.number_input(
                "Allocation weight (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(row["weight"]),
                step=5.0,
                key=f"portfolio_weight_{row_id}",
            )

        with remove_col:

            st.write("")

            if st.button(
                "Remove",
                key=f"remove_stock_{row_id}",
                help="Remove this stock from the portfolio.",
            ):

                st.session_state.portfolio_rows = [
                    item
                    for item in st.session_state.portfolio_rows
                    if item["id"] != row_id
                ]

                st.session_state.pop(
                    f"portfolio_matches_{row_id}",
                    None,
                )

                st.rerun()

    # ----------------------------------------------
    # ADD A NEW STOCK
    # ----------------------------------------------

    st.markdown("")

    if st.button(
        "＋ Add another stock",
        use_container_width=True,
    ):

        new_id = st.session_state.next_portfolio_id

        st.session_state.next_portfolio_id += 1

        st.session_state.portfolio_rows.append(
            {
                "id": new_id,
                "ticker": "",
                "weight": 0.0,
            }
        )

        st.rerun()

    # ----------------------------------------------
    # ALLOCATION VALIDATION
    # ----------------------------------------------

    rows = st.session_state.portfolio_rows

    total_weight = sum(
        float(row["weight"])
        for row in rows
    )

    st.divider()

    metric_col1, metric_col2 = st.columns(2)

    metric_col1.metric(
        "Total allocation",
        f"{total_weight:.2f}%",
    )

    metric_col2.metric(
        "Number of holdings",
        len(rows),
    )

    if abs(total_weight - 100.0) < 0.01:

        st.success("Your portfolio weights total 100%.")

    else:

        st.warning(
            f"Your weights currently total {total_weight:.2f}%. "
            "Adjust them to total exactly 100% before analysis."
        )

    portfolio_period = st.selectbox(
        "Portfolio historical period",
        ["6mo", "1y", "2y", "5y"],
        index=1,
        key="portfolio_historical_period",
    )

    # ----------------------------------------------
    # RUN PORTFOLIO ANALYSIS
    # ----------------------------------------------

    if st.button(
        "Analyze Portfolio",
        type="primary",
        use_container_width=True,
        key="analyze_portfolio_button",
    ):

        try:

            rows = st.session_state.portfolio_rows

            if not rows:
                raise ValueError(
                    "Add at least one stock to your portfolio."
                )

            tickers = [
                row["ticker"].strip().upper()
                for row in rows
            ]

            if any(not ticker for ticker in tickers):

                raise ValueError(
                    "Every holding needs a ticker. "
                    "Remove unused rows or select a company."
                )

            if len(tickers) != len(set(tickers)):

                raise ValueError(
                    "Duplicate tickers found. "
                    "Each company should appear only once."
                )

            weights_percent = [
                float(row["weight"])
                for row in rows
            ]

            if any(weight <= 0 for weight in weights_percent):

                raise ValueError(
                    "Every selected stock must have a positive weight."
                )

            if abs(sum(weights_percent) - 100.0) > 0.01:

                raise ValueError(
                    "Portfolio weights must total exactly 100%."
                )

            weights = {
                row["ticker"].strip().upper():
                float(row["weight"]) / 100.0
                for row in rows
            }

            # Download actual market data for selected tickers.
            with st.spinner(
                "Downloading market data for your selected stocks..."
            ):

                price_series = {}

                for ticker in weights:

                    try:

                        prices = download_price_history(
                            ticker,
                            portfolio_period,
                        )

                        price_series[ticker] = prices.rename(ticker)

                    except Exception as error:

                        raise ValueError(
                            f"Could not retrieve data for {ticker}. "
                            f"Check the ticker and provider availability. "
                            f"Details: {error}"
                        ) from error

                price_data = pd.concat(
                    list(price_series.values()),
                    axis=1,
                    sort=False,
                )

                price_data = price_data.sort_index().dropna(how="any")

                if len(price_data) < 2:

                    raise ValueError(
                        "There are not enough common trading dates "
                        "across your selected stocks. Try a longer period "
                        "or different tickers."
                    )

                results = analyze_portfolio(
                    price_data,
                    weights,
                )

            # --------------------------------------
            # PORTFOLIO PERFORMANCE METRICS
            # --------------------------------------

            st.divider()

            st.subheader("Portfolio performance and risk")

            st.caption(
                f"Historical data through {price_data.index[-1].date()}"
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Annualized return",
                format_percentage(results["annualized_return"]),
            )

            c2.metric(
                "Annualized volatility",
                format_percentage(
                    results["annualized_volatility"]
                ),
            )

            c3.metric(
                "Sharpe ratio",
                f"{results['sharpe_ratio']:.2f}",
            )

            c4, c5, c6 = st.columns(3)

            c4.metric(
                "Sortino ratio",
                f"{results['sortino_ratio']:.2f}",
            )

            c5.metric(
                "Maximum drawdown",
                format_percentage(results["maximum_drawdown"]),
            )

            c6.metric(
                "95% daily historical VaR",
                format_percentage(results["historical_var_95"]),
            )

            # --------------------------------------
            # INDIVIDUAL STOCK PERFORMANCE
            # --------------------------------------

            st.subheader("Individual stock performance")

            individual_df = pd.DataFrame.from_dict(
                results["individual_results"],
                orient="index",
            )

            individual_df.index.name = "Ticker"

            st.dataframe(
                individual_df.style.format({
                    "weight": "{:.1%}",
                    "annualized_return": "{:.2%}",
                    "annualized_volatility": "{:.2%}",
                }),
                use_container_width=True,
            )

            # --------------------------------------
            # RISK CONTRIBUTIONS
            # --------------------------------------

            st.subheader("Contribution to portfolio risk")

            risk_df = pd.DataFrame.from_dict(
                results["risk_contributions"],
                orient="index",
            )

            risk_df.index.name = "Ticker"

            st.dataframe(
                risk_df.style.format({
                    "annualized_volatility_contribution": "{:.2%}",
                    "percentage_of_portfolio_risk": "{:.2%}",
                }),
                use_container_width=True,
            )

            # --------------------------------------
            # CORRELATION MATRIX
            # --------------------------------------

            st.subheader("Stock return correlation")

            st.dataframe(
                results["correlation_matrix"].style.format("{:.2f}"),
                use_container_width=True,
            )

            # --------------------------------------
            # NORMALIZED PRICE COMPARISON
            # --------------------------------------

            st.subheader("Historical stock performance")

            normalized_prices = (
                price_data / price_data.iloc[0]
            ) * 100

            normalized_prices.index.name = "Date"

            chart_data = normalized_prices.reset_index().melt(
                id_vars="Date",
                var_name="Ticker",
                value_name="Growth of 100",
            )

            fig = px.line(
                chart_data,
                x="Date",
                y="Growth of 100",
                color="Ticker",
                title="Growth of 100 units invested at the start",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.caption(
                "Source: Yahoo Finance via yfinance. "
                "Metrics are historical estimates, not predictions "
                "of future performance."
            )

        except Exception as error:

            st.error(
                f"Portfolio analysis failed: {error}"
            )

# ==================================================
# TAB 3: PORTFOLIO OPTIMIZATION
# ==================================================

if selected_page == "Portfolio Optimization":

    st.subheader("Portfolio Optimization")

    st.write(
        "Optimize your selected portfolio using historical market data. "
        "Compare a minimum-volatility portfolio with a maximum-Sharpe "
        "portfolio and explore the efficient frontier."
    )

    st.info(
        "Optimization uses historical returns and long-only allocations. "
        "Results are estimates, not predictions or investment advice."
    )

    optimization_period = st.selectbox(
        "Optimization historical period",
        ["6mo", "1y", "2y", "5y"],
        index=1,
        key="optimization_period",
    )

    if st.button(
        "Optimize Portfolio",
        type="primary",
        use_container_width=True,
        key="run_portfolio_optimization",
    ):

        try:
            rows = st.session_state.get("portfolio_rows", [])

            # Read the tickers selected in Portfolio Analysis.
            tickers = [
                row["ticker"].strip().upper()
                for row in rows
                if row.get("ticker", "").strip()
            ]

            if len(tickers) < 2:
                raise ValueError(
                    "Select at least two stocks in the Portfolio "
                    "Analysis tab before optimizing."
                )

            if len(tickers) != len(set(tickers)):
                raise ValueError(
                    "Duplicate tickers found. Keep each stock only once."
                )

            with st.spinner(
                "Downloading historical prices and optimizing allocations..."
            ):
                price_series = {}

                for ticker in tickers:
                    prices = download_price_history(
                        ticker,
                        optimization_period,
                    )
                    price_series[ticker] = prices.rename(ticker)

                price_data = pd.concat(
                    list(price_series.values()),
                    axis=1,
                    sort=False,
                ).sort_index().dropna(how="any")

                if price_data.shape[0] < 30:
                    raise ValueError(
                        "Insufficient overlapping price history. "
                        "Try a longer period or different stocks."
                    )

                if price_data.shape[1] < 2:
                    raise ValueError(
                        "At least two stocks need valid overlapping prices."
                    )

                optimization = optimize_portfolio(price_data)

            st.session_state["last_optimization"] = optimization
            st.session_state["last_optimization_tickers"] = tickers
            st.session_state["last_optimization_period"] = (
                optimization_period
            )

        except Exception as error:
            st.error(f"Portfolio optimization failed: {error}")

    # Show the most recent successful results.
    optimization = st.session_state.get("last_optimization")

    if optimization is not None:

        previous_tickers = st.session_state.get(
            "last_optimization_tickers", []
        )

        current_tickers = [
            row["ticker"].strip().upper()
            for row in st.session_state.get("portfolio_rows", [])
            if row.get("ticker", "").strip()
        ]

        # Avoid presenting old results as if they match new selections.
        if set(previous_tickers) != set(current_tickers):

            st.warning(
                "Your portfolio selections have changed. Click "
                "'Optimize Portfolio' again to refresh the results."
            )

        else:
            st.divider()

            st.caption(
                "Historical data period: "
                f"{st.session_state.get('last_optimization_period', '')}"
            )

            min_vol = optimization["minimum_volatility"]
            max_sharpe = optimization["maximum_sharpe"]

            st.subheader("1. Minimum-Volatility Portfolio")

            st.write(
                "Finds an allocation designed to minimize estimated "
                "annualized portfolio volatility."
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Annualized return",
                f"{min_vol['annualized_return']:.2%}",
            )

            c2.metric(
                "Annualized volatility",
                f"{min_vol['annualized_volatility']:.2%}",
            )

            c3.metric(
                "Sharpe ratio",
                f"{min_vol['sharpe_ratio']:.2f}",
            )

            min_weights_df = pd.DataFrame({
                "Ticker": list(min_vol["weights"].keys()),
                "Optimized allocation (%)": [
                    weight * 100
                    for weight in min_vol["weights"].values()
                ],
            })

            st.dataframe(
                min_weights_df.style.format({
                    "Optimized allocation (%)": "{:.2f}%"
                }),
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("2. Maximum-Sharpe Portfolio")

            st.write(
                "Finds an allocation designed to maximize estimated "
                "return relative to volatility, using the optimizer's "
                "configured risk-free rate."
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Annualized return",
                f"{max_sharpe['annualized_return']:.2%}",
            )

            c2.metric(
                "Annualized volatility",
                f"{max_sharpe['annualized_volatility']:.2%}",
            )

            c3.metric(
                "Sharpe ratio",
                f"{max_sharpe['sharpe_ratio']:.2f}",
            )

            max_weights_df = pd.DataFrame({
                "Ticker": list(max_sharpe["weights"].keys()),
                "Optimized allocation (%)": [
                    weight * 100
                    for weight in max_sharpe["weights"].values()
                ],
            })

            st.dataframe(
                max_weights_df.style.format({
                    "Optimized allocation (%)": "{:.2f}%"
                }),
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("3. Compare Optimized Allocations")

            current_weights = {
                row["ticker"].strip().upper():
                float(row["weight"]) / 100
                for row in st.session_state.get("portfolio_rows", [])
                if row.get("ticker", "").strip()
            }

            comparison = pd.DataFrame({
                "Ticker": optimization["tickers"],
                "Current allocation (%)": [
                    current_weights.get(ticker, 0) * 100
                    for ticker in optimization["tickers"]
                ],
                "Minimum-volatility (%)": [
                    min_vol["weights"].get(ticker, 0) * 100
                    for ticker in optimization["tickers"]
                ],
                "Maximum-Sharpe (%)": [
                    max_sharpe["weights"].get(ticker, 0) * 100
                    for ticker in optimization["tickers"]
                ],
            })

            st.dataframe(
                comparison.style.format({
                    "Current allocation (%)": "{:.2f}%",
                    "Minimum-volatility (%)": "{:.2f}%",
                    "Maximum-Sharpe (%)": "{:.2f}%",
                }),
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("4. Efficient Frontier")

            frontier = optimization["efficient_frontier"]

            if frontier is not None and not frontier.empty:

                fig = px.scatter(
                    frontier,
                    x="Annualized Volatility (%)",
                    y="Annualized Return (%)",
                    color="Sharpe Ratio",
                    title="Historical Risk–Return Trade-off",
                    hover_data=["Sharpe Ratio"],
                )

                fig.add_scatter(
                    x=[min_vol["annualized_volatility"] * 100],
                    y=[min_vol["annualized_return"] * 100],
                    mode="markers",
                    name="Minimum Volatility",
                    marker={"size": 13, "symbol": "diamond"},
                )

                fig.add_scatter(
                    x=[max_sharpe["annualized_volatility"] * 100],
                    y=[max_sharpe["annualized_return"] * 100],
                    mode="markers",
                    name="Maximum Sharpe",
                    marker={"size": 13, "symbol": "star"},
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

                st.caption(
                    "Each point represents a feasible optimized allocation "
                    "for a target historical return. The frontier is an "
                    "approximation based on the available data."
                )

            else:
                st.warning(
                    "The optimizer did not produce enough efficient-frontier "
                    "points to display a chart."
                )

            st.caption(
                "Methodology: long-only weights, no leverage, historical "
                "daily returns annualized using 252 trading days. "
                "Past performance does not guarantee future results."
            )

if selected_page == "Backtesting":
    st.header("Strategy Backtesting")
    st.write(
        "Test historical trading strategies against a buy-and-hold benchmark. "
        "Past performance does not guarantee future results."
    )

    col1, col2 = st.columns(2)

    with col1:
        backtest_ticker = st.text_input(
            "Stock ticker",
            value="NVDA",
            key="backtest_ticker",
            help="Examples: NVDA, AAPL, RELIANCE.NS, TCS.NS",
        ).strip().upper()

        backtest_period = st.selectbox(
            "Historical period",
            ["6mo", "1y", "2y", "5y"],
            index=2,
            key="backtest_period",
        )

        strategy_name = st.selectbox(
            "Trading strategy",
            ["SMA Crossover", "Momentum", "Buy and Hold"],
            key="backtest_strategy_name",
        )

    with col2:
        short_window = st.number_input(
            "Short moving-average window (days)",
            min_value=2,
            max_value=100,
            value=20,
            key="backtest_short_window",
        )

        long_window = st.number_input(
            "Long moving-average window (days)",
            min_value=3,
            max_value=250,
            value=50,
            key="backtest_long_window",
        )

        momentum_window = st.number_input(
            "Momentum lookback (days)",
            min_value=1,
            max_value=250,
            value=20,
            key="backtest_momentum_window",
        )

        transaction_cost_bps = st.number_input(
            "Transaction cost (basis points)",
            min_value=0,
            max_value=100,
            value=10,
            key="backtest_transaction_cost",
            help="10 basis points = 0.10% per position change.",
        )

    if st.button("Run Backtest", type="primary", key="run_backtest"):
        try:
            if not backtest_ticker:
                st.error("Enter a valid stock ticker.")
            elif (
                strategy_name == "SMA Crossover"
                and short_window >= long_window
            ):
                st.error(
                    "The short moving-average window must be smaller "
                    "than the long window."
                )
            else:
                with st.spinner(
                    f"Downloading historical data for {backtest_ticker}..."
                ):
                    price_data = download_price_history(
                        backtest_ticker,
                        backtest_period,
                    )

                strategy_map = {
                    "SMA Crossover": "sma_crossover",
                    "Momentum": "momentum",
                    "Buy and Hold": "buy_and_hold",
                }

                result = backtest_strategy(
                    price_data,
                    strategy=strategy_map[strategy_name],
                    short_window=int(short_window),
                    long_window=int(long_window),
                    momentum_window=int(momentum_window),
                    transaction_cost_bps=int(transaction_cost_bps),
                )

                metrics = result["metrics"]
                benchmark = result["benchmark_metrics"]

                st.subheader(f"Results for {backtest_ticker}")
                st.caption(
                    f"Period selected: {backtest_period} | "
                    f"Strategy: {strategy_name} | "
                    f"Transaction cost: {transaction_cost_bps} bps"
                )

                metric_cols = st.columns(5)

                metric_cols[0].metric(
                    "Strategy return",
                    f"{metrics['total_return']:.2%}",
                )
                metric_cols[1].metric(
                    "Annualized return",
                    f"{metrics['annualized_return']:.2%}",
                )
                metric_cols[2].metric(
                    "Annualized volatility",
                    f"{metrics['annualized_volatility']:.2%}",
                )
                metric_cols[3].metric(
                    "Sharpe ratio",
                    f"{metrics['sharpe_ratio']:.2f}",
                )
                metric_cols[4].metric(
                    "Maximum drawdown",
                    f"{metrics['maximum_drawdown']:.2%}",
                )

                st.write(
                    f"**Position changes:** {result['number_of_trades']}"
                )

                st.subheader("Strategy vs. Buy and Hold")

                comparison = pd.DataFrame(
                    {
                        "Strategy": [
                            metrics["total_return"],
                            metrics["annualized_return"],
                            metrics["annualized_volatility"],
                            metrics["sharpe_ratio"],
                            metrics["maximum_drawdown"],
                        ],
                        "Buy and Hold": [
                            benchmark["total_return"],
                            benchmark["annualized_return"],
                            benchmark["annualized_volatility"],
                            benchmark["sharpe_ratio"],
                            benchmark["maximum_drawdown"],
                        ],
                    },
                    index=[
                        "Total Return",
                        "Annualized Return",
                        "Annualized Volatility",
                        "Sharpe Ratio",
                        "Maximum Drawdown",
                    ],
                )

                st.dataframe(comparison, use_container_width=True)

                results = result["results"]

                st.subheader("Equity Curve")

                equity_chart = px.line(
                    results,
                    y=["Strategy Equity", "Buy and Hold Equity"],
                    labels={
                        "index": "Date",
                        "value": "Growth of ₹1",
                        "variable": "Portfolio",
                    },
                    title="Growth of ₹1 Under Each Approach",
                )
                st.plotly_chart(equity_chart, use_container_width=True)

                st.subheader("Drawdown Comparison")

                drawdown_chart = px.line(
                    results,
                    y=[
                        "Strategy Drawdown",
                        "Buy and Hold Drawdown",
                    ],
                    labels={
                        "index": "Date",
                        "value": "Drawdown",
                        "variable": "Portfolio",
                    },
                    title="Decline from Previous Equity Peak",
                )
                drawdown_chart.update_yaxes(tickformat=".0%")
                st.plotly_chart(
                    drawdown_chart,
                    use_container_width=True,
                )

                with st.expander("View daily backtest results"):
                    st.dataframe(
                        results,
                        use_container_width=True,
                    )

                st.info(
                    "Backtest results are historical simulations, not "
                    "investment recommendations. They may exclude taxes, "
                    "slippage, and other real-world trading costs."
                )

        except Exception as e:
            st.error(f"Backtest could not be completed: {e}")

if selected_page == "Fundamentals & Valuation":
    st.header("Stock Fundamentals & Valuation")
    st.write(
        "Explore company financials, valuation ratios, profitability, "
        "balance-sheet indicators, and cash flow."
    )

    ticker_input = st.text_input(
        "Enter stock ticker",
        value="NVDA",
        key="fundamentals_ticker",
        help="Examples: NVDA, AAPL, RELIANCE.NS, TCS.NS",
    ).strip().upper()

    st.caption(
        "Data source: Yahoo Finance via yfinance. "
        "Availability and reporting periods vary by company."
    )

    if st.button(
        "Load Fundamentals",
        type="primary",
        key="load_fundamentals",
    ):
        try:
            if not ticker_input:
                st.error("Please enter a stock ticker.")
            else:
                with st.spinner(
                    f"Loading fundamentals for {ticker_input}..."
                ):
                    fundamentals = get_company_fundamentals(ticker_input)

                st.session_state["fundamentals_result"] = fundamentals
                st.session_state["fundamentals_loaded_ticker"] = ticker_input

        except Exception as exc:
            st.error(f"Could not load fundamentals: {exc}")

    fundamentals = st.session_state.get("fundamentals_result")
    loaded_ticker = st.session_state.get("fundamentals_loaded_ticker")

    if fundamentals and loaded_ticker == ticker_input:
        profile = fundamentals["profile"]
        valuation = fundamentals["valuation"]

        st.divider()
        st.subheader(profile["company_name"])

        profile_details = [
            f"Ticker: {profile['ticker']}",
            f"Sector: {profile['sector'] or 'Not available'}",
            f"Industry: {profile['industry'] or 'Not available'}",
            f"Exchange: {profile['exchange'] or 'Not available'}",
            f"Currency: {profile['currency'] or 'Not available'}",
            f"Country: {profile['country'] or 'Not available'}",
        ]
        st.write(" | ".join(profile_details))

        st.caption(
            f"Retrieved at: {fundamentals['retrieved_at_utc']} "
            "(UTC; retrieval time, not provider update time)"
        )

        def available(value):
            if value is None:
                return False
            try:
                return bool(pd.notna(value))
            except (TypeError, ValueError):
                return False

        def compact_number(value, currency=""):
            if not available(value):
                return "Not available"

            value = float(value)
            absolute = abs(value)

            if absolute >= 1_000_000_000_000:
                text = f"{value / 1_000_000_000_000:,.2f}T"
            elif absolute >= 1_000_000_000:
                text = f"{value / 1_000_000_000:,.2f}B"
            elif absolute >= 1_000_000:
                text = f"{value / 1_000_000:,.2f}M"
            else:
                text = f"{value:,.2f}"

            return f"{text} {currency}".strip()

        def ratio_text(value):
            if not available(value):
                return "Not available"
            return f"{float(value):,.2f}x"

        def percent_text(value):
            if not available(value):
                return "Not available"
            return f"{float(value) * 100:.2f}%"

        currency = profile.get("currency") or ""

        st.subheader("Key Financial Indicators")

        key_metrics = [
            (
                "Market Capitalization",
                compact_number(valuation.get("Market Capitalization"), currency),
            ),
            (
                "Revenue",
                compact_number(valuation.get("Revenue"), currency),
            ),
            (
                "Net Income",
                compact_number(valuation.get("Net Income"), currency),
            ),
            (
                "Free Cash Flow",
                compact_number(valuation.get("Free Cash Flow"), currency),
            ),
        ]

        metric_cols = st.columns(4)
        for col, (label, value) in zip(metric_cols, key_metrics):
            col.metric(label, value)

        st.subheader("Valuation Ratios")

        valuation_rows = [
            ("Trailing P/E", valuation.get("Trailing P/E"), ratio_text),
            ("Forward P/E", valuation.get("Forward P/E"), ratio_text),
            ("Price-to-Book", valuation.get("Price-to-Book"), ratio_text),
            ("Price-to-Sales", valuation.get("Price-to-Sales"), ratio_text),
            ("EV/EBITDA", valuation.get("EV/EBITDA"), ratio_text),
            ("EV/Revenue", valuation.get("EV/Revenue"), ratio_text),
            ("Trailing EPS", valuation.get("Trailing EPS"), None),
            ("Forward EPS", valuation.get("Forward EPS"), None),
            ("Dividend Yield", valuation.get("Dividend Yield"), None),
        ]

        valuation_table = []
        for label, value, formatter in valuation_rows:
            if label == "Dividend Yield":
                # Yahoo Finance dividend-yield units can vary by field/version.
                if available(value):
                    numeric_value = float(value)
                    display_value = (
                        f"{numeric_value:.2f}%"
                        if numeric_value > 0.1
                        else f"{numeric_value * 100:.2f}%"
                    )
                else:
                    display_value = "Not available"
            elif formatter is not None:
                display_value = formatter(value)
            elif available(value):
                display_value = (
                    f"{float(value):,.2f} {currency}".strip()
                )
            else:
                display_value = "Not available"

            valuation_table.append(
                {"Metric": label, "Value": display_value}
            )

        st.dataframe(
            pd.DataFrame(valuation_table),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Profitability and Financial Health")

        health_rows = [
            ("Gross Margin", valuation.get("Gross Margin"), percent_text),
            ("Operating Margin", valuation.get("Operating Margin"), percent_text),
            ("Net Profit Margin", valuation.get("Net Profit Margin"), percent_text),
            ("Return on Equity", valuation.get("Return on Equity"), percent_text),
            ("Return on Assets", valuation.get("Return on Assets"), percent_text),
            ("Debt-to-Equity", valuation.get("Debt-to-Equity"), ratio_text),
            ("Current Ratio", valuation.get("Current Ratio"), ratio_text),
            ("Beta", valuation.get("Beta"), ratio_text),
        ]

        health_table = []
        for label, value, formatter in health_rows:
            health_table.append(
                {
                    "Metric": label,
                    "Value": formatter(value),
                }
            )

        st.dataframe(
            pd.DataFrame(health_table),
            use_container_width=True,
            hide_index=True,
        )

        statement_tabs = st.tabs(
            ["Income Statement", "Balance Sheet", "Cash Flow"]
        )

        statement_items = [
            ("income_statement", statement_tabs[0]),
            ("balance_sheet", statement_tabs[1]),
            ("cash_flow", statement_tabs[2]),
        ]

        for statement_name, statement_tab in statement_items:
            with statement_tab:
                statement = fundamentals[statement_name]

                if statement.empty:
                    st.info(
                        "This financial statement is not available "
                        "for the selected company."
                    )
                else:
                    st.caption(
                        "Values and reporting periods are provided by "
                        "Yahoo Finance. Financial statement amounts may "
                        "be in the company's reporting currency and units."
                    )
                    st.dataframe(
                        statement,
                        use_container_width=True,
                    )

        business_summary = profile.get("business_summary")
        if business_summary:
            with st.expander("About the company"):
                st.write(business_summary)

        st.caption(
            "Research tool only, not investment advice. Valuation ratios "
            "should be compared with relevant peers, growth, accounting "
            "quality, and business risks. Missing data is not estimated."
        )

        st.divider()
        st.subheader("Financial Trend Analysis")

        trends = build_financial_trends(fundamentals)

        if trends.empty:
            st.warning(
                "No usable historical financial data is available "
                "for trend analysis."
            )
        else:
            trend_display = trends.copy()
            trend_display.index = trend_display.index.strftime("%Y-%m-%d")

            st.markdown("**Historical financial performance**")

            chart_metrics = [
                metric
                for metric in [
                    "Revenue",
                    "Net Income",
                    "EPS",
                    "Free Cash Flow",
                ]
                if metric in trends.columns
                and trends[metric].notna().any()
            ]

            if chart_metrics:
                selected_metric = st.selectbox(
                    "Financial metric",
                    chart_metrics,
                    key=f"trend_metric_{loaded_ticker}",
                )

                metric_chart = px.line(
                    trend_display.reset_index(names="Reporting Date"),
                    x="Reporting Date",
                    y=selected_metric,
                    markers=True,
                    title=f"{selected_metric} by Reporting Period",
                )

                metric_chart.update_layout(
                    xaxis_title="Reporting Date",
                    yaxis_title=(
                        f"Value in reported units ({currency})"
                        if selected_metric != "EPS"
                        else f"EPS ({currency} per share)"
                    ),
                )

                st.plotly_chart(
                    metric_chart,
                    use_container_width=True,
                )
            else:
                st.info("No chartable financial metrics are available.")

            st.markdown("**Year-over-year growth**")

            growth_columns = [
                column
                for column in trends.columns
                if column.endswith("YoY Growth")
                and trends[column].notna().any()
            ]

            if growth_columns:
                growth_display = (
                    trends[growth_columns] * 100
                ).copy()
                growth_display.index = growth_display.index.strftime(
                    "%Y-%m-%d"
                )
                growth_display.index.name = "Reporting Date"

                st.dataframe(
                    growth_display.style.format("{:.2f}%"),
                    use_container_width=True,
                )
            else:
                st.info(
                    "Comparable annual growth rates are not available."
                )

            st.markdown("**Data-based observations**")

            for observation in summarize_financial_trends(trends):
                st.write(f"- {observation}")

            st.caption(
                "Financial statement values are shown in the provider's "
                "reported units. Growth rates require comparable annual "
                "observations. These historical trends are not forecasts."
            )

    elif ticker_input and loaded_ticker != ticker_input:
        st.info("Click **Load Fundamentals** to retrieve this company.")

if selected_page == "AI Research Agent":
    st.header("AI Investment Research Agent")

    st.write(
        "Generate evidence-based stock research using historical "
        "market data, quantitative risk metrics, company fundamentals, "
        "and financial trends."
    )

    st.info(
        "Market data is retrieved from Yahoo Finance and AI analysis "
        "is generated using Groq. Results are for research and "
        "educational purposes, not personalized investment advice."
    )

    with st.form("ai_research_form"):
        ticker = st.text_input(
            "Stock ticker",
            value="RELIANCE.NS",
            help=(
                "Examples: RELIANCE.NS or TCS.NS for Indian stocks; "
                "NVDA or AAPL for US stocks."
            ),
        )

        period = st.selectbox(
            "Historical data period",
            options=["6mo", "1y", "2y", "5y"],
            index=1,
        )

        question = st.text_area(
            "Research question",
            value=(
                "Analyze the company's historical risk and return, "
                "valuation metrics, profitability, and financial trends. "
                "Identify evidence-supported strengths, risks, and "
                "limitations. Do not make a buy or sell recommendation. "
                "Do not make unsupported industry comparisons."
            ),
            height=140,
        )

        run_research = st.form_submit_button(
            "Generate Research Report",
            type="primary",
            use_container_width=True,
        )

    if run_research:
        if not ticker.strip():
            st.warning("Please enter a valid stock ticker.")
        elif not question.strip():
            st.warning("Please enter a research question.")
        else:
            with st.spinner(
                "Retrieving market data and generating your research report..."
            ):
                try:
                    report = research_stock(
                        ticker=ticker.strip().upper(),
                        question=question.strip(),
                        period=period,
                    )

                    st.subheader(
                        f"Research Report: {ticker.strip().upper()}"
                    )
                    st.markdown(report)

                    st.divider()
                    st.caption(
                        "Source: Yahoo Finance via yfinance for available "
                        "market and fundamental data; Groq for AI-generated "
                        "analysis. Historical metrics are not forecasts. "
                        "Verify important figures against primary sources."
                    )

                except Exception as exc:
                    st.error(
                        "The research report could not be generated."
                    )
                    st.warning(str(exc))
                    st.caption(
                        "Check the ticker, internet connection, data "
                        "availability, and Groq API configuration. "
                        "Your existing dashboard features should remain "
                        "available."
                    )
