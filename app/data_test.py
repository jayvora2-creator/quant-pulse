import yfinance as yf

# Indian stock
india = yf.download(
    "RELIANCE.NS",
    period="5d",
    interval="1d",
    auto_adjust=False
)

# US stock
us = yf.download(
    "NVDA",
    period="5d",
    interval="1d",
    auto_adjust=False
)

print("\n===== RELIANCE (INDIA) =====")
print(india)

print("\n===== NVIDIA (USA) =====")
print(us)