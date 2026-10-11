from app.data_provider import get_stock_data


india = get_stock_data("RELIANCE.NS", period="1mo")
us = get_stock_data("NVDA", period="1mo")

print("\n===== INDIA: RELIANCE =====")
print(india.tail())

print("\n===== USA: NVIDIA =====")
print(us.tail())