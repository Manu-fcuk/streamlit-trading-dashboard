
import yfinance as yf
import pandas as pd

INDICES = {
    "Nasdaq 100": "^NDX",
    "DAX (Germany)": "^GDAXI",
    "Dow Jones Industrial Average": "^DJI",
    "Hang Seng (Hong Kong)": "^HSI",
    "SSE Composite (China)": "000001.SS"
}

def test_fetch():
    print("Starting yfinance test...")
    for name, ticker in INDICES.items():
        print(f"\nTesting {name} ({ticker})...")
        try:
            # Try fetching standard data
            data = yf.download(ticker, period="1d", interval="1m", progress=False, prepost=True)
            if data.empty:
                print(f"FAILED: {name} - Returned empty DataFrame.")
            else:
                print(f"SUCCESS: {name} - Returned {len(data)} rows.")
                print("Columns:", data.columns)
                print(data.tail(3))
        except Exception as e:
            print(f"ERROR: {name} - Exception: {e}")

    # Test DIA as alternative for DJI
    print("\nTesting DIA (Dow ETF)...")
    try:
        data = yf.download("DIA", period="1d", interval="1m", progress=False, prepost=True)
        if not data.empty:
            print(f"SUCCESS: DIA - Returned {len(data)} rows.")
            print("Columns:", data.columns)
    except:
        pass

if __name__ == "__main__":
    test_fetch()
