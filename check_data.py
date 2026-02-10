
import yfinance as yf
import pandas as pd

indices = {
    "Nasdaq 100": "^NDX",
    "Dow Jones": "^DJI",
    "DAX": "^GDAXI"
}

intervals = ["1m", "5m", "15m", "1h", "1d"]

print(f"{'Index':<15} | {'Interval':<10} | {'Status':<10} | {'Rows':<5}")
print("-" * 50)

for name, ticker in indices.items():
    for interval in intervals:
        period = "5d" if interval == "1m" else "1mo"
        if interval == "1d": period = "1y"
        
        try:
            df = yf.download(ticker, interval=interval, period=period, progress=False)
            status = "OK" if not df.empty else "EMPTY"
            rows = len(df) if not df.empty else 0
            print(f"{name:<15} | {interval:<10} | {status:<10} | {rows:<5}")
        except Exception as e:
            print(f"{name:<15} | {interval:<10} | ERROR      | 0")
