
import yfinance as yf
import pandas as pd
import pandas_ta as ta

# Mocking the app's get_data function
def get_data(ticker, period="5d", interval="1m"):
    print(f"Fetching {ticker} (period={period}, interval={interval})...")
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, prepost=True)
        
        if data.empty:
            print(" -> Empty DataFrame")
            return None
            
        if isinstance(data.columns, pd.MultiIndex):
            print(" -> Flattening MultiIndex columns")
            data.columns = data.columns.get_level_values(0)
            
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in data.columns for col in required_cols):
            print(f" -> Missing columns. Found: {data.columns.tolist()}")
            return None
            
        print(f" -> OK. {len(data)} rows.")
        return data
    except Exception as e:
        print(f" -> Exception: {e}")
        return None

# Test logic
indices = {
    "Nasdaq 100": "^NDX",
    "Dow Jones": "^DJI"
}

period_map = {
    "1m": "5d",
    "5m": "1mo",
    "15m": "1mo",
    "4h": "1y"
}

for name, ticker in indices.items():
    # Test 1m
    get_data(ticker, period="5d", interval="1m")
    
    # Test 4h logic (fetch 1h)
    df_1h = get_data(ticker, period="1y", interval="1h")
    if df_1h is not None:
        print(" -> Mocking 4h resampling...")
        agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
        try:
            # Ensure Volume exists
            if 'Volume' not in df_1h.columns:
                 df_1h['Volume'] = 0
            
            df_4h = df_1h[list(agg_dict.keys())].resample('4h').agg(agg_dict).dropna()
            print(f" -> Resampled to 4h: {len(df_4h)} rows.")
        except Exception as e:
            print(f" -> Resample Error: {e}")
