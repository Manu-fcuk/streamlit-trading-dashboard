
import yfinance as yf
import pandas as pd
import pandas_ta as ta

def get_data_debug(ticker, period="5d", interval="1m"):
    print(f"1. Downloading {ticker}...")
    data = yf.download(ticker, period=period, interval=interval, progress=False, prepost=True, auto_adjust=False)
    
    print(f"   Raw shape: {data.shape}")
    print(f"   Columns: {data.columns}")
    
    if data.empty:
        print("   -> EMPTY DATA returned.")
        return None
        
    if isinstance(data.columns, pd.MultiIndex):
        print("   -> MultiIndex detected. Flattening...")
        # debug levels
        print(f"      L0: {data.columns.get_level_values(0).tolist()[:5]}")
        print(f"      L1: {data.columns.get_level_values(1).tolist()[:5]}")
        
        if 'Close' in data.columns.get_level_values(0):
             data.columns = data.columns.get_level_values(0)
        elif 'Close' in data.columns.get_level_values(1):
             data.columns = data.columns.get_level_values(1)
        print(f"   -> Flattened columns: {data.columns.tolist()}")

    required_cols = ['Open', 'High', 'Low', 'Close']
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        print(f"   -> MISSING columns: {missing}")
        return None
        
    return data

def calculate_indicators_debug(df):
    print("2. Calculating Indicators...")
    # Ensure numerical
    cols = ['Open', 'High', 'Low', 'Close']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    before = len(df)
    df.dropna(subset=cols, inplace=True)
    after = len(df)
    print(f"   -> Dropna: {before} -> {after} rows")
    
    if len(df) < 50:
        print("   -> Too few rows for indicators.")
        return df
    
    # RSI
    df['RSI'] = ta.rsi(df['Close'], length=14)
    print(f"   -> RSI calculated. NaNs: {df['RSI'].isna().sum()}")
    
    # MACD
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    if macd is not None:
        df = pd.concat([df, macd], axis=1)
        print(f"   -> MACD columns added: {macd.columns.tolist()}")

    return df

print("--- DEBUGGING NASDAQ 100 (^NDX) ---")
df = get_data_debug("^NDX", period="5d", interval="1m")
if df is not None:
    df = calculate_indicators_debug(df)
    print("3. Final Data Head:")
    print(df[['Close', 'RSI']].tail())
else:
    print("DATA FETCH FAILED.")
