import time
import pandas as pd
import pandas_ta as ta
from etoro_client import EtoroClient
import config
from datetime import datetime

# Global configuration
TIMEFRAME = '15m'
TRADE_AMOUNT = 100
LEVERAGE = 20
SCAN_INTERVAL_SECONDS = 15 * 60  # 15 minutes

def analyze_market(df):
    """
    Applies MACD and RSI strategy.
    
    Strategy Logic:
    1. RSI (14)
    2. MACD (12, 26, 9)
    
    Signal:
    - BUY:  MACD > Signal AND RSI < 70 (Momentum Up)
    - SELL: MACD < Signal OR  RSI > 70 (Momentum Down or Overbought)
    
    Returns: 'buy', 'sell', or None
    """
    if df is None or len(df) < 50:
        return None
        
    # Calculate Indicators
    # RSI
    df['rsi'] = ta.rsi(df['close'], length=14)
    
    # MACD
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    # Using default column names from pandas_ta: MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
    # We rename or access generic names if possible, but pandas_ta is explicit.
    # Let's verify columns or just assign standard names
    df = pd.concat([df, macd], axis=1)
    
    # Identify the specific column names pandas_ta generated
    macd_col = 'MACD_12_26_9'
    signal_col = 'MACDs_12_26_9'
    
    if macd_col not in df.columns:
        # Fallback if names differ
        cols = [c for c in df.columns if c.startswith('MACD')]
        if len(cols) >= 2:
            macd_col = cols[0]
            signal_col = cols[1]

    # Get latest candle values
    last_row = df.iloc[-1]
    last_rsi = last_row['rsi']
    last_macd = last_row[macd_col]
    last_signal = last_row[signal_col]
    
    # Previous candle for crossover check
    prev_row = df.iloc[-2]
    prev_macd = prev_row[macd_col]
    prev_signal = prev_row[signal_col]
    
    # Crossover Logic
    bullish_cross = (prev_macd < prev_signal) and (last_macd > last_signal)
    bearish_cross = (prev_macd > prev_signal) and (last_macd < last_signal)
    
    # Decisions
    if bullish_cross and last_rsi < 70:
        return 'buy'
    elif bearish_cross or last_rsi > 70:
        return 'sell'
        
    return None

def run_bot_cycle(client):
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Scan Cycle...")
    
    try:
        indices = client.get_indices()
    except Exception as e:
        print(f"Error fetching indices: {e}")
        return

    print(f" -> Found {len(indices)} indices.")
    
    for idx, index in enumerate(indices):
        name = index.get('InstrumentDisplayName')
        inst_id = index.get('InstrumentID')
        symbol = index.get('SymbolFull')
        
        # Optimize: Don't print every single scan to keep logs clean, only significant ones
        # print(f"Scanning {symbol}...")  
        
        # 1. Get Data
        df = client.get_candles(inst_id, period=TIMEFRAME, count=100)
        
        # 2. Analyze
        signal = analyze_market(df)
        
        # 3. Execute
        if signal:
            print(f" >> SIGNAL: {signal.upper()} on {symbol}")
            client.place_order(inst_id, side=signal, amount=TRADE_AMOUNT, leverage=LEVERAGE)
        else:
             # Just a log to show aliveness for first few items
             if idx < 3: 
                 print(f"    {symbol}: No signal.")

    print("Cycle complete.")

def main():
    print("---------------------------------------")
    print(f" eToro Bot | Interval: {TIMEFRAME} | Strategy: MACD+RSI")
    print("---------------------------------------")
    
    client = EtoroClient(config.API_KEY)
    
    while True:
        run_bot_cycle(client)
        
        print(f"Sleeping for {SCAN_INTERVAL_SECONDS} seconds...")
        time.sleep(SCAN_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
