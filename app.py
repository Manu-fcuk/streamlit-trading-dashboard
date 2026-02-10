
import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="Global Indices Trading Signals", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .metric-container {
        padding: 10px;
        border-radius: 5px;
        background-color: #0e1117;
        margin-bottom: 10px;
        border: 1px solid #30333d;
    }
    .signal-buy {
        color: #00ff00;
        font-weight: bold;
        font-size: 24px;
    }
    .signal-sell {
        color: #ff0000;
        font-weight: bold;
        font-size: 24px;
    }
    .signal-neutral {
        color: #888888;
        font-weight: bold;
        font-size: 24px;
    }
</style>
""", unsafe_allow_html=True)

# 1. Define Tickers
INDICES = {
    "Nasdaq 100": "^NDX",
    "DAX (Germany)": "^GDAXI",
    "Dow Jones Industrial Average": "^DJI",
    "Hang Seng (Hong Kong)": "^HSI",
    "SSE Composite (China)": "000001.SS"
}

# 2. Fetch Data Function
def get_data(ticker, period="5d", interval="1m"):
    try:
        # Download data
        data = yf.download(ticker, period=period, interval=interval, progress=False, prepost=True, auto_adjust=False)
        
        if data.empty:
            return None
            
        # VERY ROBUST MultiIndex Flattening for yfinance v0.2.x
        if isinstance(data.columns, pd.MultiIndex):
            # If for some reason it's a MultiIndex, we only care about the level that has OHLC
            # Often it's (Level 0: Price, Level 1: Ticker)
            # Level 0 usually contains 'Open', 'High', 'Low', 'Close', 'Volume'
            # We will flatten by taking the level that contains 'Close'
            for i in range(data.columns.nlevels):
                if 'Close' in data.columns.get_level_values(i):
                    data.columns = data.columns.get_level_values(i)
                    break
        
        # Select and reorder columns to ensure we have a clean DataFrame
        cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [c for c in cols_to_keep if c in data.columns]
        data = data[available_cols].copy()

        # Ensure numeric and drop any NaNs in price data
        for col in available_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        data.dropna(subset=['Close'], inplace=True)
        return data
    except Exception as e:
        return None

# 3. Strategy Calculation
def calculate_indicators(df):
    if df is None or len(df) == 0: return df
    df = df.copy()

    # Double check numerical
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # RSI
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # MACD
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    if macd is not None:
        # Avoid column name collisions and handle MultiIndex if ta returns it
        macd = macd.astype(float)
        m_col = [c for c in macd.columns if 'MACD_' in str(c) and 'h' not in str(c) and 's' not in str(c)]
        s_col = [c for c in macd.columns if 'MACDs_' in str(c)]
        h_col = [c for c in macd.columns if 'MACDh_' in str(c)]
        
        if m_col: df['MACD'] = macd[m_col[0]].values
        if s_col: df['MACD_Signal'] = macd[s_col[0]].values
        if h_col: df['MACD_Hist'] = macd[h_col[0]].values

    # EMAs
    df['EMA_9'] = ta.ema(df['Close'], length=9)
    df['EMA_21'] = ta.ema(df['Close'], length=21)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    
    # Bollinger Bands
    bb = ta.bbands(df['Close'], length=20, std=2)
    if bb is not None:
        l_col = [c for c in bb.columns if c.startswith('BBL')]
        u_col = [c for c in bb.columns if c.startswith('BBU')]
        if l_col: df['BB_Lower'] = bb[l_col[0]].values
        if u_col: df['BB_Upper'] = bb[u_col[0]].values

    return df

def apply_strategy(df, strategy_name):
    if df is None or len(df) < 5: return df
    
    df['Signal_Point'] = 0.0
    
    # 1. Momentum Strategy
    if "Momentum" in strategy_name and 'MACD' in df.columns:
        m = df['MACD'].fillna(0)
        s = df['MACD_Signal'].fillna(0)
        cross_up = (m.shift(1) <= s.shift(1)) & (m > s)
        cross_down = (m.shift(1) >= s.shift(1)) & (m < s)
        
        ema200 = df['EMA_200'].fillna(method='ffill')
        uptrend = df['Close'] > ema200
        downtrend = df['Close'] < ema200
        
        df.loc[uptrend & cross_up & (df['RSI'] < 70), 'Signal_Point'] = 1.0
        df.loc[downtrend & cross_down & (df['RSI'] > 30), 'Signal_Point'] = -1.0
        
    # 2. Mean Reversion
    elif "Mean Reversion" in strategy_name and 'BB_Lower' in df.columns:
        buy_cond = (df['Close'] < df['BB_Lower']) & (df['RSI'] < 35)
        sell_cond = (df['Close'] > df['BB_Upper']) & (df['RSI'] > 65)
        df.loc[buy_cond, 'Signal_Point'] = 1.0
        df.loc[sell_cond, 'Signal_Point'] = -1.0
        
    # 3. Scalping
    elif "Scalping" in strategy_name and 'EMA_9' in df.columns:
        e9 = df['EMA_9'].fillna(0)
        e21 = df['EMA_21'].fillna(0)
        cross_up = (e9.shift(1) <= e21.shift(1)) & (e9 > e21)
        cross_down = (e9.shift(1) >= e21.shift(1)) & (e9 < e21)
        df.loc[cross_up, 'Signal_Point'] = 1.0
        df.loc[cross_down, 'Signal_Point'] = -1.0

    # 4. ORB
    elif "ORB" in strategy_name:
        df['Date_Str'] = df.index.date
        for date, day_data in df.groupby('Date_Str'):
            needed = 30 
            if len(day_data) < needed: continue
            
            opening_range = day_data.iloc[:needed]
            o_high = float(opening_range['High'].max())
            o_low = float(opening_range['Low'].min())
            
            mask_day = df['Date_Str'] == date
            cutoff = day_data.index[needed-1]
            
            break_up = (df.index > cutoff) & (mask_day) & (df['Close'] > o_high) & (df['Close'].shift(1) <= o_high)
            break_down = (df.index > cutoff) & (mask_day) & (df['Close'] < o_low) & (df['Close'].shift(1) >= o_low)
            
            df.loc[break_up, 'Signal_Point'] = 1.0
            df.loc[break_down, 'Signal_Point'] = -1.0
            df.loc[mask_day, 'ORB_High'] = o_high
            df.loc[mask_day, 'ORB_Low'] = o_low
            
    return df

def generate_current_signal(df):
    if df is None or len(df) < 50:
        return "NEUTRAL", "Insufficient Data"
    
    temp = df[df['Signal_Point'] != 0]
    if temp.empty:
         return "NEUTRAL", "No signals in loaded history"
         
    # Last signal
    last_idx = temp.index[-1]
    signal_val = temp.loc[last_idx, 'Signal_Point']
    
    # Calculate bars ago
    try:
        # Find integer location of last_idx
        if not df.index.is_unique:
             # If duplicate indices, take the last one
             loc = df.index.get_loc(last_idx)
             if isinstance(loc, slice):
                 loc = loc.stop - 1
             elif isinstance(loc, np.ndarray):
                 loc = np.where(loc)[0][-1]
        else:
             loc = df.index.get_loc(last_idx)
             
        bars_ago = len(df) - loc - 1
    except:
        bars_ago = "?"
    
    signal_str = "BUY" if signal_val == 1 else "SELL"
    reason = f"Signal generated {bars_ago} bars ago ({last_idx})"
    
    return signal_str, reason

# 4. Visualization
def plot_chart(df, ticker_name):
    if df is None:
        st.write("No data frame to plot.")
        return
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except:
            pass

    # Filter
    df_plot = df.tail(300).copy()
    
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f"{ticker_name} Price & EMA", "RSI (14)", "MACD")
    )

    # 1. Price
    fig.add_trace(go.Candlestick(
        x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'],
        name='Price'
    ), row=1, col=1)
    
    # EMAs
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['EMA_50'], line=dict(color='#FFA500', width=1), name='EMA 50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['EMA_200'], line=dict(color='#0000FF', width=2), name='EMA 200'), row=1, col=1)

    # Signals
    buys = df_plot[df_plot['Signal_Point'] == 1]
    sells = df_plot[df_plot['Signal_Point'] == -1]
    
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys.index, y=buys['Low']*0.9995, mode='markers', 
            marker=dict(symbol='triangle-up', size=12, color='green'), name='BUY'
        ), row=1, col=1)
    
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells.index, y=sells['High']*1.0005, mode='markers', 
            marker=dict(symbol='triangle-down', size=12, color='red'), name='SELL'
        ), row=1, col=1)

    # 2. RSI
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], line=dict(color='#A020F0', width=2), name='RSI'), row=2, col=1)
    fig.add_shape(type="line", x0=df_plot.index[0], x1=df_plot.index[-1], y0=70, y1=70, line=dict(color="red", width=1, dash="dash"), row=2, col=1)
    fig.add_shape(type="line", x0=df_plot.index[0], x1=df_plot.index[-1], y0=30, y1=30, line=dict(color="green", width=1, dash="dash"), row=2, col=1)

    # 3. MACD
    # Fillna for colors to avoid error
    hist_vals = df_plot['MACD_Hist'].fillna(0)
    hist_colors = ['green' if v >= 0 else 'red' for v in hist_vals]
    
    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['MACD_Hist'], name='Hist', marker_color=hist_colors), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MACD'], line=dict(color='#0000FF', width=1), name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MACD_Signal'], line=dict(color='#FFA500', width=1), name='Signal'), row=3, col=1)

    fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    return


# 5. Main App Layout
# 5. Main App Layout
st.sidebar.title("Configuration")

# Sidebar - Index Selection
selected_index_name = st.sidebar.radio("Select Active Index:", list(INDICES.keys()), index=1)
selected_ticker = INDICES[selected_index_name]

# Sidebar - Strategy Selection
strategy_type = st.sidebar.radio("Select Strategy:", [
    "Momentum (EMA + MACD + RSI)", 
    "Mean Reversion (Bollinger + RSI)", 
    "Scalping (EMA 9/21 Crossover)",
    "ORB (Opening Range Breakout)"
], index=0)

# Sidebar - Timeframe Selection
timeframe = st.sidebar.selectbox("Select Timeframe:", ["1m", "5m", "15m", "1h", "4h", "1d"], index=0)

auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=True)

# 2. Fetch Data Function
def get_data(ticker, period="5d", interval="1m"):
    try:
        # We use a very clean download approach
        temp = yf.download(ticker, period=period, interval=interval, progress=False, prepost=True, auto_adjust=True)
        if temp.empty: return None

        # Flatten any MultiIndex immediately and explicitly
        if isinstance(temp.columns, pd.MultiIndex):
            # Typically (Price, Ticker) or (Ticker, Price)
            # Find the index of the level that says 'Close'
            for i in range(temp.columns.nlevels):
                if 'Close' in temp.columns.get_level_values(i):
                    temp.columns = temp.columns.get_level_values(i)
                    break
        
        # Select only core columns to start fresh
        df = pd.DataFrame(index=temp.index)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in temp.columns:
                series = temp[col]
                # If it's a dataframe, take the first column
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
                df[col] = pd.to_numeric(series, errors='coerce').astype(float)
        
        # Deduplicate column names if any and drop NaNs
        df = df.T.drop_duplicates().T
        df.dropna(subset=['Close'], inplace=True)
        df = df[df['Close'] > 0]
        
        return df
    except Exception:
        return None

# 3. Strategy Calculation
def calculate_indicators(df):
    if df is None or len(df) == 0: return df
    df = df.copy()

    # technical indicators (strictly as floats)
    try:
        df['RSI'] = ta.rsi(df['Close'], length=14).astype(float)
        
        # MACD
        macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        if macd is not None:
            # Rename columns to standard names
            for col in macd.columns:
                if 'MACD_' in col and 'h' not in col and 's' not in col: df['MACD'] = macd[col].astype(float)
                if 'MACDs_' in col: df['MACD_Signal'] = macd[col].astype(float)
                if 'MACDh_' in col: df['MACD_Hist'] = macd[col].astype(float)

        # EMAs
        df['EMA_9'] = ta.ema(df['Close'], length=9).astype(float)
        df['EMA_21'] = ta.ema(df['Close'], length=21).astype(float)
        df['EMA_50'] = ta.ema(df['Close'], length=50).astype(float)
        df['EMA_200'] = ta.ema(df['Close'], length=200).astype(float)
        
        # Bollinger
        bb = ta.bbands(df['Close'], length=20, std=2)
        if bb is not None:
            for col in bb.columns:
                if col.startswith('BBL'): df['BB_Lower'] = bb[col].astype(float)
                if col.startswith('BBU'): df['BB_Upper'] = bb[col].astype(float)
    except:
        pass

    return df

def apply_strategy(df, strategy_name):
    if df is None or len(df) < 5: return df
    df['Signal_Point'] = 0.0
    
    # Use fillna(0) locally for indicator logic ONLY to prevent breaks
    # but don't fill the price columns
    
    # 1. Momentum
    if "Momentum" in strategy_name and 'MACD' in df.columns:
        m = df['MACD'].fillna(0)
        s = df['MACD_Signal'].fillna(0)
        cross_up = (m.shift(1) <= s.shift(1)) & (m > s)
        cross_down = (m.shift(1) >= s.shift(1)) & (m < s)
        
        if 'EMA_200' in df.columns:
            ema200 = df['EMA_200'].fillna(method='ffill')
            uptrend = df['Close'] > ema200
            downtrend = df['Close'] < ema200
            df.loc[uptrend & cross_up & (df['RSI'].fillna(50) < 70), 'Signal_Point'] = 1.0
            df.loc[downtrend & cross_down & (df['RSI'].fillna(50) > 30), 'Signal_Point'] = -1.0
            
    # 2. Mean Reversion
    elif "Mean Reversion" in strategy_name and 'BB_Lower' in df.columns:
        buy_cond = (df['Close'] < df['BB_Lower']) & (df['RSI'].fillna(50) < 35)
        sell_cond = (df['Close'] > df['BB_Upper']) & (df['RSI'].fillna(50) > 65)
        df.loc[buy_cond, 'Signal_Point'] = 1.0
        df.loc[sell_cond, 'Signal_Point'] = -1.0
        
    # 3. Scalping
    elif "Scalping" in strategy_name and 'EMA_9' in df.columns:
        e9 = df['EMA_9'].fillna(0)
        e21 = df['EMA_21'].fillna(0)
        cross_up = (e9.shift(1) <= e21.shift(1)) & (e9 > e21)
        cross_down = (e9.shift(1) >= e21.shift(1)) & (e9 < e21)
        df.loc[cross_up, 'Signal_Point'] = 1.0
        df.loc[cross_down, 'Signal_Point'] = -1.0

    # 4. ORB
    elif "ORB" in strategy_name:
        df['Date_Str'] = df.index.date
        for date, day_data in df.groupby('Date_Str'):
            if len(day_data) < 30: continue
            opening_range = day_data.iloc[:30]
            oh = float(opening_range['High'].max())
            ol = float(opening_range['Low'].min())
            cutoff = day_data.index[29]
            mask = (df.index.date == date) & (df.index > cutoff)
            bu = mask & (df['Close'] > oh) & (df['Close'].shift(1) <= oh)
            bd = mask & (df['Close'] < ol) & (df['Close'].shift(1) >= ol)
            df.loc[bu, 'Signal_Point'] = 1.0
            df.loc[bd, 'Signal_Point'] = -1.0
            df.loc[df.index.date == date, 'ORB_High'] = oh
            df.loc[df.index.date == date, 'ORB_Low'] = ol
            
    return df

def generate_current_signal(df):
    if df is None or 'Signal_Point' not in df.columns:
        return "NEUTRAL", "No indicators"
    
    recent_signals = df[df['Signal_Point'] != 0.0].tail(1)
    if recent_signals.empty:
        return "NEUTRAL", "No signals in loaded history"
    
    val = recent_signals['Signal_Point'].values[0]
    time = recent_signals.index[0]
    return ("BUY" if val > 0 else "SELL"), f"Triggered at {time.strftime('%H:%M')}"

# 4. Visualization
def plot_chart(df, ticker_name, strategy_name, timeframe):
    if df is None or len(df) < 2: return

    # Filter for session
    if timeframe in ["1m", "5m"]:
        last_date = df.index[-1].date()
        df_p = df[df.index.date == last_date].copy()
    else:
        df_p = df.tail(300).copy()

    if len(df_p) < 2:
        st.info("Market session just starting...")
        return

    # Create subplots
    rows = 2
    heights = [0.7, 0.3]
    titles = ("Price Action", "RSI")
    if "Momentum" in strategy_name:
        rows = 3
        heights = [0.6, 0.2, 0.2]
        titles = ("Price Action", "RSI", "MACD")
    
    # Final categorical index conversion to fix Plotly scaling issues
    df_p.index = df_p.index.strftime('%H:%M')
    
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=heights, subplot_titles=titles)

    # 1. Price candles
    fig.add_trace(go.Candlestick(
        x=df_p.index, open=df_p['Open'], high=df_p['High'],
        low=df_p['Low'], close=df_p['Close'], name='Market'
    ), row=1, col=1)
    
    # Overlays
    if "Momentum" in strategy_name and 'EMA_200' in df_p.columns:
        fig.add_trace(go.Scatter(x=df_p.index, y=df_p['EMA_200'], name='EMA 200', line=dict(color='blue', width=1.5)), row=1, col=1)
    elif "Mean Reversion" in strategy_name and 'BB_Upper' in df_p.columns:
        fig.add_trace(go.Scatter(x=df_p.index, y=df_p['BB_Upper'], name='Upper BB', line=dict(color='rgba(173,216,230,0.5)', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_p.index, y=df_p['BB_Lower'], name='Lower BB', line=dict(color='rgba(173,216,230,0.5)', width=1)), row=1, col=1)
    elif "Scalping" in strategy_name and 'EMA_9' in df_p.columns:
        fig.add_trace(go.Scatter(x=df_p.index, y=df_p['EMA_9'], name='EMA 9', line=dict(color='green', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_p.index, y=df_p['EMA_21'], name='EMA 21', line=dict(color='red', width=1)), row=1, col=1)
    elif "ORB" in strategy_name and 'ORB_High' in df_p.columns:
        fig.add_trace(go.Scatter(x=df_p.index, y=df_p['ORB_High'], name='ORB High', line=dict(color='orange', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_p.index, y=df_p['ORB_Low'], name='ORB Low', line=dict(color='orange', width=1, dash='dash')), row=1, col=1)

    # Signals
    buys = df_p[df_p['Signal_Point'] == 1.0]
    sells = df_p[df_p['Signal_Point'] == -1.0]
    if not buys.empty:
        fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.9997, mode='markers', marker=dict(symbol='triangle-up', size=14, color='#00ff00'), name='BUY'), row=1, col=1)
    if not sells.empty:
        fig.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.0003, mode='markers', marker=dict(symbol='triangle-down', size=14, color='#ff0000'), name='SELL'), row=1, col=1)

    # 2. RSI
    if 'RSI' in df_p.columns:
        fig.add_trace(go.Scatter(x=df_p.index, y=df_p['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line=dict(color="red", width=1, dash="dash"), row=2, col=1)
        fig.add_hline(y=30, line=dict(color="green", width=1, dash="dash"), row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=2, col=1)

    # 3. MACD
    if "Momentum" in strategy_name and 'MACD' in df_p.columns:
        fig.add_trace(go.Bar(x=df_p.index, y=df_p['MACD_Hist'], name='Hist', marker_color='#888888'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_p.index, y=df_p['MACD'], name='MACD', line=dict(color='blue', width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_p.index, y=df_p['MACD_Signal'], name='Signal', line=dict(color='orange', width=1)), row=3, col=1)

    fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
    # Force categorical axis
    fig.update_xaxes(type='category')
    st.plotly_chart(fig, use_container_width=True)

# Main Logic for Selected Index
strategy_info = {
    "Momentum (EMA + MACD + RSI)": "Best for **Trending Markets**. Buys when price is above EMA 200 and MACD crosses up. Sells when price is below EMA 200 and MACD crosses down.",
    "Mean Reversion (Bollinger + RSI)": "Best for **Ranging/Sideways Markets**. Buys when price touches Lower Bollinger Band (Oversold). Sells when price touches Upper Band (Overbought).",
    "Scalping (EMA 9/21 Crossover)": "Best for **Fast Moves**. Buys when Fast EMA (9) crosses above Slow EMA (21). Sells when Fast crosses below Slow.",
    "ORB (Opening Range Breakout)": "Best for **Market Open**. Trades breakouts of the first 30 mins (default) of the active session. **Requires 1m or 5m timeframe.**"
}

st.title(f"📊 {selected_index_name} Trading Dashboard")
st.info(f"**Strategy:** {strategy_type}\n\n{strategy_info[strategy_type]}")

# Determine proper fetch period
period_map = {
    "1m": "5d",
    "5m": "1mo",
    "15m": "1mo",
    "1h": "3mo", 
    "4h": "1y",
    "1d": "2y"
}
fetch_period = period_map.get(timeframe, "1mo")
yf_interval = timeframe
if timeframe == "4h":
    yf_interval = "1h" 

# Main Logic
if selected_ticker:
    with st.spinner(f"Loading {selected_index_name}..."):
        df = get_data(selected_ticker, period=fetch_period, interval=yf_interval)

    if df is not None:
        if timeframe == "4h":
            df = df.resample('4H').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()

        df = calculate_indicators(df)
        df = apply_strategy(df, strategy_type)
        
        # Dashboard Top
        sig, res = generate_current_signal(df)
        m1, m2, m3 = st.columns(3)
        m1.metric("Current Price", f"{df['Close'].iloc[-1]:,.2f}")
        c = "#00ff00" if sig == "BUY" else "#ff0000" if sig == "SELL" else "#cccccc"
        m2.markdown(f"**SIGNAL:** <span style='color:{c}; font-size:26px;'>{sig}</span>", unsafe_allow_html=True)
        m3.info(f"**Logic:** {res}")
        
        # Analysis
        plot_chart(df, selected_index_name, strategy_type, timeframe)
        
        # HTML Table Rendering (Bypasses Arrow completely)
        with st.expander("Debug Raw Data"):
            debug_df = df.tail(10)[['Close', 'RSI', 'Signal_Point']].copy()
            st.write(debug_df.to_html(classes='table table-dark table-striped'), unsafe_allow_html=True)
    else:
        st.error("Data source unavailable.")

# Auto-Refresh (Compatible with older versions)
if auto_refresh:
    time.sleep(30)
    st.experimental_rerun()
