
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
        # st.write(f"Debug: Downloading {ticker} p={period} i={interval}")
        data = yf.download(ticker, period=period, interval=interval, progress=False, prepost=True, auto_adjust=False)
        
        if data.empty:
            return None
            
        # Robust MultiIndex Flattening
        # yfinance v0.2+ returns MultiIndex columns (Price, Ticker) by default
        if isinstance(data.columns, pd.MultiIndex):
            # We want to drop the Ticker level
            try:
                data.columns = data.columns.droplevel(1)
            except:
                pass
            
            # If that failed or wasn't enough, try explicit selection
            if isinstance(data.columns, pd.MultiIndex):
                if 'Close' in data.columns.get_level_values(0):
                     data.columns = data.columns.get_level_values(0)
                elif 'Close' in data.columns.get_level_values(1):
                     data.columns = data.columns.get_level_values(1)
            
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in data.columns for col in required_cols):
            # st.error(f"Missing columns: {data.columns.tolist()}")
            return None
            
        return data
    except Exception as e:
        # st.error(f"Ex: {e}")
        return None

# 3. Strategy Calculation
def calculate_indicators(df):
    if df is None:
        return df
    
    # Copy to avoid SettingWithCopy warnings    
    df = df.copy()

    # Ensure numerical columns
    cols = ['Open', 'High', 'Low', 'Close']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    df.dropna(subset=cols, inplace=True)
    
    if len(df) < 50:
        return df
    
    # RSI
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # MACD
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    if macd is not None:
        df = pd.concat([df, macd], axis=1)
        cols = df.columns
        try:
            # Flexible column finding for MACD
            # pandas_ta columns usually: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
            # But sometimes names differ based on version
            macd_col = [c for c in cols if str(c).startswith('MACD_') and 'h' not in str(c) and 's' not in str(c)][0]
            signal_col = [c for c in cols if str(c).startswith('MACDs_')][0]
            hist_col = [c for c in cols if str(c).startswith('MACDh_')][0]
            
            df['MACD'] = df[macd_col]
            df['MACD_Signal'] = df[signal_col]
            df['MACD_Hist'] = df[hist_col]
        except IndexError:
            pass

    # Moving Averages
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    
    # --- Generate Historical Signals for Plotting ---
    df['Signal_Point'] = 0 
    
    # Shift
    df['MACD_Prev'] = df['MACD'].shift(1)
    df['Signal_Prev'] = df['MACD_Signal'].shift(1)
    
    # Logic
    # Fill NA to avoid comparison errors
    df['MACD'] = df['MACD'].fillna(0)
    df['MACD_Signal'] = df['MACD_Signal'].fillna(0)
    df['MACD_Prev'] = df['MACD_Prev'].fillna(0)
    df['Signal_Prev'] = df['Signal_Prev'].fillna(0)
    df['EMA_200'] = df['EMA_200'].fillna(0)
    df['RSI'] = df['RSI'].fillna(50) 
    
    bullish_cross = (df['MACD_Prev'] <= df['Signal_Prev']) & (df['MACD'] > df['MACD_Signal'])
    bearish_cross = (df['MACD_Prev'] >= df['Signal_Prev']) & (df['MACD'] < df['MACD_Signal'])
    
    uptrend = df['Close'] > df['EMA_200']
    downtrend = df['Close'] < df['EMA_200']
    
    # Only generate signals where EMA_200 is valid (after 200 bars)
    valid_data = df['EMA_200'] != 0
    
    df.loc[valid_data & uptrend & bullish_cross & (df['RSI'] < 70), 'Signal_Point'] = 1
    df.loc[valid_data & downtrend & bearish_cross & (df['RSI'] > 30), 'Signal_Point'] = -1

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

# Strategy Description Helper
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
yf_interval = timeframe
if timeframe == "4h":
    yf_interval = "1h" 
fetch_period = period_map.get(timeframe, "1mo")

# 2. Fetch Data Function
def calculate_indicators(df):
    if df is None: return df
    df = df.copy()
    
    # Ensure numerical
    cols = ['Open', 'High', 'Low', 'Close']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=cols, inplace=True)
    if len(df) < 50: return df
    
    # --- Common Indicators ---
    # RSI
    df['RSI'] = ta.rsi(df['Close'], length=14)
    # MACD
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    if macd is not None:
        df = pd.concat([df, macd], axis=1)
        # Rename columns standardly
        df.rename(columns={df.columns[-3]: 'MACD', df.columns[-2]: 'MACD_Hist', df.columns[-1]: 'MACD_Signal'}, inplace=True)
            
    # EMAs
    df['EMA_9'] = ta.ema(df['Close'], length=9)
    df['EMA_21'] = ta.ema(df['Close'], length=21)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    
    # Bollinger Bands
    bb = ta.bbands(df['Close'], length=20, std=2)
    if bb is not None:
        df = pd.concat([df, bb], axis=1)
        cols = df.columns
        try:
             df['BB_Lower'] = df[[c for c in cols if c.startswith('BBL')][0]]
             df['BB_Upper'] = df[[c for c in cols if c.startswith('BBU')][0]]
        except: pass
        
    return df

def apply_strategy(df, strategy_name):
    if df is None or len(df) < 50: return df
    
    df['Signal_Point'] = 0
    
    # 1. Momentum Strategy (Original)
    if "Momentum" in strategy_name:
        # Fill only indicators for cross logic
        macd = df['MACD'].fillna(0)
        signal = df['MACD_Signal'].fillna(0)
        
        macd_cross_up = (macd.shift(1) <= signal.shift(1)) & (macd > signal)
        macd_cross_down = (macd.shift(1) >= signal.shift(1)) & (macd < signal)
        
        uptrend = df['Close'] > df['EMA_200']
        downtrend = df['Close'] < df['EMA_200']
        
        df.loc[uptrend & macd_cross_up & (df['RSI'] < 70), 'Signal_Point'] = 1
        df.loc[downtrend & macd_cross_down & (df['RSI'] > 30), 'Signal_Point'] = -1
        
    # 2. Mean Reversion (Bollinger)
    elif "Mean Reversion" in strategy_name:
        if 'BB_Lower' in df.columns and 'BB_Upper' in df.columns:
            buy_cond = (df['Close'] < df['BB_Lower']) & (df['RSI'] < 30)
            sell_cond = (df['Close'] > df['BB_Upper']) & (df['RSI'] > 70)
            df.loc[buy_cond, 'Signal_Point'] = 1
            df.loc[sell_cond, 'Signal_Point'] = -1
        
    # 3. Scalping (EMA Cross)
    elif "Scalping" in strategy_name:
        e9 = df['EMA_9'].fillna(0)
        e21 = df['EMA_21'].fillna(0)
        cross_up = (e9.shift(1) <= e21.shift(1)) & (e9 > e21)
        cross_down = (e9.shift(1) >= e21.shift(1)) & (e9 < e21)
        df.loc[cross_up, 'Signal_Point'] = 1
        df.loc[cross_down, 'Signal_Point'] = -1

    # 4. ORB (Opening Range Breakout)
    elif "ORB" in strategy_name:
        df['Date_Str'] = df.index.date
        if len(df) > 0:
            for date, day_data in df.groupby('Date_Str'):
                if len(day_data) < 30: continue
                opening_range = day_data.iloc[:30] 
                orb_high = opening_range['High'].max()
                orb_low = opening_range['Low'].min()
                mask_day = df['Date_Str'] == date
                
                if len(day_data) > 30:
                    cutoff_time = day_data.index[29]
                    breakout_up = (df.index > cutoff_time) & (mask_day) & \
                                  (df['Close'] > orb_high) & \
                                  (df['Close'].shift(1) <= orb_high)
                    breakout_down = (df.index > cutoff_time) & (mask_day) & \
                                    (df['Close'] < orb_low) & \
                                    (df['Close'].shift(1) >= orb_low)
                    
                    df.loc[breakout_up, 'Signal_Point'] = 1
                    df.loc[breakout_down, 'Signal_Point'] = -1
                    df.loc[mask_day, 'ORB_High'] = orb_high
                    df.loc[mask_day, 'ORB_Low'] = orb_low
        
    return df

# ... generate_current_signal
# 4. Visualization (Updated for strategies)
def plot_chart(df, ticker_name, strategy_name, timeframe):
    if df is None: return
    
    # Ensure datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        try: df.index = pd.to_datetime(df.index)
        except: pass

    # Default to last 300 bars
    df_plot = df.tail(300).copy()

    # Special handling for short timeframes (1m, 5m) to show clearly the current session
    if timeframe in ["1m", "5m"]:
         # Get the date of the last available data point
         last_date = df.index[-1].date()
         # Filter to show only data from that last date (Today's session)
         df_today = df[df.index.date == last_date]
         
         # If "Today" has very few bars (e.g. market just opened), we might want to show previous day too?
         # User request: "I dont see the chart clearly ... because also previous days are shown"
         # So we strictly honor showing ONLY the current day if possible.
         if len(df_today) > 0:
             df_plot = df_today.copy()
    
    # Dynamic Subplots based on Strategy
    rows = 2
    heights = [0.7, 0.3]
    titles = (f"{ticker_name} Price", "RSI")
    
    if "Momentum" in strategy_name:
        rows = 3
        heights = [0.6, 0.2, 0.2]
        titles = (f"{ticker_name} Price & EMA 200", "RSI", "MACD")
    elif "Scalping" in strategy_name:
         titles = (f"{ticker_name} Price & EMAs", "RSI")
    elif "ORB" in strategy_name:
         titles = (f"{ticker_name} Price & Opening Range", "RSI")
         
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=heights, subplot_titles=titles)

    # 1. Price Chart
    fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name='Price'), row=1, col=1)

    # Overlays
    if "Momentum" in strategy_name:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['EMA_200'], line=dict(color='blue', width=2), name='EMA 200'), row=1, col=1)
    elif "Mean Reversion" in strategy_name:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['BB_Upper'], line=dict(color='gray', width=1, dash='dash'), name='Upper BB'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['BB_Lower'], line=dict(color='gray', width=1, dash='dash'), name='Lower BB'), row=1, col=1)
    elif "Scalping" in strategy_name:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['EMA_9'], line=dict(color='green', width=1), name='EMA 9'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['EMA_21'], line=dict(color='red', width=1), name='EMA 21'), row=1, col=1)
    elif "ORB" in strategy_name and 'ORB_High' in df_plot.columns:
        # Plot ORB Levels for the most recent day in the view
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['ORB_High'], line=dict(color='orange', width=1, dash='dash'), name='ORB High'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['ORB_Low'], line=dict(color='orange', width=1, dash='dash'), name='ORB Low'), row=1, col=1)

    # Signals
    buys = df_plot[df_plot['Signal_Point'] == 1]
    sells = df_plot[df_plot['Signal_Point'] == -1]
    if not buys.empty:
        fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.9995, mode='markers', marker=dict(symbol='triangle-up', size=14, color='#00ff00'), name='BUY'), row=1, col=1)
    if not sells.empty:
        fig.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.0005, mode='markers', marker=dict(symbol='triangle-down', size=14, color='#ff0000'), name='SELL'), row=1, col=1)

    # 2. RSI (Common)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], line=dict(color='purple', width=2), name='RSI'), row=2, col=1)
    fig.add_shape(type="line", x0=df_plot.index[0], x1=df_plot.index[-1], y0=70, y1=70, line=dict(color="red", width=1, dash="dash"), row=2, col=1)
    fig.add_shape(type="line", x0=df_plot.index[0], x1=df_plot.index[-1], y0=30, y1=30, line=dict(color="green", width=1, dash="dash"), row=2, col=1)

    # 3. MACD (Only Momentum)
    if "Momentum" in strategy_name:
        colors = ['green' if v >= 0 else 'red' for v in df_plot['MACD_Hist']]
        fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['MACD_Hist'], marker_color=colors, name='Hist'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MACD'], line=dict(color='blue'), name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MACD_Signal'], line=dict(color='orange'), name='Sig'), row=3, col=1)

    fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

# Main Logic for Selected Index
if selected_ticker:
    with st.spinner(f"Fetching data for {selected_index_name} ({timeframe})..."):
        df = get_data(selected_ticker, period=fetch_period, interval=yf_interval)

    if df is not None:
        # Resample to 4H if requested
        if timeframe == "4h":
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Resample logic
            agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            # Drop columns not in agg_dict to avoid errors
            try:
                df = df[list(agg_dict.keys())].resample('4h').agg(agg_dict).dropna()
            except:
                pass

        # 1. Calculate Indicators
        df = calculate_indicators(df)
        
        # 2. Apply Selected Strategy
        df = apply_strategy(df, strategy_type)
        
        # Generate Signal Text
        signal, reason = generate_current_signal(df)
        last_price = df['Close'].iloc[-1]
        
        # Header Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"{last_price:.2f}")
        
        signal_color = "gray"
        if signal == "BUY": signal_color = "green"
        elif signal == "SELL": signal_color = "red"
        
        col2.markdown(f"**Signal:** <span style='color:{signal_color}; font-size: 24px;'>{signal}</span>", unsafe_allow_html=True)
        col3.write(f"**Reason:** {reason}")
        
        # Charts
        st.subheader(f"Chart Analysis ({timeframe})")
        plot_chart(df, selected_index_name, strategy_type, timeframe)
        
        # Recent Data Table
        with st.expander("View Recent Data"):
            st.dataframe(df.tail(10)[['Open', 'High', 'Low', 'Close', 'RSI', 'Signal_Point']].style.format("{:.2f}"))

    else:
        st.error(f"No data found for {selected_index_name} ({selected_ticker}).")
        st.info("Troubleshooting:")
        st.write("1. **Market Hours**: Ensure the market is currently open.")
        st.write("2. **Data Source**: Yahoo Finance may briefly lack data for this interval.")

# Auto-Refresh Logic (Placed at end to avoid blocking render)
if auto_refresh:
    time.sleep(30)
    st.rerun()




