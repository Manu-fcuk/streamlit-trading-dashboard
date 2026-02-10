
import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import numpy as np

# 1. Page Config
st.set_page_config(page_title="Global Indices Trading Signals", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .metric-container { padding: 10px; border-radius: 5px; background-color: #0e1117; margin-bottom: 10px; border: 1px solid #30333d; }
    .signal-buy { color: #00ff00; font-weight: bold; font-size: 24px; }
    .signal-sell { color: #ff0000; font-weight: bold; font-size: 24px; }
    .signal-neutral { color: #888888; font-weight: bold; font-size: 24px; }
</style>
""", unsafe_allow_html=True)

INDICES = {
    "Nasdaq 100": "^NDX",
    "DAX (Germany)": "^GDAXI",
    "Dow Jones Industrial Average": "^DJI",
    "Hang Seng (Hong Kong)": "^HSI",
    "SSE Composite (China)": "000001.SS"
}

# 2. Sidebar
st.sidebar.title("Configuration")
selected_index_name = st.sidebar.radio("Select Active Index:", list(INDICES.keys()), index=1)
selected_ticker = INDICES[selected_index_name]

strategy_type = st.sidebar.radio("Select Strategy:", [
    "Momentum (EMA + MACD + RSI)", 
    "Mean Reversion (Bollinger + RSI)", 
    "Scalping (EMA 9/21 Crossover)",
    "ORB (Opening Range Breakout)"
], index=0)

timeframe = st.sidebar.selectbox("Select Timeframe:", ["1m", "5m", "15m", "1h", "4h", "1d"], index=0)
auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=True)

# 3. Functions
def get_data(ticker, period="5d", interval="1m"):
    try:
        temp = yf.download(ticker, period=period, interval=interval, progress=False, prepost=True, auto_adjust=True)
        if temp.empty: return None

        # Flatten MultiIndex
        if isinstance(temp.columns, pd.MultiIndex):
            for i in range(temp.columns.nlevels):
                if 'Close' in temp.columns.get_level_values(i):
                    temp.columns = temp.columns.get_level_values(i)
                    break
        
        # Clean DataFrame
        df = pd.DataFrame(index=temp.index)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in temp.columns:
                series = temp[col]
                if isinstance(series, pd.DataFrame): series = series.iloc[:, 0]
                df[col] = pd.to_numeric(series, errors='coerce').astype(float)
        
        df = df.T.drop_duplicates().T
        df.dropna(subset=['Close'], inplace=True)
        df = df[df['Close'] > 0]
        return df
    except Exception:
        return None

def calculate_indicators(df):
    if df is None or len(df) == 0: return df
    df = df.copy()
    try:
        df['RSI'] = ta.rsi(df['Close'], length=14).astype(float)
        macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        if macd is not None:
            for col in macd.columns:
                if 'MACD_' in col and 'h' not in col and 's' not in col: df['MACD'] = macd[col].astype(float)
                if 'MACDs_' in col: df['MACD_Signal'] = macd[col].astype(float)
                if 'MACDh_' in col: df['MACD_Hist'] = macd[col].astype(float)
        df['EMA_9'] = ta.ema(df['Close'], length=9).astype(float)
        df['EMA_21'] = ta.ema(df['Close'], length=21).astype(float)
        df['EMA_50'] = ta.ema(df['Close'], length=50).astype(float)
        df['EMA_200'] = ta.ema(df['Close'], length=200).astype(float)
        bb = ta.bbands(df['Close'], length=20, std=2)
        if bb is not None:
            for col in bb.columns:
                if col.startswith('BBL'): df['BB_Lower'] = bb[col].astype(float)
                if col.startswith('BBU'): df['BB_Upper'] = bb[col].astype(float)
    except: pass
    return df

def apply_strategy(df, strategy_name):
    if df is None or len(df) < 5: return df
    df['Signal_Point'] = 0.0
    if "Momentum" in strategy_name and 'MACD' in df.columns:
        m, s = df['MACD'].fillna(0), df['MACD_Signal'].fillna(0)
        cross_up = (m.shift(1) <= s.shift(1)) & (m > s)
        cross_down = (m.shift(1) >= s.shift(1)) & (m < s)
        if 'EMA_200' in df.columns:
            ema200 = df['EMA_200'].ffill()
            df.loc[(df['Close'] > ema200) & cross_up & (df['RSI'].fillna(50) < 70), 'Signal_Point'] = 1.0
            df.loc[(df['Close'] < ema200) & cross_down & (df['RSI'].fillna(50) > 30), 'Signal_Point'] = -1.0
    elif "Mean Reversion" in strategy_name and 'BB_Lower' in df.columns:
        df.loc[(df['Close'] < df['BB_Lower']) & (df['RSI'].fillna(50) < 35), 'Signal_Point'] = 1.0
        df.loc[(df['Close'] > df['BB_Upper']) & (df['RSI'].fillna(50) > 65), 'Signal_Point'] = -1.0
    elif "Scalping" in strategy_name and 'EMA_9' in df.columns:
        e9, e21 = df['EMA_9'].fillna(0), df['EMA_21'].fillna(0)
        df.loc[(e9.shift(1) <= e21.shift(1)) & (e9 > e21), 'Signal_Point'] = 1.0
        df.loc[(e9.shift(1) >= e21.shift(1)) & (e9 < e21), 'Signal_Point'] = -1.0
    elif "ORB" in strategy_name:
        df['Date_Str'] = df.index.date
        for date, day_data in df.groupby('Date_Str'):
            if len(day_data) < 30: continue
            oh, ol = float(day_data.iloc[:30]['High'].max()), float(day_data.iloc[:30]['Low'].min())
            cutoff = day_data.index[29]
            mask = (df.index.date == date) & (df.index > cutoff)
            df.loc[mask & (df['Close'] > oh) & (df['Close'].shift(1) <= oh), 'Signal_Point'] = 1.0
            df.loc[mask & (df['Close'] < ol) & (df['Close'].shift(1) >= ol), 'Signal_Point'] = -1.0
            df.loc[df.index.date == date, 'ORB_High'], df.loc[df.index.date == date, 'ORB_Low'] = oh, ol
    return df

def plot_chart(df, ticker_name, strategy_name, timeframe):
    if df is None or len(df) < 2: return
    df_p = df.tail(150).copy()
    if len(df_p) < 2: return
    
    rows, heights, titles = (2, [0.7, 0.3], ("Price", "RSI"))
    if "Momentum" in strategy_name: rows, heights, titles = (3, [0.6, 0.2, 0.2], ("Price", "RSI", "MACD"))
    
    # Use categorical X-axis to prevent gaps and ensure scale
    idx_str = df_p.index.strftime('%H:%M')
    
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=heights, subplot_titles=titles)
    fig.add_trace(go.Candlestick(x=idx_str, open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close'], name='Market'), row=1, col=1)
    
    # Overlays
    if "Momentum" in strategy_name and 'EMA_200' in df_p.columns:
        fig.add_trace(go.Scatter(x=idx_str, y=df_p['EMA_200'], name='EMA 200', line=dict(color='blue', width=1.5)), row=1, col=1)
    elif "Mean Reversion" in strategy_name and 'BB_Upper' in df_p.columns:
        fig.add_trace(go.Scatter(x=idx_str, y=df_p['BB_Upper'], name='BB Upper', line=dict(color='gray', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=idx_str, y=df_p['BB_Lower'], name='BB Lower', line=dict(color='gray', width=1)), row=1, col=1)
    
    # Signals
    buys = df_p[df_p['Signal_Point'] == 1.0]; sells = df_p[df_p['Signal_Point'] == -1.0]
    if not buys.empty: fig.add_trace(go.Scatter(x=buys.index.strftime('%H:%M'), y=buys['Low']*0.9997, mode='markers', marker=dict(symbol='triangle-up', size=15, color='#00ff00'), name='BUY'), row=1, col=1)
    if not sells.empty: fig.add_trace(go.Scatter(x=sells.index.strftime('%H:%M'), y=sells['High']*1.0003, mode='markers', marker=dict(symbol='triangle-down', size=15, color='#ff0000'), name='SELL'), row=1, col=1)

    if 'RSI' in df_p.columns:
        fig.add_trace(go.Scatter(x=idx_str, y=df_p['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=2, col=1)
    if rows == 3 and 'MACD' in df_p.columns:
        fig.add_trace(go.Bar(x=idx_str, y=df_p['MACD_Hist'], name='Hist'), row=3, col=1)
        fig.add_trace(go.Scatter(x=idx_str, y=df_p['MACD'], name='MACD'), row=3, col=1)
    
    fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_dark", showlegend=False)
    fig.update_xaxes(type='category')
    st.plotly_chart(fig, use_container_width=True)

# 4. Main Execution
st.title(f"📊 {selected_index_name} Dashboard")

period_map = {"1m":"5d", "5m":"1mo", "15m":"1mo", "1h":"3mo", "4h":"1y", "1d":"2y"}
yf_interval = "1h" if timeframe == "4h" else timeframe

with st.spinner("Fetching data..."):
    df = get_data(selected_ticker, period=period_map.get(timeframe, "1mo"), interval=yf_interval)

if df is not None:
    if timeframe == "4h":
        df = df.resample('4H').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
    df = calculate_indicators(df)
    df = apply_strategy(df, strategy_type)
    
    # Metrics
    m1, m2, m3 = st.columns(3)
    curr_price = df['Close'].iloc[-1]
    m1.metric("Current Price", f"{curr_price:,.2f}")
    
    recent = df[df['Signal_Point'] != 0].tail(1)
    if not recent.empty:
        sig = "BUY" if recent['Signal_Point'].values[0] > 0 else "SELL"
        color = "#00ff00" if sig == "BUY" else "#ff0000"
        m2.markdown(f"**LAST SIGNAL:** <span style='color:{color}; font-size:24px;'>{sig}</span>", unsafe_allow_html=True)
        m3.info(f"At {recent.index[0].strftime('%H:%M')}")
    else:
        m2.write("No signals")

    plot_chart(df, selected_index_name, strategy_type, timeframe)
    
    with st.expander("Debug Raw Data"):
        st.write(df.tail(10).to_html(classes='table table-dark'), unsafe_allow_html=True)
else:
    st.error("Data source unavailable.")

if auto_refresh:
    time.sleep(30)
    st.rerun()
