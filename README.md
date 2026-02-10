# Streamlit Trading Dashboard - Global Indices 📊

This is a **Streamlit** dashboard designed to track major global indices (Nasdaq 100, DAX, Dow Jones, etc.) and generate trading signals based on technical indicators.

### Features
*   **Real-time Data**: Fetches data using `yfinance` (1m, 5m, 15m, 1h, 4h, 1d timeframes).
*   **Multiple Strategies**:
    *   **Momentum**: EMA 200 Trend + MACD Crossover + RSI.
    *   **Mean Reversion**: Bollinger Bands + RSI Oversold/Overbought.
    *   **Scalping**: EMA 9 / EMA 21 Crossover.
    *   **ORB (Opening Range Breakout)**: Trade breakouts of the first 30 mins (best on 1m/5m).
*   **Intraday Focus**: Automatically filters 1m/5m charts to show only the current trading session.
*   **Interactive Charts**: Powered by Plotly with dynamic overlays and signal markers.

### Installation & Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Manu-fcuk/streamlit-trading-dashboard.git
    cd streamlit-trading-dashboard
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**:
    ```bash
    streamlit run app.py
    ```

### Deployment on Streamlit Cloud

1.  **Push to GitHub**:
    Ensure this code is pushed to your GitHub repository (`Manu-fcuk/streamlit-trading-dashboard`).
    ```bash
    git init
    git add .
    git commit -m "Initial commit"
    git branch -M main
    git remote add origin https://github.com/Manu-fcuk/streamlit-trading-dashboard.git
    git push -u origin main
    ```

2.  **Connect to Streamlit Cloud**:
    *   Go to [share.streamlit.io](https://share.streamlit.io/).
    *   Sign in with your GitHub account.
    *   Click **"New app"**.
    *   Select the repository: `Manu-fcuk/streamlit-trading-dashboard`.
    *   Branch: `main`.
    *   Main file path: `app.py`.
    *   Click **"Deploy!"**.

### Notes
*   **Data Latency**: Indices like DAX (`^GDAXI`) may have 15-20 min delayed data due to Yahoo Finance limitations. US Indices (`^NDX`, `^DJI`) are generally closer to real-time.
*   **Auto-Refresh**: The app auto-refreshes every 30 seconds to fetch new data.

---
Built with ❤️ by Manu-fcuk
