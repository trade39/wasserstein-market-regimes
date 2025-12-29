import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands

# --- Configuration & Styles ---
st.set_page_config(page_title="Wasserstein Market Regimes & Forecasting", layout="wide")

# Sharp, High-Contrast Palette (Cyan, Magenta, Lime, Amber, Purple)
SHARP_PALETTE = ['#00E5FF', '#FF4081', '#00E676', '#FFC400', '#651FFF']
BACKGROUND_COLOR = '#0E1117' 

plt.rcParams.update({
    "figure.facecolor": BACKGROUND_COLOR,
    "axes.facecolor": BACKGROUND_COLOR,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "white",
    "text.color": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "#333333",
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

st.markdown("""
<style>
    .reportview-container { margin-top: -2em; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    div[data-testid="stImage"] {background-color: transparent;}
    
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { 
        height: 45px; 
        white-space: pre-wrap; 
        background-color: #161B22; 
        border: 1px solid #333; 
        border-radius: 4px; 
        padding: 0px 20px;
        color: #888;
    }
    .stTabs [aria-selected="true"] { 
        background-color: #00E5FF20; 
        border: 1px solid #00E5FF; 
        color: #00E5FF;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def get_ticker_and_fallback(asset_name):
    mapping = {
        "Gold": ("GC=F", "GLD"),
        "EURUSD": ("EURUSD=X", "FXE"),
        "ES (S&P 500 E-mini)": ("ES=F", "SPY"),
        "NQ (Nasdaq 100 E-mini)": ("NQ=F", "QQQ"),
        "BTC (Bitcoin)": ("BTC-USD", "BITO")
    }
    return mapping.get(asset_name, ("SPY", "SPY"))

@st.cache_data
def fetch_data(primary_ticker, fallback_ticker, start_date, end_date):
    def download_safe(t):
        try:
            return yf.download(t, start=start_date, end=end_date, progress=False, auto_adjust=False)
        except:
            return pd.DataFrame()

    data = download_safe(primary_ticker)
    used_ticker = primary_ticker
    if data.empty:
        data = download_safe(fallback_ticker)
        used_ticker = fallback_ticker

    if data.empty:
        return None, None

    if isinstance(data.columns, pd.MultiIndex):
        try:
            if used_ticker in data.columns.get_level_values(1):
                 data = data.xs(used_ticker, axis=1, level=1)
            else:
                 data.columns = data.columns.get_level_values(0)
        except:
             data.columns = data.columns.get_level_values(0)

    if 'Adj Close' in data.columns:
        price_col = 'Adj Close'
    elif 'Close' in data.columns:
        price_col = 'Close'
    else:
        return None, None

    data = data.copy()
    if price_col != 'Adj Close':
        data['Adj Close'] = data[price_col]

    data['LogReturn'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    data = data.dropna()
    return data, used_ticker

def lift_data(log_returns, h1, h2):
    n = len(log_returns)
    segments = []
    indices = []
    
    for i in range(0, n - h1 + 1, h2):
        window = log_returns.iloc[i : i + h1].values
        segments.append(np.sort(window))
        indices.append(log_returns.index[i + h1 - 1])
        
    return np.array(segments), indices

def wasserstein_barycenter_1d(segments):
    return np.median(segments, axis=0)

def wk_means_clustering(segments, k, max_iter=100, tol=1e-4):
    n_segments, window_size = segments.shape
    rng = np.random.default_rng(42)
    initial_indices = rng.choice(n_segments, size=k, replace=False)
    centroids = segments[initial_indices]
    
    prev_loss = float('inf')
    labels = np.zeros(n_segments, dtype=int)
    distances_to_centroid = np.zeros(n_segments)
    
    for iteration in range(max_iter):
        distances = np.zeros((n_segments, k))
        for j in range(k):
            diff = np.abs(segments - centroids[j]) 
            distances[:, j] = np.mean(diff, axis=1)
            
        labels = np.argmin(distances, axis=1)
        distances_to_centroid = np.min(distances, axis=1)
        current_loss = np.sum(distances_to_centroid)
        
        if np.abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss
        
        new_centroids = []
        for j in range(k):
            cluster_members = segments[labels == j]
            if len(cluster_members) > 0:
                new_centroids.append(wasserstein_barycenter_1d(cluster_members))
            else:
                new_centroids.append(segments[rng.choice(n_segments)])
        centroids = np.array(new_centroids)
        
    return labels, centroids, distances_to_centroid

# --- New Forecasting Logic (Review PDF) ---

def add_technical_indicators(df):
    """Adds indicators suggested in the ML Review PDF (RSI, MA, etc.)"""
    df = df.copy()
    
    # 1. Momentum: RSI
    rsi = RSIIndicator(close=df["Adj Close"], window=14)
    df["RSI"] = rsi.rsi()
    
    # 2. Trend: MACD
    macd = MACD(close=df["Adj Close"])
    df["MACD"] = macd.macd()
    
    # 3. Volatility: Bollinger Bands Width
    bb = BollingerBands(close=df["Adj Close"], window=20, window_dev=2)
    # FIX: Use .bollinger_mavg() instead of .bollinger_mband()
    df["BB_Width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    
    # 4. Simple Moving Averages
    df["SMA_20"] = SMAIndicator(close=df["Adj Close"], window=20).sma_indicator()
    df["Dist_SMA_20"] = (df["Adj Close"] - df["SMA_20"]) / df["SMA_20"] # Normalized distance
    
    # 5. Lagged Returns (Autoregression)
    df["Lag_1"] = df["LogReturn"].shift(1)
    df["Lag_2"] = df["LogReturn"].shift(2)
    df["Lag_5"] = df["LogReturn"].shift(5)
    
    return df.dropna()

def run_ml_forecasting(df, feature_cols, target_col="Target", test_size=0.2):
    """
    Trains a Random Forest to predict next day return.
    Uses 'Regime' as a key feature (Regime-Enhanced Forecasting).
    """
    # Create Target: Next Day Return
    data = df.copy()
    data["Target"] = data["LogReturn"].shift(-1)
    data = data.dropna()
    
    # Split Time Series (Sequential)
    split_idx = int(len(data) * (1 - test_size))
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]
    
    X_train = train[feature_cols]
    y_train = train[target_col]
    X_test = test[feature_cols]
    y_test = test[target_col]
    
    # Model: Random Forest (Robust default choice from literature)
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    pred_test = model.predict(X_test)
    
    # Metrics
    mse = mean_squared_error(y_test, pred_test)
    
    # Directional Accuracy (Up/Down)
    # 1 if (Actual * Pred) > 0 else 0
    correct_direction = np.sign(y_test) == np.sign(pred_test)
    acc = correct_direction.mean()
    
    return test.index, y_test, pred_test, mse, acc, model

# --- Main App ---

st.title("Regime-Aware Market Forecasting")
st.markdown("Combines **Wasserstein Clustering** (Regime Detection) with **ML Forecasting** (Random Forest + Tech Indicators).")

# Sidebar
st.sidebar.header("Data Source")
asset_select = st.sidebar.selectbox("Asset", ["Gold", "EURUSD", "ES (S&P 500 E-mini)", "NQ (Nasdaq 100 E-mini)", "BTC (Bitcoin)"])
primary_t, fallback_t = get_ticker_and_fallback(asset_select)
start_date = st.sidebar.date_input("Start", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End", pd.to_datetime("today"))

st.sidebar.divider()
st.sidebar.header("Model Parameters")
k_clusters = st.sidebar.slider("Regimes (k)", 2, 5, 2)
window_size = st.sidebar.slider("Window Size (h1)", 20, 100, 50)
step_size = st.sidebar.slider("Slide Step (h2)", 1, 20, 5)

if st.button("RUN ANALYSIS", type="primary", use_container_width=True):
    with st.spinner("Step 1: Calculating Regimes (Wasserstein)..."):
        df, ticker_used = fetch_data(primary_t, fallback_t, start_date, end_date)
        
        if df is not None and len(df) > window_size:
            # 1. Wasserstein Clustering (Regimes)
            segments, time_indices = lift_data(df['LogReturn'], window_size, step_size)
            labels, centroids, anomaly_scores = wk_means_clustering(segments, k_clusters)
            
            # Align Regimes to DataFrame
            res = pd.DataFrame(index=time_indices)
            res['Regime'] = labels
            
            main_df = df.join(res)
            # FFill regimes for days between steps (approximation for continuous features)
            main_df['Regime'] = main_df['Regime'].fillna(method='ffill')
            main_df = main_df.dropna(subset=['Regime'])
            
            # 2. Feature Engineering (Review PDF)
            with st.spinner("Step 2: Generating ML Features..."):
                main_df = add_technical_indicators(main_df)
                
                # One-Hot Encode Regime (Treat as Categorical Feature)
                # This lets the RF model learn different rules for Bull vs Bear markets
                regime_dummies = pd.get_dummies(main_df['Regime'], prefix='Regime')
                main_df = pd.concat([main_df, regime_dummies], axis=1)
                
            # 3. Forecasting
            with st.spinner("Step 3: Training Random Forest..."):
                # Define Feature Set
                feature_cols = ["RSI", "MACD", "BB_Width", "Dist_SMA_20", "Lag_1", "Lag_2", "Lag_5"]
                # Add regime columns
                feature_cols += [c for c in main_df.columns if c.startswith("Regime_")]
                
                test_dates, y_true, y_pred, mse, acc, model = run_ml_forecasting(main_df, feature_cols)

            # --- Visualization ---
            
            st.markdown("### 🔍 Model Performance")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Ticker", ticker_used)
            m2.metric("Directional Accuracy", f"{acc:.1%}", help="How often did the model predict the correct sign (Up/Down)?")
            m3.metric("RMSE", f"{np.sqrt(mse):.5f}", help="Root Mean Squared Error")
            m4.metric("Last Regime", f"Regime {int(main_df['Regime'].iloc[-1])}")
            
            st.markdown("---")
            
            tab1, tab2, tab3 = st.tabs(["🔮 FORECAST RESULTS", "📊 FEATURE IMPORTANCE", "📈 REGIMES (Validation)"])
            
            with tab1:
                # Cumulative Returns Comparison
                # Strategy: If Pred > 0 Buy, else Cash
                strat_signal = np.sign(y_pred)
                strat_ret = strat_signal * y_true
                
                cum_strat = (1 + strat_ret).cumprod()
                cum_bh = (1 + y_true).cumprod()
                
                fig_f, ax_f = plt.subplots(figsize=(12, 5))
                ax_f.plot(test_dates, cum_bh, color='white', alpha=0.3, ls='--', label='Buy & Hold')
                ax_f.plot(test_dates, cum_strat, color=SHARP_PALETTE[0], lw=2, label='ML Forecast Strategy')
                
                ax_f.set_title("Out-of-Sample Forecasting Performance (Test Set)", color='#888', loc='left')
                ax_f.legend(frameon=False)
                st.pyplot(fig_f)
                
                # Scatter of Pred vs Actual
                fig_s, ax_s = plt.subplots(figsize=(6, 6))
                ax_s.scatter(y_true, y_pred, alpha=0.5, color=SHARP_PALETTE[1], s=10)
                ax_s.axhline(0, color='white', alpha=0.1)
                ax_s.axvline(0, color='white', alpha=0.1)
                ax_s.set_xlabel("Actual Return")
                ax_s.set_ylabel("Predicted Return")
                ax_s.set_title("Prediction Accuracy Scatter", color='#888')
                st.pyplot(fig_s)
                
            with tab2:
                # Feature Importance
                importances = model.feature_importances_
                feat_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
                feat_df = feat_df.sort_values(by='Importance', ascending=False)
                
                fig_i, ax_i = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=feat_df, ax=ax_i, palette="viridis")
                ax_i.set_title("Which Features Matter Most?", color='#888', loc='left')
                st.pyplot(fig_i)
                st.caption("Note how 'Regime' features interact with technicals. If a Regime feature is high, it means the market state is a critical predictor.")

            with tab3:
                # Regime Plot (Validation)
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(main_df.index, main_df['Adj Close'], color='#FFFFFF', alpha=0.15, lw=1)
                
                colors = plt.cm.plasma(np.linspace(0, 1, k_clusters))
                for i in range(k_clusters):
                    subset = main_df[main_df['Regime'] == i]
                    ax.scatter(subset.index, subset['Adj Close'], 
                               color=colors[i], s=5, label=f'Regime {i}', zorder=5)
                
                ax.set_title(f"{asset_select} | Identified Regimes", fontsize=10, loc='left', color='#888')
                ax.legend(frameon=False)
                st.pyplot(fig)

        else:
            st.warning("Insufficient data. Try increasing the Date Range.")
else:
    st.info("Click 'RUN ANALYSIS' to train the model.")
