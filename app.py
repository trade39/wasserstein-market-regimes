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
from ta.trend import SMAIndicator, MACD
from ta.volatility import BollingerBands

# --- Configuration & Styles ---
st.set_page_config(page_title="Wasserstein AI Trader", layout="wide")

# Sharp, High-Contrast Palette
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
        padding: 0px 10px;
        color: #888;
        font-size: 0.9em;
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

def moment_kmeans_clustering(log_returns, h1, h2, k):
    moments = []
    indices = []
    n = len(log_returns)
    
    for i in range(0, n - h1 + 1, h2):
        window = log_returns.iloc[i : i + h1].values
        m_vec = [np.mean(window), np.std(window), skew(window), kurtosis(window)]
        moments.append(m_vec)
        indices.append(log_returns.index[i + h1 - 1])
    
    moments = np.array(moments)
    scaler = StandardScaler()
    moments_scaled = scaler.fit_transform(moments)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(moments_scaled)
    
    return labels, indices

def calculate_transition_matrix(labels):
    df = pd.DataFrame({'Current': labels[:-1], 'Next': labels[1:]})
    matrix = pd.crosstab(df['Current'], df['Next'], normalize='index')
    return matrix

def run_backtest(price_series, labels, time_indices):
    signal_df = pd.DataFrame(index=time_indices)
    signal_df['Regime'] = labels
    full_returns = price_series.pct_change()
    aligned_signals = signal_df.reindex(price_series.index).fillna(method='ffill')
    
    regime_vols = {}
    unique_regimes = np.unique(labels)
    for r in unique_regimes:
        mask = aligned_signals['Regime'] == r
        regime_vols[r] = full_returns[mask].std()
        
    safe_regime = min(regime_vols, key=regime_vols.get)
    aligned_signals['Position'] = np.where(aligned_signals['Regime'] == safe_regime, 1, 0)
    aligned_signals['Position'] = aligned_signals['Position'].shift(1)
    
    aligned_signals['Strategy_Return'] = aligned_signals['Position'] * full_returns
    aligned_signals['BuyHold_Return'] = full_returns
    aligned_signals['Strategy_Equity'] = (1 + aligned_signals['Strategy_Return'].fillna(0)).cumprod()
    aligned_signals['BuyHold_Equity'] = (1 + aligned_signals['BuyHold_Return'].fillna(0)).cumprod()
    
    return aligned_signals, safe_regime

# --- Forecasting Functions ---

def add_technical_indicators(df):
    df = df.copy()
    
    # RSI
    rsi = RSIIndicator(close=df["Adj Close"], window=14)
    df["RSI"] = rsi.rsi()
    
    # MACD
    macd = MACD(close=df["Adj Close"])
    df["MACD"] = macd.macd()
    
    # BB Width - FIXED .bollinger_mavg()
    bb = BollingerBands(close=df["Adj Close"], window=20, window_dev=2)
    df["BB_Width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    
    # SMA Dist
    df["SMA_20"] = SMAIndicator(close=df["Adj Close"], window=20).sma_indicator()
    df["Dist_SMA_20"] = (df["Adj Close"] - df["SMA_20"]) / df["SMA_20"] 
    
    # Lags
    df["Lag_1"] = df["LogReturn"].shift(1)
    df["Lag_2"] = df["LogReturn"].shift(2)
    df["Lag_5"] = df["LogReturn"].shift(5)
    
    return df.dropna()

def run_ml_forecasting(df, feature_cols, target_col="Target", test_size=0.2):
    data = df.copy()
    data["Target"] = data["LogReturn"].shift(-1)
    data = data.dropna()
    
    split_idx = int(len(data) * (1 - test_size))
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]
    
    X_train = train[feature_cols]
    y_train = train[target_col]
    X_test = test[feature_cols]
    y_test = test[target_col]
    
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    pred_test = model.predict(X_test)
    mse = mean_squared_error(y_test, pred_test)
    
    correct_direction = np.sign(y_test) == np.sign(pred_test)
    acc = correct_direction.mean()
    
    return test.index, y_test, pred_test, mse, acc, model

# --- Main App ---

st.title("Wasserstein Market Regimes & Forecasting")
st.markdown("Distributional Clustering + ML Forecasting")

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

if st.button("RUN FULL ANALYSIS", type="primary", use_container_width=True):
    with st.spinner("Processing Wasserstein Regimes & ML Models..."):
        df, ticker_used = fetch_data(primary_t, fallback_t, start_date, end_date)
        
        if df is not None and len(df) > window_size:
            # 1. Regimes
            segments, time_indices = lift_data(df['LogReturn'], window_size, step_size)
            labels, centroids, anomaly_scores = wk_means_clustering(segments, k_clusters)
            
            res = pd.DataFrame(index=time_indices)
            res['Regime'] = labels
            res['Anomaly_Score'] = anomaly_scores
            
            main_df = df.join(res)
            main_df['Regime'] = main_df['Regime'].fillna(method='ffill')
            main_df['Anomaly_Score'] = main_df['Anomaly_Score'].fillna(method='ffill')
            main_df = main_df.dropna(subset=['Regime'])
            main_df['Regime'] = main_df['Regime'].astype(int)

            # 2. Forecasting Prep
            main_df = add_technical_indicators(main_df)
            regime_dummies = pd.get_dummies(main_df['Regime'], prefix='Regime')
            main_df = pd.concat([main_df, regime_dummies], axis=1)
            
            feature_cols = ["RSI", "MACD", "BB_Width", "Dist_SMA_20", "Lag_1", "Lag_2", "Lag_5"]
            feature_cols += [c for c in main_df.columns if c.startswith("Regime_")]
            
            test_dates, y_true, y_pred, mse, acc, model = run_ml_forecasting(main_df, feature_cols)

            # --- Header Metrics ---
            last_regime = int(main_df['Regime'].iloc[-1])
            regime_stats = []
            for i in range(k_clusters):
                idx = np.where(labels == i)[0]
                flat_rets = segments[idx].flatten()
                regime_stats.append({
                    "vol": np.std(flat_rets) * np.sqrt(252),
                    "ret": np.mean(flat_rets) * 252
                })
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Regime", f"Regime {last_regime}")
            c2.metric("Regime Volatility", f"{regime_stats[last_regime]['vol']:.1%}")
            c3.metric("Forecast Accuracy", f"{acc:.1%}", help="Directional Accuracy (Test Set)")
            c4.metric("Ticker", ticker_used)
            
            st.markdown("---")

            # --- TABS (6 TOTAL) ---
            t1, t2, t3, t4, t5, t6 = st.tabs([
                "📈 REGIMES", 
                "🔮 AI FORECAST", 
                "📐 SHAPES", 
                "⚠️ RISK", 
                "💰 STRATEGY", 
                "⚖️ BENCHMARK"
            ])

            # Tab 1: Price & Regimes
            with t1:
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(main_df.index, main_df['Adj Close'], color='#FFFFFF', alpha=0.15, lw=1)
                for i in range(k_clusters):
                    subset = main_df[main_df['Regime'] == i]
                    ax.scatter(subset.index, subset['Adj Close'], 
                               color=SHARP_PALETTE[i], s=8, label=f'Regime {i}', zorder=5)
                ax.set_title(f"{asset_select} | Regime Identification", color='#888')
                ax.legend(frameon=False)
                st.pyplot(fig)
                
                st.markdown("###### Regime Stats")
                st.dataframe(pd.DataFrame(regime_stats).style.format("{:.2%}"))

            # Tab 2: Forecast (New)
            with t2:
                col_a, col_b = st.columns([2,1])
                with col_a:
                    st.markdown("**Out-of-Sample Performance**")
                    strat_signal = np.sign(y_pred)
                    strat_ret = strat_signal * y_true
                    cum_strat = (1 + strat_ret).cumprod()
                    cum_bh = (1 + y_true).cumprod()
                    
                    fig_f, ax_f = plt.subplots(figsize=(10, 5))
                    ax_f.plot(test_dates, cum_bh, color='white', alpha=0.3, ls='--', label='Buy & Hold')
                    ax_f.plot(test_dates, cum_strat, color=SHARP_PALETTE[0], lw=2, label='AI Forecast Strategy')
                    ax_f.legend(frameon=False)
                    st.pyplot(fig_f)
                
                with col_b:
                    st.markdown("**Feature Importance**")
                    importances = model.feature_importances_
                    feat_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values('Importance', ascending=False)
                    fig_i, ax_i = plt.subplots(figsize=(5, 6))
                    sns.barplot(x='Importance', y='Feature', data=feat_df, ax=ax_i, palette="viridis")
                    st.pyplot(fig_i)

            # Tab 3: Shapes
            with t3:
                c1, c2 = st.columns(2)
                with c1:
                    window_means = np.mean(segments, axis=1) * 252
                    window_vols = np.std(segments, axis=1) * np.sqrt(252)
                    fig_mv, ax_mv = plt.subplots(figsize=(6, 5))
                    for i in range(k_clusters):
                        mask = labels == i
                        ax_mv.scatter(window_vols[mask], window_means[mask], color=SHARP_PALETTE[i], s=15, alpha=0.6, label=f'Regime {i}')
                    ax_mv.set_xlabel("Volatility")
                    ax_mv.set_ylabel("Return")
                    ax_mv.legend(frameon=False)
                    st.pyplot(fig_mv)
                with c2:
                    fig_skew, ax_skew = plt.subplots(figsize=(6, 5))
                    for i in range(k_clusters):
                        cluster_skews = [skew(seg) for seg in segments[labels == i]]
                        sns.kdeplot(cluster_skews, ax=ax_skew, color=SHARP_PALETTE[i], fill=True, alpha=0.2)
                    ax_skew.set_title("Skewness Distribution")
                    st.pyplot(fig_skew)

            # Tab 4: Risk
            with t4:
                c1, c2 = st.columns(2)
                with c1:
                    trans_mat = calculate_transition_matrix(labels)
                    fig_hm, ax_hm = plt.subplots()
                    sns.heatmap(trans_mat, annot=True, fmt=".2f", cmap=sns.dark_palette(SHARP_PALETTE[0], as_cmap=True), cbar=False, ax=ax_hm)
                    st.pyplot(fig_hm)
                with c2:
                    fig_ad, ax_ad = plt.subplots(figsize=(6, 4))
                    ax_ad.plot(main_df.index, main_df['Anomaly_Score'], color=SHARP_PALETTE[1], lw=0.8)
                    ax_ad.set_title("Wasserstein Anomaly Score")
                    st.pyplot(fig_ad)

            # Tab 5: Strategy
            with t5:
                backtest_df, safe_regime = run_backtest(main_df['Adj Close'], main_df['Regime'], main_df.index)
                fig_bt, ax_bt = plt.subplots(figsize=(12, 5))
                ax_bt.plot(backtest_df.index, backtest_df['BuyHold_Equity'], color='#FFFFFF', alpha=0.3, ls='--')
                ax_bt.plot(backtest_df.index, backtest_df['Strategy_Equity'], color=SHARP_PALETTE[2], lw=2)
                ax_bt.set_title(f"Regime Strategy (Long Regime {safe_regime}, Cash otherwise)")
                st.pyplot(fig_bt)

            # Tab 6: Benchmark
            with t6:
                bench_labels, bench_indices = moment_kmeans_clustering(df['LogReturn'], window_size, step_size, k_clusters)
                bench_df = pd.DataFrame(index=bench_indices)
                bench_df['Bench_Cluster'] = bench_labels
                plot_df = df.join(bench_df).dropna()
                
                fig_b, ax_b = plt.subplots(figsize=(12, 5))
                ax_b.plot(plot_df.index, plot_df['Adj Close'], color='#FFFFFF', alpha=0.1)
                BENCH_PALETTE = ['#FF9100', '#00B0FF', '#D500F9', '#FF1744']
                for i in range(k_clusters):
                    subset = plot_df[plot_df['Bench_Cluster'] == i]
                    ax_b.scatter(subset.index, subset['Adj Close'], color=BENCH_PALETTE[i], s=8, alpha=0.7)
                st.pyplot(fig_b)
                st.caption("Benchmark: Standard K-Means on [Mean, Std, Skew, Kurtosis].")

        else:
            st.warning("Insufficient data.")
else:
    st.info("Configure and click RUN FULL ANALYSIS.")
