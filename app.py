import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from ta.volatility import BollingerBands
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon for sentiment analysis
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

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
    
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { 
        height: 50px; 
        white-space: pre-wrap; 
        background-color: #161B22; 
        border: 1px solid #333; 
        border-radius: 4px; 
        padding: 0px 10px;
        color: #888;
        font-size: 0.85em;
    }
    .stTabs [aria-selected="true"] { 
        background-color: #00E5FF10; 
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
        "Silver": ("SI=F", "SLV"), # Added for Pairs
        "EURUSD": ("EURUSD=X", "FXE"),
        "GBPUSD": ("GBPUSD=X", "FXB"), # Added for Pairs
        "ES (S&P 500)": ("ES=F", "SPY"),
        "NQ (Nasdaq)": ("NQ=F", "QQQ"),
        "RTY (Russell 2000)": ("RTY=F", "IWM"), # Added for Pairs
        "BTC (Bitcoin)": ("BTC-USD", "BITO"),
        "ETH (Ethereum)": ("ETH-USD", "ETHE") # Added for Pairs
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

def fetch_vix(start_date, end_date):
    """Fetches VIX data to use as a Sentiment/Fear proxy for the ML model."""
    try:
        vix = yf.download("^VIX", start=start_date, end=end_date, progress=False, auto_adjust=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        
        # We only need the Close price (Fear Level)
        return vix['Close'].rename("VIX")
    except:
        return None

def get_live_news_sentiment(ticker_symbol):
    """Fetches live news and calculates sentiment using NLTK VADER."""
    try:
        t = yf.Ticker(ticker_symbol)
        news = t.news
        if not news:
            return [], 0
        
        sia = SentimentIntensityAnalyzer()
        scored_news = []
        total_score = 0
        
        for item in news:
            title = item.get('title', '')
            score = sia.polarity_scores(title)['compound']
            scored_news.append({'title': title, 'score': score, 'link': item.get('link', '#'), 'publisher': item.get('publisher', 'Unknown')})
            total_score += score
            
        avg_score = total_score / len(news) if news else 0
        return scored_news, avg_score
    except Exception as e:
        return [], 0

# --- Core Math Functions ---

def lift_data(log_returns, h1, h2):
    n = len(log_returns)
    segments = []
    indices = []
    
    for i in range(0, n - h1 + 1, h2):
        window = log_returns.iloc[i : i + h1].values
        segments.append(np.sort(window))
        indices.append(log_returns.index[i + h1 - 1])
        
    return np.array(segments), indices

def calculate_wasserstein_distance_series(returns_a, returns_b, window_size):
    """
    Calculates rolling Wasserstein distance between two assets.
    Idea 3: Pairs Trading via Optimal Transport.
    """
    # Align dates
    df = pd.DataFrame({'A': returns_a, 'B': returns_b}).dropna()
    
    distances = []
    indices = []
    
    for i in range(window_size, len(df)):
        win_a = df['A'].iloc[i-window_size:i].values
        win_b = df['B'].iloc[i-window_size:i].values
        # Wasserstein Distance (L1 between sorted distributions)
        d = wasserstein_distance(win_a, win_b)
        distances.append(d)
        indices.append(df.index[i])
        
    return pd.Series(distances, index=indices)

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
                new_centroids.append(np.median(cluster_members, axis=0))
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
    scaler = StandardScaler()
    moments_scaled = scaler.fit_transform(moments)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    return kmeans.fit_predict(moments_scaled), indices

def calculate_transition_matrix(labels):
    df = pd.DataFrame({'Current': labels[:-1], 'Next': labels[1:]})
    return pd.crosstab(df['Current'], df['Next'], normalize='index')

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

def add_technical_indicators(df, vix_data=None):
    df = df.copy()
    
    # Technicals
    df["RSI"] = RSIIndicator(close=df["Adj Close"], window=14).rsi()
    df["MACD"] = MACD(close=df["Adj Close"]).macd()
    bb = BollingerBands(close=df["Adj Close"], window=20, window_dev=2)
    df["BB_Width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    df["SMA_20"] = SMAIndicator(close=df["Adj Close"], window=20).sma_indicator()
    df["Dist_SMA_20"] = (df["Adj Close"] - df["SMA_20"]) / df["SMA_20"] 
    
    # Auto-Regression
    df["Lag_1"] = df["LogReturn"].shift(1)
    df["Lag_2"] = df["LogReturn"].shift(2)
    
    # Idea 2: Add VIX (Market Sentiment) to Feature Set
    if vix_data is not None:
        df = df.join(vix_data, how='left')
        df['VIX'] = df['VIX'].fillna(method='ffill') # Fill holidays
        df['VIX_Change'] = df['VIX'].pct_change()
    
    return df.dropna()

def run_ml_forecasting(df, feature_cols, target_col="Target", test_size=0.2):
    data = df.copy()
    data["Target"] = data["LogReturn"].shift(-1)
    data = data.dropna()
    
    split_idx = int(len(data) * (1 - test_size))
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]
    
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(train[feature_cols], train[target_col])
    
    pred_test = model.predict(test[feature_cols])
    mse = mean_squared_error(test[target_col], pred_test)
    acc = (np.sign(test[target_col]) == np.sign(pred_test)).mean()
    
    return test.index, test[target_col], pred_test, mse, acc, model

# --- Main App ---

st.title("Wasserstein AI Trader: Hybrid Regime-Sentiment & Arbitrage")
st.markdown("Optimal Transport Clustering | Hybrid ML Forecasting | Statistical Arbitrage")

# Sidebar
st.sidebar.header("Asset Selection")
asset_select = st.sidebar.selectbox("Primary Asset", ["Gold", "ES (S&P 500)", "NQ (Nasdaq)", "BTC (Bitcoin)", "EURUSD"])
primary_t, fallback_t = get_ticker_and_fallback(asset_select)

st.sidebar.markdown("### Pairs Trading (Idea 3)")
pair_select = st.sidebar.selectbox("Comparison Asset (for Stat Arb)", ["Silver", "RTY (Russell 2000)", "ETH (Ethereum)", "GBPUSD", "Gold", "ES (S&P 500)"], index=0)
pair_t, pair_fallback_t = get_ticker_and_fallback(pair_select)

st.sidebar.divider()
st.sidebar.header("Data Range")
start_date = st.sidebar.date_input("Start", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End", pd.to_datetime("today"))

st.sidebar.header("Model Parameters")
k_clusters = st.sidebar.slider("Regimes (k)", 2, 5, 2)
window_size = st.sidebar.slider("Window Size (h1)", 20, 100, 50)
step_size = st.sidebar.slider("Slide Step (h2)", 1, 20, 5)

if st.button("RUN FULL ANALYSIS", type="primary", use_container_width=True):
    with st.spinner("Initializing Hybrid Model..."):
        # Fetch Data
        df, ticker_used = fetch_data(primary_t, fallback_t, start_date, end_date)
        
        # Idea 2: Fetch VIX for Hybrid Forecasting
        vix_df = fetch_vix(start_date, end_date)
        
        # Idea 3: Fetch Pair Data
        df_pair, pair_ticker_used = fetch_data(pair_t, pair_fallback_t, start_date, end_date)
        
        if df is not None and len(df) > window_size:
            # --- 1. Wasserstein Regimes ---
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

            # --- 2. Hybrid Forecasting (Regime + VIX) ---
            main_df = add_technical_indicators(main_df, vix_df)
            regime_dummies = pd.get_dummies(main_df['Regime'], prefix='Regime')
            main_df = pd.concat([main_df, regime_dummies], axis=1)
            
            feature_cols = ["RSI", "MACD", "BB_Width", "Dist_SMA_20", "Lag_1", "Lag_2"]
            if 'VIX' in main_df.columns: feature_cols += ['VIX', 'VIX_Change'] # Hybrid Feature
            feature_cols += [c for c in main_df.columns if c.startswith("Regime_")]
            
            test_dates, y_true, y_pred, mse, acc, model = run_ml_forecasting(main_df, feature_cols)

            # --- 3. Live Sentiment (Idea 2) ---
            news_items, news_score = get_live_news_sentiment(ticker_used)

            # --- 4. Pairs Trading (Idea 3) ---
            dist_series = None
            if df_pair is not None:
                # Synchronize data indices
                common_idx = df.index.intersection(df_pair.index)
                dist_series = calculate_wasserstein_distance_series(
                    df.loc[common_idx, 'LogReturn'], 
                    df_pair.loc[common_idx, 'LogReturn'], 
                    window_size
                )

            # --- Dashboard ---
            last_regime = int(main_df['Regime'].iloc[-1])
            regime_vol = np.std(segments[labels == last_regime].flatten()) * np.sqrt(252)
            
            # Header
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Regime", f"Regime {last_regime}", help="Classified by Wasserstein clustering")
            c2.metric("Forecast Accuracy", f"{acc:.1%}", help="Out-of-sample directional accuracy")
            c3.metric("Live Sentiment", f"{news_score:.2f}", delta="Bullish" if news_score > 0.05 else "Bearish" if news_score < -0.05 else "Neutral")
            c4.metric("Pair Divergence", f"{dist_series.iloc[-1]:.4f}" if dist_series is not None else "N/A", help="Current Wasserstein Distance to Pair")
            
            st.markdown("---")

            # TABS
            tabs = st.tabs([
                "📈 REGIMES", 
                "🔮 AI FORECAST (Hybrid)", 
                "⚖️ PAIRS TRADER",
                "📰 LIVE SENTIMENT",
                "⚠️ RISK & SHAPE", 
                "💰 STRATEGY", 
                "⚖️ BENCHMARK"
            ])

            # 1. Regimes
            with tabs[0]:
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(main_df.index, main_df['Adj Close'], color='#FFFFFF', alpha=0.15, lw=1)
                for i in range(k_clusters):
                    subset = main_df[main_df['Regime'] == i]
                    ax.scatter(subset.index, subset['Adj Close'], color=SHARP_PALETTE[i], s=8, label=f'Regime {i}', zorder=5)
                ax.legend(frameon=False)
                ax.set_title(f"{asset_select} | Regime Identification")
                st.pyplot(fig)

            # 2. Forecast
            with tabs[1]:
                c1, c2 = st.columns([3, 1])
                with c1:
                    strat_ret = np.sign(y_pred) * y_true
                    cum_strat = (1 + strat_ret).cumprod()
                    cum_bh = (1 + y_true).cumprod()
                    
                    fig_f, ax_f = plt.subplots(figsize=(10, 5))
                    ax_f.plot(test_dates, cum_bh, color='white', alpha=0.3, ls='--', label='Buy & Hold')
                    ax_f.plot(test_dates, cum_strat, color=SHARP_PALETTE[0], lw=2, label='Hybrid AI Strategy')
                    ax_f.legend(frameon=False)
                    ax_f.set_title("Forecast Performance (Incorporating VIX + Regimes)")
                    st.pyplot(fig_f)
                with c2:
                    st.markdown("**Feature Importance**")
                    imps = pd.DataFrame({'Feature': feature_cols, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
                    fig_i, ax_i = plt.subplots(figsize=(4, 6))
                    sns.barplot(x='Importance', y='Feature', data=imps.head(10), ax=ax_i, palette="viridis")
                    st.pyplot(fig_i)

            # 3. Pairs Trader (Idea 3)
            with tabs[2]:
                if dist_series is not None:
                    st.markdown(f"**Optimal Transport Arbitrage: {asset_select} vs {pair_select}**")
                    st.caption("Plots the Wasserstein Distance between the return distributions of the two assets. Spikes indicate structural decoupling (Trade Opportunity).")
                    
                    # Normalize distance (Z-Score)
                    z_score = (dist_series - dist_series.mean()) / dist_series.std()
                    
                    fig_p, ax_p = plt.subplots(figsize=(12, 5))
                    ax_p.plot(dist_series.index, z_score, color=SHARP_PALETTE[2], lw=1.5, label='Wasserstein Divergence (Z-Score)')
                    ax_p.axhline(2, color='red', ls='--', alpha=0.5, label='Sell Threshold (+2 Std)')
                    ax_p.axhline(-2, color='green', ls='--', alpha=0.5, label='Buy Threshold (-2 Std)')
                    ax_p.set_title("Distributional Divergence Signal")
                    ax_p.legend(frameon=False)
                    st.pyplot(fig_p)
                else:
                    st.error("Could not fetch pair data.")

            # 4. Live Sentiment (Idea 2)
            with tabs[3]:
                st.markdown(f"**Latest News & Sentiment: {ticker_used}**")
                st.caption("Powered by NLTK VADER (Valence Aware Dictionary and sEntiment Reasoner).")
                
                if news_items:
                    for item in news_items[:5]:
                        emoji = "🟢" if item['score'] > 0.05 else "🔴" if item['score'] < -0.05 else "⚪"
                        st.markdown(f"{emoji} **[{item['score']}]** [{item['title']}]({item['link']}) *({item['publisher']})*")
                else:
                    st.info("No recent news found via API.")

            # 5. Risk & Shape
            with tabs[4]:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Risk Profile (Return vs Vol)**")
                    w_means = np.mean(segments, axis=1) * 252
                    w_vols = np.std(segments, axis=1) * np.sqrt(252)
                    fig_mv, ax_mv = plt.subplots(figsize=(6, 5))
                    for i in range(k_clusters):
                        m = labels == i
                        ax_mv.scatter(w_vols[m], w_means[m], color=SHARP_PALETTE[i], s=15, alpha=0.6, label=f'Regime {i}')
                    ax_mv.legend(frameon=False)
                    st.pyplot(fig_mv)
                with c2:
                    st.markdown("**Wasserstein Anomaly Score**")
                    fig_ad, ax_ad = plt.subplots(figsize=(6, 5))
                    ax_ad.plot(main_df.index, main_df['Anomaly_Score'], color=SHARP_PALETTE[1], lw=1)
                    st.pyplot(fig_ad)

            # 6. Strategy
            with tabs[5]:
                backtest_df, safe_regime = run_backtest(main_df['Adj Close'], main_df['Regime'], main_df.index)
                fig_bt, ax_bt = plt.subplots(figsize=(12, 5))
                ax_bt.plot(backtest_df.index, backtest_df['Strategy_Equity'], color=SHARP_PALETTE[3], lw=2, label='Regime Strategy')
                ax_bt.plot(backtest_df.index, backtest_df['BuyHold_Equity'], color='white', alpha=0.3, ls='--', label='Buy & Hold')
                ax_bt.legend(frameon=False)
                ax_bt.set_title(f"Long Regime {safe_regime} / Cash Regime {1 if safe_regime==0 else 0}")
                st.pyplot(fig_bt)

            # 7. Benchmark
            with tabs[6]:
                blabels, bindices = moment_kmeans_clustering(df['LogReturn'], window_size, step_size, k_clusters)
                bdf = pd.DataFrame({'Cluster': blabels}, index=bindices).join(df[['Adj Close']])
                fig_b, ax_b = plt.subplots(figsize=(12, 5))
                ax_b.plot(bdf.index, bdf['Adj Close'], color='#FFFFFF', alpha=0.1)
                for i in range(k_clusters):
                    s = bdf[bdf['Cluster'] == i]
                    ax_b.scatter(s.index, s['Adj Close'], s=8, alpha=0.6)
                st.pyplot(fig_b)

        else:
            st.warning("Insufficient data. Adjust window size.")
else:
    st.info("Set parameters and click RUN FULL ANALYSIS.")
