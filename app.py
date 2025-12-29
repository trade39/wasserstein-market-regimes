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

# Download VADER lexicon
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# --- Configuration & Styles ---
st.set_page_config(page_title="Wasserstein AI Trader", layout="wide")

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

def get_asset_config(asset_name):
    """Returns tickers and news proxies for each asset."""
    mapping = {
        "Gold": {
            "primary": "GC=F", 
            "price_fallback": "GLD", 
            "news_proxies": ["GC=F", "GLD", "NEM", "GOLD", "GDX"]
        },
        "Silver": {
            "primary": "SI=F", 
            "price_fallback": "SLV", 
            "news_proxies": ["SI=F", "SLV", "PAAS", "AG"]
        },
        "EURUSD": {
            "primary": "EURUSD=X", 
            "price_fallback": "FXE", 
            "news_proxies": ["EURUSD=X", "FXE", "UUP"]
        },
        "GBPUSD": {
            "primary": "GBPUSD=X", 
            "price_fallback": "FXB", 
            "news_proxies": ["GBPUSD=X", "FXB", "UUP"]
        },
        "ES (S&P 500)": {
            "primary": "ES=F", 
            "price_fallback": "SPY", 
            "news_proxies": ["ES=F", "SPY", "IVV", "VOO"]
        },
        "NQ (Nasdaq)": {
            "primary": "NQ=F", 
            "price_fallback": "QQQ", 
            "news_proxies": ["NQ=F", "QQQ", "TQQQ", "AAPL", "MSFT"]
        },
        "RTY (Russell 2000)": {
            "primary": "RTY=F", 
            "price_fallback": "IWM", 
            "news_proxies": ["RTY=F", "IWM", "TNA"]
        },
        "BTC (Bitcoin)": {
            "primary": "BTC-USD", 
            "price_fallback": "BITO", 
            "news_proxies": ["BTC-USD", "BITO", "COIN", "MSTR"]
        },
        "ETH (Ethereum)": {
            "primary": "ETH-USD", 
            "price_fallback": "ETHE", 
            "news_proxies": ["ETH-USD", "ETHE", "COIN"]
        }
    }
    return mapping.get(asset_name, {"primary": "SPY", "price_fallback": "SPY", "news_proxies": ["SPY"]})

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
    try:
        vix = yf.download("^VIX", start=start_date, end=end_date, progress=False, auto_adjust=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        return vix['Close'].rename("VIX")
    except:
        return None

def get_live_news_sentiment(news_proxies):
    """
    Tries to fetch news. If API fails (returns empty), returns None 
    so the app knows to switch to Technical Sentiment.
    """
    sia = SentimentIntensityAnalyzer()
    
    def fetch_valid_news(t):
        try:
            raw_news = yf.Ticker(t).news
            # Relaxed Filter: Only needs a Title.
            valid = [item for item in raw_news if item.get('title')]
            return valid
        except:
            return []

    found_news = []
    source_used = "None"

    for ticker in news_proxies:
        found_news = fetch_valid_news(ticker)
        if found_news:
            source_used = ticker
            break
            
    # If API is completely blocked/empty
    if not found_news:
        return [], 0.0, None

    scored_news = []
    total_score = 0
    
    for item in found_news:
        title = item.get('title', 'No Title')
        score = sia.polarity_scores(title)['compound']
        publisher = item.get('publisher', 'Unknown')
        link = item.get('link', '#')
        
        scored_news.append({
            'title': title, 
            'score': score, 
            'link': link, 
            'publisher': publisher
        })
        total_score += score
        
    avg_score = total_score / len(found_news) if found_news else 0
    return scored_news, avg_score, source_used

def get_technical_sentiment(last_row):
    """Calculates a Synthetic Sentiment Score based on Technicals if News fails."""
    score = 0
    reasons = []
    
    # RSI Logic
    if last_row['RSI'] > 50: 
        score += 0.3
        reasons.append("RSI Bullish (>50)")
    else:
        score -= 0.3
        reasons.append("RSI Bearish (<50)")
        
    # Trend Logic (Price vs SMA)
    if last_row['Dist_SMA_20'] > 0:
        score += 0.4
        reasons.append("Price above SMA20")
    else:
        score -= 0.4
        reasons.append("Price below SMA20")
        
    # MACD Logic
    if last_row['MACD'] > 0:
        score += 0.2
        reasons.append("MACD Positive")
    else:
        score -= 0.2
        reasons.append("MACD Negative")
        
    # Normalize to -1 to 1 range (approx)
    score = max(min(score, 0.9), -0.9)
    
    # Create fake "News Items" to display the reasons
    dummy_news = []
    for r in reasons:
        dummy_news.append({
            'title': r,
            'score': 0.5 if "Bullish" in r or "above" in r or "Positive" in r else -0.5,
            'link': '#',
            'publisher': 'Technical Indicator'
        })
        
    return dummy_news, score, "Technical Indicators (Fallback)"

# --- Math Functions ---

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
    df = pd.DataFrame({'A': returns_a, 'B': returns_b}).dropna()
    distances = []
    indices = []
    for i in range(window_size, len(df)):
        win_a = df['A'].iloc[i-window_size:i].values
        win_b = df['B'].iloc[i-window_size:i].values
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

# --- Forecasting ---

def add_technical_indicators(df, vix_data=None):
    df = df.copy()
    df["RSI"] = RSIIndicator(close=df["Adj Close"], window=14).rsi()
    df["MACD"] = MACD(close=df["Adj Close"]).macd()
    bb = BollingerBands(close=df["Adj Close"], window=20, window_dev=2)
    df["BB_Width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    df["SMA_20"] = SMAIndicator(close=df["Adj Close"], window=20).sma_indicator()
    df["Dist_SMA_20"] = (df["Adj Close"] - df["SMA_20"]) / df["SMA_20"] 
    df["Lag_1"] = df["LogReturn"].shift(1)
    df["Lag_2"] = df["LogReturn"].shift(2)
    if vix_data is not None:
        df = df.join(vix_data, how='left')
        df['VIX'] = df['VIX'].fillna(method='ffill') 
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
asset_select = st.sidebar.selectbox("Primary Asset", ["Gold", "ES (S&P 500)", "NQ (Nasdaq)", "BTC (Bitcoin)", "EURUSD", "Silver", "RTY (Russell 2000)", "GBPUSD"])
config = get_asset_config(asset_select)
primary_t = config['primary']
fallback_t = config['price_fallback']
news_proxies = config['news_proxies']

st.sidebar.markdown("### Pairs Trading")
pair_select = st.sidebar.selectbox("Comparison Asset", ["Silver", "RTY (Russell 2000)", "ETH (Ethereum)", "GBPUSD", "Gold", "ES (S&P 500)"], index=0)
pair_config = get_asset_config(pair_select)
pair_t = pair_config['primary']
pair_fallback_t = pair_config['price_fallback']

st.sidebar.divider()
st.sidebar.header("Settings")
start_date = st.sidebar.date_input("Start", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End", pd.to_datetime("today"))
k_clusters = st.sidebar.slider("Regimes (k)", 2, 5, 2)
window_size = st.sidebar.slider("Window Size (h1)", 20, 100, 50)
step_size = st.sidebar.slider("Slide Step (h2)", 1, 20, 5)

if st.button("RUN FULL ANALYSIS", type="primary", use_container_width=True):
    with st.spinner("Initializing Models..."):
        df, ticker_used = fetch_data(primary_t, fallback_t, start_date, end_date)
        vix_df = fetch_vix(start_date, end_date)
        df_pair, pair_ticker_used = fetch_data(pair_t, pair_fallback_t, start_date, end_date)
        
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

            # 2. Forecast
            main_df = add_technical_indicators(main_df, vix_df)
            regime_dummies = pd.get_dummies(main_df['Regime'], prefix='Regime')
            main_df = pd.concat([main_df, regime_dummies], axis=1)
            feature_cols = ["RSI", "MACD", "BB_Width", "Dist_SMA_20", "Lag_1", "Lag_2"]
            if 'VIX' in main_df.columns: feature_cols += ['VIX', 'VIX_Change']
            feature_cols += [c for c in main_df.columns if c.startswith("Regime_")]
            test_dates, y_true, y_pred, mse, acc, model = run_ml_forecasting(main_df, feature_cols)

            # 3. Sentiment (with Technical Fallback)
            news_items, news_score, news_source = get_live_news_sentiment(news_proxies)
            
            # FAILSAFE: If News API is blocked, use Technicals
            if news_source is None:
                news_items, news_score, news_source = get_technical_sentiment(main_df.iloc[-1])

            # 4. Pairs
            dist_series = None
            if df_pair is not None:
                common_idx = df.index.intersection(df_pair.index)
                dist_series = calculate_wasserstein_distance_series(
                    df.loc[common_idx, 'LogReturn'], 
                    df_pair.loc[common_idx, 'LogReturn'], 
                    window_size
                )

            # Dashboard
            last_regime = int(main_df['Regime'].iloc[-1])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Regime", f"Regime {last_regime}")
            c2.metric("Forecast Accuracy", f"{acc:.1%}")
            c3.metric(f"Sentiment ({news_source})", f"{news_score:.2f}", delta="Bullish" if news_score > 0.05 else "Bearish" if news_score < -0.05 else "Neutral")
            c4.metric("Pair Divergence", f"{dist_series.iloc[-1]:.4f}" if dist_series is not None else "N/A")
            
            st.markdown("---")
            tabs = st.tabs(["📈 REGIMES", "🔮 FORECAST", "⚖️ PAIRS TRADER", "📰 LIVE SENTIMENT", "⚠️ RISK", "💰 STRATEGY", "⚖️ BENCHMARK"])

            with tabs[0]:
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(main_df.index, main_df['Adj Close'], color='#FFFFFF', alpha=0.15, lw=1)
                for i in range(k_clusters):
                    subset = main_df[main_df['Regime'] == i]
                    ax.scatter(subset.index, subset['Adj Close'], color=SHARP_PALETTE[i], s=8, label=f'Regime {i}', zorder=5)
                ax.legend(frameon=False)
                st.pyplot(fig)

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
                    st.pyplot(fig_f)
                with c2:
                    imps = pd.DataFrame({'Feature': feature_cols, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
                    fig_i, ax_i = plt.subplots(figsize=(4, 6))
                    sns.barplot(x='Importance', y='Feature', data=imps.head(10), ax=ax_i, palette="viridis")
                    st.pyplot(fig_i)

            with tabs[2]:
                if dist_series is not None:
                    z_score = (dist_series - dist_series.mean()) / dist_series.std()
                    fig_p, ax_p = plt.subplots(figsize=(12, 5))
                    ax_p.plot(dist_series.index, z_score, color=SHARP_PALETTE[2], lw=1.5, label='Wasserstein Divergence (Z-Score)')
                    ax_p.axhline(2, color='red', ls='--', alpha=0.5, label='Sell Threshold (+2 Std)')
                    ax_p.axhline(-2, color='green', ls='--', alpha=0.5, label='Buy Threshold (-2 Std)')
                    ax_p.legend(frameon=False)
                    st.pyplot(fig_p)

            with tabs[3]:
                st.markdown(f"**Source: {news_source}**")
                for item in news_items[:5]:
                    emoji = "🟢" if item['score'] > 0.05 else "🔴" if item['score'] < -0.05 else "⚪"
                    st.markdown(f"{emoji} **[{item['score']:.2f}]** {item['title']} *({item['publisher']})*")

            with tabs[4]:
                c1, c2 = st.columns(2)
                with c1:
                    w_means = np.mean(segments, axis=1) * 252
                    w_vols = np.std(segments, axis=1) * np.sqrt(252)
                    fig_mv, ax_mv = plt.subplots(figsize=(6, 5))
                    for i in range(k_clusters):
                        m = labels == i
                        ax_mv.scatter(w_vols[m], w_means[m], color=SHARP_PALETTE[i], s=15, alpha=0.6, label=f'Regime {i}')
                    st.pyplot(fig_mv)
                with c2:
                    fig_ad, ax_ad = plt.subplots(figsize=(6, 5))
                    ax_ad.plot(main_df.index, main_df['Anomaly_Score'], color=SHARP_PALETTE[1], lw=1)
                    st.pyplot(fig_ad)

            with tabs[5]:
                backtest_df, safe_regime = run_backtest(main_df['Adj Close'], main_df['Regime'], main_df.index)
                fig_bt, ax_bt = plt.subplots(figsize=(12, 5))
                ax_bt.plot(backtest_df.index, backtest_df['Strategy_Equity'], color=SHARP_PALETTE[3], lw=2)
                ax_bt.plot(backtest_df.index, backtest_df['BuyHold_Equity'], color='white', alpha=0.3, ls='--')
                st.pyplot(fig_bt)

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
            st.warning("Insufficient data.")
else:
    st.info("Set parameters and click RUN FULL ANALYSIS.")
