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
st.set_page_config(
    page_title="Wasserstein Quant Terminal", 
    layout="wide",
    page_icon="📈"
)

# --- QUANT PALETTE (White/Blue Theme) ---
# Professional, Institutional Colors
QUANT_PALETTE = ['#0D47A1', '#1976D2', '#42A5F5', '#90CAF9', '#2962FF'] # Deep Blue to Light Blue
STRATEGY_COLOR = '#002984' # Dark Royal Blue
BENCHMARK_COLOR = '#9E9E9E' # Grey
POSITIVE_COLOR = '#2E7D32' # Professional Green
NEGATIVE_COLOR = '#C62828' # Professional Red
BACKGROUND_COLOR = '#FFFFFF'
TEXT_COLOR = '#212121'

# Global Plot Settings for "Research Report" Look
plt.rcParams.update({
    "figure.facecolor": BACKGROUND_COLOR,
    "axes.facecolor": BACKGROUND_COLOR,
    "axes.edgecolor": "#E0E0E0", # Light grey borders
    "axes.labelcolor": "#424242",
    "text.color": TEXT_COLOR,
    "xtick.color": "#616161",
    "ytick.color": "#616161",
    "grid.color": "#EEEEEE", # Very subtle grid
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "axes.spines.top": True, # Boxed look
    "axes.spines.right": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "font.family": "sans-serif",
})

# Custom CSS for Streamlit
st.markdown(f"""
<style>
    /* Force Light Theme Logic */
    .stApp {{
        background-color: #F5F7F9; /* Very light blue-grey background */
    }}
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {{
        color: #0D47A1;
        font-family: 'Roboto Mono', monospace;
    }}
    
    /* Professional Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: transparent;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 45px;
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 4px;
        color: #616161;
        font-weight: 500;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: #E3F2FD;
        border: 1px solid #1976D2;
        color: #0D47A1;
        font-weight: bold;
    }}
    
    /* Buttons */
    div.stButton > button {{
        background-color: #1565C0;
        color: white;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    div.stButton > button:hover {{
        background-color: #0D47A1;
        color: white;
    }}
    
    h1, h2, h3 {{
        color: #0D47A1;
        font-weight: 700;
    }}
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def get_asset_config(asset_name):
    mapping = {
        "Gold": {
            "primary": "GC=F", "price_fallback": "GLD", 
            "news_proxies": ["GC=F", "GLD", "NEM", "GOLD", "GDX"],
            "candidates": ["SI=F", "GDX", "DX-Y.NYB", "^TNX", "AUDUSD=X"] 
        },
        "Silver": {
            "primary": "SI=F", "price_fallback": "SLV", 
            "news_proxies": ["SI=F", "SLV", "PAAS", "AG"],
            "candidates": ["GC=F", "COPX", "GDX", "DX-Y.NYB"]
        },
        "EURUSD": {
            "primary": "EURUSD=X", "price_fallback": "FXE", 
            "news_proxies": ["EURUSD=X", "FXE", "UUP"],
            "candidates": ["GBPUSD=X", "DX-Y.NYB", "CHF=X", "GC=F"] 
        },
        "GBPUSD": {
            "primary": "GBPUSD=X", "price_fallback": "FXB", 
            "news_proxies": ["GBPUSD=X", "FXB", "UUP"],
            "candidates": ["EURUSD=X", "DX-Y.NYB", "EWU"]
        },
        "ES (S&P 500)": {
            "primary": "ES=F", "price_fallback": "SPY", 
            "news_proxies": ["ES=F", "SPY", "IVV", "VOO"],
            "candidates": ["NQ=F", "RTY=F", "^VIX", "HYG", "JNK"] 
        },
        "NQ (Nasdaq)": {
            "primary": "NQ=F", "price_fallback": "QQQ", 
            "news_proxies": ["NQ=F", "QQQ", "TQQQ", "AAPL", "MSFT"],
            "candidates": ["ES=F", "XLK", "SMH", "ARKK"] 
        },
        "RTY (Russell 2000)": {
            "primary": "RTY=F", "price_fallback": "IWM", 
            "news_proxies": ["RTY=F", "IWM", "TNA"],
            "candidates": ["ES=F", "MDY", "KRE"] 
        },
        "BTC (Bitcoin)": {
            "primary": "BTC-USD", "price_fallback": "BITO", 
            "news_proxies": ["BTC-USD", "BITO", "COIN", "MSTR"],
            "candidates": ["ETH-USD", "COIN", "MSTR", "NQ=F", "GLD"] 
        },
        "ETH (Ethereum)": {
            "primary": "ETH-USD", "price_fallback": "ETHE", 
            "news_proxies": ["ETH-USD", "ETHE", "COIN"],
            "candidates": ["BTC-USD", "SOL-USD", "AVAX-USD"]
        }
    }
    return mapping.get(asset_name, {"primary": "SPY", "price_fallback": "SPY", "news_proxies": ["SPY"], "candidates": ["QQQ"]})

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

@st.cache_data
def find_best_pair(primary_df, candidates, start_date, end_date):
    best_ticker = None
    best_corr = -1
    best_df = None
    primary_rets = primary_df['LogReturn']
    
    for cand in candidates:
        try:
            df_cand = yf.download(cand, start=start_date, end=end_date, progress=False, auto_adjust=False)
            if isinstance(df_cand.columns, pd.MultiIndex):
                df_cand.columns = df_cand.columns.get_level_values(0)
            if 'Adj Close' in df_cand.columns:
                df_cand['LogReturn'] = np.log(df_cand['Adj Close'] / df_cand['Adj Close'].shift(1))
            elif 'Close' in df_cand.columns:
                 df_cand['LogReturn'] = np.log(df_cand['Close'] / df_cand['Close'].shift(1))
            else:
                continue
            df_cand = df_cand.dropna()
            common_idx = primary_rets.index.intersection(df_cand.index)
            if len(common_idx) < 50: continue 
            corr = primary_rets.loc[common_idx].corr(df_cand.loc[common_idx]['LogReturn'])
            abs_corr = abs(corr)
            if abs_corr > best_corr:
                best_corr = abs_corr
                best_ticker = cand
                best_df = df_cand
                real_corr = corr
        except:
            continue
    return best_ticker, best_df, best_corr, real_corr if best_ticker else 0

def fetch_vix(start_date, end_date):
    try:
        vix = yf.download("^VIX", start=start_date, end=end_date, progress=False, auto_adjust=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        return vix['Close'].rename("VIX")
    except:
        return None

def get_live_news_sentiment(news_proxies):
    sia = SentimentIntensityAnalyzer()
    def fetch_valid_news(t):
        try:
            raw_news = yf.Ticker(t).news
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
            
    if not found_news:
        return [], 0.0, None

    scored_news = []
    total_score = 0
    for item in found_news:
        title = item.get('title', 'No Title')
        score = sia.polarity_scores(title)['compound']
        publisher = item.get('publisher', 'Unknown')
        link = item.get('link', '#')
        scored_news.append({'title': title, 'score': score, 'link': link, 'publisher': publisher})
        total_score += score
    avg_score = total_score / len(found_news) if found_news else 0
    return scored_news, avg_score, source_used

def get_technical_sentiment(last_row):
    score = 0
    reasons = []
    if last_row['RSI'] > 50: 
        score += 0.3
        reasons.append("RSI Bullish (>50)")
    else:
        score -= 0.3
        reasons.append("RSI Bearish (<50)")
    if last_row['Dist_SMA_20'] > 0:
        score += 0.4
        reasons.append("Price above SMA20")
    else:
        score -= 0.4
        reasons.append("Price below SMA20")
    if last_row['MACD'] > 0:
        score += 0.2
        reasons.append("MACD Positive")
    else:
        score -= 0.2
        reasons.append("MACD Negative")
    score = max(min(score, 0.9), -0.9)
    dummy_news = []
    for r in reasons:
        dummy_news.append({'title': r, 'score': 0.5 if "Bullish" in r or "above" in r or "Positive" in r else -0.5, 'link': '#', 'publisher': 'Technical Indicator'})
    return dummy_news, score, "Technical Indicators (Fallback)"

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
        if np.abs(prev_loss - current_loss) < tol: break
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

def run_pro_backtest(price_series, labels, time_indices, target_vol_ann, cost_bps):
    signal_df = pd.DataFrame(index=time_indices)
    signal_df['Regime'] = labels
    full_returns = price_series.pct_change()
    aligned_signals = signal_df.reindex(price_series.index).fillna(method='ffill')
    regime_vols = {}
    unique_regimes = np.unique(labels)
    for r in unique_regimes:
        mask = aligned_signals['Regime'] == r
        regime_vols[r] = full_returns[mask].std() * np.sqrt(252)
    safe_regime = min(regime_vols, key=regime_vols.get)
    def get_position_size(row):
        r = row['Regime']
        if r == safe_regime:
            r_vol = regime_vols[r]
            if r_vol == 0: return 0.0
            size = target_vol_ann / r_vol
            return min(size, 1.5)
        else:
            return 0.0
    aligned_signals['Target_Pos'] = aligned_signals.apply(get_position_size, axis=1)
    aligned_signals['Position'] = aligned_signals['Target_Pos'].shift(1).fillna(0)
    aligned_signals['Asset_Ret'] = full_returns
    aligned_signals['Gross_Ret'] = aligned_signals['Position'] * aligned_signals['Asset_Ret']
    aligned_signals['Pos_Change'] = aligned_signals['Position'].diff().abs()
    aligned_signals['Cost'] = aligned_signals['Pos_Change'] * (cost_bps / 10000)
    aligned_signals['Net_Ret'] = aligned_signals['Gross_Ret'] - aligned_signals['Cost']
    aligned_signals['Strategy_Equity'] = (1 + aligned_signals['Net_Ret'].fillna(0)).cumprod()
    aligned_signals['BuyHold_Equity'] = (1 + aligned_signals['Asset_Ret'].fillna(0)).cumprod()
    return aligned_signals, safe_regime, regime_vols

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

st.title("Wasserstein Quant Terminal")
st.markdown("**Distributional Market Regime & Arbitrage Analysis**")

# Sidebar
st.sidebar.header("Data Parameters")
asset_select = st.sidebar.selectbox("Asset", ["Gold", "ES (S&P 500)", "NQ (Nasdaq)", "BTC (Bitcoin)", "EURUSD", "Silver", "RTY (Russell 2000)", "GBPUSD"])
config = get_asset_config(asset_select)
primary_t = config['primary']
fallback_t = config['price_fallback']
news_proxies = config['news_proxies']
candidate_list = config['candidates']

st.sidebar.divider()
st.sidebar.header("Backtest Configuration")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
k_clusters = st.sidebar.slider("Regimes (k)", 2, 5, 2)
window_size = st.sidebar.slider("Window Size (days)", 20, 100, 50)
target_vol = st.sidebar.number_input("Target Volatility (%)", value=10.0, step=1.0) / 100.0
txn_cost = st.sidebar.number_input("Trans. Cost (bps)", value=5.0, step=1.0)

if st.button("RUN ANALYSIS", type="primary", use_container_width=True):
    with st.spinner("Fetching Market Data..."):
        df, ticker_used = fetch_data(primary_t, fallback_t, start_date, end_date)
        vix_df = fetch_vix(start_date, end_date)
        st.toast(f"Correlating pair candidates...")
        best_pair_ticker, df_pair, best_corr, real_corr_val = find_best_pair(df, candidate_list, start_date, end_date)
        
        if df is not None and len(df) > window_size:
            # 1. Regimes
            segments, time_indices = lift_data(df['LogReturn'], window_size, 5) # Default step 5
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

            # 3. Sentiment
            news_items, news_score, news_source = get_live_news_sentiment(news_proxies)
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

            # Dashboard Header
            last_regime = int(main_df['Regime'].iloc[-1])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Regime", f"Regime {last_regime}", help="Current market distributional state")
            c2.metric("ML Forecast Acc.", f"{acc:.1%}", help="Out-of-sample directional prediction accuracy")
            c3.metric(f"Sentiment ({news_source})", f"{news_score:.2f}", delta="Bullish" if news_score > 0.05 else "Bearish" if news_score < -0.05 else "Neutral")
            pair_label = f"{best_pair_ticker}" if best_pair_ticker else "N/A"
            c4.metric(f"Paired Asset", pair_label, delta=f"Corr: {real_corr_val:.2f}" if best_pair_ticker else None)
            
            st.markdown("---")
            
            # --- TABS ---
            t1, t2, t3, t4, t5, t6, t7 = st.tabs([
                "REGIME ID", "ML FORECAST", "PAIRS ARB", "SENTIMENT", "RISK PROFILE", "BACKTEST", "BENCHMARK"
            ])

            # 1. Regimes
            with t1:
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(main_df.index, main_df['Adj Close'], color='#BDBDBD', alpha=0.3, lw=1)
                for i in range(k_clusters):
                    subset = main_df[main_df['Regime'] == i]
                    ax.scatter(subset.index, subset['Adj Close'], color=QUANT_PALETTE[i], s=10, label=f'Regime {i}', zorder=5)
                ax.legend(frameon=True, facecolor='white', framealpha=1)
                ax.set_title(f"{asset_select} Price Path by Regime", fontsize=10, fontweight='bold', color=STRATEGY_COLOR)
                st.pyplot(fig)

            # 2. Forecast
            with t2:
                c_a, c_b = st.columns([2, 1])
                with c_a:
                    strat_ret = np.sign(y_pred) * y_true
                    cum_strat = (1 + strat_ret).cumprod()
                    cum_bh = (1 + y_true).cumprod()
                    fig_f, ax_f = plt.subplots(figsize=(10, 5))
                    ax_f.plot(test_dates, cum_bh, color=BENCHMARK_COLOR, alpha=0.5, ls='--', label='Buy & Hold')
                    ax_f.plot(test_dates, cum_strat, color=STRATEGY_COLOR, lw=2, label='ML Strategy')
                    ax_f.legend(frameon=True, facecolor='white')
                    ax_f.set_title("ML Directional Strategy (Out-of-Sample)", fontsize=10, fontweight='bold', color=STRATEGY_COLOR)
                    st.pyplot(fig_f)
                with c_b:
                    imps = pd.DataFrame({'Feature': feature_cols, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
                    fig_i, ax_i = plt.subplots(figsize=(4, 6))
                    sns.barplot(x='Importance', y='Feature', data=imps.head(10), ax=ax_i, palette="Blues_r")
                    ax_i.set_title("Feature Importance", fontsize=10, fontweight='bold')
                    st.pyplot(fig_i)

            # 3. Pairs
            with t3:
                if dist_series is not None:
                    z_score = (dist_series - dist_series.mean()) / dist_series.std()
                    fig_p, ax_p = plt.subplots(figsize=(12, 5))
                    ax_p.plot(dist_series.index, z_score, color=QUANT_PALETTE[2], lw=1.5, label='Wasserstein Z-Score')
                    ax_p.axhline(2, color=NEGATIVE_COLOR, ls='--', alpha=0.6, label='Sell Threshold')
                    ax_p.axhline(-2, color=POSITIVE_COLOR, ls='--', alpha=0.6, label='Buy Threshold')
                    ax_p.legend(frameon=True, facecolor='white')
                    ax_p.set_title(f"Statistical Arbitrage: {asset_select} vs {best_pair_ticker}", fontsize=10, fontweight='bold', color=STRATEGY_COLOR)
                    st.pyplot(fig_p)
                else:
                    st.warning("No correlation data available.")

            # 4. Sentiment
            with t4:
                st.markdown(f"**News Source: {news_source}**")
                for item in news_items[:5]:
                    color = POSITIVE_COLOR if item['score'] > 0.05 else NEGATIVE_COLOR if item['score'] < -0.05 else BENCHMARK_COLOR
                    st.markdown(f"<span style='color:{color}'>●</span> **{item['score']:.2f}** {item['title']}", unsafe_allow_html=True)

            # 5. Risk
            with t5:
                c_1, c_2 = st.columns(2)
                with c_1:
                    w_means = np.mean(segments, axis=1) * 252
                    w_vols = np.std(segments, axis=1) * np.sqrt(252)
                    fig_mv, ax_mv = plt.subplots(figsize=(6, 5))
                    for i in range(k_clusters):
                        m = labels == i
                        ax_mv.scatter(w_vols[m], w_means[m], color=QUANT_PALETTE[i], s=15, alpha=0.7, label=f'Regime {i}')
                    ax_mv.set_xlabel("Annualized Volatility")
                    ax_mv.set_ylabel("Annualized Return")
                    ax_mv.legend(frameon=True, facecolor='white')
                    st.pyplot(fig_mv)
                with c_2:
                    fig_ad, ax_ad = plt.subplots(figsize=(6, 5))
                    ax_ad.plot(main_df.index, main_df['Anomaly_Score'], color=NEGATIVE_COLOR, lw=1)
                    ax_ad.set_title("Market fragility (Anomaly Score)", fontsize=10)
                    st.pyplot(fig_ad)

            # 6. Backtest
            with t6:
                backtest_df, safe_regime, r_vols = run_pro_backtest(main_df['Adj Close'], main_df['Regime'], main_df.index, target_vol, txn_cost)
                tot_ret = backtest_df['Strategy_Equity'].iloc[-1] - 1
                bh_ret = backtest_df['BuyHold_Equity'].iloc[-1] - 1
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Strategy Return", f"{tot_ret:.2%}", delta=f"{tot_ret-bh_ret:.2%}")
                col_b.metric("Safe Regime Vol", f"{r_vols[safe_regime]:.1%}")
                col_c.metric("Avg Leverage", f"{backtest_df['Position'].mean():.2f}x")

                fig_bt, ax_bt = plt.subplots(figsize=(12, 5))
                ax_bt.plot(backtest_df.index, backtest_df['Strategy_Equity'], color=STRATEGY_COLOR, lw=2, label="Vol-Targeted Strategy")
                ax_bt.plot(backtest_df.index, backtest_df['BuyHold_Equity'], color=BENCHMARK_COLOR, alpha=0.5, ls='--', label="Buy & Hold")
                
                ax2 = ax_bt.twinx()
                ax2.fill_between(backtest_df.index, backtest_df['Position'], color=QUANT_PALETTE[3], alpha=0.15)
                ax2.set_ylabel("Leverage", color=QUANT_PALETTE[1])
                
                ax_bt.legend(loc='upper left', frameon=True, facecolor='white')
                ax_bt.set_title("Backtest Performance (Net of Costs)", fontsize=10, fontweight='bold', color=STRATEGY_COLOR)
                st.pyplot(fig_bt)

            # 7. Benchmark
            with t7:
                blabels, bindices = moment_kmeans_clustering(df['LogReturn'], window_size, 5, k_clusters)
                bdf = pd.DataFrame({'Cluster': blabels}, index=bindices).join(df[['Adj Close']])
                fig_b, ax_b = plt.subplots(figsize=(12, 5))
                ax_b.plot(bdf.index, bdf['Adj Close'], color='#BDBDBD', alpha=0.3)
                for i in range(k_clusters):
                    s = bdf[bdf['Cluster'] == i]
                    ax_b.scatter(s.index, s['Adj Close'], color=['orange','purple','brown','cyan'][i%4], s=10, alpha=0.6)
                st.pyplot(fig_b)

        else:
            st.warning("Insufficient data.")
else:
    st.info("Set parameters and click RUN ANALYSIS.")
