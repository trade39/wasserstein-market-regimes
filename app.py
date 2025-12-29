import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Configuration & Styles ---
st.set_page_config(page_title="Wasserstein Market Regimes", layout="wide")

# Define a sharp, high-contrast palette (Cyan, Magenta, Lime, Amber, Purple)
SHARP_PALETTE = ['#00E5FF', '#FF4081', '#00E676', '#FFC400', '#651FFF']
BACKGROUND_COLOR = '#0E1117' # Streamlit Dark BG

# Configure Global Matplotlib Theme for "Sleek" look
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
    
    /* Sleek Tabs */
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

# --- Main App ---

st.title("Wasserstein Market Regimes")
st.markdown("Distributional Clustering of Financial Time Series")

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

if st.button("RUN MODEL", type="primary", use_container_width=True):
    with st.spinner("Calculating Wasserstein distances..."):
        df, ticker_used = fetch_data(primary_t, fallback_t, start_date, end_date)
        
        if df is not None and len(df) > window_size:
            # Calculation
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

            # --- Header Metrics ---
            last_regime = int(res['Regime'].iloc[-1])
            last_dist = res['Anomaly_Score'].iloc[-1]
            avg_dist = np.mean(anomaly_scores)
            
            # Compute stats for display
            regime_stats = []
            for i in range(k_clusters):
                idx = np.where(labels == i)[0]
                flat_rets = segments[idx].flatten()
                regime_stats.append({
                    "vol": np.std(flat_rets) * np.sqrt(252),
                    "ret": np.mean(flat_rets) * 252
                })
            current_vol = regime_stats[last_regime]['vol']
            
            # Color assignment for metrics
            metric_color = SHARP_PALETTE[last_regime % len(SHARP_PALETTE)]
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Regime", f"Regime {last_regime}", delta=None)
            col2.metric("Regime Volatility", f"{current_vol:.1%}")
            col3.metric("Anomaly Score", f"{last_dist:.4f}", delta=f"{(last_dist - avg_dist)*-1:.4f}", delta_color="normal")
            col4.metric("Ticker", ticker_used)
            
            st.markdown("---")

            # --- TABS ---
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "PRICE & REGIMES", 
                "SHAPE ANALYSIS", 
                "TRANSITIONS", 
                "STRATEGY", 
                "BENCHMARK"
            ])

            # 1. Price Path
            with tab1:
                fig, ax = plt.subplots(figsize=(12, 5))
                # Plot Base Line (thin, sleek)
                ax.plot(main_df.index, main_df['Adj Close'], color='#FFFFFF', alpha=0.15, lw=1)
                
                # Plot Colored Regimes
                for i in range(k_clusters):
                    subset = main_df[main_df['Regime'] == i]
                    # Usescatter for sharp points
                    ax.scatter(subset.index, subset['Adj Close'], 
                               color=SHARP_PALETTE[i], s=8, label=f'Regime {i}', zorder=5, alpha=0.9)
                
                ax.set_title(f"{asset_select} | Regime Segmentation", fontsize=10, loc='left', color='#888')
                ax.legend(frameon=False, fontsize=9, loc='upper left')
                ax.grid(True, which='major', alpha=0.1)
                st.pyplot(fig)
                
                # Summary Stats
                st.markdown("###### Regime Characteristics")
                stats_df = pd.DataFrame(regime_stats)
                stats_df.index.name = "Regime"
                stats_df.columns = ["Ann. Volatility", "Ann. Return"]
                
                # Style the table
                st.dataframe(
                    stats_df.style.format("{:.2%}")
                    .background_gradient(cmap='Greys', subset=['Ann. Volatility'])
                )

            # 2. Shape Analysis
            with tab2:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Risk-Return Profile**")
                    window_means = np.mean(segments, axis=1) * 252
                    window_vols = np.std(segments, axis=1) * np.sqrt(252)
                    
                    fig_mv, ax_mv = plt.subplots(figsize=(6, 5))
                    for i in range(k_clusters):
                        mask = labels == i
                        ax_mv.scatter(window_vols[mask], window_means[mask], 
                                      color=SHARP_PALETTE[i], s=20, alpha=0.6, 
                                      edgecolor='none', label=f'Regime {i}')
                    ax_mv.set_xlabel("Volatility", fontsize=8, color='#888')
                    ax_mv.set_ylabel("Return", fontsize=8, color='#888')
                    ax_mv.legend(frameon=False, fontsize=8)
                    st.pyplot(fig_mv)
                    
                with c2:
                    st.markdown("**Distribution Tails (Skewness)**")
                    fig_skew, ax_skew = plt.subplots(figsize=(6, 5))
                    for i in range(k_clusters):
                        cluster_skews = [skew(seg) for seg in segments[labels == i]]
                        sns.kdeplot(cluster_skews, ax=ax_skew, color=SHARP_PALETTE[i], 
                                    fill=True, alpha=0.2, linewidth=1.5)
                    ax_skew.set_xlabel("Window Skewness", fontsize=8, color='#888')
                    ax_skew.set_ylabel("Density", fontsize=8, color='#888')
                    ax_skew.grid(False)
                    st.pyplot(fig_skew)

            # 3. Transitions
            with tab3:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Transition Probabilities**")
                    trans_mat = calculate_transition_matrix(labels)
                    fig_hm, ax_hm = plt.subplots(figsize=(5,4))
                    # Use a dark, sleek heatmap
                    sns.heatmap(trans_mat, annot=True, fmt=".2f", 
                                cmap=sns.dark_palette(SHARP_PALETTE[0], as_cmap=True), 
                                cbar=False, ax=ax_hm, square=True)
                    ax_hm.set_ylabel("From Regime", color='#888')
                    ax_hm.set_xlabel("To Regime", color='#888')
                    st.pyplot(fig_hm)
                    
                with c2:
                    st.markdown("**Regime Stability (Distance to Center)**")
                    fig_ad, ax_ad = plt.subplots(figsize=(6, 4))
                    ax_ad.plot(res.index, res['Anomaly_Score'], color=SHARP_PALETTE[1], lw=0.8)
                    ax_ad.fill_between(res.index, res['Anomaly_Score'], 0, color=SHARP_PALETTE[1], alpha=0.1)
                    ax_ad.set_title("Wasserstein Distance Anomaly Score", fontsize=8, color='#888', loc='left')
                    ax_ad.grid(False)
                    st.pyplot(fig_ad)

            # 4. Strategy
            with tab4:
                backtest_df, safe_regime = run_backtest(main_df['Adj Close'], main_df['Regime'], main_df.index)
                
                strat_ret = backtest_df['Strategy_Equity'].iloc[-1] - 1
                bh_ret = backtest_df['BuyHold_Equity'].iloc[-1] - 1
                
                col_a, col_b = st.columns(2)
                col_a.metric("Strategy Return", f"{strat_ret:.2%}", delta=f"{strat_ret-bh_ret:.2%}")
                col_b.caption(f"Strategy: Long Regime {safe_regime} (Low Vol), Cash otherwise.")

                fig_bt, ax_bt = plt.subplots(figsize=(12, 5))
                ax_bt.plot(backtest_df.index, backtest_df['BuyHold_Equity'], color='#FFFFFF', alpha=0.3, linestyle='--', label='Buy & Hold')
                ax_bt.plot(backtest_df.index, backtest_df['Strategy_Equity'], color=SHARP_PALETTE[2], lw=1.5, label='Regime Strategy')
                ax_bt.legend(frameon=False)
                ax_bt.set_title("Cumulative Equity Curve", fontsize=10, color='#888', loc='left')
                st.pyplot(fig_bt)

            # 5. Benchmark
            with tab5:
                bench_labels, bench_indices = moment_kmeans_clustering(df['LogReturn'], window_size, step_size, k_clusters)
                bench_df = pd.DataFrame(index=bench_indices)
                bench_df['Bench_Cluster'] = bench_labels
                plot_df = df.join(bench_df).dropna()
                
                fig_b, ax_b = plt.subplots(figsize=(12, 5))
                ax_b.plot(plot_df.index, plot_df['Adj Close'], color='#FFFFFF', alpha=0.1, lw=1)
                
                # Use secondary palette for benchmark
                BENCH_PALETTE = ['#FF9100', '#00B0FF', '#D500F9', '#FF1744']
                for i in range(k_clusters):
                    subset = plot_df[plot_df['Bench_Cluster'] == i]
                    ax_b.scatter(subset.index, subset['Adj Close'], color=BENCH_PALETTE[i], s=8, alpha=0.7, label=f'Moment {i}')
                    
                ax_b.set_title("Benchmark: Standard Moment K-Means", fontsize=10, color='#888', loc='left')
                ax_b.legend(frameon=False)
                st.pyplot(fig_b)
                st.caption("Moment K-Means uses Euclidean distance on [Mean, Std, Skew, Kurtosis]. It often fails to separate regimes as cleanly as Optimal Transport.")

        else:
            st.warning("Insufficient data. Please adjust the window size or date range.")
else:
    st.info("Configure parameters on the left and click RUN MODEL.")
