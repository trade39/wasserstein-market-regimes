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
plt.style.use('dark_background')

st.markdown("""
<style>
    .reportview-container { margin-top: -2em; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    div[data-testid="stImage"] {background-color: transparent;}
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #0E1117; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #262730; border-bottom: 2px solid #FF4B4B; }
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
    """Lifts returns into sorted segments (quantiles) for Wasserstein distance."""
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
    """Wasserstein K-Means Algorithm."""
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
            # W1 distance for 1D = L1 distance between sorted vectors
            diff = np.abs(segments - centroids[j]) 
            distances[:, j] = np.mean(diff, axis=1)
            
        labels = np.argmin(distances, axis=1)
        # Store min distance for anomaly detection
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
    """Benchmark: Standard K-Means on statistical moments."""
    moments = []
    indices = []
    n = len(log_returns)
    
    for i in range(0, n - h1 + 1, h2):
        window = log_returns.iloc[i : i + h1].values
        # Feature vector: Mean, Std, Skew, Kurtosis
        m_vec = [np.mean(window), np.std(window), skew(window), kurtosis(window)]
        moments.append(m_vec)
        indices.append(log_returns.index[i + h1 - 1])
    
    moments = np.array(moments)
    # Standardize features (crucial for Euclidean distance)
    scaler = StandardScaler()
    moments_scaled = scaler.fit_transform(moments)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(moments_scaled)
    
    return labels, indices

def calculate_transition_matrix(labels):
    """Calculates probability of switching regimes."""
    df = pd.DataFrame({'Current': labels[:-1], 'Next': labels[1:]})
    matrix = pd.crosstab(df['Current'], df['Next'], normalize='index')
    return matrix

def run_backtest(price_series, labels, time_indices):
    """Simple strategy: Long lowest vol regime, Cash otherwise."""
    # Align labels with prices
    # Note: Labels correspond to window ending at time t. 
    # We trade at t+1 based on signal at t.
    
    # Create DataFrame aligned with original indices
    signal_df = pd.DataFrame(index=time_indices)
    signal_df['Regime'] = labels
    
    # Calculate volatility of each regime to identify "Safe" vs "Risky"
    # We need the original full price series to calculate daily returns
    full_returns = price_series.pct_change()
    
    # Reindex signal to full series (forward fill for step size gaps)
    aligned_signals = signal_df.reindex(price_series.index).fillna(method='ffill')
    
    # Determine which regime is "Low Vol"
    regime_vols = {}
    unique_regimes = np.unique(labels)
    
    for r in unique_regimes:
        # Get returns where regime is r
        mask = aligned_signals['Regime'] == r
        regime_vols[r] = full_returns[mask].std()
        
    safe_regime = min(regime_vols, key=regime_vols.get)
    
    # Strategy: 1 if Safe Regime, 0 if Risky
    # Shift by 1 to avoid lookahead bias (trade tomorrow based on today's regime)
    aligned_signals['Position'] = np.where(aligned_signals['Regime'] == safe_regime, 1, 0)
    aligned_signals['Position'] = aligned_signals['Position'].shift(1)
    
    aligned_signals['Strategy_Return'] = aligned_signals['Position'] * full_returns
    aligned_signals['BuyHold_Return'] = full_returns
    
    # Cumulative Returns
    aligned_signals['Strategy_Equity'] = (1 + aligned_signals['Strategy_Return'].fillna(0)).cumprod()
    aligned_signals['BuyHold_Equity'] = (1 + aligned_signals['BuyHold_Return'].fillna(0)).cumprod()
    
    return aligned_signals, safe_regime

# --- Main App ---

st.title("Wasserstein Market Regimes Dashboard")
st.markdown("Advanced clustering of financial time series using Optimal Transport (Wasserstein Distance).")

# Sidebar
st.sidebar.header("Configuration")
asset_select = st.sidebar.selectbox("Select Asset", ["Gold", "EURUSD", "ES (S&P 500 E-mini)", "NQ (Nasdaq 100 E-mini)", "BTC (Bitcoin)"])
primary_t, fallback_t = get_ticker_and_fallback(asset_select)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

st.sidebar.subheader("WK-Means Parameters")
k_clusters = st.sidebar.slider("Regimes (k)", 2, 5, 2)
window_size = st.sidebar.slider("Window (h1)", 20, 100, 50)
step_size = st.sidebar.slider("Step (h2)", 1, 20, 5)

if st.button("Run Analysis", type="primary"):
    with st.spinner("Processing..."):
        df, ticker_used = fetch_data(primary_t, fallback_t, start_date, end_date)
        
        if df is not None and len(df) > window_size:
            # 1. Lift Data & Run WK-Means
            segments, time_indices = lift_data(df['LogReturn'], window_size, step_size)
            labels, centroids, anomaly_scores = wk_means_clustering(segments, k_clusters)
            
            # DataFrame Construction
            res = pd.DataFrame(index=time_indices)
            res['Regime'] = labels
            res['Anomaly_Score'] = anomaly_scores
            
            # Merge for plotting
            main_df = df.join(res)
            main_df['Regime'] = main_df['Regime'].fillna(method='ffill')
            main_df['Anomaly_Score'] = main_df['Anomaly_Score'].fillna(method='ffill')
            main_df = main_df.dropna(subset=['Regime'])
            main_df['Regime'] = main_df['Regime'].astype(int)

            # --- Dashboard Header (Real-Time State) ---
            last_regime = int(res['Regime'].iloc[-1])
            last_dist = res['Anomaly_Score'].iloc[-1]
            
            # Determine Regime Stats
            regime_stats = []
            for i in range(k_clusters):
                idx = np.where(labels == i)[0]
                flat_rets = segments[idx].flatten()
                regime_stats.append({
                    "vol": np.std(flat_rets) * np.sqrt(252),
                    "ret": np.mean(flat_rets) * 252,
                    "skew": skew(flat_rets)
                })
            
            current_vol = regime_stats[last_regime]['vol']
            avg_dist = np.mean(anomaly_scores)
            
            st.markdown("### Current Market State")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Regime", f"Regime {last_regime}", delta=None)
            c2.metric("Regime Volatility", f"{current_vol:.1%}", help="Annualized Volatility of current regime")
            c3.metric("Anomaly Score", f"{last_dist:.4f}", delta=f"{last_dist - avg_dist:.4f}", delta_color="inverse", help="Distance to Regime Centroid. Higher = Abnormal behavior.")
            c4.metric("Ticker Used", ticker_used)
            
            st.markdown("---")

            # --- Tabs for Analysis ---
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Price & Regimes", 
                "🔍 Validation & Shape", 
                "⚠️ Risk & Transitions", 
                "💰 Backtest Strategy", 
                "⚖️ Benchmark (Moment K-Means)"
            ])

            colors = plt.cm.plasma(np.linspace(0, 1, k_clusters))

            # TAB 1: Price Path
            with tab1:
                st.subheader("Regime Identification")
                fig, ax = plt.subplots(figsize=(12, 5))
                fig.patch.set_facecolor('#0E1117')
                ax.set_facecolor('#0E1117')
                ax.plot(main_df.index, main_df['Adj Close'], color='#555555', alpha=0.5, lw=1)
                
                for i in range(k_clusters):
                    subset = main_df[main_df['Regime'] == i]
                    ax.scatter(subset.index, subset['Adj Close'], color=colors[i], s=10, label=f'Regime {i}', zorder=5)
                
                ax.set_title(f"{asset_select} Price Path", color='white')
                ax.legend(facecolor='#0E1117', labelcolor='white')
                ax.grid(True, alpha=0.2)
                ax.tick_params(colors='white')
                st.pyplot(fig)
                
                # Stats Table
                st.caption("Regime Statistics")
                stats_df = pd.DataFrame(regime_stats)
                stats_df.index.name = "Regime"
                stats_df.columns = ["Volatility (Ann.)", "Return (Ann.)", "Skewness"]
                st.dataframe(stats_df.style.format("{:.2%}"))

            # TAB 2: Validation (Mean-Variance & Shapes)
            with tab2:
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown("**1. Mean-Variance Projection**")
                    st.caption("Visual validation: Do clusters separate by risk/return?")
                    
                    # Calculate stats per window for scatter plot
                    window_means = np.mean(segments, axis=1) * 252
                    window_vols = np.std(segments, axis=1) * np.sqrt(252)
                    
                    fig_mv, ax_mv = plt.subplots(figsize=(6, 5))
                    fig_mv.patch.set_facecolor('#0E1117')
                    ax_mv.set_facecolor('#0E1117')
                    
                    for i in range(k_clusters):
                        mask = labels == i
                        ax_mv.scatter(window_vols[mask], window_means[mask], color=colors[i], s=15, alpha=0.6, label=f'Regime {i}')
                    
                    ax_mv.set_xlabel("Window Volatility (Ann.)", color='white')
                    ax_mv.set_ylabel("Window Mean Return (Ann.)", color='white')
                    ax_mv.legend(facecolor='#0E1117', labelcolor='white')
                    ax_mv.tick_params(colors='white')
                    ax_mv.grid(True, alpha=0.2)
                    st.pyplot(fig_mv)

                with col_right:
                    st.markdown("**2. Skewness & Shape**")
                    st.caption("Do regimes exhibit different tail risks (skewness)?")
                    
                    fig_skew, ax_skew = plt.subplots(figsize=(6, 5))
                    fig_skew.patch.set_facecolor('#0E1117')
                    ax_skew.set_facecolor('#0E1117')
                    
                    for i in range(k_clusters):
                        # Plot Skewness distribution of windows in this cluster
                        cluster_skews = [skew(seg) for seg in segments[labels == i]]
                        sns.kdeplot(cluster_skews, ax=ax_skew, color=colors[i], fill=True, alpha=0.3, label=f'Regime {i}')
                    
                    ax_skew.set_xlabel("Window Skewness", color='white')
                    ax_skew.set_title("Distribution of Skewness", color='white')
                    ax_skew.legend(facecolor='#0E1117', labelcolor='white')
                    ax_skew.tick_params(colors='white')
                    st.pyplot(fig_skew)

            # TAB 3: Transitions & Risk
            with tab3:
                c1, c2 = st.columns(2)
                
                with c1:
                    st.markdown("**Transition Matrix**")
                    st.caption("Probability of moving from Regime X to Regime Y.")
                    trans_mat = calculate_transition_matrix(labels)
                    
                    fig_hm, ax_hm = plt.subplots()
                    fig_hm.patch.set_facecolor('#0E1117')
                    sns.heatmap(trans_mat, annot=True, fmt=".2f", cmap="magma", ax=ax_hm, cbar=False)
                    ax_hm.set_ylabel("Current Regime", color='white')
                    ax_hm.set_xlabel("Next Regime", color='white')
                    ax_hm.tick_params(colors='white')
                    st.pyplot(fig_hm)
                    
                with c2:
                    st.markdown("**Anomaly Detection**")
                    st.caption("Wasserstein Distance to Centroid over time. Spikes indicate regime breakdown.")
                    
                    fig_ad, ax_ad = plt.subplots(figsize=(6, 4))
                    fig_ad.patch.set_facecolor('#0E1117')
                    ax_ad.set_facecolor('#0E1117')
                    ax_ad.plot(res.index, res['Anomaly_Score'], color='#FF4B4B', lw=1)
                    ax_ad.set_ylabel("Distance", color='white')
                    ax_ad.tick_params(colors='white')
                    ax_ad.grid(True, alpha=0.2)
                    st.pyplot(fig_ad)

            # TAB 4: Backtest
            with tab4:
                st.subheader("Strategy Backtest")
                st.caption("Strategy: Long 100% in 'Safe' (Low Vol) Regime, Cash (0%) in 'Risky' Regime.")
                
                # Use raw Adjusted Close for backtest to include dividends/splits accurately
                backtest_df, safe_regime = run_backtest(main_df['Adj Close'], main_df['Regime'], main_df.index)
                
                fig_bt, ax_bt = plt.subplots(figsize=(12, 5))
                fig_bt.patch.set_facecolor('#0E1117')
                ax_bt.set_facecolor('#0E1117')
                
                ax_bt.plot(backtest_df.index, backtest_df['BuyHold_Equity'], label='Buy & Hold', color='white', alpha=0.6, linestyle='--')
                ax_bt.plot(backtest_df.index, backtest_df['Strategy_Equity'], label=f'Regime Strategy (Safe={safe_regime})', color='#00FF00', lw=2)
                
                ax_bt.set_title("Equity Curve", color='white')
                ax_bt.set_ylabel("Normalized Value", color='white')
                ax_bt.legend(facecolor='#0E1117', labelcolor='white')
                ax_bt.tick_params(colors='white')
                ax_bt.grid(True, alpha=0.2)
                st.pyplot(fig_bt)
                
                # Metrics
                strat_ret = backtest_df['Strategy_Equity'].iloc[-1] - 1
                bh_ret = backtest_df['BuyHold_Equity'].iloc[-1] - 1
                st.metric("Strategy Total Return", f"{strat_ret:.2%}", delta=f"{strat_ret - bh_ret:.2%}")

            # TAB 5: Benchmark
            with tab5:
                st.markdown("### Benchmark: Moment K-Means")
                st.caption("Comparison with standard K-Means clustering on [Mean, Variance, Skew, Kurtosis].")
                
                # Run Benchmark
                bench_labels, bench_indices = moment_kmeans_clustering(df['LogReturn'], window_size, step_size, k_clusters)
                
                # Align benchmark results
                bench_df = pd.DataFrame(index=bench_indices)
                bench_df['Bench_Cluster'] = bench_labels
                plot_df = df.join(bench_df).dropna()
                
                fig_b, ax_b = plt.subplots(figsize=(12, 5))
                fig_b.patch.set_facecolor('#0E1117')
                ax_b.set_facecolor('#0E1117')
                ax_b.plot(plot_df.index, plot_df['Adj Close'], color='#555555', alpha=0.5)
                
                # Use a different colormap for benchmark to distinguish
                b_colors = plt.cm.cool(np.linspace(0, 1, k_clusters))
                
                for i in range(k_clusters):
                    subset = plot_df[plot_df['Bench_Cluster'] == i]
                    ax_b.scatter(subset.index, subset['Adj Close'], color=b_colors[i], s=10, label=f'Moment-Cluster {i}')
                    
                ax_b.set_title("Moment K-Means Clustering Results", color='white')
                ax_b.legend(facecolor='#0E1117', labelcolor='white')
                ax_b.tick_params(colors='white')
                st.pyplot(fig_b)
                
                st.info("Notice how Moment-based clustering might be more sensitive to outliers or fail to capture subtle regime shifts compared to the Wasserstein approach (compare with Tab 1).")

        else:
            st.error("Not enough data. Increase date range or decrease window size.")
else:
    st.info("Adjust parameters and click **Run Analysis**.")
