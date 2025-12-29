import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration & Styles ---
st.set_page_config(page_title="Wasserstein Market Regimes", layout="wide")

# Apply Dark Background for Matplotlib globally
plt.style.use('dark_background')

# Custom CSS for Streamlit
st.markdown("""
<style>
    .reportview-container { margin-top: -2em; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    div[data-testid="stImage"] {background-color: transparent;}
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def get_ticker_and_fallback(asset_name):
    """
    Returns a primary ticker and a safer fallback (ETF/Index) 
    in case the primary (usually Futures) fails.
    """
    mapping = {
        "Gold": ("GC=F", "GLD"),                 # Future -> ETF
        "EURUSD": ("EURUSD=X", "FXE"),           # Forex -> ETF
        "ES (S&P 500 E-mini)": ("ES=F", "SPY"),  # Future -> ETF
        "NQ (Nasdaq 100 E-mini)": ("NQ=F", "QQQ"), # Future -> ETF
        "BTC (Bitcoin)": ("BTC-USD", "BITO")     # Crypto -> ETF
    }
    return mapping.get(asset_name, ("SPY", "SPY"))

@st.cache_data
def fetch_data(primary_ticker, fallback_ticker, start_date, end_date):
    """
    Robust data fetching. Tries primary ticker first; if empty, uses fallback.
    """
    def download_safe(t):
        try:
            # force auto_adjust=False to ensure we get 'Adj Close' or 'Close' columns consistently
            d = yf.download(t, start=start_date, end=end_date, progress=False, auto_adjust=False)
            return d
        except Exception:
            return pd.DataFrame()

    # 1. Try Primary
    data = download_safe(primary_ticker)
    
    # 2. Check if valid. If not, try Fallback.
    used_ticker = primary_ticker
    if data.empty:
        st.warning(f"Could not fetch data for {primary_ticker} (likely a Yahoo Finance futures issue). Switching to fallback: {fallback_ticker}...")
        data = download_safe(fallback_ticker)
        used_ticker = fallback_ticker

    if data.empty:
        st.error(f"No data returned for {primary_ticker} or {fallback_ticker}. Please check the date range.")
        return None, None

    # 3. Handle MultiIndex columns (New yfinance versions)
    if isinstance(data.columns, pd.MultiIndex):
        try:
            # Try to flatten if the level 1 is the ticker name
            if used_ticker in data.columns.get_level_values(1):
                 data = data.xs(used_ticker, axis=1, level=1)
            else:
                 # Just take the top level if structure is unexpected
                 data.columns = data.columns.get_level_values(0)
        except Exception:
             # Last resort flatten
             data.columns = data.columns.get_level_values(0)

    # 4. Standardize Price Column
    if 'Adj Close' in data.columns:
        price_col = 'Adj Close'
    elif 'Close' in data.columns:
        price_col = 'Close'
    else:
        st.error(f"Missing price columns. Available: {data.columns}")
        return None, None

    data = data.copy()
    if price_col != 'Adj Close':
        data['Adj Close'] = data[price_col]

    # 5. Log Returns
    data['LogReturn'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    data = data.dropna()
    
    return data, used_ticker

def lift_data(log_returns, h1, h2):
    """Lifts the stream of returns into a stream of segments."""
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
    
    for iteration in range(max_iter):
        distances = np.zeros((n_segments, k))
        for j in range(k):
            diff = np.abs(segments - centroids[j]) 
            distances[:, j] = np.mean(diff, axis=1)
            
        labels = np.argmin(distances, axis=1)
        current_loss = np.sum(np.min(distances, axis=1))
        
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
        
    return labels, centroids

def generate_context_summary(stats_df):
    summary = []
    max_vol_regime = stats_df['Volatility (Ann.)'].idxmax()
    min_vol_regime = stats_df['Volatility (Ann.)'].idxmin()

    summary.append(f"**Regime {max_vol_regime} (High Volatility):** Exhibits highest annualized volatility ({stats_df.loc[max_vol_regime, 'Volatility (Ann.)']:.2%}). "
                   f"Often resembles crisis or correction behavior.")
    
    if max_vol_regime != min_vol_regime:
        summary.append(f"**Regime {min_vol_regime} (Low Volatility):** More stable with lower volatility ({stats_df.loc[min_vol_regime, 'Volatility (Ann.)']:.2%}). "
                       f"Typically represents calm or trending periods.")
    return summary

# --- Main App Interface ---

st.title("Clustering Market Regimes using Wasserstein Distance")
st.markdown("""
[cite_start]This app applies **Wasserstein k-means** to cluster market behavior[cite: 7]. 
[cite_start]It groups time periods by the **geometry of their return distribution**, capturing shifts in volatility and tail risk [cite: 30-32].
""")

# Sidebar
st.sidebar.header("Configuration")
asset_select = st.sidebar.selectbox("Select Asset", ["Gold", "EURUSD", "ES (S&P 500 E-mini)", "NQ (Nasdaq 100 E-mini)", "BTC (Bitcoin)"])
primary_t, fallback_t = get_ticker_and_fallback(asset_select)

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = col2.date_input("End Date", pd.to_datetime("today"))

st.sidebar.subheader("WK-Means Hyperparameters")
k_clusters = st.sidebar.slider("Number of Regimes (k)", 2, 5, 2)
window_size = st.sidebar.slider("Window Size (h1)", 20, 100, 50, help="Days in each distribution segment.")
step_size = st.sidebar.slider("Step Size (h2)", 1, 20, 5, help="Days to slide forward.")

# Execution
if st.button("Run Clustering"):
    with st.spinner(f"Fetching data for {asset_select}..."):
        df, actual_ticker = fetch_data(primary_t, fallback_t, start_date, end_date)
        
        if df is not None and len(df) > window_size:
            if actual_ticker != primary_t:
                st.caption(f"Note: Using data from **{actual_ticker}** as proxy.")
                
            segments_sorted, time_indices = lift_data(df['LogReturn'], window_size, step_size)
            
            if len(segments_sorted) < k_clusters:
                st.error("Not enough data points. Try increasing the date range or decreasing the window size.")
            else:
                labels, centroids = wk_means_clustering(segments_sorted, k_clusters)
                
                results_df = pd.DataFrame(index=time_indices)
                results_df['Cluster'] = labels
                merged_df = df.join(results_df)
                merged_df['Cluster'] = merged_df['Cluster'].fillna(method='ffill')
                merged_df = merged_df.dropna(subset=['Cluster'])
                merged_df['Cluster'] = merged_df['Cluster'].astype(int)

                # --- Statistics ---
                stats_data = []
                for i in range(k_clusters):
                    cluster_indices = np.where(labels == i)[0]
                    if len(cluster_indices) > 0:
                        all_returns = segments_sorted[cluster_indices].flatten()
                        vol = np.std(all_returns) * np.sqrt(252)
                        mu = np.mean(all_returns) * 252
                        stats_data.append({
                            "Regime": i,
                            "Count": len(cluster_indices),
                            "Mean Return (Ann.)": mu,
                            "Volatility (Ann.)": vol
                        })
                stats_df_numeric = pd.DataFrame(stats_data).set_index("Regime")

                # --- Visualizations ---
                
                # 1. Price Path
                st.subheader(f"{asset_select} ({actual_ticker}) Price Path by Regime")
                fig, ax = plt.subplots(figsize=(12, 6))
                fig.patch.set_facecolor('#0E1117') 
                ax.set_facecolor('#0E1117')
                
                ax.plot(merged_df.index, merged_df['Adj Close'], color='#444444', alpha=0.5, label='Price', linewidth=1)
                
                colors = plt.cm.plasma(np.linspace(0, 1, k_clusters))
                
                for cluster_id in range(k_clusters):
                    cluster_data = merged_df[merged_df['Cluster'] == cluster_id]
                    ax.scatter(cluster_data.index, cluster_data['Adj Close'], 
                               color=colors[cluster_id], s=15, label=f'Regime {cluster_id}', zorder=10)
                
                ax.set_title("Identified Market Regimes", color='white', fontsize=14)
                ax.set_ylabel("Price", color='white')
                ax.tick_params(colors='white')
                ax.grid(True, color='#333333', linestyle='--', alpha=0.5)
                
                leg = ax.legend(facecolor='#0E1117', edgecolor='white')
                for text in leg.get_texts():
                    text.set_color("white")
                st.pyplot(fig)

                # 2. Context Summary
                st.subheader("Context Summary")
                summary_text = generate_context_summary(stats_df_numeric)
                for line in summary_text:
                    st.info(line)

                # 3. Distributions
                st.subheader("Regime Return Distributions (Centroids)")
                col_chart, col_stats = st.columns([2, 1])
                
                with col_chart:
                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    fig2.patch.set_facecolor('#0E1117')
                    ax2.set_facecolor('#0E1117')
                    
                    for i in range(k_clusters):
                        sns.kdeplot(centroids[i], ax=ax2, color=colors[i], label=f'Regime {i}', fill=True, alpha=0.2, linewidth=2)
                        
                    ax2.set_title("Distribution Shape (Barycenters)", color='white')
                    ax2.set_xlabel("Log Return", color='white')
                    ax2.tick_params(colors='white')
                    ax2.grid(True, color='#333333', linestyle='--', alpha=0.5)
                    leg2 = ax2.legend(facecolor='#0E1117', edgecolor='white')
                    for text in leg2.get_texts():
                        text.set_color("white")
                    st.pyplot(fig2)

                # 4. Stats Table
                with col_stats:
                    st.markdown("##### Detailed Statistics")
                    display_stats = stats_df_numeric.copy()
                    display_stats['Mean Return (Ann.)'] = display_stats['Mean Return (Ann.)'].apply(lambda x: f"{x:.2%}")
                    display_stats['Volatility (Ann.)'] = display_stats['Volatility (Ann.)'].apply(lambda x: f"{x:.2%}")
                    st.dataframe(display_stats)

        else:
            st.warning("No data found. Try expanding the date range.")

else:
    st.info("Select parameters and click 'Run Clustering' to start.")
