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

# Custom CSS for Streamlit to ensure tables and text look good
st.markdown("""
<style>
    .reportview-container { margin-top: -2em; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    /* Force dark background for plots if transparent */
    div[data-testid="stImage"] {background-color: transparent;}
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def get_ticker(asset_name):
    """Map user friendly names to Yahoo Finance Tickers."""
    mapping = {
        "Gold": "GC=F",
        "EURUSD": "EURUSD=X",
        "ES (S&P 500 E-mini)": "ES=F",
        "NQ (Nasdaq 100 E-mini)": "NQ=F",
        "BTC (Bitcoin)": "BTC-USD"
    }
    return mapping.get(asset_name, "SPY")

@st.cache_data
def fetch_data(ticker, start_date, end_date):
    """Fetch closing prices and calculate log returns with robust column handling."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            st.error(f"No data returned for {ticker}. The ticker might be delisted or the date range is invalid.")
            return None
        
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Determine price column
        if 'Adj Close' in data.columns:
            price_col = 'Adj Close'
        elif 'Close' in data.columns:
            price_col = 'Close'
        else:
            st.error(f"Could not find 'Adj Close' or 'Close' in data columns: {data.columns}")
            return None
            
        data = data.copy()
        
        # Standardize column name for plotting
        if price_col != 'Adj Close':
            data['Adj Close'] = data[price_col]

        # Log Returns
        data['LogReturn'] = np.log(data[price_col] / data[price_col].shift(1))
        
        data = data.dropna()
        return data

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def lift_data(log_returns, h1, h2):
    """Lifts the stream of returns into a stream of segments (empirical measures)."""
    n = len(log_returns)
    segments = []
    indices = []
    
    for i in range(0, n - h1 + 1, h2):
        window = log_returns.iloc[i : i + h1].values
        segments.append(np.sort(window))
        indices.append(log_returns.index[i + h1 - 1])
        
    return np.array(segments), indices

def wasserstein_barycenter_1d(segments):
    """Calculates the 1-Wasserstein Barycenter (Component-wise Median)."""
    return np.median(segments, axis=0)

def wk_means_clustering(segments, k, max_iter=100, tol=1e-4):
    """The Wasserstein K-Means Algorithm."""
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
    """Generates a text summary interpreting the regimes."""
    summary = []
    
    # Identify Volatility Extremes
    max_vol_regime = stats_df['Volatility (Ann.)'].idxmax()
    min_vol_regime = stats_df['Volatility (Ann.)'].idxmin()
    
    # Identify Return Extremes
    max_ret_regime = stats_df['Mean Return (Ann.)'].idxmax()
    min_ret_regime = stats_df['Mean Return (Ann.)'].idxmin()

    summary.append(f"**Regime {max_vol_regime} (High Volatility):** This regime exhibits the highest annualized volatility ({stats_df.loc[max_vol_regime, 'Volatility (Ann.)']:.2%}). "
                   f"It often corresponds to 'Crisis' or 'Correction' periods (resembling Bear markets).")
    
    if max_vol_regime != min_vol_regime:
        summary.append(f"**Regime {min_vol_regime} (Low Volatility):** This regime is more stable with lower volatility ({stats_df.loc[min_vol_regime, 'Volatility (Ann.)']:.2%}). "
                       f"It typically represents 'Calm' or 'Bullish' trending periods.")
        
    return summary

# --- Main App Interface ---

st.title("Clustering Market Regimes using Wasserstein Distance")
st.markdown("""
This app applies **Wasserstein k-means** to cluster market behavior. 
It groups time periods not just by price, but by the **shape of the return distribution** (volatility, tails, skew).
""")

# Sidebar
st.sidebar.header("Configuration")
asset_select = st.sidebar.selectbox("Select Asset", ["Gold", "EURUSD", "ES (S&P 500 E-mini)", "NQ (Nasdaq 100 E-mini)", "BTC (Bitcoin)"])
ticker = get_ticker(asset_select)

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = col2.date_input("End Date", pd.to_datetime("today"))

st.sidebar.subheader("WK-Means Hyperparameters")
k_clusters = st.sidebar.slider("Number of Regimes (k)", 2, 5, 2)
window_size = st.sidebar.slider("Window Size (h1)", 20, 100, 50, help="Days in each distribution segment.")
step_size = st.sidebar.slider("Step Size (h2)", 1, 20, 5, help="Days to slide forward.")

# Execution
if st.button("Run Clustering"):
    with st.spinner("Fetching data and calculating regimes..."):
        df = fetch_data(ticker, start_date, end_date)
        
        if df is not None and len(df) > window_size:
            segments_sorted, time_indices = lift_data(df['LogReturn'], window_size, step_size)
            
            if len(segments_sorted) < k_clusters:
                st.error("Not enough data points for the chosen parameters.")
            else:
                labels, centroids = wk_means_clustering(segments_sorted, k_clusters)
                
                results_df = pd.DataFrame(index=time_indices)
                results_df['Cluster'] = labels
                merged_df = df.join(results_df)
                merged_df['Cluster'] = merged_df['Cluster'].fillna(method='ffill')
                merged_df = merged_df.dropna(subset=['Cluster'])
                merged_df['Cluster'] = merged_df['Cluster'].astype(int)

                # --- Calculate Stats for Context ---
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

                # --- Visualizations (Dark Mode) ---
                
                # 1. Price Path
                st.subheader(f"{asset_select} Price Path by Regime")
                
                # Setup Dark Figure
                fig, ax = plt.subplots(figsize=(12, 6))
                fig.patch.set_facecolor('#0E1117') # Streamlit Dark BG color approximation
                ax.set_facecolor('#0E1117')
                
                # Plot Grey line
                ax.plot(merged_df.index, merged_df['Adj Close'], color='#444444', alpha=0.5, label='Price', linewidth=1)
                
                # Plot Regimes
                colors = plt.cm.plasma(np.linspace(0, 1, k_clusters)) # Plasma looks good on dark
                
                for cluster_id in range(k_clusters):
                    cluster_data = merged_df[merged_df['Cluster'] == cluster_id]
                    ax.scatter(cluster_data.index, cluster_data['Adj Close'], 
                               color=colors[cluster_id], s=15, label=f'Regime {cluster_id}', zorder=10)
                
                ax.set_title("Identified Market Regimes", color='white', fontsize=14)
                ax.set_ylabel("Price", color='white')
                ax.tick_params(colors='white')
                ax.grid(True, color='#333333', linestyle='--', alpha=0.5)
                
                # Legend with dark text
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

                # 4. Stats Table (Formatted)
                with col_stats:
                    st.markdown("##### Detailed Statistics")
                    # Format for display
                    display_stats = stats_df_numeric.copy()
                    display_stats['Mean Return (Ann.)'] = display_stats['Mean Return (Ann.)'].apply(lambda x: f"{x:.2%}")
                    display_stats['Volatility (Ann.)'] = display_stats['Volatility (Ann.)'].apply(lambda x: f"{x:.2%}")
                    st.dataframe(display_stats)

        else:
            st.warning("No data found or data length is shorter than window size.")

else:
    st.info("Select parameters and click 'Run Clustering' to start.")

# --- Explainer ---
st.markdown("---")
st.markdown("### Methodology")
st.markdown("""
This tool uses the **Wasserstein Distance** (Earth Mover's Distance) to compare windows of returns. 
[cite_start]Unlike standard correlation or mean-variance clustering, this method compares the **entire geometry** of the return distribution [cite: 7, 30-32].

* **Regimes:** Clusters are formed by grouping time windows with similar return distributions.
* [cite_start]**Barycenters:** The curves plotted above are the "average" distribution for that regime [cite: 32, 241-243].
* [cite_start]**Context:** High volatility regimes often align with market crashes (fat tails), while low volatility regimes align with steady growth [cite: 377-382].
""")
