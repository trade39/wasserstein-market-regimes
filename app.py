import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance

# --- Configuration & Styles ---
st.set_page_config(page_title="Wasserstein Market Regimes", layout="wide")
st.markdown("""
<style>
    .reportview-container { margin-top: -2em; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
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
    """Fetch closing prices and calculate log returns."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return None
        # Calculate Log Returns: r_i = log(P_t) - log(P_{t-1})
        data['LogReturn'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
        data = data.dropna()
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def lift_data(log_returns, h1, h2):
    """
    Lifts the stream of returns into a stream of segments (empirical measures).
    h1: Window size (number of atoms in the distribution)
    h2: Step size (stride)
    """
    n = len(log_returns)
    segments = []
    indices = []
    
    # Sliding window
    for i in range(0, n - h1 + 1, h2):
        window = log_returns.iloc[i : i + h1].values
        # We sort the window immediately because W_1 distance between 1D empirical 
        # measures is simply the L1 distance between their sorted vectors (quantiles).
        segments.append(np.sort(window))
        # Store the middle timestamp of the window for plotting
        indices.append(log_returns.index[i + h1 - 1])
        
    return np.array(segments), indices

def wasserstein_barycenter_1d(segments):
    """
    Calculates the 1-Wasserstein Barycenter for a set of 1D empirical measures.
    According to the paper (Prop 2.6), for p=1, this is the element-wise Median
    of the sorted atom vectors.
    """
    # segments shape: (M, N) where M is num_segments in cluster, N is window_size
    # Calculate column-wise median
    return np.median(segments, axis=0)

def wk_means_clustering(segments, k, max_iter=100, tol=1e-4):
    """
    The Wasserstein K-Means Algorithm.
    """
    n_segments, window_size = segments.shape
    
    # 1. Initialization: Randomly choose k segments as initial centroids
    rng = np.random.default_rng(42)
    initial_indices = rng.choice(n_segments, size=k, replace=False)
    centroids = segments[initial_indices]
    
    prev_loss = float('inf')
    labels = np.zeros(n_segments, dtype=int)
    
    for iteration in range(max_iter):
        # 2. Assignment Step
        # Calculate distance from every segment to every centroid
        # Since segments and centroids are sorted 1D arrays, W_1 is mean(|u - v|)
        distances = np.zeros((n_segments, k))
        for j in range(k):
            # Vectorized L1 distance between sorted vectors
            # shape broadcast: (n_segments, window_size) - (window_size,)
            diff = np.abs(segments - centroids[j]) 
            distances[:, j] = np.mean(diff, axis=1) # Average over atoms (1/N sum |a_i - b_i|)
            
        # Assign to closest centroid
        labels = np.argmin(distances, axis=1)
        current_loss = np.sum(np.min(distances, axis=1))
        
        # Check convergence
        if np.abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss
        
        # 3. Update Step (Barycenter)
        new_centroids = []
        for j in range(k):
            cluster_members = segments[labels == j]
            if len(cluster_members) > 0:
                # Update centroid to be the Wasserstein Barycenter of the cluster
                new_centroid = wasserstein_barycenter_1d(cluster_members)
                new_centroids.append(new_centroid)
            else:
                # Handle empty cluster by re-initializing (rare but possible)
                new_centroids.append(segments[rng.choice(n_segments)])
        centroids = np.array(new_centroids)
        
    return labels, centroids

# --- Main App Interface ---

st.title("Clustering Market Regimes using Wasserstein Distance")
st.markdown("""
This app applies the **Wasserstein k-means (WK-means)** algorithm to financial time series. 
It treats sliding windows of returns as probability distributions and clusters them based on the 
Earth Mover's Distance (1-Wasserstein), capturing shifts in market behavior (volatility, skewness, etc.) 
without relying on rigid parametric assumptions.
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
window_size = st.sidebar.slider("Window Size (h1)", 20, 100, 50, help="Number of days in each distribution segment.")
step_size = st.sidebar.slider("Step Size (h2)", 1, 20, 5, help="How many days to slide the window forward.")

# Execution
if st.button("Run Clustering"):
    with st.spinner("Fetching data and calculating regimes..."):
        # 1. Get Data
        df = fetch_data(ticker, start_date, end_date)
        
        if df is not None and len(df) > window_size:
            # 2. Lift Data (Create Empirical Measures)
            # segments shape: (Num_Windows, Window_Size) - these are sorted vectors
            segments_sorted, time_indices = lift_data(df['LogReturn'], window_size, step_size)
            
            if len(segments_sorted) < k_clusters:
                st.error("Not enough data points for the chosen window size and step size.")
            else:
                # 3. Run Algorithm
                labels, centroids = wk_means_clustering(segments_sorted, k_clusters)
                
                # 4. Process Results for Visualization
                results_df = pd.DataFrame(index=time_indices)
                results_df['Cluster'] = labels
                # Align with original price data
                # We map the cluster of the window ending at t to time t
                merged_df = df.join(results_df)
                
                # Forward fill cluster labels for the "step size" gaps to make the plot continuous
                merged_df['Cluster'] = merged_df['Cluster'].fillna(method='ffill')
                # Drop initial NaN rows created by window lag
                merged_df = merged_df.dropna(subset=['Cluster'])
                merged_df['Cluster'] = merged_df['Cluster'].astype(int)

                # --- Visualizations ---
                
                # A. Price Path Colored by Regime
                st.subheader(f"{asset_select} Price Path by Regime")
                
                # Create segments for coloring
                # We can't easily color a single line with multiple colors in standard matplotlib 
                # without iterating segments. Using scatter for simplicity or broken line segments.
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot underlying grey line for continuity
                ax.plot(merged_df.index, merged_df['Adj Close'], color='lightgrey', alpha=0.5, label='Price')
                
                colors = plt.cm.viridis(np.linspace(0, 1, k_clusters))
                
                for cluster_id in range(k_clusters):
                    # Filter data for this cluster
                    cluster_data = merged_df[merged_df['Cluster'] == cluster_id]
                    ax.scatter(cluster_data.index, cluster_data['Adj Close'], 
                               color=colors[cluster_id], s=10, label=f'Regime {cluster_id}')
                
                ax.set_title("Identified Market Regimes")
                ax.set_ylabel("Price")
                ax.legend()
                st.pyplot(fig)

                # B. Centroid Distributions (The "Shape" of the Regime)
                st.subheader("Regime Return Distributions (Centroids)")
                st.markdown("These curves represent the 'typical' distribution of returns for each identified regime (the Wasserstein Barycenters).")
                
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                for i in range(k_clusters):
                    # Centroids are sorted vectors (Quantile Functions essentially). 
                    # We can plot their KDE or Histogram to see the distribution shape.
                    sns.kdeplot(centroids[i], ax=ax2, color=colors[i], label=f'Regime {i}', fill=True, alpha=0.3)
                    
                ax2.set_title("Probability Density of Cluster Centroids")
                ax2.set_xlabel("Log Return")
                ax2.legend()
                st.pyplot(fig2)
                
                # C. Regime Statistics
                st.subheader("Regime Statistics")
                
                stats = []
                for i in range(k_clusters):
                    # Get all actual returns belonging to this cluster
                    # Note: We use the centroid for "Ideal" stats, or actual data for "Realized" stats.
                    # Let's use the actual classified segments to compute realized volatility.
                    cluster_indices = np.where(labels == i)[0]
                    if len(cluster_indices) > 0:
                        # flatten all windows in this cluster
                        all_returns = segments_sorted[cluster_indices].flatten()
                        
                        # Annualize (assuming daily data, 252 days)
                        vol = np.std(all_returns) * np.sqrt(252)
                        mu = np.mean(all_returns) * 252
                        
                        stats.append({
                            "Regime": i,
                            "Count (Windows)": len(cluster_indices),
                            "Mean Return (Ann.)": f"{mu:.2%}",
                            "Volatility (Ann.)": f"{vol:.2%}",
                            "Min Return": f"{np.min(all_returns):.2%}",
                            "Max Return": f"{np.max(all_returns):.2%}"
                        })
                
                st.table(pd.DataFrame(stats).set_index("Regime"))

        else:
            st.warning("No data found or data length is shorter than window size.")

else:
    st.info("Select parameters and click 'Run Clustering' to start.")

# --- Explainer ---
st.markdown("---")
st.markdown("### How it works (Based on PDF)")
st.markdown("""
1. **Lift**: The price series is converted into log-returns and sliced into overlapping windows (length $h_1$).
2. **Empirical Measure**: Each window is treated as a probability distribution.
3. **Wasserstein Distance**: The algorithm calculates the distance between these distributions. For 1D data, this is the $L_1$ distance between sorted returns.
4. **WK-Means**: 
    - **Assignment**: Windows are assigned to the closest regime centroid.
    - **Update**: Centroids are updated by calculating the **Wasserstein Barycenter** (which, for $p=1$, is the component-wise median of sorted distributions).
""")
