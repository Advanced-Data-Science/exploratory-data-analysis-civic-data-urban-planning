import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker # Import for axis formatting

# Suppress warnings
warnings.filterwarnings('ignore')

# --- File Paths ---
INPUT_FILE = 'combined_311_with_sales_data.csv'


def load_and_preprocess_data(file_path):
    """Loads the combined dataset and performs initial cleanup and aggregation."""
    print("--- 1. Data Loading and Aggregation ---")
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{file_path}'. Please ensure the combining script ran successfully.")
        return None

    # Check for empty data after the last script's filtering
    if df.empty:
        print("CRITICAL: The dataset is empty after filtering (0 successful matches). Cannot perform analysis.")
        print("Please review the Date Range Debugging output in combine_datasets.py to find why matches failed.")
        return None
    
    print(f"Successfully loaded {len(df)} records with matching sales data.")

    # Ensure 'Created Date' is datetime and aggregate key is present
    df['Created Date'] = pd.to_datetime(df['Created Date'], errors='coerce')
    df['Year-Month'] = df['Created Date'].dt.to_period('M').astype(str)

    # Aggregate complaints and sales price by Borough and Month
    # This creates the core table for most analyses
    df_agg = df.groupby(['Borough', 'Year-Month']).agg(
        Complaint_Count=('Unique Key', 'count'),
        Median_Sales_Price=('Median_Sales_Price', 'mean') # Use mean of sales price per month/borough
    ).reset_index()

    print(f"Aggregated data into {len(df_agg)} unique Borough-Month combinations.\n")
    return df_agg


def pattern_analysis(df_agg):
    """
    Identifies and documents trends, detects outliers, and identifies clusters.
    """
    print("--- 2. Pattern Analysis ---")

    # A. Outlier Detection (using Z-score, method explanation)
    print("\n[A] Outlier Detection (Z-Score Method)")
    print("Method: Calculating Z-scores for Complaint Count. Values |Z| > 3 are considered outliers.")
    
    # Calculate Z-score for Complaint Count
    df_agg['Z_Score_Complaints'] = zscore(df_agg['Complaint_Count'])
    outliers = df_agg[np.abs(df_agg['Z_Score_Complaints']) > 3].sort_values(by='Z_Score_Complaints', ascending=False)
    
    if not outliers.empty:
        print(f"Detected {len(outliers)} significant outliers (where monthly complaint count is statistically unusual):")
        print(outliers[['Borough', 'Year-Month', 'Complaint_Count', 'Median_Sales_Price']])
    else:
        print("No significant outliers (Z-score > 3) found in monthly Complaint Counts.")

    # B. Clustering (K-Means)
    print("\n[B] Clustering/Grouping (K-Means)")
    print("Approach: Using K-Means to group Borough-Months based on Complaint Count and Sales Price.")
    
    # Prepare features for clustering: Complaint Count and Sales Price
    features = df_agg[['Complaint_Count', 'Median_Sales_Price']].copy()
    
    # Optional: Add Borough as a feature using Label Encoding for clustering
    le = LabelEncoder()
    features['Borough_Encoded'] = le.fit_transform(df_agg['Borough'])
    
    # Standardize the features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Use K=3 clusters (e.g., Low, Medium, High Activity/Price)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    df_agg['Cluster'] = kmeans.fit_predict(scaled_features)
    
    # Document Clusters
    cluster_summary = df_agg.groupby('Cluster').agg(
        Avg_Complaints=('Complaint_Count', 'mean'),
        Avg_Sales_Price=('Median_Sales_Price', 'mean'),
        Count=('Borough', 'count')
    ).sort_values(by='Avg_Complaints')
    
    print("Cluster Summary (Groups based on Complaint Count and Sales Price):")
    print(cluster_summary.round(0))
    print("\nInterpretation of Potential Clusters:")
    print("Cluster 0 (Lowest Complaint/Price)")
    print("Cluster 1 (Mid-Range)")
    print("Cluster 2 (Highest Complaint/Price)\n")


def segmentation_analysis(df_agg):
    """
    Identifies meaningful segments (Boroughs).
    """
    print("--- 3. Segmentation Analysis (Segmented by Borough) ---")

    # Define segmentation criteria (Borough is the segment)
    print("Segmentation Approach: Borough (geographical segmentation)")
    print("Comparison Criteria: Average Monthly Complaint Count, Average Median Sales Price.")

    # Compare characteristics across segments (Boroughs)
    segment_summary = df_agg.groupby('Borough').agg(
        Total_Complaint_Count=('Complaint_Count', 'sum'),
        Avg_Monthly_Complaints=('Complaint_Count', 'mean'),
        Avg_Monthly_Sales_Price=('Median_Sales_Price', 'mean'),
        Num_Months=('Year-Month', 'count')
    ).reset_index().sort_values(by='Avg_Monthly_Complaints', ascending=False)

    print("\nSummary Statistics for Each Borough Segment:")
    print(segment_summary.to_markdown(index=False, floatfmt=".0f"))
    
    print("\nDiscussion of Segment Differences and Implications:")
    # Identify highest and lowest
    highest_complaint_borough = segment_summary.iloc[0]['Borough']
    lowest_complaint_borough = segment_summary.iloc[-1]['Borough']
    
    print(f"Segment Differences:")
    print(f"- {highest_complaint_borough} consistently has the highest average monthly complaints.")
    print(f"- {lowest_complaint_borough} has the lowest average.")

def time_series_analysis(df_agg):
    """
    Identifies and visualizes temporal trends.
    """
    print("--- 4. Time Series Analysis (Temporal Trends) ---")
    
    # Aggregate only by month (across all boroughs)
    df_ts = df_agg.groupby('Year-Month').agg(
        Total_Complaints=('Complaint_Count', 'sum'),
        Avg_Sales_Price_NYC=('Median_Sales_Price', 'mean')
    ).reset_index()

    # Convert 'Year-Month' back to datetime objects for plotting/sorting
    df_ts['Date'] = pd.to_datetime(df_ts['Year-Month'])
    df_ts = df_ts.sort_values('Date')
    df_ts['Time_Index'] = np.arange(len(df_ts)) # Recalculate Time_Index for trend lines

    if len(df_ts) < 5:
        print("Temporal Data too sparse or covers a very short period. Trend analysis is limited.")
        print(f"Data available for {len(df_ts)} months: {df_ts['Year-Month'].tolist()}")
        return

    # Identify temporal trend
    time_span = (df_ts['Date'].max() - df_ts['Date'].min()).days / 30
    
    # Calculate simple correlation between Time Index and Complaints/Sales Price
    corr_complaints = df_ts['Time_Index'].corr(df_ts['Total_Complaints'])
    corr_sales = df_ts['Time_Index'].corr(df_ts['Avg_Sales_Price_NYC'])
    
    print(f"Data covers {len(df_ts)} months (approx. {time_span:.1f} months).")
    print(f"Correlation (Time vs. Total Complaints): {corr_complaints:.2f}")
    print(f"Correlation (Time vs. Average Sales Price): {corr_sales:.2f}")

    print("\nDiscussion of Temporal Dynamics:")
    if abs(corr_complaints) > 0.5:
        print("- There is a noticeable linear trend in total complaints over time.")
    else:
        print("- Total complaints show no strong linear trend over the observed period, suggesting seasonality or short-term volatility dominate.")
    
    if abs(corr_sales) > 0.5:
        print("- Average sales price shows a noticeable linear trend over time, indicating general market movement.")
    else:
        print("- Average sales price shows no strong linear trend, suggesting market stability or high volatility.\n")


def generate_visualizations(df_agg):
    """Generates and saves the required analytical plots."""
    print("\n--- 5. Generating Visualizations ---")

    # Helper function to format dollar amounts in millions
    def dollar_formatter(x, pos):
        """Formats an axis tick value as Millions of USD."""
        return f'${x*1e-6:.1f}M'

    # 1. Cluster Visualization (Scatter Plot)
    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(
        x='Complaint_Count',
        y='Median_Sales_Price',
        hue='Cluster',
        data=df_agg,
        palette='viridis',
        style='Borough',
        s=100
    )
    
    # --- ADD SCATTERPLOT TREND LINE ---
    # Add an overall linear regression line (trend line)
    sns.regplot(
        x='Complaint_Count',
        y='Median_Sales_Price',
        data=df_agg,
        scatter=False,  # Plot only the line
        color='black',
        line_kws={'linestyle': '--', 'linewidth': 1.5, 'label': 'Overall Linear Trend'}
    )
    # -----------------------------------
    
    # Format the Y-axis to show large dollar amounts (in Millions)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))

    # Add legend that includes the regression line
    h, l = ax.get_legend_handles_labels()
    # The first 3 or 4 are clusters/boroughs, the last is the trend line
    ax.legend(h, l, title=ax.get_legend().get_title().get_text(), loc='upper right')

    plt.title('Pattern Analysis: Monthly Activity Clusters (Complaints vs. Sales Price)')
    plt.xlabel('Monthly Complaint Count')
    plt.ylabel('Median Sales Price') 
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('viz_1_clustering_scatter.png')
    print("Saved viz_1_clustering_scatter.png (Complaints vs. Sales Price by Cluster) with trend line.")
    plt.close()

    # 2. Segmentation Visualization (Bar Plot)
    # Recalculate segment summary for consistent plotting data
    segment_summary = df_agg.groupby('Borough').agg(
        Avg_Monthly_Complaints=('Complaint_Count', 'mean')
    ).reset_index().sort_values(by='Avg_Monthly_Complaints', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Avg_Monthly_Complaints',
        y='Borough',
        data=segment_summary,
        palette='rocket'
    )
    plt.title('Segmentation Analysis: Average Monthly Complaints by Borough')
    plt.xlabel('Average Monthly Complaint Count')
    plt.ylabel('Borough')
    plt.savefig('viz_2_segmentation_bar.png')
    print("Saved viz_2_segmentation_bar.png (Avg Complaints by Borough)")
    plt.close()

    # 3. Time Series Visualization (Dual-Axis Line Plot)
    df_ts = df_agg.groupby('Year-Month').agg(
        Total_Complaints=('Complaint_Count', 'sum'),
        Avg_Sales_Price_NYC=('Median_Sales_Price', 'mean')
    ).reset_index()
    df_ts['Date'] = pd.to_datetime(df_ts['Year-Month'])
    df_ts = df_ts.sort_values('Date')
    df_ts['Time_Index'] = np.arange(len(df_ts)) # Ensure Time_Index is available for polyfit
    
    if len(df_ts) >= 5: # Only plot if time series is long enough
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Calculate Trend Lines (Linear Fit)
        # Complains Trend (ax1)
        z_comp = np.polyfit(df_ts['Time_Index'], df_ts['Total_Complaints'], 1)
        p_comp = np.poly1d(z_comp)
        
        # Sales Price Trend (ax2)
        z_sales = np.polyfit(df_ts['Time_Index'], df_ts['Avg_Sales_Price_NYC'], 1)
        p_sales = np.poly1d(z_sales)

        # Plot Complaints on primary axis (ax1)
        color = 'tab:blue'
        ax1.set_xlabel('Time (Year-Month)')
        ax1.set_ylabel('Total Complaints (NYC)', color=color)
        ax1.plot(df_ts['Year-Month'], df_ts['Total_Complaints'], color=color, label='Total Complaints', marker='o')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # --- ADD COMPLAINTS TREND LINE ---
        ax1.plot(df_ts['Year-Month'], p_comp(df_ts['Time_Index']), color=color, linestyle=':', linewidth=2, label='Complaint Trend')

        # Create secondary axis for Sales Price (ax2)
        ax2 = ax1.twinx()  
        color = 'tab:red'
        ax2.set_ylabel('Avg Sales Price', color=color) # Adjusted label
        ax2.plot(df_ts['Year-Month'], df_ts['Avg_Sales_Price_NYC'], color=color, linestyle='--', label='Avg Sales Price', marker='x')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # --- ADD SALES PRICE TREND LINE ---
        ax2.plot(df_ts['Year-Month'], p_sales(df_ts['Time_Index']), color=color, linestyle=':', linewidth=2, label='Sales Price Trend')
     
        # Format secondary Y-axis (Sales Price) in Millions of USD
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        tick_interval = max(1, len(df_ts['Year-Month']) // 10)
        plt.xticks(df_ts['Year-Month'][::tick_interval], rotation=45, ha='right')

        fig.tight_layout()
        plt.title('Time Series Analysis: Complaints vs. Sales Price Trend (NYC)')
        plt.savefig('viz_3_timeseries_dual_axis.png')
        print("Saved viz_3_timeseries_dual_axis.png (Complaints & Sales Price over Time) with trend lines.")
        plt.close()
    else:
        print("Skipping time series visualization: Time data is too sparse.")


def main():
    """Main function to run all analysis steps."""
    
    df_agg = load_and_preprocess_data(INPUT_FILE)
    
    if df_agg is not None:
        # Run analysis functions
        pattern_analysis(df_agg)
        segmentation_analysis(df_agg)
        time_series_analysis(df_agg)
        
        # Generate and save visualizations
        generate_visualizations(df_agg)
        
        print("\n--- ANALYSIS COMPLETE ---")
        print("The analysis and visualizations have been generated.")
        print("Look for the three image files (viz_1_*, viz_2_*, viz_3_*) in the file output.")

if __name__ == '__main__':
    main()
