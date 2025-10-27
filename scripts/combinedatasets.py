import pandas as pd
import warnings

# Suppress pandas warning about future changes in Series.dt.month/year
warnings.simplefilter(action='ignore', category='FutureWarning')

# --- File Paths ---
COMPLAINTS_FILE = '311_Noise_Complaints_processed.csv'
SALES_FILE = 'Processed_MedianSales.csv'
OUTPUT_FILE = 'combined_311_with_sales_data.csv'

# Standard mapping to ensure consistent Borough names (often Staten Island vs. Richmond)
BOROUGH_MAPPING = {
    'MANHATTAN': 'MANHATTAN',
    'BRONX': 'BRONX',
    'BROOKLYN': 'BROOKLYN',
    'QUEENS': 'QUEENS',
    'STATEN ISLAND': 'STATEN ISLAND',
    'RICHMOND': 'STATEN ISLAND' # Standardize Richmond to Staten Island
}


def clean_boroughs(df, column_name='Borough'):
    """Performs robust cleaning and standardization of borough names."""
    # 1. Convert to uppercase, strip whitespace, and handle NaN/None
    df[column_name] = df[column_name].astype(str).str.upper().str.strip()
    
    # 2. Apply the canonical mapping
    # This ensures names like 'RICHMOND' (the county name) map to 'STATEN ISLAND' (the borough name)
    df[column_name] = df[column_name].map(BOROUGH_MAPPING)
    
    # 3. Handle any remaining non-matching/NaN values (set to UNKNOWN for clear exclusion)
    df[column_name] = df[column_name].fillna('UNKNOWN')
    
    return df


def load_and_prepare_data(complaints_file, sales_file):
    """
    Loads both datasets and prepares them for merging.
    """
    print("Loading datasets...")

    # Load 311 Complaints Data
    try:
        df_311 = pd.read_csv(complaints_file)
    except FileNotFoundError:
        print(f"Error: 311 Complaints file not found at '{complaints_file}'")
        return None, None

    # Load Median Sales/Rent Data
    try:
        df_sales = pd.read_csv(sales_file)
    except FileNotFoundError:
        print(f"Error: Sales/Rent file not found at '{sales_file}'")
        return None, None

    # --- 1. Prepare Sales/Rent Data (Melt from Wide to Long Format) ---
    print("Preparing sales/rent data...")

    # Identify date columns
    id_vars = ['areaName', 'Borough', 'areaType']
    date_columns = [col for col in df_sales.columns if col not in id_vars]

    # Melt the DataFrame to create one row per Borough, Month, and Value
    df_sales_long = pd.melt(
        df_sales,
        id_vars=id_vars,
        value_vars=date_columns,
        var_name='Year-Month',  # e.g., '2016-01'
        value_name='Median_Sales_Price'
    )

    df_sales_long['Year-Month'] = df_sales_long['Year-Month'].astype(str).str.strip()

    # Clean up the Borough names robustly
    df_sales_long = clean_boroughs(df_sales_long, 'Borough')

    # --- 2. Prepare 311 Complaints Data ---
    print("Preparing 311 complaints data...")

    # Convert 'Created Date' to datetime objects
    df_311['Created Date'] = pd.to_datetime(df_311['Created Date'], errors='coerce', utc=True).dt.tz_localize(None)

    # Filter out rows where date conversion failed
    df_311.dropna(subset=['Created Date'], inplace=True)

    # Extract the Year-Month string (e.g., '2025-10') for the join key
    df_311['Year-Month'] = df_311['Created Date'].dt.strftime('%Y-%m')

    # Clean up the Borough names
    df_311 = clean_boroughs(df_311, 'Borough')
    
    # --- Debugging Output ---
    print("\n--- Debugging Boroughs ---")
    print(f"Unique 311 Boroughs after cleaning: {df_311['Borough'].unique()}")
    print(f"Unique Sales Boroughs after cleaning: {df_sales_long['Borough'].unique()}")
    print("--------------------------\n")

    print(f"311 data has {len(df_311)} rows.")
    # Filter sales data to only include valid boroughs before printing size
    valid_sales_rows = df_sales_long[df_sales_long['Borough'].isin(BOROUGH_MAPPING.values())]
    print(f"Sales/Rent data has {len(valid_sales_rows)} valid Borough-Month combinations.")

    return df_311, df_sales_long


def merge_and_save(df_311, df_sales_long, output_file):
    """
    Merges the two prepared DataFrames using a memory-efficient lookup and saves the result,
    dropping any 311 records that do not find a matching sales price.
    """
    if df_311 is None or df_sales_long is None:
        return

    print("Optimized merging datasets using dictionary lookup...")

    # 1. Create a composite key for the sales data
    df_sales_long['key'] = df_sales_long['Borough'] + '_' + df_sales_long['Year-Month']

    # 2. Convert the sales data into a dictionary for fast, memory-efficient lookups
    # This creates a map: {'BOROUGH_YYYY-MM': Median_Sales_Price}
    sales_map = df_sales_long.drop_duplicates(subset=['key']).set_index('key')['Median_Sales_Price'].to_dict()

    # 3. Create the corresponding composite key in the 311 data
    df_311['key'] = df_311['Borough'] + '_' + df_311['Year-Month']

    # --- DEBUGGING: Check keys and date ranges ---
    print("\n--- Key and Date Range Debugging ---")
    # Sample Keys
    sample_sales_keys = list(sales_map.keys())[:5]
    print(f"Sample Sales Keys (from map): {sample_sales_keys}")
    
    # Filter for non-UNKNOWN boroughs before getting samples
    valid_311_keys = df_311[df_311['Borough'] != 'UNKNOWN']['key'].unique().tolist()
    sample_311_keys = valid_311_keys[:5]
    print(f"Sample 311 Keys (for lookup): {sample_311_keys}")

    # Date Range Overlap
    try:
        min_311_date = df_311['Created Date'].min().strftime('%Y-%m')
        max_311_date = df_311['Created Date'].max().strftime('%Y-%m')
        min_sales_date = df_sales_long['Year-Month'].min()
        max_sales_date = df_sales_long['Year-Month'].max()

        print(f"311 Complaint Date Range: {min_311_date} to {max_311_date}")
        print(f"Sales Data Date Range: {min_sales_date} to {max_sales_date}")
    except Exception as e:
        print(f"Could not calculate date ranges: {e}")
    print("------------------------------------\n")
    # --------------------------------------------------

    # 4. Use the .map() function to perform the lookup
    df_311['Median_Sales_Price'] = df_311['key'].map(sales_map)
    
    # *** Drop rows where Median_Sales_Price is NaN (i.e., no match was found) ***
    initial_count = len(df_311)
    df_311.dropna(subset=['Median_Sales_Price'], inplace=True)
    mapped_count = len(df_311)
    
    # --- Debugging check for successful filtering ---
    dropped_count = initial_count - mapped_count
    print(f"Initial 311 complaint count: {initial_count}")
    print(f"Number of 311 complaints successfully matched and KEPT: {mapped_count}")
    print(f"Number of 311 complaints DROPPED due to missing sales price: {dropped_count}")

    if mapped_count == 0:
        print("CRITICAL: Zero matches found. Please check the date range and sample keys output above.")
    
    # Clean up the temporary key and Year-Month column
    df_combined = df_311.drop(columns=['Year-Month', 'key'])


    # Save the final combined DataFrame to a new CSV file
    try:
        df_combined.to_csv(output_file, index=False)
        print(f"\nSuccess! The matched dataset has been saved to: {output_file}")
        print(f"Final combined dataset size: {len(df_combined)} rows.")
        print("Only rows that found a 'Median_Sales_Price' match have been kept.")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}. PLEASE ENSURE THE FILE IS CLOSED BEFORE RUNNING.")


if __name__ == '__main__':
    # Load and prepare the data
    complaints_data, sales_data = load_and_prepare_data(COMPLAINTS_FILE, SALES_FILE)

    # Merge and save the result
    merge_and_save(complaints_data, sales_data, OUTPUT_FILE)
