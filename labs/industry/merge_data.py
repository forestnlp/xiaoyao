import pandas as pd
import os

# Define constants for key columns and the data directory
# Assumes the script is in 'labs', and data is in a parallel 'data' directory
DATA_DIR = os.path.abspath(r'D:\workspace\xiaoyao\data')

KEY_COLUMNS = ['date', 'stock_code']

def get_data_path(filename: str) -> str:
    """Constructs the full, absolute path to a data file."""
    return os.path.join(DATA_DIR, filename)

def load_data(file_paths: dict[str, str]) -> dict[str, pd.DataFrame]:
    """
    Loads multiple parquet files into a dictionary of DataFrames.
    Handles FileNotFoundError and provides clear error messages.
    """
    dataframes = {}
    print("--- 1. Loading Data ---")
    try:
        for name, path in file_paths.items():
            df = pd.read_parquet(path)
            print(f"  - Successfully loaded '{name}'. Shape: {df.shape}")
            dataframes[name] = df
        return dataframes
    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e.filename}")
        print(f"Please ensure all required .parquet files are in the directory: '{os.path.abspath(DATA_DIR)}'")
        raise

def standardize_date_columns(dataframes: dict[str, pd.DataFrame]) -> None:
    """
    Standardizes the 'date' column to datetime.date objects in place
    for all DataFrames in the dictionary.
    """
    print("\n--- 2. Standardizing Date Columns ---")
    for name, df in dataframes.items():
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date
    print("  - Date columns standardized successfully.")

def merge_dataframes(base_df: pd.DataFrame, dfs_to_merge: list[tuple[pd.DataFrame, str]]) -> pd.DataFrame:
    """
    Merges a list of DataFrames into a base DataFrame.
    Adds a specified prefix to all non-key columns to prevent conflicts.
    """
    print("\n--- 3. Merging DataFrames ---")
    merged_df = base_df.copy()
    print(f"  - Base DataFrame initial shape: {merged_df.shape}")

    for df_to_merge, prefix in dfs_to_merge:
        df_renamed = df_to_merge.copy()
        
        # Identify columns to be renamed (all columns except the key columns)
        cols_to_rename = [col for col in df_renamed.columns if col not in KEY_COLUMNS]
        
        # Create the renaming dictionary, e.g., {'volume': 'auc_volume'}
        rename_map = {col: f"{prefix}_{col}" for col in cols_to_rename}
        df_renamed.rename(columns=rename_map, inplace=True)
        
        # Perform the left merge
        merged_df = pd.merge(merged_df, df_renamed, on=KEY_COLUMNS, how='inner')
        print(f"  - Merged with '{prefix}' data. New shape: {merged_df.shape}")
        
    return merged_df

def main():
    """Main function to orchestrate the data loading, processing, and merging."""
    print("====== Starting Data Merging Pipeline ======")
    
    # Define the paths for the input data files
    file_paths = {
        "price": get_data_path('stock_daily_price.parquet'),
        "auction": get_data_path('stock_daily_auction.parquet'),
        "valuation": get_data_path('stock_daily_marketcap.parquet'),
        "industry": get_data_path('stock_daily_industry.parquet')
    }

    # Execute the pipeline steps
    all_dfs = load_data(file_paths)
    standardize_date_columns(all_dfs)

    # Prepare for merging
    price_df = all_dfs.pop("price")
    
    # Ensure 'stock_code' column exists for merging, renaming 'instrument' if necessary
    if 'instrument' in price_df.columns and 'stock_code' not in price_df.columns:
        price_df.rename(columns={'instrument': 'stock_code'}, inplace=True)
        print("  - Renamed 'instrument' to 'stock_code' in price_df for consistency.")

    # Define the list of DataFrames to merge into the base table
    dfs_to_merge_list = [
        (all_dfs["auction"], 'auc'),
        (all_dfs["valuation"], 'val'),
        (all_dfs["industry"], 'ind')
    ]
    
    # Perform the merge
    wide_table = merge_dataframes(base_df=price_df, dfs_to_merge=dfs_to_merge_list)

    # Save the final merged DataFrame
    print("\n--- 4. Saving Result ---")
    output_path = get_data_path('wide_table.parquet')
    wide_table.to_parquet(output_path, index=False)
    
    print(f"  - Successfully saved wide table to: {output_path}")
    print(f"  - Final shape of the merged table: {wide_table.shape}")
    
    print("\n====== Pipeline Finished ======")
    print("\nFinal Merged DataFrame Head:")
    print(wide_table.head())

if __name__ == "__main__":
    main()