import pandas as pd
import numpy as np
import os

# --- Configuration ---
DATA_DIR = os.path.abspath(r'D:\workspace\xiaoyao\data')

INPUT_FILENAME = 'wide_table.parquet'
OUTPUT_FILENAME = 'wide_table.parquet'

# --- Functions ---

def get_data_path(filename: str) -> str:
    """Constructs the full, absolute path to a data file."""
    return os.path.join(DATA_DIR, filename)

def load_data(file_path: str) -> pd.DataFrame:
    """Loads the wide table, handling potential errors."""
    print(f"--- 1. Loading Data from {file_path} ---")
    try:
        df = pd.read_parquet(file_path)
        print(f"  - Successfully loaded data. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"\n[ERROR] File not found: {file_path}")
        print(f"Please run the previous scripts to generate the wide table.")
        raise

def print_stats(df: pd.DataFrame, stage: str):
    """Prints statistics about trading days and stocks per day."""
    print(f"\n--- {stage} Statistics ---")
    if df.empty:
        print("  - DataFrame is empty. No statistics to show.")
        return
    
    trading_days_count = df['date'].nunique()
    stocks_per_day = df.groupby('date').size()
    
    print(f"  - Total unique trading days: {trading_days_count}")
    print(f"  - Average stocks per day: {stocks_per_day.mean():.2f}")
    print(f"  - Max stocks on a single day: {stocks_per_day.max()}")
    print(f"  - Min stocks on a single day: {stocks_per_day.min()}")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the DataFrame by removing paused and invalid records."""
    print("\n--- 3. Cleaning Data ---")
    initial_rows = len(df)
    print(f"  - Initial number of records: {initial_rows}")

    # Filter out paused stocks
    if 'paused' in df.columns:
        df = df[df['paused'] != 1]
        rows_after_pause_filter = len(df)
        print(f"  - Records removed due to 'paused == 1': {initial_rows - rows_after_pause_filter}")
    else:
        print("  - 'paused' column not found, skipping this filter.")

    # Define columns to check for NaN or infinity
    columns_to_check = ['interday_return', 'intraday_return', 'auction_money_ratio']
    
    # Ensure columns exist before trying to clean them
    existing_columns_to_check = [col for col in columns_to_check if col in df.columns]
    if not existing_columns_to_check:
        print("  - None of the specified return/ratio columns found. Skipping NaN/inf filter.")
        return df

    # Drop rows with NaN or infinite values in the specified columns
    rows_before_nan_filter = len(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=existing_columns_to_check, inplace=True)
    rows_after_nan_filter = len(df)
    print(f"  - Records removed due to NaN/infinity in {existing_columns_to_check}: {rows_before_nan_filter - rows_after_nan_filter}")
    
    print(f"  - Final number of records after cleaning: {len(df)}")
    return df

def main():
    """Main function to orchestrate the data analysis and cleaning pipeline."""
    print("====== Starting Data Analysis and Cleaning Pipeline (Step 1) ======")
    
    input_path = get_data_path(INPUT_FILENAME)
    
    # Load data
    wide_df = load_data(input_path)
    
    # Initial statistics
    print_stats(wide_df, "2. Pre-Cleaning")
    
    # Clean data
    cleaned_df = clean_data(wide_df)
    
    # Final statistics
    print_stats(cleaned_df, "4. Post-Cleaning")
    
    # Save the cleaned data for the next steps
    output_path = get_data_path(OUTPUT_FILENAME)
    print(f"\n--- 5. Saving Cleaned Data to {output_path} ---")
    cleaned_df.to_parquet(output_path, index=False)
    print(f"  - Successfully saved cleaned data. Final shape: {cleaned_df.shape}")
    
    print("\n====== Pipeline Step 1 Finished ======")

if __name__ == "__main__":
    main()