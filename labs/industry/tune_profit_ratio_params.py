import pandas as pd
import os
import itertools

# --- Configuration ---
DATA_DIR = os.path.abspath(r'D:\workspace\xiaoyao\data')
INPUT_FILENAME = 'wide_table.parquet'

# --- Parameter Grid ---
# Define the range of parameters to test
AUCTION_RATIO_THRESHOLDS = [5, 10, 15, 20, 25, 30, 40, 50]
PROFIT_THRESHOLDS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]

# --- Functions ---

def get_data_path(filename: str, directory: str) -> str:
    """Constructs the full, absolute path to a data file."""
    return os.path.join(directory, filename)

def load_cleaned_data(file_path: str) -> pd.DataFrame:
    """Loads the cleaned data table."""
    print(f"--- 1. Loading Cleaned Data from {file_path} ---")
    try:
        df = pd.read_parquet(file_path)
        print(f"  - Successfully loaded data. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"\n[ERROR] File not found: {file_path}")
        raise

def run_tuning_loop(df: pd.DataFrame, industry_column: str = 'ind_sw_l3_industry_name'):
    """
    Loops through parameter combinations, calculates correlations, 
    and returns a DataFrame with the results.
    """
    print("\n--- 2. Starting Parameter Tuning Loop ---")
    
    results = []
    
    # Create all combinations of parameters
    param_grid = list(itertools.product(AUCTION_RATIO_THRESHOLDS, PROFIT_THRESHOLDS))
    total_combinations = len(param_grid)
    
    for i, (auction_thresh, profit_thresh) in enumerate(param_grid):
        print(f"  - Running combination {i+1}/{total_combinations}: auction_thresh={auction_thresh}, profit_thresh={profit_thresh}")
        
        # Define aggregation logic using helper functions for the current parameters
        def ratio_greater_than(series, threshold):
            return (series > threshold).sum() / series.count() if series.count() > 0 else 0

        aggregations = {
            'auction_money_ratio': lambda x: ratio_greater_than(x, auction_thresh),
            'interday_return': lambda x: ratio_greater_than(x, profit_thresh),
            'open': 'count' # Keep for context if needed
        }
        
        # Group by date and industry, then aggregate
        industry_df = df.groupby(['date', industry_column]).agg(aggregations)
        
        # Rename columns for clarity
        industry_df.rename(columns={
            'auction_money_ratio': 'high_auction_ratio_pct',
            'interday_return': 'interday_profit_ratio',
        }, inplace=True)
        
        # Calculate correlation if there's enough data
        if not industry_df.empty and industry_df.shape[0] > 1:
            correlation = industry_df['high_auction_ratio_pct'].corr(industry_df['interday_profit_ratio'])
            if pd.notna(correlation):
                results.append({
                    'auction_threshold': auction_thresh,
                    'profit_threshold': profit_thresh,
                    'correlation': correlation
                })

    print("\n--- 3. Tuning Loop Finished ---")
    
    if not results:
        print("  - No results were generated. This might be due to empty dataframes after aggregation.")
        return pd.DataFrame()

    # Convert results to a DataFrame and sort
    results_df = pd.DataFrame(results)
    results_df.sort_values(by='correlation', ascending=False, inplace=True)
    
    return results_df

def main():
    """Main function to orchestrate the parameter tuning."""
    print("====== Starting Parameter Tuning Pipeline ======")
    
    input_path = get_data_path(INPUT_FILENAME, DATA_DIR)
    
    # 1. Load data
    cleaned_df = load_cleaned_data(input_path)
    
    # 2. Run the tuning loop
    results_df = run_tuning_loop(cleaned_df, industry_column='ind_sw_l3_industry_name')
    
    # 3. Display results
    print("\n--- 4. Top 10 Parameter Combinations by Correlation ---")
    print(results_df.head(10).to_string())
    
    if not results_df.empty:
        best_params = results_df.iloc[0]
        print("\n--- 5. Best Result ---")
        print(f"The highest correlation of {best_params['correlation']:.6f} was achieved with:")
        print(f"  - auction_threshold: {best_params['auction_threshold']}")
        print(f"  - profit_threshold:  {best_params['profit_threshold']:.2f}")

    print("\n====== Tuning Finished ======")

if __name__ == "__main__":
    main()