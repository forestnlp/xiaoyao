import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from typing import List

# --- Configuration ---
DATA_DIR = os.path.abspath(r'D:\workspace\xiaoyao\data')
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILENAME = 'wide_table.parquet'
OUTPUT_FILENAME = 'industry_profit_ratio_table.parquet'
OUTPUT_PLOT_DIR = os.path.join(CURRENT_DIR, 'plots')

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
        print(f"Please run the 'analyze_data.py' script first to generate the cleaned table.")
        raise

def calculate_industry_ratios(df: pd.DataFrame, industry_column: str = 'ind_sw_l3_industry_name') -> pd.DataFrame:
    """Calculates various ratios at the industry level, including profit ratios."""
    print(f"\n--- 2. Calculating Industry-Level Ratios (Grouping by '{industry_column}') ---")
    
    # Define aggregation logic using helper functions
    def ratio_greater_than(series, threshold):
        return (series > threshold).sum() / series.count() if series.count() > 0 else 0

    aggregations = {
        'auction_money_ratio': lambda x: ratio_greater_than(x, 20),
        'interday_return': lambda x: ratio_greater_than(x, 0.08),
        'intraday_return': lambda x: ratio_greater_than(x, 0.08),
        'open': 'count'
    }
    
    # Group by date and industry, then aggregate
    industry_df = df.groupby(['date', industry_column]).agg(aggregations)
    
    # Rename columns for clarity
    industry_df.rename(columns={
        'auction_money_ratio': 'high_auction_ratio_pct',
        'interday_return': 'interday_profit_ratio',
        'intraday_return': 'intraday_profit_ratio',
        'open': 'stock_count'
    }, inplace=True)
    
    # Reset index to make 'date' and industry column regular columns
    industry_df.reset_index(inplace=True)
    
    print(f"  - Successfully created industry ratio table. Shape: {industry_df.shape}")
    print("  - Displaying last 5 rows of the new table:")
    print(industry_df.tail())
    return industry_df

def analyze_correlation(df: pd.DataFrame, columns: List[str]):
    """Calculates and prints the correlation matrix for the given columns."""
    print("\n--- 4. Correlation Analysis ---")
    correlation_matrix = df[columns].corr()
    print("Correlation Matrix:")
    print(correlation_matrix)



def main():
    """Main function to orchestrate the industry profit ratio analysis."""
    print("====== Starting Industry Profit Ratio Analysis Pipeline ======")
    
    input_path = get_data_path(INPUT_FILENAME, DATA_DIR)
    output_path = get_data_path(OUTPUT_FILENAME, DATA_DIR)
    
    # 1. Load data
    cleaned_df = load_cleaned_data(input_path)
    
    # 2. Calculate industry ratios
    industry_ratios_df = calculate_industry_ratios(cleaned_df, industry_column='ind_sw_l3_industry_name')
    
    # 3. Save the new table
    print(f"\n--- 3. Saving Industry Ratios Table to {output_path} ---")
    industry_ratios_df.to_parquet(output_path, index=False)
    print("  - Save complete.")

    # 4. Define columns for analysis
    analysis_columns = ['high_auction_ratio_pct', 'interday_profit_ratio', 'intraday_profit_ratio']
    
    # 5. Correlation Analysis
    analyze_correlation(industry_ratios_df, analysis_columns)
    
    # 6. Visualization
    print(f"\n--- 6. Visualizing Relationships ---")
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
    print(f"  - Plots will be saved to: {OUTPUT_PLOT_DIR}")

    # Scatter plot for high_auction_ratio_pct vs interday_profit_ratio
    plt.figure(figsize=(10, 6))
    sns.regplot(data=industry_ratios_df, x='high_auction_ratio_pct', y='interday_profit_ratio',
                scatter_kws={'alpha': 0.3, 's': 10}, line_kws={'color': 'red'})
    plt.title('Industry High Auction Ratio vs. Interday Profit Ratio (Profit > 1%)')
    plt.xlabel('High Auction Ratio Stock Percentage')
    plt.ylabel('Interday Profit Ratio (Profit > 1%)')
    interday_plot_path = os.path.join(OUTPUT_PLOT_DIR, 'industry_profit_ratio_vs_interday_profit_ratio.png')
    plt.savefig(interday_plot_path)
    plt.close()
    print(f"  - Saved scatter plot to: {interday_plot_path}")

    # Scatter plot for high_auction_ratio_pct vs intraday_profit_ratio
    plt.figure(figsize=(10, 6))
    sns.regplot(data=industry_ratios_df, x='high_auction_ratio_pct', y='intraday_profit_ratio',
                scatter_kws={'alpha': 0.3, 's': 10}, line_kws={'color': 'red'})
    plt.title('Industry High Auction Ratio vs. Intraday Profit Ratio (Profit > 1%)')
    plt.xlabel('High Auction Ratio Stock Percentage')
    plt.ylabel('Intraday Profit Ratio (Profit > 1%)')
    intraday_plot_path = os.path.join(OUTPUT_PLOT_DIR, 'industry_profit_ratio_vs_intraday_profit_ratio.png')
    plt.savefig(intraday_plot_path)
    plt.close()
    print(f"  - Saved scatter plot to: {intraday_plot_path}")
    
    print("\n====== Analysis Finished ======")

if __name__ == "__main__":
    main()