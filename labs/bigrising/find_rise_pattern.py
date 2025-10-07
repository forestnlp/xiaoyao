import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import talib

# --- Constants ---
LABS_DIR = r'd:\workspace\xiaoyao\labs\bigrising'
DATA_DIR = r'D:\workspace\xiaoyao\data'
POSITIVE_FEATURES_FILENAME = 'pre_rise_features.csv'
WIDE_TABLE_FILENAME = 'wide_table.parquet'

def enrich_with_talib_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches the dataframe with TA-Lib features, calculating them per stock.
    """
    print("  - Enriching data with TA-Lib features for all stocks...")
    
    # Ensure data is sorted for consistent TA calculations
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['stock_code', 'date'])

    # Use a list to collect enriched data for each stock
    enriched_parts = []
    
    # Group by stock and calculate features
    for stock, stock_df in df.groupby('stock_code'):
        # Ensure we have enough data to calculate features
        if len(stock_df) > 14: # 14 is a common lookback period
            stock_df = stock_df.copy() # Avoid SettingWithCopyWarning
            stock_df['rsi_14'] = talib.RSI(stock_df['close'], timeperiod=14)
            macd, macdsignal, macdhist = talib.MACD(stock_df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
            stock_df['macd'] = macd
            stock_df['macdsignal'] = macdsignal
            stock_df['macdhist'] = macdhist
            upper, middle, lower = talib.BBANDS(stock_df['close'], timeperiod=20)
            stock_df['bband_upper'] = upper
            stock_df['bband_middle'] = middle
            stock_df['bband_lower'] = lower
            stock_df['atr_14'] = talib.ATR(stock_df['high'], stock_df['low'], stock_df['close'], timeperiod=14)
            stock_df['adx_14'] = talib.ADX(stock_df['high'], stock_df['low'], stock_df['close'], timeperiod=14)
            enriched_parts.append(stock_df)
            
    if not enriched_parts:
        print("  - WARNING: No stock had enough data to calculate features.")
        return pd.DataFrame()

    # Combine all processed stock dataframes
    enriched_df = pd.concat(enriched_parts)
    
    # Drop rows where initial feature values are NaN due to lookback periods
    enriched_df.dropna(inplace=True)
    
    print(f"  - Feature enrichment complete. Resulting shape: {enriched_df.shape}")
    return enriched_df

def load_and_prepare_data():
    """Loads positive and negative samples and prepares them for modeling."""
    print("--- Loading and Preparing Data ---")
    
    # 1. Load Positive Samples (days before a rise)
    positive_path = os.path.join(LABS_DIR, POSITIVE_FEATURES_FILENAME)
    positives = pd.read_csv(positive_path)
    positives['target'] = 1
    print(f"  - Loaded {len(positives)} positive samples.")

    # 2. Load all data and enrich with features to create a pool for negative samples
    print("  - Loading full dataset for negative sampling...")
    wide_table_path = os.path.join(DATA_DIR, WIDE_TABLE_FILENAME)
    all_data = pd.read_parquet(wide_table_path)
    
    # Enrich the entire dataset with TA-Lib features
    all_data_featured = enrich_with_talib_features(all_data)
    
    if all_data_featured.empty:
        print("  - ERROR: Feature enrichment resulted in an empty DataFrame. Cannot proceed.")
        return pd.DataFrame(), []

    # 3. Create Negative Samples
    print("  - Creating negative samples...")
    
    # Identify the positive samples in the full dataset to exclude them from negative sampling
    # Convert dates to a consistent string format for comparison
    positives['date'] = pd.to_datetime(positives['date']).dt.strftime('%Y-%m-%d')
    all_data_featured['date'] = pd.to_datetime(all_data_featured['date']).dt.strftime('%Y-%m-%d')
    
    positive_indices = positives.set_index(['stock_code', 'date']).index
    all_data_featured_indexed = all_data_featured.set_index(['stock_code', 'date'])
    
    # Get indices that are NOT in the positive set
    negative_candidates_indices = all_data_featured_indexed.index.difference(positive_indices)
    negatives_candidates = all_data_featured_indexed.loc[negative_candidates_indices].reset_index()

    # Sample a number of negative examples equal to the positive ones for a balanced dataset
    if len(negatives_candidates) < len(positives):
        print(f"  - WARNING: Not enough negative candidates ({len(negatives_candidates)}) to match positive samples ({len(positives)}). Using all available candidates.")
        n_samples = len(negatives_candidates)
    else:
        n_samples = len(positives)

    negatives = negatives_candidates.sample(n=n_samples, random_state=42)
    negatives['target'] = 0
    print(f"  - Sampled {len(negatives)} negative samples.")

    # 4. Combine and Finalize
    feature_cols = ['rsi_14', 'macd', 'macdsignal', 'macdhist', 'bband_upper', 'bband_middle', 'bband_lower', 'atr_14', 'adx_14']
    final_df = pd.concat([positives[feature_cols + ['target']], negatives[feature_cols + ['target']]], ignore_index=True)
    
    # Final check for any NaNs
    model_df = final_df.dropna()
    
    print(f"  - Final combined dataset shape: {model_df.shape}")
    return model_df, feature_cols

def train_and_evaluate(df: pd.DataFrame, feature_cols: list):
    """Trains a decision tree and evaluates its performance."""
    print("\n--- Training and Evaluating Decision Tree ---")
    
    X = df[feature_cols]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # We use a shallow tree to get simple, interpretable rules
    tree_classifier = DecisionTreeClassifier(max_depth=4, random_state=42, min_samples_leaf=100)
    tree_classifier.fit(X_train, y_train)
    
    # --- Evaluation ---
    y_pred = tree_classifier.predict(X_test)
    precision = precision_score(y_test, y_pred)
    
    print(f"  - Model Precision on Test Set: {precision:.2%}")
    
    if precision < 0.30:
        print("  - WARNING: Model precision is below the 30% target.")
    else:
        print("  - SUCCESS: Model precision meets the 30% target!")

    # --- Feature Importance ---
    print("\n  - Feature Importances:")
    importances = pd.Series(tree_classifier.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(importances)

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pre-Rise'], yticklabels=['Normal', 'Pre-Rise'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    cm_path = os.path.join(LABS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"  - Confusion matrix saved to {cm_path}")

    # --- Extract and Display Rules ---
    print("\n--- Decision Tree Rules (Potential Pre-Rise Patterns) ---")
    rules = export_text(tree_classifier, feature_names=feature_cols)
    print(rules)
    
    return rules

def main():
    """Main function to run the pattern finding pipeline."""
    dataset, features = load_and_prepare_data()
    if not dataset.empty:
        train_and_evaluate(dataset, features)

if __name__ == "__main__":
    main()