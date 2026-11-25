"""
src/data/loader.py
------------------
Loads the Twitter Bot Detection dataset (CSV) and returns a DataFrame.
Supports both 'twitter_human_bots_dataset.csv' (primary, real-world dataset)
and legacy 'bot_detection_data.csv'.
"""

import pandas as pd
import os

PRIMARY_DATASET_PATH = "twitter_human_bots_dataset.csv"
LEGACY_DATASET_PATH = "bot_detection_data.csv"


def load_bot_dataset(path: str = None) -> pd.DataFrame:
    """
    Load the Twitter Bot Detection dataset from a CSV file.

    Parameters
    ----------
    path : str, optional
        Path to the CSV file. If None, checks for PRIMARY_DATASET_PATH,
        then LEGACY_DATASET_PATH.

    Returns
    -------
    pd.DataFrame
        Raw dataset with all original columns.
    """
    if path is None:
        if os.path.exists(PRIMARY_DATASET_PATH):
            path = PRIMARY_DATASET_PATH
        elif os.path.exists(LEGACY_DATASET_PATH):
            path = LEGACY_DATASET_PATH
        else:
            raise FileNotFoundError(
                f"No bot dataset found. Please place '{PRIMARY_DATASET_PATH}' "
                "in the project root directory."
            )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)
    print(f"[loader] Loaded {len(df):,} rows from '{path}'.")
    return df


def get_column_summary(df: pd.DataFrame) -> None:
    """Print a summary of dataset columns and target label distribution."""
    print("\n--- Dataset Summary ---")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    for target_col in ["account_type", "Bot Label"]:
        if target_col in df.columns:
            print(f"\nTarget distribution ({target_col}):\n{df[target_col].value_counts()}")
    print("----------------------\n")
