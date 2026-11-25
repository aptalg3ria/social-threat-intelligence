"""
src/data/loader.py
------------------
Loads the bot detection dataset (CSV) and returns a cleaned DataFrame.
"""

import pandas as pd
import os


BOT_DATASET_PATH = "bot_detection_data.csv"


def load_bot_dataset(path: str = BOT_DATASET_PATH) -> pd.DataFrame:
    """
    Load the Twitter Bot Detection dataset from a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file. Defaults to 'bot_detection_data.csv'.

    Returns
    -------
    pd.DataFrame
        Raw dataset with all original columns.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place 'bot_detection_data.csv' in the project root directory."
        )

    df = pd.read_csv(path)
    print(f"[loader] Loaded {len(df):,} rows from '{path}'.")
    print(f"[loader] Columns: {list(df.columns)}")
    return df


def get_column_summary(df: pd.DataFrame) -> None:
    """Print a quick summary of the dataset columns and class distribution."""
    print("\n--- Dataset Summary ---")
    print(f"Shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    if "Bot Label" in df.columns:
        print(f"\nBot Label distribution:\n{df['Bot Label'].value_counts()}")
    print("----------------------\n")
