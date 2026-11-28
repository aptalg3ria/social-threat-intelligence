"""
src/data/loader.py
------------------
Loads the Twitter Bot Detection dataset (CSV) and returns a DataFrame.
Resolves dataset paths relative to the project root so loading works reliably
regardless of the current working directory or operating system.

Supports:
  - 'twitter_human_bots_dataset.csv'
  - 'twitter_human_bots_dataset-1.csv' (and numbered variants)
  - 'Data/raw/' or 'Data/' dataset subdirectories
  - legacy 'bot_detection_data.csv'
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.utils.helpers import PROJECT_ROOT


PRIMARY_DATASET_PATH = "twitter_human_bots_dataset.csv"
ALT_PRIMARY_DATASET_PATH = "twitter_human_bots_dataset-1.csv"
LEGACY_DATASET_PATH = "bot_detection_data.csv"

_CANDIDATE_RELATIVE_PATHS: tuple[Path, ...] = (
    Path(PRIMARY_DATASET_PATH),
    Path(ALT_PRIMARY_DATASET_PATH),
    Path("Data") / "raw" / PRIMARY_DATASET_PATH,
    Path("Data") / "raw" / ALT_PRIMARY_DATASET_PATH,
    Path("Data") / PRIMARY_DATASET_PATH,
    Path("Data") / ALT_PRIMARY_DATASET_PATH,
    Path("data") / "raw" / PRIMARY_DATASET_PATH,
    Path("data") / "raw" / ALT_PRIMARY_DATASET_PATH,
    Path("data") / PRIMARY_DATASET_PATH,
    Path("data") / ALT_PRIMARY_DATASET_PATH,
    Path(LEGACY_DATASET_PATH),
)


def resolve_bot_dataset_path(path: str | os.PathLike | None = None) -> Path:
    """
    Locate the Twitter Bot Detection CSV path deterministically relative to
    PROJECT_ROOT (with current-working-directory fallback for custom paths).
    """
    if path is not None:
        explicit = Path(path)
        if explicit.is_absolute():
            if explicit.is_file():
                return explicit
            raise FileNotFoundError(f"Dataset not found at: {path}")

        for base in (Path.cwd(), PROJECT_ROOT):
            candidate = (base / explicit).resolve()
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"Dataset not found at: {path}")

    for base in (PROJECT_ROOT, Path.cwd()):
        for rel in _CANDIDATE_RELATIVE_PATHS:
            candidate = (base / rel).resolve()
            if candidate.is_file():
                return candidate

        for search_dir in (base, base / "Data" / "raw", base / "Data", base / "data"):
            if search_dir.is_dir():
                matches = sorted(search_dir.glob("twitter_human_bots_dataset*.csv"))
                for match in matches:
                    if match.is_file():
                        return match.resolve()

    raise FileNotFoundError(
        f"No bot dataset found under '{PROJECT_ROOT}'. "
        f"Expected '{PRIMARY_DATASET_PATH}' (or '{ALT_PRIMARY_DATASET_PATH}') "
        "in the project root or 'Data/raw/'."
    )


def load_bot_dataset(path: str | os.PathLike | None = None) -> pd.DataFrame:
    """
    Load the Twitter Bot Detection dataset from a CSV file.

    Parameters
    ----------
    path : str or PathLike, optional
        Explicit path to the CSV file. If None, searches standard project-root
        locations automatically.

    Returns
    -------
    pd.DataFrame
        Raw dataset with all original columns.
    """
    resolved_path = resolve_bot_dataset_path(path)
    df = pd.read_csv(resolved_path)
    print(f"[loader] Loaded {len(df):,} rows from '{resolved_path.name}'.")
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
