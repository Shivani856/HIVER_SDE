import pandas as pd
import numpy as np
from pathlib import Path


def find_project_root(marker: str = "requirements.txt") -> Path:
    """Same logic as prepare_data.py: walk up until requirements.txt is found,
    so this script finds apple_conversations.csv regardless of whether it's
    run from the project root or from inside src/."""
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / marker).exists():
            return candidate
    return Path.cwd()


PROJECT_ROOT = find_project_root()
DEFAULT_INPUT = PROJECT_ROOT / "apple_conversations.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "golden_set.csv"


def create_golden_set(input_csv=DEFAULT_INPUT, output_csv=DEFAULT_OUTPUT, n_samples=200):
    print(f"Loading conversations from {input_csv}...")
    if not Path(input_csv).exists():
        raise FileNotFoundError(
            f"Could not find {input_csv}. Run 'python prepare_data.py' first "
            f"(from any folder — it now always saves to the project root)."
        )
    df = pd.read_csv(input_csv).dropna(subset=['customer_query', 'brand_reply'])

    # We want a mix of short and long queries to test the agent properly
    # Calculate word count of customer queries
    df['query_length'] = df['customer_query'].apply(lambda x: len(str(x).split()))

    # Stratify by length: short (<10 words), medium (10-25 words), long (>25 words)
    df['length_category'] = pd.cut(df['query_length'], bins=[0, 10, 25, 1000], labels=['short', 'medium', 'long'])

    # Sample proportionally or evenly. Let's do roughly evenly.
    sampled = df.groupby('length_category', observed=False, group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), n_samples // 3), random_state=42)
    ).reset_index(drop=True)

    # If we are short of n_samples, pad with random samples
    if len(sampled) < n_samples:
        remaining = n_samples - len(sampled)
        pad = df.drop(sampled.index, errors='ignore').sample(n=min(remaining, len(df) - len(sampled)), random_state=42)
        sampled = pd.concat([sampled, pad]).reset_index(drop=True)

    sampled = sampled.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

    # Add empty columns for human labelling
    sampled['Golden_Intent'] = ""
    sampled['Golden_Escalate'] = ""  # True/False
    sampled['Golden_Reply_Quality'] = ""  # 1 to 5 scale

    # Drop intermediate columns
    sampled = sampled.drop(columns=['query_length', 'length_category'])

    # Keep only what's needed for labeling
    output_cols = [
        'customer_tweet_id', 'customer_query', 'brand_reply',
        'Golden_Intent', 'Golden_Escalate', 'Golden_Reply_Quality'
    ]
    sampled[output_cols].to_csv(output_csv, index=False)

    print(f"Golden set template created at {output_csv} with {len(sampled)} rows.")
    print("Please open this file in Excel/Sheets and fill in the 'Golden_' columns to complete your Golden Set.")


if __name__ == "__main__":
    create_golden_set()
