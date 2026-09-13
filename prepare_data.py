import kagglehub
import os
import pandas as pd
from pathlib import Path


def find_project_root(marker: str = "requirements.txt") -> Path:
    """Walks up from this file's location until it finds requirements.txt,
    so the output CSV always lands in the same place no matter which folder
    you ran 'python prepare_data.py' from."""
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / marker).exists():
            return candidate
    return Path.cwd()


PROJECT_ROOT = find_project_root()
OUTPUT_PATH = PROJECT_ROOT / "apple_conversations.csv"

print("Downloading/Locating dataset...")
dataset_path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
csv_path = os.path.join(dataset_path, 'twcs', 'twcs.csv')

print("Reading twcs.csv...")
df = pd.read_csv(csv_path)

print("Extracting AppleSupport replies...")
apple_replies = df[df['author_id'] == 'AppleSupport']

print("Joining with customer questions...")
# We want the customer question that AppleSupport replied to.
# We match apple_replies.in_response_to_tweet_id with df.tweet_id
conversations = apple_replies.merge(
    df,
    left_on='in_response_to_tweet_id',
    right_on='tweet_id',
    suffixes=('_brand', '_customer')
)

# Keep relevant columns
cleaned_data = conversations[[
    'tweet_id_customer',
    'author_id_customer',
    'text_customer',
    'tweet_id_brand',
    'text_brand'
]]

cleaned_data = cleaned_data.rename(columns={
    'tweet_id_customer': 'customer_tweet_id',
    'author_id_customer': 'customer_id',
    'text_customer': 'customer_query',
    'tweet_id_brand': 'brand_tweet_id',
    'text_brand': 'brand_reply'
})

print(f"Found {len(cleaned_data)} full conversations.")
cleaned_data.to_csv(OUTPUT_PATH, index=False)
print(f"Saved to {OUTPUT_PATH}")

# Display a few samples
print("\nSamples:")
for i, row in cleaned_data.head(3).iterrows():
    print(f"Customer: {row['customer_query']}")
    print(f"AppleSupport: {row['brand_reply']}\n")
