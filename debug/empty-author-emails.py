import pandas as pd
import sys
from pathlib import Path

# Directory containing parquet files
parquet_dir = sys.argv[1] if len(sys.argv) > 1 else "data/processed/rq2/commit_patterns"

try:
    # Read all parquet files in directory
    print("\n=== Reading parquet files ===")
    df = pd.read_parquet(parquet_dir)

    print("\n=== Author Email Analysis ===")
    print("\nUnique email value counts (top 20):")
    print(df["author_email"].value_counts().head(20))

    print("\n=== Special Cases ===")
    # Check for various types of empty/special values
    print("\nUnique values with length <= 1:")
    # Fix: First get unique values, then filter
    unique_emails = df["author_email"].unique()
    short_emails = [email for email in unique_emails if len(str(email)) <= 1]
    if short_emails:
        print(short_emails)
    else:
        print("No emails with length <= 1 found")

    print("\nSuspicious values check:")
    suspicious = [email for email in unique_emails if not "@" in str(email)]
    suspicious_emails_info = df[df["author_email"].isin(suspicious)][
        ["repo_name", "author_email", "commit_hash"]
    ]
    if not suspicious_emails_info.empty:
        print(suspicious_emails_info)
    else:
        print("No suspicious emails found")
    if suspicious:
        print("\nEmails without '@' symbol:")
        print(suspicious)
    else:
        print("All emails contain '@' symbol")

except Exception as e:
    print(f"Error reading parquet files from {parquet_dir}")
    print(f"Error: {str(e)}")
