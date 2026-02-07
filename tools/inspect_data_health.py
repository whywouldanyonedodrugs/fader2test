import pandas as pd
import glob
import os
from pathlib import Path

# Use the linked full directory
parquet_dir = Path("shortonly/parquet")

# Find one parquet file
files = list(parquet_dir.glob("*.parquet"))
if not files:
    print("Error: No parquet files found in shortonly/parquet")
    exit(1)

target = files[0]
print(f"Inspecting: {target.name}")

try:
    df = pd.read_parquet(target)
    print("\nColumns found:", list(df.columns))
    print(f"Total Rows: {len(df)}")
    
    # Check Funding
    if "funding_rate" in df.columns:
        valid = df["funding_rate"].notna().sum()
        print(f"Funding Rate: {valid} valid rows ({valid/len(df):.1%} coverage)")
        if valid > 0:
            print(f"Sample Funding: {df['funding_rate'].dropna().iloc[0]}")
    else:
        print("Funding Rate: COLUMN MISSING")

    # Check OI
    if "open_interest" in df.columns:
        valid = df["open_interest"].notna().sum()
        print(f"Open Interest: {valid} valid rows ({valid/len(df):.1%} coverage)")
    else:
        print("Open Interest: COLUMN MISSING")

except Exception as e:
    print(f"Error reading file: {e}")
