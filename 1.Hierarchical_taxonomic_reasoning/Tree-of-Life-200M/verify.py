
import os
import pandas as pd
import argparse

def report_missing(df):

    levels = ['phylum', 'class', 'order', 'family', 'genus', 'species', 'scientific_name', 'common']
    
    for col in levels:
        if col not in df.columns:
            print(f"  - {col:15}: COLUMN MISSING")
            continue
            
        is_missing = (
            df[col].isna() | 
            (df[col].astype(str).str.lower().str.strip() == 'nan') | 
            (df[col].astype(str).str.strip() == '')
        )
        
        nan_count = is_missing.sum()
        print(f"  - {col:15}: {nan_count} missing")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True, help="Path to the directory containing CSVs")
    args = parser.parse_args()

    csv_files = []
    for root, _, files in os.walk(args.base_dir):
        for file in files:
            if file.lower().endswith(".csv") and not file.endswith("-QA.csv"):
                csv_files.append(os.path.join(root, file))

    if not csv_files:
        print(f"No CSV files found in {args.base_dir}")
    else:
        for csv_path in csv_files:
            print("\n" + "="*60)
            print(f"FILE: {os.path.basename(csv_path)}")

            try:
                df = pd.read_csv(csv_path)
                print(f"Total Rows: {len(df)}")
                report_missing(df)
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")

    print("\n" + "="*60)
    print("[DONE] Verification process complete.")