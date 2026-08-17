import os
import glob
import re
import argparse
import pandas as pd


def process_chunks(base_path):

    # 1. Iterate over taxonomic group directories (e.g., Amphibians-10K)
    for tax_dir in os.listdir(base_path):
        tax_path = os.path.join(base_path, tax_dir)
        if not os.path.isdir(tax_path):
            continue
            
        # 2. Iterate over items inside each taxonomic group
        for model_dir in os.listdir(tax_path):
            model_path = os.path.join(tax_path, model_dir)
            
            # Ensure the path is actually a directory (safeguard against stray files)
            if not os.path.isdir(model_path):
                continue

            # 3. Search for any chunk files in this directory
            search_pattern = os.path.join(model_path, '*chunk*.csv')
            
            chunk_files = sorted(
                glob.glob(search_pattern),
                key=lambda p: int(re.search(r'chunk(\d+)', os.path.basename(p), re.I).group(1))
            )

            # 4. Check condition: are there chunk files?
            has_chunks = len(chunk_files) > 0

            # 5. Apply the skip logic: skip if no chunks found
            if not has_chunks:
                print(f"Skipping: {tax_dir}/{model_dir} (No chunks found)")
                continue

            print(f"Merging {len(chunk_files)} chunks in {tax_dir}/{model_dir}...")
            
            # 6. Read and concatenate all chunk DataFrames
            df_list = [pd.read_csv(f) for f in chunk_files]
            merged_df = pd.concat(df_list, ignore_index=True)
            
            # 7. Generate the new merged filename
            sample_filename = os.path.basename(chunk_files[0])
            
            # Remove "chunk{x}_"
            new_filename = re.sub(r'chunk\d+_', '', sample_filename)
            
            # Capitalize 'b' to 'B' only if it directly follows a digit (e.g., 32b -> 32B)
            new_filename = re.sub(r'(\d)b', r'\g<1>B', new_filename)
            
            output_filepath = os.path.join(model_path, new_filename)
            
            # 8. Save the merged file
            merged_df.to_csv(output_filepath, index=False)
            print(f"Saved: {output_filepath}\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Merge chunk csvs into a single csv for each model directory."
    )
    parser.add_argument(
        "base_dir",
        help="Base directory containing all dataset folders.",
    )

    args = parser.parse_args()
    base_path = args.base_dir

    process_chunks(base_path)
    print("Processing complete.")

