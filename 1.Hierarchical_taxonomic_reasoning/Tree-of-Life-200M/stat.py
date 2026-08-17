import os
import argparse
import pandas as pd


def generate_dataset_stats(base_dir):

    base_dir = base_dir
    
    output_file = os.path.join(base_dir, "data-stats.txt")
    
    # Taxonomic columns to track
    taxa_cols = [
        'kingdom', 'phylum', 'class', 'order', 'family', 
        'genus', 'species', 'scientific_name', 'common'
    ]
    
    # Find the correct CSVs: {subdir_name}/{subdir_name}.csv
    csv_files = []
    
    if os.path.exists(base_dir):
        # Iterate through items in the base directory
        for item in sorted(os.listdir(base_dir)):
            subdir_path = os.path.join(base_dir, item)
            # If it's a directory, check if the matching .csv exists inside
            if os.path.isdir(subdir_path):
                expected_csv = os.path.join(subdir_path, f"{item}.csv")
                if os.path.exists(expected_csv):
                    csv_files.append(expected_csv)
    
    if not csv_files:
        print(f"No valid {{subdir_name}}.csv files found in subdirectories of {base_dir}")
        return

    all_dfs = []

    with open(output_file, 'w') as f:
        f.write("=========================================\n")
        f.write("      Bio-VQA Dataset Statistics\n")
        f.write("=========================================\n\n")

        # Process each CSV individually
        for csv_path in csv_files:
            dataset_name = os.path.basename(os.path.dirname(csv_path))
            f.write(f"--- {dataset_name} ---\n")
            
            try:
                df = pd.read_csv(csv_path)
                all_dfs.append(df)
                
                # Calculate unique taxa
                for col in taxa_cols:
                    if col in df.columns:
                        unique_vals = df[col].dropna().unique()
                        unique_count = len(unique_vals)
                        
                        # Format the extra string if 1 or 2 values
                        val_str = ""
                        if unique_count in [1, 2, 3, 4, 5]:
                            val_list = [str(v) for v in unique_vals]
                            val_str = f" [ {', '.join(val_list)} ]"
                            
                        f.write(f"  Unique {col.capitalize():<25}: {unique_count}{val_str}\n")
                    else:
                        f.write(f"  Unique {col.capitalize():<25}: [Column Missing]\n")
                
                # Calculate unique images
                if 'local_path' in df.columns:
                    unique_images = df['local_path'].nunique(dropna=True)
                    f.write(f"  Unique Images                   : {unique_images}\n")
                else:
                    f.write("   Unique Images                   : [Column 'local_path' Missing]\n")
                
                
            except Exception as e:
                f.write(f"  [ERROR] Could not read file: {e}\n\n")
                print(f"Error reading {csv_path}: {e}")

        # Process overall merged stats
        f.write("=========================================\n")
        f.write("    OVERALL STATS (All Datasets Merged)\n")
        f.write("=========================================\n")
        
        if all_dfs:
            merged_df = pd.concat(all_dfs, ignore_index=True)
            
            for col in taxa_cols:
                if col in merged_df.columns:
                    unique_vals = merged_df[col].dropna().unique()
                    unique_count = len(unique_vals)
                    
                    # Format the extra string if 1 or 2 values for the overall stats
                    val_str = ""
                    if unique_count in [1, 2, 3, 4, 5]:
                        val_list = [str(v) for v in unique_vals]
                        val_str = f" [ {', '.join(val_list)} ]"
                        
                    f.write(f"  Total Unique {col.capitalize():<19}: {unique_count}{val_str}\n")
            
            if 'local_path' in merged_df.columns:
                unique_images = merged_df['local_path'].nunique(dropna=True)
                f.write(f"  Total Unique Images             : {unique_images}\n")
                
        else:
            f.write("No data available to merge.\n")

    print(f"Process complete. Statistics successfully written to:\n{output_file}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate dataset statistics for Bio-VQA datasets."
    )

    parser.add_argument(
        "--base_dir",
        required=True,
        help="Base directory containing all dataset folders.",
    )

    args = parser.parse_args()

    generate_dataset_stats(args.base_dir)