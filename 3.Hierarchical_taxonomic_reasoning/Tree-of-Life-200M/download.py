import duckdb
import pandas as pd
import argparse
import os
import requests
import threading
import csv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# --- Global Sync Objects ---
csv_lock = threading.Lock()
stop_event = threading.Event()
success_counter = 0

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100)
session.mount('http://', adapter)
session.mount('https://', adapter)

def download_image(url, save_path, min_size_kb):
    min_bytes = min_size_kb * 1024
    try:
        r = session.get(url, stream=True, timeout=10)
        if r.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            if os.path.exists(save_path) and os.path.getsize(save_path) >= min_bytes:
                return True
            else:
                if os.path.exists(save_path): os.remove(save_path)
    except:
        if os.path.exists(save_path): os.remove(save_path)
    return False

def process_species_task(group_df, image_dir, images_per_species, min_size_kb, csv_writer, stop_limit, pbar):
    global success_counter
    
    potential_rows = group_df.sample(frac=1)
    local_success = 0
    
    for _, row in potential_rows.iterrows():
        if stop_event.is_set() or success_counter >= stop_limit:
            return
        
        if local_success >= images_per_species:
            break
            
        url = row.get("source_url")
        if url is None or not isinstance(url, str):
            continue

        uuid = row["uuid"]
        ext = url.split(".")[-1].lower() if "." in url else "jpg"
        if len(ext) > 4 or "?" in ext: ext = "jpg"
        
        filename = f"{uuid}.{ext}"
        path = os.path.join(image_dir, filename)
        
        if download_image(url, path, min_size_kb):
            d = row.to_dict()
            d["local_path"] = f"Images/{filename}"
            
            with csv_lock:
                if success_counter < stop_limit:
                    csv_writer.writerow(d)
                    success_counter += 1
                    local_success += 1
                    pbar.update(1)
                
                if success_counter >= stop_limit:
                    stop_event.set()
                    return

def cleanup_stray_images(output_csv, image_dir):
    print("\n[STEP 3] Cleaning up stray images...")
    df = pd.read_csv(output_csv)
    valid_filenames = set(df['local_path'].apply(lambda x: os.path.basename(x)))
    
    all_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    deleted_count = 0
    
    for f in all_files:
        if f not in valid_filenames:
            try:
                os.remove(os.path.join(image_dir, f))
                deleted_count += 1
            except:
                pass
    print(f"[INFO] Deleted {deleted_count} stray images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--taxon_rank", required=True)
    parser.add_argument("--taxon_value", required=True)
    parser.add_argument("--images_per_species", type=int, default=10)
    parser.add_argument("--stop_limit", type=int, default=10000)
    parser.add_argument("--min_size_kb", type=float, default=1024)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--skip_order", action="store_true", help="Skip the 'order' column check (useful for Reptiles)")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    image_dir = os.path.join(args.output_dir, "Images")
    os.makedirs(image_dir, exist_ok=True)

    con = duckdb.connect()

    # --- SQL Taxa Filter ---
    # Define fields to check for NOT NULL
    tax_fields = ["kingdom", "phylum", "class", "family", "genus", "species", "scientific_name", "common"]
    if not args.skip_order:
        tax_fields.append('"order"') # DuckDB requires quotes for the keyword 'order'

    null_filter = " AND ".join([f"{f} IS NOT NULL" for f in tax_fields])
    
    taxon_filter = f"{args.taxon_rank} = '{args.taxon_value}'"
    if "(" in args.taxon_value:
         taxon_filter = f"{args.taxon_rank} IN {args.taxon_value}"

    print(f"[STEP 1] Querying species with COMPLETE metadata for {args.taxon_value}...")
    
    # We query ALL species that match the criteria, ordered by frequency
    query = f"""
        SELECT scientific_name, COUNT(*) as freq
        FROM read_parquet('{args.catalog}')
        WHERE ({taxon_filter}) AND ({null_filter}) AND source_url IS NOT NULL
        GROUP BY scientific_name
        ORDER BY freq DESC
    """
    species_df = con.execute(query).df()
    target_list = species_df["scientific_name"].tolist()
    
    print(f"[INFO] Found {len(target_list)} species with valid metadata.")

    # Prepare CSV header
    sample_row = con.execute(f"SELECT * FROM read_parquet('{args.catalog}') LIMIT 1").df()
    header = list(sample_row.columns) + ["local_path"]

    with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        pbar = tqdm(total=args.stop_limit, desc="Downloading")

        # Load data chunks to avoid memory issues if target_list is huge
        # We load enough data to hit our stop_limit easily
        query_str = ", ".join([f"'{s}'" for s in target_list[:args.stop_limit]]) # Limit query to a safe buffer
        df_full = con.sql(f"SELECT * FROM read_parquet('{args.catalog}') WHERE scientific_name IN ({query_str}) AND ({null_filter})").df()
        grouped = df_full.groupby("scientific_name")

        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            for s_name in target_list:
                if s_name in grouped.groups:
                    executor.submit(process_species_task, grouped.get_group(s_name), 
                                    image_dir, args.images_per_species, args.min_size_kb, 
                                    writer, args.stop_limit, pbar)
            executor.shutdown(wait=True)
        pbar.close()

    cleanup_stray_images(args.output_csv, image_dir)

    # Final Sanity Check
    print(f"\n[STEP 4] Final Sanity Check...")
    final_csv_rows = len(pd.read_csv(args.output_csv))
    final_images = len([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])

    print(f"---------------------------")
    print(f"Target Stop Limit: {args.stop_limit}")
    print(f"Final CSV Rows:    {final_csv_rows}")
    print(f"Files in Images/:  {final_images}")
    print(f"---------------------------")

    if final_csv_rows == final_images:
        print("✅ SUCCESS: Data is clean and complete.")
    else:
        print("❌ ERROR: Mismatch detected.")