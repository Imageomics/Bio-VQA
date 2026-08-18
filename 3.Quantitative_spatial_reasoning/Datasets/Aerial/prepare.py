import os
import pandas as pd
import shutil
import argparse

def get_full_path(img_rel_path, base_dir):
    return os.path.normpath(os.path.join(base_dir, img_rel_path))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fixed Aerial Animal Dataset Selection")
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--output_base_dir", type=str, required=True)
    parser.add_argument("--file_size_kb", type=int, default=20)
    parser.add_argument("--dataset_size", type=int, default=500)

    args = parser.parse_args()

    file_size_limit_bytes = args.file_size_kb * 1024
    dataset_size = args.dataset_size

    csv_path = os.path.join(args.base_dir, "metadata.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        exit()

    df = pd.read_csv(csv_path)

    df['class_name'] = df['class_name'].replace('camelus', 'camel')

    df['full_path'] = df['image_path'].apply(lambda x: get_full_path(x, args.base_dir))

    unique_paths = df['full_path'].unique()

    valid_paths = {p for p in unique_paths if os.path.exists(p) and os.path.getsize(p) > file_size_limit_bytes}
    
    df = df[df['full_path'].isin(valid_paths)].copy()
    
    if df.empty:
        print(f"Error: No images found larger than {args.file_size_kb}KB at {args.base_dir}")
        exit()

    counts = df.groupby(['image_path', 'class_name', 'full_path']).size().reset_index(name='num_animals')
    
    config = {
        'camel':  {'tiers': [(5, 10, 300), (11, 30, 200)]},
        'cattle': {'tiers': [(5, 10, 200), (11, 30, 200), (31, 9999, 100)]},
        'kiang':  {'tiers': [(2, 4, 150), (5, 10, 300), (11, 30, 50)]},
        'seal':   {'tiers': [(31, 9999, 75), (11, 30, 200), (5, 10, 225)]},
        'sheep':  {'tiers': [(5, 10, 200), (11, 30, 200), (31, 9999, 100)]},
        'zebra':  {'tiers': [(2, 4, 100), (5, 10, 250), (11, 30, 150)]}
    }

    for target_animal, setup in config.items():
        animal_dir_name = f"{target_animal.capitalize()}-{dataset_size}"
        cat_out_dir = os.path.join(args.output_base_dir, animal_dir_name)
        img_out_dir = os.path.join(cat_out_dir, "Images")

        cat_pool = counts[counts['class_name'] == target_animal].copy()
        if cat_pool.empty:
            print(f"Skipping {target_animal}: No data after filtering.")
            continue

        cat_pool['file_size'] = cat_pool['full_path'].apply(os.path.getsize)
        
        selected_dfs = []
        already_selected = set()
        
        for low, high, target in setup['tiers']:
            pool = cat_pool[(cat_pool['num_animals'] >= low) & 
                            (cat_pool['num_animals'] <= high) & 
                            (~cat_pool['image_path'].isin(already_selected))]
            
            picked = pool.sort_values(by=['file_size', 'num_animals'], ascending=[False, False]).head(target)
            selected_dfs.append(picked)
            already_selected.update(picked['image_path'].tolist())

        current_total = sum(len(x) for x in selected_dfs)
        if current_total < dataset_size:
            shortfall = dataset_size - current_total
            remaining_pool = cat_pool[~cat_pool['image_path'].isin(already_selected)]
            backfill = remaining_pool.sort_values(by='file_size', ascending=False).head(shortfall)
            selected_dfs.append(backfill)

        final_cat_selection = pd.concat(selected_dfs).head(dataset_size)
        
        selected_image_paths = set(final_cat_selection['image_path'])
        final_metadata = df[(df['image_path'].isin(selected_image_paths)) & 
                            (df['class_name'] == target_animal)].copy()
                            
        os.makedirs(img_out_dir, exist_ok=True)
        final_metadata.to_csv(os.path.join(cat_out_dir, "metadata.csv"), index=False)
        
        for img_p in final_cat_selection['full_path']:
            shutil.copy2(img_p, os.path.join(img_out_dir, os.path.basename(img_p)))