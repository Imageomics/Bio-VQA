import os
import glob
import pandas as pd
import argparse
import random
import hashlib
from PIL import Image

def get_plural(cls_name):
    cls_name = str(cls_name).lower().strip()
    if cls_name == 'kiang':
        return "kiangs (wild asses)"
    elif cls_name in ['cattle', 'sheep']: 
        return cls_name
    else:
        return f"{cls_name}s"

def concat_images(path1, path2, out_path):
    try:
        img1 = Image.open(path1).convert('RGB')
        img2 = Image.open(path2).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        return False
        
    # Get dimensions
    h1, w1 = img1.height, img1.width
    h2, w2 = img2.height, img2.width
    
    # Resize img2 to have the same height as img1, maintaining aspect ratio
    if h1 != h2:
        new_w2 = int(w2 * (h1 / h2))
        img2_resized = img2.resize((new_w2, h1), Image.Resampling.LANCZOS)
    else:
        new_w2 = w2
        img2_resized = img2
        
    # Create new image canvas (no whitespace)
    new_img = Image.new('RGB', (w1 + new_w2, h1))
    
    # Paste images
    new_img.paste(img1, (0, 0))
    new_img.paste(img2_resized, (w1, 0))
    
    # Save output
    new_img.save(out_path)
    return True

def process_datasets(base_dir, output_dir, num_pairs_total, seed):
    # Set seed for reproducibility
    random.seed(seed)
    print(f"Random seed set to: {seed}")

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    images_out_dir = os.path.join(output_dir, "Images")
    os.makedirs(images_out_dir, exist_ok=True)
    
    datasets = [
        "Bees", "Beetles-500", "Camel-500", "Cattle-500", 
        "Fish", "GrevysZebra", "Kiang-500", "Seal-500", 
        "Sheep-500", "Zebra-500"
    ]

    # Organize images by dataset
    images_by_dataset = {ds: [] for ds in datasets}

    print("--- STEP 1: Collecting Metadata ---")
    for dataset in datasets:
        dataset_path = os.path.join(base_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue
            
        csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
        if not csv_files:
            continue
            
        df = pd.read_csv(csv_files[0])
        print(f"Parsing {dataset}...")

        # 1. Bees
        if "Bees" in dataset:
            df_unique = df.drop_duplicates(subset=['image'])
            for _, row in df_unique.iterrows():
                img_name = row['image']
                full_path = os.path.join(dataset_path, "Images", img_name)
                if os.path.exists(full_path):
                    images_by_dataset[dataset].append({
                        'dataset': dataset, 'img_name': img_name, 'full_path': full_path,
                        'qa_data': [{'animal': 'bees', 'answer': row['num_bees']}]
                    })

        # 2. Beetles
        elif "Beetles" in dataset:
            df_unique = df.drop_duplicates(subset=['group_image_path'])
            for _, row in df_unique.iterrows():
                img_name = os.path.basename(row['group_image_path'])
                full_path = os.path.join(dataset_path, row['group_image_path'])
                if os.path.exists(full_path):
                    images_by_dataset[dataset].append({
                        'dataset': dataset, 'img_name': img_name, 'full_path': full_path,
                        'qa_data': [{'animal': 'beetles', 'answer': row['no_of_beetles_in_image']}]
                    })

        # 3. Fish
        elif "Fish" in dataset:
            df_unique = df.drop_duplicates(subset=['image_name'])
            for _, row in df_unique.iterrows():
                img_name = row['image_name']
                full_path = os.path.join(dataset_path, "Images", img_name)
                if os.path.exists(full_path):
                    images_by_dataset[dataset].append({
                        'dataset': dataset, 'img_name': img_name, 'full_path': full_path,
                        'qa_data': [{'animal': 'fish', 'answer': row['num_fish']}]
                    })

        # 4. GrevysZebra
        elif "GrevysZebra" in dataset:
            for img_fname, group in df.groupby('image fname'):
                total_animals = len(group)
                grevys_count = len(group[group['annot species'] == 'zebra_grevys'])
                zebras_count = len(group[group['annot species'].str.contains('zebra', case=False, na=False)])
                
                full_path = os.path.join(dataset_path, "Images", img_fname)
                if os.path.exists(full_path):
                    images_by_dataset[dataset].append({
                        'dataset': dataset, 'img_name': img_fname, 'full_path': full_path,
                        'qa_data': [
                            {'animal': 'animals', 'answer': total_animals},
                            {'animal': 'grevys zebras', 'answer': grevys_count},
                            {'animal': 'zebras', 'answer': zebras_count}
                        ]
                    })

        # 5. WAID Datasets
        else:
            df_unique = df.drop_duplicates(subset=['image_path'])
            for _, row in df_unique.iterrows():
                img_name = os.path.basename(row['image_path'])
                full_path = os.path.join(dataset_path, "Images", img_name)
                
                plural_name = get_plural(row['class_name'])
                
                if os.path.exists(full_path):
                    images_by_dataset[dataset].append({
                        'dataset': dataset, 'img_name': img_name, 'full_path': full_path,
                        'qa_data': [{'animal': plural_name, 'answer': row['num_animals_in_image']}]
                    })

    # Count actual valid datasets to distribute pairs evenly
    valid_datasets = [ds for ds in datasets if len(images_by_dataset[ds]) > 0]
    target_per_dataset = num_pairs_total // len(valid_datasets)
    
    print(f"\n--- STEP 2: Generating Pairs ({target_per_dataset} per dataset) ---")
    pairs = []
    
    for dataset_name in valid_datasets:
        pool = images_by_dataset[dataset_name]
        if len(pool) < 2:
            print(f"Skipping {dataset_name}: Not enough images to form pairs.")
            continue
            
        seen_pairs = set()
        dataset_pairs = []
        attempts = 0
        max_attempts = target_per_dataset * 50 # Safety net
        
        while len(dataset_pairs) < target_per_dataset and attempts < max_attempts:
            attempts += 1
            
            item1 = random.choice(pool)
            item2 = random.choice(pool)
            
            # Order matters for left/right, so (A, B) is different from (B, A)
            pair_id = (item1['full_path'], item2['full_path'])
            
            # 1. Check if we've seen this exact layout before
            if pair_id in seen_pairs:
                continue
                
            # 2. Prevent concatenating an image with itself
            if item1['full_path'] == item2['full_path']:
                continue
                
            # 3. Prevent having the same number of animals
            # We compare the primary count (the first item in qa_data list)
            if item1['qa_data'][0]['answer'] == item2['qa_data'][0]['answer']:
                continue
                
            # Valid pair found
            dataset_pairs.append((item1, item2))
            seen_pairs.add(pair_id)
            
        pairs.extend(dataset_pairs)
        print(f"Generated {len(dataset_pairs)} pairs for {dataset_name} (Attempts: {attempts})")

    print(f"\nTotal image pairs to generate: {len(pairs)}")
    
    all_qa_data = []
    
    print("\n--- STEP 3: Concatenating Images & Building CSV ---")
    for idx, (item1, item2) in enumerate(pairs):
        if idx % 1000 == 0 and idx != 0:
            print(f"Processed {idx}/{len(pairs)} concatenated images...")
            
        # Clean filenames for safety
        name1 = os.path.splitext(item1['img_name'])[0]
        name2 = item2['img_name']
        
        # Adding a short random hash to prevent naming collisions if same images are paired inversely elsewhere
        hash_str = hashlib.md5((item1['full_path'] + item2['full_path']).encode()).hexdigest()[:6]
        concat_img_name = f"concat_{item1['dataset']}_{name1}_{name2}_{hash_str}.jpg"
            
        concat_rel_path = f"Images/{concat_img_name}"
        concat_full_path = os.path.join(output_dir, concat_rel_path)
        
        # Only process image if it doesn't already exist (allows pausing/resuming)
        if not os.path.exists(concat_full_path):
            success = concat_images(item1['full_path'], item2['full_path'], concat_full_path)
        else:
            success = True
        
        if success:
            # Left image questions
            for qa in item1['qa_data']:
                all_qa_data.append({
                    'image': concat_img_name,
                    'image_path': concat_rel_path,
                    'question': f"How many {qa['animal']} are there in the left image?",
                    'answer': qa['answer']
                })
                
            # Right image questions
            for qa in item2['qa_data']:
                all_qa_data.append({
                    'image': concat_img_name,
                    'image_path': concat_rel_path,
                    'question': f"How many {qa['animal']} are there in the right image?",
                    'answer': qa['answer']
                })

    # Save final CSV
    if all_qa_data:
        final_df = pd.DataFrame(all_qa_data)
        out_csv_path = os.path.join(output_dir, "qa.csv")
        final_df.to_csv(out_csv_path, index=False)
        print(f"\nSuccess! Saved {len(final_df)} QA pairs and generated {len(pairs)} concatenated images to {output_dir}")
    else:
        print("No valid data was generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Hard VQA Counting CSV (Same Dataset, Different Counts)")
    parser.add_argument("--base_dir", required=True, help="Base directory containing dataset folders")
    parser.add_argument("--output_dir", required=True, help="Output directory for the qa.csv file and Images folder")
    parser.add_argument("--num_pairs", type=int, default=10000, help="Total number of concatenated image pairs to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    process_datasets(args.base_dir, args.output_dir, args.num_pairs, args.seed)