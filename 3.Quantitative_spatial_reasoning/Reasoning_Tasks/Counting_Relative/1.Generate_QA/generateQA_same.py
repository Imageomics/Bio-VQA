# import os
# import glob
# import pandas as pd
# import argparse
# import random
# import hashlib
# from PIL import Image

# def get_plural(cls_name):
#     cls_name = str(cls_name).lower().strip()
#     if cls_name == 'kiang':
#         return "kiangs (wild asses)"
#     elif cls_name in ['cattle', 'sheep']: 
#         return cls_name
#     else:
#         return f"{cls_name}s"

# def concat_4_images(p1, p2, p3, p4, out_path):
#     try:
#         i1 = Image.open(p1).convert('RGB')
#         i2 = Image.open(p2).convert('RGB')
#         i3 = Image.open(p3).convert('RGB')
#         i4 = Image.open(p4).convert('RGB')
#     except Exception as e:
#         print(f"Error opening image: {e}")
#         return False

#     # Row 1 (c1): Resize img2 to match img1's height
#     h1 = i1.height
#     w2_new = int(i2.width * (h1 / i2.height))
#     i2_res = i2.resize((w2_new, h1), Image.Resampling.LANCZOS)
#     row1_w = i1.width + w2_new

#     row1 = Image.new('RGB', (row1_w, h1))
#     row1.paste(i1, (0, 0))
#     row1.paste(i2_res, (i1.width, 0))

#     # Row 2 (c2): Resize img4 to match img3's height
#     h3 = i3.height
#     w4_new = int(i4.width * (h3 / i4.height))
#     i4_res = i4.resize((w4_new, h3), Image.Resampling.LANCZOS)
#     row2_w_temp = i3.width + w4_new

#     row2_temp = Image.new('RGB', (row2_w_temp, h3))
#     row2_temp.paste(i3, (0, 0))
#     row2_temp.paste(i4_res, (i3.width, 0))

#     # Scale Row 2 proportionally so its width matches Row 1
#     scale = row1_w / row2_w_temp
#     row2_h_final = int(h3 * scale)
#     row2_final = row2_temp.resize((row1_w, row2_h_final), Image.Resampling.LANCZOS)

#     # Final Canvas: Combine Row 1 and Row 2 vertically
#     final_img = Image.new('RGB', (row1_w, h1 + row2_h_final))
#     final_img.paste(row1, (0, 0))
#     final_img.paste(row2_final, (0, h1))

#     final_img.save(out_path)
#     return True

# def process_datasets(base_dir, output_dir, num_quads_total, seed):
#     random.seed(seed)
#     print(f"Random seed set to: {seed}")

#     os.makedirs(output_dir, exist_ok=True)
#     images_out_dir = os.path.join(output_dir, "Images")
#     os.makedirs(images_out_dir, exist_ok=True)
    
#     datasets = [
#         "Bees", "Beetles-500", "Camel-500", "Cattle-500", 
#         "Fish", "GrevysZebra", "Kiang-500", "Seal-500", 
#         "Sheep-500", "Zebra-500"
#     ]

#     images_by_dataset = {ds: [] for ds in datasets}

#     print("--- STEP 1: Collecting Metadata ---")
#     for dataset in datasets:
#         dataset_path = os.path.join(base_dir, dataset)
#         if not os.path.isdir(dataset_path):
#             continue
            
#         csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
#         if not csv_files:
#             continue
            
#         df = pd.read_csv(csv_files[0])
#         print(f"Parsing {dataset}...")

#         # 1. Bees
#         if "Bees" in dataset:
#             df_unique = df.drop_duplicates(subset=['image'])
#             for _, row in df_unique.iterrows():
#                 img_name = row['image']
#                 full_path = os.path.join(dataset_path, "Images", img_name)
#                 if os.path.exists(full_path):
#                     images_by_dataset[dataset].append({
#                         'full_path': full_path,
#                         'qa_data': [{'animal': 'bees', 'answer': row['num_bees']}]
#                     })

#         # 2. Beetles
#         elif "Beetles" in dataset:
#             df_unique = df.drop_duplicates(subset=['group_image_path'])
#             for _, row in df_unique.iterrows():
#                 full_path = os.path.join(dataset_path, row['group_image_path'])
#                 if os.path.exists(full_path):
#                     images_by_dataset[dataset].append({
#                         'full_path': full_path,
#                         'qa_data': [{'animal': 'beetles', 'answer': row['no_of_beetles_in_image']}]
#                     })

#         # 3. Fish
#         elif "Fish" in dataset:
#             df_unique = df.drop_duplicates(subset=['image_name'])
#             for _, row in df_unique.iterrows():
#                 img_name = row['image_name']
#                 full_path = os.path.join(dataset_path, "Images", img_name)
#                 if os.path.exists(full_path):
#                     images_by_dataset[dataset].append({
#                         'full_path': full_path,
#                         'qa_data': [{'animal': 'fish', 'answer': row['num_fish']}]
#                     })

#         # 4. GrevysZebra
#         elif "GrevysZebra" in dataset:
#             for img_fname, group in df.groupby('image fname'):
#                 total_animals = len(group)
#                 grevys_count = len(group[group['annot species'] == 'zebra_grevys'])
#                 zebras_count = len(group[group['annot species'].str.contains('zebra', case=False, na=False)])
                
#                 full_path = os.path.join(dataset_path, "Images", img_fname)
#                 if os.path.exists(full_path):
#                     images_by_dataset[dataset].append({
#                         'full_path': full_path,
#                         'qa_data': [
#                             {'animal': 'animals', 'answer': total_animals},
#                             {'animal': 'grevys zebras', 'answer': grevys_count},
#                             {'animal': 'zebras', 'answer': zebras_count}
#                         ]
#                     })

#         # 5. WAID Datasets
#         else:
#             df_unique = df.drop_duplicates(subset=['image_path'])
#             for _, row in df_unique.iterrows():
#                 img_name = os.path.basename(row['image_path'])
#                 full_path = os.path.join(dataset_path, "Images", img_name)
                
#                 if os.path.exists(full_path):
#                     images_by_dataset[dataset].append({
#                         'full_path': full_path,
#                         'qa_data': [{'animal': get_plural(row['class_name']), 'answer': row['num_animals_in_image']}]
#                     })

#     valid_datasets = [ds for ds in datasets if len(images_by_dataset[ds]) >= 4]
#     if not valid_datasets:
#         print("No datasets have enough images (minimum 4) to generate quadruplets.")
#         return

#     target_per_dataset = num_quads_total // len(valid_datasets)
    
#     print(f"\n--- STEP 2: Generating Quadruplets ({target_per_dataset} per dataset) ---")
#     all_quadruplets = []
    
#     labels = [
#         "Image A (top-left)", 
#         "Image B (top-right)", 
#         "Image C (bottom-left)", 
#         "Image D (bottom-right)"
#     ]

#     for dataset_name in valid_datasets:
#         pool = images_by_dataset[dataset_name]
#         seen_combinations = set()
#         dataset_quads = []
#         attempts = 0
#         max_attempts = target_per_dataset * 100 
        
#         while len(dataset_quads) < target_per_dataset and attempts < max_attempts:
#             attempts += 1
            
#             # Pick 4 random, distinct images from the pool
#             selected = random.sample(pool, 4)
#             i1, i2, i3, i4 = selected
            
#             # Hash a sorted tuple of paths so we never pick the EXACT same 4 images again, 
#             # even in a different permutation, to maximize diversity.
#             combo_signature = tuple(sorted([i1['full_path'], i2['full_path'], i3['full_path'], i4['full_path']]))
#             if combo_signature in seen_combinations:
#                 continue
                
#             valid_questions_for_this_quad = []
            
#             # Check all properties (1 for most datasets, 3 for GrevysZebra)
#             for q_idx in range(len(i1['qa_data'])):
#                 animal = i1['qa_data'][q_idx]['animal']
#                 counts = [
#                     i1['qa_data'][q_idx]['answer'],
#                     i2['qa_data'][q_idx]['answer'],
#                     i3['qa_data'][q_idx]['answer'],
#                     i4['qa_data'][q_idx]['answer']
#                 ]
#                 max_count = max(counts)
                
#                 # The question is only answerable if there is exactly ONE image with the maximum count.
#                 if counts.count(max_count) == 1:
#                     max_index = counts.index(max_count)
#                     valid_questions_for_this_quad.append({
#                         'animal': animal,
#                         'answer': labels[max_index]
#                     })
                    
#             if not valid_questions_for_this_quad:
#                 # Discard this quad entirely, there is no unambiguous winner for any category
#                 continue

#             # Store successful quad
#             seen_combinations.add(combo_signature)
#             dataset_quads.append({
#                 'dataset': dataset_name,
#                 'images': [i1, i2, i3, i4],
#                 'questions': valid_questions_for_this_quad
#             })
            
#         all_quadruplets.extend(dataset_quads)
#         print(f"Generated {len(dataset_quads)} valid quadruplets for {dataset_name} (Attempts: {attempts})")

#     print(f"\nTotal quadruplets to generate: {len(all_quadruplets)}")
    
#     all_qa_data = []
    
#     print("\n--- STEP 3: Concatenating Images & Building CSV ---")
#     for idx, quad in enumerate(all_quadruplets):
#         if idx % 1000 == 0 and idx != 0:
#             print(f"Processed {idx}/{len(all_quadruplets)} quadruplet grids...")
            
#         i1, i2, i3, i4 = quad['images']
        
#         # Generate a safe, short hash for the filename based on the 4 filepaths
#         paths_concat = "".join([i['full_path'] for i in quad['images']])
#         short_hash = hashlib.md5(paths_concat.encode()).hexdigest()[:8]
#         concat_img_name = f"quad_{quad['dataset']}_{short_hash}.jpg"
            
#         concat_rel_path = f"Images/{concat_img_name}"
#         concat_full_path = os.path.join(output_dir, concat_rel_path)
        
#         if not os.path.exists(concat_full_path):
#             success = concat_4_images(i1['full_path'], i2['full_path'], i3['full_path'], i4['full_path'], concat_full_path)
#         else:
#             success = True
        
#         if success:
#             for q_data in quad['questions']:
#                 all_qa_data.append({
#                     'image': concat_img_name,
#                     'image_path': concat_rel_path,
#                     'question': f"Which of these 4 images contains most number of {q_data['animal']}: Image A (top-left), Image B (top-right), Image C (bottom-left), Image D (bottom-right)?",
#                     'answer': q_data['answer']
#                 })

#     # Save final CSV
#     if all_qa_data:
#         final_df = pd.DataFrame(all_qa_data)
#         out_csv_path = os.path.join(output_dir, "qa.csv")
#         final_df.to_csv(out_csv_path, index=False)
#         print(f"\nSuccess! Saved {len(final_df)} QA pairs and generated {len(all_quadruplets)} grid images to {output_dir}")
#     else:
#         print("No valid data was generated.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Generate Quad VQA Counting CSV (2x2 Grid)")
#     parser.add_argument("--base_dir", required=True, help="Base directory containing dataset folders")
#     parser.add_argument("--output_dir", required=True, help="Output directory for the qa.csv file and Images folder")
#     parser.add_argument("--num_quads", type=int, default=10000, help="Total number of 2x2 grid images to generate")
#     parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
#     args = parser.parse_args()
#     process_datasets(args.base_dir, args.output_dir, args.num_quads, args.seed)





import os
import glob
import pandas as pd
import argparse
import random
import hashlib
from PIL import Image, ImageOps

def get_plural(cls_name):
    cls_name = str(cls_name).lower().strip()
    if cls_name == 'kiang':
        return "kiangs (wild asses)"
    elif cls_name in ['cattle', 'sheep']: 
        return cls_name
    else:
        return f"{cls_name}s"

def concat_4_images(p1, p2, p3, p4, out_path, border_width=3):
    try:
        i1 = Image.open(p1).convert('RGB')
        i2 = Image.open(p2).convert('RGB')
        i3 = Image.open(p3).convert('RGB')
        i4 = Image.open(p4).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        return False

    # --- Row 1 ---
    # Resize img2 to match img1's height
    h1 = i1.height
    w2_new = int(i2.width * (h1 / i2.height))
    i2_res = i2.resize((w2_new, h1), Image.Resampling.LANCZOS)
    
    # Add borders
    i1_b = ImageOps.expand(i1, border=border_width, fill='black')
    i2_b = ImageOps.expand(i2_res, border=border_width, fill='black')
    
    # Merge Row 1
    row1_w = i1_b.width + i2_b.width
    row1_h = i1_b.height
    row1 = Image.new('RGB', (row1_w, row1_h), 'black')
    row1.paste(i1_b, (0, 0))
    row1.paste(i2_b, (i1_b.width, 0))

    # --- Row 2 ---
    # Resize img4 to match img3's height
    h3 = i3.height
    w4_new = int(i4.width * (h3 / i4.height))
    i4_res = i4.resize((w4_new, h3), Image.Resampling.LANCZOS)
    
    # Add borders
    i3_b = ImageOps.expand(i3, border=border_width, fill='black')
    i4_b = ImageOps.expand(i4_res, border=border_width, fill='black')
    
    # Merge Row 2
    row2_w_temp = i3_b.width + i4_b.width
    row2_h_temp = i3_b.height
    row2_temp = Image.new('RGB', (row2_w_temp, row2_h_temp), 'black')
    row2_temp.paste(i3_b, (0, 0))
    row2_temp.paste(i4_b, (i3_b.width, 0))

    # --- Final Composition ---
    # Scale Row 2 proportionally so its width perfectly matches Row 1
    scale = row1_w / row2_w_temp
    row2_h_final = int(row2_h_temp * scale)
    row2_final = row2_temp.resize((row1_w, row2_h_final), Image.Resampling.LANCZOS)

    # Combine vertically
    final_img = Image.new('RGB', (row1_w, row1_h + row2_h_final), 'black')
    final_img.paste(row1, (0, 0))
    final_img.paste(row2_final, (0, row1_h))

    final_img.save(out_path)
    return True

def process_datasets(base_dir, output_dir, num_quads_total, seed):
    random.seed(seed)
    print(f"Random seed set to: {seed}")

    os.makedirs(output_dir, exist_ok=True)
    images_out_dir = os.path.join(output_dir, "Images")
    os.makedirs(images_out_dir, exist_ok=True)
    
    datasets = [
        "Bees", "Beetles-500", "Camel-500", "Cattle-500", 
        "Fish", "GrevysZebra", "Kiang-500", "Seal-500", 
        "Sheep-500", "Zebra-500"
    ]

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

        if "Bees" in dataset:
            df_unique = df.drop_duplicates(subset=['image'])
            for _, row in df_unique.iterrows():
                img_name = row['image']
                full_path = os.path.join(dataset_path, "Images", img_name)
                if os.path.exists(full_path):
                    images_by_dataset[dataset].append({
                        'full_path': full_path,
                        'qa_data': [{'animal': 'bees', 'answer': row['num_bees']}]
                    })

        elif "Beetles" in dataset:
            df_unique = df.drop_duplicates(subset=['group_image_path'])
            for _, row in df_unique.iterrows():
                full_path = os.path.join(dataset_path, row['group_image_path'])
                if os.path.exists(full_path):
                    images_by_dataset[dataset].append({
                        'full_path': full_path,
                        'qa_data': [{'animal': 'beetles', 'answer': row['no_of_beetles_in_image']}]
                    })

        elif "Fish" in dataset:
            df_unique = df.drop_duplicates(subset=['image_name'])
            for _, row in df_unique.iterrows():
                img_name = row['image_name']
                full_path = os.path.join(dataset_path, "Images", img_name)
                if os.path.exists(full_path):
                    images_by_dataset[dataset].append({
                        'full_path': full_path,
                        'qa_data': [{'animal': 'fish', 'answer': row['num_fish']}]
                    })

        elif "GrevysZebra" in dataset:
            for img_fname, group in df.groupby('image fname'):
                total_animals = len(group)
                grevys_count = len(group[group['annot species'] == 'zebra_grevys'])
                zebras_count = len(group[group['annot species'].str.contains('zebra', case=False, na=False)])
                
                full_path = os.path.join(dataset_path, "Images", img_fname)
                if os.path.exists(full_path):
                    images_by_dataset[dataset].append({
                        'full_path': full_path,
                        'qa_data': [
                            {'animal': 'animals', 'answer': total_animals},
                            {'animal': 'grevys zebras', 'answer': grevys_count},
                            {'animal': 'zebras', 'answer': zebras_count}
                        ]
                    })

        else:
            df_unique = df.drop_duplicates(subset=['image_path'])
            for _, row in df_unique.iterrows():
                img_name = os.path.basename(row['image_path'])
                full_path = os.path.join(dataset_path, "Images", img_name)
                
                if os.path.exists(full_path):
                    images_by_dataset[dataset].append({
                        'full_path': full_path,
                        'qa_data': [{'animal': get_plural(row['class_name']), 'answer': row['num_animals_in_image']}]
                    })

    valid_datasets = [ds for ds in datasets if len(images_by_dataset[ds]) >= 4]
    if not valid_datasets:
        print("No datasets have enough images (minimum 4) to generate quadruplets.")
        return

    target_per_dataset = num_quads_total // len(valid_datasets)
    
    print(f"\n--- STEP 2: Generating Quadruplets ({target_per_dataset} per dataset) ---")
    all_quadruplets = []
    
    labels = [
        "Image A (top-left)", 
        "Image B (top-right)", 
        "Image C (bottom-left)", 
        "Image D (bottom-right)"
    ]

    for dataset_name in valid_datasets:
        pool = images_by_dataset[dataset_name]
        seen_combinations = set()
        dataset_quads = []
        attempts = 0
        
        # Slightly higher max attempts as finding both unique min and max is statistically harder
        max_attempts = target_per_dataset * 200 
        
        while len(dataset_quads) < target_per_dataset and attempts < max_attempts:
            attempts += 1
            
            # Pick 4 random, distinct images from the pool
            selected = random.sample(pool, 4)
            i1, i2, i3, i4 = selected
            
            combo_signature = tuple(sorted([i1['full_path'], i2['full_path'], i3['full_path'], i4['full_path']]))
            if combo_signature in seen_combinations:
                continue
                
            valid_questions_for_this_quad = []
            
            # Check all properties (1 for most datasets, 3 for GrevysZebra)
            for q_idx in range(len(i1['qa_data'])):
                animal = i1['qa_data'][q_idx]['animal']
                counts = [
                    i1['qa_data'][q_idx]['answer'],
                    i2['qa_data'][q_idx]['answer'],
                    i3['qa_data'][q_idx]['answer'],
                    i4['qa_data'][q_idx]['answer']
                ]
                
                max_count = max(counts)
                min_count = min(counts)
                
                # To get double QA pairs, BOTH the highest and lowest count must be unique
                if counts.count(max_count) == 1 and counts.count(min_count) == 1:
                    max_index = counts.index(max_count)
                    min_index = counts.index(min_count)
                    
                    # Add Most Question
                    valid_questions_for_this_quad.append({
                        'animal': animal,
                        'type': 'most',
                        'answer': labels[max_index]
                    })
                    # Add Least Question
                    valid_questions_for_this_quad.append({
                        'animal': animal,
                        'type': 'least',
                        'answer': labels[min_index]
                    })
                    
            if not valid_questions_for_this_quad:
                # Discard this quad entirely, it doesn't satisfy both unique max/min conditions
                continue

            seen_combinations.add(combo_signature)
            dataset_quads.append({
                'dataset': dataset_name,
                'images': [i1, i2, i3, i4],
                'questions': valid_questions_for_this_quad
            })
            
        all_quadruplets.extend(dataset_quads)
        print(f"Generated {len(dataset_quads)} valid quadruplets for {dataset_name} (Attempts: {attempts})")

    print(f"\nTotal quadruplets to generate: {len(all_quadruplets)}")
    
    all_qa_data = []
    
    print("\n--- STEP 3: Concatenating Images & Building CSV ---")
    for idx, quad in enumerate(all_quadruplets):
        if idx % 500 == 0 and idx != 0:
            print(f"Processed {idx}/{len(all_quadruplets)} quadruplet grids...")
            
        i1, i2, i3, i4 = quad['images']
        
        paths_concat = "".join([i['full_path'] for i in quad['images']])
        short_hash = hashlib.md5(paths_concat.encode()).hexdigest()[:8]
        concat_img_name = f"quad_{quad['dataset']}_{short_hash}.jpg"
            
        concat_rel_path = f"Images/{concat_img_name}"
        concat_full_path = os.path.join(output_dir, concat_rel_path)
        
        if not os.path.exists(concat_full_path):
            success = concat_4_images(i1['full_path'], i2['full_path'], i3['full_path'], i4['full_path'], concat_full_path, border_width=3)
        else:
            success = True
        
        if success:
            for q_data in quad['questions']:
                # Dynamically insert "most" or "least"
                all_qa_data.append({
                    'image': concat_img_name,
                    'image_path': concat_rel_path,
                    'question': f"Which of these 4 images contains {q_data['type']} number of {q_data['animal']}: Image A (top-left), Image B (top-right), Image C (bottom-left), Image D (bottom-right)?",
                    'answer': q_data['answer']
                })

    if all_qa_data:
        final_df = pd.DataFrame(all_qa_data)
        out_csv_path = os.path.join(output_dir, "qa.csv")
        final_df.to_csv(out_csv_path, index=False)
        print(f"\nSuccess! Saved {len(final_df)} QA pairs and generated {len(all_quadruplets)} grid images to {output_dir}")
    else:
        print("No valid data was generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Quad Most/Least VQA Counting CSV")
    parser.add_argument("--base_dir", required=True, help="Base directory containing dataset folders")
    parser.add_argument("--output_dir", required=True, help="Output directory for the qa.csv file and Images folder")
    parser.add_argument("--num_quads", type=int, default=10000, help="Total number of 2x2 grid images to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    process_datasets(args.base_dir, args.output_dir, args.num_quads, args.seed)