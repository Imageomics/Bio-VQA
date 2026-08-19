import os
import glob
import pandas as pd
import argparse

def process_datasets(base_dir, output_dir):
    all_qa_data = []

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # List of expected datasets
    datasets = [
        "Bees", "Beetles-500", "Camel-500", "Cattle-500", 
        "Fish", "GrevysZebra", "Kiang-500", "Seal-500", 
        "Sheep-500", "Zebra-500"
    ]

    for dataset in datasets:
        dataset_path = os.path.join(base_dir, dataset)
        if not os.path.isdir(dataset_path):
            print(f"Warning: Directory not found for {dataset}. Skipping.")
            continue
            
        # Find the CSV file in the dataset directory
        csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
        if not csv_files:
            print(f"Warning: No CSV found in {dataset}. Skipping.")
            continue
            
        csv_path = csv_files[0]
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue

        print(f"Processing {dataset}...")

        # 1. Bees
        if "Bees" in dataset:
            df_unique = df.drop_duplicates(subset=['image'])
            for _, row in df_unique.iterrows():
                img_name = row['image']
                all_qa_data.append({
                    'image': img_name,
                    'image_path': f"{dataset}/Images/{img_name}",
                    'question': "How many bees are there in this image?",
                    'answer': row['num_bees']
                })

        # 2. Beetles
        elif "Beetles" in dataset:
            df_unique = df.drop_duplicates(subset=['group_image_path'])
            for _, row in df_unique.iterrows():
                rel_path = row['group_image_path'] # e.g., 'Images/A00000001831.jpg'
                img_name = os.path.basename(rel_path)
                all_qa_data.append({
                    'image': img_name,
                    'image_path': f"{dataset}/{rel_path}",
                    'question': "How many beetles are there in this image?",
                    'answer': row['no_of_beetles_in_image']
                })

        # 3. Fish
        elif "Fish" in dataset:
            df_unique = df.drop_duplicates(subset=['image_name'])
            for _, row in df_unique.iterrows():
                img_name = row['image_name']
                all_qa_data.append({
                    'image': img_name,
                    'image_path': f"{dataset}/Images/{img_name}",
                    'question': "How many fish are there in this image?",
                    'answer': row['num_fish']
                })

        # 4. GrevysZebra (Complex Logic)
        elif "GrevysZebra" in dataset:
            # Group by image filename to count occurrences
            for img_fname, group in df.groupby('image fname'):
                # Total annotations (all animals)
                total_animals = len(group)
                
                # Total Grevys zebras
                grevys_count = len(group[group['annot species'] == 'zebra_grevys'])
                
                # Total zebras (any species containing 'zebra')
                zebras_count = len(group[group['annot species'].str.contains('zebra', case=False, na=False)])
                
                img_path = f"{dataset}/Images/{img_fname}"
                
                # Add 3 separate questions per image
                all_qa_data.append({
                    'image': img_fname, 'image_path': f"{dataset}/Images/{img_fname}",
                    'question': "How many animals are there in this image?",
                    'answer': total_animals
                })
                all_qa_data.append({
                    'image': img_fname, 'image_path': f"{dataset}/Images/{img_fname}",
                    'question': "How many grevys zebras are there in this image?",
                    'answer': grevys_count
                })
                all_qa_data.append({
                    'image': img_fname, 'image_path': f"{dataset}/Images/{img_fname}",
                    'question': "How many zebras are there in this image?",
                    'answer': zebras_count
                })


        else:
            df_unique = df.drop_duplicates(subset=['image_path'])
            for _, row in df_unique.iterrows():

                rel_path = row['image_path']
                img_name = os.path.basename(rel_path)
            
                cls_name = str(row['class_name']).lower().strip()

                if cls_name == 'kiang':
                    plural_name = "kiangs (wild asses)"
                elif cls_name in ['cattle', 'sheep']: 
                    plural_name = cls_name
                else:
                    plural_name = f"{cls_name}s"

                all_qa_data.append({
                    'image': img_name,
                    'image_path': f"{dataset}/Images/{img_name}",
                    'question': f"How many {plural_name} are there in this image?",
                    'answer': row['num_animals_in_image']
                })

    # Save consolidated dataframe to CSV
    if all_qa_data:
        final_df = pd.DataFrame(all_qa_data)
        out_file = os.path.join(output_dir, "qa.csv")
        final_df.to_csv(out_file, index=False)
        print(f"\nSuccess! Saved {len(final_df)} QA pairs to {out_file}")
    else:
        print("No data was processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate VQA Counting CSV")
    parser.add_argument("--base_dir", required=True, help="Base directory containing dataset folders")
    parser.add_argument("--output_dir", required=True, help="Output directory for the qa.csv file")
    
    args = parser.parse_args()
    process_datasets(args.base_dir, args.output_dir)