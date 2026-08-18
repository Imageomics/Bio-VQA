import os
import xml.etree.ElementTree as ET
import pandas as pd
import shutil
import argparse

def parse_voc_xml(xml_path): 
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bees = []
        for obj in root.findall('object'):
            if obj.find('name').text == 'bee':
                bbox = obj.find('bndbox')
                xmin, ymin = int(bbox.find('xmin').text), int(bbox.find('ymin').text)
                xmax, ymax = int(bbox.find('xmax').text), int(bbox.find('ymax').text)
                bees.append({
                    'xmin': xmin, 'ymin': ymin,
                    'w': xmax - xmin, 'h': ymax - ymin,
                    'truncated': int(obj.find('truncated').text),
                    'difficult': int(obj.find('difficult').text)
                })
        return bees
    except Exception:
        return []


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process Bee Dataset into Difficulty Tiers.")
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--filtered_dir", type=str, required=True)
    parser.add_argument("--filtered_csv", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=200)

    args = parser.parse_args()

    os.makedirs(args.filtered_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.filtered_csv), exist_ok=True)

    all_data = []
    xml_files = [f for f in os.listdir(args.base_dir) if f.endswith('.xml')]
    
    print(f"Scanning {len(xml_files)} files...")

    for xml_file in xml_files:
        
        if "Chueried_HempBox" in xml_file: # Exclude Hempbox, bbox coordinates not accurate
            continue

        image_name = xml_file.replace('.xml', '.jpg')
        image_path = os.path.join(args.base_dir, image_name)
        
        if not os.path.exists(image_path):
            continue
            
        # 2. Filter by Size (> 200KB) : there are many .jpg files that cannot be opened
        file_size_kb = os.path.getsize(image_path) / 1024
        if file_size_kb <= args.image_size:
            continue
        
        bees = parse_voc_xml(os.path.join(args.base_dir, xml_file))
        num_bees = len(bees)
        
        bees.sort(key=lambda b: (round(b['ymin'] / 50) * 50, b['xmin']))
        
        for i, bee in enumerate(bees, 1):
            all_data.append({
                'image': image_name,
                'num_bees': num_bees,
                'bee index': f"bee_{i}",
                'x': bee['xmin'], 'y': bee['ymin'],
                'w': bee['w'], 'h': bee['h'],
                'difficulty_tier': 'easy' if 5 <= num_bees <= 20 else ('medium' if 20 < num_bees <= 60 else 'hard' if num_bees > 60 else 'too_easy')
            })

    df = pd.DataFrame(all_data)
    output_csv_path = os.path.join(args.output_csv, "metadata.csv")
    df.to_csv(output_csv_path, index=False)
    
    unique_df = df[['image', 'num_bees', 'difficulty_tier']].drop_duplicates()

    easy_pool = unique_df[unique_df['difficulty_tier'] == 'easy'].sort_values(by='num_bees', ascending=False).head(250)
    med_pool = unique_df[unique_df['difficulty_tier'] == 'medium'].sort_values(by='num_bees', ascending=False).head(150)
    hard_pool = unique_df[unique_df['difficulty_tier'] == 'hard'].sort_values(by='num_bees', ascending=False).head(100)

    final_images = pd.concat([easy_pool, med_pool, hard_pool])
    image_list = final_images['image'].tolist()

    df_filtered = df[df['image'].isin(image_list)].copy()
    
    tier_order = {'easy': 0, 'medium': 1, 'hard': 2}
    df_filtered['tier_rank'] = df_filtered['difficulty_tier'].map(tier_order)
    df_filtered = df_filtered.sort_values(by=['tier_rank', 'num_bees', 'image', 'bee index'], ascending=[True, False, True, True])
    df_filtered = df_filtered.drop(columns=['tier_rank'])

    df_filtered.to_csv(args.filtered_csv, index=False)
    
    for img_name in image_list:
        shutil.copy2(os.path.join(args.base_dir, img_name), os.path.join(args.filtered_dir, img_name))