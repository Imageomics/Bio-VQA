import os
import argparse
import pandas as pd
import random
import json

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def get_label_type(kingdom):
    if pd.isna(kingdom):
        return "organism"
    k = str(kingdom).lower()
    if 'animalia' in k:
        return "animal"
    elif 'plantae' in k:
        return "plant"
    return "organism"

def generate_questions(row, genus_map, family_map, order_map, class_map, global_list):

    label = get_label_type(row['kingdom'])
    
    qa_list = []
    
    levels = ['phylum', 'class', 'order', 'family', 'genus', 'species', 'scientific_name', 'common_name']
    
    for level in levels:
        answer = row.get(level)

        if pd.isna(answer) or str(answer).strip() == "" or str(answer).lower() == 'nan':
            continue
            
        answer = str(answer).strip()

        display_level = "species epithet" if level == 'species' else level.replace('_', ' ')
        
        templates = [
            f"What is the {display_level} of the {label} shown in this image?",
            f"Which {display_level} does the {label} in this image belong to?",
            f"Identify the {display_level} of the {label} shown in this image.",
            f"Name the {display_level} of the {label} in this image.",
            f"What is the correct {display_level} label for the {label} in this image?"
        ]
        
        name_templates = [
            f"What is the {display_level} of the {label} shown in this image?",
            f"Identify the {display_level} of the {label} shown in this image.",
            f"Name the {display_level} of the {label} shown in this image."
        ]

        chosen_q = random.choice(templates if level not in ['scientific_name', 'common_name'] else name_templates)

        qa_list.append({'Q': chosen_q, 'A': answer})

        if level == 'common_name':

            genus = row.get('genus')
            family = row.get('family')
            order = row.get('order')
            tax_class = row.get('class')

            distractor_pool = set()
            
            for map_obj, key in [(genus_map, genus), (family_map, family), (order_map, order), (class_map, tax_class)]:
                if key and key in map_obj:
                    distractor_pool.update(map_obj[key])
            
            distractor_pool.discard(answer)
            
            distractors = list(distractor_pool)

            if len(distractors) < 3:
                global_pool = [n for n in global_list if n != answer]
                needed = 3 - len(distractors)
                additional = random.sample(global_pool, min(needed, len(global_pool)))
                distractors.extend(additional)

            distractors.sort()
            
            final_distractors = random.sample(distractors, 3)
            
            options = final_distractors + [answer]
            
            random.shuffle(options)
            
            opt_str = ", ".join(options)
            
            mcq_q = f"Among the four options, what is the common name of the {label} shown in this image? {opt_str}"
            
            qa_list.append({'Q': mcq_q, 'A': answer})

    return qa_list



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir", required=True)

    args = parser.parse_args()
    
    csv_files = [os.path.join(r, f) for r, _, files in os.walk(args.base_dir) for f in files if f.endswith("-10K.csv")]
    csv_files.sort()

    for csv_path in csv_files:
        random.seed(RANDOM_SEED)
        print(f"\nPROCESSING: {os.path.basename(csv_path)}")
        df = pd.read_csv(csv_path)

        if 'common' in df.columns:
            df = df.rename(columns={'common': 'common_name'})

        valid_common = df.dropna(subset=['common_name']).copy()
        def get_sorted_list(group): return sorted(list(set(group)))

        genus_map = valid_common.groupby('genus')['common_name'].apply(get_sorted_list).to_dict()
        family_map = valid_common.groupby('family')['common_name'].apply(get_sorted_list).to_dict()
        order_map = valid_common.groupby('order')['common_name'].apply(get_sorted_list).to_dict()
        class_map = valid_common.groupby('class')['common_name'].apply(get_sorted_list).to_dict()
        global_list = valid_common['common_name'].unique().tolist()

        all_qa_data = []
        for _, row in df.iterrows():
            qa_pairs = generate_questions(row, genus_map, family_map, order_map, class_map, global_list)
            for pair in qa_pairs:
                all_qa_data.append({
                    'uuid': row['uuid'],
                    'source_url': row['source_url'],
                    'local_path': row['local_path'],
                    'qa_pairs': json.dumps(pair)
                })

        output_name = csv_path.replace(".csv", "-QA.csv")
        pd.DataFrame(all_qa_data).to_csv(output_name, index=False)
        print(f"[DONE] Saved {len(all_qa_data)} QA pairs to {output_name}")