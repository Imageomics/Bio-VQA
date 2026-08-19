import os
import pandas as pd
import ast
import cv2
import uuid
import json
import random
import numpy as np
from tqdm import tqdm
import argparse

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def draw_dotted_rect(img, top_left, bottom_right, color, thickness=3, dash_length=15):
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    def draw_dashed_line(pt1, pt2):
        dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
        dashes = max(1, int(dist / dash_length))
        for i in range(dashes):
            start = [int(pt1[0] + (pt2[0]-pt1[0])*i/dashes), int(pt1[1] + (pt2[1]-pt1[1])*i/dashes)]
            end = [int(pt1[0] + (pt2[0]-pt1[0])*(i+0.5)/dashes), int(pt1[1] + (pt2[1]-pt1[1])*(i+0.5)/dashes)]
            cv2.line(img, tuple(start), tuple(end), color, thickness)
            
    draw_dashed_line((x1, y1), (x2, y1))
    draw_dashed_line((x2, y1), (x2, y2))
    draw_dashed_line((x2, y2), (x1, y2))
    draw_dashed_line((x1, y2), (x1, y1))

def get_matching_bbox(line_coords, bboxes_normalized, img_w, img_h):
    mx = (line_coords['x1'] + line_coords['x2']) / 2.0
    my = (line_coords['y1'] + line_coords['y2']) / 2.0
    
    for box in bboxes_normalized:
        xmin, ymin = box['x_min'] * img_w, box['y_min'] * img_h
        xmax, ymax = box['x_max'] * img_w, box['y_max'] * img_h
        
        if xmin <= mx <= xmax and ymin <= my <= ymax:
            return (int(xmin), int(ymin)), (int(xmax), int(ymax))
            
    padding = 30
    xmin = min(line_coords['x1'], line_coords['x2']) - padding
    ymin = min(line_coords['y1'], line_coords['y2']) - padding
    xmax = max(line_coords['x1'], line_coords['x2']) + padding
    ymax = max(line_coords['y1'], line_coords['y2']) + padding
    return (int(max(0, xmin)), int(max(0, ymin))), (int(min(img_w, xmax)), int(min(img_h, ymax)))

def process_beetle_dataset(base_dir, output_base_dir, target_count=10000, seed=42):
    set_seed(seed)
    
    image_source_dir = os.path.join(base_dir, "Images")
    metadata_csv = os.path.join(base_dir, "metadata.csv")
    detections_csv = os.path.join(base_dir, "Detections/Detections.csv")
    
    output_dir = os.path.join(output_base_dir, "Images")
    output_csv_path = os.path.join(output_base_dir, "qa.csv")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Metadata
    meta_df = pd.read_csv(metadata_csv)
    # Keeping the 10-30 constraint as requested previously, but it will yield much more now
    meta_df = meta_df[meta_df['no_of_beetles_in_image'].between(10, 30)]

    # 2. Load Detections
    det_df = pd.read_csv(detections_csv)
    detections_dict = {}
    for _, row in det_df.iterrows():
        try:
            coord_str = str(row['coordinates']).replace('""', '"')
            detections_dict[row['image_name']] = json.loads(coord_str)
        except Exception:
            try:
                detections_dict[row['image_name']] = ast.literal_eval(row['coordinates'])
            except:
                pass

    grouped = meta_df.groupby('group_image_path')
    qa_data = []
    
    color_map = {
        "A": (0, 0, 255),      # Red
        "B": (255, 0, 0),      # Blue
        "C": (0, 255, 0),      # Green
        "D": (255, 255, 255)   # White
    }

    group_names = sorted(list(grouped.groups.keys()))
    random.shuffle(group_names)
    
    # --- YIELD CONTROLS ---
    max_pairs_per_image = 20 # Prevents one image from dominating the dataset
    max_attempts_per_image = 500 # How many random combinations to try before giving up on an image

    for img_rel_path in tqdm(group_names, desc="Processing Images"):
        if len(qa_data) >= target_count:
            break

        group = grouped.get_group(img_rel_path)
        img_name = os.path.basename(img_rel_path)
        
        if len(group) < 4 or img_name not in detections_dict:
            continue

        bboxes_normalized = detections_dict[img_name]
        beetle_pool = group.to_dict('records')
        
        pairs_from_this_image = 0
        attempts = 0
        
        # Load image once per group to save I/O time
        img_path = os.path.join(image_source_dir, img_name)
        base_img = cv2.imread(img_path)
        if base_img is None:
            continue
        img_h, img_w = base_img.shape[:2]

        while pairs_from_this_image < max_pairs_per_image and attempts < max_attempts_per_image:
            attempts += 1
            if len(qa_data) >= target_count:
                break

            # Randomly pick 4 beetles from the pool (combinations!)
            selected_beetles = random.sample(beetle_pool, 4)
            
            m_type = random.choice(['length', 'width'])
            val_key = f'elytra_{m_type}_cm'
            coord_key = f'elytra_{m_type}_coords'

            # --- DYNAMIC FILTERING GAP (RELAXED) ---
            # 10% gap for width, 5% gap for length
            gap_percent = 0.90 if m_type == 'width' else 0.95
            
            vals = [b[val_key] for b in selected_beetles]
            max_val = max(vals)
            threshold = max_val * gap_percent
            
            others = vals.copy()
            others.remove(max_val)
            
            if any(v > threshold for v in others) or vals.count(max_val) > 1:
                continue 

            # We found a valid combination! Make a copy of the base image to draw on
            img = base_img.copy()

            measurements_info = []
            for i, label in enumerate(["A", "B", "C", "D"]):
                beetle = selected_beetles[i]
                line_coords = ast.literal_eval(beetle[coord_key])
                
                pt1, pt2 = get_matching_bbox(line_coords, bboxes_normalized, img_w, img_h)
                draw_dotted_rect(img, pt1, pt2, color_map[label], thickness=4, dash_length=15)

                cv2.line(img, (int(line_coords['x1']), int(line_coords['y1'])), 
                         (int(line_coords['x2']), int(line_coords['y2'])), color_map[label], 6)
                
                measurements_info.append({"label": label, "value": beetle[val_key]})

            winner_label = max(measurements_info, key=lambda x: x['value'])['label']
            
            unique_suffix = uuid.uuid4().hex[:6]
            new_img_name = f"{os.path.splitext(img_name)[0]}_set{pairs_from_this_image}_{unique_suffix}.jpg"
            cv2.imwrite(os.path.join(output_dir, new_img_name), img)

            question = (f"Among the four highlighted beetle specimens, which one has the largest elytral {m_type}? "
                        f"Beetle A (red dotted box/line), Beetle B (blue dotted box/line), "
                        f"Beetle C (green dotted box/line), and Beetle D (white dotted box/line)")
            
            qa_data.append({
                "image": new_img_name,
                "qa_pairs": json.dumps({"Q": question, "A": f"Beetle {winner_label}"})
            })
            
            pairs_from_this_image += 1

    output_df = pd.DataFrame(qa_data)
    output_df.to_csv(output_csv_path, index=False)
    print(f"\nProcessing Complete. Created {len(output_df)} valid QA samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Beetle Relative Reasoning Generator")
    parser.add_argument("--base_dir", type=str, required=True, help="Path to Beetles-500")
    parser.add_argument("--output_base_dir", type=str, required=True, help="Save path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    process_beetle_dataset(args.base_dir, args.output_base_dir, seed=args.seed)