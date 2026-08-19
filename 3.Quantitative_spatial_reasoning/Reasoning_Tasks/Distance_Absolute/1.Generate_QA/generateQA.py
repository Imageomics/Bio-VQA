import os
import pandas as pd
import ast
import cv2
import uuid
import json
import numpy as np
from tqdm import tqdm
import argparse


def draw_bordered_line(img, p1, p2, line_color, border_color, thickness=7, border_thickness=3):
    p1 = np.array([p1[0], p1[1]], dtype=float)
    p2 = np.array([p2[0], p2[1]], dtype=float)
    
    vec = p2 - p1
    length = np.linalg.norm(vec)
    if length == 0: return
    
    # 1. Draw the main measurement line first
    cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)), line_color, thickness)
    
    # 2. Calculate dynamic width (Square-like framing)
    # Instead of a fixed 15, let's use 20% of the line length for the 'box' height
    # This ensures the border is always proportional to the measurement.
    unit_vec = vec / length
    perp_vec = np.array([-unit_vec[1], unit_vec[0]])
    
    extension_factor = 0.20
    ext = length * extension_factor
    
    # Define the framing box corners
    # We move 20% past the points and 20% perpendicular to them
    rect_width = ext 
    
    rect_p1_start = p1 - (unit_vec * ext)
    rect_p2_end = p2 + (unit_vec * ext)
    
    pts = np.array([
        rect_p1_start + (perp_vec * rect_width),
        rect_p1_start - (perp_vec * rect_width),
        rect_p2_end - (perp_vec * rect_width),
        rect_p2_end + (perp_vec * rect_width)
    ], np.int32)
    
    # 3. Draw the border
    cv2.polylines(img, [pts], isClosed=True, color=border_color, thickness=border_thickness)

    

def process_beetle_data(base_dir, output_base_dir):
    image_source_dir = os.path.join(base_dir, "Images")
    csv_path = os.path.join(base_dir, "metadata.csv")
    output_img_dir = os.path.join(output_base_dir, "Images")
    output_csv_path = os.path.join(output_base_dir, "qa.csv")
    
    os.makedirs(output_img_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    
    # Filter: 10 to 30 beetles inclusive
    df = df[(df['no_of_beetles_in_image'] >= 10) & (df['no_of_beetles_in_image'] <= 30)]
    
    qa_data = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Beetles"):
        try:
            scale_coords = ast.literal_eval(row['scalebar'])
            len_coords = ast.literal_eval(row['elytra_length_coords'])
            wid_coords = ast.literal_eval(row['elytra_width_coords'])

            img_filename = os.path.basename(row['group_image_path'])
            img_path = os.path.join(image_source_dir, img_filename)
            
            if not os.path.exists(img_path): continue

            measurements = [
                ('length', len_coords, row['elytra_length_cm']),
                ('width', wid_coords, row['elytra_width_cm'])
            ]

            for m_type, m_coords, m_val in measurements:
                img = cv2.imread(img_path)
                if img is None: continue

                # Draw Scalebar: Blue line (255,0,0) with Green border (0,255,0)
                draw_bordered_line(img, 
                                   (scale_coords['x1'], scale_coords['y1']), 
                                   (scale_coords['x2'], scale_coords['y2']), 
                                   (255, 0, 0), (0, 255, 0))

                # Draw Measurement: Red line (0,0,255) with White border (255,255,255)
                draw_bordered_line(img, 
                                   (m_coords['x1'], m_coords['y1']), 
                                   (m_coords['x2'], m_coords['y2']), 
                                   (0, 0, 255), (255, 255, 255))

                unique_id = uuid.uuid4().hex[:8]
                new_img_name = f"{os.path.splitext(img_filename)[0]}_{m_type}_{unique_id}.jpg"
                cv2.imwrite(os.path.join(output_img_dir, new_img_name), img)

                question = (f"What is the measurement of the elytra {m_type} (marked with red line with white border) "
                            f"with respect to the scalebar (marked with blue line with green border) in cm?")
                answer = f"{round(m_val, 3)} cm"
                
                qa_data.append({
                    "image": os.path.join("Images", new_img_name),
                    "qa_pairs": json.dumps({"Q": question, "A": answer})
                })

        except Exception as e:
            continue

    pd.DataFrame(qa_data).to_csv(output_csv_path, index=False)
    print(f"Success! Processed {len(qa_data)} QA pairs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--output_base_dir", type=str, required=True)
    args = parser.parse_args()
    process_beetle_data(args.base_dir, args.output_base_dir)