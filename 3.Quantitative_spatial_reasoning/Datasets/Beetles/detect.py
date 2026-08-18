import os
import argparse
import pandas as pd
import moondream as md
from PIL import Image, ImageDraw
import time
import random
import json

def run_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True, help="Path to Beetles-500")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--num_tasks", type=int, default=6)
    args = parser.parse_args()

    api_key = os.environ.get("MOONDREAM_API_KEY")
    model = md.vl(api_key=api_key)
    
    # Setup Paths relative to base_dir
    metadata_path = os.path.join(args.base_dir, "metadata.csv")
    output_dir = os.path.join(args.base_dir, "Detections")
    plotted_dir = os.path.join(output_dir, "plotted_images")
    
    os.makedirs(plotted_dir, exist_ok=True)

    # 1. Load and Filter for 10-20 range
    df = pd.read_csv(metadata_path)
    df_unique = df.groupby("group_image_path", as_index=False).first()
    df_filtered = df_unique[
        (df_unique["no_of_beetles_in_image"] >= 10) & 
        (df_unique["no_of_beetles_in_image"] < 20)
    ].copy()
    
    # 2. Parallel Chunking
    total_images = len(df_filtered)
    chunk_size = (total_images // args.num_tasks) + 1
    start_idx = args.task_id * chunk_size
    end_idx = min(start_idx + chunk_size, total_images)
    
    my_chunk = df_filtered.iloc[start_idx:end_idx]
    task_csv = os.path.join(output_dir, f"detections_{args.task_id}.csv")
    
    print(f"🚀 Task {args.task_id}: Processing {len(my_chunk)} images (Range: 10-20)...")

    results = []
    for _, row in my_chunk.iterrows():
        rel_path = row["group_image_path"]
        full_path = os.path.join(args.base_dir, rel_path)
        ground_truth_count = int(row["no_of_beetles_in_image"])
        img_filename = os.path.basename(rel_path)

        if not os.path.exists(full_path):
            continue

        max_retries = 5
        for attempt in range(max_retries):
            try:
                time.sleep(random.uniform(0.1, 0.3))
                image = Image.open(full_path).convert("RGB")
                width, height = image.size
                
                result = model.detect(image, "beetle")
                objects = result.get("objects", [])
                detected_count = len(objects)

                # Validation Gate
                if detected_count == ground_truth_count:
                    # 1. Plot and Save Image
                    draw = ImageDraw.Draw(image)
                    for obj in objects:
                        x_min, y_min = int(obj["x_min"] * width), int(obj["y_min"] * height)
                        x_max, y_max = int(obj["x_max"] * width), int(obj["y_max"] * height)
                        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
                    
                    image.save(os.path.join(plotted_dir, img_filename))

                    # 2. Log Data
                    results.append({
                        "image_name": img_filename,
                        "image_path": rel_path,
                        "no_of_beetles": ground_truth_count,
                        "coordinates": json.dumps(objects)
                    })
                    
                    # Incremental save for safety
                    pd.DataFrame(results).to_csv(task_csv, index=False)
                
                break # Success, move to next image

            except Exception as e:
                if "429" in str(e):
                    time.sleep(10 * (2 ** attempt))
                else:
                    print(f"❌ Error {rel_path}: {e}")
                    break 

    print(f"✅ Task {args.task_id} finished.")

if __name__ == "__main__":
    run_inference()