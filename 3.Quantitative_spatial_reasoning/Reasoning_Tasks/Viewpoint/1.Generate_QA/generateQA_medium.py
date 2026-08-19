import os
import uuid
import argparse
import pandas as pd
from PIL import Image



def concat_images_horizontally(img_paths):
    
    imgs = [Image.open(p).convert("RGB") for p in img_paths]

    # resize to same height
    min_h = min(img.height for img in imgs)
    imgs = [
        img.resize((int(img.width * min_h / img.height), min_h))
        for img in imgs
    ]

    total_width = sum(img.width for img in imgs)

    new_img = Image.new("RGB", (total_width, min_h))

    x_offset = 0
    for img in imgs:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width

    return new_img


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir", required=True)

    parser.add_argument("--input_basename", default="Viewpoint/Easy")

    parser.add_argument("--output_basename", default="Viewpoint/Medium")

    args = parser.parse_args()

    input_path = os.path.join(args.base_dir, args.input_basename)
    images_dir = os.path.join(input_path, "Images")
    csv_path = os.path.join(input_path, "qa.csv")

    output_path = os.path.join(args.base_dir, args.output_basename)
    out_img_dir = os.path.join(output_path, "Images")

    os.makedirs(out_img_dir, exist_ok=True)

    df = pd.read_csv(csv_path)


    selected = {}


    for dataset, group in df.groupby("dataset"):

        candidates = []

        for _, row in group.iterrows():
            img_name = row["image"]
            img_path = os.path.join(images_dir, img_name)

            if not os.path.exists(img_path):
                continue

            size = os.path.getsize(img_path)
            viewpoint = row["answer"]

            candidates.append((size, img_name, viewpoint))

        candidates.sort(key=lambda x: (-x[0], x[1]))

        selected[dataset] = candidates[:500]

    datasets = list(selected.keys())

    qa_rows = []
    image_count = 0

    for i in range(500):

        triplet = []
        meta = []

        for d in datasets:
            size, img_name, viewpoint = selected[d][i]
            img_path = os.path.join(images_dir, img_name)

            triplet.append(img_path)
            meta.append((d, viewpoint))

        # pick ANY 3 out of 4 datasets (sliding window trick)
        combos = [
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3),
            (1, 2, 3),
        ]

        for combo in combos:

            imgs = [triplet[i] for i in combo]
            metas = [meta[i] for i in combo]

            try:
                merged = concat_images_horizontally(imgs)
            except Exception as e:
                print(f"Skipping merge: {e}")
                continue

            uid = str(uuid.uuid4())
            new_name = f"{uid}.jpg"
            save_path = os.path.join(out_img_dir, new_name)

            merged.save(save_path)
            
            for d, vp in metas:
                qa_rows.append({
                    "image": new_name,
                    "question": f"What is the viewpoint of the {d} in this image?",
                    "answer": vp,
                    "dataset": d
                })

            image_count += 1


    out_csv = os.path.join(output_path, "qa.csv")
    pd.DataFrame(qa_rows).to_csv(out_csv, index=False)

    print(f"Saved {image_count} images")
    print(f"Saved {len(qa_rows)} QA pairs")