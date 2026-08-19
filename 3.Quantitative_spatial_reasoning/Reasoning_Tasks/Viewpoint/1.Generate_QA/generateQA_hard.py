import os
import uuid
import argparse
import random
import pandas as pd
from PIL import Image


# ---------------- CONCAT ---------------- #
def concat_images(img_paths):
    imgs = [Image.open(p).convert("RGB") for p in img_paths]

    min_h = min(img.height for img in imgs)
    resized = [
        img.resize((int(img.width * min_h / img.height), min_h))
        for img in imgs
    ]

    total_w = sum(img.width for img in resized)
    out = Image.new("RGB", (total_w, min_h))

    x = 0
    for img in resized:
        out.paste(img, (x, 0))
        x += img.width

    return out


# ---------------- MAIN ---------------- #
def main(args):

    input_path = os.path.join(args.base_dir, args.input_basename)
    images_dir = os.path.join(input_path, "Images")
    csv_path = os.path.join(input_path, "qa.csv")

    output_path = os.path.join(args.base_dir, args.output_basename)
    out_img_dir = os.path.join(output_path, "Images")

    os.makedirs(out_img_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    qa_rows = []
    total_images = 0

    random.seed(42)

    # ---------------- PER DATASET ---------------- #
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

        # deterministic ordering
        candidates.sort(key=lambda x: (-x[0], x[1]))

        # shuffle for diversity
        random.shuffle(candidates)

        triplets = []
        used_triplets = set()

        i = 0

        # ---------------- 250 TRIPLETS ---------------- #
        while len(triplets) < 250:

            a = candidates[i % len(candidates)]
            b = candidates[(i + 1) % len(candidates)]
            c = candidates[(i + 2) % len(candidates)]

            ids = tuple(sorted([a[1], b[1], c[1]]))

            if ids not in used_triplets:
                used_triplets.add(ids)
                triplets.append([a, b, c])

            i += 3

        # ---------------- BUILD IMAGES + QA ---------------- #
        for triplet in triplets:

            img_paths = []
            viewpoints = []

            for _, img_name, vp in triplet:
                img_paths.append(os.path.join(images_dir, img_name))
                viewpoints.append(vp)

            merged = concat_images(img_paths)

            uid = str(uuid.uuid4())
            out_name = f"{dataset}_{uid}.jpg"
            out_path = os.path.join(out_img_dir, out_name)

            merged.save(out_path)

            positions = ["left", "middle", "right"]

            for pos, vp in zip(positions, viewpoints):
                qa_rows.append({
                    "image": out_name,
                    "question": f"What is the viewpoint of the {dataset} marked in the {pos} bounding box?",
                    "answer": vp,
                    "dataset": dataset,
                    "position": pos
                })

            total_images += 1

    # ---------------- SAVE ---------------- #
    out_csv = os.path.join(output_path, "qa.csv")
    pd.DataFrame(qa_rows).to_csv(out_csv, index=False)

    print(f"Saved images: {total_images}")
    print(f"Saved QA pairs: {len(qa_rows)}")


# ---------------- CLI ---------------- #
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_dir",
        required=True
    )

    parser.add_argument(
        "--input_basename",
        default="Viewpoint/Easy"
    )

    parser.add_argument(
        "--output_basename",
        default="Viewpoint/Hard"
    )

    args = parser.parse_args()

    main(args)