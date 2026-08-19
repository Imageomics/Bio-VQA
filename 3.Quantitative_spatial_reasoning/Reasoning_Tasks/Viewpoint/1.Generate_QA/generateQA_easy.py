import os
import json
import uuid
import argparse
import pandas as pd
from PIL import Image, ImageDraw
from collections import defaultdict


# ---------------- DRAW BBOX ---------------- #
def draw_bbox(image_path, bbox):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    x, y, w, h = bbox
    draw.rectangle([x, y, x + w, y + h], outline="red", width=4)

    return img


# ---------------- CSV (Grevy) ---------------- #
def process_csv(csv_path, images_path, n_samples):

    df = pd.read_csv(csv_path)

    grouped = df.groupby("image")
    valid = [g for _, g in grouped if len(g) == 1]

    candidates = []

    for g in valid:
        row = g.iloc[0]

        image_name = row["image"]
        viewpoint = row["viewpoint"]

        # ✅ filter viewpoint EARLY
        if pd.isna(viewpoint) or str(viewpoint).strip() == "":
            continue

        bbox = [row["bbox x"], row["bbox y"], row["bbox w"], row["bbox h"]]

        image_path = os.path.join(images_path, image_name)

        # ✅ ensure file exists
        if not os.path.exists(image_path):
            continue

        size = os.path.getsize(image_path)

        candidates.append((size, image_path, bbox, viewpoint, image_name))

    # ✅ deterministic sort (size desc + tie-break path)
    candidates.sort(key=lambda x: (-x[0], x[1]))

    selected = candidates[:n_samples]

    return [(img_path, bbox, viewpoint, name) for _, img_path, bbox, viewpoint, name in selected]


# ---------------- COCO ---------------- #
def process_coco(json_path, images_path, n_samples):

    with open(json_path, "r") as f:
        data = json.load(f)

    annotations = data["annotations"]
    images = {img["id"]: img["file_name"] for img in data["images"]}

    img_to_anns = defaultdict(list)
    for ann in annotations:
        img_to_anns[ann["image_id"]].append(ann)

    candidates = []

    for anns in img_to_anns.values():

        if len(anns) != 1:
            continue

        ann = anns[0]

        viewpoint = ann.get("viewpoint", None)

        if viewpoint is None or str(viewpoint).strip() == "":
            continue

        image_id = ann["image_id"]
        image_name = images[image_id]
        bbox = ann["bbox"]

        image_path = os.path.join(images_path, image_name)

        if not os.path.exists(image_path):
            continue

        size = os.path.getsize(image_path)

        candidates.append((size, image_path, bbox, viewpoint, image_name))


    candidates.sort(key=lambda x: (-x[0], x[1]))

    selected = candidates[:n_samples]

    return [(img_path, bbox, viewpoint, name) for _, img_path, bbox, viewpoint, name in selected]


# ---------------- MAIN ---------------- #
def main(args):

    os.makedirs(os.path.join(args.output_dir, "Images"), exist_ok=True)

    qa_data = []

    grevy_data = process_csv(args.grevys_csv, args.grevys_images, args.num_samples)
    hyena_data = process_coco(args.hyena_json, args.hyena_images, args.num_samples)
    leopard_data = process_coco(args.leopard_json, args.leopard_images, args.num_samples)
    whaleshark_data = process_coco(args.whaleshark_json, args.whaleshark_images, args.num_samples)

    all_data = (
        [("zebra", x) for x in grevy_data]
        + [("hyena", x) for x in hyena_data]
        + [("leopard", x) for x in leopard_data]
        + [("whaleshark", x) for x in whaleshark_data]
    )

    for dataset, (img_path, bbox, viewpoint, orig_name) in all_data:

        try:
            img = draw_bbox(img_path, bbox)
        except Exception as e:
            print(f"Skipping {img_path}: {e}")
            continue

        uid = str(uuid.uuid4())
        new_name = f"{os.path.splitext(orig_name)[0]}_{uid}.jpg"
        save_path = os.path.join(args.output_dir, "Images", new_name)

        img.save(save_path)

        qa_data.append({
            "image": new_name,
            "question": "What is the viewpoint of the animal (red bounding box) shown in this image?",
            "answer": viewpoint,
            "dataset": dataset
        })

    df = pd.DataFrame(qa_data)
    df.to_csv(os.path.join(args.output_dir, "qa.csv"), index=False)

    print(f"Saved {len(df)} QA pairs.")


# ---------------- CLI ---------------- #
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--grevys_csv")
    parser.add_argument("--grevys_images")

    parser.add_argument("--hyena_json")
    parser.add_argument("--hyena_images")

    parser.add_argument("--leopard_json")
    parser.add_argument("--leopard_images")

    parser.add_argument("--whaleshark_json")
    parser.add_argument("--whaleshark_images")

    parser.add_argument("--output_dir")
    parser.add_argument("--num_samples", type=int, default=1000)

    args = parser.parse_args()

    main(args)