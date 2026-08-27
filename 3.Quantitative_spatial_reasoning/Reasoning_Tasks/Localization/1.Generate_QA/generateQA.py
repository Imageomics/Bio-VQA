import os
import csv
import json
import argparse
import pandas as pd
from collections import defaultdict


# ---------------- FORMAT BBOX ---------------- #
def format_bbox(bbox):
    return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]


# ---------------- GREVY CSV ---------------- #
def process_csv(csv_path, images_path, n_samples):

    df = pd.read_csv(csv_path)
    grouped = df.groupby("image")

    valid = [g for _, g in grouped if len(g) == 1]

    candidates = []

    for g in valid:
        row = g.iloc[0]

        image_name = row["image"]
        bbox = [row["bbox x"], row["bbox y"], row["bbox w"], row["bbox h"]]

        image_path = os.path.join(images_path, image_name)

        if not os.path.exists(image_path):
            continue

        size = os.path.getsize(image_path)

        candidates.append((size, image_path, bbox, image_name))

    candidates.sort(key=lambda x: (-x[0], x[1]))

    return candidates[:n_samples]


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

        image_id = ann["image_id"]
        image_name = images[image_id]
        bbox = ann["bbox"]

        image_path = os.path.join(images_path, image_name)

        if not os.path.exists(image_path):
            continue

        size = os.path.getsize(image_path)

        candidates.append((size, image_path, bbox, image_name))

    candidates.sort(key=lambda x: (-x[0], x[1]))

    return candidates[:n_samples]


# ---------------- MAIN ---------------- #
def main(args):

    qa_data = []

    # ---------------- ZEBRA ---------------- #
    zebra = process_csv(args.grevys_csv, args.grevys_images, args.num_samples)

    for _, img_path, bbox, name in zebra:

        rel_path = os.path.relpath(img_path, args.base_dir)

        qa_data.append({
            "image": name,
            "image_path": rel_path,
            "question": "Provide bounding box coordinates for the subject animal (zebra) in this image.",
            "answer": format_bbox(bbox),
            "dataset": "zebra"
        })

    # ---------------- HYENA ---------------- #
    hyena = process_coco(args.hyena_json, args.hyena_images, args.num_samples)

    for _, img_path, bbox, name in hyena:

        rel_path = os.path.relpath(img_path, args.base_dir)

        qa_data.append({
            "image": name,
            "image_path": rel_path,
            "question": "Provide bounding box coordinates for the subject animal (hyena) in this image.",
            "answer": format_bbox(bbox),
            "dataset": "hyena"
        })

    # ---------------- LEOPARD ---------------- #
    leopard = process_coco(args.leopard_json, args.leopard_images, args.num_samples)

    for _, img_path, bbox, name in leopard:

        rel_path = os.path.relpath(img_path, args.base_dir)

        qa_data.append({
            "image": name,
            "image_path": rel_path,
            "question": "Provide bounding box coordinates for the subject animal (leopard) in this image.",
            "answer": format_bbox(bbox),
            "dataset": "leopard"
        })

    # ---------------- WHALESHARK ---------------- #
    whale = process_coco(args.whaleshark_json, args.whaleshark_images, args.num_samples)

    for _, img_path, bbox, name in whale:

        rel_path = os.path.relpath(img_path, args.base_dir)

        qa_data.append({
            "image": name,
            "image_path": rel_path,
            "question": "Provide bounding box coordinates for the subject animal (whale shark) in this image.",
            "answer": format_bbox(bbox),
            "dataset": "whaleshark"
        })

    # ---------------- OUTPUT DIR ---------------- #
    os.makedirs(args.output_dir, exist_ok=True)

    output_path = os.path.join(args.output_dir, "qa.csv")

    # ---------------- WRITE CSV ---------------- #
    with open(output_path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "image_path",
                "question",
                "answer",
                "dataset"
            ]
        )

        writer.writeheader()

        for item in qa_data:
            writer.writerow({
                "image": item["image"],
                "image_path": item["image_path"],
                "question": item["question"],
                "answer": str(item["answer"]),
                "dataset": item["dataset"]
            })

    print(f"Saved {len(qa_data)} samples to {output_path}")


# ---------------- CLI ---------------- #
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir", required=True)

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