import os
import shutil
import argparse
import uuid

import pandas as pd
from PIL import Image


SPLITS = ["Train", "Test", "Valid"]
SPECIES_NAME = "Etroplus_maculatus"


def read_yolo_boxes(label_path):

    boxes = []

    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                _, x_c, y_c, w, h = map(float, parts)
            except ValueError:
                continue
            boxes.append((x_c, y_c, w, h))

    return boxes


def to_corner_box(box, img_w, img_h):

    x_c, y_c, w_n, h_n = box

    abs_w = w_n * img_w
    abs_h = h_n * img_h

    x_min = (x_c * img_w) - (abs_w / 2.0)
    y_min = (y_c * img_h) - (abs_h / 2.0)

    # Clamp to image bounds.
    x_min = max(0.0, min(x_min, img_w))
    y_min = max(0.0, min(y_min, img_h))
    abs_w = min(abs_w, img_w - x_min)
    abs_h = min(abs_h, img_h - y_min)

    return x_min, y_min, abs_w, abs_h


def process_fish_data(base_dir, output_dir, file_size_kb, top_k):

    img_out_dir = os.path.join(output_dir, "Images")
    os.makedirs(img_out_dir, exist_ok=True)

    all_candidates = []

    for split in SPLITS:

        img_folder = os.path.join(base_dir, split, "Images")
        lbl_folder = os.path.join(base_dir, split, "Labels")

        if not os.path.exists(img_folder):
            continue

        for img_name in os.listdir(img_folder):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(img_folder, img_name)

            if os.path.getsize(img_path) < (file_size_kb * 1024):
                continue

            label_path = os.path.join(
                lbl_folder, os.path.splitext(img_name)[0] + ".txt"
            )

            boxes = read_yolo_boxes(label_path)

            if not boxes:
                continue

            all_candidates.append(
                {
                    "img_path": img_path,
                    "img_name": img_name,
                    "boxes": boxes,
                }
            )

    top_selection = sorted(
        all_candidates, key=lambda x: len(x["boxes"]), reverse=True
    )[:top_k]

    final_metadata = []
    seen_names = set()

    for item in top_selection:

        # Duplicate basenames can occur across splits; skip rather than
        # silently overwrite an already-copied image.
        if item["img_name"] in seen_names:
            print(f"[WARN] Duplicate image name across splits: {item['img_name']}")
            continue
        seen_names.add(item["img_name"])

        try:
            with Image.open(item["img_path"]) as img:
                img_w, img_h = img.size
        except Exception as e:
            print(f"[WARN] Unreadable image {item['img_path']}: {e}")
            continue
        
        shutil.copy2(item["img_path"], os.path.join(img_out_dir, item["img_name"]))

        num_fish = len(item["boxes"])

        for box in item["boxes"]:
            x_min, y_min, abs_w, abs_h = to_corner_box(box, img_w, img_h)

            final_metadata.append(
                {
                    "annot_uuid": str(uuid.uuid4()),
                    "image_name": item["img_name"],
                    "species": SPECIES_NAME,
                    "bbox_x": x_min,
                    "bbox_y": y_min,
                    "bbox_w": abs_w,
                    "bbox_h": abs_h,
                    "num_fish": num_fish,
                }
            )

    if not final_metadata:
        raise SystemExit(
            "No annotated images survived filtering. Check --base_dir layout "
            "(expects <split>/Images and <split>/Labels) and --file_size_kb."
        )

    df = pd.DataFrame(final_metadata)
    df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

    print(f"Wrote {len(df)} annotations over {df['image_name'].nunique()} images.")


def visualize(output_dir, n):
    """Draw boxes on the first n images for visual verification."""
    import cv2

    csv_path = os.path.join(output_dir, "metadata.csv")
    img_dir = os.path.join(output_dir, "Images")
    viz_dir = os.path.join(output_dir, "Visualize")
    os.makedirs(viz_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    for img_name in df["image_name"].unique()[:n]:
        image = cv2.imread(os.path.join(img_dir, img_name))
        if image is None:
            continue

        rows = df[df["image_name"] == img_name]

        for _, r in rows.iterrows():
            x, y = int(round(r["bbox_x"])), int(round(r["bbox_y"]))
            w, h = int(round(r["bbox_w"])), int(round(r["bbox_h"]))
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.putText(
            image,
            f"Fish: {len(rows)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
        )

        cv2.imwrite(os.path.join(viz_dir, f"viz_{img_name}"), image)

    print(f"Wrote up to {n} verification images to: {viz_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Build the fish counting subset with corner-form bounding boxes"
    )
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--file_size_kb", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=500)
    parser.add_argument(
        "--viz_n",
        type=int,
        default=0,
        help="If > 0, render this many images with boxes drawn for verification",
    )

    args = parser.parse_args()

    process_fish_data(args.base_dir, args.output_dir, args.file_size_kb, args.top_k)

    if args.viz_n > 0:
        visualize(args.output_dir, args.viz_n)