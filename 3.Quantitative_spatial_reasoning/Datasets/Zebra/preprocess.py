import os
import shutil
import argparse
import pandas as pd
import itertools


MIN_SIZE_MB = 2
MIN_SIZE_BYTES = MIN_SIZE_MB * 1024 * 1024


def compute_iou(box1, box2):

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter_area = inter_w * inter_h

    union = (w1 * h1) + (w2 * h2) - inter_area

    return inter_area / union if union > 0 else 0


def has_excessive_overlap(group, iou_threshold=0.5, max_pairs=2):

    boxes = group[["bbox x", "bbox y", "bbox w", "bbox h"]].values

    count = 0
    for b1, b2 in itertools.combinations(boxes, 2):
        if compute_iou(b1, b2) > iou_threshold:
            count += 1
            if count > max_pairs:
                return True
    return False

def get_top_k_images(csv_path, image_dir, k=500):

    df = pd.read_csv(csv_path)
    
    df = df[~df["annot species"].str.lower().str.contains("human", na=False)]
    
    zebra_mask = df["annot species"].str.lower().str.contains("zebra", na=False)
    zebra_images = set(df[zebra_mask]["image uuid"])
    df = df[df["image uuid"].isin(zebra_images)]
    
    valid_groups = []
    for img_uuid, group in df.groupby("image uuid"):
        if not has_excessive_overlap(group):
            valid_groups.append(group)

    df = pd.concat(valid_groups, ignore_index=True)
    
    valid_uuids = []
    for img_uuid, group in df.groupby("image uuid"):
        img_name = group.iloc[0]["image fname"]
        img_path = os.path.join(image_dir, img_name)

        if os.path.exists(img_path) and os.path.getsize(img_path) >= MIN_SIZE_BYTES:
            valid_uuids.append(img_uuid)

    df = df[df["image uuid"].isin(valid_uuids)]
    
    counts = (
        df.groupby("image uuid")["annot uuid"]
        .nunique()
        .reset_index(name="num_animals")
    )

    top_images = counts.sort_values("num_animals", ascending=False).head(k)

    return df, top_images


def copy_images_and_metadata(df, top_images, image_dir, output_dir):

    output_img_dir = os.path.join(output_dir, "Images")
    os.makedirs(output_img_dir, exist_ok=True)

    selected_uuids = set(top_images["image uuid"])

    kept_uuids = []
    copied = 0

    for img_uuid in selected_uuids:
        rows = df[df["image uuid"] == img_uuid]
        img_name = rows.iloc[0]["image fname"]

        src = os.path.join(image_dir, img_name)

        if not os.path.exists(src):
            print(f"[Missing] {src}")
            continue
        
        if os.path.getsize(src) < MIN_SIZE_BYTES:
            print(f"[Too small] {img_name}")
            continue

        dst = os.path.join(output_img_dir, img_name)
        shutil.copy2(src, dst)

        kept_uuids.append(img_uuid)
        copied += 1

    print(f"Copied {copied} images (>= {MIN_SIZE_MB}MB)")
    
    filtered_df = df[df["image uuid"].isin(kept_uuids)]

    return filtered_df


def save_metadata(filtered_df, output_dir):

    output_csv = os.path.join(output_dir, "metadata.csv")
    os.makedirs(output_dir, exist_ok=True)

    filtered_df.to_csv(output_csv, index=False)

    print(f"Metadata saved to: {output_csv}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Extract top-K zebra images with quality + overlap filtering")

    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=500)

    args = parser.parse_args()

    df, top_images = get_top_k_images(
        args.csv_path,
        args.image_dir,
        args.top_k
    )

    filtered_df = copy_images_and_metadata(
        df,
        top_images,
        args.image_dir,
        args.output_dir
    )

    filtered_df = filtered_df[filtered_df["census annot region"] == False]

    save_metadata(filtered_df, args.output_dir)

    print("Done.")