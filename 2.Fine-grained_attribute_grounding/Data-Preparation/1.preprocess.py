import os
import json
import numpy as np
import pandas as pd


BASE_DIR = "/path/to/your/dataset/directory"  # Update this to your dataset directory

OUT_FILE = os.path.join(BASE_DIR, "CUB200_attributes.csv")

SCI_MAP_FILE = os.path.join(BASE_DIR, "taxon_name_mapping.json")

ATTR_FILE = os.path.join(BASE_DIR, "attributes/image_attribute_labels.txt")
ATTR_NAME_FILE = os.path.join(BASE_DIR, "attributes/attributes.txt")
CERTAINTY_FILE = os.path.join(BASE_DIR, "attributes/certainties.txt")
CLASSES_FILE = os.path.join(BASE_DIR, "classes.txt")
IMAGE_FILE = os.path.join(BASE_DIR, "images.txt")
IMAGE_CLASS_FILE = os.path.join(BASE_DIR, "image_class_labels.txt")
BBOX_FILE = os.path.join(BASE_DIR, "bounding_boxes.txt")
PARTS_FILE = os.path.join(BASE_DIR, "parts/parts.txt")
PART_LOCS_FILE = os.path.join(BASE_DIR, "parts/part_locs.txt")
SPLIT_FILE = os.path.join(BASE_DIR, "train_test_split.txt")


def load_attributes(file_path):

    rows = []
    skipped = 0
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    rows.append([
                        int(parts[0]),     # image_id
                        int(parts[1]),     # attr_id
                        int(parts[2]),     # present
                        int(parts[3]),     # certainty
                        float(parts[-1])   # score
                    ])
                except Exception:
                    skipped += 1
            else:
                skipped += 1

    df = pd.DataFrame(rows, columns=["image_id", "attr_id", "present", "certainty", "score"])
    print(f"Loaded {len(df)} attribute rows, skipped {skipped}")
    return df


def load_simple_mapping(file_path, delimiter=" ", value_type=str):
    mapping = {}
    if os.path.exists(file_path):
        with open(file_path) as f:
            for line in f:
                parts = line.strip().split(delimiter, 1)
                if len(parts) == 2:
                    mapping[int(parts[0])] = value_type(parts[1])
    return mapping


def load_images_and_classes(img_file, class_file):
    img_files = load_simple_mapping(img_file)
    img_classes = load_simple_mapping(class_file, value_type=int)
    return img_files, img_classes


def load_bboxes(file_path):
    bboxes = {}
    with open(file_path) as f:
        for line in f:
            idx, x, y, w, h = line.strip().split()
            bboxes[int(idx)] = [float(x), float(y), float(w), float(h)]
    return bboxes


def load_parts(parts_file, loc_file):
    parts_mapping = load_simple_mapping(parts_file)
    part_locs = {}

    with open(loc_file) as f:
        for line in f:
            img_id, part_id, x, y, visible = line.strip().split()
            img_id, part_id, visible = int(img_id), int(part_id), int(visible)
            
            part_name = parts_mapping.get(part_id, f"part_{part_id}")

            if img_id not in part_locs:
                part_locs[img_id] = {}
            
            part_locs[img_id][part_name] = {
                "visible": visible,
                "x": float(x) if visible == 1 else 0.0,
                "y": float(y) if visible == 1 else 0.0
            }

    return parts_mapping, part_locs


def load_train_test_splits(split_file):
    return load_simple_mapping(split_file, value_type=int)


def build_trait_dict(attr_df, attr_names, certainties):
    trait_dict_per_image = {}
    attr_names_list = [attr_names.get(i, f"attr_{i}") for i in range(1, len(attr_names) + 1)]
    
    for _, r in attr_df.iterrows():
        img_id = int(r.image_id)
        attr_id = int(r.attr_id) - 1
        
        trait_name = attr_names_list[attr_id]
        certainty_str = certainties.get(int(r.certainty), str(int(r.certainty)))

        if img_id not in trait_dict_per_image:
            trait_dict_per_image[img_id] = {}

        trait_dict_per_image[img_id][trait_name] = {
            "present": int(r.present),
            "certainty": certainty_str,
            "score": float(r.score)
        }

    return trait_dict_per_image


def build_csv():
    print("Loading core mappings...")
    classes = load_simple_mapping(CLASSES_FILE)
    img_files, img_classes = load_images_and_classes(IMAGE_FILE, IMAGE_CLASS_FILE)
    bboxes = load_bboxes(BBOX_FILE)
    splits = load_train_test_splits(SPLIT_FILE)
    
    print("Loading attributes and metadata...")
    attr_df = load_attributes(ATTR_FILE)
    attr_names = load_simple_mapping(ATTR_NAME_FILE)
    certainties = load_simple_mapping(CERTAINTY_FILE)
    
    print("Building enriched trait dicts...")
    trait_dict_per_image = build_trait_dict(attr_df, attr_names, certainties)

    print("Loading parts...")
    parts_mapping, part_locs = load_parts(PARTS_FILE, PART_LOCS_FILE)

    print("Loading Taxon names...")
    sci_map = {}
    if os.path.exists(SCI_MAP_FILE):
        with open(SCI_MAP_FILE, "r") as f:
            sci_map = json.load(f)
    else:
        print(f"Warning: Taxon mapping missing at {SCI_MAP_FILE}")

    print("Compiling final dataset...")
    rows = []
    
    default_parts = {
        name: {"visible": 0, "x": 0.0, "y": 0.0} 
        for name in parts_mapping.values()
    }

    for img_id, fname in img_files.items():
        raw_class_name = classes.get(img_classes[img_id], "")
        clean_class_name = raw_class_name.split(".")[-1].replace("_", " ")
        
        row = {
            "image_id": img_id,
            "filename": fname,
            "class_id": img_classes[img_id],
            "class_name": clean_class_name,
            "taxon": sci_map.get(clean_class_name, "Unknown"),
            "is_training_image": splits.get(img_id, -1),
            "bbox": bboxes.get(img_id, []),
            "attributes": json.dumps(trait_dict_per_image.get(img_id, {})),
            "parts": json.dumps(part_locs.get(img_id, default_parts))
        }
        rows.append(row)

    df_final = pd.DataFrame(rows)
    return df_final


if __name__ == "__main__":
    
    df = build_csv()
    
    print(f"\nTotal images processed: {len(df)}")
    print(f"Sample Classes:\n{df[['class_name', 'taxon']].head(3)}\n")
    
    df.to_csv(OUT_FILE, index=False)
    
    print(f"Saved merged dataset successfully to: {OUT_FILE}")