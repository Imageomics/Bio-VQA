import os
import csv
import uuid
import argparse

def get_classes(root_path):
    classes_file = os.path.join(root_path, "classes.txt")
    if not os.path.exists(classes_file):
        raise FileNotFoundError(f"classes.txt not found at {root_path}")
    with open(classes_file, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def create_metadata(root_path, output_file):
    original_classes = get_classes(root_path)
    splits = ['train', 'test', 'valid']
    
    header = [
        "annot_uuid", 
        "image_path", 
        "split", 
        "class_name", 
        "x_center", "y_center", "width", "height", 
        "num_animals_in_image"
    ]

    rows = []

    for split in splits:
        img_dir = os.path.join(root_path, "images", split)
        lbl_dir = os.path.join(root_path, "labels", split)

        if not os.path.exists(img_dir):
            continue

        print(f"Processing {split} split...")

        for img_name in os.listdir(img_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            img_path_rel = os.path.join("images", split, img_name)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(lbl_dir, label_name)

            if not os.path.exists(label_path):
                continue

            with open(label_path, 'r') as f:
                annotations = [line.strip().split() for line in f.readlines() if line.strip()]
            
            num_animals = len(annotations)

            for ann in annotations:
                class_idx = int(ann[0])
                
                # According to our finding: 
                # If label index is 3, it should be Kiang.
                # If label index is 4, it should be Camel.
                if class_idx == 3:
                    class_name = "kiang"
                elif class_idx == 4:
                    class_name = "camel"
                else:
                    # Fallback to original list for 0, 1, 2, 5...
                    class_name = original_classes[class_idx] if class_idx < len(original_classes) else "unknown"

                # Ensure 'camelus' becomes 'camel' globally
                if class_name == "camelus":
                    class_name = "camel"

                rows.append([
                    str(uuid.uuid4()), 
                    img_path_rel,
                    split,
                    class_name,
                    ann[1], ann[2], ann[3], ann[4],
                    num_animals
                ])

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate FIXED metadata.csv from WAID YOLO dataset")
    parser.add_argument("base_dir", type=str, help="Path to the root WAID directory")
    
    args = parser.parse_args()
    output_file = os.path.join(args.base_dir, "metadata.csv")

    create_metadata(args.base_dir, output_file)