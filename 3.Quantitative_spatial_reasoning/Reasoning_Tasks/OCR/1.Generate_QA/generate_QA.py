import os
import pandas as pd
import csv
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_dir",
        required=True,
        help="Root dataset directory"
    )

    parser.add_argument(
        "--barcode_dir_basename",
        required=True,
        help="Relative folder containing barcode dataset"
    )

    parser.add_argument(
        "--barcode_csv_basename",
        default="Barcodes.csv",
        help="CSV filename inside barcode folder"
    )

    args = parser.parse_args()

    barcode_dir = os.path.join(args.base_dir, args.barcode_dir_basename)

    basecode_image_dir_basename = "Barcodes"

    basecode_image_dir = os.path.join(barcode_dir, basecode_image_dir_basename)

    barcode_csv = os.path.join(barcode_dir, args.barcode_csv_basename)

    barcode_data = pd.read_csv(barcode_csv)

    output_csv = os.path.join(barcode_dir, "qa.csv")

    question = 'What is the text under the barcode (starting with A0...) shown in this image?'

    rows = []

    for _, row in barcode_data.iterrows():

        image_path = row["image_path"] if "image_path" in row else row["image"]
        barcode = row["barcode"]

        image_basename = os.path.basename(image_path)
        relative_image_path = os.path.join(args.barcode_dir_basename, basecode_image_dir_basename, image_basename)

        qa_dict = {
            "Q": question,
            "A": str(barcode)
        }

        rows.append({
            "image_path": relative_image_path,
            "qa": str(qa_dict)
        })

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "qa"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved QA CSV to: {output_csv}")
    print(f"Total samples: {len(rows)}")