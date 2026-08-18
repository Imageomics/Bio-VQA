import pandas as pd
import ast
import os
import argparse
import shutil


def resolve_duplicates(group):
    if len(group) == 1:
        return group

    rows_i = group[group["user_name"] == "IsaFluck"]
    
    if len(rows_i) > 0:
        return rows_i.iloc[[0]]
    else:
        return group.iloc[[0]]


def parse_coords(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x


def make_specimen_name(row):
    return row["pictureID"].replace(".jpg", "") + f"_specimen_{int(row['individual'])}.png"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build merged beetle dataset")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory of dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output")

    args = parser.parse_args()
    base_dir = args.base_dir
    output_dir = args.output_dir

    bm_path = os.path.join(base_dir, "BeetleMeasurements.csv")
    ind_path = os.path.join(base_dir, "individual_specimens", "metadata.csv")
    grp_dir = os.path.join(base_dir, "group_images_resized")
    grp_path = os.path.join(grp_dir, "metadata.csv")

    output_csv = os.path.join(output_dir, "metadata.csv")
    output_images_dir = os.path.join(output_dir, "Images")
    os.makedirs(output_images_dir, exist_ok=True)

    bm = pd.read_csv(bm_path)
    ind = pd.read_csv(ind_path)
    grp = pd.read_csv(grp_path)

    bm["coords_pix"] = bm["coords_pix"].apply(parse_coords)
    bm["scalebar"] = bm["scalebar"].apply(parse_coords)

    length_df = bm[bm["structure"] == "ElytraLength"].copy()
    width_df  = bm[bm["structure"] == "ElytraWidth"].copy()

    merged = pd.merge(
        length_df,
        width_df,
        on=["pictureID", "individual"],
        suffixes=("_len", "_wid")
    )

    merged["file_name"] = merged.apply(make_specimen_name, axis=1)


    beetle_counts = merged.groupby("pictureID")["individual"].nunique()

    final = pd.merge(
        merged,
        ind,
        on="file_name",
        how="left"
    )

    final = pd.merge(
        final,
        grp,
        on="pictureID",
        how="left",
        suffixes=("", "_grp")
    )

    final_df = pd.DataFrame({
        "beetle_name": final["file_name"],
        "group_image_path": final["pictureID"].apply(
            lambda x: os.path.join("Images", x) if pd.notna(x) else None
        ),
        "no_of_beetles_in_image": final["pictureID"].map(beetle_counts),
        "scalebar": final["scalebar_len"],
        "elytra_length_coords": final["coords_pix_len"],
        "elytra_width_coords": final["coords_pix_wid"],
        "elytra_length_cm": final["dist_cm_len"],
        "elytra_width_cm": final["dist_cm_wid"],
        "site": final["siteID_grp"],
        "scientificName": final["scientificName_grp"],
        "genus": final["genus"],
        "species_epithet": final["species"],
        "barcode_text": final["pictureID"]
            .where(final["pictureID"].str.startswith("A", na=False))
            .str.replace(".jpg", "", regex=False),
        "user_name": final["user_name_len"]
    })

    final_df = (
        final_df
        .groupby("beetle_name", group_keys=False)
        .apply(resolve_duplicates, include_groups=False)
        .reset_index(drop=True)
    )

    final_df = final_df[final_df["no_of_beetles_in_image"] >= 10].reset_index(drop=True)

    final_df.to_csv(output_csv, index=False)

    unique_image_names = (
        final_df["group_image_path"]
        .dropna()
        .apply(os.path.basename)
        .unique()
    )

    for img_name in unique_image_names:

        src_path = os.path.join(grp_dir, img_name)  
        dst_path = os.path.join(output_images_dir, img_name)

        if not os.path.exists(src_path):
            print(f"⚠️ Source missing: {src_path}")
            continue

        if not os.path.exists(dst_path):
            try:
                shutil.copy(src_path, dst_path)
            except Exception as e:
                print(f"❌ Failed to copy {src_path}: {e}")