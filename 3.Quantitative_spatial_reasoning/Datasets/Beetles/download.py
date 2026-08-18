import argparse
from huggingface_hub import snapshot_download, hf_hub_download


def download_beetles_dataset(local_dir: str):
    
    snapshot_download(
        repo_id="imageomics/2018-NEON-beetles",
        repo_type="dataset",
        allow_patterns="group_images_resized/*",
        local_dir=local_dir,
    )

    hf_hub_download(
        repo_id="imageomics/2018-NEON-beetles",
        repo_type="dataset",
        filename="individual_specimens/metadata.csv",
        local_dir=local_dir,
    )

    hf_hub_download(
        repo_id="imageomics/2018-NEON-beetles",
        repo_type="dataset",
        filename="BeetleMeasurements.csv",
        local_dir=local_dir,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download NEON Beetles dataset")
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    download_beetles_dataset(args.output_dir)