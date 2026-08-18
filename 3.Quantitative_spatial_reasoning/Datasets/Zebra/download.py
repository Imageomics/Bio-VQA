import argparse
from huggingface_hub import snapshot_download, hf_hub_download


def download_zebra_dataset(local_dir: str):

    snapshot_download(
        repo_id="imageomics/GZCD",
        repo_type="dataset",
        allow_patterns="images/*",
        local_dir=local_dir,
    )

    hf_hub_download(
        repo_id="imageomics/GZCD",
        repo_type="dataset",
        filename="final_metadata.json",
        local_dir=local_dir,
    )

    hf_hub_download(
        repo_id="imageomics/GZCD",
        repo_type="dataset",
        filename="initial_metadata.json",
        local_dir=local_dir,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download GZCD zebra dataset")

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    download_zebra_dataset(args.output_dir)