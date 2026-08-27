import os
import requests
import argparse
from tqdm import tqdm
import zipfile
import tarfile


def download_file(url, output_path):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))

        with open(output_path, "wb") as f, tqdm(
            desc="Downloading",
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    size = f.write(chunk)
                    bar.update(size)

    print(f"Download complete! Saved to: {output_path}")




def extract_archive(file_path, extract_to=None):
    if extract_to is None:
        extract_to = os.path.splitext(file_path)[0]

    os.makedirs(extract_to, exist_ok=True)

    # ZIP
    if file_path.endswith(".zip"):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

    # TAR / TAR.GZ / TGZ
    elif file_path.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(file_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)

    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    print(f"Extracted to: {extract_to}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download and extract dataset")
    parser.add_argument("--url", type=str, required=True, help="File URL")
    parser.add_argument("--output", type=str, required=True, help="Local file path")
    parser.add_argument("--extract", action="store_true", help="Extract after download")

    args = parser.parse_args()

    download_file(args.url, args.output)

    if args.extract:
        extract_archive(args.output)
