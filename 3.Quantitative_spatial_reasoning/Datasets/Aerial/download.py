import os
import argparse
import subprocess


def clone_repo(repo_url, local_path) :

    os.makedirs(local_path, exist_ok=True)
    
    if os.listdir(local_path):
        print(f"[WARNING] Target directory is not empty: {local_path}")

    try:
        subprocess.run(
            ["git", "clone", repo_url, local_path],
            check=True
        )
        print(f"Cloned successfully into: {local_path}")

    except subprocess.CalledProcessError as e:
        print("Error during cloning:", e)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="GitHub repository URL"
    )

    parser.add_argument(
        "--local_path",
        type=str,
        required=True,
        help="Local directory to clone into"
    )

    args = parser.parse_args()

    clone_repo(args.repo, args.local_path)