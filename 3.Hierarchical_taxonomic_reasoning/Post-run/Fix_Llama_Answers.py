import os
import argparse
import pandas as pd

MODEL_DIR_NAME = "Llama-3.2-11B-Vision"


def find_merged_csv(model_dir):
    csvs = [
        f for f in os.listdir(model_dir)
        if f.lower().endswith(".csv")
    ]

    merged = [
        f for f in csvs
        if "chunk" not in f.lower()
    ]

    if not merged:
        return None

    qa = [
        f for f in merged
        if "qa_difficulty" in f.lower()
    ]

    candidates = qa if qa else merged

    candidates.sort(
        key=lambda x: os.path.getsize(os.path.join(model_dir, x)),
        reverse=True,
    )

    return os.path.join(model_dir, candidates[0])


def find_answer_column(df):
    candidates = [
        c for c in df.columns
        if c.lower().endswith("_answer")
    ]

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one *_answer column. Found: {candidates}"
        )

    return candidates[0]


def clean_answer(ans):
    if pd.isna(ans):
        return ans

    ans = str(ans).strip()

    ans = ans.replace("**", "")
    ans = ans.strip()

    # Keep only text before the first period
    ans = ans.split(".", 1)[0].strip()

    return ans


def main(base_dir):
    processed = 0

    print("=" * 80)
    print("Cleaning Llama-3.2-11B-Vision answers")
    print("=" * 80)

    for dataset in sorted(os.listdir(base_dir)):
        dataset_dir = os.path.join(base_dir, dataset)

        if not os.path.isdir(dataset_dir):
            continue

        model_dir = os.path.join(dataset_dir, MODEL_DIR_NAME)

        if not os.path.isdir(model_dir):
            print(f"[SKIP] {dataset} (no {MODEL_DIR_NAME})")
            continue

        merged_csv = find_merged_csv(model_dir)

        if merged_csv is None:
            print(f"[SKIP] {dataset} (no merged csv)")
            continue

        print(f"\nProcessing: {merged_csv}")

        df = pd.read_csv(merged_csv)

        answer_col = find_answer_column(df)

        before = df[answer_col].copy()

        df[answer_col] = df[answer_col].apply(clean_answer)

        changed = (before.astype(str) != df[answer_col].astype(str)).sum()

        df.to_csv(merged_csv, index=False)

        print(f"  Answer column : {answer_col}")
        print(f"  Rows modified : {changed}")
        print("  Saved.")

        processed += 1

    print("\n" + "=" * 80)
    print(f"Finished. Processed {processed} datasets.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean Llama-3.2-11B-Vision answer columns."
    )
    parser.add_argument(
        "base_dir",
        help="Base directory containing all dataset folders.",
    )

    args = parser.parse_args()

    main(args.base_dir)