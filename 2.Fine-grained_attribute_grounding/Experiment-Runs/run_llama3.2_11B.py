import os
import json
import argparse
import math
import time
import torch
import pandas as pd
import cv2
import glob

# --- PIL ROBUSTNESS FIXES ---
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
# ----------------------------

from transformers import MllamaForConditionalGeneration, AutoProcessor


def build_image_path(base_dir, local_path):
    local_path = str(local_path).strip()
    if os.path.isabs(local_path):
        return local_path
    return os.path.join(base_dir, local_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, type=str)
    parser.add_argument("--csv_path", required=True, type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--chunk_id", default=0, type=int)
    parser.add_argument("--num_chunks", default=1, type=int)
    args = parser.parse_args()

    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    print(f"Loading {model_id} with SDPA...", flush=True)

    # --- HPC RACE CONDITION FIX: Retry loop for model loading ---
    max_retries = 3
    model = None
    processor = None
    
    for attempt in range(max_retries):
        try:
            model = MllamaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="sdpa",
            )
            processor = AutoProcessor.from_pretrained(model_id)
            processor.tokenizer.padding_side = "left"
            break  # Success, exit the retry loop
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[WARNING] Failed to load model/processor (attempt {attempt+1}/{max_retries}). Retrying in 30s... Error: {e}", flush=True)
                time.sleep(30)
            else:
                print(f"[ERROR] Completely failed to load model after {max_retries} attempts.", flush=True)
                raise
    # ------------------------------------------------------------

    df = pd.read_csv(args.csv_path)

    if args.num_chunks > 1:
        chunk_size = math.ceil(len(df) / args.num_chunks)
        start_row = args.chunk_id * chunk_size
        end_row = min(start_row + chunk_size, len(df))
        df = df.iloc[start_row:end_row].copy()
        print(f"Running Chunk {args.chunk_id + 1}/{args.num_chunks}. Rows {start_row} to {end_row}.", flush=True)

    answers = []
    total_rows = len(df)
    print(f"Total Rows for this job: {total_rows}\n", flush=True)

    seen_images = set()
    row_counter = 0

    for start_idx in range(0, total_rows, args.batch_size):
        batch_df = df.iloc[start_idx:start_idx + args.batch_size]

        messages_batch = []
        images_batch = []
        meta_batch = []

        for idx, row in batch_df.iterrows():
            try:
                image_path = build_image_path(args.dataset_dir, row["filename"])
                if not os.path.exists(image_path):
                    print(f"Missing image: {image_path}", flush=True)
                    continue

                final_image_path = image_path
                try:
                    with Image.open(image_path) as img:
                        img.verify()
                except Exception:
                    print(f"\n[WARNING] Unidentified file (likely hidden video): {image_path}", flush=True)
                    cap = cv2.VideoCapture(image_path)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        final_image_path = f"temp_frame_chunk{args.chunk_id}_row{idx}.jpg"
                        cv2.imwrite(final_image_path, frame)
                        print(f"Success! Extracted frame saved as {final_image_path}\n", flush=True)
                    else:
                        print(f"Error: Could not extract frame from {image_path}. Skipping.\n", flush=True)
                        continue

                pil_image = Image.open(final_image_path).convert("RGB")

                base_filename = os.path.basename(image_path)
                seen_images.add(base_filename)

                base_question = str(row["question"]).strip()
                ground_truth = str(row["answer"]).strip()
                q_lower = base_question.lower()

                if "options:" in q_lower:
                    system_instruction = (
                        "\n\n[CRITICAL INSTRUCTIONS - NO VIOLATIONS ALLOWED]:\n"
                        "1. Locate the 'Options:' section in the question above.\n"
                        "2. Identify the single correct choice from that list.\n"
                        "3. Output ONLY the matching option letter in parentheses and its exact trailing text string."
                        "4. DO NOT truncate the response to just the option letter.\n"
                        "5. DO NOT invent, hallucinate, or insert any placeholder words from external examples. "
                        "Absolutely DO NOT output any word unless it is explicitly written inside the chosen option string above.\n"
                        "6. Output exactly zero extra words, zero markdown formatting, and zero punctuation outside the option string."
                    )
                elif "scientific name" in q_lower:
                    system_instruction = " Answer with ONLY the scientific name of the bird. Do not write full sentences."
                elif "genus" in q_lower:
                    system_instruction = " Answer with ONLY the capitalized Genus name of the bird. Do not write full sentences."
                elif "common name" in q_lower or "what kind" in q_lower:
                    system_instruction = " Answer with ONLY the everyday English common name of the bird. Do not write full sentences."
                else:
                    system_instruction = " Answer directly and concisely with only the exact terms requested. Do not write full sentences."

                question = base_question + system_instruction

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": question},
                        ],
                    }
                ]

                messages_batch.append(messages)
                images_batch.append([pil_image]) 
                meta_batch.append({
                    "filename": base_filename,
                    "question": base_question,
                    "ground_truth": ground_truth,
                })

            except Exception as e:
                print(f"Row {idx} error: {e}", flush=True)

        if len(messages_batch) == 0:
            continue

        texts = [
            processor.apply_chat_template(msg, add_generation_prompt=True)
            for msg in messages_batch
        ]

        inputs = processor(
            images=images_batch,
            text=texts,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=30, do_sample=False)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        outputs = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        current_batch_answers = [x.strip() for x in outputs]


        for i, ans in enumerate(current_batch_answers):
            row_counter += 1
            if row_counter % 10 == 0 or row_counter == 1:
                print("-" * 80, flush=True)
                print(f"[PROGRESS LOG] Question {row_counter}/{total_rows} | Unique Images Seen: {len(seen_images)}", flush=True)
                print(f"File:     {meta_batch[i]['filename']}", flush=True)
                print(f"Query:    {meta_batch[i]['question']}", flush=True)
                print(f"Answer:   {ans}", flush=True)
                print(f"G-Truth:  {meta_batch[i]['ground_truth']}", flush=True)
                print("-" * 80 + "\n", flush=True)

        answers.extend(current_batch_answers)

        for temp_file in glob.glob(f"temp_frame_chunk{args.chunk_id}_*.jpg"):
            try:
                os.remove(temp_file)
            except Exception:
                pass

    if len(answers) != len(df):
        print(f"WARNING: answers={len(answers)} rows={len(df)}", flush=True)

    df = df.iloc[:len(answers)].copy()
    df["llama3.2_11B_answer"] = answers
    base_name = os.path.splitext(args.csv_path)[0]
    if args.num_chunks > 1:
        output_csv = f"{base_name}_chunk{args.chunk_id}_llama3.2_11B.csv"
    else:
        output_csv = f"{base_name}_llama3.2_11B.csv"

    df.to_csv(output_csv, index=False)
    print(f"Saved completed dataset to: {output_csv}", flush=True)


if __name__ == "__main__":
    main()