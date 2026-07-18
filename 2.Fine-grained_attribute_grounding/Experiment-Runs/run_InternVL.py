import os
import json
import argparse
import math
import torch
import pandas as pd
import cv2
import glob

from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
PreTrainedModel.all_tied_weights_keys = property(lambda self: {})

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------- InternVL dynamic-tiling image preprocessing ----------
def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values
# -----------------------------------------------------------------


def build_image_path(base_dir, local_path):
    local_path = str(local_path).strip()
    if os.path.isabs(local_path):
        return local_path
    return os.path.join(base_dir, local_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, type=str)
    parser.add_argument("--csv_path", required=True, type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--chunk_id", default=0, type=int)
    parser.add_argument("--num_chunks", default=1, type=int)
    parser.add_argument("--model_id", default="OpenGVLab/InternVL3_5-8B-Instruct", type=str)
    parser.add_argument("--max_num", default=12, type=int,
                        help="max image tiles; lower (e.g. 6) to save memory on big models")
    parser.add_argument("--use_flash_attn", action="store_true",
                        help="enable flash-attention (must be installed)")
    args = parser.parse_args()

    model_id = args.model_id
    print(f"Loading {model_id} (trust_remote_code, flash_attn={args.use_flash_attn})...", flush=True)

    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=args.use_flash_attn,
        trust_remote_code=True,
        device_map="auto"
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    generation_config = dict(
    max_new_tokens=30,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,   # <-- silences the per-call log line
    )

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

        pixel_values_list = []
        num_patches_list = []
        questions = []
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
                        "3. Output ONLY the matching option letter in parentheses and its exact trailing text string. "
                        "For example, if the correct option is '(X) alpha bravo charlie', your entire response must be exactly: (X) alpha bravo charlie\n"
                        "4. DO NOT truncate the response to just the option letter (e.g., do not output just '(X)').\n"
                        "5. DO NOT invent, hallucinate, or insert any placeholder words from external examples. "
                        "Absolutely DO NOT output any word unless it is explicitly written inside the chosen option string above.\n"
                        "6. Output exactly zero extra words, zero markdown formatting, and zero punctuation outside the option string."
                    )
                elif "scientific name" in q_lower:
                    system_instruction = " Please answer in English. Answer with ONLY the scientific name of the bird. Do not write full sentences."
                elif "genus" in q_lower:
                    system_instruction = " Please answer in English. Answer with ONLY the capitalized Genus name of the bird. Do not write full sentences."
                elif "common name" in q_lower or "what kind" in q_lower:
                    system_instruction = " Please answer in English. Answer with ONLY the everyday English common name of the bird. Do not write full sentences."
                else:
                    system_instruction = " Please answer in English. Answer directly and concisely with only the exact terms requested. Do not write full sentences."

                question = base_question + system_instruction

                pv = load_image(final_image_path, max_num=args.max_num)
                pixel_values_list.append(pv)
                num_patches_list.append(pv.size(0))
                questions.append("<image>\n" + question)

                meta_batch.append({
                    "filename": base_filename,
                    "question": base_question,
                    "ground_truth": ground_truth,
                })

            except Exception as e:
                print(f"Row {idx} error: {e}", flush=True)

        if len(questions) == 0:
            continue

        pixel_values = torch.cat(pixel_values_list, dim=0).to(torch.bfloat16).to(model.device)
        
        with torch.no_grad():
            outputs = model.batch_chat(
                tokenizer,
                pixel_values,
                num_patches_list=num_patches_list,
                questions=questions,
                generation_config=generation_config,
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

    model_tag = model_id.split("/")[-1].replace(".", "_").replace("-", "_")
    df[f"{model_tag}_answer"] = answers
    base_name = os.path.splitext(args.csv_path)[0]
    if args.num_chunks > 1:
        output_csv = f"{base_name}_chunk{args.chunk_id}_{model_tag}.csv"
    else:
        output_csv = f"{base_name}_{model_tag}.csv"

    df.to_csv(output_csv, index=False)
    print(f"Saved completed dataset to: {output_csv}", flush=True)


if __name__ == "__main__":
    main()