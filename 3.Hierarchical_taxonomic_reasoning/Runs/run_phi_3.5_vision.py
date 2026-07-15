import os
import json
import argparse
import math
import torch
import pandas as pd
import cv2
import glob

# --- PIL ROBUSTNESS FIXES ---
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
# ----------------------------

# Phi-3.5-vision is custom-code (phi3_v). Needs trust_remote_code.
#   pip install "transformers>=4.43" accelerate torch pillow opencv-python pandas
# Optional speed-up: install flash-attn and pass --attn flash_attention_2.
# 'eager' is the safe default everywhere.
from transformers import AutoModelForCausalLM, AutoProcessor

# =====================================================================
# COMPATIBILITY SHIMS
# Phi-3.5-vision's remote modeling code was written against an OLD
# transformers Cache API. Newer transformers (>= ~4.46) progressively
# removed/renamed those methods, so generate() crashes with errors like:
#     'DynamicCache' object has no attribute 'seen_tokens'
#     'DynamicCache' object has no attribute 'get_usable_length'
# These shims re-attach the legacy names by mapping them to the current
# API, so the script runs unchanged on old OR new transformers WITHOUT
# touching your venv (important: your Phi-4 runner pins transformers==4.48.2).
# =====================================================================
try:
    from transformers.cache_utils import DynamicCache

    # 1) seen_tokens -> current sequence length
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())

    # 2) get_max_length -> renamed to get_max_cache_shape
    if not hasattr(DynamicCache, "get_max_length"):
        if hasattr(DynamicCache, "get_max_cache_shape"):
            DynamicCache.get_max_length = DynamicCache.get_max_cache_shape
        else:
            DynamicCache.get_max_length = lambda self: None

    # 3) get_usable_length -> faithful re-implementation of the removed method
    if not hasattr(DynamicCache, "get_usable_length"):
        def _get_usable_length(self, new_seq_length, layer_idx=0):
            if hasattr(self, "get_max_cache_shape"):
                max_length = self.get_max_cache_shape()
            elif hasattr(self, "get_max_length"):
                max_length = self.get_max_length()
            else:
                max_length = None
            previous_seq_length = self.get_seq_length(layer_idx)
            if max_length is not None and previous_seq_length + new_seq_length > max_length:
                return max_length - new_seq_length
            return previous_seq_length
        DynamicCache.get_usable_length = _get_usable_length
except Exception as _e:
    print(f"[WARN] Could not apply DynamicCache shim: {_e}", flush=True)
# =====================================================================


def build_image_path(base_dir, local_path):
    local_path = str(local_path).strip()
    if os.path.isabs(local_path):
        return local_path
    return os.path.join(base_dir, local_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, type=str)
    parser.add_argument("--csv_path", required=True, type=str)
    # batch_size accepted for interface parity with the other runners, but
    # Phi-3-vision does not batch cleanly, so generation is sequential.
    # This value only controls the progress-log cadence grouping.
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--chunk_id", default=0, type=int)
    parser.add_argument("--num_chunks", default=1, type=int)
    parser.add_argument("--num_crops", default=16, type=int,
                        help="16 = best for single images; lower to save memory")
    parser.add_argument("--attn", default="eager", type=str,
                        help="eager (safe default) | sdpa | flash_attention_2")
    args = parser.parse_args()

    model_id = "microsoft/Phi-3.5-vision-instruct"
    print(f"Loading {model_id} (trust_remote_code, attn={args.attn})...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        _attn_implementation=args.attn,
    ).eval()

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        num_crops=args.num_crops,
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

    for idx, row in df.iterrows():
        try:
            image_path = build_image_path(args.dataset_dir, row["local_path"])
            if not os.path.exists(image_path):
                print(f"Missing image: {image_path}", flush=True)
                answers.append("")
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
                    answers.append("")
                    continue

            pil_image = Image.open(final_image_path).convert("RGB")

            base_filename = os.path.basename(image_path)
            seen_images.add(base_filename)

            qa = json.loads(row["qa_pairs"])
            base_question = qa["Q"]
            ground_truth = qa["A"]
            q_lower = base_question.lower()

            taxonomic_level = None
            keywords = [
                "phylum", "class", "order", "family", "genus",
                "species epithet", "scientific name",
                "among the four options", "common name",
            ]
            for keyword in keywords:
                if keyword in q_lower:
                    taxonomic_level = keyword
                    break

            instruction_router = {
                "phylum": " Answer with ONLY the Phylum name (e.g., Arthropoda). Do not include class, order, or any other taxonomic ranks. Do not write full sentences.",
                "class": " Answer with ONLY the Class name (e.g., Arachnida, Insecta). Do not include order, family, or any other taxonomic ranks. Do not write full sentences.",
                "order": " Answer with ONLY the Order name (e.g., Araneae). Do not include family, genus, or any other taxonomic ranks. Do not write full sentences.",
                "family": " Answer with ONLY the Family name (e.g., Araneidae). Do not include genus, species, or any other taxonomic ranks. Do not write full sentences.",
                "genus": " Answer with ONLY the Genus name (capitalized, e.g., Argiope). Do not include the species epithet or family name. Do not write full sentences.",
                "species epithet": " Answer with ONLY the species epithet (the single, lowercase second word of the scientific name, e.g., aurantia). Absolutely DO NOT include the capitalized Genus name. Do not write full sentences.",
                "scientific name": " Answer with ONLY the full binomial scientific name (Genus and species, e.g., Argiope aurantia). Do not include the common name or higher taxonomic ranks. Do not write full sentences.",
                "among the four options": " Answer with ONLY the exact correct option from the four choices provided in the question. Absolutely DO NOT write full sentences, and do not provide any extra context or scientific names.",
                "common name": " Answer with ONLY the everyday English common name (e.g., Orb weavers). Absolutely DO NOT provide the Latin scientific name. Do not write full sentences.",
            }

            if taxonomic_level in instruction_router:
                system_instruction = instruction_router[taxonomic_level]
            else:
                system_instruction = " Answer directly with ONLY the exact taxonomic term requested. Do not write full sentences and provide no additional context."

            question = base_question + system_instruction

            # Phi-3 vision image placeholder format
            messages = [
                {"role": "user", "content": "<|image_1|>\n" + question},
            ]
            prompt = processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = processor(prompt, [pil_image], return_tensors="pt").to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    max_new_tokens=15,
                    do_sample=False,
                )

            generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
            ans = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            # post-process to the requested taxonomic term
            if "species epithet" in q_lower and len(ans.split()) > 1:
                ans = ans.split()[-1].lower()
            elif any(rank in q_lower for rank in ["phylum", "class", "order", "family", "genus"]):
                if len(ans.split()) > 1:
                    ans = ans.split()[0].replace(',', '')

            answers.append(ans)

            row_counter += 1
            if row_counter % 10 == 0 or row_counter == 1:
                print("-" * 80, flush=True)
                print(f"[PROGRESS LOG] Question {row_counter}/{total_rows} | Unique Images Seen: {len(seen_images)}", flush=True)
                print(f"File:     {base_filename}", flush=True)
                print(f"Query:    {base_question}", flush=True)
                print(f"Answer:   {ans}", flush=True)
                print(f"G-Truth:  {ground_truth}", flush=True)
                print("-" * 80 + "\n", flush=True)

        except Exception as e:
            print(f"Row {idx} error: {e}", flush=True)
            answers.append("")

        for temp_file in glob.glob(f"temp_frame_chunk{args.chunk_id}_*.jpg"):
            try:
                os.remove(temp_file)
            except Exception:
                pass

    if len(answers) != len(df):
        print(f"WARNING: answers={len(answers)} rows={len(df)}", flush=True)

    df = df.iloc[:len(answers)].copy()
    df["phi35_vision_answer"] = answers

    csv_dir = os.path.dirname(args.csv_path)
    output_dir = os.path.join(csv_dir, "Phi-3.5-vision")
    os.makedirs(output_dir, exist_ok=True)

    csv_filename = os.path.basename(args.csv_path)
    base_name = os.path.splitext(csv_filename)[0]
    if args.num_chunks > 1:
        output_csv = os.path.join(output_dir, f"{base_name}_chunk{args.chunk_id}_phi35_vision.csv")
    else:
        output_csv = os.path.join(output_dir, f"{base_name}_phi35_vision.csv")

    df.to_csv(output_csv, index=False)
    print(f"Saved completed dataset to: {output_csv}", flush=True)


if __name__ == "__main__":
    main()