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

from transformers import (
    Qwen3VLForConditionalGeneration, # <-- Reverted back to the dense architecture class
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info


def build_image_path(base_dir, local_path):
    local_path = str(local_path).strip()
    if os.path.isabs(local_path):
        return local_path
    return os.path.join(base_dir, local_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, type=str)
    parser.add_argument("--csv_path", required=True, type=str)
    parser.add_argument("--batch_size", default=4, type=int) # Lowered default for dense 32B
    parser.add_argument("--chunk_id", default=0, type=int)
    parser.add_argument("--num_chunks", default=1, type=int)

    args = parser.parse_args()

    # Updated Model ID to 32B Instruct
    model_id = "Qwen/Qwen3-VL-32B-Instruct"
    print(f"Loading {model_id} with SDPA...", flush=True)

    # Reverted to standard Dense vision-language class
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa" 
    )

    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"

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
        meta_batch = []

        for idx, row in batch_df.iterrows():
            try:
                image_path = build_image_path(args.dataset_dir, row["local_path"])
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

                qa = json.loads(row["qa_pairs"])
                base_question = qa["Q"]
                ground_truth = qa["A"]  
                q_lower = base_question.lower()

                taxonomic_level = None
                keywords = [
                    "phylum", "class", "order", "family", "genus", 
                    "species epithet", "scientific name", 
                    "among the four options", "common name"
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
                    "common name": " Answer with ONLY the everyday English common name (e.g., Orb weavers). Absolutely DO NOT provide the Latin scientific name. Do not write full sentences."
                }

                if taxonomic_level in instruction_router:
                    system_instruction = instruction_router[taxonomic_level]
                else:
                    system_instruction = " Answer directly with ONLY the exact taxonomic term requested. Do not write full sentences and provide no additional context."

                question = base_question + system_instruction

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image", 
                                "image": f"file://{os.path.abspath(final_image_path)}",
                                "max_pixels": 501760
                            },
                            {"type": "text", "text": question}
                        ]
                    }
                ]

                messages_batch.append(messages)
                meta_batch.append({
                    "filename": base_filename,
                    "question": base_question,   
                    "ground_truth": ground_truth 
                })

            except Exception as e:
                print(f"Row {idx} error: {e}", flush=True)

        if len(messages_batch) == 0:
            continue

        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_batch
        ]

        image_inputs, video_inputs = process_vision_info(messages_batch)

        processor_kwargs = {"text": texts, "padding": True, "return_tensors": "pt"}
        if image_inputs: processor_kwargs["images"] = image_inputs
        if video_inputs: processor_kwargs["videos"] = video_inputs

        inputs = processor(**processor_kwargs).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=15, do_sample=False)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        outputs = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        current_batch_answers = [x.strip() for x in outputs]

        for i, ans in enumerate(current_batch_answers):
            q_text = meta_batch[i]['question'].lower()
            if "species epithet" in q_text and len(ans.split()) > 1:
                current_batch_answers[i] = ans.split()[-1].lower()
            elif any(rank in q_text for rank in ["phylum", "class", "order", "family", "genus"]):
                 if len(ans.split()) > 1:
                     current_batch_answers[i] = ans.split()[0].replace(',', '')

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
    
    # Updated output identifier
    df["qwen3_32B_answer"] = answers

    # Ensure the Qwen3-VL-32B subdirectory exists via os.path.join
    csv_dir = os.path.dirname(args.csv_path)
    output_dir = os.path.join(csv_dir, "Qwen3-VL-32B")
    os.makedirs(output_dir, exist_ok=True)

    csv_filename = os.path.basename(args.csv_path)
    base_name = os.path.splitext(csv_filename)[0]
    
    if args.num_chunks > 1:
        output_csv = os.path.join(output_dir, f"{base_name}_chunk{args.chunk_id}_qwen3_32b.csv")
    else:
        output_csv = os.path.join(output_dir, f"{base_name}_qwen3_32b.csv")
    
    df.to_csv(output_csv, index=False)
    print(f"Saved completed dataset to: {output_csv}", flush=True)

if __name__ == "__main__":
    main()