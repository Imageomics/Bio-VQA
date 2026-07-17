import os
import random
import pandas as pd
import json
import uuid
import cv2
import math


random.seed(42)

BASE_DIR = "/path/to/your/dataset/directory" # Update this to your dataset directory

IN_CSV = os.path.join(BASE_DIR, "CUB200_attributes.csv")
OUT_CSV = os.path.join(BASE_DIR, "QA.csv")


IMAGES_DIR = os.path.join(BASE_DIR, "images") 
ANNOTATED_DIR = os.path.join(BASE_DIR, "Images_Annotated")
os.makedirs(ANNOTATED_DIR, exist_ok=True)


_LOGGED_GENUS = False


COLORS = {
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "white": (255, 255, 255),
    "yellow": (0, 255, 255)
}

SHAPES = ["dot", "cross", "plus", "star"]


def parse_json_safely(data_str):

    if pd.isna(data_str):
        return {}
    try:
        return json.loads(data_str)
    except json.JSONDecodeError:
        data_str = data_str.replace('""', '"')
        try:
            return json.loads(data_str)
        except json.JSONDecodeError:
            return {}


def build_attribute_pools(df):

    group_pool = {}
    full_trait_pool = set()
    for attr_str in df["attributes"]:
        attrs = parse_json_safely(attr_str)
        for k, v in attrs.items():
            if "::" in k:
                group, value = k.split("::", 1)
                group = group.replace("has_", "").replace("_", " ")
                if group == "size":
                    continue
                value = value.replace("_", " ")
                group_pool.setdefault(group, set()).add(value)
                full_trait_pool.add(f"{group}: {value}")
    return {k: list(v) for k, v in group_pool.items()}, list(full_trait_pool)



def process_attributes(attr_dict):

    grouped = {}
    definite_traits = []
    all_present_traits = set() 
    for k, v in attr_dict.items():
        if not isinstance(v, dict) or v.get("present") != 1:
            continue
        if "::" not in k:
            continue
        group, value = k.split("::", 1)
        group = group.replace("has_", "").replace("_", " ")
        if group == "size":
            continue
        value = value.replace("_", " ")
        grouped.setdefault(group, []).append(value)
        trait_string = f"{group}: {value}"
        all_present_traits.add(trait_string)
        if v.get("certainty") == "definitely" or v.get("score", 0.0) >= 4.0:
            definite_traits.append(trait_string)
    return grouped, definite_traits, all_present_traits

def generate_mcq(group, true_values, pool, num_options=5):
    correct_val = random.choice(true_values)
    distractors = [v for v in pool if v not in true_values]
    num_distractors = min(num_options - 1, len(distractors))
    selected_distractors = random.sample(distractors, num_distractors)
    options = [correct_val] + selected_distractors
    random.shuffle(options)
    letters = ["A", "B", "C", "D", "E"]
    formatted_options = []
    correct_letter = ""
    for i, opt in enumerate(options):
        formatted_options.append(f"({letters[i]}) {opt}")
        if opt == correct_val:
            correct_letter = letters[i]
    q_text = f"What is the {group} of this bird? Options: " + ", ".join(formatted_options)
    a_text = f"({correct_letter}) {correct_val}"
    return q_text, a_text

def draw_mark(img, x, y, color_bgr, shape, size=10):
    x, y = int(x), int(y)
    if shape == "dot":
        cv2.circle(img, (x, y), size - 2, color_bgr, -1)
    elif shape == "cross":
        cv2.line(img, (x - size, y - size), (x + size, y + size), color_bgr, 3)
        cv2.line(img, (x + size, y - size), (x - size, y + size), color_bgr, 3)
    elif shape == "plus":
        cv2.line(img, (x, y - size), (x, y + size), color_bgr, 3)
        cv2.line(img, (x - size, y), (x + size, y), color_bgr, 3)
    elif shape == "star":
        # Combine a slightly thinner plus and cross
        cv2.line(img, (x, y - size), (x, y + size), color_bgr, 2)
        cv2.line(img, (x - size, y), (x + size, y), color_bgr, 2)
        offset = int(size * 0.75)
        cv2.line(img, (x - offset, y - offset), (x + offset, y + offset), color_bgr, 2)
        cv2.line(img, (x + offset, y - offset), (x - offset, y + offset), color_bgr, 2)


def is_far_enough(x, y, existing_points, min_dist=40):

    for (ex, ey) in existing_points:
        if math.hypot(x - ex, y - ey) < min_dist:
            return False
    return True



def make_qa(row, group_pool, full_trait_pool, category_pool):
    global _LOGGED_GENUS
    qa = []

    # 1. Nomenclature QA
    class_name = str(row.get("class_name", "")).split(".")[-1].replace("_", " ").strip()
    qa.append({"Q": "What is the common name of the bird?", "A": class_name})

    common_words = class_name.split()
    if len(common_words) >= 2:
        category = common_words[-1] 
        distractors = [c for c in category_pool if c != category]
        num_distractors = min(4, len(distractors))
        selected_distractors = random.sample(distractors, num_distractors)
        options = [category] + selected_distractors
        random.shuffle(options)
        letters = ["A", "B", "C", "D", "E"]
        formatted_options = []
        correct_letter = ""
        for i, opt in enumerate(options):
            formatted_options.append(f"({letters[i]}) {opt}")
            if opt == category:
                correct_letter = letters[i]
        qa.append({
            "Q": "What type of bird is this in the image? Options: " + ", ".join(formatted_options),
            "A": f"({correct_letter}) {category}"
        })
        qa.append({
            "Q": f"What kind of {category} is the bird in this image?", 
            "A": class_name
        })

    taxon_val = str(row.get("taxon", "Unknown")).strip()
    words = taxon_val.split()
    if len(words) >= 2:
        qa.append({"Q": "What is the scientific name of the bird?", "A": taxon_val})
        qa.append({"Q": "What is the genus of the bird?", "A": words[0]})
    elif len(words) == 1 and words[0] != "Unknown":
        qa.append({"Q": "What is the genus of the bird?", "A": words[0]})
        if not _LOGGED_GENUS:
            print(f"[*] Log: Found one-word taxon (Genus) -> '{words[0]}' for image ID {row.get('image_id')}")
            _LOGGED_GENUS = True


    attrs = parse_json_safely(row.get("attributes", "{}"))
    grouped, definite_traits, all_present_traits = process_attributes(attrs)

    for g, values in grouped.items():
        if len(values) == 0:
            continue
        g_pool = group_pool.get(g, [])
        q_text, a_text = generate_mcq(g, values, g_pool)
        qa.append({"Q": q_text, "A": a_text})

    for def_trait in definite_traits:
        distractors = [t for t in full_trait_pool if t not in all_present_traits]
        num_distractors = min(4, len(distractors))
        selected_distractors = random.sample(distractors, num_distractors)
        options = [def_trait] + selected_distractors
        random.shuffle(options)
        letters = ["A", "B", "C", "D", "E"]
        formatted_options = []
        correct_letter = ""
        for i, opt in enumerate(options):
            formatted_options.append(f"({letters[i]}) {opt}")
            if opt == def_trait:
                correct_letter = letters[i]
        q_text = "Which of the following is the most prominently visible physical trait of the bird in this image? Options: " + ", ".join(formatted_options)
        a_text = f"({correct_letter}) {def_trait}"
        qa.append({"Q": q_text, "A": a_text})




    parts = parse_json_safely(row.get("parts", "{}"))
    pairs = {
        "eye": ["left eye", "right eye"],
        "leg": ["left leg", "right leg"],
        "wing": ["left wing", "right wing"]
    }
    
    processed_parts = []
    for p_name, side_keys in pairs.items():
        left_vis = parts.get(side_keys[0], {}).get("visible", 0) == 1
        right_vis = parts.get(side_keys[1], {}).get("visible", 0) == 1
        if left_vis and right_vis:
            correct_ans = "Both"
        elif left_vis:
            correct_ans = "Left"
        elif right_vis:
            correct_ans = "Right"
        else:
            correct_ans = "None"
            
        options = ["Left", "Right", "Both", "None"]
        letters = ["A", "B", "C", "D"]
        formatted_options = [f"({letters[i]}) {opt}" for i, opt in enumerate(options)]
        correct_letter = letters[options.index(correct_ans)]
        qa.append({
            "Q": f"Which of the {p_name}s of the bird are visible in this image? Options: " + ", ".join(formatted_options), 
            "A": f"({correct_letter}) {correct_ans}"
        })
        processed_parts.extend(side_keys)



    visible = [k for k, v in parts.items() if v.get("visible") == 1]
    hidden = [k for k, v in parts.items() if v.get("visible") == 0]
    
    for i in range(10):
        if i == 0 and len(hidden) >= 3:
            choices = random.sample(hidden, 3)
            ans_formatted = "(D) None"
        elif i == 1 and len(visible) >= 3:
            choices = random.sample(visible, 3)
            ans_formatted = "(E) All three"
        elif len(visible) >= 1 and len(hidden) >= 2:
            v = random.sample(visible, 1)
            h = random.sample(hidden, 2)
            choices = v + h
            random.shuffle(choices)
            correct_idx = choices.index(v[0])
            letters = ["A", "B", "C"]
            ans_formatted = f"({letters[correct_idx]}) {v[0]}"
        else:
            if len(hidden) >= 3:
                choices = random.sample(hidden, 3)
                ans_formatted = "(D) None"
            elif len(visible) >= 3:
                choices = random.sample(visible, 3)
                ans_formatted = "(E) All three"
            else:
                continue

        letters = ["A", "B", "C"]
        options_str = ", ".join([f"({letters[j]}) {c}" for j, c in enumerate(choices)])
        options_str += ", (D) None, (E) All three"
        q_text = f"Which of the following parts of the bird are visible in this image? Options: {options_str}"
        qa.append({"Q": q_text, "A": ans_formatted})

    return qa


def categorize_qa_metadata(question):
    
    q_lower = question.lower()
    
    if any(x in q_lower for x in ["genus", "type of bird"]):
        return "Taxonomy & Naming"

    if any(x in q_lower for x in ["scientific name", "common name", "kind of"]):
        return "Taxonomy & Naming"

    if "most prominently visible" in q_lower:
        return "Relative Salience"

    if "are visible in this image" in q_lower:
        return "Anatomical Localization"

    if "color" in q_lower:
        return "Morphological Attributes"

    if any(x in q_lower for x in ["shape", "length", "pattern"]):
        return "Morphological Attributes"

    if "body part of this bird marked with" in q_lower:
        return "Anatomical Localization"

    return "Morphological Attributes"



if __name__ == "__main__":

    print(f"Loading dataset from: {IN_CSV}")
    df = pd.read_csv(IN_CSV)

    print("Building global vocabulary pools for MCQ distractors...")
    category_pool = set()
    for c_name in df["class_name"].dropna():
        clean_name = str(c_name).split(".")[-1].replace("_", " ").strip()
        words = clean_name.split()
        if len(words) >= 2:
            category_pool.add(words[-1])
    category_pool = list(category_pool)

    group_pool, full_trait_pool = build_attribute_pools(df)

    print("Generating textual QA pairs and Visual Grounding tasks...")
    final_rows = []
    
    paired_parts = {
        "left eye": "right eye", "right eye": "left eye",
        "left leg": "right leg", "right leg": "left leg",
        "left wing": "right wing", "right wing": "left wing"
    }

    for idx, row in df.iterrows():
        
        base_qa = make_qa(row, group_pool, full_trait_pool, category_pool)
        
        final_rows.append({
            "image_id": row["image_id"],
            "filename": row["filename"],
            "qa_pairs": json.dumps(base_qa)
        })
        
        img_filename = row["filename"]
        img_path = os.path.join(IMAGES_DIR, img_filename)
        
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            
            parts = parse_json_safely(row.get("parts", "{}"))
            visible_parts = {k: v for k, v in parts.items() if v.get("visible") == 1}
            hidden_parts = [k for k, v in parts.items() if v.get("visible") == 0]
            
            if visible_parts:

                target_part_name = random.choice(list(visible_parts.keys()))
                part_data = visible_parts[target_part_name]
                
                img_copy = img.copy()
                
                target_color_name = random.choice(list(COLORS.keys()))
                target_shape = random.choice(SHAPES)

                draw_mark(img_copy, part_data["x"], part_data["y"], COLORS[target_color_name], target_shape, size=12)
                occupied_points = [(part_data["x"], part_data["y"])]

                distractor_colors = []
                for color in COLORS.keys():
                    distractor_colors.extend([color, color])
                distractor_colors.remove(target_color_name)
                
                distractor_shapes = []
                for shape in SHAPES:
                    if shape == target_shape:
                        distractor_shapes.extend([shape, shape, shape])
                    else:
                        distractor_shapes.extend([shape, shape])
                        
                distractor_colors.remove(target_color_name)
                
                safe_shapes = [s for s in distractor_shapes if s != target_shape]
                safe_shape_choice = random.choice(safe_shapes)
                distractor_shapes.remove(safe_shape_choice)
                
                distractor_pairs = [(target_color_name, safe_shape_choice)]
                
                random.shuffle(distractor_colors)
                random.shuffle(distractor_shapes)
                distractor_pairs.extend(list(zip(distractor_colors, distractor_shapes)))
                
                
                for d_color, d_shape in distractor_pairs:
                    attempts = 0
                    while attempts < 100:
                        dx = random.randint(15, w - 15)
                        dy = random.randint(15, h - 15)
                        
                        if is_far_enough(dx, dy, occupied_points, min_dist=40):
                            occupied_points.append((dx, dy))
                            draw_mark(img_copy, dx, dy, COLORS[d_color], d_shape, size=10)
                            break
                        attempts += 1
                
                options_pool = set()
                
                if target_part_name in paired_parts:
                    options_pool.add(paired_parts[target_part_name])
                
                hidden_pool_shuffled = hidden_parts.copy()
                random.shuffle(hidden_pool_shuffled)
                for hp in hidden_pool_shuffled:
                    if len(options_pool) < 4:
                        options_pool.add(hp)
                
                visible_pool_shuffled = [p for p in visible_parts.keys() if p != target_part_name]
                random.shuffle(visible_pool_shuffled)
                for vp in visible_pool_shuffled:
                    if len(options_pool) < 4 and vp not in options_pool:
                        options_pool.add(vp)

                final_options = list(options_pool)
                final_options.append(target_part_name)
                random.shuffle(final_options)
                
                letters = ["A", "B", "C", "D", "E"]
                formatted_grounding_options = []
                correct_letter = ""
                for i, opt in enumerate(final_options):
                    formatted_grounding_options.append(f"({letters[i]}) {opt}")
                    if opt == target_part_name:
                        correct_letter = letters[i]
                
                options_str = ", ".join(formatted_grounding_options)
                
                file_uuid = uuid.uuid4().hex[:5]
                base_name = os.path.basename(img_filename).split(".")[0]
                new_filename = f"{base_name}_{file_uuid}.jpg"
                new_filepath = os.path.join(ANNOTATED_DIR, new_filename)
                cv2.imwrite(new_filepath, img_copy)
                
                final_rows.append({
                    "image_id": file_uuid,
                    "filename": f"Images_Annotated/{new_filename}",
                    "qa_pairs": json.dumps([{
                        "Q": f"What is the body part of this bird marked with the {target_color_name} {target_shape}? Options: {options_str}",
                        "A": f"({correct_letter}) {target_part_name}"
                    }])
                })

    df_out = pd.DataFrame(final_rows)

    df_out = pd.read_csv(file_path)

    df_out["qa_pairs"] = df_out["qa_pairs"].apply(json.loads)

    df_exploded = df_out.explode("qa_pairs").reset_index(drop=True)
    
    qa_normalized = pd.json_normalize(df_exploded["qa_pairs"])

    qa_normalized = qa_normalized.rename(columns={"Q": "question", "A": "answer"})

    final_df = pd.concat([df_exploded[["image_id", "filename"]], qa_normalized], axis=1)

    final_df["filename"] = final_df["filename"].apply(
        lambda x: str(x) if str(x).startswith("Images_Annotated/") else f"images/{x}"
    )

    final_df[["category"]] = final_df["question"].apply(lambda q: pd.Series(categorize_qa_metadata(q)))

    final_df.to_csv(OUT_CSV, index=False)
    
    print(f"\nSuccessfully saved enriched QA dataset to: {OUT_CSV}")
    print(f"Total rows generated (Original + Grounding): {len(final_df)}")