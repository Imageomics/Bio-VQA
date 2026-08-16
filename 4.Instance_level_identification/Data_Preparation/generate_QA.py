#!/usr/bin/env python3
"""
generate_QA.py

Build binary same-individual verification pairs from a COCO-format
re-identification dataset, at a controlled appearance-difficulty level.

Difficulty conditions
---------------------
hard    different-individual pairs drawn from the NEAREST DINOv2 neighbours
        same-individual pairs drawn from the FURTHEST within-identity neighbours
        -> appearance similarity is anti-correlated with the label

easy    different-individual pairs drawn from the FURTHEST DINOv2 neighbours
        same-individual pairs drawn from the CLOSEST within-identity neighbours
        -> appearance similarity is correlated with the label

random  both classes sampled uniformly at random
        -> appearance similarity is uncorrelated with the label

Viewpoint is held constant within every pair in all three conditions, so
difficulty is the only manipulated factor.

Safety
------
Nothing is written to disk until pair selection, image validation and capacity
checks have all succeeded. Images are written atomically (tmp + os.replace).
An existing non-empty output directory is never silently clobbered.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
import shutil
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import faiss

SCRIPT_VERSION = "2.0"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
QUESTION = (
    "This image shows two images placed side by side. "
    "Is the main subject in the left image the very same individual animal as the main subject in the right image?"
)
SEARCH_CHUNK = 4096

# accepted field names for the individual label and the viewpoint, in priority
# order; covers both the COCO re-ID exports and the flat zebra metadata
IDENTITY_KEYS = ("name_uuid", "name", "annot_name_uuid", "individual_id", "identity")
VIEWPOINT_KEYS = ("viewpoint", "annot_viewpoint", "view")
IMAGE_KEYS = ("image", "image_fname", "file_name", "filename", "image_path", "path")


# --------------------------------------------------------------------------- #
# utilities
# --------------------------------------------------------------------------- #

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fail(msg: str) -> "None":
    # stdout, not stderr, so the reason appears inline in the SLURM .out file
    print(f"\nERROR: {msg}\n", flush=True)
    sys.exit(1)


def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def index_image_files(img_dir: str) -> dict:
    """Map image stem -> path, walking subdirectories, tolerant of any extension."""
    mapping = {}
    collisions = 0
    for root, dirs, files in os.walk(img_dir):
        # never index our own generated output or the embedding store
        dirs[:] = sorted(d for d in dirs
                         if not d.startswith("instance_identification_")
                         and d != "embeddings_dinov2")
        for fname in sorted(files):
            if not fname.lower().endswith(IMG_EXTS):
                continue
            key = stem(fname)
            if key in mapping:
                collisions += 1
                continue
            mapping[key] = os.path.join(root, fname)
    if collisions:
        log(f"  warning: {collisions} image stem collisions ignored")
    return mapping


def load_annotations_coco(annotation_file: str):
    """COCO layout: instances_train{year}.json with images[] and annotations[]."""
    with open(annotation_file, "r") as f:
        data = json.load(f)

    if "images" not in data or "annotations" not in data:
        fail(f"{annotation_file} does not look like COCO "
             f"(missing 'images' or 'annotations')")

    id_to_stem = {img["id"]: stem(img["file_name"]) for img in data["images"]}

    identity, viewpoint = {}, {}
    for ann in data["annotations"]:
        key = id_to_stem.get(ann["image_id"])
        if key is None:
            continue
        for k in IDENTITY_KEYS:
            if ann.get(k) is not None:
                identity[key] = ann[k]
                break
        for k in VIEWPOINT_KEYS:
            if ann.get(k) is not None:
                viewpoint[key] = ann[k]
                break

    return identity, viewpoint


def load_annotations_flat(metadata_file: str):
    """
    Flat layout: one record per image. Accepts CSV/TSV (e.g. the Grevy's zebra
    metadata.csv) or a JSON list. Column/field names are normalised to
    lowercase-underscore before matching, so 'name uuid' and 'name_uuid' both work.
    """
    ext = os.path.splitext(metadata_file)[1].lower()

    if ext in (".csv", ".tsv", ".tab", ".txt"):
        sep = "\t" if ext in (".tsv", ".tab") else ","
        try:
            df = pd.read_csv(metadata_file, sep=sep)
        except Exception as exc:
            fail(f"could not read {metadata_file}: {exc}")
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        records = df.to_dict("records")
    else:
        with open(metadata_file, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for k in ("images", "records", "data", "annotations"):
                if isinstance(data.get(k), list):
                    data = data[k]
                    break
        if not isinstance(data, list):
            fail(f"unrecognised flat metadata structure in {metadata_file}")
        records = [{str(k).strip().lower().replace(" ", "_"): v
                    for k, v in rec.items()}
                   for rec in data if isinstance(rec, dict)]

    def clean(value):
        if value is None:
            return None
        if isinstance(value, float) and value != value:   # NaN
            return None
        text = str(value).strip()
        return text if text and text.lower() not in ("nan", "none", "null") else None

    identity, viewpoint = {}, {}
    for rec in records:
        img = next((rec[k] for k in IMAGE_KEYS if clean(rec.get(k))), None)
        if img is None:
            continue
        key = stem(str(img))
        ident = next((clean(rec[k]) for k in IDENTITY_KEYS if clean(rec.get(k))), None)
        if ident is not None:
            identity[key] = ident
        view = next((clean(rec[k]) for k in VIEWPOINT_KEYS if clean(rec.get(k))), None)
        if view is not None:
            viewpoint[key] = view

    return identity, viewpoint


def flat_metadata_candidates(base: str):
    """Filenames tried, in order, when auto-detecting a flat-layout dataset."""
    return [os.path.join(base, n) for n in
            ("metadata.csv", "metadata.json", "metadata.tsv")]


def resolve_paths(args):
    """Work out layout, image dir, annotation file and embedding dir."""
    base = args.base_dir

    def under(p):
        return p if os.path.isabs(p) else os.path.join(base, p)

    emb_dir = under(args.emb_dir)
    layout = args.layout
    ann_dir = os.path.join(base, "annotations")

    if layout == "auto":
        if os.path.isdir(ann_dir):
            layout = "coco"
        elif args.metadata and os.path.isfile(under(args.metadata)):
            layout = "flat"
        elif any(os.path.isfile(c) for c in flat_metadata_candidates(base)):
            layout = "flat"
        else:
            tried = "\n         ".join([f"{ann_dir}/  (coco)"] +
                                       [f"{c}  (flat)" for c in
                                        flat_metadata_candidates(base)])
            fail(f"cannot determine layout for {base}\n"
                 f"       tried: {tried}\n"
                 f"       pass --layout flat --annotations <file> to override.")

    if layout == "coco":
        year = args.year
        if args.annotations:
            ann_file = under(args.annotations)
        else:
            if not year:
                found = sorted(f for f in os.listdir(ann_dir)
                               if f.startswith("instances_train") and f.endswith(".json"))
                if len(found) != 1:
                    fail(f"could not infer --year: found {len(found)} "
                         f"instances_train*.json files in {ann_dir}")
                year = found[0][len("instances_train"):-len(".json")]
                log(f"  inferred year: {year}")
            ann_file = os.path.join(ann_dir, f"instances_train{year}.json")

        if args.image_dir:
            img_dir = under(args.image_dir)
        else:
            img_dir = os.path.join(base, "images", f"train{year}") if year else ""
            if not os.path.isdir(img_dir):
                img_dir = os.path.join(base, "images")
    else:
        if args.annotations:
            ann_file = under(args.annotations)
        elif args.metadata:
            ann_file = under(args.metadata)
        else:
            ann_file = next((c for c in flat_metadata_candidates(base)
                             if os.path.isfile(c)), flat_metadata_candidates(base)[0])
        if args.image_dir:
            img_dir = under(args.image_dir)
        else:
            img_dir = next((d for d in (os.path.join(base, "Images"),
                                        os.path.join(base, "images"))
                            if os.path.isdir(d)), base)

    return layout, img_dir, ann_file, emb_dir


def load_embeddings(emb_dir: str, img_map: dict, identity: dict, viewpoint: dict):
    """Load .npy embeddings, keeping only entries with an image and an identity."""
    vecs, keys, skipped = [], [], defaultdict(int)
    dim = None

    for fname in sorted(os.listdir(emb_dir)):
        if not fname.endswith(".npy"):
            continue
        key = stem(fname)

        if key not in img_map:
            skipped["no_image_file"] += 1
            continue
        if key not in identity:
            skipped["no_identity"] += 1
            continue

        try:
            emb = np.load(os.path.join(emb_dir, fname))
        except Exception:
            skipped["unreadable_npy"] += 1
            continue

        emb = np.asarray(emb, dtype="float32").reshape(-1)
        if dim is None:
            dim = emb.shape[0]
        if emb.shape[0] != dim:
            skipped["dim_mismatch"] += 1
            continue
        if not np.isfinite(emb).all() or float(np.linalg.norm(emb)) == 0.0:
            skipped["degenerate_vector"] += 1
            continue

        vecs.append(emb)
        keys.append(key)

    if not vecs:
        fail(f"no usable embeddings found in {emb_dir}")

    for reason, n in sorted(skipped.items()):
        log(f"  skipped {n} embeddings ({reason})")

    embeddings = np.ascontiguousarray(np.vstack(vecs), dtype="float32")
    identities = [identity[k] for k in keys]
    viewpoints = [viewpoint.get(k) for k in keys]
    return embeddings, keys, identities, viewpoints


# --------------------------------------------------------------------------- #
# pair selection
# --------------------------------------------------------------------------- #

def take_greedy(candidates, target, max_uses, used_pairs, use_count):
    """Consume a pre-ordered candidate list under dedup + per-image use caps."""
    chosen = []
    for i, j, sim in candidates:
        key = (i, j) if i < j else (j, i)
        if key in used_pairs:
            continue
        if use_count[i] >= max_uses or use_count[j] >= max_uses:
            continue
        used_pairs.add(key)
        use_count[i] += 1
        use_count[j] += 1
        chosen.append((i, j, sim))
        if len(chosen) >= target:
            break
    return chosen


def take_round_robin(per_group, target, max_uses, used_pairs, use_count, rng):
    """Cycle across groups so no single identity or viewpoint dominates."""
    # key=str is required: viewpoint keys may mix None with strings when the
    # annotations have missing viewpoints, and bare sorted() raises TypeError.
    keys = sorted(per_group.keys(), key=str)
    rng.shuffle(keys)
    ptr = {k: 0 for k in keys}
    chosen = []

    progressed = True
    while len(chosen) < target and progressed:
        progressed = False
        for k in keys:
            lst = per_group[k]
            p = ptr[k]
            while p < len(lst):
                i, j, sim = lst[p]
                p += 1
                pair_key = (i, j) if i < j else (j, i)
                if pair_key in used_pairs:
                    continue
                if use_count[i] >= max_uses or use_count[j] >= max_uses:
                    continue
                used_pairs.add(pair_key)
                use_count[i] += 1
                use_count[j] += 1
                chosen.append((i, j, sim))
                progressed = True
                break
            ptr[k] = p
            if len(chosen) >= target:
                break
    return chosen


def select_different_pairs(difficulty, embeddings, identities, viewpoints,
                           target, knn, max_uses, used_pairs, use_count, rng,
                           log_groups=True):
    """
    One candidate partner per anchor.

    The search runs WITHIN each viewpoint group rather than filtering matches
    after retrieval. This matters for 'easy': viewpoint is the dominant axis of
    variation in DINOv2 space, so an anchor's most dissimilar neighbours are
    overwhelmingly the opposite viewpoint. Post-filtering would reject almost
    every candidate; searching inside the group cannot.

    Candidates are then round-robined across viewpoint groups so that all three
    difficulty conditions end up with comparable viewpoint distributions.
    """
    vp_groups = defaultdict(list)
    for i, vp in enumerate(viewpoints):
        vp_groups[vp].append(i)

    per_group = {}
    for vp in sorted(vp_groups.keys(), key=str):
        members = vp_groups[vp]
        if len(members) < 2:
            continue

        cands = []

        if difficulty in ("hard", "easy"):
            # hard -> nearest neighbours; easy -> furthest, obtained by querying
            # with negated unit-norm vectors so that maximum inner product
            # corresponds to minimum cosine similarity.
            sign = 1.0 if difficulty == "hard" else -1.0
            sub = np.ascontiguousarray(embeddings[members], dtype="float32")
            sub_index = faiss.IndexFlatIP(sub.shape[1])
            sub_index.add(sub)
            k = min(knn, len(members))

            for start in range(0, len(members), SEARCH_CHUNK):
                block = sub[start:start + SEARCH_CHUNK]
                query = np.ascontiguousarray(block * sign, dtype="float32")
                D, I = sub_index.search(query, k)
                for row in range(I.shape[0]):
                    i = members[start + row]
                    for rank in range(I.shape[1]):
                        loc = int(I[row, rank])
                        if loc < 0:
                            continue
                        j = members[loc]
                        if j == i or identities[i] == identities[j]:
                            continue
                        cands.append((i, j, float(sign * D[row, rank])))
                        break

            # hardest first for 'hard' (most similar), easiest first for 'easy'
            cands.sort(key=lambda t: t[2], reverse=(difficulty == "hard"))

        else:  # random
            order = list(members)
            rng.shuffle(order)
            for i in order:
                for _ in range(64):
                    j = members[rng.randrange(len(members))]
                    if j != i and identities[i] != identities[j]:
                        cands.append((i, j, float(np.dot(embeddings[i], embeddings[j]))))
                        break
            rng.shuffle(cands)

        if cands:
            per_group[vp] = cands
            if log_groups:
                log(f"  viewpoint {str(vp):>12}: {len(members)} images, "
                    f"{len(cands)} candidate pairs")

    if not per_group:
        return []

    return take_round_robin(per_group, target, max_uses, used_pairs, use_count, rng)


def select_same_pairs(difficulty, embeddings, identities, viewpoints,
                      target, max_uses, max_group, max_per_identity,
                      used_pairs, use_count, rng):
    """
    Enumerate EVERY valid within-identity, within-viewpoint pair, ordered by
    difficulty, then round-robin across identities.

    Enumerating all pairs rather than one best partner per anchor matters: with
    a best-partner-only pool, mutual best partners collapse under dedup and an
    identity with n images yields roughly n/2 pairs instead of the C(n,2) that
    exist. On datasets with small identities that starves the same-individual
    class long before the images run out.
    """
    groups = defaultdict(list)
    for idx, ident in enumerate(identities):
        groups[ident].append(idx)

    per_identity = {}
    for ident in sorted(groups.keys(), key=str):
        idxs = groups[ident]
        if len(idxs) < 2:
            continue
        if len(idxs) > max_group:
            idxs = sorted(rng.sample(idxs, max_group))

        # split the identity by viewpoint, then take all pairs inside each half
        by_vp = defaultdict(list)
        for i in idxs:
            by_vp[viewpoints[i]].append(i)

        cands = []
        for vp in sorted(by_vp.keys(), key=str):
            members = by_vp[vp]
            if len(members) < 2:
                continue
            sub = embeddings[members]
            sims = sub @ sub.T
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    cands.append((members[a], members[b], float(sims[a, b])))

        if not cands:
            continue

        if difficulty == "hard":
            cands.sort(key=lambda t: t[2])                 # furthest first
        elif difficulty == "easy":
            cands.sort(key=lambda t: t[2], reverse=True)   # closest first
        else:
            rng.shuffle(cands)

        per_identity[ident] = cands[:max_per_identity]

    if not per_identity:
        return []

    return take_round_robin(per_identity, target, max_uses, used_pairs, use_count, rng)


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

def run_tasks(fn, tasks, workers, total):
    """Map fn over tasks, in parallel when workers > 1, logging progress."""
    results = []
    if workers <= 1:
        for n, task in enumerate(tasks, 1):
            results.append(fn(task))
            if n % 200 == 0:
                log(f"  {n}/{total}")
    else:
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(workers) as pool:
            for n, r in enumerate(pool.imap_unordered(fn, tasks, chunksize=4), 1):
                results.append(r)
                if n % 200 == 0:
                    log(f"  {n}/{total}")
    return results


def validate_one(path):
    try:
        with Image.open(path) as im:
            im.load()
        return path, True
    except Exception:
        return path, False


def open_rgb(path, max_height=0):
    with Image.open(path) as im:
        if max_height and im.height > max_height:
            # lets the JPEG decoder work at a reduced DCT scale: much faster
            im.draft("RGB", (im.width, max_height))
        return im.convert("RGB")


def concat_and_save(img1_path, img2_path, save_path, max_height=0):
    img1 = open_rgb(img1_path, max_height)
    img2 = open_rgb(img2_path, max_height)

    target_h = min(img1.height, img2.height)
    if max_height:
        target_h = min(target_h, max_height)

    def resize_keep_aspect(img, h):
        w = max(1, int(round(img.width * (h / img.height))))
        return img.resize((w, h), Image.BILINEAR)

    if img1.height != target_h:
        img1 = resize_keep_aspect(img1, target_h)
    if img2.height != target_h:
        img2 = resize_keep_aspect(img2, target_h)

    canvas = Image.new("RGB", (img1.width + img2.width, target_h))
    canvas.paste(img1, (0, 0))
    canvas.paste(img2, (img1.width, 0))

    tmp_path = save_path + ".tmp"
    canvas.save(tmp_path, format="PNG")
    with Image.open(tmp_path) as check:   # confirm the written file decodes
        check.load()
    os.replace(tmp_path, save_path)


def render_one(task):
    img1, img2, out_path, max_height = task
    try:
        concat_and_save(img1, img2, out_path, max_height)
        return out_path, True, ""
    except Exception as exc:
        try:
            if os.path.exists(out_path + ".tmp"):
                os.remove(out_path + ".tmp")
        except OSError:
            pass
        return out_path, False, repr(exc)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_dir", type=str, required=True)
    p.add_argument("--difficulty", type=str, required=True,
                   choices=["easy", "hard", "random"])
    p.add_argument("--num", type=int, default=500,
                   help="pairs PER CLASS; total = 2 * num")
    p.add_argument("--layout", type=str, default="auto",
                   choices=["auto", "coco", "flat"],
                   help="coco: annotations/instances_train{year}.json + images/train{year}; "
                        "flat: a JSON list of per-image records + Images/")
    p.add_argument("--year", type=str, default=None,
                   help="coco layout only; inferred from annotations/ if omitted")
    p.add_argument("--metadata", type=str, default=None,
                   help="flat layout only; defaults to metadata.csv / metadata.json "
                        "in --base_dir. Relative paths resolve against --base_dir")
    p.add_argument("--annotations", type=str, default=None,
                   help="explicit annotation/metadata file, overrides --year/--metadata")
    p.add_argument("--image_dir", type=str, default=None,
                   help="explicit image directory, overrides layout defaults")
    p.add_argument("--emb_dir", type=str, default="embeddings_dinov2")
    p.add_argument("--question", type=str, default=QUESTION,
                   help="question text written into every qa_pair")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=0,
                   help="parallel image workers; 0 = use SLURM_CPUS_PER_TASK "
                        "(capped at 12), 1 = serial")
    p.add_argument("--max_height", type=int, default=0,
                   help="cap the concatenated image height in pixels; 0 = keep "
                        "full resolution. Large values dominate runtime")
    p.add_argument("--knn", type=int, default=50,
                   help="neighbours retrieved per anchor, within its viewpoint group")
    p.add_argument("--max_uses_per_image", type=int, default=2,
                   help="cap on how many pairs a single image may appear in")
    p.add_argument("--max_group_size", type=int, default=400,
                   help="cap on images per identity for the within-identity sim matrix")
    p.add_argument("--max_pairs_per_identity", type=int, default=64,
                   help="cap on candidate pairs kept per identity")
    p.add_argument("--no_auto_relax", dest="auto_relax", action="store_false",
                   help="fail instead of raising --max_uses_per_image when short")
    p.add_argument("--ignore_viewpoint", action="store_true",
                   help="do not require matching viewpoint within a pair")
    p.add_argument("--reserve_frac", type=float, default=0.15,
                   help="extra candidates selected to absorb unreadable images")
    p.add_argument("--allow_short", action="store_true",
                   help="proceed even if fewer than --num pairs are available")
    p.add_argument("--overwrite", action="store_true",
                   help="delete an existing output directory before writing")
    p.add_argument("--dry_run", action="store_true",
                   help="select and validate everything, write nothing")
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    base = args.base_dir
    if not os.path.isdir(base):
        fail(f"base_dir does not exist: {base}")

    layout, img_dir, ann_file, emb_dir = resolve_paths(args)

    for path, label in ((img_dir, "image dir"), (emb_dir, "embedding dir")):
        if not os.path.isdir(path):
            fail(f"{label} does not exist: {path}")
    if not os.path.isfile(ann_file):
        fail(f"annotation file does not exist: {ann_file}")

    log(f"layout={layout}")
    log(f"  images      : {img_dir}")
    log(f"  annotations : {ann_file}")
    log(f"  embeddings  : {emb_dir}")

    out_root = os.path.join(base, f"instance_identification_{args.num * 2}_{args.difficulty}")
    out_img_dir = os.path.join(out_root, "concatenated_images")

    if os.path.exists(out_root) and os.listdir(out_root):
        if not args.overwrite:
            fail(f"output already exists and is non-empty: {out_root}\n"
                 f"       pass --overwrite to replace it.")
        if not args.dry_run:
            log(f"removing existing output: {out_root}")
            shutil.rmtree(out_root)

    workers = args.workers
    if workers <= 0:
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 0) or (os.cpu_count() or 1))
        workers = max(1, min(workers, 12))

    log(f"difficulty={args.difficulty}  num={args.num} per class  "
        f"seed={args.seed}  workers={workers}")

    log("Indexing image files...")
    img_map = index_image_files(img_dir)
    log(f"  {len(img_map)} images on disk")

    log("Loading annotations...")
    if layout == "coco":
        identity_map, viewpoint_map = load_annotations_coco(ann_file)
    else:
        identity_map, viewpoint_map = load_annotations_flat(ann_file)
    if not identity_map:
        fail(f"no identity labels parsed from {ann_file}; "
             f"expected one of {IDENTITY_KEYS}")
    log(f"  {len(identity_map)} identity labels, {len(viewpoint_map)} viewpoint labels")

    log("Loading embeddings...")
    embeddings, keys, identities, viewpoints = load_embeddings(
        emb_dir, img_map, identity_map, viewpoint_map)
    if args.ignore_viewpoint:
        viewpoints = [None] * len(viewpoints)
    elif not any(v is not None for v in viewpoints):
        log("  warning: no viewpoint labels present; the viewpoint constraint is inert")
    log(f"  {embeddings.shape[0]} usable embeddings, dim {embeddings.shape[1]}")

    groups = defaultdict(list)
    for i, ident in enumerate(identities):
        groups[ident].append(i)
    n_ident = len(groups)
    n_multi = sum(1 for v in groups.values() if len(v) >= 2)
    log(f"  {n_ident} identities, {n_multi} with >= 2 images")
    if n_multi == 0:
        fail("no identity has two or more images; same-individual pairs are impossible")

    log("Normalizing embeddings...")
    faiss.normalize_L2(embeddings)

    reserve = max(20, int(round(args.num * args.reserve_frac)))
    want = args.num + reserve
    UNLIMITED = 10 ** 9

    caps = [args.max_uses_per_image]
    if args.auto_relax:
        caps += [c for c in (4, 8, UNLIMITED) if c > args.max_uses_per_image]

    same_pairs, diff_pairs, used_cap = [], [], args.max_uses_per_image
    for attempt, cap in enumerate(caps):
        used_pairs, use_count = set(), defaultdict(int)
        attempt_rng = random.Random(args.seed)
        used_cap = cap

        if attempt:
            shown = "unlimited" if cap >= UNLIMITED else cap
            log(f"Target not met; retrying with per-image use cap = {shown}...")

        # The same-individual class is the scarce one, bounded by how many
        # images each identity actually has. It gets first claim on the image
        # budget; different-individual pairs are effectively unlimited.
        log(f"Selecting SAME-individual pairs ({args.difficulty})...")
        same_pairs = select_same_pairs(
            args.difficulty, embeddings, identities, viewpoints,
            want, cap, args.max_group_size, args.max_pairs_per_identity,
            used_pairs, use_count, attempt_rng)
        log(f"  {len(same_pairs)} candidates")

        log(f"Selecting DIFFERENT-individual pairs ({args.difficulty})...")
        diff_pairs = select_different_pairs(
            args.difficulty, embeddings, identities, viewpoints,
            want, args.knn, cap, used_pairs, use_count, attempt_rng,
            log_groups=(attempt == 0))
        log(f"  {len(diff_pairs)} candidates")

        if len(same_pairs) >= args.num and len(diff_pairs) >= args.num:
            break

    if used_cap != args.max_uses_per_image:
        shown = "unlimited" if used_cap >= UNLIMITED else used_cap
        log(f"  note: per-image use cap relaxed {args.max_uses_per_image} -> {shown}; "
            f"some images appear in more than {args.max_uses_per_image} pairs")

    log(f"Validating source images ({workers} worker(s))...")
    needed = sorted({img_map[keys[i]] for pair in (diff_pairs + same_pairs)
                     for i in pair[:2]})
    good = {p for p, ok in run_tasks(validate_one, needed, workers, len(needed)) if ok}
    n_bad = len(needed) - len(good)
    if n_bad:
        log(f"  warning: {n_bad} unreadable images excluded")

    def usable(pairs):
        return [p for p in pairs
                if img_map[keys[p[0]]] in good and img_map[keys[p[1]]] in good]

    diff_pairs = usable(diff_pairs)[:args.num]
    same_pairs = usable(same_pairs)[:args.num]

    if (len(diff_pairs) < args.num or len(same_pairs) < args.num) and not args.allow_short:
        fail(
            f"insufficient pairs: {len(diff_pairs)}/{args.num} different, "
            f"{len(same_pairs)}/{args.num} same. Nothing was written.\n"
            f"       see the per-viewpoint candidate counts logged above.\n"
            f"       the per-image use cap was already relaxed to "
            f"{'unlimited' if used_cap >= UNLIMITED else used_cap} without success, "
            f"so this dataset structurally cannot supply {args.num} pairs per class.\n"
            f"       remedies: lower --num, raise --max_pairs_per_identity "
            f"(currently {args.max_pairs_per_identity}), "
            f"or pass --ignore_viewpoint / --allow_short."
        )

    log(f"Final selection: {len(diff_pairs)} different + {len(same_pairs)} same "
        f"= {len(diff_pairs) + len(same_pairs)} pairs")

    if args.dry_run:
        log("dry run complete; no files written.")
        return

    os.makedirs(out_img_dir, exist_ok=True)

    plan = [(p, "Yes") for p in same_pairs] + [(p, "No") for p in diff_pairs]

    tasks = []
    for (i, j, _sim), _answer in plan:
        out_name = f"{keys[i]}_{keys[j]}.png"
        tasks.append((img_map[keys[i]], img_map[keys[j]],
                      os.path.join(out_img_dir, out_name), args.max_height))

    log(f"Rendering {len(tasks)} concatenated images ({workers} worker(s))...")
    outcome = {path: (ok, err) for path, ok, err
               in run_tasks(render_one, tasks, workers, len(tasks))}

    # rows are rebuilt in plan order, so parallel completion order is irrelevant
    rows, meta_rows, failures = [], [], []
    for ((i, j, sim), answer), task in zip(plan, tasks):
        out_path = task[2]
        out_name = os.path.basename(out_path)
        ok, err = outcome.get(out_path, (False, "not attempted"))
        if not ok:
            failures.append((out_name, err))
            continue

        rows.append({
            "image_path": f"concatenated_images/{out_name}",
            "qa_pair": json.dumps({"Q": args.question, "A": answer}),
        })
        meta_rows.append({
            "image_path": f"concatenated_images/{out_name}",
            "label": answer,
            "difficulty": args.difficulty,
            "cosine_similarity": round(sim, 6),
            "image_left": keys[i],
            "image_right": keys[j],
            "identity_left": identities[i],
            "identity_right": identities[j],
            "viewpoint": viewpoints[i],
        })

    if failures:
        log(f"  warning: {len(failures)} pairs failed to render")

    rng.shuffle(rows)
    order = {r["image_path"]: idx for idx, r in enumerate(rows)}
    meta_rows.sort(key=lambda r: order[r["image_path"]])

    csv_path = os.path.join(out_root, "QA.csv")
    meta_path = os.path.join(out_root, "pairs_metadata.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    pd.DataFrame(meta_rows).to_csv(meta_path, index=False)

    sims_same = [r["cosine_similarity"] for r in meta_rows if r["label"] == "Yes"]
    sims_diff = [r["cosine_similarity"] for r in meta_rows if r["label"] == "No"]
    mean_same = float(np.mean(sims_same)) if sims_same else float("nan")
    mean_diff = float(np.mean(sims_diff)) if sims_diff else float("nan")

    with open(os.path.join(out_root, "metadata.json"), "w") as f:
        json.dump({
            "script_version": SCRIPT_VERSION,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "args": vars(args),
            "layout": layout,
            "question": args.question,
            "paths": {"image_dir": img_dir, "annotation_file": ann_file,
                      "embedding_dir": emb_dir},
            "counts": {
                "embeddings_used": int(embeddings.shape[0]),
                "identities": n_ident,
                "identities_with_multiple_images": n_multi,
                "pairs_same": len(sims_same),
                "pairs_different": len(sims_diff),
                "render_failures": len(failures),
                "workers": workers,
                "max_uses_per_image_applied": (None if used_cap >= UNLIMITED
                                               else used_cap),
            },
            "viewpoint_distribution": {
                "same": {str(k): v for k, v in sorted(
                    pd.Series([r["viewpoint"] for r in meta_rows
                               if r["label"] == "Yes"]).astype(str)
                    .value_counts().items())},
                "different": {str(k): v for k, v in sorted(
                    pd.Series([r["viewpoint"] for r in meta_rows
                               if r["label"] == "No"]).astype(str)
                    .value_counts().items())},
            },
            "similarity": {
                "mean_same_individual": mean_same,
                "mean_different_individual": mean_diff,
                "margin_same_minus_different": mean_same - mean_diff,
            },
            "render_failures": failures[:50],
        }, f, indent=2)

    log(f"Saved CSV: {csv_path}")
    log(f"Saved metadata: {meta_path}")
    log(f"mean cosine similarity  same={mean_same:.4f}  different={mean_diff:.4f}  "
        f"margin={mean_same - mean_diff:+.4f}")
    log("Done!")


if __name__ == "__main__":
    main()