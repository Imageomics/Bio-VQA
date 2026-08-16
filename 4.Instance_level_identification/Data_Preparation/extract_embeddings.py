#!/usr/bin/env python3
"""
extract_embeddings.py

Extract DINOv2 embeddings for a re-identification dataset, in either layout:

  coco   <root>/images/train{year}/*.jpg          (Beluga, Hyena, Leopard, Whaleshark)
  flat   <root>/metadata.{json,csv} listing images (Grevy's zebra)

One .npy per image, named by image stem, written to --output_dir. That is the
exact contract generate_QA.py expects: it maps image stem -> embedding file.

Sharding
--------
Work is split by STRIDE, not by contiguous index ranges:

    my_images = all_images[shard::num_shards]

Every image is covered as long as the array size equals --num_shards, whatever
the dataset size. Contiguous START/END chunking silently drops the tail when the
array is too short for the dataset.

Resumable: existing .npy files are skipped, so a requeued job picks up where it
stopped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
IMAGE_KEYS = ("image", "image_fname", "file_name", "filename", "image_path", "path")


def log(msg):
    print(msg, flush=True)


def fail(msg):
    print(f"\nERROR: {msg}\n", flush=True)
    sys.exit(1)


def stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def is_image_file(name):
    return name.lower().endswith(IMG_EXTS) and not name.startswith(".")


# --------------------------------------------------------------------------- #
# listing images
# --------------------------------------------------------------------------- #

def infer_year(root):
    img_root = os.path.join(root, "images")
    if not os.path.isdir(img_root):
        return None
    years = sorted(d[len("train"):] for d in os.listdir(img_root)
                   if d.startswith("train") and os.path.isdir(os.path.join(img_root, d)))
    if len(years) == 1:
        return years[0]
    if len(years) > 1:
        fail(f"multiple train dirs under {img_root}: {years}; pass --year")
    return None


def list_images_coco(root, year, image_dir):
    if image_dir:
        img_dir = image_dir
    else:
        if year is None:
            year = infer_year(root)
            if year:
                log(f"  inferred year: {year}")
        img_dir = os.path.join(root, "images", f"train{year}") if year else ""
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(root, "images")

    if not os.path.isdir(img_dir):
        fail(f"image dir does not exist: {img_dir}")

    paths = []
    for r, dirs, files in os.walk(img_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for f in sorted(files):
            if is_image_file(f):
                paths.append(os.path.join(r, f))
    return img_dir, paths


def read_flat_records(meta_path):
    ext = os.path.splitext(meta_path)[1].lower()
    if ext in (".csv", ".tsv", ".tab"):
        import pandas as pd
        sep = "\t" if ext in (".tsv", ".tab") else ","
        df = pd.read_csv(meta_path, sep=sep)
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        return df.to_dict("records")

    with open(meta_path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for k in ("images", "records", "data", "annotations"):
            if isinstance(data.get(k), list):
                data = data[k]
                break
    if not isinstance(data, list):
        fail(f"unrecognised flat metadata structure in {meta_path}")
    return [{str(k).strip().lower().replace(" ", "_"): v for k, v in rec.items()}
            for rec in data if isinstance(rec, dict)]


def list_images_flat(root, meta_path):
    if not os.path.isfile(meta_path):
        fail(f"metadata file does not exist: {meta_path}")

    records = read_flat_records(meta_path)
    paths, missing = [], 0

    for rec in records:
        raw = next((rec[k] for k in IMAGE_KEYS
                    if rec.get(k) is not None and str(rec[k]).strip()
                    and str(rec[k]).strip().lower() not in ("nan", "none", "null")), None)
        if raw is None:
            continue
        raw = str(raw).strip()

        # metadata may hold absolute paths, or paths relative to the root
        cand = raw if os.path.isabs(raw) else os.path.join(root, raw)
        if not os.path.exists(cand):
            for sub in ("Images", "images", ""):
                alt = os.path.join(root, sub, os.path.basename(raw))
                if os.path.exists(alt):
                    cand = alt
                    break
        if not os.path.exists(cand):
            missing += 1
            continue
        paths.append(cand)

    if missing:
        log(f"  warning: {missing} records reference images not found on disk")
    return meta_path, paths


def find_flat_metadata(root, explicit):
    if explicit:
        return explicit if os.path.isabs(explicit) else os.path.join(root, explicit)
    for name in ("metadata.json", "metadata.csv", "metadata.tsv"):
        p = os.path.join(root, name)
        if os.path.isfile(p):
            return p
    return os.path.join(root, "metadata.json")


def resolve_layout(root, layout, metadata):
    if layout != "auto":
        return layout
    if os.path.isdir(os.path.join(root, "annotations")) or \
       os.path.isdir(os.path.join(root, "images")):
        return "coco"
    if os.path.isfile(find_flat_metadata(root, metadata)):
        return "flat"
    fail(f"cannot determine layout for {root}; pass --layout coco|flat")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def get_transform(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class ImageList(Dataset):
    def __init__(self, paths, tf):
        self.paths = paths
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        try:
            with Image.open(path) as im:
                x = self.tf(im.convert("RGB"))
            return x, path
        except Exception as exc:
            print(f"  skip {os.path.basename(path)}: {exc}", flush=True)
            return None


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs = torch.stack([b[0] for b in batch])
    return xs, [b[1] for b in batch]


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

def load_model(model_name, device):
    log(f"Loading model: {model_name}")
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    return model.eval().to(device)


def embed_batch(model, x):
    with torch.no_grad():
        out = model.forward_features(x)
    if isinstance(out, dict):
        out = out.get("x_norm_clstoken", next(iter(out.values())))
    return out.detach().float().cpu().numpy()


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--output_dir", default=None,
                   help="default: <dataset_root>/embeddings_dinov2")
    p.add_argument("--layout", default="auto", choices=["auto", "coco", "flat"])
    p.add_argument("--year", default=None, help="coco only; inferred if omitted")
    p.add_argument("--metadata", default=None, help="flat only; file name or path")
    p.add_argument("--image_dir", default=None, help="explicit image dir override")
    p.add_argument("--model", default="dinov2_vitl14")
    p.add_argument("--image_size", type=int, default=518)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=-1,
                   help="-1 = SLURM_CPUS_PER_TASK - 1, capped at 8")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dry_run", action="store_true",
                   help="list and count work, load nothing")
    args = p.parse_args()

    root = args.dataset_root
    if not os.path.isdir(root):
        fail(f"dataset_root does not exist: {root}")
    if args.num_shards < 1 or not (0 <= args.shard < args.num_shards):
        fail(f"invalid shard {args.shard}/{args.num_shards}")

    out_dir = args.output_dir or os.path.join(root, "embeddings_dinov2")

    layout = resolve_layout(root, args.layout, args.metadata)
    if layout == "coco":
        source, paths = list_images_coco(root, args.year, args.image_dir)
    else:
        source, paths = list_images_flat(root, find_flat_metadata(root, args.metadata))

    log(f"dataset : {root}")
    log(f"layout  : {layout}   source: {source}")
    log(f"output  : {out_dir}")

    # generate_QA.py keys embeddings by image stem, so collapse stem duplicates
    # here rather than letting two images fight over one .npy
    by_stem, dupes = {}, 0
    for path in sorted(paths):
        k = stem(path)
        if k in by_stem:
            dupes += 1
            continue
        by_stem[k] = path
    if dupes:
        log(f"  warning: {dupes} duplicate image stems ignored")

    all_paths = [by_stem[k] for k in sorted(by_stem)]
    log(f"  {len(all_paths)} unique images")

    mine = all_paths[args.shard::args.num_shards]
    todo = [p for p in mine
            if not os.path.exists(os.path.join(out_dir, stem(p) + ".npy"))]
    log(f"shard {args.shard}/{args.num_shards}: {len(mine)} assigned, "
        f"{len(mine) - len(todo)} already done, {len(todo)} to process")

    if args.dry_run:
        log("dry run; nothing computed.")
        return
    if not todo:
        log("Nothing to do.")
        return

    os.makedirs(out_dir, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log("  note: running on CPU; DINOv2 ViT-L is slow here")
    log(f"device  : {device}")

    workers = args.num_workers
    if workers < 0:
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 0) or (os.cpu_count() or 2)) - 1
        workers = max(0, min(workers, 8))

    model = load_model(args.model, device)
    loader = DataLoader(ImageList(todo, get_transform(args.image_size)),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=workers, collate_fn=collate,
                        pin_memory=(device == "cuda"))

    written, failed = 0, 0
    for batch in tqdm(loader, total=(len(todo) + args.batch_size - 1) // args.batch_size):
        if batch is None:
            continue
        xs, batch_paths = batch
        try:
            embs = embed_batch(model, xs.to(device, non_blocking=True))
        except Exception as exc:
            failed += len(batch_paths)
            print(f"  batch failed: {exc}", flush=True)
            continue

        for emb, path in zip(embs, batch_paths):
            save_path = os.path.join(out_dir, stem(path) + ".npy")
            tmp = save_path + ".tmp.npy"          # atomic: no half-written .npy
            np.save(tmp, np.asarray(emb, dtype="float32").reshape(-1))
            os.replace(tmp, save_path)
            written += 1

    log(f"Done. {written} embeddings written, {failed} failed, dim "
        f"{np.load(os.path.join(out_dir, stem(todo[0]) + '.npy')).shape if written else 'n/a'}")


if __name__ == "__main__":
    main()