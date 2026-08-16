#!/usr/bin/env python3
"""
BioVQA C4 - specialist re-identification baseline.

Answers "are these pairs solvable at all?" with a purpose-built wildlife re-ID
embedding, so the VLM results can be read against an achievable target rather
than against nothing.

  # 1. extract embeddings (GPU, ~1-2h for all five species)
  python megadescriptor.py --base_dir /path/to/4-Instance-Level-Identification \\
      --mode extract

  # 2. evaluate (CPU, seconds)
  python megadescriptor.py --base_dir /path/to/4-Instance-Level-Identification \\
      --mode evaluate

WHY NOT DINOv2
The pairs were constructed by thresholding DINOv2 cosine similarity, so a
DINOv2 classifier separates easy pairs almost perfectly and hard pairs almost
perfectly backwards - by definition, not by perception. That number is
reported here anyway, clearly labelled, because it quantifies exactly how much
of the difficulty effect is baked into the construction. It is a construction
check, NOT a baseline.

WHY AUC RATHER THAN A THRESHOLD
Any fixed cosine threshold is arbitrary and dataset-specific. AUC is
threshold-free and is the standard metric for verification tasks. It converts
to d' as d' = sqrt(2) * z(AUC), which puts this baseline in the same units as
the VLM evaluation. Accuracy at a threshold is also reported, but the
threshold is FITTED on the random condition and applied unchanged to easy and
hard, so those two are out-of-fit. An oracle (best-possible) threshold is
reported per cell as an upper bound.
"""

import argparse
import os
import re
import sys
import time
from statistics import NormalDist

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

_Z = NormalDist().inv_cdf
_PHI = NormalDist().cdf
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# MegaDescriptor: trained on wildlife re-identification datasets, so it is a
# fair "specialist" reference point rather than a generic feature extractor.
DEFAULT_MODEL = "hf-hub:BVRA/MegaDescriptor-L-384"
DEFAULT_EMB_DIR = "embeddings_megadescriptor"


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def stem(path):
    return os.path.splitext(os.path.basename(str(path)))[0]


# --------------------------------------------------------------------------- #
# source image discovery (mirrors generate_QA.py)
# --------------------------------------------------------------------------- #

def find_image_dir(species_dir):
    for cand in ("Images", "images"):
        p = os.path.join(species_dir, cand)
        if os.path.isdir(p):
            return p
    p = os.path.join(species_dir, "images")
    return p if os.path.isdir(p) else species_dir


def index_images(image_dir):
    mapping = {}
    for root, dirs, files in os.walk(image_dir):
        dirs[:] = [d for d in dirs
                   if not d.startswith("instance_identification_")
                   and not d.startswith("embeddings_")]
        for f in sorted(files):
            if f.lower().endswith(IMG_EXTS):
                mapping.setdefault(stem(f), os.path.join(root, f))
    return mapping


def species_dirs(base_dir):
    """Directories that directly contain instance_identification_* subdirs."""
    found = set()
    for root, dirs, _ in os.walk(base_dir):
        if any(d.startswith("instance_identification_") for d in dirs):
            found.add(root)
    return sorted(found)


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #

def extract(base_dir, model_name, emb_dir_name, batch_size, image_size,
            shard_id=0, num_shards=1, io_workers=8):
    import torch
    import timm
    from PIL import Image, ImageFile
    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    log("loading " + model_name)
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model = model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    log("device: " + device)

    cfg = timm.data.resolve_data_config({}, model=model)
    if image_size:
        cfg["input_size"] = (3, image_size, image_size)
    transform = timm.data.create_transform(**cfg)
    log("input size: %s" % (cfg["input_size"],))
    if num_shards > 1:
        log("shard %d of %d" % (shard_id, num_shards))

    max_h = cfg["input_size"][1]

    def load_one(item):
        """Decode + transform one image. Runs on a worker thread: PIL decode
        releases the GIL, so this genuinely overlaps across threads."""
        k, path = item
        try:
            with Image.open(path) as im:
                # reduced-scale JPEG decode: the source frames are far larger
                # than the model input, so full decode is wasted work
                if im.height > max_h * 2:
                    im.draft("RGB", (im.width // 2, im.height // 2))
                return k, transform(im.convert("RGB"))
        except Exception as exc:
            log("  unreadable %s: %r" % (path, exc))
            return k, None

    for sp in species_dirs(base_dir):
        img_dir = find_image_dir(sp)
        images = index_images(img_dir)
        out_dir = os.path.join(sp, emb_dir_name)
        os.makedirs(out_dir, exist_ok=True)

        # ORDER MATTERS. Shard assignment is computed on the FULL sorted list
        # and only then filtered for work already done. Doing it the other way
        # round makes membership depend on what happens to be on disk when the
        # shard starts, so concurrent shards stride over different lists and
        # silently skip images.
        #
        # Striding rather than contiguous slicing: contiguous chunking drops
        # the tail when the count is not divisible by the shard count, and it
        # unbalances load when file sizes vary across the sorted order.
        assigned = sorted(images.items())
        if num_shards > 1:
            assigned = assigned[shard_id::num_shards]
        todo = [(k, p) for k, p in assigned
                if not os.path.exists(os.path.join(out_dir, k + ".npy"))]

        log("%s: %d images, %d assigned to this shard, %d still to extract"
            % (os.path.basename(sp), len(images), len(assigned), len(todo)))
        if not todo:
            continue

        pool = ThreadPoolExecutor(max_workers=max(1, io_workers))
        for i in range(0, len(todo), batch_size):
            chunk = todo[i:i + batch_size]
            loaded = list(pool.map(load_one, chunk))
            keys = [k for k, t in loaded if t is not None]
            tensors = [t for _k, t in loaded if t is not None]
            if not tensors:
                continue
            batch = torch.stack(tensors).to(device)
            with torch.no_grad():
                feats = model(batch).float().cpu().numpy()
            for k, v in zip(keys, feats):
                # np.save APPENDS ".npy" when given a path that lacks it, so
                # np.save("x.npy.tmp", ...) silently writes "x.npy.tmp.npy".
                # Writing through an open file handle suppresses that.
                tmp = os.path.join(out_dir, k + ".npy.tmp")
                with open(tmp, "wb") as fh:
                    np.save(fh, v.astype("float32"))
                os.replace(tmp, os.path.join(out_dir, k + ".npy"))
            if (i // batch_size) % 20 == 0:
                log("  %d/%d" % (min(i + batch_size, len(todo)), len(todo)))
        pool.shutdown(wait=False)

    log("extraction complete")


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def auc_score(scores, labels):
    """
    Rank-based AUC (Mann-Whitney U), ties handled by average ranks.
    scores: similarity, higher = more likely same individual
    labels: 1 = same individual, 0 = different
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within tie groups
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


def auc_to_dprime(auc):
    """d' = sqrt(2) * z(AUC). Same units as the VLM evaluation."""
    if auc != auc:
        return float("nan")
    auc = min(max(auc, 1e-6), 1 - 1e-6)
    return float(np.sqrt(2.0) * _Z(auc))


def best_threshold(scores, labels):
    """Threshold maximising accuracy, plus that accuracy (the oracle bound)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(scores) == 0:
        return float("nan"), float("nan")
    cands = np.unique(scores)
    mids = np.concatenate(([cands[0] - 1e-6],
                           (cands[:-1] + cands[1:]) / 2.0,
                           [cands[-1] + 1e-6]))
    best_t, best_a = float("nan"), -1.0
    for t in mids:
        acc = float(((scores >= t).astype(int) == labels).mean())
        if acc > best_a:
            best_a, best_t = acc, float(t)
    return best_t, best_a


def cell_metrics(scores, labels, threshold):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    auc = auc_score(scores, labels)
    o_t, o_a = best_threshold(scores, labels)

    pred = (scores >= threshold).astype(int) if threshold == threshold else None
    row = {
        "n": len(scores),
        "auc": auc,
        "d_prime_from_auc": auc_to_dprime(auc),
        "oracle_threshold": o_t,
        "oracle_accuracy": o_a,
        "mean_sim_same": float(scores[labels == 1].mean()) if (labels == 1).any() else float("nan"),
        "mean_sim_diff": float(scores[labels == 0].mean()) if (labels == 0).any() else float("nan"),
    }
    row["margin_same_minus_diff"] = row["mean_sim_same"] - row["mean_sim_diff"]
    if pred is None:
        row.update({"threshold": float("nan"), "accuracy": float("nan"),
                    "balanced_accuracy": float("nan")})
        return row
    tp = int(((labels == 1) & (pred == 1)).sum())
    fn = int(((labels == 1) & (pred == 0)).sum())
    fp = int(((labels == 0) & (pred == 1)).sum())
    tn = int(((labels == 0) & (pred == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    row.update({
        "threshold": float(threshold),
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": 0.5 * (sens + spec),
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
    })
    return row


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #

def load_pairs(task_dir):
    """(labels, left_stems, right_stems, stored_dinov2_sims) or None."""
    meta = os.path.join(task_dir, "pairs_metadata.csv")
    if not os.path.isfile(meta):
        return None
    df = pd.read_csv(meta)
    need = {"label", "image_left", "image_right"}
    if not need <= set(df.columns):
        return None
    labels = (df["label"].astype(str).str.strip().str.lower() == "yes").astype(int).values
    sims = (df["cosine_similarity"].values
            if "cosine_similarity" in df.columns else None)
    return labels, df["image_left"].astype(str).values, \
        df["image_right"].astype(str).values, sims


def pair_similarities(emb_dir, left, right):
    """Cosine similarity per pair; NaN where an embedding is missing."""
    cache = {}

    def get(k):
        if k not in cache:
            p = os.path.join(emb_dir, str(k) + ".npy")
            if not os.path.exists(p):
                cache[k] = None
            else:
                v = np.load(p).astype("float64").reshape(-1)
                n = np.linalg.norm(v)
                cache[k] = v / n if n > 0 else None
        return cache[k]

    out = np.full(len(left), np.nan)
    for i, (a, b) in enumerate(zip(left, right)):
        va, vb = get(a), get(b)
        if va is not None and vb is not None and va.shape == vb.shape:
            out[i] = float(np.dot(va, vb))
    return out


def evaluate(base_dir, emb_dir_name, out_dir):
    rows = []
    for sp in species_dirs(base_dir):
        species = re.sub(r"\.coco$", "", os.path.basename(sp))
        emb_dir = os.path.join(sp, emb_dir_name)
        has_specialist = os.path.isdir(emb_dir)

        for d in sorted(os.listdir(sp)):
            if not d.startswith("instance_identification_"):
                continue
            task_dir = os.path.join(sp, d)
            condition = d.split("_")[-1]
            pairs = load_pairs(task_dir)
            if pairs is None:
                log("  no pairs_metadata.csv in %s - skipping" % task_dir)
                continue
            labels, left, right, dino = pairs

            if has_specialist:
                sims = pair_similarities(emb_dir, left, right)
                ok = ~np.isnan(sims)
                if ok.sum() >= 10:
                    rows.append(dict(embedding="specialist", species=species,
                                     condition=condition, n_missing=int((~ok).sum()),
                                     **{k: v for k, v in
                                        cell_metrics(sims[ok], labels[ok],
                                                     float("nan")).items()}))
            if dino is not None:
                rows.append(dict(embedding="dinov2_construction_check",
                                 species=species, condition=condition,
                                 n_missing=0,
                                 **cell_metrics(np.asarray(dino, dtype=float),
                                                labels, float("nan"))))

    if not rows:
        sys.exit("no pairs_metadata.csv found under " + base_dir)

    df = pd.DataFrame(rows)

    # Fit one threshold per (embedding, species) on the RANDOM condition, then
    # apply it unchanged to easy and hard so those are out-of-fit.
    fitted = {}
    for (emb, species), g in df.groupby(["embedding", "species"]):
        r = g[g.condition == "random"]
        fitted[(emb, species)] = (float(r.oracle_threshold.iloc[0])
                                  if len(r) else float("nan"))

    out = []
    for sp in species_dirs(base_dir):
        species = re.sub(r"\.coco$", "", os.path.basename(sp))
        emb_dir = os.path.join(sp, emb_dir_name)
        for d in sorted(os.listdir(sp)):
            if not d.startswith("instance_identification_"):
                continue
            task_dir = os.path.join(sp, d)
            condition = d.split("_")[-1]
            pairs = load_pairs(task_dir)
            if pairs is None:
                continue
            labels, left, right, dino = pairs
            for emb, sims in (("specialist",
                               pair_similarities(emb_dir, left, right)
                               if os.path.isdir(emb_dir) else None),
                              ("dinov2_construction_check",
                               np.asarray(dino, dtype=float)
                               if dino is not None else None)):
                if sims is None:
                    continue
                ok = ~np.isnan(sims)
                if ok.sum() < 10:
                    continue
                t = fitted.get((emb, species), float("nan"))
                m = cell_metrics(sims[ok], labels[ok], t)
                m.update({"embedding": emb, "species": species,
                          "condition": condition,
                          "threshold_fitted_on": "random",
                          "in_fit": condition == "random",
                          "n_missing": int((~ok).sum())})
                out.append(m)

    res = pd.DataFrame(out)
    lead = ["embedding", "species", "condition", "n", "auc", "d_prime_from_auc",
            "accuracy", "balanced_accuracy", "threshold", "in_fit",
            "oracle_threshold", "oracle_accuracy", "mean_sim_same",
            "mean_sim_diff", "margin_same_minus_diff", "n_missing"]
    res = res[lead + [c for c in res.columns if c not in lead]]
    res = res.sort_values(["embedding", "species", "condition"])

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "baseline_results.csv")
    res.to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)

    write_report(res, os.path.join(out_dir, "baseline_report.txt"), base_dir)
    log("written to " + out_dir)
    return res


def write_report(res, path, base_dir):
    L, A = [], None
    A = L.append
    A("=" * 78)
    A("BioVQA C4 - specialist re-identification baseline")
    A("Generated: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    A("Source:    %s" % base_dir)
    A("=" * 78)
    A("")
    A("AUC is threshold-free. d' = sqrt(2) * z(AUC), so it is directly")
    A("comparable with the d' column of the VLM evaluation.")
    A("Thresholds are fitted on the random condition per species and applied")
    A("unchanged to easy and hard, which are therefore out-of-fit.")
    A("")

    for emb, g in res.groupby("embedding"):
        A("-" * 78)
        title = ("SPECIALIST EMBEDDING" if emb == "specialist"
                 else "DINOv2 - CONSTRUCTION CHECK, NOT A BASELINE")
        A(title)
        A("-" * 78)
        if emb != "specialist":
            A("The pairs were built by thresholding these similarities, so")
            A("these numbers are near-deterministic. They quantify how much of")
            A("the difficulty effect is baked into the construction.")
            A("")
        A("  %-14s %-8s %6s %7s %8s %9s %9s" % (
            "species", "cond", "n", "AUC", "d'", "acc", "oracle"))
        for sp, gg in g.groupby("species"):
            for cond in ("easy", "random", "hard"):
                r = gg[gg.condition == cond]
                if r.empty:
                    continue
                r = r.iloc[0]
                A("  %-14s %-8s %6d %7.3f %8.3f %9.3f %9.3f" % (
                    sp[:14], cond, r.n, r.auc, r.d_prime_from_auc,
                    r.accuracy, r.oracle_accuracy))
        A("")
        rnd = g[g.condition == "random"]
        if len(rnd):
            A("  neutral-condition mean AUC %.3f  (d' %.3f)"
              % (rnd.auc.mean(), rnd.d_prime_from_auc.mean()))
        A("")

    spec = res[res.embedding == "specialist"]
    if len(spec):
        r = spec[spec.condition == "random"]
        A("=" * 78)
        A("INTERPRETATION")
        A("=" * 78)
        A("Specialist neutral-condition d' = %.3f (mean across species)."
          % r.d_prime_from_auc.mean())
        A("")
        A("Compare with the best VLM in the same condition. If the specialist")
        A("is clearly higher, the pairs are solvable and the VLM shortfall is")
        A("a capability gap rather than an unsolvable task - the stronger")
        A("claim. If the specialist is also low, report the task as hard for")
        A("everything and drop any language implying VLMs specifically fail.")
        A("")
        h = spec[spec.condition == "hard"]
        if len(h):
            A("Specialist hard-condition d' = %.3f. If this stays positive"
              % h.d_prime_from_auc.mean())
            A("while every VLM goes negative, that is direct evidence the hard")
            A("pairs are adversarial to appearance matching specifically,")
            A("rather than simply impossible.")
        A("")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_dir", required=True)
    p.add_argument("--mode", choices=["extract", "evaluate", "both"],
                   default="both")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--emb_dir", default=DEFAULT_EMB_DIR,
                   help="subdirectory name written under each species dir")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--image_size", type=int, default=0,
                   help="override the model's default input size")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--shard_id", type=int, default=0,
                   help="stride-sharded extraction; use with a SLURM array")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--io_workers", type=int, default=8,
                   help="threads decoding images ahead of the GPU")
    args = p.parse_args()

    base = os.path.abspath(args.base_dir)
    if not os.path.isdir(base):
        sys.exit("base_dir does not exist: " + base)
    out_dir = args.out_dir or os.path.join(base, "evaluation")

    if args.mode in ("extract", "both"):
        extract(base, args.model, args.emb_dir, args.batch_size, args.image_size,
                args.shard_id, args.num_shards, args.io_workers)
    if args.mode in ("evaluate", "both"):
        evaluate(base, args.emb_dir, out_dir)


if __name__ == "__main__":
    main()