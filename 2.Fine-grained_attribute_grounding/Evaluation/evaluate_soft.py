from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 0. Constants / lexicons
# --------------------------------------------------------------------------

LETTERS = ["A", "B", "C", "D", "E"]

# sRGB anchors for the 15 CUB colour terms. `iridescent` has no fixed
# chromaticity and is deliberately excluded from the perceptual space.
COLOR_RGB = {
    "black":   (0, 0, 0),
    "white":   (255, 255, 255),
    "grey":    (128, 128, 128),
    "gray":    (128, 128, 128),
    "brown":   (139, 69, 19),
    "buff":    (240, 220, 180),
    "rufous":  (168, 84, 45),
    "red":     (220, 20, 20),
    "orange":  (255, 165, 0),
    "yellow":  (255, 225, 30),
    "olive":   (128, 128, 0),
    "green":   (30, 150, 50),
    "blue":    (30, 60, 220),
    "purple":  (128, 0, 128),
    "pink":    (255, 170, 190),
}
NONCHROMATIC = {"iridescent"}  # exact-match only

# ordinal scales: value -> rank
ORDINAL_SCALES = {
    "bill length": {
        "shorter than head": 0,
        "about the same as head": 1,
        "longer than head": 2,
    },
    "size": {
        "very small (3 - 5 in)": 0,
        "small (5 - 9 in)": 1,
        "medium (9 - 16 in)": 2,
        "large (16 - 32 in)": 3,
        "very large (32 - 72 in)": 4,
    },
}

# nominal pattern confusion: "solid" is qualitatively opposed to the
# structured patterns, which are mutually confusable.
PATTERN_SIM = {
    frozenset(["spotted", "striped"]): 0.5,
    frozenset(["spotted", "multi-colored"]): 0.5,
    frozenset(["striped", "multi-colored"]): 0.5,
    frozenset(["solid", "spotted"]): 0.0,
    frozenset(["solid", "striped"]): 0.0,
    frozenset(["solid", "multi-colored"]): 0.0,
}

# Schematic bird, facing right, y pointing down, z = lateral offset.
# Used when empirical part locations are not supplied.
SCHEMATIC_PARTS = {
    "beak":      (0.95, 0.35, 0.00),
    "forehead":  (0.85, 0.28, 0.00),
    "crown":     (0.78, 0.22, 0.00),
    "left eye":  (0.83, 0.32, -0.03),
    "right eye": (0.83, 0.32, 0.03),
    "nape":      (0.72, 0.25, 0.00),
    "throat":    (0.82, 0.45, 0.00),
    "breast":    (0.68, 0.55, 0.00),
    "belly":     (0.50, 0.62, 0.00),
    "back":      (0.55, 0.30, 0.00),
    "left wing": (0.55, 0.45, -0.05),
    "right wing":(0.55, 0.45, 0.05),
    "left leg":  (0.45, 0.85, -0.04),
    "right leg": (0.45, 0.85, 0.04),
    "tail":      (0.10, 0.45, 0.00),
}

# CUB attribute-group region names -> annotated part(s) whose centroid stands
# in for that region. GLOBAL means "the whole bird" (no localisable region).
REGION_TO_PARTS = {
    "bill":        ["beak"],
    "wing":        ["left wing", "right wing"],
    "upperparts":  ["back"],
    "underparts":  ["belly"],
    "breast":      ["breast"],
    "back":        ["back"],
    "tail":        ["tail"],
    "upper tail":  ["tail"],
    "under tail":  ["tail"],
    "head":        ["crown", "forehead"],
    "throat":      ["throat"],
    "eye":         ["left eye", "right eye"],
    "forehead":    ["forehead"],
    "nape":        ["nape"],
    "belly":       ["belly"],
    "crown":       ["crown"],
    "leg":         ["left leg", "right leg"],
    "primary":     ["GLOBAL"],
    "shape":       ["GLOBAL"],
    "size":        ["GLOBAL"],
}

PROPERTIES = ["color", "colour", "pattern", "shape", "length"]

# Question families
F_ATTR_BARE   = "attr_bare_value"     # "What is the wing color ...?"  options are bare values
F_ATTR_PAIR   = "attr_group_value"    # salience Qs, options are "group: value"
F_SIDE        = "visibility_side"     # Left / Right / Both / None
F_PARTSET     = "visibility_partset"  # (A)(B)(C) / None / All three
F_MARKER      = "marker_localization" # "part marked with the red star"
F_CATEGORY    = "category_mcq"        # "What type of bird is this?"
F_COMMON      = "open_common_name"
F_SCI         = "open_scientific_name"
F_GENUS       = "open_genus"
F_KINDOF      = "open_kind_of"
F_OTHER       = "other"

OPEN_FAMILIES = {F_COMMON, F_SCI, F_GENUS, F_KINDOF}


# --------------------------------------------------------------------------
# 1. Colour space: sRGB -> CIELAB -> CIEDE2000  (pure numpy, no deps)
# --------------------------------------------------------------------------

def _srgb_to_xyz(rgb):
    r, g, b = [c / 255.0 for c in rgb]

    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = lin(r), lin(g), lin(b)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    return x, y, z


def _xyz_to_lab(xyz):
    # D65 reference white
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    x, y, z = xyz[0] / xn, xyz[1] / yn, xyz[2] / zn

    def f(t):
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t + 16 / 116)

    fx, fy, fz = f(x), f(y), f(z)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def rgb_to_lab(rgb):
    return _xyz_to_lab(_srgb_to_xyz(rgb))


def ciede2000(lab1, lab2):
    """Sharma, Wu & Dalal (2005) implementation of CIE Delta-E 2000."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    kL = kC = kH = 1.0

    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    Cbar = 0.5 * (C1 + C2)
    G = 0.5 * (1 - math.sqrt(Cbar ** 7 / (Cbar ** 7 + 25.0 ** 7))) if Cbar > 0 else 0.5
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = math.hypot(a1p, b1), math.hypot(a2p, b2)

    def hp(ap, bp):
        if ap == 0 and bp == 0:
            return 0.0
        h = math.degrees(math.atan2(bp, ap))
        return h + 360 if h < 0 else h

    h1p, h2p = hp(a1p, b1), hp(a2p, b2)

    dLp = L2 - L1
    dCp = C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    else:
        dh = h2p - h1p
        if dh > 180:
            dh -= 360
        elif dh < -180:
            dh += 360
        dhp = dh
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2))

    Lbp = 0.5 * (L1 + L2)
    Cbp = 0.5 * (C1p + C2p)
    if C1p * C2p == 0:
        hbp = h1p + h2p
    else:
        if abs(h1p - h2p) <= 180:
            hbp = 0.5 * (h1p + h2p)
        elif h1p + h2p < 360:
            hbp = 0.5 * (h1p + h2p + 360)
        else:
            hbp = 0.5 * (h1p + h2p - 360)

    T = (1
         - 0.17 * math.cos(math.radians(hbp - 30))
         + 0.24 * math.cos(math.radians(2 * hbp))
         + 0.32 * math.cos(math.radians(3 * hbp + 6))
         - 0.20 * math.cos(math.radians(4 * hbp - 63)))
    dtheta = 30 * math.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * math.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7)) if Cbp > 0 else 0.0
    Sl = 1 + (0.015 * (Lbp - 50) ** 2) / math.sqrt(20 + (Lbp - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -math.sin(math.radians(2 * dtheta)) * Rc

    return math.sqrt(
        (dLp / (kL * Sl)) ** 2
        + (dCp / (kC * Sc)) ** 2
        + (dHp / (kH * Sh)) ** 2
        + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh))
    )


_LAB = {c: rgb_to_lab(v) for c, v in COLOR_RGB.items()}
_DE_MAX = max(ciede2000(_LAB[a], _LAB[b]) for a in _LAB for b in _LAB)


def color_sim(c1, c2, kernel="linear", tau=25.0):
    c1, c2 = c1.strip().lower(), c2.strip().lower()
    if c1 == c2:
        return 1.0
    if c1 in NONCHROMATIC or c2 in NONCHROMATIC:
        return 0.0
    if c1 not in _LAB or c2 not in _LAB:
        return None  # not a colour pair
    dE = ciede2000(_LAB[c1], _LAB[c2])
    if kernel == "exp":
        return float(math.exp(-dE / tau))
    return float(max(0.0, 1.0 - dE / _DE_MAX))


# --------------------------------------------------------------------------
# 2. Anatomy kernel
# --------------------------------------------------------------------------

class Anatomy:
    """Pairwise part similarity from a 3-D layout (schematic or empirical)."""

    def __init__(self, coords: dict):
        self.coords = {k: np.asarray(v, dtype=float) for k, v in coords.items()}
        names = list(self.coords)
        dmax = 0.0
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                dmax = max(dmax, float(np.linalg.norm(self.coords[a] - self.coords[b])))
        self.dmax = dmax if dmax > 0 else 1.0

    def part_sim(self, p1, p2):
        p1, p2 = _norm(p1), _norm(p2)
        if p1 == p2:
            return 1.0
        if p1 not in self.coords or p2 not in self.coords:
            return None
        d = float(np.linalg.norm(self.coords[p1] - self.coords[p2]))
        return max(0.0, 1.0 - d / self.dmax)

    def region_centroid(self, region):
        parts = REGION_TO_PARTS.get(region)
        if not parts or parts == ["GLOBAL"]:
            return None
        pts = [self.coords[p] for p in parts if p in self.coords]
        return np.mean(pts, axis=0) if pts else None

    def region_sim(self, r1, r2):
        r1, r2 = _norm(r1), _norm(r2)
        if r1 == r2:
            return 1.0
        g1 = REGION_TO_PARTS.get(r1) == ["GLOBAL"]
        g2 = REGION_TO_PARTS.get(r2) == ["GLOBAL"]
        if g1 and g2:
            return 1.0
        if g1 or g2:
            return 0.0
        c1, c2 = self.region_centroid(r1), self.region_centroid(r2)
        if c1 is None or c2 is None:
            return 0.0
        d = float(np.linalg.norm(c1 - c2))
        return max(0.0, 1.0 - d / self.dmax)


def build_empirical_anatomy(part_locs_file, images_file, bboxes_file):
    """
    Estimate mean bbox-normalised (x, y) per part from CUB part_locs.txt.
    Lateral z is kept from the schematic so that left/right pairs remain
    distinguishable (they are near-coincident in 2-D projection).
    """
    # part id -> name (parts.txt lives next to part_locs.txt)
    parts_txt = os.path.join(os.path.dirname(part_locs_file), "parts.txt")
    id2name = {}
    with open(parts_txt) as f:
        for line in f:
            i, name = line.strip().split(" ", 1)
            id2name[int(i)] = name.strip()

    bboxes = {}
    with open(bboxes_file) as f:
        for line in f:
            i, x, y, w, h = line.split()
            bboxes[int(i)] = (float(x), float(y), float(w), float(h))

    acc = defaultdict(list)
    with open(part_locs_file) as f:
        for line in f:
            img_id, pid, x, y, vis = line.split()
            if int(vis) != 1:
                continue
            img_id, pid = int(img_id), int(pid)
            if img_id not in bboxes:
                continue
            bx, by, bw, bh = bboxes[img_id]
            if bw <= 0 or bh <= 0:
                continue
            acc[id2name[pid]].append(((float(x) - bx) / bw, (float(y) - by) / bh))

    coords = {}
    for name, pts in acc.items():
        arr = np.asarray(pts)
        mx, my = float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))
        z = SCHEMATIC_PARTS.get(name, (0, 0, 0))[2]
        coords[name] = (mx, my, z)
    for k, v in SCHEMATIC_PARTS.items():   # backfill anything missing
        coords.setdefault(k, v)
    return coords


# --------------------------------------------------------------------------
# 3. Text normalisation, option parsing, prediction resolution
# --------------------------------------------------------------------------

_WS = re.compile(r"\s+")
_PUNC = re.compile(r"[^\w\s:\-\(\)]")


def _norm(s):
    s = str(s).strip().lower().replace("_", " ")
    s = s.replace("gray", "grey")
    s = _PUNC.sub(" ", s)
    return _WS.sub(" ", s).strip()


_OPT_RE = re.compile(r"\(([A-E])\)\s*(.*?)\s*(?=,\s*\([A-E]\)|$)", re.S)


def parse_options(question: str):
    """Return OrderedDict letter -> option text, or {} for open-ended items."""
    if "Options:" not in question:
        return {}
    tail = question.split("Options:", 1)[1]
    return {m.group(1): m.group(2).strip() for m in _OPT_RE.finditer(tail)}


_GOLD_RE = re.compile(r"^\s*\(([A-E])\)\s*(.*)$", re.S)
_LEAD_LETTER = re.compile(r"^\s*\(?([A-E])\)?\s*[\.\):\-]?\s*(?=$|[^\w])")

_PREFIXES = [
    "the answer is", "answer:", "answer is", "final answer:", "final answer",
    "the correct answer is", "correct answer:", "i think", "it is", "this is",
]


def clean_pred(text):
    t = str(text).strip()
    t = t.strip("` *")
    low = t.lower()
    for p in _PREFIXES:
        if low.startswith(p):
            t = t[len(p):].strip(" :*")
            low = t.lower()
    # keep only the first line -- most models put the choice first
    t = t.split("\n")[0].strip()
    return t


def resolve_prediction(pred_raw, options, prefer="letter", fuzzy_thr=0.86):
    """
    -> (letter or None, text or None, status)
    status in {"letter", "text", "fuzzy", "letter_text_mismatch", "no_parse", "open"}
    """
    pred = clean_pred(pred_raw)
    if not options:
        return None, pred, "open"

    inv = {_norm(v): k for k, v in options.items()}

    # (a) explicit parenthesised letter
    letter = None
    m = re.search(r"\(([A-E])\)", pred)
    if m:
        letter = m.group(1)
    else:
        m2 = _LEAD_LETTER.match(pred)
        if m2:
            letter = m2.group(1)

    # (b) verbatim option text anywhere in the prediction
    text_letter = None
    npred = _norm(pred)
    for otext_n, olet in inv.items():
        if otext_n and otext_n in npred:
            text_letter = olet
            break

    if letter and text_letter:
        if letter == text_letter:
            return letter, options[letter], "letter"
        chosen = letter if prefer == "letter" else text_letter
        return chosen, options[chosen], "letter_text_mismatch"
    if letter and letter in options:
        return letter, options[letter], "letter"
    if text_letter:
        return text_letter, options[text_letter], "text"

    # (c) fuzzy fallback
    best, best_r = None, 0.0
    for olet, otext in options.items():
        r = difflib.SequenceMatcher(None, npred, _norm(otext)).ratio()
        if r > best_r:
            best, best_r = olet, r
    if best_r >= fuzzy_thr:
        return best, options[best], "fuzzy"

    return None, pred, "no_parse"


def parse_gold(ans, options):
    m = _GOLD_RE.match(str(ans))
    if m:
        letter, text = m.group(1), m.group(2).strip()
        if options and letter in options:
            return letter, options[letter]
        return letter, text
    return None, str(ans).strip()


# --------------------------------------------------------------------------
# 4. Question-family router
# --------------------------------------------------------------------------

def question_family(question: str, options: dict) -> str:
    q = question.lower()
    if "body part of this bird marked with" in q:
        return F_MARKER
    if "most prominently visible physical trait" in q:
        return F_ATTR_PAIR
    if "are visible in this image" in q:
        opt_texts = {_norm(v) for v in options.values()}
        if opt_texts >= {"left", "right", "both", "none"}:
            return F_SIDE
        return F_PARTSET
    if "what type of bird is this" in q:
        return F_CATEGORY
    if "common name" in q:
        return F_COMMON
    if "scientific name" in q:
        return F_SCI
    if "genus" in q:
        return F_GENUS
    if q.startswith("what kind of"):
        return F_KINDOF
    if re.match(r"what is the .* of this bird\?", q) and options:
        return F_ATTR_BARE
    return F_OTHER


_ATTR_Q_RE = re.compile(r"what is the (.+?) of this bird\?", re.I)


def implicit_group(question):
    m = _ATTR_Q_RE.search(question)
    return _norm(m.group(1)) if m else None


def split_group(group):
    """'upper tail color' -> ('upper tail', 'color');  'shape' -> ('shape','global')"""
    g = _norm(group)
    toks = g.split()
    if toks and toks[-1] in PROPERTIES:
        prop = toks[-1].replace("colour", "color")
        region = " ".join(toks[:-1]) or "GLOBAL"
        return region, prop
    return g, "global"


def split_trait(text):
    """'wing color: blue' -> ('wing color', 'blue')"""
    if ":" in text:
        g, v = text.split(":", 1)
        return _norm(g), _norm(v)
    return None, _norm(text)


# --------------------------------------------------------------------------
# 5. Atom extraction + kernels
# --------------------------------------------------------------------------

def token_f1(a, b):
    ta, tb = set(_norm(a).split()), set(_norm(b).split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    if inter == 0:
        return 0.0
    p, r = inter / len(ta), inter / len(tb)
    return 2 * p * r / (p + r)


def value_sim(group, v1, v2, color_kernel="exp", tau=30.0):
    v1, v2 = _norm(v1), _norm(v2)
    if v1 == v2:
        return 1.0
    g = _norm(group) if group else ""
    if g in ORDINAL_SCALES:
        sc = ORDINAL_SCALES[g]
        if v1 in sc and v2 in sc:
            n = max(sc.values())
            return 1.0 - abs(sc[v1] - sc[v2]) / n
    if g.endswith("pattern"):
        key = frozenset([v1, v2])
        if key in PATTERN_SIM:
            return PATTERN_SIM[key]
    cs = color_sim(v1, v2, kernel=color_kernel, tau=tau)
    if cs is not None:
        return cs
    return token_f1(v1, v2)


def attr_pair_sim(t1, t2, anat: Anatomy, alpha=0.5, color_kernel="exp", tau=30.0):
    """Similarity between two 'group: value' traits."""
    g1, v1 = split_trait(t1)
    g2, v2 = split_trait(t2)
    if g1 is None or g2 is None:
        return token_f1(t1, t2)
    if g1 == g2:
        gs = 1.0
    else:
        r1, p1 = split_group(g1)
        r2, p2 = split_group(g2)
        rs = anat.region_sim(r1, r2)
        ps = 1.0 if p1 == p2 else 0.0
        gs = 0.5 * (rs if rs is not None else 0.0) + 0.5 * ps
    vs = value_sim(g1 if g1 == g2 else None, v1, v2, color_kernel, tau)
    return alpha * gs + (1 - alpha) * vs


SIDE_MAP = {"left": {"L"}, "right": {"R"}, "both": {"L", "R"}, "none": set()}


def partset_atoms(option_text, options):
    """Map a (A)/(B)/(C)/None/All three option to the set of parts it asserts visible."""
    t = _norm(option_text)
    named = [_norm(v) for k, v in options.items()
             if _norm(v) not in ("none", "all three")]
    if t == "none":
        return frozenset()
    if t == "all three":
        return frozenset(named)
    return frozenset([t])


def soft_jaccard(A, B, kern):
    A, B = list(A), list(B)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    used, inter = set(), 0.0
    for a in A:
        best, bi = 0.0, None
        for j, b in enumerate(B):
            if j in used:
                continue
            k = kern(a, b)
            k = 0.0 if k is None else k
            if k > best:
                best, bi = k, j
        if bi is not None and best > 0:
            used.add(bi)
            inter += best
    union = len(A) + len(B) - inter
    return float(inter / union) if union > 0 else 1.0


# ---- open-ended naming --------------------------------------------------

def head_noun(name):
    toks = _norm(name).split()
    return toks[-1] if toks else ""


def common_name_sim(p, g, taxon_map=None):
    p_n, g_n = _norm(p), _norm(g)
    if p_n == g_n:
        return 1.0
    s = 0.6 * token_f1(p_n, g_n) + 0.4 * (1.0 if head_noun(p_n) == head_noun(g_n) else 0.0)
    if taxon_map:
        tp, tg = taxon_map.get(p_n), taxon_map.get(g_n)
        if tp and tg and tp.split()[0].lower() == tg.split()[0].lower():
            s = max(s, 0.75)          # same genus
    return min(1.0, s)


def sci_name_sim(p, g):
    pt, gt = _norm(p).split(), _norm(g).split()
    if not pt or not gt:
        return 0.0
    gen = 1.0 if pt[0] == gt[0] else difflib.SequenceMatcher(None, pt[0], gt[0]).ratio() * 0.3
    if len(pt) > 1 and len(gt) > 1:
        ep = 1.0 if pt[1] == gt[1] else difflib.SequenceMatcher(None, pt[1], gt[1]).ratio() * 0.3
    else:
        ep = 0.0
    return 0.5 * gen + 0.5 * ep


def genus_sim(p, g):
    p_n, g_n = _norm(p).split(), _norm(g).split()
    if not p_n or not g_n:
        return 0.0
    if p_n[0] == g_n[0]:
        return 1.0
    return max(0.0, difflib.SequenceMatcher(None, p_n[0], g_n[0]).ratio() - 0.5) * 2 * 0.3


# --------------------------------------------------------------------------
# 6. Per-item scoring
# --------------------------------------------------------------------------

def score_pair(family, pred_text, gold_text, options, anat, cfg):
    """Raw similarity in [0,1] between a candidate answer and the gold answer."""
    if pred_text is None:
        return 0.0

    if family == F_ATTR_PAIR:
        return attr_pair_sim(pred_text, gold_text, anat, alpha=cfg["alpha"],
                             color_kernel=cfg["color_kernel"], tau=cfg["tau"])

    if family == F_ATTR_BARE:
        g = cfg.get("_implicit_group")
        return value_sim(g, pred_text, gold_text, cfg["color_kernel"], cfg["tau"])

    if family == F_SIDE:
        A = SIDE_MAP.get(_norm(pred_text))
        B = SIDE_MAP.get(_norm(gold_text))
        if A is None or B is None:
            return 0.0
        return soft_jaccard(A, B, lambda a, b: 1.0 if a == b else 0.0)

    if family == F_PARTSET:
        A = partset_atoms(pred_text, options)
        B = partset_atoms(gold_text, options)
        if cfg["anatomy_in_visibility"]:
            kern = lambda a, b: anat.part_sim(a, b) or (1.0 if a == b else 0.0)
        else:
            kern = lambda a, b: 1.0 if a == b else 0.0
        return soft_jaccard(A, B, kern)

    if family == F_MARKER:
        s = anat.part_sim(pred_text, gold_text)
        return s if s is not None else token_f1(pred_text, gold_text)

    if family == F_CATEGORY:
        tm = cfg.get("taxon_map")
        if tm:
            # category head nouns are quasi-taxonomic; reuse genus agreement
            return common_name_sim(pred_text, gold_text, tm)
        return token_f1(pred_text, gold_text)

    if family == F_COMMON or family == F_KINDOF:
        return common_name_sim(pred_text, gold_text, cfg.get("taxon_map"))
    if family == F_SCI:
        return sci_name_sim(pred_text, gold_text)
    if family == F_GENUS:
        return genus_sim(pred_text, gold_text)

    return token_f1(pred_text, gold_text)


def chance_baseline(family, gold_text, options, anat, cfg, open_pool=None):
    """Mean raw similarity of the item's distractors to the gold answer."""
    if options:
        dists = [v for v in options.values() if _norm(v) != _norm(gold_text)]
    elif open_pool:
        dists = [v for v in open_pool if _norm(v) != _norm(gold_text)]
        if len(dists) > cfg["open_pool_sample"]:
            rng = np.random.default_rng(cfg["seed"])
            dists = list(rng.choice(dists, cfg["open_pool_sample"], replace=False))
    else:
        return None
    if not dists:
        return None
    return float(np.mean([score_pair(family, d, gold_text, options, anat, cfg) for d in dists]))


def cags(s_raw, b, eps=1e-6, clip=True):
    if b is None or (1.0 - b) < eps:
        return s_raw
    v = (s_raw - b) / (1.0 - b)
    return float(max(0.0, min(1.0, v))) if clip else float(v)


# --------------------------------------------------------------------------
# 7. Driver
# --------------------------------------------------------------------------

def bootstrap_ci(vals, n=1000, seed=0, alpha=0.05):
    vals = np.asarray([v for v in vals if v is not None and not np.isnan(v)])
    if len(vals) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n)]
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


# --------------------------------------------------------------------------
# 7. Scoring driver (single dataframe)
# --------------------------------------------------------------------------

def score_dataframe(df, args, anat, taxon_map, q_col, a_col, pred_col):
    cfg = dict(alpha=args.alpha, color_kernel=args.color_kernel, tau=args.tau,
               anatomy_in_visibility=args.anatomy_in_visibility,
               taxon_map=taxon_map, open_pool_sample=args.open_pool_sample,
               seed=args.seed)

    open_pool = sorted({_norm(a) for q, a in zip(df[q_col], df[a_col])
                        if "Options:" not in str(q)}) or None

    recs = []
    for _, r in df.iterrows():
        q, gold_raw, pred_raw = str(r[q_col]), r[a_col], r[pred_col]
        options = parse_options(q)
        fam = question_family(q, options)
        cfg["_implicit_group"] = implicit_group(q) if fam == F_ATTR_BARE else None

        g_let, g_txt = parse_gold(gold_raw, options)
        p_let, p_txt, status = resolve_prediction(pred_raw, options, prefer=args.prefer)

        s_raw = 0.0 if status == "no_parse" else score_pair(fam, p_txt, g_txt, options, anat, cfg)

        pool = open_pool if fam in OPEN_FAMILIES else None
        b = chance_baseline(fam, g_txt, options, anat, cfg, open_pool=pool)
        c = cags(s_raw, b, clip=args.clip)

        exact = int((p_txt is not None) and _norm(p_txt) == _norm(g_txt))

        recs.append(dict(
            image_id=r.get("image_id"), family=fam,
            category=r.get("category"),
            gold_letter=g_let, gold_text=g_txt,
            pred_letter=p_let, pred_text=p_txt, resolve_status=status,
            exact=exact, s_raw=round(s_raw, 4),
            chance_b=None if b is None else round(b, 4),
            cags=round(c, 4),
            severity=None if exact else round(1.0 - c, 4),
        ))
    return pd.DataFrame(recs)


def summarize(rows, args):
    def agg(sub):
        n = len(sub)
        wrong = sub[sub.exact == 0]
        lo, hi = bootstrap_ci(sub.cags.values, args.bootstrap, args.seed)
        return pd.Series({
            "n": n,
            "EM": round(sub.exact.mean(), 4),
            "GS_raw": round(sub.s_raw.mean(), 4),
            "CAGS": round(sub.cags.mean(), 4),
            "CAGS_lo": round(lo, 4), "CAGS_hi": round(hi, 4),
            "MistakeSeverity": round(wrong.severity.mean(), 4) if len(wrong) else 0.0,
            "parse_rate": round((sub.resolve_status != "no_parse").mean(), 4),
            "letter_text_mismatch": round((sub.resolve_status == "letter_text_mismatch").mean(), 4),
        })

    def group_agg(col):
        g = rows.groupby(col, dropna=False)
        try:
            return g.apply(agg, include_groups=False)   # pandas >= 2.2
        except TypeError:
            return g.apply(agg)                          # pandas < 2.2

    overall = agg(rows).to_frame("ALL").T
    by_fam = group_agg("family")
    by_cat = group_agg("category")
    return overall, by_fam, by_cat


METRIC_LEGEND = """\
METRIC GLOSSARY
  n                    number of questions in the slice.
  EM                   exact-match accuracy (0-1): fraction where the resolved
                       prediction equals the gold answer after normalisation.
  GS_raw               mean RAW graded similarity (0-1) BEFORE chance adjustment
                       -- the kernel score (colour / anatomy / ordinal / pattern
                       / naming / set-overlap depending on the question family).
  CAGS                 Chance-Adjusted Graded Score (the headline number):
                       (GS_raw - b) / (1 - b) per item, where b is the mean raw
                       similarity of that item's own distractors to the gold
                       answer. 1 = perfect, 0 = no better than a confused guess,
                       < 0 = worse than chance. Reported UNCLIPPED.
  CAGS_lo / CAGS_hi    95% bootstrap confidence interval on the mean CAGS. Two
                       models whose intervals overlap are statistically tied.
  MistakeSeverity      mean (1 - CAGS) over WRONG items only -- how far the misses
                       land from the gold answer. > 1 means the average miss is
                       worse than a random distractor (confidently wrong).
  parse_rate           fraction of MCQ predictions that yielded a usable choice.
  letter_text_mismatch fraction where the chosen letter and the spelled-out
                       option text disagreed (a format-following symptom)."""

FILES_LEGEND = """\
OUTPUT FILES  (paths relative to base_dir)
  <model>/Evaluation/eval_soft_scored.csv    one row per question: family, gold vs
                                             predicted text, resolve_status, s_raw
                                             (raw kernel sim), chance_b, cags,
                                             severity. This is the atom of everything.
  <model>/Evaluation/eval_soft_summary.csv   overall / per-family / per-category
                                             tables in tidy long form (slice, key, ...).
  <model>/Evaluation/eval_soft_stats.txt     the per-model human-readable report.
  Results/Eval - CAGS/eval_soft_leaderboard.csv      one row per model, sorted by CAGS.
  Results/Eval - CAGS/eval_soft_by_category_ALL.csv  model x category CAGS matrix.
  Results/Eval - CAGS/eval_soft_by_family_ALL.csv    model x family CAGS matrix.
  Results/Eval - CAGS/eval_soft_stats_ALL.txt        this merged report."""


def run_one(csv_path, pred_col, out_dir, anat, taxon_map, args, tag=""):
    df = pd.read_csv(csv_path)
    for c in (args.q_col, args.a_col, pred_col):
        if c not in df.columns:
            print(f"    [skip] missing column '{c}' in {os.path.basename(csv_path)}")
            return None

    rows = score_dataframe(df, args, anat, taxon_map, args.q_col, args.a_col, pred_col)
    overall, by_fam, by_cat = summarize(rows, args)

    pd.set_option("display.width", 200)
    print(f"\n=== {tag or os.path.basename(csv_path)}  (pred: {pred_col}, n={len(rows)}) ===")
    print(overall.to_string())
    print("-- by family --");     print(by_fam.to_string())
    print("-- by category --");   print(by_cat.to_string())

    rows_path = os.path.join(out_dir, args.out_rows)
    rows.to_csv(rows_path, index=False)

    summ = pd.concat([
        overall.assign(slice="overall", key="ALL").reset_index(drop=True).set_index("key"),
        by_fam.assign(slice="family").reset_index().rename(columns={"family": "key"}).set_index("key"),
        by_cat.assign(slice="category").reset_index().rename(columns={"category": "key"}).set_index("key"),
    ])
    summ_path = os.path.join(out_dir, args.out_summary)
    summ.to_csv(summ_path)

    o = overall.iloc[0].to_dict()
    o["pred_col"] = pred_col
    o["n"] = len(rows)

    # human-readable per-model report (mirrors the exact-match evaluator)
    L = ["=" * 80, tag or os.path.basename(csv_path), "=" * 80,
         f"CSV                 : {os.path.basename(csv_path)}",
         f"Prediction column   : {pred_col}",
         f"Questions           : {len(rows)}",
         "",
         f"EM (exact match)    : {o.get('EM')}",
         f"GS_raw              : {o.get('GS_raw')}",
         f"CAGS                : {o.get('CAGS')}   "
         f"[{o.get('CAGS_lo')}, {o.get('CAGS_hi')}]",
         f"Mistake Severity    : {o.get('MistakeSeverity')}",
         f"Parse rate          : {o.get('parse_rate')}",
         f"Letter/text mismatch: {o.get('letter_text_mismatch')}",
         "",
         "PER QUESTION FAMILY (n, EM, GS_raw, CAGS, CI, severity, parse ...)",
         "-" * 80,
         by_fam.sort_values("CAGS", ascending=False).to_string(),
         "",
         "PER CATEGORY",
         "-" * 80,
         by_cat.sort_values("CAGS", ascending=False).to_string(),
         "",
         METRIC_LEGEND]
    report = "\n".join(L)
    stats_path = os.path.join(out_dir, args.out_stats)
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"    [write] {rows_path}\n    [write] {summ_path}\n    [write] {stats_path}")

    return o, report, by_fam["CAGS"], by_cat["CAGS"]


# --------------------------------------------------------------------------
# 8. Auto-discovery of model directories
# --------------------------------------------------------------------------

OUTPUT_BASENAMES = set()   # populated in main() from args to avoid re-reading our own outputs


def find_pred_col(df):
    cands = [c for c in df.columns if "_answer" in c and c != "answer"]
    return cands


def discover_models(base_dir):
    """
    Yield (model_name, consolidated_csv_path, pred_col) for every model dir.

    A subdirectory is treated as a MODEL dir iff it contains at least one
    '*chunk*.csv'. Within it, the consolidated file is the .csv that
      (a) has no 'chunk' in its name,
      (b) is not one of our own output files,
      (c) contains a 'question' column and exactly-one '*_answer' column.
    """
    for name in sorted(os.listdir(base_dir)):
        d = os.path.join(base_dir, name)
        if not os.path.isdir(d):
            continue
        csvs = [f for f in os.listdir(d) if f.lower().endswith(".csv")]
        if not any("chunk" in f.lower() for f in csvs):
            continue  # not a model dir

        candidates = [f for f in csvs
                      if "chunk" not in f.lower() and f not in OUTPUT_BASENAMES]
        if not candidates:
            print(f"[warn] {name}: no non-chunk consolidated CSV found -> skipping")
            continue

        chosen, chosen_pred = None, None
        for f in sorted(candidates):
            path = os.path.join(d, f)
            try:
                cols = pd.read_csv(path, nrows=0).columns.tolist()
            except Exception as e:
                print(f"[warn] {name}/{f}: unreadable header ({e})")
                continue
            if "question" not in cols:
                continue
            preds = [c for c in cols if "_answer" in c and c != "answer"]
            if len(preds) >= 1:
                chosen, chosen_pred = f, preds[0]
                if len(preds) > 1:
                    print(f"[warn] {name}/{f}: multiple pred cols {preds}; using '{preds[0]}'")
                break

        if chosen is None:
            print(f"[warn] {name}: no consolidated CSV with a '_answer' column -> skipping")
            continue
        yield name, os.path.join(d, chosen), chosen_pred


# --------------------------------------------------------------------------
# 9. Main
# --------------------------------------------------------------------------
# No hardcoded data root: --base_dir is required and the taxon/part/bbox side
# files are derived from it inside main().


def main():
    ap = argparse.ArgumentParser()
    # --- auto-discovery mode (default) ---
    ap.add_argument("--base_dir", required=True,
                    help="root holding one sub-directory per model (REQUIRED); "
                         "Results/ is written here")
    ap.add_argument("--data_dir", default=None,
                    help="root holding the resource side-files "
                         "(taxon_name_mapping.json, parts/part_locs.txt, "
                         "bounding_boxes.txt). Default: same as --base_dir. "
                         "Use this when the models and the Data/ folder are "
                         "in different places.")
    ap.add_argument("--skip_done", action="store_true",
                    help="skip a model dir if its out_rows already exists")
    # --- single-file mode (optional override) ---
    ap.add_argument("--csv", default=None, help="score a single CSV instead of auto-discovery")
    ap.add_argument("--pred_col", default=None, help="prediction column for single-file mode")
    # --- fixed resources (default under --data_dir) ---
    ap.add_argument("--taxon_map", default=None,
                    help="default: <data_dir>/taxon_name_mapping.json")
    ap.add_argument("--part_locs", default=None,
                    help="default: <data_dir>/parts/part_locs.txt")
    ap.add_argument("--bboxes",    default=None,
                    help="default: <data_dir>/bounding_boxes.txt")
    # --- output names (written inside each model dir) ---
    ap.add_argument("--out_rows", default="eval_soft_scored.csv")
    ap.add_argument("--out_summary", default="eval_soft_summary.csv")
    ap.add_argument("--out_stats", default="eval_soft_stats.txt",
                    help="per-model human-readable report (in each Evaluation/)")
    ap.add_argument("--leaderboard", default="eval_soft_leaderboard.csv",
                    help="combined per-model overall table, written under base_dir/Results/Eval - CAGS")
    # --- columns ---
    ap.add_argument("--q_col", default="question")
    ap.add_argument("--a_col", default="answer")
    # --- scoring knobs ---
    ap.add_argument("--prefer", choices=["letter", "text"], default="letter")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--color_kernel", choices=["linear", "exp"], default="exp")
    ap.add_argument("--tau", type=float, default=30.0)
    ap.add_argument("--anatomy_in_visibility", action="store_true")
    # CAGS is reported UNCLIPPED by default (exact chance floor of 0).
    ap.add_argument("--clip", action="store_true",
                    help="clip CAGS to [0,1] (default: unclipped headline score)")
    ap.add_argument("--no_clip", action="store_true",
                    help=argparse.SUPPRESS)   # accepted for backward compat, no-op
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--open_pool_sample", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # --clip wins if both given; default stays unclipped.
    args.clip = bool(args.clip)

    # Resource side-files live under --data_dir (which defaults to --base_dir).
    # This lets the models sit in one folder and the Data/ folder sit elsewhere.
    data_dir = args.data_dir or args.base_dir
    if args.taxon_map is None:
        args.taxon_map = os.path.join(data_dir, "taxon_name_mapping.json")
    if args.part_locs is None:
        args.part_locs = os.path.join(data_dir, "parts", "part_locs.txt")
    if args.bboxes is None:
        args.bboxes = os.path.join(data_dir, "bounding_boxes.txt")

    # First line out the door, flushed, BEFORE touching the filesystem -- so a
    # slow/stale mount shows up as "stuck after this line" instead of silence.
    print(f"[start] evaluate_soft | base_dir = {args.base_dir} | data_dir = {data_dir}",
          flush=True)

    global OUTPUT_BASENAMES
    OUTPUT_BASENAMES = {args.out_rows, args.out_summary, args.leaderboard}

    # ---- load fixed resources ONCE, reused across all models ----
    if args.part_locs and args.bboxes and os.path.exists(args.part_locs) and os.path.exists(args.bboxes):
        coords = build_empirical_anatomy(args.part_locs, None, args.bboxes)
        print(f"[anatomy] empirical layout from {args.part_locs}")
    else:
        coords = SCHEMATIC_PARTS
        print("[anatomy] schematic layout (part_locs/bboxes not found)")
    anat = Anatomy(coords)

    taxon_map = None
    if args.taxon_map and os.path.exists(args.taxon_map):
        raw = json.load(open(args.taxon_map))
        taxon_map = {_norm(k): v for k, v in raw.items()}
        print(f"[taxonomy] loaded {len(taxon_map)} name->taxon entries")

    # ---- single-file mode ----
    if args.csv:
        out_dir = os.path.dirname(os.path.abspath(args.csv))
        pred_col = args.pred_col
        if pred_col is None:
            preds = find_pred_col(pd.read_csv(args.csv, nrows=0))
            if not preds:
                sys.exit("no '*_answer' column found; pass --pred_col")
            pred_col = preds[0]
            print(f"[auto] pred_col = {pred_col}")
        run_one(args.csv, pred_col, out_dir, anat, taxon_map, args)
        return

    # ---- auto-discovery mode ----
    if not os.path.isdir(args.base_dir):
        sys.exit(f"[error] base_dir does not exist: {args.base_dir}")
    print(f"[discover] scanning {args.base_dir}")
    board = []
    reports = []
    cat_matrix, fam_matrix = {}, {}
    n_models = 0
    for model_name, csv_path, pred_col in discover_models(args.base_dir):
        n_models += 1
        out_dir = os.path.join(os.path.dirname(csv_path), "Evaluation")
        os.makedirs(out_dir, exist_ok=True)
        rows_path = os.path.join(out_dir, args.out_rows)
        if args.skip_done and os.path.exists(rows_path):
            print(f"\n[skip_done] {model_name}: {args.out_rows} exists")
            continue
        print(f"\n>>> MODEL: {model_name}\n    csv : {os.path.basename(csv_path)}\n    pred: {pred_col}")
        res = run_one(csv_path, pred_col, out_dir, anat, taxon_map, args, tag=model_name)
        if res is not None:
            o, report, fam_cags, cat_cags = res
            board.append({"model": model_name, **o})
            reports.append(report)
            fam_matrix[model_name] = fam_cags
            cat_matrix[model_name] = cat_cags

    if board:
        lb = pd.DataFrame(board)
        front = ["model", "pred_col", "n", "EM", "GS_raw", "CAGS",
                 "CAGS_lo", "CAGS_hi", "MistakeSeverity", "parse_rate", "letter_text_mismatch"]
        lb = lb[[c for c in front if c in lb.columns]]
        lb = lb.sort_values("CAGS", ascending=False).reset_index(drop=True)
        cags_dir = os.path.join(args.base_dir, "Results", "Eval - CAGS")
        os.makedirs(cags_dir, exist_ok=True)
        lb_path = os.path.join(cags_dir, args.leaderboard)
        lb.to_csv(lb_path, index=False)

        # merged CAGS-by-category / CAGS-by-family matrices (rows = model)
        cat_df = pd.DataFrame(cat_matrix).T.sort_index()
        fam_df = pd.DataFrame(fam_matrix).T.sort_index()
        cat_df.round(4).to_csv(os.path.join(cags_dir, "eval_soft_by_category_ALL.csv"))
        fam_df.round(4).to_csv(os.path.join(cags_dir, "eval_soft_by_family_ALL.csv"))

        # merged human-readable report
        all_path = os.path.join(cags_dir, "eval_soft_stats_ALL.txt")
        with open(all_path, "w", encoding="utf-8") as f:
            f.write("SOFT (GRADED) EVALUATION - MERGED REPORT\n")
            f.write("=" * 100 + "\n\n")
            f.write("WHAT THIS IS\n")
            f.write("  A graded companion to exact match. Instead of right/wrong, every answer\n")
            f.write("  earns partial credit for being CLOSE (a near colour, an adjacent body\n")
            f.write("  part, an off-by-one size, a shared genus), then that credit is adjusted\n")
            f.write("  for how easy the question's distractors made it to look good. The\n")
            f.write("  resulting CAGS is the headline; EM is kept alongside for reference.\n\n")
            f.write(METRIC_LEGEND + "\n\n")
            f.write(FILES_LEGEND + "\n\n")
            f.write("HOW TO READ THE LEADERBOARD\n")
            f.write("  Sorted by CAGS (higher = better). Where CAGS > EM, partial credit lifted\n")
            f.write("  the model (its misses were near-misses). Where CAGS < EM, its misses were\n")
            f.write("  worse than chance (see MistakeSeverity > 1). Overlapping CAGS_lo/CAGS_hi\n")
            f.write("  between two rows means the gap is not statistically meaningful.\n\n")
            f.write("=" * 100 + "\n")
            f.write("OVERALL MODEL COMPARISON  (sorted by CAGS)\n")
            f.write("=" * 100 + "\n\n")
            f.write(lb.to_string(index=False))
            f.write("\n\nCAGS BY CATEGORY (rows = model)\n")
            f.write("-" * 100 + "\n")
            f.write(cat_df.round(4).to_string())
            f.write("\n\nCAGS BY QUESTION FAMILY (rows = model)\n")
            f.write("-" * 100 + "\n")
            f.write(fam_df.round(4).to_string())
            f.write("\n\n" + "=" * 100 + "\n")
            f.write("INDIVIDUAL MODEL REPORTS\n")
            f.write("=" * 100 + "\n\n")
            f.write(("\n\n" + "=" * 100 + "\n\n").join(reports))

        print("\n" + "=" * 80)
        print("LEADERBOARD (sorted by CAGS)")
        print("=" * 80)
        print(lb.to_string(index=False))
        print(f"\n[write] {lb_path}\n[write] {all_path}")
    print(f"\n[done] {n_models} model dir(s) processed.")


if __name__ == "__main__":
    main()