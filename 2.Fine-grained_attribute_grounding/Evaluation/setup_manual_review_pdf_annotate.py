#!/usr/bin/env python3
"""
render_annotation_cards.py
=========================
Turn each row of annotation_sheet.csv into a single self-contained "card"
image: the bird photo on top, then the question with its options, the CORRECT
answer, and the MODEL's answer rendered as text below. A rater looks at one
image and writes a rating -- no need to cross-reference the CSV.

The card is BLIND by construction: it never shows CAGS, the band, or which
model produced the answer (annotation_sheet.csv doesn't contain those). It
shows the correct answer and the model answer clearly labelled, because the
task is "how correct is the MODEL answer, given the correct answer + image".

Usage
-----
  # default: reads base_dir/Results/Manual Review/annotation_sheet.csv
  python SampleHumanAnnotators.py --base_dir <dataset_root>
  python SampleHumanAnnotators.py --base_dir <dataset_root> --pdf     # also bundle one PDF
  python SampleHumanAnnotators.py --sheet <path.csv> --limit 5        # test on 5 rows

Outputs (in <sheet_dir>/cards/ by default):
  IT0001.png, IT0002.png, ...           one card per item
  annotation_cards.pdf                  (if --pdf) all cards, one per page

Requires: Pillow, pandas.
"""

from __future__ import annotations
import argparse, os, re, sys
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import evaluate_soft as S   # single source of truth for the dataset root

# ---- palette ----
BG        = (255, 255, 255)
INK       = (28, 28, 30)
MUTED     = (110, 110, 118)
RULE      = (222, 222, 228)
Q_LABEL   = (60, 60, 70)
GOLD_CLR   = (20, 120, 55)     # correct answer
MODEL_CLR  = (40, 70, 175)     # model answer
HEADER_BG = (245, 246, 248)
CARD_BORDER = (200, 200, 208)

FONT_PATHS = {
    "reg":  ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"],
    "bold": ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"],
}


def load_font(kind, size):
    for p in FONT_PATHS[kind]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def wrap(draw, text, font, max_w):
    """Pixel-accurate word wrap."""
    out = []
    for para in str(text).split("\n"):
        words = para.split()
        if not words:
            out.append("")
            continue
        cur = words[0]
        for w in words[1:]:
            if draw.textlength(cur + " " + w, font=font) <= max_w:
                cur += " " + w
            else:
                out.append(cur); cur = w
        out.append(cur)
    return out


_OPT_RE = re.compile(r"\(([A-E])\)\s*(.*?)\s*(?=,\s*\([A-E]\)|$)", re.S)


def split_question(question):
    """Return (stem, [(letter, text), ...]). Options=[] for open-ended."""
    q = str(question)
    if "Options:" not in q:
        return q.strip(), []
    stem, tail = q.split("Options:", 1)
    opts = [(m.group(1), m.group(2).strip()) for m in _OPT_RE.finditer(tail)]
    return stem.strip(), opts


class Pen:
    """Cursor that draws top-to-bottom on a tall scratch canvas."""
    def __init__(self, draw, x0, w, y0):
        self.d = draw; self.x = x0; self.w = w; self.y = y0

    def text_block(self, text, font, color, lh_mult=1.35, indent=0, gap_after=0):
        lines = wrap(self.d, text, font, self.w - indent)
        asc, desc = font.getmetrics()
        lh = int((asc + desc) * lh_mult)
        for ln in lines:
            self.d.text((self.x + indent, self.y), ln, font=font, fill=color)
            self.y += lh
        self.y += gap_after
        return self.y

    def label_value(self, label, value, lfont, vfont, lcolor, vcolor, gap_after=6):
        self.d.text((self.x, self.y), label, font=lfont, fill=lcolor)
        lw = self.d.textlength(label + "  ", font=lfont)
        asc, desc = vfont.getmetrics(); lh = int((asc + desc) * 1.35)
        # wrap value in the remaining width, hanging-indent to align
        lines = wrap(self.d, value, vfont, self.w - lw)
        for i, ln in enumerate(lines):
            xx = self.x + (lw if i == 0 else lw)
            self.d.text((xx, self.y), ln, font=vfont, fill=vcolor)
            self.y += lh
        self.y += gap_after
        return self.y

    def rule(self, gap=10):
        self.y += gap
        self.d.line((self.x, self.y, self.x + self.w, self.y), fill=RULE, width=1)
        self.y += gap


def render_card(row, fonts, width, max_img_h, scale_hint):
    M = 44
    inner_w = width - 2 * M
    scratch_h = 4000
    card = Image.new("RGB", (width, scratch_h), BG)
    d = ImageDraw.Draw(card)

    # ---- header bar ----
    hb_h = 74
    d.rectangle((0, 0, width, hb_h), fill=HEADER_BG)
    d.line((0, hb_h, width, hb_h), fill=RULE, width=1)
    d.text((M, 16), f"{row['item_id']}", font=fonts["h_bold"], fill=INK)
    d.text((M, 44), scale_hint, font=fonts["small"], fill=MUTED)

    y = hb_h + 26

    # ---- bird image ----
    ip = str(row["image_path"])
    try:
        im = Image.open(ip).convert("RGB")
        scale = min(inner_w / im.width, max_img_h / im.height)
        nw, nh = max(1, int(im.width * scale)), max(1, int(im.height * scale))
        im = im.resize((nw, nh), Image.LANCZOS)
        ix = (width - nw) // 2
        card.paste(im, (ix, y))
        d.rectangle((ix, y, ix + nw, y + nh), outline=CARD_BORDER, width=1)
        y += nh + 22
    except Exception as e:
        d.text((M, y), f"[image not found: {ip}]", font=fonts["reg"], fill=(180, 40, 40))
        y += 40

    pen = Pen(d, M, inner_w, y)

    # ---- question stem ----
    stem, opts = split_question(row["question"])
    pen.text_block("QUESTION", fonts["label"], Q_LABEL, gap_after=4)
    pen.text_block(stem, fonts["reg"], INK, gap_after=8)

    # ---- options, one per line ----
    if opts:
        for letter, otext in opts:
            asc, desc = fonts["reg"].getmetrics(); lh = int((asc + desc) * 1.35)
            d.text((pen.x + 8, pen.y), f"({letter})", font=fonts["opt_bold"], fill=INK)
            ow = d.textlength(f"({letter})  ", font=fonts["opt_bold"])
            for i, ln in enumerate(wrap(d, otext, fonts["reg"], inner_w - 8 - ow)):
                d.text((pen.x + 8 + ow, pen.y), ln, font=fonts["reg"], fill=INK)
                pen.y += lh
        pen.y += 6

    pen.rule()

    # ---- correct answer + model answer ----
    pen.label_value("CORRECT ANSWER:", str(row["gold_answer"]),
                    fonts["label"], fonts["reg_bold"], Q_LABEL, GOLD_CLR, gap_after=10)
    pen.label_value("MODEL ANSWER:", str(row["model_answer"]),
                    fonts["label"], fonts["reg_bold"], Q_LABEL, MODEL_CLR, gap_after=14)

    pen.rule()

    # ---- rating prompt ----
    pen.text_block("YOUR RATING  (write it in the CSV next to " + str(row["item_id"]) + "):",
                   fonts["reg"], INK, gap_after=6)
    pen.text_block(scale_hint, fonts["small"], MUTED, gap_after=10)

    final_h = min(pen.y + M, scratch_h)
    card = card.crop((0, 0, width, final_h))
    # thin outer border
    ImageDraw.Draw(card).rectangle((0, 0, width - 1, final_h - 1), outline=CARD_BORDER, width=1)
    return card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True,
                    help="dataset root (REQUIRED); the annotation sheet is read from "
                         "base_dir/Results/Manual Review/ unless --sheet is given")
    ap.add_argument("--sheet", default=None,
                    help="explicit path to annotation_sheet.csv "
                         "(default: base_dir/Results/Manual Review/annotation_sheet.csv)")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--max_img_h", type=int, default=560)
    ap.add_argument("--pdf", action="store_true", help="also write one multi-page PDF")
    ap.add_argument("--limit", type=int, default=0, help="render only first N (0 = all)")
    ap.add_argument("--scale_hint",
                    default="Rate the MODEL ANSWER:   0 = wrong    1 = partially right    2 = right")
    args = ap.parse_args()
    print(f"[start] manual-review render | base_dir = {args.base_dir}", flush=True)

    if args.sheet is None:
        args.sheet = os.path.join(args.base_dir, "Results", "Manual Review",
                                  "annotation_sheet.csv")

    if not os.path.exists(args.sheet):
        sys.exit(f"[error] sheet not found: {args.sheet}")
    df = pd.read_csv(args.sheet)
    need = {"item_id", "image_path", "question", "gold_answer", "model_answer"}
    if not need.issubset(df.columns):
        sys.exit(f"[error] sheet missing columns: {need - set(df.columns)}")
    if args.limit > 0:
        df = df.head(args.limit)

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.sheet)), "cards")
    os.makedirs(out_dir, exist_ok=True)

    fonts = dict(
        h_bold=load_font("bold", 26),
        small=load_font("reg", 16),
        label=load_font("bold", 18),
        reg=load_font("reg", 22),
        reg_bold=load_font("bold", 22),
        opt_bold=load_font("bold", 22),
    )

    pages, n_missing = [], 0
    for _, row in df.iterrows():
        card = render_card(row, fonts, args.width, args.max_img_h, args.scale_hint)
        p = os.path.join(out_dir, f"{row['item_id']}.png")
        card.save(p)
        if args.pdf:
            pages.append(card.convert("RGB"))
        if not os.path.exists(str(row["image_path"])):
            n_missing += 1

    print(f"[cards] wrote {len(df)} card image(s) -> {out_dir}")
    if n_missing:
        print(f"[warn] {n_missing} row(s) had a missing source image (rendered with a placeholder)")
    if args.pdf and pages:
        pdf_path = os.path.join(out_dir, "annotation_cards.pdf")
        pages[0].save(pdf_path, save_all=True, append_images=pages[1:])
        print(f"[pdf] {pdf_path}  ({len(pages)} pages)")


if __name__ == "__main__":
    main()