from __future__ import annotations

import argparse
import itertools
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

import evaluate_soft as S


# --------------------------------------------------------------------------
# rank-correlation helpers (hand-rolled: no scipy dependency)
# --------------------------------------------------------------------------

def _ranks(x):
    """Average ranks (1..n), ties averaged."""
    x = np.asarray(x, float)
    order = np.argsort(-x, kind="mergesort")       # descending: best -> rank 1
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(1, len(x) + 1)
    # average ties
    _, inv, counts = np.unique(-x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return ranks


def spearman(a, b):
    ra, rb = _ranks(a), _ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 1.0


def kendall_tau(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a); c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            if s > 0: c += 1
            elif s < 0: d += 1
    tot = c + d
    return float((c - d) / tot) if tot else 1.0


def rank_vector(scores):
    """scores: dict model->cags  ->  dict model->rank (1=best)."""
    models = list(scores)
    r = _ranks([scores[m] for m in models])
    return {m: int(round(r[i])) for i, m in enumerate(models)}


# --------------------------------------------------------------------------
# config grid
# --------------------------------------------------------------------------

def make_args(defaults, **over):
    a = SimpleNamespace(**vars(defaults))
    for k, v in over.items():
        setattr(a, k, v)
    return a


def build_grid(d, mode, with_policy):
    """
    Return list of (name, args). 'd' is the default-args namespace.
    Deduplicates configs identical to default; drops tau when kernel=linear.
    """
    configs = [("default", d)]

    tau_vals   = [20.0, 30.0, 40.0, 50.0]
    alpha_vals = [0.3, 0.4, 0.5, 0.6, 0.7]
    kern_vals  = ["exp", "linear"]

    if mode == "oat":
        # one knob at a time, others at default
        for t in tau_vals:
            if t != d.tau and d.color_kernel == "exp":
                configs.append((f"tau={t:g}", make_args(d, tau=t)))
        for al in alpha_vals:
            if al != d.alpha:
                configs.append((f"alpha={al:g}", make_args(d, alpha=al)))
        for k in kern_vals:
            if k != d.color_kernel:
                configs.append((f"kernel={k}", make_args(d, color_kernel=k)))
    else:  # full cross of tau x alpha x kernel
        for k, al, t in itertools.product(kern_vals, alpha_vals, tau_vals):
            if k == "linear" and t != d.tau:
                continue  # tau irrelevant for linear -> avoid duplicates
            name = f"k={k},a={al:g}" + (f",t={t:g}" if k == "exp" else "")
            configs.append((name, make_args(d, color_kernel=k, alpha=al, tau=t)))

    if with_policy:
        configs.append(("prefer=text", make_args(d, prefer="text")))
        configs.append(("anat_in_vis=1", make_args(d, anatomy_in_visibility=True)))

    # dedupe by the tuple of the knobs that matter
    seen, uniq = set(), []
    for name, a in configs:
        key = (round(a.tau, 3), round(a.alpha, 3), a.color_kernel,
               a.prefer, bool(a.anatomy_in_visibility))
        if key in seen and name != "default":
            continue
        seen.add(key); uniq.append((name, a))
    return uniq


# --------------------------------------------------------------------------
# stratified subsample (paired across configs, per model)
# --------------------------------------------------------------------------

def stratified_sample(df, q_col, n_sample, seed):
    """Proportional stratified subsample by question family.

    Version-agnostic: samples index labels per group directly instead of
    groupby().apply(), which changed grouping-column semantics across pandas
    releases (include_groups). Same seed -> same rows across all configs.
    """
    if n_sample <= 0 or n_sample >= len(df):
        return df
    fams = df[q_col].astype(str).map(
        lambda q: S.question_family(q, S.parse_options(q)))
    frac = n_sample / len(df)
    keep = []
    for _, idx in df.groupby(fams, sort=False).groups.items():
        idx = pd.Index(idx)
        k = min(len(idx), max(1, int(round(len(idx) * frac))))
        keep.extend(pd.Series(idx).sample(k, random_state=seed).tolist())
    return df.loc[keep]


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True,
                    help="dataset root holding the model sub-dirs (REQUIRED)")
    ap.add_argument("--data_dir", default=None,
                    help="root holding taxon/part/bbox side-files "
                         "(default: same as --base_dir)")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--taxon_map", default=None)
    ap.add_argument("--part_locs", default=None)
    ap.add_argument("--bboxes",    default=None)
    ap.add_argument("--q_col", default="question")
    ap.add_argument("--a_col", default="answer")
    ap.add_argument("--grid", choices=["oat", "full"], default="oat")
    ap.add_argument("--with_policy", action="store_true",
                    help="also vary --prefer and anatomy-in-visibility")
    ap.add_argument("--n_sample", type=int, default=60000,
                    help="stratified rows per model (0 = full dataset)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="20k subsample, OAT grid")
    # metric defaults (must match the main run)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=30.0)
    ap.add_argument("--color_kernel", choices=["linear", "exp"], default="exp")
    ap.add_argument("--prefer", choices=["letter", "text"], default="letter")
    ap.add_argument("--anatomy_in_visibility", action="store_true")
    ap.add_argument("--open_pool_sample", type=int, default=40)
    args = ap.parse_args()

    if args.quick:
        args.n_sample = 20000
        args.grid = "oat"

    # resource side-files live under --data_dir (defaults to --base_dir)
    data_dir = args.data_dir or args.base_dir
    if args.taxon_map is None:
        args.taxon_map = os.path.join(data_dir, "taxon_name_mapping.json")
    if args.part_locs is None:
        args.part_locs = os.path.join(data_dir, "parts", "part_locs.txt")
    if args.bboxes is None:
        args.bboxes = os.path.join(data_dir, "bounding_boxes.txt")

    print(f"[start] sensitivity_sweep | base_dir = {args.base_dir} | data_dir = {data_dir}"
          f"{' (+policy)' if args.with_policy else ''}", flush=True)

    out_dir = args.out_dir or os.path.join(args.base_dir, "Results", "Eval - CAGS")
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.isdir(args.base_dir):
        sys.exit(f"[error] base_dir not found: {args.base_dir}")

    # ---- fixed resources, loaded once ----
    if os.path.exists(args.part_locs) and os.path.exists(args.bboxes):
        coords = S.build_empirical_anatomy(args.part_locs, None, args.bboxes)
        print(f"[anatomy] empirical layout")
    else:
        coords = S.SCHEMATIC_PARTS
        print("[anatomy] schematic layout")
    anat = S.Anatomy(coords)

    taxon_map = None
    if os.path.exists(args.taxon_map):
        import json
        taxon_map = {S._norm(k): v for k, v in json.load(open(args.taxon_map)).items()}
        print(f"[taxonomy] {len(taxon_map)} entries")

    # default-args namespace for score_dataframe
    defaults = SimpleNamespace(
        alpha=args.alpha, tau=args.tau, color_kernel=args.color_kernel,
        prefer=args.prefer, anatomy_in_visibility=args.anatomy_in_visibility,
        open_pool_sample=args.open_pool_sample, seed=args.seed, clip=False)

    grid = build_grid(defaults, args.grid, args.with_policy)
    S.OUTPUT_BASENAMES = {"eval_soft_scored.csv", "eval_soft_summary.csv", "eval_soft_leaderboard.csv"}
    print(f"[grid] {len(grid)} configs: " + ", ".join(n for n, _ in grid))

    models = list(S.discover_models(args.base_dir))
    if not models:
        sys.exit("[error] no model dirs discovered")
    print(f"[models] {len(models)} discovered\n")

    # cags_matrix[model][config_name] = mean CAGS
    cags_matrix = {}
    for model_name, csv_path, pred_col in models:
        df = pd.read_csv(csv_path)
        sub = stratified_sample(df, args.q_col, args.n_sample, args.seed)
        row = {}
        for cname, cargs in grid:
            scored = S.score_dataframe(sub, cargs, anat, taxon_map,
                                       args.q_col, args.a_col, pred_col)
            row[cname] = float(scored["cags"].mean())
        cags_matrix[model_name] = row
        print(f"  {model_name:<24} n={len(sub):>7}  "
              f"default CAGS={row['default']:.4f}  "
              f"spread=[{min(row.values()):.4f},{max(row.values()):.4f}]")

    M = pd.DataFrame(cags_matrix).T           # rows=models, cols=configs
    M = M[[n for n, _ in grid]]               # preserve config order
    M.index.name = "model"                    # so the CSV's first column is labelled
    config_names = list(M.columns)

    # rank matrix (1 = best per config)
    R = pd.DataFrame({c: pd.Series(rank_vector(M[c].to_dict())) for c in config_names})
    R = R.loc[M.index]
    R.index.name = "model"

    # stability of each config vs default
    default_scores = M["default"]
    default_ranks = R["default"]
    top3_default = set(default_ranks.sort_values().index[:3])

    stab = []
    for c in config_names:
        rho = spearman(default_scores.values, M[c].values)
        ktau = kendall_tau(default_scores.values, M[c].values)
        disp = int((R[c] - default_ranks).abs().max())
        top3 = set(R[c].sort_values().index[:3])
        stab.append(dict(config=c, spearman_rho=round(rho, 4),
                         kendall_tau=round(ktau, 4),
                         max_rank_displacement=disp,
                         top3_stable=(top3 == top3_default),
                         mean_abs_dCAGS=round(float((M[c] - default_scores).abs().mean()), 4)))
    stab_df = pd.DataFrame(stab)

    # ---- console ----
    pd.set_option("display.width", 200)
    print("\n" + "=" * 78)
    print("PER-CONFIG STABILITY vs DEFAULT")
    print("=" * 78)
    print(stab_df.to_string(index=False))

    non_default = stab_df[stab_df.config != "default"]
    min_rho = float(non_default.spearman_rho.min())
    min_tau = float(non_default.kendall_tau.min())
    max_disp = int(non_default.max_rank_displacement.max())
    n_top3 = int(non_default.top3_stable.sum())
    n_nd = len(non_default)
    worst_cfg = str(non_default.loc[non_default.mean_abs_dCAGS.idxmax(), "config"])
    worst_dcags = float(non_default.mean_abs_dCAGS.max())

    print("\n" + "-" * 78)
    print("HEADLINE (exclude default row):")
    print(f"  min Spearman rho          = {min_rho:.4f}")
    print(f"  min Kendall tau           = {min_tau:.4f}")
    print(f"  max rank displacement     = {max_disp} position(s)")
    print(f"  top-3 set stable in       = {n_top3}/{n_nd} configs")
    print(f"  mean |dCAGS| (worst cfg)  = {worst_dcags:.4f}  ({worst_cfg})")
    print("-" * 78)

    # ---- which models actually move, and between whom ----
    rank_min = R.min(axis=1)
    rank_max = R.max(axis=1)
    movers = [(m, int(rank_min[m]), int(rank_max[m]))
              for m in R.index if rank_min[m] != rank_max[m]]
    winner = str(default_ranks.sort_values().index[0])
    winner_pinned = (R.loc[winner].nunique() == 1)

    # verdict phrasing driven by the actual numbers
    if min_rho >= 0.99 and max_disp <= 1 and n_top3 == n_nd:
        verdict = "NEAR-INVARIANT: the ranking is effectively independent of these knobs."
    elif min_rho >= 0.95 and max_disp <= 2:
        verdict = "HIGHLY STABLE: minor local reshuffling only; the story is unchanged."
    else:
        verdict = ("MODERATELY STABLE: inspect the movers below before making "
                   "invariance claims.")

    # ---- filenames: keep the two runs separate ----
    suffix = "_with_policy" if args.with_policy else ""
    m_path = os.path.join(out_dir, f"sweep_cags_matrix{suffix}.csv")
    r_path = os.path.join(out_dir, f"sweep_rank_matrix{suffix}.csv")
    s_path = os.path.join(out_dir, f"sweep_stability{suffix}.csv")
    t_path = os.path.join(out_dir, f"sweep_report{suffix}.txt")
    M.round(4).to_csv(m_path)
    R.to_csv(r_path)
    stab_df.to_csv(s_path, index=False)

    n_cfg = len(config_names)
    swept = ", ".join(n for n, _ in grid if n != "default")
    mode = "WITH policy toggles (prefer, anatomy-in-visibility)" if args.with_policy \
        else "base grid (tau / alpha / kernel)"
    n_sample_txt = "full dataset" if args.n_sample <= 0 else f"{args.n_sample} rows/model"

    with open(t_path, "w", encoding="utf-8") as f:
        f.write("METRIC SENSITIVITY / RANKING-STABILITY REPORT\n")
        f.write(f"run mode: {mode}\n")
        f.write("=" * 90 + "\n\n")

        f.write("WHAT THIS TESTS\n")
        f.write("  A benchmark leaderboard is only trustworthy if it reflects the models, not\n")
        f.write("  the arbitrary constants inside the scoring metric. This sweep re-scores every\n")
        f.write("  model under many settings of the metric's free parameters and measures how\n")
        f.write("  far the leaderboard moves relative to the default. Small movement => the\n")
        f.write("  ranking is a property of the models; large movement => the ranking is an\n")
        f.write("  artefact of parameter choices and cannot be reported as-is.\n\n")

        f.write(f"CONFIGURATIONS ({n_cfg} total; grid='{args.grid}')\n")
        f.write("  default    the settings used for the headline run.\n")
        f.write("  tau=*      exponential colour-kernel temperature. Larger tau = a gentler\n")
        f.write("             fall-off, so more distant colours still earn partial credit.\n")
        f.write("             Only touches colour-bearing questions.\n")
        f.write("  alpha=*    in the paired-attribute kernel, the weight on the attribute GROUP\n")
        f.write("             (which region/property) vs the VALUE (which colour/pattern).\n")
        f.write("             alpha high = rewards naming the right body region; alpha low =\n")
        f.write("             rewards naming the right value.\n")
        f.write("  kernel=*   linear vs exponential mapping of CIEDE2000 colour distance to a\n")
        f.write("             similarity. Tests whether the colour story depends on that shape.\n")
        if args.with_policy:
            f.write("  prefer=text     when a prediction gives BOTH a letter and spelled-out text\n")
            f.write("                  that disagree, resolve toward the text instead of the letter.\n")
            f.write("                  Tests robustness to the letter-vs-text tie-break policy.\n")
            f.write("  anat_in_vis=1   fold anatomy (part-location) similarity into the visibility\n")
            f.write("                  set kernel instead of scoring parts as exact-match only.\n")
            f.write("                  Tests whether crediting near-miss parts changes conclusions.\n")
        f.write(f"  swept: {swept}\n\n")

        f.write("SAMPLING\n")
        f.write(f"  {n_sample_txt}, stratified by question family, seed={args.seed}. The SAME rows\n")
        f.write("  are scored under every configuration, so every difference is attributable to\n")
        f.write("  the knob and nothing else (no sampling noise between configs).\n\n")

        f.write("OUTPUT FILES  (this directory; the '%s' suffix marks the policy run)\n"
                % (suffix or "(none)"))
        f.write(f"  sweep_cags_matrix{suffix}.csv  rows = model (first col 'model'), cols = config;\n")
        f.write("                                 each cell = that model's mean CAGS under that config.\n")
        f.write(f"  sweep_rank_matrix{suffix}.csv  same shape, but ranks (1 = best) within each config.\n")
        f.write(f"  sweep_stability{suffix}.csv    one row per config vs default (the table below).\n\n")

        f.write("HOW TO READ THE STABILITY TABLE\n")
        f.write("  spearman_rho / kendall_tau  rank correlation between that config's ordering and\n")
        f.write("                              the default ordering. 1.0 = identical order; >=0.99\n")
        f.write("                              is very strong; tau is the stricter of the two.\n")
        f.write("  max_rank_displacement       the largest number of positions any single model\n")
        f.write("                              moves under that config (0 = nobody moved).\n")
        f.write("  top3_stable                 True iff the SET of top-3 models is unchanged\n")
        f.write("                              (their internal order may still differ).\n")
        f.write("  mean_abs_dCAGS              mean |CAGS(config) - CAGS(default)| across models --\n")
        f.write("                              the size of the SCORE shift, independent of ranking.\n")
        f.write("                              Small here means absolute numbers are also stable,\n")
        f.write("                              not just their order.\n\n")

        f.write("=" * 90 + "\n")
        f.write("PER-CONFIG STABILITY vs DEFAULT\n")
        f.write("=" * 90 + "\n")
        f.write(stab_df.to_string(index=False))
        f.write("\n\nHEADLINE (excluding the default row)\n")
        f.write("-" * 90 + "\n")
        f.write(f"  min Spearman rho        = {min_rho:.4f}\n")
        f.write(f"  min Kendall tau         = {min_tau:.4f}\n")
        f.write(f"  max rank displacement   = {max_disp} position(s)\n")
        f.write(f"  top-3 set stable in     = {n_top3}/{n_nd} configs\n")
        f.write(f"  mean |dCAGS| worst cfg  = {worst_dcags:.4f}  (at {worst_cfg})\n\n")

        f.write("=" * 90 + "\n")
        f.write("WHAT THIS RUN IMPLIES\n")
        f.write("=" * 90 + "\n")
        f.write(f"  VERDICT: {verdict}\n\n")
        f.write(f"  * Rank #1 ({winner}) is "
                + ("PINNED across every configuration.\n" if winner_pinned
                   else "NOT stable -- it changes under some configuration (investigate).\n"))
        if movers:
            f.write(f"  * Only {len(movers)} of {len(R.index)} models change rank at all; the rest are\n")
            f.write("    pinned. The models that move, and the rank window they move within:\n")
            for m, lo, hi in sorted(movers, key=lambda x: x[1]):
                f.write(f"      - {m:<24} ranks {lo}-{hi}\n")
            f.write("    Movement confined to adjacent ranks means the metric is only reordering\n")
            f.write("    models that were already near-tied; check sweep_cags_matrix to confirm the\n")
            f.write("    swapped pairs differ by a hair, and cross-reference their bootstrap CIs in\n")
            f.write("    the CAGS leaderboard -- overlapping CIs make the swap a tie, not instability.\n")
        else:
            f.write("  * No model changes rank under ANY configuration: the ordering is identical\n")
            f.write("    everywhere. This is the strongest possible stability result.\n")
        f.write("\n")
        f.write("  IMPLICATION FOR THE PAPER\n")
        if min_rho >= 0.99 and max_disp <= 1 and n_top3 == n_nd:
            f.write("    You can state that model rankings are insensitive to the metric's free\n")
            f.write("    parameters within the swept ranges, and that the only rank exchanges occur\n")
            f.write("    between statistically-tied models. This pre-empts the reviewer objection\n")
            f.write("    that CAGS results depend on hand-tuned constants.\n")
        else:
            f.write("    Report the observed correlations and, for any config that moves a model by\n")
            f.write("    more than one rank, show that the pair's bootstrap CIs overlap before\n")
            f.write("    claiming invariance. Otherwise scope the claim to the stable subset.\n")
        if args.with_policy:
            f.write("    Because this is the POLICY run, the letter-vs-text tie-break and the\n")
            f.write("    anatomy-in-visibility choice are now covered too: if they land in the same\n")
            f.write("    stable band, those design decisions are shown not to drive the results.\n")
        f.write("\n  CAVEATS (state these so stability doesn't read as trivial)\n")
        f.write("    - Stability is demonstrated WITHIN the swept ranges; name them explicitly.\n")
        f.write("    - The graded kernels perturb a partial-credit layer sitting on top of a\n")
        f.write("      stable exact-match / chance-adjustment backbone, which is part of why the\n")
        f.write("      ranking barely moves. That is expected, not a flaw -- but say it.\n")
        f.write(f"    - The largest score shift was mean |dCAGS| = {worst_dcags:.4f} at '{worst_cfg}';\n")
        f.write("      if that config also reorders a pair, give it a sentence of its own.\n\n")

        f.write("=" * 90 + "\n")
        f.write("MEAN CAGS PER CONFIG (rows = model)\n")
        f.write("-" * 90 + "\n")
        f.write(M.round(4).to_string())
        f.write("\n\nRANK PER CONFIG (rows = model, 1 = best)\n")
        f.write("-" * 90 + "\n")
        f.write(R.to_string())

    print(f"\n[write] {m_path}\n[write] {r_path}\n[write] {s_path}\n[write] {t_path}")


if __name__ == "__main__":
    main()