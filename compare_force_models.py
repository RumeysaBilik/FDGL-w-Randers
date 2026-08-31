#!/usr/bin/env python3
"""
compare_force_models.py -- [OURS 2026-08-31, per explicit user request --
"bu paperın methodu ve UMAP'in methodunun neden bu kadar farklı sonuçlar
verdiklerini güzelce analiz etmek istiyorum"] quantifies WHY
force_model="fr_gravity" (Bannister et al./Fruchterman-Reingold, the new
default) and force_model="umap" (the original (a,b)-curve law) produce such
different embeddings, on the SAME D_asym / same everything-else, differing
ONLY in force_model.

Two datasets, two different kinds of evidence:

  --dataset swiss_roll (default): the intrinsic coordinate t is known
      EXACTLY (it's how the data was generated), so we can directly test
      whether each force model correctly "unrolls" the strip or spuriously
      closes it into a ring. This matters because Bannister et al.'s own
      stated design goal is literally "vertices placed uniformly in a disk"
      (Section 3 of arXiv:1209.0748) -- a good property for a generic
      social-network graph, but actively WRONG for an open-strip manifold:
      the two true ends of the roll (t near t_min and t near t_max) should
      stay far apart in the embedding, not become neighbours.

  --dataset digits: sklearn's offline load_digits() (10 real classes),
      a real, generic (non-synthetic-geometry) dataset -- same common
      metrics as swiss_roll below, minus end_to_end (which needs a known
      1-D coordinate).

Metrics
-------
common (both datasets):
  - asymmetry_score [OURS 2026-08-31, per explicit user request --
    "spearmani cikarip asimetri skor ekler misin"] ported from
    randers_bridge.asymmetry_score(D, bln): for each real graph edge
    (i,j), |D[i,j]-D[j,i]| / (D[i,j]+D[j,i]), in [0,1), then averaged.
    Computed TWICE: once on the raw D_asym itself (the "target" asymmetry
    the data actually has -- printed once, doesn't depend on force_model),
    and once per force model on rho_reconstructed = ||y_i-y_j|| +
    b_i.(y_j-y_i) built from that model's own final (Y,B) -- i.e. how much
    of the target asymmetry actually survived training under each force
    law. This is exactly the "ne kadar kaybettik / ne kadar sakladik"
    comparison asymmetry_score's own docstring describes, applied here to
    force_model instead of to a locate-vs-live comparison. Works
    identically for both datasets (unlike the old spearman check, which
    needed swiss roll's ground-truth t and so couldn't run on digits).
  - edge_length_cv : coefficient of variation (std/mean) of embedding
    distance along the TRUE k-NN graph edges. Tests whether force_model=
    "fr_gravity" imposes a more UNIFORM edge length (its own built-in
    equilibrium spacing k, see module docstring below) than UMAP's locally
    density-adaptive edges.
  - extent

swiss_roll only:
  - end_to_end / edge_len : mean embedding distance between the lowest-5%-t
    group and the highest-5%-t group (the strip's two true ends), in units
    of a typical k-NN edge length. LOW = the two ends have been pulled
    artificially close -- a ring-closure artifact.

Why an equilibrium spacing exists for force_model="fr_gravity" but not for
"umap": at a real k-NN edge (i,j), fr_gravity's net radial force is
    f_a - f_r = rho/k - k^2/rho^2,
which is zero exactly at rho=k -- every real edge is pulled toward the SAME
length k regardless of local density. UMAP's own force has no such crossing
point built from a single shared constant: attraction is suppressed by mu
and vanishes as rho->0, repulsion is suppressed by (1-mu) and is near-zero
for real neighbours (mu~1) in the first place, so real edges are pulled
in almost unopposed until the attraction curve itself decays -- there is no
single shared equilibrium length, each neighbourhood settles at whatever
scale its own local density (baked into mu via smooth_knn_dist's rho_i/
sigma_i) implies.

Usage
-----
    python3 compare_force_models.py --dataset swiss_roll --n 1500 --epochs 500
    python3 compare_force_models.py --dataset digits --n 1200 --epochs 300
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from randers_bridge import compute_dist_matrix, asymmetry_score
from randers_umap import randers_umap_fit, fuzzy_simplicial_set


# ─────────────────────────────────────────────────────────────────────────
# data loaders
# ─────────────────────────────────────────────────────────────────────────
def make_swiss_roll(n, seed=42):
    """Ported unchanged from run_swiss_roll.py's make_swiss_roll_randers."""
    from run_swiss_roll import make_swiss_roll_randers
    return make_swiss_roll_randers(n, seed=seed)


def load_digits_data(n, seed):
    from sklearn.datasets import load_digits
    d = load_digits()
    X, y = d.data.astype(np.float64), d.target.astype(np.int64)
    if n is not None and n < X.shape[0]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=n, replace=False)
        X, y = X[idx], y[idx]
    return X, y


# ─────────────────────────────────────────────────────────────────────────
# shared metrics
# ─────────────────────────────────────────────────────────────────────────
def edge_length_stats(Y, knn_mask):
    """
    Returns (cv, mean_len) of embedding distance along the TRUE k-NN graph
    edges (knn_mask, symmetrised). cv (std/mean) low = force_model imposes
    a near-uniform edge length across the whole embedding, high = edge
    length varies a lot from one neighbourhood to another. mean_len is used
    elsewhere as a robust, local "typical neighbour gap" normalization
    scale (see end_to_end_ratio).
    """
    A = knn_mask | knn_mask.T
    diff = Y[np.newaxis, :, :] - Y[:, np.newaxis, :]
    d = np.sqrt((diff ** 2).sum(-1))
    lens = d[A]
    if lens.size == 0:
        return float("nan"), float("nan")
    return float(lens.std() / max(lens.mean(), 1e-12)), float(lens.mean())


def reconstruct_rho(Y, B):
    """
    rho(i->j) = ||y_i-y_j|| + b_i.(y_j-y_i) -- same formula
    randers_umap_fit's own training loop and test.py/
    compare_live_vs_frozen_direction.py use, standalone here so
    asymmetry_score can be evaluated on the TRAINED embedding's own
    reconstructed distances, not just on the raw input D_asym.
    """
    diff = Y[np.newaxis, :, :] - Y[:, np.newaxis, :]
    d = np.sqrt((diff ** 2).sum(-1))
    proj = (B[:, np.newaxis, :] * diff).sum(-1)
    rho = d + proj
    np.fill_diagonal(rho, 0.0)
    return rho


def end_to_end_ratio(Y, t, mean_edge_len, frac=0.05):
    """
    Mean embedding distance between the lowest-`frac` and highest-`frac`
    percentile-t groups (the strip's two true ends), normalized by the
    MEAN TRUE k-NN EDGE LENGTH (a robust, local "typical neighbour gap"
    scale) rather than the embedding's overall extent -- extent itself can
    be blown up by a handful of far-flung outlier points (which is exactly
    what happens under force_model="umap", see module analysis), which
    would make an extent-normalized ratio LOOK artificially low even when
    the two ends are not actually close. LOW here = the two ends really
    are only a few typical-neighbour-gaps apart (ring-closure-like
    artifact); HIGH = they are many neighbour-gaps apart, i.e. properly
    unrolled.
    """
    lo_cut = np.quantile(t, frac)
    hi_cut = np.quantile(t, 1.0 - frac)
    lo = Y[t <= lo_cut]
    hi = Y[t >= hi_cut]
    d = np.linalg.norm(lo[:, None, :] - hi[None, :, :], axis=-1)
    return float(d.mean() / max(mean_edge_len, 1e-12))


# ─────────────────────────────────────────────────────────────────────────
# run one condition
# ─────────────────────────────────────────────────────────────────────────
def run_condition(D_asym, bln, args, force_model, label, extra):
    with np.errstate(invalid="ignore", divide="ignore"):
        out = randers_umap_fit(D_asym, n_neighbors=args.emb_k, n_negative_samples=args.neg,
                                n_epochs=args.epochs, use_drift=True, B_fixed=None,
                                clip_delta=args.clip_delta, ramp=False,
                                seed=args.seed, verbose=not args.quiet,
                                force_model=force_model)
    Y, B, knn_mask = out["Y"], out["B"], out["knn_mask"]
    ext = float(Y.max() - Y.min())
    cv, mean_edge = edge_length_stats(Y, knn_mask)

    # [OURS 2026-08-31] how much of D_asym's own target asymmetry survived
    # training under THIS force model -- reconstruct rho from this run's
    # own (Y,B), score it the same way asymmetry_score scores D_asym
    # itself, using the SAME real edges (bln) both times so the two
    # numbers are directly comparable.
    with np.errstate(invalid="ignore", divide="ignore"):
        rho_rec = reconstruct_rho(Y, B)
        _, asym_rec = asymmetry_score(rho_rec, bln)

    result = {"Y": Y, "B": B, "label": label, "extent": ext, "edge_length_cv": cv,
              "mean_edge_len": mean_edge, "asym_score": asym_rec}

    if extra["dataset"] == "swiss_roll":
        t = extra["t"]
        result["end_to_end"] = end_to_end_ratio(Y, t, mean_edge)
        if not args.quiet:
            print(f"[{label}] extent={ext:.2f}  edge_len_cv={cv:.4f}  "
                  f"asym_score={asym_rec:.4f}  end_to_end/edge_len={result['end_to_end']:.4f}")
    else:
        if not args.quiet:
            print(f"[{label}] extent={ext:.2f}  edge_len_cv={cv:.4f}  asym_score={asym_rec:.4f}")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["swiss_roll", "digits"], default="swiss_roll")
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--k", type=int, default=20, help="geodesic backbone k-NN")
    p.add_argument("--emb-k", type=int, default=20)
    p.add_argument("--neg", type=int, default=10)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--clip-delta", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="compare_force_models")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    verbose = not args.quiet

    if args.dataset == "digits" and args.n > 1797:
        args.n = 1797

    if verbose:
        print(f"Loading {args.dataset} (n={args.n})...")

    if args.dataset == "swiss_roll":
        X, omega, t = make_swiss_roll(args.n, seed=42)
        y_color = t
        cmap = "viridis"
        D_asym, _, bln = compute_dist_matrix(X, n_neighbors=args.k, path_method="auto",
                                              randers_field=omega, return_adjacency=True)
        extra = {"dataset": "swiss_roll", "t": t}
    else:
        X, y = load_digits_data(args.n, args.seed)
        y_color = y
        cmap = "tab10"
        # digits has no natural omega field -- use a generic PCA-direction
        # asymmetric field only so D_asym is genuinely directed, matching
        # how the isumap-family scripts build their own D_asym; the force-
        # model comparison itself does not depend on this choice.
        from sklearn.decomposition import PCA
        pcs = PCA(n_components=2, random_state=args.seed).fit_transform(X)
        omega = np.zeros_like(X)
        omega[:, :2] = 0.05 * pcs / max(np.linalg.norm(pcs, axis=1).mean(), 1e-9)
        D_asym, _, bln = compute_dist_matrix(X, n_neighbors=args.k, path_method="auto",
                                              randers_field=omega, return_adjacency=True)
        extra = {"dataset": "digits", "y": y}

    n = X.shape[0]
    # [OURS 2026-08-31] target asymmetry -- a property of D_asym itself,
    # independent of force_model, printed once for reference. Each
    # condition's own "asym_score" (in run_condition) is the SAME
    # computation applied to that model's own reconstructed rho, so the two
    # numbers are directly comparable: how much of THIS survived training.
    _, asym_target = asymmetry_score(D_asym, bln)
    if verbose:
        print(f"X: {X.shape}  D_asym symmetric={np.allclose(D_asym, D_asym.T)}  (should be False)")
        print(f"target asymmetry_score (D_asym itself) = {asym_target:.4f}")
        print(f"\n=== Running BOTH force models at epochs={args.epochs} ===")

    res_fr = run_condition(D_asym, bln, args, "fr_gravity", "fr_gravity (new default)", extra)
    res_umap = run_condition(D_asym, bln, args, "umap", "umap (original)", extra)

    print(f"\n--- summary (epochs={args.epochs}, n={n}, dataset={args.dataset}, "
          f"target asym_score={asym_target:.4f}) ---")
    if args.dataset == "swiss_roll":
        print(f"{'':26} {'extent':>8} {'edge_cv':>9} {'asym_score':>10} {'end2end/edge':>13}")
        for r in (res_fr, res_umap):
            print(f"{r['label']:26} {r['extent']:8.2f} {r['edge_length_cv']:9.4f} "
                  f"{r['asym_score']:10.4f} {r['end_to_end']:13.4f}")
    else:
        print(f"{'':26} {'extent':>8} {'edge_cv':>9} {'asym_score':>10}")
        for r in (res_fr, res_umap):
            print(f"{r['label']:26} {r['extent']:8.2f} {r['edge_length_cv']:9.4f} "
                  f"{r['asym_score']:10.4f}")

    # ---- side-by-side plot -----------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, r in zip(axes, (res_fr, res_umap)):
        Y = r["Y"]
        sc = ax.scatter(Y[:, 0], Y[:, 1], c=y_color, cmap=cmap, s=8, alpha=0.85, linewidths=0)
        ax.set_xticks([]); ax.set_yticks([])
        if args.dataset == "swiss_roll":
            ax.set_title(f"{r['label']}\nasym_score={r['asym_score']:.3f}  "
                          f"end2end/edge={r['end_to_end']:.3f}  edge_cv={r['edge_length_cv']:.3f}",
                          fontsize=10)
        else:
            ax.set_title(f"{r['label']}\nasym_score={r['asym_score']:.3f}  "
                          f"edge_cv={r['edge_length_cv']:.3f}", fontsize=10)
    fig.colorbar(sc, ax=axes, label=("t" if args.dataset == "swiss_roll" else "digit"), shrink=0.7)
    fig.suptitle(f"force_model comparison, {args.dataset} n={n}, epochs={args.epochs}", fontsize=13)
    out_path = os.path.join(HERE, f"{args.out}_{args.dataset}.png")
    fig.savefig(out_path, dpi=150)
    if verbose:
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
