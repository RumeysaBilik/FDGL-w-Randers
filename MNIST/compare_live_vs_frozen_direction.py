#!/usr/bin/env python3
"""
compare_live_vs_frozen_direction.py -- runs the SAME D_asym (IsUMap's own
asymmetric distance, PCA-first exactly like embed_MNIST_pca.py) through
randers_umap_fit TWICE at the same epoch count, differing ONLY in how B's
DIRECTION is handled:

  1. "live" (current default, B_fixed=None, use_drift=True): compute_drift()
     is called every epoch on the CURRENT, evolving Y -- both magnitude AND
     direction change over training. This is what embed_MNIST.py/
     embed_MNIST_pca.py/embed_MNIST_raw.py/embed_BreastCancer.py all use
     today.
  2. "frozen" (B_fixed): compute_drift() is called exactly ONCE, on the
     untrained spectral_layout Y_init (same construction randers_umap_fit
     would build internally), then that single (n,d) array is passed as
     B_fixed and never touched again for the rest of training -- direction
     AND magnitude both locked at their epoch-0 value. This is the exact
     mechanism run_swiss_roll_isumap.py's locate_B_from_D_asym() implements
     (see that file for the full derivation/history) -- ported here
     unchanged, just applied to MNIST's own D_asym instead of swiss roll's.

[OURS 2026-08-28, per explicit user request -- "bir MNIST datasinin
sonuclarini 100 epochta fixed direction drift ve su anki hali ile
karsilastirir misin"]

Why the comparison is fair: D_asym itself is built ONCE and reused for both
runs (same k, same distance_graph_generation call) -- the only thing that
differs between the two randers_umap_fit calls is the B_fixed argument, so
any difference in the resulting embedding is attributable to the live-vs-
frozen mechanism itself, not to any other confound.

Dataset note: real MNIST (data_and_plots.load_MNIST) needs either an
internet connection (sklearn fetch_openml on first use) or a pre-cached
Dataset_files/mnist_784.pkl -- NEITHER is available in this sandbox (no
network, no cache found). --dataset digits uses sklearn's bundled
load_digits() (1797 samples, 8x8=64-dim handwritten digits 0-9, no
network/cache needed at all) as a real, offline, MNIST-like stand-in so
this comparison can actually be RUN and shown here; --dataset mnist is the
real thing and should be used when running this on a machine with internet
access (or an already-cached mnist_784.pkl from a prior embed_MNIST*.py
run).

Usage
-----
    python3 MNIST/compare_live_vs_frozen_direction.py --dataset digits --epochs 100
    python3 MNIST/compare_live_vs_frozen_direction.py --dataset mnist --n 5000 --epochs 100
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from distance_graph_generation import distance_graph_generation
from randers_umap import randers_umap_fit, fuzzy_simplicial_set, spectral_layout, compute_drift


def load_data(args):
    if args.dataset == "digits":
        from sklearn.datasets import load_digits
        d = load_digits()
        X, y = d.data.astype(np.float64), d.target.astype(np.int64)
        if args.n is not None and args.n < X.shape[0]:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(X.shape[0], size=args.n, replace=False)
            X, y = X[idx], y[idx]
        return X, y, X.shape[1]  # already low-dim (64), no PCA needed
    else:
        from data_and_plots import load_MNIST
        dataset_path = os.path.join(ROOT, "Dataset_files") + os.sep
        X, y = load_MNIST(args.n, datasetPath=dataset_path)
        pca_dim = min(args.pca_dim, X.shape[0], X.shape[1])
        X = PCA(n_components=pca_dim, random_state=args.seed).fit_transform(X)
        return X, y, pca_dim


def build_D_asym(X, k, verbose):
    n = X.shape[0]
    isumap_dist = distance_graph_generation(
        X, k=k, normalize=True, distBeyondNN=True, verbose=verbose,
        dataIsDistMatrix=False, dataIsGeodesicDistMatrix=False, saveDistMatrix=False,
    )
    asymm_distance = isumap_dist[0]
    D_sparse = np.full((n, n), np.inf)
    np.fill_diagonal(D_sparse, 0.0)
    for (i, j, kk), value in asymm_distance.items():
        if i == j:
            D_sparse[i, kk] = value
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    rows, cols = np.nonzero(np.isfinite(D_sparse) & (D_sparse > 0))
    nbg = csr_matrix((D_sparse[rows, cols], (rows, cols)), shape=(n, n))
    D_asym, _ = shortest_path(nbg, method="auto", directed=True, return_predecessors=True)
    return D_asym


def locate_B_from_D_asym(D_asym, emb_k, clip_delta=0.01, seed=0, verbose=True):
    """
    Ported unchanged from run_swiss_roll_isumap.py -- B derived ENTIRELY
    from D_asym's own asymmetry (no ground truth involved anywhere), but
    computed ONCE on the untrained spectral_layout init and FROZEN, instead
    of live/per-epoch. See that file's own module docstring for the full
    history of why this specific mechanism (not virtual-point, not live)
    was settled on for a "frozen direction" comparison.
    """
    n = D_asym.shape[0]
    mu, knn_mask = fuzzy_simplicial_set(D_asym, emb_k)
    Y_init = spectral_layout(mu, d=2, seed=seed)
    N = (D_asym - D_asym.T) / (D_asym + D_asym.T + 1e-12)
    N = np.where(np.isfinite(N), N, 0.0)
    B_located = compute_drift(N, knn_mask, emb_k, Y_init, clip_delta=clip_delta)
    if verbose:
        bn = np.linalg.norm(B_located, axis=1)
        limit = 1.0 - clip_delta
        print(f"B located (from D_asym asymmetry only, frozen): mean||b||={bn.mean():.4f}  "
              f"max||b||={bn.max():.4f}  clipped={(bn >= limit - 1e-9).sum()}/{n}")
    return B_located


def reconstruct_rho(Y, B):
    diff = Y[np.newaxis, :, :] - Y[:, np.newaxis, :]
    d = np.sqrt((diff ** 2).sum(-1))
    proj = (B[:, np.newaxis, :] * diff).sum(-1)
    rho = d + proj
    np.fill_diagonal(rho, 0.0)
    return rho


def stress(D_target, D_reconstructed):
    mask = ~np.eye(len(D_target), dtype=bool)
    finite = mask & np.isfinite(D_target) & np.isfinite(D_reconstructed)
    num = ((D_target[finite] - D_reconstructed[finite]) ** 2).sum()
    den = (D_target[finite] ** 2).sum() + 1e-12
    return float(np.sqrt(num) / np.sqrt(den))


def knn_purity(Y, y, k=10, seed=0):
    """
    Quick, defensible quality proxy: train/test split, k-NN classifier on
    the 2D embedding coordinates only, accuracy against the TRUE digit
    label -- higher means the embedding kept same-digit points closer to
    each other than to other digits, i.e. class structure survived the
    embedding. Not a claim about drift-direction correctness specifically,
    just a general "did the embedding preserve label structure" number so
    the two conditions can be compared by something other than eyeballing
    the scatter plots.
    """
    Ytr, Yte, ytr, yte = train_test_split(Y, y, test_size=0.3, random_state=seed, stratify=y)
    clf = KNeighborsClassifier(n_neighbors=k).fit(Ytr, ytr)
    return float(clf.score(Yte, yte))


def run_condition(D_asym, X_shape0, args, B_fixed, label, y):
    with np.errstate(invalid="ignore", divide="ignore"):
        out = randers_umap_fit(D_asym, n_neighbors=args.emb_k, n_negative_samples=args.neg,
                                n_epochs=args.epochs, use_drift=(B_fixed is None), B_fixed=B_fixed,
                                clip_delta=args.clip_delta, ramp=False,
                                seed=args.seed, verbose=not args.quiet)
    Y, B = out["Y"], out["B"]
    bn = np.linalg.norm(B, axis=1)
    rho = reconstruct_rho(Y, B)
    s = stress(D_asym, rho)
    purity = knn_purity(Y, y, seed=args.seed)
    if not args.quiet:
        print(f"[{label}] mean||b||={bn.mean():.4f}  max||b||={bn.max():.4f}  "
              f"stress={s:.4f}  knn_purity={purity:.4f}")
    return {"Y": Y, "B": B, "bn": bn, "stress": s, "purity": purity, "label": label}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["mnist", "digits"], default="mnist")
    p.add_argument("--n", type=int, default=5000,
                    help="MNIST subsample size (ignored for --dataset digits unless smaller "
                         "than 1797, digits' own total size)")
    p.add_argument("--pca-dim", type=int, default=50, help="only used for --dataset mnist")
    p.add_argument("--k", type=int, default=30,
                    help="IsUMap distance_graph_generation neighbourhood size")
    p.add_argument("--emb-k", type=int, default=20)
    p.add_argument("--neg", type=int, default=10)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--clip-delta", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="compare_live_vs_frozen")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    verbose = not args.quiet

    save_dir = HERE

    if args.dataset == "digits" and (args.n is None or args.n > 1797):
        args.n = 1797  # load_digits() only has 1797 samples total

    if verbose:
        print(f"Loading {args.dataset} (n={args.n})...")
    X, y, dim = load_data(args)
    n = X.shape[0]
    if verbose:
        print(f"X: {X.shape}")

    if verbose:
        print("\nBuilding D_asym (shared by both conditions)...")
    D_asym = build_D_asym(X, args.k, verbose)
    is_symmetric = np.allclose(D_asym, D_asym.T)
    if verbose:
        print(f"D_asym: {D_asym.shape}  symmetric={is_symmetric}  (should be False)")

    if verbose:
        print("\nLocating frozen B (once, from D_asym asymmetry, before any training)...")
    B_frozen = locate_B_from_D_asym(D_asym, args.emb_k, clip_delta=args.clip_delta,
                                     seed=args.seed, verbose=verbose)

    if verbose:
        print(f"\n=== Running BOTH conditions at epochs={args.epochs} ===")
    res_live = run_condition(D_asym, n, args, B_fixed=None, label="live (current)", y=y)
    res_frozen = run_condition(D_asym, n, args, B_fixed=B_frozen, label="frozen direction", y=y)

    print(f"\n--- summary (epochs={args.epochs}, n={n}, dataset={args.dataset}) ---")
    print(f"{'':20} {'mean||b||':>10} {'max||b||':>10} {'stress':>10} {'knn_purity':>11}")
    for r in (res_live, res_frozen):
        print(f"{r['label']:20} {r['bn'].mean():10.4f} {r['bn'].max():10.4f} "
              f"{r['stress']:10.4f} {r['purity']:11.4f}")

    # ---- side-by-side plot ---------------------------------------------
    n_classes = int(y.max()) + 1
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, r in zip(axes, (res_live, res_frozen)):
        Y, B = r["Y"], r["B"]
        sc = ax.scatter(Y[:, 0], Y[:, 1], c=y, cmap="tab10", s=8, alpha=0.85,
                         linewidths=0, vmin=0, vmax=n_classes - 1)
        bn = np.linalg.norm(B, axis=1)
        big = np.argsort(bn)[::-1][:25]
        if bn.max() > 0:
            sc_scale = 0.12 * (Y.max() - Y.min()) / bn.max()
            ax.quiver(Y[big, 0], Y[big, 1], B[big, 0] * sc_scale, B[big, 1] * sc_scale,
                      color="k", alpha=0.6, width=0.004, scale=1, scale_units="xy")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{r['label']}\nstress={r['stress']:.4f}  knn_purity={r['purity']:.4f}",
                     fontsize=11)
    fig.colorbar(sc, ax=axes, label="digit", ticks=range(n_classes), shrink=0.7)
    fig.suptitle(f"Live vs. frozen-direction drift, {args.dataset} n={n}, epochs={args.epochs}",
                 fontsize=13)
    out_path = os.path.join(save_dir, f"{args.out}.png")
    fig.savefig(out_path, dpi=150)
    if verbose:
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
