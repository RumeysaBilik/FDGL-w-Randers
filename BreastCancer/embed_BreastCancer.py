#!/usr/bin/env python3
"""
embed_BreastCancer.py -- applies our IsUMap + Randers-UMAP pipeline (same
core mechanism as MNIST/embed_MNIST_raw.py: IsUMap's own distance_graph_
generation for the asymmetric distance, our own randers_umap_fit for the
embedding, no PCA, no virtual points/class-centroid drift) to the Wisconsin
Breast Cancer Diagnostic dataset (BreastCancerDataset.csv, 569 samples, 30
real-valued features, binary diagnosis label M/B).

[OURS 2026-08-26]

Differences from the MNIST scripts (dataset-specific, mechanism unchanged):
  - Loading: plain pandas.read_csv instead of load_MNIST/fetch_openml. Two
    columns are dropped before anything else touches the data: `id`
    (a row index, not a feature) and `Unnamed: 32` (entirely NaN -- an
    artifact of a trailing comma in the source CSV, not real data).
  - Feature scaling: [IMPORTANT, genuinely new relative to MNIST] MNIST's
    784 pixel features are already on one shared [0,1] scale, so no
    additional scaling was needed there. This dataset's 30 features are NOT
    comparable in scale (e.g. area_mean ranges ~150-2500, smoothness_mean
    ranges ~0.05-0.16) -- IsUMap's own k-NN search (plain Euclidean) would
    otherwise be dominated entirely by the large-magnitude features
    (area, perimeter) and effectively ignore the small-magnitude ones
    (smoothness, symmetry, fractal_dimension). sklearn's StandardScaler
    (zero mean, unit variance per feature) is applied before
    distance_graph_generation ever sees the data -- this is the standard,
    close-to-mandatory preprocessing step for this exact dataset in the
    wider ML literature, not an optional choice.
  - Labels: diagnosis (M=malignant, B=benign) is a binary categorical label,
    not a 10-way digit -- plotted with a 2-colour map (red=M, green=B) and
    an explicit legend instead of a 0-9 colourbar.
  - Dataset is small (n=569 total) -- no --n subsampling flag, the whole
    dataset is used every run; correspondingly no on-disk caching either
    (a full run is fast).

Usage
-----
    python3 BreastCancer/embed_BreastCancer.py
    python3 BreastCancer/embed_BreastCancer.py --epochs 500 --snapshot-every 50
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler

from distance_graph_generation import distance_graph_generation
from randers_umap import randers_umap_fit


def load_breast_cancer_csv(path):
    """
    Loads BreastCancerDataset.csv, drops the non-feature columns (`id`,
    and `Unnamed: 32` which is entirely NaN -- an artifact of a trailing
    comma in the source file, not real data), returns:
        X_raw : (n, 30) float64 raw feature matrix (NOT yet scaled)
        y     : (n,) int64, 1=malignant (M), 0=benign (B)
        feature_names : list of the 30 column names, in order
    """
    df = pd.read_csv(path)
    drop_cols = [c for c in ("id", "Unnamed: 32") if c in df.columns]
    df = df.drop(columns=drop_cols)
    y = (df["diagnosis"] == "M").astype(np.int64).values
    df = df.drop(columns=["diagnosis"])
    feature_names = df.columns.tolist()
    X_raw = df.values.astype(np.float64)
    return X_raw, y, feature_names


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=os.path.join(ROOT, "BreastCancerDataset.csv"),
                    help="path to BreastCancerDataset.csv")
    p.add_argument("--k", type=int, default=15,
                    help="IsUMap's own distance_graph_generation neighbourhood size "
                         "(smaller than MNIST's default 30 -- dataset only has n=569)")
    p.add_argument("--emb-k", type=int, default=10,
                    help="n_neighbors for our own randers_umap_fit's UMAP-style graph")
    p.add_argument("--neg", type=int, default=10)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--snapshot-every", type=int, default=None,
                    help="if given, also save <out>_snapshots.png: the embedding every "
                         "N epochs (from init to final), side by side.")
    p.add_argument("--gravity", action="store_true",
                    help="add per-node gravity toward xi_i=y_i+b_i (Bannister et al. "
                         "f_g=gamma*M[i]*b_i), weighted by --gravity-neighbor-weight "
                         "unless disabled.")
    p.add_argument("--gravity-strength", type=float, default=1.0,
                    help="gamma in Bannister et al.'s gravity force. Only matters with "
                         "--gravity.")
    p.add_argument("--no-gravity-neighbor-weight", action="store_true",
                    help="disable the neighbour-plausibility weighting (revert to the "
                         "old unconditional gravity pull). Only matters with --gravity.")
    p.add_argument("--ramp", action="store_true",
                    help="ramp drift's magnitude 0->1 over epochs instead of applying it "
                         "at full strength from epoch 0 (off by default, matching the "
                         "MNIST scripts' own --ramp convention).")
    p.add_argument("--force-model", choices=["fr_gravity", "umap"], default="fr_gravity",
                    help="[OURS 2026-09-02] attraction/repulsion law passed to randers_umap_fit "
                         "-- 'fr_gravity' (default) = Bannister et al.'s spring/inverse-square "
                         "law, 'umap' = UMAP's own fitted (a,b)-curve.")
    p.add_argument("--fr-k", type=float, default=None,
                    help="[OURS 2026-09-02] natural edge-length constant for force_model="
                         "fr_gravity (default None -> 1/sqrt(n)). Ignored for force_model=umap.")
    p.add_argument("--neg-sampling", action="store_true",
                    help="[OURS 2026-09-02] use TRUE stochastic negative sampling for repulsion "
                         "instead of the dense/exact sum.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="breastcancer_embedding")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    verbose = not args.quiet

    save_dir = os.path.join(HERE, "")
    os.makedirs(save_dir, exist_ok=True)

    # ---- load + scale -------------------------------------------------------
    if verbose:
        print(f"Loading {args.csv} ...")
    X_raw, y, feature_names = load_breast_cancer_csv(args.csv)
    n = X_raw.shape[0]
    if verbose:
        print(f"X_raw: {X_raw.shape}  ({X_raw.shape[1]} features, no PCA)  "
              f"malignant={y.sum()}  benign={(1 - y).sum()}")

    # [OURS 2026-08-26] StandardScaler -- see module docstring's "Feature
    # scaling" section for why this is required here but was not needed for
    # MNIST's already-uniform [0,1] pixel features.
    X = StandardScaler().fit_transform(X_raw)
    if verbose:
        print("Applied StandardScaler (zero mean, unit variance per feature).")

    # ---- IsUMap's own local, pre-symmetrization asymmetric distance -------
    isumap_dist = distance_graph_generation(
        X, k=args.k, normalize=True, distBeyondNN=True, verbose=verbose,
        dataIsDistMatrix=False, dataIsGeodesicDistMatrix=False, saveDistMatrix=False,
    )
    asymm_distance = isumap_dist[0]

    # [OURS 2026-08-26] same i==j fix as MNIST/embed_MNIST_{pca,raw}.py --
    # see those files' own comments for the full bug writeup (R's key
    # (i,j,k) = "distance from j to k in neighbourhood i"; the real direct
    # i-to-neighbour distance only appears when i==j, third element is the
    # real target).
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

    is_symmetric = np.allclose(D_asym, D_asym.T)
    if verbose:
        n_inf = (~np.isfinite(D_asym)).sum()
        print(f"D_asym: {D_asym.shape}  symmetric={is_symmetric}  (should be False)  "
              f"unreachable pairs={n_inf}")

    np.save(os.path.join(save_dir, "asymm_matrix_breastcancer.npy"), D_asym)
    np.save(os.path.join(save_dir, "labels_breastcancer.npy"), y)

    # ---- embed with our own randers_umap_fit -------------------------------
    with np.errstate(invalid="ignore", divide="ignore"):
        out = randers_umap_fit(D_asym, n_neighbors=args.emb_k, n_negative_samples=args.neg,
                                n_epochs=args.epochs, use_drift=True,
                                snapshot_every=args.snapshot_every,
                                use_gravity=args.gravity,
                                gravity_strength=args.gravity_strength,
                                gravity_neighbor_weight=not args.no_gravity_neighbor_weight,
                                ramp=args.ramp,
                                force_model=args.force_model, fr_k=args.fr_k,
                                negative_sampling=args.neg_sampling,
                                seed=args.seed, verbose=verbose)
    Y, B = out["Y"], out["B"]

    cmap2 = ListedColormap(["tab:green", "tab:red"])  # 0=benign, 1=malignant
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color="tab:green", label="Benign (B)"),
        Line2D([0], [0], marker="o", linestyle="", color="tab:red", label="Malignant (M)"),
    ]

    # ---- plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(Y[:, 0], Y[:, 1], c=y, cmap=cmap2, s=14, alpha=0.85, linewidths=0)
    ax.legend(handles=legend_handles, loc="best")

    bn = np.linalg.norm(B, axis=1)
    big = np.argsort(bn)[::-1][:25]
    if bn.max() > 0:
        sc_scale = 0.12 * (Y.max() - Y.min()) / bn.max()
        ax.quiver(Y[big, 0], Y[big, 1], B[big, 0] * sc_scale, B[big, 1] * sc_scale,
                  color="k", alpha=0.6, width=0.004, scale=1, scale_units="xy")

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Randers-UMAP on Breast Cancer dataset (30D, StandardScaled, n={n}, epochs={args.epochs})",
                 fontsize=10)
    fig.tight_layout()
    out_path = os.path.join(save_dir, f"{args.out}.png")
    fig.savefig(out_path, dpi=150)

    np.savez(os.path.join(save_dir, f"{args.out}.npz"), Y=Y, B=B, labels=y,
             feature_names=feature_names)

    if verbose:
        print(f"\nwrote {out_path} and {args.out}.npz (in {save_dir})")

    # ---- snapshot grid: init -> every N epochs -> final, side by side -----
    if args.snapshot_every is not None:
        snaps = out["snapshots"]
        n_snap = len(snaps)
        ncols = min(n_snap, 6)
        nrows = int(np.ceil(n_snap / ncols))
        fig2, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows), squeeze=False)
        for idx, snap in enumerate(snaps):
            ax2 = axes[idx // ncols][idx % ncols]
            Yi, Bi = snap["Y"], snap["B"]
            ax2.scatter(Yi[:, 0], Yi[:, 1], c=y, cmap=cmap2, s=10, alpha=0.85, linewidths=0)
            bni = np.linalg.norm(Bi, axis=1)
            bigi = np.argsort(bni)[::-1][:25]
            if bni.max() > 0:
                sc_scale_i = 0.12 * (Yi.max() - Yi.min()) / bni.max()
                ax2.quiver(Yi[bigi, 0], Yi[bigi, 1], Bi[bigi, 0] * sc_scale_i, Bi[bigi, 1] * sc_scale_i,
                          color="k", alpha=0.6, width=0.006, scale=1, scale_units="xy")
            ax2.set_title(f"epoch {snap['epoch']}", fontsize=9)
            ax2.set_xticks([]); ax2.set_yticks([])
        for idx in range(n_snap, nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")

        fig2.suptitle(f"Randers-UMAP on Breast Cancer dataset, training trajectory "
                      f"(n={n}, snapshot_every={args.snapshot_every})", fontsize=11)
        fig2.legend(handles=legend_handles, loc="lower center", ncol=2)
        snap_path = os.path.join(save_dir, f"{args.out}_snapshots.png")
        fig2.savefig(snap_path, dpi=150)
        if verbose:
            print(f"wrote {snap_path}")


if __name__ == "__main__":
    main()
