#!/usr/bin/env python3
"""
embed_scRNA.py -- applies our IsUMap + Randers-UMAP pipeline (same core
mechanism as MNIST/embed_MNIST_pca.py and BreastCancer/embed_BreastCancer.py:
IsUMap's own distance_graph_generation for the asymmetric distance, our own
randers_umap_fit for the embedding, live drift B_fixed=None/use_drift=True)
to IsUMap's own bundled scRNA-seq example dataset (colorectal cancer mouse
model, "CRCC_AKPE_scRNASeq_2024", from
https://github.com/LUK4S-B/IsUMap/tree/main/Dataset_files/
scRNA_dataset-1_CRCC_AKPE_scRNASeq_2024).

[OURS 2026-08-28, per explicit user request -- "bu data setini methodumuza
uygulayabilir miyiz"]

Dataset
-------
Two Seurat-exported CSVs are used (the gene-count files smoc2_counts.csv/
smoc2_data.csv are NOT used here -- they only contain a single gene's
(SMOC2) expression value per cell, not a full count matrix, so they cannot
serve as a general high-dimensional feature input):
  - sct_pca_embeddings.csv : (n_cells, 50) SCTransform+PCA embedding,
    ALREADY dimensionality-reduced by Seurat -- used directly as X, exactly
    the same role PCA plays in embed_MNIST_pca.py's "option 2" (PCA first,
    then IsUMap's own asymmetric distance on the reduced space), except here
    the PCA was computed upstream by Seurat, not by us.
  - sct_cluster_labels.csv : (n_cells,) Seurat SNN cluster id (resolution
    0.3, 7 clusters, ids 0-6) -- used ONLY for colouring the final scatter
    plot (ground-truth-ish structure to visually check against), never fed
    into the embedding itself.
  Real full dataset size (verified 2026-08-28): 11,505 cells across 2
  samples ("sample1_..."/"sample2_..." barcode prefixes), 50 PCs, cluster
  sizes [0:4086, 1:2787, 2:1477, 3:1186, 4:1116, 5:692, 6:161].

Why no StandardScaler (unlike BreastCancer): the 50 columns here are PCA
components, already ordered by explained variance -- rescaling them to unit
variance each would erase that ordering/importance signal, which is exactly
why embed_MNIST_pca.py doesn't rescale its own PCA output either. Contrast
with BreastCancer's 30 RAW, differently-scaled physical measurements, where
StandardScaler is necessary.

Why the barcode JOIN, not a positional merge: verified directly (via
browser fetch of both raw CSVs) that sct_pca_embeddings.csv and
sct_cluster_labels.csv do NOT share row order -- naively zipping the two
files position-by-position would silently mismatch every cell's PCA vector
with the WRONG cluster label. load_scrna_csv() below joins on the barcode
string (each file's own row index) instead.

Local smoke-test files: sct_pca_embeddings_sample.csv/
sct_cluster_labels_sample.csv (also in this folder) are a deterministic
100-cell random subsample of the two real files above (seed=42, same
barcode-matched pair), fetched directly from GitHub -- included so this
script can be smoke-tested without needing a live network connection at
run time. For a real run, download the two full CSVs from the GitHub path
above into this folder (or point --pca-csv/--cluster-csv elsewhere) and use
--n to subsample down to something tractable (see --n's own help for why).

Usage
-----
    python3 scRNA/embed_scRNA.py --n 100 --epochs 300 \\
        --pca-csv scRNA/sct_pca_embeddings_sample.csv \\
        --cluster-csv scRNA/sct_cluster_labels_sample.csv   # smoke test
    python3 scRNA/embed_scRNA.py --n 3000 --epochs 500       # real run,
        # once the full sct_pca_embeddings.csv/sct_cluster_labels.csv are
        # downloaded into this folder
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

from distance_graph_generation import distance_graph_generation
from randers_umap import randers_umap_fit


def load_scrna_csv(pca_path, cluster_path):
    """
    Loads the two Seurat-exported CSVs and joins them on the barcode
    (each file's own row index) -- NOT a positional merge, since the two
    files are not in the same row order (verified directly, see module
    docstring). Returns:
        X       : (n, 50) float64 PCA embedding, in barcode-sorted order
        y       : (n,) int64 Seurat SNN cluster id (0-6)
        barcodes: list of the n barcodes, in the same order as X/y
    """
    pca_df = pd.read_csv(pca_path, index_col=0)
    clu_df = pd.read_csv(cluster_path, index_col=0)
    clu_col = clu_df.columns[0]  # "SCT_snn_res.0.3"

    common = pca_df.index.intersection(clu_df.index)
    if len(common) == 0:
        raise ValueError("No shared barcodes between --pca-csv and --cluster-csv -- "
                          "check that both files are from the same dataset export.")
    common = sorted(common)
    pca_df = pca_df.loc[common]
    clu_df = clu_df.loc[common]

    X = pca_df.values.astype(np.float64)
    y = clu_df[clu_col].astype(np.int64).values
    return X, y, list(common)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pca-csv", default=os.path.join(HERE, "sct_pca_embeddings.csv"),
                    help="path to sct_pca_embeddings.csv (50-dim Seurat PCA). Falls back "
                         "to sct_pca_embeddings_sample.csv (100-cell local smoke-test "
                         "subsample, bundled in this folder) if the full file isn't found.")
    p.add_argument("--cluster-csv", default=os.path.join(HERE, "sct_cluster_labels.csv"),
                    help="path to sct_cluster_labels.csv (Seurat SNN cluster id, for "
                         "colouring only). Falls back to sct_cluster_labels_sample.csv "
                         "the same way --pca-csv does.")
    p.add_argument("--n", type=int, default=2000,
                    help="[OURS 2026-08-28] subsample this many cells (random, seeded by "
                         "--seed) before running the pipeline. The real dataset has 11,505 "
                         "cells -- randers_umap_fit's own dense (n,n,d) drift/gradient "
                         "arrays (see compute_drift's docstring in randers_umap.py) scale "
                         "as O(n^2), so n=11505 would need several GB of memory just for "
                         "those intermediate arrays; 2000-3000 is a reasonable default for "
                         "a single-machine run. Pass --n -1 to use every cell (only advisable "
                         "with a machine that has enough memory headroom).")
    p.add_argument("--k", type=int, default=30,
                    help="IsUMap's own distance_graph_generation neighbourhood size "
                         "(matches the MNIST scripts' default)")
    p.add_argument("--emb-k", type=int, default=20,
                    help="n_neighbors for our own randers_umap_fit's UMAP-style graph")
    p.add_argument("--neg", type=int, default=10)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--snapshot-every", type=int, default=None,
                    help="if given, also save <out>_snapshots.png: the embedding every "
                         "N epochs (from init to final), side by side.")
    p.add_argument("--gravity", action="store_true",
                    help="add per-node gravity toward xi_i=y_i+b_i (Bannister et al. "
                         "f_g=gamma*M[i]*b_i), weighted by --gravity-neighbor-weight "
                         "unless disabled. Works here since B is live (use_drift=True), "
                         "exactly as in the MNIST/BreastCancer scripts.")
    p.add_argument("--gravity-strength", type=float, default=1.0,
                    help="gamma in Bannister et al.'s gravity force. Only matters with "
                         "--gravity.")
    p.add_argument("--no-gravity-neighbor-weight", action="store_true",
                    help="disable the neighbour-plausibility weighting (revert to the "
                         "old unconditional gravity pull). Only matters with --gravity.")
    p.add_argument("--ramp", action="store_true",
                    help="ramp drift's magnitude 0->1 over epochs instead of applying it "
                         "at full strength from epoch 0 (off by default, matching the "
                         "other run_*.py/embed_*.py scripts' own --ramp convention).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="scrna_embedding")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    verbose = not args.quiet

    save_dir = HERE
    os.makedirs(save_dir, exist_ok=True)

    # ---- resolve CSV paths, falling back to the bundled small sample -----
    pca_csv = args.pca_csv
    cluster_csv = args.cluster_csv
    if not os.path.exists(pca_csv):
        fallback = os.path.join(HERE, "sct_pca_embeddings_sample.csv")
        if verbose:
            print(f"{pca_csv} not found -- falling back to bundled smoke-test sample "
                  f"{fallback} (100 cells only; download the full CSV from IsUMap's repo "
                  f"for a real run).")
        pca_csv = fallback
    if not os.path.exists(cluster_csv):
        fallback = os.path.join(HERE, "sct_cluster_labels_sample.csv")
        if verbose:
            print(f"{cluster_csv} not found -- falling back to bundled smoke-test sample "
                  f"{fallback}.")
        cluster_csv = fallback

    # ---- load + join on barcode --------------------------------------------
    if verbose:
        print(f"Loading {pca_csv} and {cluster_csv} ...")
    X_full, y_full, barcodes_full = load_scrna_csv(pca_csv, cluster_csv)
    if verbose:
        print(f"X_full: {X_full.shape}  (50 Seurat PCA dims, already reduced)  "
              f"n_clusters={len(np.unique(y_full))}")

    # ---- subsample ----------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    if args.n is not None and args.n > 0 and args.n < X_full.shape[0]:
        idx = rng.choice(X_full.shape[0], size=args.n, replace=False)
        idx.sort()
        X, y = X_full[idx], y_full[idx]
        barcodes = [barcodes_full[i] for i in idx]
    else:
        X, y, barcodes = X_full, y_full, barcodes_full
    n = X.shape[0]
    if verbose:
        print(f"Using n={n} cells "
              f"({'full dataset' if n == X_full.shape[0] else f'subsampled from {X_full.shape[0]}'})")

    # ---- IsUMap's own local, pre-symmetrization asymmetric distance -------
    isumap_dist = distance_graph_generation(
        X, k=args.k, normalize=True, distBeyondNN=True, verbose=verbose,
        dataIsDistMatrix=False, dataIsGeodesicDistMatrix=False, saveDistMatrix=False,
    )
    asymm_distance = isumap_dist[0]

    # [OURS 2026-08-28] same i==j fix as MNIST/embed_MNIST_{pca,raw}.py and
    # BreastCancer/embed_BreastCancer.py -- comp_graph()'s key (i,j,k) means
    # "distance from j to k, as measured in neighbourhood i"; the real,
    # directly-measured i-to-neighbour distance only appears when i==j.
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

    np.save(os.path.join(save_dir, "asymm_matrix_scrna.npy"), D_asym)
    np.save(os.path.join(save_dir, "labels_scrna.npy"), y)

    # ---- embed with our own randers_umap_fit -------------------------------
    # Live drift, B_fixed=None -- same mechanism as MNIST/BreastCancer, and
    # per the 2026-08-28 discussion this is the CURRENT, explicitly-chosen
    # design for real (non-synthetic) datasets in this project, not a
    # frozen/located B (that hybrid was tried in the isumap scripts and
    # explicitly reverted -- see run_swiss_roll_isumap.py's own comments).
    with np.errstate(invalid="ignore", divide="ignore"):
        out = randers_umap_fit(D_asym, n_neighbors=args.emb_k, n_negative_samples=args.neg,
                                n_epochs=args.epochs, use_drift=True, B_fixed=None,
                                snapshot_every=args.snapshot_every,
                                use_gravity=args.gravity,
                                gravity_strength=args.gravity_strength,
                                gravity_neighbor_weight=not args.no_gravity_neighbor_weight,
                                ramp=args.ramp,
                                seed=args.seed, verbose=verbose)
    Y, B = out["Y"], out["B"]

    n_clusters = int(y.max()) + 1

    # ---- plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(Y[:, 0], Y[:, 1], c=y, cmap="tab10", s=10, alpha=0.85, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Seurat SNN cluster", ticks=range(n_clusters))

    bn = np.linalg.norm(B, axis=1)
    big = np.argsort(bn)[::-1][:25]
    if bn.max() > 0:
        sc_scale = 0.12 * (Y.max() - Y.min()) / bn.max()
        ax.quiver(Y[big, 0], Y[big, 1], B[big, 0] * sc_scale, B[big, 1] * sc_scale,
                  color="k", alpha=0.6, width=0.004, scale=1, scale_units="xy")

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Randers-UMAP on scRNA (CRCC_AKPE, 50D Seurat PCA, n={n}, epochs={args.epochs})", fontsize=10)
    fig.tight_layout()
    out_path = os.path.join(save_dir, f"{args.out}.png")
    fig.savefig(out_path, dpi=150)

    np.savez(os.path.join(save_dir, f"{args.out}.npz"), Y=Y, B=B, labels=y, barcodes=barcodes)

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
            ax2.scatter(Yi[:, 0], Yi[:, 1], c=y, cmap="tab10", s=8, alpha=0.85, linewidths=0)
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

        fig2.suptitle(f"Randers-UMAP on scRNA, training trajectory "
                      f"(n={n}, snapshot_every={args.snapshot_every})", fontsize=11)
        snap_path = os.path.join(save_dir, f"{args.out}_snapshots.png")
        fig2.savefig(snap_path, dpi=150)
        if verbose:
            print(f"wrote {snap_path}")


if __name__ == "__main__":
    main()
