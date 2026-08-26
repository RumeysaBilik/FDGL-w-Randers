#!/usr/bin/env python3
"""
embed_MNIST_pca.py -- the "PCA-first" variant of MNIST + IsUMap + our own
Randers-UMAP pipeline, per the advisor's second suggested approach ([OURS
2026-08-26] "hocam demisti ki MNIST'i uygulamanin iki yolu var: 1) direkt
28x28'lik dimensiondan baslayip metodumuzu uygulamak, 2) once lower
dimension'a cekip sonra metodumuzu uygulamak" -- this file implements
option 2).

This is a single, self-contained script combining what asymm_dist_MNIST.py
+ embed_MNIST.py do in two steps, with ONE new step inserted between them:
a linear PCA projection from the raw 784-dim pixel space down to
--pca-dim dimensions, BEFORE IsUMap's own distance-graph construction ever
sees the data. Everything else -- which asymmetry mechanism is used
(IsUMap's own local, pre-symmetrization neighbourhood metric, exactly as
in asymm_dist_MNIST.py/distance_graph_generation.py), and which embedding
method reads that D_asym (our own randers_umap_fit, exactly as in
embed_MNIST.py) -- is UNCHANGED, so this is a controlled comparison against
the existing 784-dim pipeline: only the ambient dimensionality the
asymmetry is computed FROM differs, not the mechanism itself.

Why PCA first, not the label-informed 784-dim drift (the advisor's other
option): see project discussion -- t-SNE-style methods traditionally
PCA-reduce first (van der Maaten's own recommendation, ~50 dims) precisely
because raw high-dimensional pixel-space distances are a noisy proxy for
perceptual similarity and because shortest-path/geodesic construction (as
used here, both by IsUMap's own local metric AND by our own
compute_dist_matrix elsewhere in this project) is more sensitive to noise
in very high ambient dimensions. This is the safer, lower-risk first
experiment; the label/class-centroid-drift, direct-784-dim variant is a
separate, more ambitious follow-up (not implemented here).

Usage
-----
    python3 MNIST/embed_MNIST_pca.py
    python3 MNIST/embed_MNIST_pca.py --n 5000 --pca-dim 50 --epochs 500
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# [OURS 2026-08-26] both the MNIST/ folder (this script's own dir) and the
# FDGL root (where data_and_plots.py, distance_graph_generation.py and
# randers_umap.py actually live) need to be importable -- the existing
# asymm_dist_MNIST.py only added HERE (MNIST/) to sys.path, which does NOT
# resolve those root-level modules and fails with "No module named
# 'data_and_plots'" when run from anywhere except a shell already cd'd into
# a location that happens to have ROOT on its path. Adding both here makes
# this script runnable regardless of cwd.
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from data_and_plots import load_MNIST
from distance_graph_generation import distance_graph_generation
from randers_umap import randers_umap_fit


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=5000,
                    help="number of MNIST points to sample (matches asymm_dist_MNIST.py's default)")
    p.add_argument("--pca-dim", type=int, default=50,
                    help="[OURS 2026-08-26] target dimension for the PCA pre-reduction step, "
                         "784 -> this. 50 is van der Maaten's own classic t-SNE-preprocessing "
                         "recommendation; adjust and compare explained-variance printed below.")
    p.add_argument("--k", type=int, default=30,
                    help="IsUMap's own distance_graph_generation neighbourhood size "
                         "(matches asymm_dist_MNIST.py's default)")
    p.add_argument("--emb-k", type=int, default=20,
                    help="n_neighbors for our own randers_umap_fit's UMAP-style graph")
    p.add_argument("--neg", type=int, default=10)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--snapshot-every", type=int, default=None,
                    help="[OURS 2026-08-26] if given, also save <out>_snapshots.png: the "
                         "embedding every N epochs (from init to final), side by side -- "
                         "same mechanism as run_swiss_roll.py's --snapshot-every.")
    p.add_argument("--gravity", action="store_true",
                    help="[OURS 2026-08-26] add per-node gravity toward xi_i=y_i+b_i "
                         "(Bannister et al. f_g=gamma*M[i]*b_i), weighted by "
                         "--gravity-neighbor-weight unless disabled. Same mechanism as "
                         "run_swiss_roll.py's --gravity -- works here too since B is live "
                         "(use_drift=True), not just when B is a frozen locate result.")
    p.add_argument("--gravity-strength", type=float, default=1.0,
                    help="[OURS 2026-08-26] gamma in Bannister et al.'s gravity force. "
                         "Only matters with --gravity.")
    p.add_argument("--no-gravity-neighbor-weight", action="store_true",
                    help="[OURS 2026-08-26] disable the neighbour-plausibility weighting "
                         "(revert to the old unconditional gravity pull). Only matters with "
                         "--gravity.")
    p.add_argument("--ramp", action="store_true",
                    help="[OURS 2026-08-26] ramp drift's magnitude 0->1 over epochs instead "
                         "of applying it at full strength from epoch 0 (default here, "
                         "matching run_swiss_roll.py's own --ramp convention -- off by "
                         "default). When on: drift held at exactly 0 for the first 30%% of "
                         "epochs, linearly ramped 0->1 over the next 40%%, full strength for "
                         "the last 30%% (randers_umap_fit's own schedule). Without this flag, "
                         "randers_umap_fit's own internal default (ramp=True) would otherwise "
                         "apply silently -- passing ramp=args.ramp here makes it explicit and "
                         "off by default, consistent with the other run_*.py scripts.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="mnist_pca_embedding")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    verbose = not args.quiet

    save_dir = os.path.join(HERE, "")
    os.makedirs(save_dir, exist_ok=True)

    # ---- load: raw 784-dim MNIST ------------------------------------------
    if verbose:
        print(f"Loading MNIST (n={args.n})...")
    dataset_path = os.path.join(ROOT, "Dataset_files") + os.sep
    X, y = load_MNIST(args.n, datasetPath=dataset_path)
    if verbose:
        print(f"X: {X.shape}  (raw pixel dimension = {X.shape[1]})")

    # ---- PCA: 784 -> --pca-dim, BEFORE IsUMap ever sees the data ----------
    # [OURS 2026-08-26, per explicit user request -- advisor's "option 2"]
    # this is the ONLY structural difference from asymm_dist_MNIST.py: the
    # rest of the pipeline (IsUMap's own distance_graph_generation, then our
    # own randers_umap_fit) is byte-for-byte the same mechanism, just fed
    # lower-dimensional input.
    pca_dim = min(args.pca_dim, X.shape[0], X.shape[1])
    pca = PCA(n_components=pca_dim, random_state=args.seed)
    X_pca = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_.sum()
    if verbose:
        print(f"PCA: {X.shape[1]} -> {pca_dim} dims, "
              f"explained variance retained = {explained:.4f}")

    # ---- IsUMap's own local, pre-symmetrization asymmetric distance -------
    isumap_dist = distance_graph_generation(
        X_pca, k=args.k, normalize=True, distBeyondNN=True, verbose=verbose,
        dataIsDistMatrix=False, dataIsGeodesicDistMatrix=False, saveDistMatrix=False,
    )
    asymm_distance = isumap_dist[0]

    n = X_pca.shape[0]

    # [OURS 2026-08-26, per explicit user request -- bug found while
    # diagnosing a structureless embedding] R's key (i,j,k) means "distance
    # from j to k, as measured in neighbourhood i" (distance_graph_
    # generation.py's own comp_graph(), line ~147). The DIRECT, real
    # measured distance from point i to one of its own neighbours only
    # appears when the key's SECOND element equals the FIRST (i==j) --
    # comp_graph() explicitly stores it there: "if j==i: R[(i,j,k)] =
    # knn_distances[i][ind_k]". The old code here (inherited from fdg's
    # asymm_dist_MNIST.py) wrote asymm_matrix[i,j]=value for EVERY entry
    # regardless of this -- for i==j entries that lands on the diagonal
    # (self-distance, useless) and OVERWRITES itself for every one of i's
    # neighbours since j==i is fixed; for i!=j entries it stores an
    # unrelated triangulated j<->k distance at position [i,j], not [i,k].
    # Net effect verified empirically: the resulting matrix's "10 nearest
    # neighbours" per row had 0% overlap with the TRUE nearest neighbours
    # in the data -- the neighbour graph randers_umap_fit trained on was
    # essentially noise, which is why the embedding showed no digit
    # clusters no matter how many epochs were run. Fixed here: only take
    # the i==j entries (the real, direct i-to-neighbour distances), write
    # them to D_sparse[i, k] (the THIRD element is the actual target).
    D_sparse = np.full((n, n), np.inf)
    np.fill_diagonal(D_sparse, 0.0)
    for (i, j, kk), value in asymm_distance.items():
        if i == j:
            D_sparse[i, kk] = value

    # D_sparse is still sparse (~k real entries per row, rest inf) -- most
    # pairs were simply never measured. Complete it into a fully finite,
    # connected DIRECTED geodesic matrix via Dijkstra, exactly like our own
    # randers_bridge.compute_dist_matrix does for the synthetic-manifold
    # scripts, so randers_umap_fit's own dense k-NN search (which assumes
    # every entry is a meaningful finite number) has real data everywhere.
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

    np.save(os.path.join(save_dir, "asymm_matrix_pca.npy"), D_asym)
    np.save(os.path.join(save_dir, "labels_pca.npy"), y)
    np.save(os.path.join(save_dir, "X_pca.npy"), X_pca)

    # ---- embed with our own randers_umap_fit -------------------------------
    # Identical call to embed_MNIST.py: use_drift=True, B_fixed=None -> B is
    # driven live by compute_drift()'s own bounded N-coefficient mechanism
    # (norm_mode="relative" default), exactly as in the existing 784-dim run.
    # [OURS 2026-08-26] some pairs remain unreachable (inf) even after the
    # directed Dijkstra completion above -- an inherent property of a
    # directed k-NN graph, not a bug. randers_umap_fit's own N computation
    # already handles this safely (np.where(isfinite(N), N, 0.0) zeroes out
    # undefined pairs -- see its own comment), but numpy still prints a
    # RuntimeWarning for the inf-inf/inf/inf arithmetic that produces those
    # NaNs before they get zeroed. Suppressed here (verified harmless via
    # direct testing) purely to keep the console output readable.
    with np.errstate(invalid="ignore", divide="ignore"):
        out = randers_umap_fit(D_asym, n_neighbors=args.emb_k, n_negative_samples=args.neg,
                                n_epochs=args.epochs, use_drift=True,
                                snapshot_every=args.snapshot_every,
                                use_gravity=args.gravity,
                                gravity_strength=args.gravity_strength,
                                gravity_neighbor_weight=not args.no_gravity_neighbor_weight,
                                ramp=args.ramp,
                                seed=args.seed, verbose=verbose)
    Y, B = out["Y"], out["B"]

    # ---- plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(Y[:, 0], Y[:, 1], c=y, cmap="tab10", s=6, alpha=0.85, linewidths=0)
    plt.colorbar(sc, ax=ax, label="digit", ticks=range(10))

    bn = np.linalg.norm(B, axis=1)
    big = np.argsort(bn)[::-1][:25]
    if bn.max() > 0:
        sc_scale = 0.12 * (Y.max() - Y.min()) / bn.max()
        ax.quiver(Y[big, 0], Y[big, 1], B[big, 0] * sc_scale, B[big, 1] * sc_scale,
                  color="k", alpha=0.6, width=0.004, scale=1, scale_units="xy")

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Randers-UMAP on MNIST (PCA {X.shape[1]}->{pca_dim}D first, "
                 f"n={n}, explained var={explained:.3f})", fontsize=10)
    fig.tight_layout()
    out_path = os.path.join(save_dir, f"{args.out}.png")
    fig.savefig(out_path, dpi=150)

    np.savez(os.path.join(save_dir, f"{args.out}.npz"), Y=Y, B=B, labels=y,
             pca_dim=pca_dim, explained_variance=explained)

    if verbose:
        print(f"\nwrote {out_path} and {args.out}.npz (in {save_dir})")

    # ---- snapshot grid: init -> every N epochs -> final, side by side -----
    # [OURS 2026-08-26, per explicit user request -- "sureci gormek istiyorum"]
    # same mechanism as run_swiss_roll.py's own --snapshot-every.
    if args.snapshot_every is not None:
        snaps = out["snapshots"]
        n_snap = len(snaps)
        ncols = min(n_snap, 6)
        nrows = int(np.ceil(n_snap / ncols))
        fig2, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows), squeeze=False)
        sc2 = None
        for idx, snap in enumerate(snaps):
            ax2 = axes[idx // ncols][idx % ncols]
            Yi, Bi = snap["Y"], snap["B"]
            sc2 = ax2.scatter(Yi[:, 0], Yi[:, 1], c=y, cmap="tab10", s=6,
                              alpha=0.85, linewidths=0, vmin=0, vmax=9)
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

        fig2.suptitle(f"Randers-UMAP on MNIST (PCA {X.shape[1]}->{pca_dim}D), training trajectory "
                      f"(n={n}, snapshot_every={args.snapshot_every})", fontsize=11)
        if sc2 is not None:
            fig2.colorbar(sc2, ax=axes, label="digit", ticks=range(10), shrink=0.6)
        snap_path = os.path.join(save_dir, f"{args.out}_snapshots.png")
        fig2.savefig(snap_path, dpi=150)
        if verbose:
            print(f"wrote {snap_path}")


if __name__ == "__main__":
    main()
