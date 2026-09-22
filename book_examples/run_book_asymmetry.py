#!/usr/bin/env python3
"""
run_book_asymmetry.py -- feed the two real asymmetric proximity datasets from
Borg & Groenen, "Modern Multidimensional Scaling" (2nd ed.), Chapter 23, into
our own Randers force-directed layout pipeline (fdgl_pipeline).

Datasets (see book_examples/*.csv, transcribed from the book's printed tables)
------------------------------------------------------------------------------
    morse : Table 4.2 (Rothkopf, 1957). 36x36 Morse code confusion
            percentages -- P[i, j] = % of trials where signal i was reported
            as signal j. This is a SIMILARITY (higher = more often confused,
            i.e. more alike), not a distance.
    cola  : Table 23.1. 15x15 brand-switching counts among cola soft drinks
            (Bell & Lattin, 1998) -- P[i, j] = number of households that
            switched from cola i to cola j. Also a SIMILARITY (higher =
            switched more often = more alike).

Both matrices are asymmetric (P[i, j] != P[j, i] in general) -- that
asymmetry is the whole point of the book chapter, and is what our own
compute_highdim_drift()/compute_drift() machinery is designed to turn into a
per-point drift field.

Why we need a similarity -> dissimilarity conversion
-----------------------------------------------------
Our pipeline (classical_mds, compute_highdim_drift, fdgl_pipeline) expects a
DISSIMILARITY convention throughout: larger value = further apart / less
alike, and 0 on the diagonal. The book's raw tables are the opposite
(similarities: larger value = more alike). We convert with

    D[i, j] = P.max() - P[i, j],   then D[i, i] is forced to 0.

This is an increasing-affine (order-reversing) transform of P, so it
preserves *which* pairs are more/less alike, just flips the scale to a
dissimilarity. One direct consequence: the SIGN of the resulting asymmetry
is flipped relative to the raw similarity matrix, because
    D[i, j] - D[j, i] = (max - P[i, j]) - (max - P[j, i]) = -(P[i, j] - P[j, i]).
That sign flip is expected and not a bug -- it just means "i is reported as
j more often than j is reported as i" (P[i,j] > P[j,i]) becomes "the
distance i->j is smaller than j->i" (D[i,j] < D[j,i]) under the
dissimilarity convention, which is the natural reading (more confusable =
closer).

Two separate conversions are used below:
    D_sym  : from the SYMMETRIC part M = (P + P.T) / 2, fed to classical_mds
             to place the n points in an ambient coordinate space X (we need
             actual coordinates for compute_highdim_drift's direction
             vectors e_ij(X); the book itself never needs this step, since
             its own models fit coordinates and asymmetry jointly or in
             sequence with different machinery).
    D_asym : from the full (asymmetric) P itself, fed to compute_highdim_drift
             as the D_asym whose row/column asymmetry becomes the per-point
             drift field omega.

Usage
-----
    python run_book_asymmetry.py --dataset morse
    python run_book_asymmetry.py --dataset cola --k 6
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))       # .../FDGL/book_examples
FDGL_ROOT = os.path.dirname(HERE)                        # .../FDGL
sys.path.insert(0, FDGL_ROOT)

from randers_fdgl import classical_mds, arrow_scale, plot_caption
from randers_bridge import fdgl_pipeline, compute_highdim_drift, compute_dist_matrix


def load_dataset(name, data_dir):
    path = os.path.join(data_dir, f"{'morse_confusion' if name == 'morse' else 'cola_brandswitch'}.csv")
    import csv
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    labels = rows[0][1:]
    P = np.array([[float(x) for x in row[1:]] for row in rows[1:]], dtype=float)
    return P, labels


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["morse", "cola"], default="morse")
    p.add_argument("--data-dir", default=None,
                    help="directory containing <dataset>_*.csv (default: "
                         "book_examples/ next to this script, or alongside "
                         "it if run from inside that folder)")
    p.add_argument("--ambient-dim", type=int, default=8,
                    help="dimension of the ambient coordinate space X, "
                         "reconstructed via classical_mds from the "
                         "symmetric part of the data.")
    p.add_argument("--k", type=int, default=None,
                    help="shared k for the ambient D_asym's own knn (drift "
                         "derivation) and fdgl_pipeline's own k-NN backbone. "
                         "Default: min(8, n-2).")
    p.add_argument("--neg", type=int, default=10)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--clip-delta", type=float, default=0.01)
    p.add_argument("--adjacency", choices=["threshold", "knn"], default="knn")
    p.add_argument("--proj-dim", type=int, default=2, choices=[2, 3])
    p.add_argument("--snapshot-every", type=int, default=None,
                    help="if given, also save <out>_snapshots.png: the "
                         "embedding every N epochs (from init to final), "
                         "side by side.")
    p.add_argument("--live-drift", action="store_true")
    p.add_argument("--init-only", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    verbose = not args.quiet

    data_dir = args.data_dir or HERE
    out_name = args.out or f"{args.dataset}_drift_embedding"

    P, labels = load_dataset(args.dataset, data_dir)
    n = P.shape[0]
    k = args.k if args.k is not None else min(8, n - 2)

    if verbose:
        print(f"Loaded {args.dataset}: P shape {P.shape}, "
              f"symmetric={np.allclose(P, P.T)}, k={k}")

    # similarity -> dissimilarity, see module docstring for the exact
    # rationale and the resulting sign flip in the asymmetry.
    D_raw = P.max() - P
    np.fill_diagonal(D_raw, 0.0)

    # Route the raw dissimilarity matrix through our own project's
    # compute_dist_matrix (randers_bridge.py) instead of using D_raw's own
    # absolute magnitude directly: knn-sparsify each row to its
    # n_neighbors smallest entries, then Dijkstra-complete the resulting
    # sparse graph into a dense geodesic distance matrix.
    # dataIsDistMatrix=True tells compute_dist_matrix that D_raw is already
    # a pairwise distance matrix (not raw coordinates), so it skips its own
    # cdist step and knn-sparsifies D_raw directly. directed=True keeps the
    # asymmetry (D_raw[i,j] != D_raw[j,i] in general) instead of collapsing
    # it to a symmetric shortest-path matrix.
    if verbose:
        print(f"\nBuilding D_asym via randers_bridge.compute_dist_matrix "
              f"(dataIsDistMatrix=True, directed knn adjacency)...")
    D_asym, _ = compute_dist_matrix(D_raw, n_neighbors=k, randers_field=None,
                                     directed=True, adjacency="knn",
                                     dataIsDistMatrix=True)
    D_sym = 0.5 * (D_asym + D_asym.T)

    if verbose:
        print(f"\nReconstructing ambient coordinates X ({args.ambient_dim}D) "
              f"via classical_mds on the symmetric part...")
    X = classical_mds(D_sym, d=args.ambient_dim, seed=args.seed)

    if verbose:
        print(f"\nDeriving drift field omega from D_asym's own asymmetry "
              f"(compute_highdim_drift)...")
    omega = compute_highdim_drift(X, D_asym, k, clip_delta=args.clip_delta)
    if verbose:
        bn0 = np.linalg.norm(omega, axis=1)
        print(f"omega: mean||omega||={bn0.mean():.4f}  max||omega||={bn0.max():.4f}")

    result = fdgl_pipeline(X, omega, k=k, emb_k=k, neg=args.neg,
                            epochs=args.epochs, clip_delta=args.clip_delta,
                            proj_dim=args.proj_dim, adjacency=args.adjacency,
                            snapshot_every=args.snapshot_every,
                            seed=args.seed, verbose=verbose,
                            apply_step=not args.init_only,
                            B_fixed=not args.live_drift)
    Y, B = result["Y"], result["B"]

    # ---- plot: 2D embedding with point labels and drift arrows -----------
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(Y[:, 0], Y[:, 1], s=30, alpha=0.8)
    for i, lab in enumerate(labels):
        ax.annotate(lab, (Y[i, 0], Y[i, 1]), fontsize=8,
                    xytext=(3, 3), textcoords="offset points")

    bn = np.linalg.norm(B, axis=1)
    if bn.max() > 0:
        sc_scale = arrow_scale(Y, bn)
        ax.quiver(Y[:, 0], Y[:, 1], B[:, 0] * sc_scale, B[:, 1] * sc_scale,
                   color="crimson", alpha=0.7, width=0.004, scale=1, scale_units="xy")

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(plot_caption(args.dataset.capitalize(), "calculated", n, k, not args.live_drift,
                               epochs=args.epochs, init_only=args.init_only), fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(data_dir, f"{out_name}.png")
    fig.savefig(out_path, dpi=150)

    np.savez(os.path.join(data_dir, f"{out_name}.npz"), Y=Y, B=B, X=X, omega=omega,
             labels=np.array(labels),
             asymmetry_score=result.get("asymmetry_score", np.nan))

    if verbose:
        print(f"\nwrote {out_path} and {out_name}.npz (in {data_dir})")

    # ---- snapshot grid: init -> every N epochs -> final, side by side -----
    if args.snapshot_every is not None and not args.init_only:
        snaps = result["snapshots"]
        n_snap = len(snaps)
        ncols = min(n_snap, 6)
        nrows = int(np.ceil(n_snap / ncols))
        fig2, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows), squeeze=False)
        for idx, snap in enumerate(snaps):
            ax2 = axes[idx // ncols][idx % ncols]
            Yi, Bi = snap["Y"], snap["B"]
            ax2.scatter(Yi[:, 0], Yi[:, 1], s=18, alpha=0.8)
            for i, lab in enumerate(labels):
                ax2.annotate(lab, (Yi[i, 0], Yi[i, 1]), fontsize=6,
                             xytext=(2, 2), textcoords="offset points")
            bni = np.linalg.norm(Bi, axis=1)
            if bni.max() > 0:
                sc_scale_i = arrow_scale(Yi, bni)
                ax2.quiver(Yi[:, 0], Yi[:, 1], Bi[:, 0] * sc_scale_i, Bi[:, 1] * sc_scale_i,
                           color="crimson", alpha=0.6, width=0.006, scale=1, scale_units="xy")
            ax2.set_title(f"epoch {snap['epoch']}", fontsize=9)
            ax2.set_xticks([]); ax2.set_yticks([])
        for idx in range(n_snap, nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")

        fig2.suptitle(plot_caption(args.dataset.capitalize(), "calculated", n, k, not args.live_drift,
                                    epochs=args.epochs) +
                      f" | snapshot_every={args.snapshot_every}", fontsize=11)
        snap_path = os.path.join(data_dir, f"{out_name}_snapshots.png")
        fig2.savefig(snap_path, dpi=150, bbox_inches="tight")
        if verbose:
            print(f"wrote {snap_path} ({n_snap} snapshots)")


if __name__ == "__main__":
    main()
