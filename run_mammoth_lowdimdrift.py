#!/usr/bin/env python3
"""
run_mammoth_lowdimdrift.py -- mammoth + Randers Force-Directed Layout with
drift computed the OLD way (asymmds/randers_umap.py's original method):
purely in the LOW-DIM embedding space, b_i = (1/k) * sum_{j in N_k(i)}
N[i,j] * e_ij(Y), where N is derived once from D_asym's own asymmetry.
Same code as MNIST/mnist_pca_lowdimdrift.py, only the data source differs
(mammoth.csv via make_mammoth_randers, no PCA step).

No omega, no 2n-point locate step, no fdgl_pipeline: D_asym is built once
(randers_field=None) and fdgl_low_dim is called directly on it.

    default           B LIVE: recomputed every epoch from the current Y
    --fixed-drift     B FIXED: computed once from Y_init = classical_mds(D_asym)

Usage
-----
    python3 run_mammoth_lowdimdrift.py
    python3 run_mammoth_lowdimdrift.py --n 2000 --k 20 --epochs 500 --fixed-drift
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "MNIST"))  # sphere_view.py lives there

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_mammoth_generated import make_mammoth_randers
from randers_fdgl import (arrow_scale, plot_caption, fdgl_low_dim, classical_mds,
                           _compute_N, compute_drift, knn_mask_from_distance_matrix)
from randers_bridge import compute_dist_matrix, asymmetry_score, reconstruct_rho
from sphere_view import stereographic_project


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=5000,
                    help="number of mammoth points to sample")
    p.add_argument("--k", type=int, default=20,
                    help="shared k: both the D_asym build (compute_dist_matrix) and "
                         "fdgl_low_dim's own k-NN backbone.")
    p.add_argument("--neg", type=int, default=10)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--clip-delta", type=float, default=0.01,
                    help="used by compute_drift to cap ||b_i||.")
    p.add_argument("--snapshot-every", type=int, default=None,
                    help="if given, also save <out>_snapshots.png: the embedding every N "
                         "epochs (from init to final), side by side.")
    p.add_argument("--gravity", action="store_true",
                    help="add per-node gravity toward xi_i=y_i+b_i "
                         "(Bannister et al. f_g=gamma*M[i]*b_i), weighted by "
                         "--gravity-neighbor-weight unless disabled.")
    p.add_argument("--gravity-strength", type=float, default=1.0,
                    help="gamma in Bannister et al.'s gravity force. Only matters with --gravity.")
    p.add_argument("--no-gravity-neighbor-weight", action="store_true",
                    help="disable the neighbour-plausibility weighting (revert to the old "
                         "unconditional gravity pull). Only matters with --gravity.")
    p.add_argument("--no-virtual-neighbor", action="store_true",
                    help="[default ON] each node's own virtual point xi_i=y_i+b_i is an "
                         "unconditional (k+1)-th attractive neighbour -- pass to disable.")
    p.add_argument("--ramp", action="store_true",
                    help="ramp drift's magnitude 0->1 over epochs instead of applying it at "
                         "full strength from epoch 0. Off by default.")
    p.add_argument("--force-model", choices=["fr_gravity", "umap"], default="umap",
                    help="attraction/repulsion law -- 'umap' (default) = UMAP's own fitted "
                         "(a,b)-curve, 'fr_gravity' = Bannister et al.'s spring/inverse-square law.")
    p.add_argument("--fr-k", type=float, default=None,
                    help="natural edge-length constant for force_model=fr_gravity "
                         "(default None -> 1/sqrt(n)). Ignored for force_model=umap.")
    p.add_argument("--neg-sampling", action="store_true",
                    help="use TRUE stochastic negative sampling for repulsion instead of the "
                         "dense/exact sum.")
    p.add_argument("--sphere-view", action="store_true",
                    help="also save <out>_sphere.png: the trained 2D embedding mapped onto a "
                         "unit sphere via inverse stereographic projection.")
    p.add_argument("--proj-dim", type=int, default=2, choices=[2, 3])
    p.add_argument("--adjacency", choices=["threshold", "knn"], default="knn")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="mammoth_lowdimdrift")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--fixed-drift", action="store_true",
                    help="[default OFF = live drift] compute B ONCE from the initial "
                         "embedding Y_init = classical_mds(D_asym) and freeze it for the whole "
                         "run (B_fixed). Default: B is recomputed every epoch from the current Y.")
    p.add_argument("--no-isumap-perturb", action="store_true",
                    help="disable IsUMap conformal distance rescaling in compute_dist_matrix "
                         "(applies wherever randers_field=None). Default: perturbation ON.")
    args = p.parse_args()
    verbose = not args.quiet

    save_dir = os.path.join(HERE, "")

    # ---- load: mammoth.csv, subsampled ------------------------------------
    if verbose:
        print(f"Loading mammoth.csv, subsampling to n={args.n}")
    X, _omega_true, z = make_mammoth_randers(args.n, seed=42, alpha=0.5)
    n = X.shape[0]
    if verbose:
        print(f"X: {X.shape}")

    # ---- D_asym: randers_field=None ---------------------------------------
    if verbose:
        print(f"\nBuilding distance matrix via randers_bridge.compute_dist_matrix "
              f"(directed knn adjacency, no field)...")
    D_asym, _, bln_asym = compute_dist_matrix(X, n_neighbors=args.k, randers_field=None,
                                               isumap_perturb=not args.no_isumap_perturb,
                                               directed=True, adjacency=args.adjacency,
                                               return_adjacency=True)
    if verbose:
        finite = np.isfinite(D_asym)
        print(f"D_asym: {D_asym.shape}  finite entries={finite.sum()}/{D_asym.size}  "
              f"symmetric={np.allclose(D_asym, D_asym.T)}")
    asym_per_node, asym_global = asymmetry_score(D_asym, bln_asym)
    if verbose:
        print(f"asymmetry_score (input D_asym): global={asym_global:.4f}")

    # ---- train directly on D_asym: NO omega, NO 2n-point locate step.
    # use_drift=True + B_fixed=None -> B recomputed every epoch from N (fixed,
    # derived once from D_asym's own asymmetry) and the CURRENT, training Y.
    B_fixed_arr, Y_init_arr = None, None
    if args.fixed_drift:
        knn_mask0 = knn_mask_from_distance_matrix(D_asym, args.k)
        N0 = _compute_N(D_asym, knn_mask=knn_mask0)
        Y_init_arr = classical_mds(D_asym, args.proj_dim, seed=args.seed)
        B_fixed_arr = compute_drift(N0, knn_mask0, args.k, Y_init_arr, args.clip_delta)
        if verbose:
            bn0 = np.linalg.norm(B_fixed_arr, axis=1)
            print(f"\nfixed drift: B computed once from Y_init, mean||b||={bn0.mean():.4f}  "
                  f"max||b||={bn0.max():.4f}")
    if verbose:
        print(f"\nTraining fdgl_low_dim directly on D_asym: use_drift=True, "
              f"B={'frozen (computed once from Y_init)' if args.fixed_drift else 'live (recomputed every epoch)'} "
              f"-- force_model={args.force_model}, epochs={args.epochs}...")
    result = fdgl_low_dim(D_asym, n_neighbors=args.k, n_negative_samples=args.neg,
                           n_epochs=args.epochs, use_drift=True, d=args.proj_dim,
                           B_fixed=B_fixed_arr, Y_init_override=Y_init_arr,
                           use_gravity=args.gravity, gravity_strength=args.gravity_strength,
                           gravity_neighbor_weight=not args.no_gravity_neighbor_weight,
                           use_virtual_neighbor=not args.no_virtual_neighbor,
                           ramp=args.ramp, snapshot_every=args.snapshot_every,
                           clip_delta=args.clip_delta, seed=args.seed, verbose=verbose,
                           force_model=args.force_model, fr_k=args.fr_k,
                           negative_sampling=args.neg_sampling)
    Y, B = result["Y"], result["B"]

    rho_final = reconstruct_rho(Y, B)
    asym_per_node_final, asym_global_final = asymmetry_score(rho_final, bln_asym)
    if verbose:
        bn = np.linalg.norm(B, axis=1)
        limit = 1.0 - args.clip_delta
        print(f"\nextent={Y.max()-Y.min():.2f}  mean||b||={bn.mean():.4f}  "
              f"max||b||={bn.max():.4f}  clipped={(bn >= limit - 1e-9).sum()}/{n}")
        pct = 100.0 * asym_global_final / max(asym_global, 1e-12)
        print(f"asymmetry_score (final embedding): {asym_global_final:.4f}  "
              f"vs target (D_asym) {asym_global:.4f}  -- {pct:.1f}% of target asymmetry preserved")

    # ---- plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(Y[:, 0], Y[:, 1], c=z, cmap="viridis", s=6, alpha=0.85, linewidths=0)
    plt.colorbar(sc, ax=ax, label="z (tail<->head)")

    bn = np.linalg.norm(B, axis=1)
    big = np.argsort(bn)[::-1][:100]
    if bn.max() > 0:
        sc_scale = arrow_scale(Y, bn)
        ax.quiver(Y[big, 0], Y[big, 1], B[big, 0] * sc_scale, B[big, 1] * sc_scale,
                  color="k", alpha=0.6, width=0.004, scale=1, scale_units="xy")

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(plot_caption("Mammoth (low-dim drift)", "calculated", n, args.k,
                               args.fixed_drift, epochs=args.epochs), fontsize=10)
    fig.tight_layout()
    out_path = os.path.join(save_dir, f"{args.out}.png")
    fig.savefig(out_path, dpi=150)

    np.savez(os.path.join(save_dir, f"{args.out}.npz"), Y=Y, B=B, z=z, X=X,
             asymmetry_score=asym_global, asymmetry_score_final=asym_global_final)

    if verbose:
        print(f"\nwrote {out_path} and {args.out}.npz (in {save_dir})")

    # ---- sphere view -------------------------------------------------------
    if args.sphere_view:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
        Y_sphere, center, scale = stereographic_project(Y)
        fig_s = plt.figure(figsize=(9, 8))
        ax_s = fig_s.add_subplot(111, projection="3d")
        sc_s = ax_s.scatter(Y_sphere[:, 0], Y_sphere[:, 1], Y_sphere[:, 2],
                            c=z, cmap="viridis", s=6, alpha=0.9, linewidths=0)
        fig_s.colorbar(sc_s, ax=ax_s, label="z (tail<->head)", shrink=0.6)

        u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
        xs, ys_, zs = np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)
        ax_s.plot_wireframe(xs, ys_, zs, color="gray", linewidth=0.3, alpha=0.2)

        if bn.max() > 0:
            heads2d = Y[big] + B[big] * sc_scale
            heads3d, _, _ = stereographic_project(heads2d, center=center, scale=scale)
            tails3d = Y_sphere[big]
            for t, h in zip(tails3d, heads3d):
                ax_s.plot([t[0], h[0]], [t[1], h[1]], [t[2], h[2]],
                          color="k", alpha=0.6, linewidth=0.8)

        ax_s.set_xticks([]); ax_s.set_yticks([]); ax_s.set_zticks([])
        ax_s.set_title(f"Randers Force-Directed Layout on Mammoth (low-dim drift, "
                        f"sphere view, n={n})", fontsize=10)
        fig_s.tight_layout()
        sphere_path = os.path.join(save_dir, f"{args.out}_sphere.png")
        fig_s.savefig(sphere_path, dpi=150)
        if verbose:
            print(f"wrote {sphere_path}")

    # ---- snapshot grid: init -> every N epochs -> final, side by side -----
    if args.snapshot_every is not None:
        snaps = result["snapshots"]
        n_snap = len(snaps)
        ncols = min(n_snap, 6)
        nrows = int(np.ceil(n_snap / ncols))
        fig2, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows), squeeze=False)
        sc2 = None
        for idx, snap in enumerate(snaps):
            ax2 = axes[idx // ncols][idx % ncols]
            Yi, Bi = snap["Y"], snap["B"]
            sc2 = ax2.scatter(Yi[:, 0], Yi[:, 1], c=z, cmap="viridis", s=6,
                              alpha=0.85, linewidths=0)
            bni = np.linalg.norm(Bi, axis=1)
            bigi = np.argsort(bni)[::-1][:100]
            if bni.max() > 0:
                sc_scale_i = arrow_scale(Yi, bni)
                ax2.quiver(Yi[bigi, 0], Yi[bigi, 1], Bi[bigi, 0] * sc_scale_i, Bi[bigi, 1] * sc_scale_i,
                          color="k", alpha=0.6, width=0.006, scale=1, scale_units="xy")
            ax2.set_title(f"epoch {snap['epoch']}", fontsize=9)
            ax2.set_xticks([]); ax2.set_yticks([])
        for idx in range(n_snap, nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")

        fig2.suptitle(plot_caption("Mammoth (low-dim drift)", "calculated", n, args.k,
                                    args.fixed_drift, epochs=args.epochs) +
                      f" | snapshot_every={args.snapshot_every}", fontsize=11)
        if sc2 is not None:
            fig2.colorbar(sc2, ax=axes, label="z (tail<->head)", shrink=0.6)
        snap_path = os.path.join(save_dir, f"{args.out}_snapshots.png")
        fig2.savefig(snap_path, dpi=150)
        if verbose:
            print(f"wrote {snap_path}")


if __name__ == "__main__":
    main()
