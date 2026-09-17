#!/usr/bin/env python3
"""
embed_MNIST_pca.py -- the "PCA-first" variant of MNIST + our own
Randers Force-Directed Layout pipeline, per the advisor's second suggested approach ([OURS
2026-08-26] "hocam demisti ki MNIST'i uygulamanin iki yolu var: 1) direkt
28x28'lik dimensiondan baslayip metodumuzu uygulamak, 2) once lower
dimension'a cekip sonra metodumuzu uygulamak" -- this file implements
option 2).

[OURS 2026-09-17] Migrated off isumap's own build_isumap_dist_matrix/
isumap_style_init/fdgl_low_dim-direct pipeline, onto the same
compute_highdim_drift + fdgl_pipeline approach run_swiss_roll_calculated.py/
run_mammoth_calculated.py/run_sphere_calculated.py now use: D_asym is built
purely via randers_bridge.compute_dist_matrix's own directed k-NN geodesic
(no isumap star-graph/t-conorm/Dijkstra-skip machinery involved at all),
omega is derived from D_asym's own asymmetry in the PCA-reduced ambient
space via randers_bridge.compute_highdim_drift, and (X_pca, omega) is fed
straight into fdgl_pipeline -- the SAME locate+apply pipeline the
"generated" family uses. This is a real change of premise from the old
version: this script no longer tests IsUMap's own asymmetric-distance
mechanism at all, only our own pipeline applied to real (PCA-reduced) data.

[OURS 2026-09-17] Also no longer imports anything from isumap/ at all --
load_MNIST now comes from mnist_loader.py (a self-contained copy of
isumap/data_and_plots.py's own load_MNIST, minus the unconditional
`import torchvision` that file carries purely for its own unrelated
load_CIFAR_10, never used in this project). This script no longer needs
torchvision installed.

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
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from mnist_loader import load_MNIST
from randers_fdgl import arrow_scale
from randers_bridge import fdgl_pipeline, compute_dist_matrix, compute_highdim_drift
from sphere_view import stereographic_project


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=5000,
                    help="number of MNIST points to sample")
    p.add_argument("--pca-dim", type=int, default=50,
                    help="target dimension for the PCA pre-reduction step, 784 -> this. "
                         "50 is van der Maaten's own classic t-SNE-preprocessing "
                         "recommendation; adjust and compare explained-variance printed below.")
    p.add_argument("--k", type=int, default=20,
                    help="shared k: both the ambient D_asym build (compute_dist_matrix, "
                         "for deriving omega) and fdgl_pipeline's own k-NN backbone "
                         "(locate/apply steps).")
    p.add_argument("--neg", type=int, default=10)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--locate-epochs", type=int, default=500,
                    help="no-op -- kept for compat with fdgl_pipeline's signature.")
    p.add_argument("--clip-delta", type=float, default=0.01,
                    help="used both when deriving omega (compute_highdim_drift) and inside "
                         "fdgl_pipeline's own apply step -- see compute_highdim_drift's "
                         "docstring for the open caveat about this being an absolute "
                         "(not X-scale-relative) cap, not yet calibrated for PCA-reduced "
                         "pixel-space coordinates.")
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
    p.add_argument("--normalize", action="store_true",
                    help="see fdgl_low_dim's scale_B_fixed_by_knn_distance docstring.")
    p.add_argument("--force-model", choices=["fr_gravity", "umap"], default="fr_gravity",
                    help="attraction/repulsion law -- 'fr_gravity' (default) = Bannister et "
                         "al.'s spring/inverse-square law, 'umap' = UMAP's own fitted (a,b)-curve.")
    p.add_argument("--fr-k", type=float, default=None,
                    help="natural edge-length constant for force_model=fr_gravity "
                         "(default None -> 1/sqrt(n)). Ignored for force_model=umap.")
    p.add_argument("--neg-sampling", action="store_true",
                    help="use TRUE stochastic negative sampling for repulsion instead of the "
                         "dense/exact sum.")
    p.add_argument("--sphere-view", action="store_true",
                    help="also save <out>_sphere.png: the trained 2D embedding mapped onto a "
                         "unit sphere via inverse stereographic projection -- purely a "
                         "post-training visualization. Off by default.")
    p.add_argument("--proj-dim", type=int, default=2, choices=[2, 3])
    p.add_argument("--adjacency", choices=["threshold", "knn"], default="knn")
    p.add_argument("--init-only", action="store_true")
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

    # ---- PCA: 784 -> --pca-dim -- the only pre-processing step; everything
    # downstream (D_asym build, omega derivation, fdgl_pipeline) is the same
    # mechanism the swiss_roll/mammoth/sphere "calculated" scripts use.
    pca_dim = min(args.pca_dim, X.shape[0], X.shape[1])
    pca = PCA(n_components=pca_dim, random_state=args.seed)
    X_pca = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_.sum()
    if verbose:
        print(f"PCA: {X.shape[1]} -> {pca_dim} dims, "
              f"explained variance retained = {explained:.4f}")
    n = X_pca.shape[0]

    if verbose:
        print(f"\nBuilding distance matrix via randers_bridge.compute_dist_matrix "
              f"(directed knn adjacency, no field)...")
    D_asym, _ = compute_dist_matrix(X_pca, n_neighbors=args.k, randers_field=None,
                                     directed=True, adjacency="knn")
    if verbose:
        print(f"D_asym: {D_asym.shape}  symmetric={np.allclose(D_asym, D_asym.T)}")

    if verbose:
        print(f"\nDeriving AMBIENT-space (PCA-reduced) omega from D_asym's own "
              f"asymmetry (compute_highdim_drift)...")
    omega = compute_highdim_drift(X_pca, D_asym, args.k, clip_delta=args.clip_delta)
    if verbose:
        bn0 = np.linalg.norm(omega, axis=1)
        print(f"omega (derived, ambient): mean||omega||={bn0.mean():.4f}  "
              f"max||omega||={bn0.max():.4f}  (X_pca itself spans "
              f"~{np.ptp(X_pca, axis=0).max():.1f} units per axis -- compare scale)")

    np.save(os.path.join(save_dir, "asymm_matrix_pca.npy"), D_asym)
    np.save(os.path.join(save_dir, "labels_pca.npy"), y)
    np.save(os.path.join(save_dir, "X_pca.npy"), X_pca)

    result = fdgl_pipeline(X_pca, omega, k=args.k, emb_k=args.k, neg=args.neg,
                               locate_epochs=args.locate_epochs, epochs=args.epochs,
                               clip_delta=args.clip_delta, use_gravity=args.gravity,
                               gravity_strength=args.gravity_strength,
                               gravity_neighbor_weight=not args.no_gravity_neighbor_weight,
                               use_virtual_neighbor=not args.no_virtual_neighbor,
                               proj_dim=args.proj_dim, adjacency=args.adjacency,
                               snapshot_every=args.snapshot_every, ramp=args.ramp,
                               seed=args.seed, verbose=verbose,
                               apply_step=not args.init_only,
                               normalize_drift_by_asymmetry=args.normalize,
                               force_model=args.force_model, fr_k=args.fr_k,
                               negative_sampling=args.neg_sampling)
    Y, B = result["Y"], result["B"]

    # ---- plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(Y[:, 0], Y[:, 1], c=y, cmap="tab10", s=6, alpha=0.85, linewidths=0)
    plt.colorbar(sc, ax=ax, label="digit", ticks=range(10))

    bn = np.linalg.norm(B, axis=1)
    big = np.argsort(bn)[::-1][:100]
    if bn.max() > 0:
        sc_scale = arrow_scale(Y, bn)
        ax.quiver(Y[big, 0], Y[big, 1], B[big, 0] * sc_scale, B[big, 1] * sc_scale,
                  color="k", alpha=0.6, width=0.004, scale=1, scale_units="xy")

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Randers Force-Directed Layout on MNIST (PCA {X.shape[1]}->{pca_dim}D, DERIVED high-dim "
                 f"omega, n={n}, explained var={explained:.3f})", fontsize=10)
    fig.tight_layout()
    out_path = os.path.join(save_dir, f"{args.out}.png")
    fig.savefig(out_path, dpi=150)

    np.savez(os.path.join(save_dir, f"{args.out}.npz"), Y=Y, B=B, labels=y,
             pca_dim=pca_dim, explained_variance=explained, omega=omega,
             asymmetry_score=result.get("asymmetry_score", np.nan))

    if verbose:
        print(f"\nwrote {out_path} and {args.out}.npz (in {save_dir})")

    # ---- sphere view: same trained Y, mapped onto a unit sphere via inverse
    # stereographic projection -- purely a post-training visualization,
    # --sphere-view opt-in, training itself is completely untouched.
    if args.sphere_view:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
        Y_sphere, center, scale = stereographic_project(Y)
        fig_s = plt.figure(figsize=(9, 8))
        ax_s = fig_s.add_subplot(111, projection="3d")
        sc_s = ax_s.scatter(Y_sphere[:, 0], Y_sphere[:, 1], Y_sphere[:, 2],
                            c=y, cmap="tab10", s=6, alpha=0.9, linewidths=0)
        fig_s.colorbar(sc_s, ax=ax_s, label="digit", ticks=range(10), shrink=0.6)

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
        ax_s.set_title(f"Randers Force-Directed Layout on MNIST (PCA {X.shape[1]}->{pca_dim}D, sphere view, "
                        f"n={n})", fontsize=10)
        fig_s.tight_layout()
        sphere_path = os.path.join(save_dir, f"{args.out}_sphere.png")
        fig_s.savefig(sphere_path, dpi=150)
        if verbose:
            print(f"wrote {sphere_path}")

    # ---- snapshot grid: init -> every N epochs -> final, side by side -----
    if args.snapshot_every is not None and not args.init_only:
        snaps = result["snapshots"]
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
            bigi = np.argsort(bni)[::-1][:100]
            if bni.max() > 0:
                sc_scale_i = arrow_scale(Yi, bni)
                ax2.quiver(Yi[bigi, 0], Yi[bigi, 1], Bi[bigi, 0] * sc_scale_i, Bi[bigi, 1] * sc_scale_i,
                          color="k", alpha=0.6, width=0.006, scale=1, scale_units="xy")
            ax2.set_title(f"epoch {snap['epoch']}", fontsize=9)
            ax2.set_xticks([]); ax2.set_yticks([])
        for idx in range(n_snap, nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")

        fig2.suptitle(f"Randers Force-Directed Layout on MNIST (PCA {X.shape[1]}->{pca_dim}D), training trajectory "
                      f"(n={n}, snapshot_every={args.snapshot_every})", fontsize=11)
        if sc2 is not None:
            fig2.colorbar(sc2, ax=axes, label="digit", ticks=range(10), shrink=0.6)
        snap_path = os.path.join(save_dir, f"{args.out}_snapshots.png")
        fig2.savefig(snap_path, dpi=150)
        if verbose:
            print(f"wrote {snap_path}")


if __name__ == "__main__":
    main()
