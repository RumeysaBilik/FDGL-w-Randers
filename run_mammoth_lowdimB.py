#!/usr/bin/env python3
"""
run_mammoth_lowdimB.py -- run_mammoth_calculated.py with ONE change: the drift B
is computed in the low-dimensional embedding (as in the first commit's
randers_umap_fit), not located from a 2n-point real + virtual MDS.

Same as run_mammoth_calculated.py: D_asym, omega (compute_highdim_drift),
training matrix D_real_asym (global Randers, from omega), D_geo, force model,
virtual neighbour, all flags and defaults. Different: no locate step.
Y_init = classical_mds on the n real points; B at epoch 0 =
compute_drift(N, mask, k, Y_init); live by default (recomputed every epoch
from the current Y), --fixed-drift freezes the epoch-0 value.
See fdgl_pipeline_lowdim_B.py.

Usage
-----
    python run_mammoth_lowdimB.py
    python run_mammoth_lowdimB.py --n 1500 --epochs 500 --fixed-drift

Outputs
-------
    <out>_3d_field.png            ambient X coloured by z, DERIVED omega arrows
    <out>_3d_drift_attached.png   ambient X -> X+omega (derived), exaggerated for visibility
    <out>.png                     embedding coloured by z, drift arrows
    <out>.npz                     Y, B, z, X, omega (derived)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_mammoth_generated import make_mammoth_randers
from randers_bridge import compute_dist_matrix, compute_highdim_drift
from fdgl_pipeline_lowdim_B import fdgl_pipeline_lowdim_B
from randers_fdgl import arrow_scale, plot_caption
from legacy_isumap_dist_matrix import dijkstra_only_dist_matrix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--k", type=int, default=20,
                    help="shared k: isumap's own D_asym (for deriving omega) AND "
                         "fdgl_pipeline_lowdim_B's own k-NN backbone (apply step).")
    p.add_argument("--neg", type=int, default=10)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--clip-delta", type=float, default=0.01,
                    help="used both when deriving omega (compute_highdim_drift) and "
                         "inside fdgl_pipeline's own apply step -- see "
                         "compute_highdim_drift's docstring for the open caveat about "
                         "this being an absolute (not X-scale-relative) cap.")
    p.add_argument("--gravity", action="store_true")
    p.add_argument("--gravity-strength", type=float, default=1.0)
    p.add_argument("--no-gravity-neighbor-weight", action="store_true")
    p.add_argument("--no-virtual-neighbor", action="store_true",
                    help="[default ON] each node's own virtual point xi_i=y_i+b_i is an "
                         "unconditional (k+1)-th attractive neighbour -- pass to disable.")
    p.add_argument("--snapshot-every", type=int, default=None)
    p.add_argument("--ramp", action="store_true")
    p.add_argument("--fixed-drift", action="store_true",
                    help="[default OFF = live drift] compute B ONCE from Y_init and freeze it "
                         "for the whole run. Default: B is recomputed every epoch from the "
                         "current Y (N and the neighbour mask stay fixed).")
    p.add_argument("--raw-init", action="store_true",
                    help="run classical_mds on D_real_asym as is, instead of on "
                         "0.5*(D_real_asym + D_real_asym^T).")
    p.add_argument("--proj-dim", type=int, default=2, choices=[2, 3])
    p.add_argument("--adjacency", choices=["threshold", "knn"], default="knn",
                    help="fdgl_pipeline's own adjacency mode -- see randers_bridge."
                         "compute_dist_matrix's adjacency docstring.")
    p.add_argument("--normalize", action="store_true",
                    help="see fdgl_low_dim's scale_B_fixed_by_knn_distance docstring.")
    p.add_argument("--force-model", choices=["fr_gravity", "umap"], default="umap")
    p.add_argument("--fr-k", type=float, default=None)
    p.add_argument("--neg-sampling", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="mammoth_lowdimB")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-isumap-perturb", action="store_true",
                    help="disable IsUMap conformal distance rescaling in compute_dist_matrix (applies wherever randers_field=None). Default: perturbation ON.")
    args = p.parse_args()
    verbose = not args.quiet

    if not args.quiet:
        print(f"Loading mammoth.csv, subsampling to n={args.n}")
  
    X, _omega_true, z = make_mammoth_randers(args.n, seed=42, alpha=0.5)
    
    n = args.n

    if not args.quiet:
        print(f"\nBuilding distance matrix via randers_bridge.compute_dist_matrix "
              f"(directed k-NN adjacency, no isumap normalization, no field)...")

    D_asym, _ = compute_dist_matrix(X, n_neighbors=args.k, randers_field=None, isumap_perturb=not args.no_isumap_perturb, directed=True, adjacency="knn")
    #D_asym = dijkstra_only_dist_matrix(X, k=args.k, verbose=verbose, directedDistances=True)
    
    emb_k = args.k
    if not args.quiet:
        print(f"D_asym: {D_asym.shape}  symmetric={np.allclose(D_asym, D_asym.T)}  "
              f"emb_k used={emb_k}")

    if not args.quiet:
        print(f"\nDeriving AMBIENT-space omega from D_asym's own asymmetry "
              f"(compute_highdim_drift)...")
    omega = compute_highdim_drift(X, D_asym, emb_k, clip_delta=args.clip_delta)
    if not args.quiet:
        bn0 = np.linalg.norm(omega, axis=1)
        print(f"omega (derived, ambient): mean||omega||={bn0.mean():.4f}  "
              f"max||omega||={bn0.max():.4f}  (X itself spans "
              f"~{np.ptp(X, axis=0).max():.1f} units per axis -- compare scale)")

    # ---- 3D plot of the ambient mammoth with the DERIVED omega field ------
    fig3d = plt.figure(figsize=(11, 9))
    ax3d = fig3d.add_subplot(111, projection="3d")
    sc3d = ax3d.scatter(X[:, 0], X[:, 1], X[:, 2], c=z, cmap="viridis", s=8,
                        alpha=0.85, linewidths=0)
    fig3d.colorbar(sc3d, ax=ax3d, label="z (tail<->head)", shrink=0.6, pad=0.08)
    rng3d = np.random.RandomState(0)
    idx3d = rng3d.choice(n, size=min(200, n), replace=False)
    bn_omega = np.linalg.norm(omega, axis=1)
    # pick a scale so the median arrow spans ~5% of X's own extent, visible either way.
    x_extent = np.ptp(X, axis=0).max()
    scale3d = (0.05 * x_extent / max(np.median(bn_omega), 1e-8)) if bn_omega.max() > 0 else 1.0
    ax3d.quiver(X[idx3d, 0], X[idx3d, 1], X[idx3d, 2],
                omega[idx3d, 0] * scale3d, omega[idx3d, 1] * scale3d, omega[idx3d, 2] * scale3d,
                color="crimson", alpha=0.8, linewidth=1.0, arrow_length_ratio=0.3)
    ax3d.set_title(f"Mammoth (ambient X, n={n}) with DERIVED omega field "
                    f"(exaggerated x{scale3d:.2g})", fontsize=11)
    ax3d.set_xlabel("x"); ax3d.set_ylabel("y (height)"); ax3d.set_zlabel("z (tail<->head)")
    fig3d.tight_layout()
    fig3d.savefig(f"{args.out}_3d_field.png", dpi=150)
    if not args.quiet:
        print(f"wrote {args.out}_3d_field.png")

    X_virtual_display = X + omega * scale3d
    fig3d_v = plt.figure(figsize=(11, 9))
    ax3d_v = fig3d_v.add_subplot(111, projection="3d")
    sc3d_v = ax3d_v.scatter(X[:, 0], X[:, 1], X[:, 2], c=z, cmap="viridis", s=8,
                             alpha=0.85, linewidths=0)
    fig3d_v.colorbar(sc3d_v, ax=ax3d_v, label="z (tail<->head)", shrink=0.6, pad=0.08)
    ax3d_v.quiver(X[idx3d, 0], X[idx3d, 1], X[idx3d, 2],
                  X_virtual_display[idx3d, 0] - X[idx3d, 0],
                  X_virtual_display[idx3d, 1] - X[idx3d, 1],
                  X_virtual_display[idx3d, 2] - X[idx3d, 2],
                  color="crimson", alpha=0.9, linewidth=1.2, arrow_length_ratio=0.25)
    ax3d_v.set_title(f"Mammoth: DERIVED drift attached (x_i -> x_i+omega_i) "
                      f"(n={n}, omega exaggerated x{scale3d:.2g} for visibility)", fontsize=10)
    ax3d_v.set_xlabel("x"); ax3d_v.set_ylabel("y (height)"); ax3d_v.set_zlabel("z (tail<->head)")
    fig3d_v.tight_layout()
    fig3d_v.savefig(f"{args.out}_3d_drift_attached.png", dpi=150)
    if not args.quiet:
        print(f"wrote {args.out}_3d_drift_attached.png")

    result = fdgl_pipeline_lowdim_B(X, omega, isumap_perturb=not args.no_isumap_perturb, k=args.k, emb_k=args.k, neg=args.neg,
                               epochs=args.epochs,
                               clip_delta=args.clip_delta, use_gravity=args.gravity,
                               snapshot_every=args.snapshot_every, ramp=args.ramp,
                               gravity_strength=args.gravity_strength,
                               gravity_neighbor_weight=not args.no_gravity_neighbor_weight,
                               use_virtual_neighbor=not args.no_virtual_neighbor,
                               proj_dim=args.proj_dim, adjacency=args.adjacency,
                               seed=args.seed, verbose=not args.quiet,
                               normalize_drift_by_asymmetry=args.normalize,
                               force_model=args.force_model, fr_k=args.fr_k,
                               negative_sampling=args.neg_sampling,
                               B_fixed=args.fixed_drift,
                               symmetrize_init=not args.raw_init)
    Y, B = result["Y"], result["B"]

    # ---- plot ------------------------------------------------------------
    bn = np.linalg.norm(B, axis=1)
    big = np.argsort(bn)[::-1][:200]
    if args.proj_dim == 3:
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=z, cmap="viridis", s=10,
                        alpha=0.85, linewidths=0)
        fig.colorbar(sc, ax=ax, label="z (tail<->head)", shrink=0.6, pad=0.08)
        if bn.max() > 0:
            sc_scale = arrow_scale(Y, bn)
            ax.quiver(Y[big, 0], Y[big, 1], Y[big, 2],
                      B[big, 0] * sc_scale, B[big, 1] * sc_scale, B[big, 2] * sc_scale,
                      color="k", alpha=0.6, linewidth=1.0, arrow_length_ratio=0.3)
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2"); ax.set_zlabel("dim 3")
    else:
        fig, ax = plt.subplots(figsize=(9, 8))
        sc = ax.scatter(Y[:, 0], Y[:, 1], c=z, cmap="viridis", s=10, alpha=0.85, linewidths=0)
        plt.colorbar(sc, ax=ax, label="z (tail<->head)")
        if bn.max() > 0:
            sc_scale = arrow_scale(Y, bn)
            ax.quiver(Y[big, 0], Y[big, 1], B[big, 0] * sc_scale, B[big, 1] * sc_scale,
                      color="k", alpha=0.6, width=0.004, scale=1, scale_units="xy")
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")

    ax.set_title(plot_caption("Mammoth", "calculated", n, args.k, args.fixed_drift,
                               epochs=args.epochs), fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)

    np.savez(f"{args.out}.npz", Y=Y, B=B, z=z, X=X, omega=omega,
             asymmetry_score=result.get("asymmetry_score", np.nan),
             asymmetry_per_node=result.get("asymmetry_per_node", np.array([])))

    if not args.quiet:
        print(f"\nwrote {args.out}.png and {args.out}.npz")

    if args.snapshot_every is not None:
        snaps = result["snapshots"]
        n_snap = len(snaps)
        ncols = min(n_snap, 6)
        nrows = int(np.ceil(n_snap / ncols))
        vmin, vmax = z.min(), z.max()
        sc2 = None

        if args.proj_dim == 3:
            fig2 = plt.figure(figsize=(3.6 * ncols, 3.6 * nrows))
            axes2 = [fig2.add_subplot(nrows, ncols, idx + 1, projection="3d")
                     for idx in range(nrows * ncols)]
            for idx, snap in enumerate(snaps):
                ax2 = axes2[idx]
                Yi, Bi = snap["Y"], snap["B"]
                sc2 = ax2.scatter(Yi[:, 0], Yi[:, 1], Yi[:, 2], c=z, cmap="viridis",
                                  s=6, alpha=0.85, linewidths=0, vmin=vmin, vmax=vmax)
                bni = np.linalg.norm(Bi, axis=1)
                bigi = np.argsort(bni)[::-1][:200]
                if bni.max() > 0:
                    sc_scale_i = arrow_scale(Yi, bni)
                    ax2.quiver(Yi[bigi, 0], Yi[bigi, 1], Yi[bigi, 2],
                              Bi[bigi, 0] * sc_scale_i, Bi[bigi, 1] * sc_scale_i,
                              Bi[bigi, 2] * sc_scale_i,
                              color="k", alpha=0.6, linewidth=0.8, arrow_length_ratio=0.3)
                ax2.set_title(f"epoch {snap['epoch']}", fontsize=9)
            for idx in range(n_snap, nrows * ncols):
                axes2[idx].axis("off")
        else:
            fig2, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows),
                                       squeeze=False)
            for idx, snap in enumerate(snaps):
                ax2 = axes[idx // ncols][idx % ncols]
                Yi, Bi = snap["Y"], snap["B"]
                sc2 = ax2.scatter(Yi[:, 0], Yi[:, 1], c=z, cmap="viridis", s=6,
                                  alpha=0.85, linewidths=0, vmin=vmin, vmax=vmax)
                bni = np.linalg.norm(Bi, axis=1)
                bigi = np.argsort(bni)[::-1][:200]
                if bni.max() > 0:
                    sc_scale_i = arrow_scale(Yi, bni)
                    ax2.quiver(Yi[bigi, 0], Yi[bigi, 1],
                              Bi[bigi, 0] * sc_scale_i, Bi[bigi, 1] * sc_scale_i,
                              color="k", alpha=0.6, width=0.006, scale=1, scale_units="xy")
                ax2.set_title(f"epoch {snap['epoch']}", fontsize=9)
                ax2.set_xticks([]); ax2.set_yticks([])
            for idx in range(n_snap, nrows * ncols):
                axes[idx // ncols][idx % ncols].axis("off")

        fig2.suptitle(plot_caption("Mammoth", "calculated", n, args.k, args.fixed_drift,
                                    epochs=args.epochs) +
                      f" | snapshot_every={args.snapshot_every}", fontsize=11)
        if sc2 is not None:
            fig2.colorbar(sc2, ax=fig2.get_axes(), label="z (tail<->head)",
                          fraction=0.02, pad=0.01)
        fig2.savefig(f"{args.out}_snapshots.png", dpi=150, bbox_inches="tight")

        if not args.quiet:
            print(f"wrote {args.out}_snapshots.png ({n_snap} snapshots)")


if __name__ == "__main__":
    main()
