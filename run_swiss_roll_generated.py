#!/usr/bin/env python3
"""
run_swiss_roll_generated.py -- Randers-UMAP on the swiss-roll dataset, but with
the Randers drift field taken EXACTLY from generated_swiss_roll-2.py (the
original DAGES-style generator the user supplied), instead of run_swiss_roll.py's
own modified field. This file is otherwise an exact copy of run_swiss_roll.py --
same CLI flags, same run_located_drift() pipeline, same plotting/snapshot code --
so the two scripts are directly comparable, differing ONLY in how omega is built.

Differences from run_swiss_roll.py's make_swiss_roll_randers():
    - vector field:  V = (-z, 0, x)            (generated_swiss_roll-2.py's own
                                                  rotational field -- a 90-degree
                                                  rotation of the (x,z) position)
                      instead of run_swiss_roll.py's tangent-to-the-spiral
                      derivative field, d/dt (t*cos t, t*sin t).
    - alpha:          alpha(t) = -0.5*cos(t) + 0.3*sin(t)   (t-dependent)
                      instead of run_swiss_roll.py's constant alpha=0.5.
    - optional additive Gaussian noise on X (--noise, default 0.0, matching
      generated_swiss_roll-2.py's own default of noise=0.0).

Everything downstream (locate step, apply step, gravity, virtual-neighbour,
adjacency mode, proj-dim, snapshotting, plotting) is identical to
run_swiss_roll.py -- only make_swiss_roll_randers_generated() differs.

Usage
-----
    python run_swiss_roll_generated.py
    python run_swiss_roll_generated.py --n 2000 --epochs 500 --gravity
    python run_swiss_roll_generated.py --adjacency knn --proj-dim 3

Outputs
-------
    <out>.png   embedding coloured by t, drift arrows on the top-N
    <out>.npz   Y, B, t, X, omega
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

from randers_bridge import compute_dist_matrix
from randers_umap import randers_umap_fit, fuzzy_simplicial_set, classical_mds
from randers_bridge import run_located_drift


def make_swiss_roll_randers_generated(n, seed=42, noise=0.0):
    """Exact construction from generated_swiss_roll-2.py (the original
    generator the user supplied), parameterised by n -- as opposed to
    run_swiss_roll.py's make_swiss_roll_randers(), which uses OUR modified
    tangent-derivative field + a constant alpha. This variant reproduces
    generated_swiss_roll-2.py's own:

        - rotational vector field V = (-z, 0, x)  (a 90-degree rotation of
          the (x, z) position -- NOT the true tangent-to-the-spiral
          derivative that run_swiss_roll.py uses instead)
        - t-dependent alpha(t) = -0.5*cos(t) + 0.3*sin(t)  (not a constant)
        - optional additive Gaussian noise on X (default 0.0, matching the
          original script's own default of noise=0.0)
    """
    rng = np.random.RandomState(seed)

    t = 1.5 * np.pi * (1 + 2 * rng.rand(n))
    height = 21 * rng.rand(n)

    x = t * np.cos(t)
    z = t * np.sin(t)
    y = height
    X = np.column_stack([x, y, z])

    if noise > 0:
        X = X + noise * rng.randn(n, 3)

    # [generated_swiss_roll-2.py's own field] rotational tangent -- a
    # 90-degree rotation of the (x, z) position.
    V = np.column_stack([-X[:, 2], np.zeros(n), X[:, 0]])
    V = V / np.linalg.norm(V, axis=1)[:, None]

    # [generated_swiss_roll-2.py's own alpha] t-dependent, not constant.
    alpha = -0.5 * np.cos(t) + 0.3 * np.sin(t)
    omega = alpha[:, None] * V

    return X, omega, t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n",      type=int, default=1000)
    p.add_argument("--k",      type=int, default=20, help="k-NN for the geodesic backbone")
    p.add_argument("--neg",    type=int, default=10)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--locate-epochs", type=int, default=500,
                    help="no-op -- locate step is a single spectral_layout/classical_mds "
                         "call, no training. Kept for compat.")
    p.add_argument("--clip-delta", type=float, default=0.01)
    p.add_argument("--noise", type=float, default=0.0,
                    help="additive Gaussian noise std on X, matching "
                         "generated_swiss_roll-2.py's own --noise-equivalent (default 0.0).")
    p.add_argument("--gravity", action="store_true",
                    help="add per-node gravity toward xi_i=y_i+b_i "
                         "(Bannister et al. f_g=gamma*M[i]*b_i), weighted by "
                         "--gravity-neighbor-weight unless disabled.")
    p.add_argument("--gravity-strength", type=float, default=1.0,
                    help="gamma_t in Bannister et al.'s gravity force. Only matters with --gravity.")
    p.add_argument("--no-gravity-neighbor-weight", action="store_true",
                    help="disable the neighbour-plausibility weighting (revert to the old "
                         "unconditional gravity pull). Only matters with --gravity.")
    p.add_argument("--no-virtual-neighbor", action="store_true",
                    help="by default, each node's own virtual point xi_i=y_i+b_i is treated "
                         "as an UNCONDITIONAL (k+1)-th attractive neighbour every epoch. Pass "
                         "this flag to disable it and revert to the old behaviour.")
    p.add_argument("--snapshot-every", type=int, default=None,
                    help="if given, also save <out>_snapshots.png: the apply-step "
                         "embedding every N epochs (from Y_real0 to final), side by side")
    p.add_argument("--ramp", action="store_true",
                    help="ramp B_fixed's magnitude 0->1 over the first 70%% of apply-step "
                         "epochs instead of applying it at full strength from epoch 0.")
    p.add_argument("--init-only", action="store_true",
                    help="stop after the locate step -- skip the force-directed apply/"
                         "training step entirely, and just plot/save the raw located "
                         "embedding (Y_real0) with its drift vectors (B_located).")
    p.add_argument("--init-method", choices=["umap", "isomap"], default="isomap",
                    help="locate step's placement method. 'isomap' (default) = classical_mds. "
                         "'umap' is kept for backward CLI compat but is now IDENTICAL to "
                         "'isomap' -- spectral_layout was removed project-wide.")
    p.add_argument("--proj-dim", type=int, default=2, choices=[2, 3],
                    help="embedding dimension for the locate step's placement AND the apply "
                         "step's force-directed training. 2 (default) = existing 2D pipeline. "
                         "3 = full 3D layout.")
    p.add_argument("--adjacency", choices=["threshold", "knn"], default="threshold",
                    help="how compute_dist_matrix builds its base adjacency graph. "
                         "'threshold' (default) = eps-threshold rule, symmetric by "
                         "construction. 'knn' = true per-point k-nearest-neighbours "
                         "membership, asymmetric in general -- see randers_bridge."
                         "compute_dist_matrix's adjacency docstring for the full explanation.")
    p.add_argument("--seed",   type=int, default=0)
    p.add_argument("--out",    default="swiss_embedding_generated")
    p.add_argument("--quiet",  action="store_true")
    args = p.parse_args()

    if not args.quiet:
        print(f"Generating swiss roll (generated_swiss_roll-2.py field): n={args.n}")
    X, omega, t = make_swiss_roll_randers_generated(args.n, seed=42, noise=args.noise)
    n = args.n

    # ---- 3D plot of the ambient swiss roll with the omega (Randers) field ---
    fig3d = plt.figure(figsize=(11, 9))
    ax3d = fig3d.add_subplot(111, projection="3d")
    sc3d = ax3d.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap="viridis", s=8,
                        alpha=0.85, linewidths=0)
    fig3d.colorbar(sc3d, ax=ax3d, label="t (intrinsic coordinate)", shrink=0.6, pad=0.08)
    rng3d = np.random.RandomState(0)
    idx3d = rng3d.choice(n, size=min(150, n), replace=False)
    scale3d = 4.0
    ax3d.quiver(X[idx3d, 0], X[idx3d, 1], X[idx3d, 2],
                omega[idx3d, 0] * scale3d, omega[idx3d, 1] * scale3d, omega[idx3d, 2] * scale3d,
                color="k", alpha=0.7, linewidth=1.0, arrow_length_ratio=0.3)
    ax3d.set_title(f"Swiss roll (ambient X, n={n}) with generated_swiss_roll-2.py's omega field", fontsize=11)
    ax3d.set_xlabel("x"); ax3d.set_ylabel("y (height)"); ax3d.set_zlabel("z")
    fig3d.tight_layout()
    fig3d.savefig(f"{args.out}_3d_field.png", dpi=150)
    if not args.quiet:
        print(f"wrote {args.out}_3d_field.png")

    result = run_located_drift(X, omega, k=args.k, emb_k=args.k, neg=args.neg,
                               locate_epochs=args.locate_epochs, epochs=args.epochs,
                               clip_delta=args.clip_delta, use_gravity=args.gravity,
                               gravity_strength=args.gravity_strength,
                               gravity_neighbor_weight=not args.no_gravity_neighbor_weight,
                               use_virtual_neighbor=not args.no_virtual_neighbor,
                               proj_dim=args.proj_dim, adjacency=args.adjacency,
                               snapshot_every=args.snapshot_every, ramp=args.ramp,
                               seed=args.seed, verbose=not args.quiet,
                               apply_step=not args.init_only, init_method=args.init_method)
    Y, B = result["Y"], result["B"]

    # ---- plot ------------------------------------------------------------
    bn = np.linalg.norm(B, axis=1)
    big = np.argsort(bn)[::-1][:200]
    if args.proj_dim == 3:
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=t, cmap="viridis", s=10,
                        alpha=0.85, linewidths=0)
        fig.colorbar(sc, ax=ax, label="t (intrinsic coordinate)", shrink=0.6, pad=0.08)
        if bn.max() > 0:
            sc_scale = 0.12 * (Y.max() - Y.min()) / bn.max()
            ax.quiver(Y[big, 0], Y[big, 1], Y[big, 2],
                      B[big, 0] * sc_scale, B[big, 1] * sc_scale, B[big, 2] * sc_scale,
                      color="k", alpha=0.6, linewidth=1.0, arrow_length_ratio=0.3)
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2"); ax.set_zlabel("dim 3")
    else:
        fig, ax = plt.subplots(figsize=(9, 8))
        sc = ax.scatter(Y[:, 0], Y[:, 1], c=t, cmap="viridis", s=10, alpha=0.85, linewidths=0)
        plt.colorbar(sc, ax=ax, label="t (intrinsic coordinate)")
        if bn.max() > 0:
            sc_scale = 0.12 * (Y.max() - Y.min()) / bn.max()
            ax.quiver(Y[big, 0], Y[big, 1], B[big, 0] * sc_scale, B[big, 1] * sc_scale,
                      color="k", alpha=0.6, width=0.004, scale=1, scale_units="xy")
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")

    if args.init_only:
        ax.set_title(f"Randers-UMAP swiss-roll (generated field), LOCATED INIT ONLY "
                     f"({args.init_method}, no training)  (n={n})", fontsize=11)
    else:
        ax.set_title(f"Randers-UMAP swiss-roll (generated field), located-drift init "
                     f"({args.init_method})  (n={n})", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)

    np.savez(f"{args.out}.npz", Y=Y, B=B, t=t, X=X, omega=omega,
             asymmetry_score=result.get("asymmetry_score", np.nan),
             asymmetry_per_node=result.get("asymmetry_per_node", np.array([])))

    if not args.quiet:
        print(f"\nwrote {args.out}.png and {args.out}.npz")

    # ---- snapshot grid: init -> every N epochs -> final, side by side -------
    if args.snapshot_every is not None and not args.init_only:
        snaps = result["snapshots"]
        n_snap = len(snaps)
        ncols = min(n_snap, 6)
        nrows = int(np.ceil(n_snap / ncols))
        vmin, vmax = t.min(), t.max()
        sc2 = None

        if args.proj_dim == 3:
            fig2 = plt.figure(figsize=(3.6 * ncols, 3.6 * nrows))
            axes2 = [fig2.add_subplot(nrows, ncols, idx + 1, projection="3d")
                     for idx in range(nrows * ncols)]
            for idx, snap in enumerate(snaps):
                ax2 = axes2[idx]
                Yi, Bi = snap["Y"], snap["B"]
                sc2 = ax2.scatter(Yi[:, 0], Yi[:, 1], Yi[:, 2], c=t, cmap="viridis",
                                  s=6, alpha=0.85, linewidths=0, vmin=vmin, vmax=vmax)
                bni = np.linalg.norm(Bi, axis=1)
                bigi = np.argsort(bni)[::-1][:200]
                if bni.max() > 0:
                    sc_scale_i = 0.12 * (Yi.max() - Yi.min()) / bni.max()
                    ax2.quiver(Yi[bigi, 0], Yi[bigi, 1], Yi[bigi, 2],
                              Bi[bigi, 0] * sc_scale_i, Bi[bigi, 1] * sc_scale_i,
                              Bi[bigi, 2] * sc_scale_i,
                              color="k", alpha=0.6, linewidth=0.8, arrow_length_ratio=0.3)
                ax2.set_title(f"epoch {snap['epoch']}", fontsize=9)
                ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([])
            for idx in range(n_snap, nrows * ncols):
                axes2[idx].axis("off")
        else:
            fig2, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows),
                                       squeeze=False)
            for idx, snap in enumerate(snaps):
                ax2 = axes[idx // ncols][idx % ncols]
                Yi, Bi = snap["Y"], snap["B"]
                sc2 = ax2.scatter(Yi[:, 0], Yi[:, 1], c=t, cmap="viridis", s=6,
                                  alpha=0.85, linewidths=0, vmin=vmin, vmax=vmax)
                bni = np.linalg.norm(Bi, axis=1)
                bigi = np.argsort(bni)[::-1][:200]
                if bni.max() > 0:
                    sc_scale_i = 0.12 * (Yi.max() - Yi.min()) / bni.max()
                    ax2.quiver(Yi[bigi, 0], Yi[bigi, 1],
                              Bi[bigi, 0] * sc_scale_i, Bi[bigi, 1] * sc_scale_i,
                              color="k", alpha=0.6, width=0.006, scale=1, scale_units="xy")
                ax2.set_title(f"epoch {snap['epoch']}", fontsize=9)
                ax2.set_xticks([]); ax2.set_yticks([])
            for idx in range(n_snap, nrows * ncols):
                axes[idx // ncols][idx % ncols].axis("off")

        fig2.suptitle(f"Randers-UMAP swiss-roll (generated field), apply-step trajectory  "
                      f"(n={n}, snapshot_every={args.snapshot_every})", fontsize=11)
        if sc2 is not None:
            fig2.colorbar(sc2, ax=fig2.get_axes(), label="t (intrinsic coordinate)",
                          fraction=0.02, pad=0.01)
        fig2.savefig(f"{args.out}_snapshots.png", dpi=150, bbox_inches="tight")

        if not args.quiet:
            print(f"wrote {args.out}_snapshots.png ({n_snap} snapshots)")


if __name__ == "__main__":
    main()
