#!/usr/bin/env python3
"""
run_mammoth.py -- Randers-UMAP on the mammoth point cloud (mammoth.csv) +
a hand-crafted Randers field, using the EXACT SAME pipeline as
run_swiss_roll.py (run_located_drift, imported directly, not duplicated).

[OURS 2026-08-16] Only the dataset changes here: mammoth.csv (~1M raw points,
subsampled) instead of the synthetic swiss roll. Everything downstream
(D_asym construction via randers_bridge.compute_dist_matrix, the locate-
then-attach B mechanism, the apply-step force-directed training) is
identical to run_swiss_roll.py -- reused via import, not copy-pasted.

Randers field on mammoth
-------------------------
mammoth.csv has no natural parametrisation (unlike the swiss roll's t),
and neither the IsUMap paper nor DAGES's own paper/code define a drift
field for this dataset -- there's no ground truth to replicate here.
We hand-craft a
DAGES-style global-axis field, the same spirit as their river/sea examples
in main_2D_maps.py (a fixed flow direction, magnitude shaped by position,
NOT derived from local manifold geometry):

    direction : constant +z (z is the tail(-)->head(+) axis in this file's
                orientation -- verified empirically: tusks/head cluster at
                z~150-170, tail curls down at z~-170, y~100-250)
    magnitude : 0.4 + 0.6 * normalized_height(y) -- stronger along the
                spine/head (high y), tapering toward the legs (low y),
                mirroring how DAGES's river field tapers across width.

Usage
-----
    python run_mammoth.py
    python run_mammoth.py --n 1500 --epochs 500

Outputs
-------
    <out>.png   embedding coloured by z (head<->tail position), drift arrows
    <out>.npz   Y, B, z, X, omega
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from randers_bridge import run_located_drift

MAMMOTH_CSV = HERE / "mammoth.csv"


def make_mammoth_randers(n, seed=42, csv_path=MAMMOTH_CSV, alpha=0.5):
    """
    Load mammoth.csv, subsample to n points, build the hand-crafted
    global-axis Randers field described in the module docstring.

    Returns
    -------
    X     : (n,3) ambient points
    omega : (n,3) Randers drift vectors, ||omega_i|| <= alpha
    z     : (n,) the point's own z-coordinate (head<->tail position),
            returned in place of swiss roll's t -- used purely for
            colouring plots, no intrinsic-manifold meaning implied.
    """
    df = pd.read_csv(csv_path)
    X_all = df[["x", "y", "z"]].values.astype(np.float64)

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X_all), size=n, replace=False)
    X = X_all[idx]

    y = X[:, 1]
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)

    omega = np.zeros_like(X)
    omega[:, 2] = 1.0
    mag = 0.4 + 0.6 * y_norm
    omega = omega * mag[:, None]
    omega = omega / np.linalg.norm(omega, axis=1, keepdims=True).max(initial=1e-12)
    omega = omega * alpha

    z = X[:, 2]
    return X, omega, z


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n",      type=int, default=1000)
    p.add_argument("--k",      type=int, default=20, help="k-NN for the geodesic backbone")
    p.add_argument("--neg",    type=int, default=10)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--locate-epochs", type=int, default=500,
                    help="no-op -- locate step is a single deterministic placement call, no "
                         "training. Kept for compat with run_swiss_roll.py's run_located_drift signature.")
    p.add_argument("--clip-delta", type=float, default=0.01)
    p.add_argument("--gravity", action="store_true",
                    help="[OURS 2026-08-17] add per-node gravity toward xi_i=y_i+b_i "
                         "(Bannister et al. f_g=gamma*M[i]*b_i).")
    p.add_argument("--gravity-strength", type=float, default=1.0)
    p.add_argument("--no-gravity-neighbor-weight", action="store_true")
    p.add_argument("--no-virtual-neighbor", action="store_true",
                    help="[OURS 2026-08-20, default ON] each node's own virtual point "
                         "xi_i=y_i+b_i is, BY DEFAULT, an unconditional (k+1)-th attractive "
                         "neighbour, pulled with UMAP's own attraction curve -- see "
                         "run_swiss_roll.py's --no-virtual-neighbor help for the full "
                         "explanation. Pass this flag to DISABLE it.")
    p.add_argument("--snapshot-every", type=int, default=None)
    p.add_argument("--ramp", action="store_true")
    p.add_argument("--init-only", action="store_true")
    p.add_argument("--init-method", choices=["umap", "isomap"], default="isomap",
                    help="[OURS 2026-08-16, default changed to 'isomap' 2026-08-18] locate "
                         "step's placement method. 'isomap' (default) = "
                         "classical_mds (Isomap's own finishing step). 'umap' is kept for "
                         "backward CLI compat but is now IDENTICAL to 'isomap' -- "
                         "spectral_layout was removed project-wide (see "
                         "randers_umap.classical_mds's own docstring).")
    p.add_argument("--alpha", type=float, default=0.5, help="max ||omega|| for the mammoth drift field")
    p.add_argument("--proj-dim", type=int, default=2, choices=[2, 3],
                    help="[OURS 2026-08-20] embedding "
                         "dimension for the locate step's placement AND the apply "
                         "step's force-directed training -- see run_swiss_roll.py's "
                         "--proj-dim help for the full explanation. 3 = full 3D "
                         "layout, main scatter plot switches to 3D axes automatically.")
    p.add_argument("--adjacency", choices=["threshold", "knn"], default="threshold",
                    help="[OURS 2026-08-20] how "
                         "compute_dist_matrix builds its base adjacency graph -- see "
                         "run_swiss_roll.py's --adjacency help / randers_bridge."
                         "compute_dist_matrix's adjacency docstring for the full "
                         "explanation. 'threshold' (default) = unchanged. 'knn' = TRUE "
                         "per-point k-nearest-neighbours membership, asymmetric in general.")
    p.add_argument("--normalize", action="store_true",
                    help="[OURS 2026-08-25] default OFF -- "
                         "B_located used as-is (prior behaviour, unchanged). If given, "
                         "replace each node's drift MAGNITUDE, every epoch, with its "
                         "LIVE distance to its own k-th nearest neighbour in the "
                         "current embedding, keeping its DIRECTION as located. See "
                         "randers_umap_fit's scale_B_fixed_by_knn_distance docstring "
                         "for the exact mechanism.")
    p.add_argument("--force-model", choices=["fr_gravity", "umap"], default="fr_gravity",
                    help="[OURS 2026-08-31] see run_swiss_roll.py's --force-model help -- "
                         "'fr_gravity' (NEW DEFAULT) = Bannister et al.'s own "
                         "Fruchterman-Reingold-style forces, 'umap' = original UMAP "
                         "(a,b)-curve law.")
    p.add_argument("--fr-k", type=float, default=None,
                    help="natural edge length for --force-model fr_gravity. None uses sqrt(1/n).")
    p.add_argument("--neg-sampling", action="store_true",
                    help="[OURS 2026-08-31] only affects --force-model umap -- see "
                         "run_swiss_roll.py's --neg-sampling help / randers_umap_fit's "
                         "negative_sampling docstring for the full explanation.")
    p.add_argument("--seed",   type=int, default=0)
    p.add_argument("--out",    default="mammoth_embedding")
    p.add_argument("--quiet",  action="store_true")
    args = p.parse_args()

    if not args.quiet:
        print(f"Loading mammoth.csv, subsampling to n={args.n}")
    X, omega, z = make_mammoth_randers(args.n, seed=42, alpha=args.alpha)
    n = args.n

    # ---- 3D plot of the ambient mammoth with the omega (Randers) field ------
    fig3d = plt.figure(figsize=(11, 9))
    ax3d = fig3d.add_subplot(111, projection="3d")
    sc3d = ax3d.scatter(X[:, 0], X[:, 1], X[:, 2], c=z, cmap="viridis", s=8,
                        alpha=0.85, linewidths=0)
    fig3d.colorbar(sc3d, ax=ax3d, label="z (tail<->head)", shrink=0.6, pad=0.08)
    rng3d = np.random.RandomState(0)
    idx3d = rng3d.choice(n, size=min(200, n), replace=False)
    scale3d = 12.0
    ax3d.quiver(X[idx3d, 0], X[idx3d, 1], X[idx3d, 2],
                omega[idx3d, 0] * scale3d, omega[idx3d, 1] * scale3d, omega[idx3d, 2] * scale3d,
                color="crimson", alpha=0.8, linewidth=1.0, arrow_length_ratio=0.3)
    ax3d.set_title(f"Mammoth (ambient X, n={n}) with omega field", fontsize=11)
    ax3d.set_xlabel("x"); ax3d.set_ylabel("y (height)"); ax3d.set_zlabel("z (tail<->head)")
    fig3d.tight_layout()
    fig3d.savefig(f"{args.out}_3d_field.png", dpi=150)
    if not args.quiet:
        print(f"wrote {args.out}_3d_field.png")

    # [OURS 2026-08-19] 3D plot of the initial
    # data with the drift ATTACHED -- i.e. the "virtual" cloud X+omega,
    # the actual point each x_i's drift vector points to (same X_virtual
    # concept used everywhere else in this project, e.g. the old
    # virtual-point locate mechanism). ||omega_i|| is only 0.2-0.5 while X
    # itself spans hundreds of units, so X+omega at TRUE scale would sit
    # visually on top of X, indistinguishable -- exaggerated by the SAME
    # factor (scale3d=12) already used for the quiver arrows above, purely
    # for this visualisation, so the two plots agree on what "the arrows"
    # mean. This does NOT touch the true omega used anywhere downstream
    # (locate step, D_asym, etc.) -- display-only.
    # [OURS 2026-08-19] drawn as crimson quiver arrows FROM each x_i TO its
    # attached virtual point x_i+omega_i, same subsample/style as the field
    # plot above, rather than (or in addition to) two overlaid clouds.
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
    ax3d_v.set_title(f"Mammoth: drift attached (x_i -> x_i+omega_i) "
                      f"(n={n}, omega exaggerated x{scale3d:.0f} for visibility)", fontsize=10)
    ax3d_v.set_xlabel("x"); ax3d_v.set_ylabel("y (height)"); ax3d_v.set_zlabel("z (tail<->head)")
    fig3d_v.tight_layout()
    fig3d_v.savefig(f"{args.out}_3d_drift_attached.png", dpi=150)
    if not args.quiet:
        print(f"wrote {args.out}_3d_drift_attached.png")

    result = run_located_drift(X, omega, k=args.k, emb_k=args.k, neg=args.neg,
                               locate_epochs=args.locate_epochs, epochs=args.epochs,
                               clip_delta=args.clip_delta, use_gravity=args.gravity,
                               snapshot_every=args.snapshot_every, ramp=args.ramp,
                               gravity_strength=args.gravity_strength,
                               gravity_neighbor_weight=not args.no_gravity_neighbor_weight,
                               use_virtual_neighbor=not args.no_virtual_neighbor,
                               proj_dim=args.proj_dim, adjacency=args.adjacency,
                               seed=args.seed, verbose=not args.quiet,
                               apply_step=not args.init_only, init_method=args.init_method,
                               normalize_drift_by_asymmetry=args.normalize,
                               force_model=args.force_model, fr_k=args.fr_k,
                               negative_sampling=args.neg_sampling)
    Y, B = result["Y"], result["B"]

    # ---- plot ------------------------------------------------------------
    # [OURS 2026-08-20] proj_dim==3 -> 3D scatter
    # + 3D quiver; proj_dim==2 -> unchanged original 2D plot.
    bn = np.linalg.norm(B, axis=1)
    big = np.argsort(bn)[::-1][:200]
    if args.proj_dim == 3:
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=z, cmap="viridis", s=10,
                        alpha=0.85, linewidths=0)
        fig.colorbar(sc, ax=ax, label="z (tail<->head)", shrink=0.6, pad=0.08)
        if bn.max() > 0:
            sc_scale = 0.12 * (Y.max() - Y.min()) / bn.max()
            ax.quiver(Y[big, 0], Y[big, 1], Y[big, 2],
                      B[big, 0] * sc_scale, B[big, 1] * sc_scale, B[big, 2] * sc_scale,
                      color="k", alpha=0.6, linewidth=1.0, arrow_length_ratio=0.3)
        # [OURS 2026-08-20] matplotlib's DEFAULT 3D
        # tick/box axes, matching FinslerMDS's utils.plot_points -- no manual
        # origin-crossing lines, no set_xticks([]) hiding. Numbers stay on.
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2"); ax.set_zlabel("dim 3")
    else:
        fig, ax = plt.subplots(figsize=(9, 8))
        sc = ax.scatter(Y[:, 0], Y[:, 1], c=z, cmap="viridis", s=10, alpha=0.85, linewidths=0)
        plt.colorbar(sc, ax=ax, label="z (tail<->head)")
        if bn.max() > 0:
            sc_scale = 0.12 * (Y.max() - Y.min()) / bn.max()
            ax.quiver(Y[big, 0], Y[big, 1], B[big, 0] * sc_scale, B[big, 1] * sc_scale,
                      color="k", alpha=0.6, width=0.004, scale=1, scale_units="xy")
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")

    if args.init_only:
        ax.set_title(f"Randers-UMAP mammoth, LOCATED INIT ONLY ({args.init_method}, no training)  (n={n})", fontsize=11)
    else:
        ax.set_title(f"Randers-UMAP mammoth, located-drift init ({args.init_method})  "
                     f"(n={n}, epochs={args.epochs})", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)

    np.savez(f"{args.out}.npz", Y=Y, B=B, z=z, X=X, omega=omega,
             asymmetry_score=result.get("asymmetry_score", np.nan),
             asymmetry_per_node=result.get("asymmetry_per_node", np.array([])))

    if not args.quiet:
        print(f"\nwrote {args.out}.png and {args.out}.npz")

    if args.snapshot_every is not None and not args.init_only:
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
                    sc_scale_i = 0.12 * (Yi.max() - Yi.min()) / bni.max()
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
                    sc_scale_i = 0.12 * (Yi.max() - Yi.min()) / bni.max()
                    ax2.quiver(Yi[bigi, 0], Yi[bigi, 1],
                              Bi[bigi, 0] * sc_scale_i, Bi[bigi, 1] * sc_scale_i,
                              color="k", alpha=0.6, width=0.006, scale=1, scale_units="xy")
                ax2.set_title(f"epoch {snap['epoch']}", fontsize=9)
                ax2.set_xticks([]); ax2.set_yticks([])
            for idx in range(n_snap, nrows * ncols):
                axes[idx // ncols][idx % ncols].axis("off")

        fig2.suptitle(f"Randers-UMAP mammoth, apply-step trajectory  (n={n}, "
                      f"snapshot_every={args.snapshot_every})", fontsize=11)
        if sc2 is not None:
            fig2.colorbar(sc2, ax=fig2.get_axes(), label="z (tail<->head)",
                          fraction=0.02, pad=0.01)
        fig2.savefig(f"{args.out}_snapshots.png", dpi=150, bbox_inches="tight")

        if not args.quiet:
            print(f"wrote {args.out}_snapshots.png ({n_snap} snapshots)")


if __name__ == "__main__":
    main()
