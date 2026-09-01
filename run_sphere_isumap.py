#!/usr/bin/env python3
"""
run_sphere_isumap.py -- sphere point cloud, but the distance matrix comes
from distance_graph_generation() (the isumap method), NOT from
randers_bridge.compute_dist_matrix() with an injected omega. EXACT SAME
pipeline as run_swiss_roll_isumap.py / run_mammoth_isumap.py
(build_isumap_dist_matrix, imported directly, not duplicated) -- only the
dataset changes.

Unlike run_sphere_tangential.py / run_sphere_radial.py, this script injects
NO vector field at all: distance_graph_generation()'s asymmetry comes
purely from the directed k-NN/star-graph structure of the raw point cloud
X (make_sphere_points() -- no omega parameter exists in that function).
The point of this script is the same as its swiss-roll/mammoth
counterparts: does a drift signal recovered purely from an OBSERVED
asymmetric dissimilarity matrix (no privileged access to any ground-truth
field) still produce a sensible, drift-like embedding?

[per the same 2026-08-18/19 back-and-forth documented in
run_swiss_roll_isumap.py's module docstring] B is derived LIVE, every
epoch, purely from D_asym's own asymmetry (compute_drift on
N=(D_asym-D_asym.T)/(D_asym+D_asym.T), no omega anywhere) -- this is the
final, reverted-to state for the other two isumap scripts, applied here
from the start rather than going through the same locate-then-freeze
detour.

Usage
-----
    python run_sphere_isumap.py
    python run_sphere_isumap.py --n 2000 --epochs 500
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

from run_sphere_tangential import make_sphere_points
from run_swiss_roll_isumap import build_isumap_dist_matrix
from randers_umap import randers_umap_fit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--radius", type=float, default=10.0)
    p.add_argument("--k", type=int, default=30, help="k for distance_graph_generation (isumap's own D_asym)")
    p.add_argument("--neg", type=int, default=10)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--gravity", action="store_true",
                    help="[OURS 2026-08-17] add per-node gravity toward xi_i=y_i+b_i "
                         "(Bannister et al. f_g=gamma*M[i]*b_i).")
    p.add_argument("--gravity-strength", type=float, default=1.0)
    p.add_argument("--no-gravity-neighbor-weight", action="store_true")
    p.add_argument("--no-virtual-neighbor", action="store_true",
                    help="[OURS 2026-08-20, default ON] each node's own virtual point "
                         "xi_i=y_i+b_i is, BY DEFAULT, an unconditional (k+1)-th attractive "
                         "neighbour, pulled with UMAP's own attraction curve -- see "
                         "randers_umap.py's use_virtual_neighbor docstring for the full "
                         "explanation. Pass this flag to DISABLE it.")
    p.add_argument("--clip-delta", type=float, default=0.01)
    p.add_argument("--ramp", action="store_true")
    p.add_argument("--init-only", action="store_true",
                    help="stop before force-directed training -- runs a single epoch with an "
                         "internal epoch-0 snapshot and returns that pre-training state.")
    p.add_argument("--snapshot-every", type=int, default=None)
    p.add_argument("--proj-dim", type=int, default=2, choices=[2, 3],
                    help="[OURS 2026-08-20, per explicit user request] embedding "
                         "dimension for randers_umap_fit's own internal spectral "
                         "init AND the apply-step training -- see run_swiss_roll.py's "
                         "--proj-dim help for the full explanation. 3 = full 3D "
                         "layout, main scatter plot switches to 3D axes automatically.")
    p.add_argument("--force-model", choices=["fr_gravity", "umap"], default="fr_gravity",
                    help="[OURS 2026-08-28, per explicit user request] see run_swiss_roll.py's "
                         "--force-model help -- 'fr_gravity' (NEW DEFAULT) = Bannister et al.'s "
                         "own Fruchterman-Reingold-style forces, 'umap' = original UMAP "
                         "(a,b)-curve law.")
    p.add_argument("--fr-k", type=float, default=None,
                    help="natural edge length for --force-model fr_gravity. None uses sqrt(1/n).")
    p.add_argument("--neg-sampling", action="store_true",
                    help="[OURS 2026-08-31] only affects --force-model umap -- see "
                         "run_swiss_roll.py's --neg-sampling help / randers_umap_fit's "
                         "negative_sampling docstring for the full explanation.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="sphere_embedding_isumap")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    apply_epochs = 1 if args.init_only else args.epochs
    apply_snapshot_every = 1 if args.init_only else args.snapshot_every

    if not args.quiet:
        print(f"Generating sphere: n={args.n}, radius={args.radius}")
    X, theta, phi = make_sphere_points(args.n, seed=42, radius=args.radius)

    if not args.quiet:
        print(f"\nBuilding distance matrix via distance_graph_generation (data_D, unfixed)...")
    D_asym = build_isumap_dist_matrix(X, k=args.k, verbose=not args.quiet)

    min_real_neighbors = int(np.isfinite(D_asym).sum(axis=1).min() - 1)
    emb_k = min(args.k, max(min_real_neighbors, 1))
    if not args.quiet:
        print(f"D_asym: {D_asym.shape}  symmetric={np.allclose(D_asym, D_asym.T)}  "
              f"min real neighbours/row={min_real_neighbors}  emb_k used={emb_k}")

    if not args.quiet:
        print(f"\nDeriving B live from D_asym's own asymmetry (no omega used) each epoch...")
    out = randers_umap_fit(D_asym, n_neighbors=emb_k, n_negative_samples=args.neg,
                            n_epochs=apply_epochs, use_drift=True, B_fixed=None,
                            d=args.proj_dim,
                            clip_delta=args.clip_delta,
                            use_gravity=args.gravity, gravity_strength=args.gravity_strength,
                            gravity_neighbor_weight=not args.no_gravity_neighbor_weight,
                            use_virtual_neighbor=not args.no_virtual_neighbor,
                            ramp=args.ramp, seed=args.seed,
                            snapshot_every=apply_snapshot_every, verbose=not args.quiet,
                            force_model=args.force_model, fr_k=args.fr_k,
                            negative_sampling=args.neg_sampling)

    if args.init_only:
        Y, B = out["snapshots"][0]["Y"], out["snapshots"][0]["B"]
    else:
        Y, B = out["Y"], out["B"]

    # ---- plot --------------------------------------------------------
    # [OURS 2026-08-20, per explicit user request] proj_dim==3 -> 3D scatter
    # + 3D quiver; proj_dim==2 -> unchanged original 2D plot.
    bn = np.linalg.norm(B, axis=1)
    big = np.argsort(bn)[::-1][:200]
    if args.proj_dim == 3:
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=theta, cmap="viridis", s=10,
                        alpha=0.85, linewidths=0)
        fig.colorbar(sc, ax=ax, label="theta (colatitude)", shrink=0.6, pad=0.08)
        if bn.max() > 0:
            sc_scale = 0.12 * (Y.max() - Y.min()) / bn.max()
            ax.quiver(Y[big, 0], Y[big, 1], Y[big, 2],
                      B[big, 0] * sc_scale, B[big, 1] * sc_scale, B[big, 2] * sc_scale,
                      color="k", alpha=0.6, linewidth=1.0, arrow_length_ratio=0.3)
        # [OURS 2026-08-20, per explicit user request -- "finslermds'in swiss
        # rollda haliyle yaptığı gibi eksene oturt"] matplotlib's DEFAULT 3D
        # tick/box axes, matching FinslerMDS's utils.plot_points -- no manual
        # origin-crossing lines, no set_xticks([]) hiding. Numbers stay on.
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2"); ax.set_zlabel("dim 3")
    else:
        fig, ax = plt.subplots(figsize=(9, 8))
        sc = ax.scatter(Y[:, 0], Y[:, 1], c=theta, cmap="viridis", s=10, alpha=0.85, linewidths=0)
        plt.colorbar(sc, ax=ax, label="theta (colatitude)")
        if bn.max() > 0:
            sc_scale = 0.12 * (Y.max() - Y.min()) / bn.max()
            ax.quiver(Y[big, 0], Y[big, 1], B[big, 0] * sc_scale, B[big, 1] * sc_scale,
                      color="k", alpha=0.6, width=0.004, scale=1, scale_units="xy")
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")

    drift_label = "live B (from D_asym asymmetry only)"
    init_suffix = ", INIT ONLY (no training)" if args.init_only else f", epochs={args.epochs}"
    ax.set_title(f"Randers-UMAP sphere, isumap-derived D, {drift_label}{init_suffix}  (n={args.n})", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)

    np.savez(f"{args.out}.npz", Y=Y, B=B, theta=theta, phi=phi, X=X)

    if not args.quiet:
        print(f"\nwrote {args.out}.png and {args.out}.npz")

    if args.snapshot_every is not None and not args.init_only:
        snaps = out["snapshots"]
        n_snap = len(snaps)
        ncols = min(n_snap, 6)
        nrows = int(np.ceil(n_snap / ncols))
        vmin, vmax = theta.min(), theta.max()
        sc2 = None

        if args.proj_dim == 3:
            fig2 = plt.figure(figsize=(3.6 * ncols, 3.6 * nrows))
            axes2 = [fig2.add_subplot(nrows, ncols, idx + 1, projection="3d")
                     for idx in range(nrows * ncols)]
            for idx, snap in enumerate(snaps):
                ax2 = axes2[idx]
                Yi, Bi = snap["Y"], snap["B"]
                sc2 = ax2.scatter(Yi[:, 0], Yi[:, 1], Yi[:, 2], c=theta, cmap="viridis",
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
                sc2 = ax2.scatter(Yi[:, 0], Yi[:, 1], c=theta, cmap="viridis", s=6,
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

        fig2.suptitle(f"Randers-UMAP sphere, isumap D, apply-step trajectory  (n={args.n}, "
                      f"snapshot_every={args.snapshot_every})", fontsize=11)
        if sc2 is not None:
            fig2.colorbar(sc2, ax=fig2.get_axes(), label="theta (colatitude)",
                          fraction=0.02, pad=0.01)
        fig2.savefig(f"{args.out}_snapshots.png", dpi=150, bbox_inches="tight")

        if not args.quiet:
            print(f"wrote {args.out}_snapshots.png ({n_snap} snapshots)")


if __name__ == "__main__":
    main()
