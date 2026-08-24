#!/usr/bin/env python3
"""
run_sphere_tangential.py -- Randers-UMAP on a synthetic sphere point cloud +
a TANGENTIAL (azimuthal) Randers field, using the EXACT SAME pipeline as
run_swiss_roll.py / run_mammoth.py (run_located_drift, imported directly,
not duplicated).

[OURS 2026-08-19, per explicit user request] New synthetic dataset,
parallel to the swiss roll: a sphere has no boundary and no single natural
"unrolled" 1D coordinate like the swiss roll's t, so (like mammoth) we use
an intrinsic coordinate purely for colouring plots -- here, theta, the
colatitude (angle from the north pole, 0 at the north pole, pi at the
south pole).

This file defines make_sphere_points() (pure geometry, no drift -- reused
by run_sphere_radial.py and run_sphere_isumap.py) and the TANGENTIAL field.
See run_sphere_radial.py for the RADIAL field and a discussion of why the
two are expected to behave very differently under this pipeline.

Tangential field
----------------
omega_i is the azimuthal ("eastward", d/dphi) unit direction at x_i,
scaled to ||omega_i|| = alpha everywhere:

    omega_i \\propto (-y_i, x_i, 0)          (rotation about the z-axis)

This is tangent to the sphere at every point (it is, by construction,
orthogonal to the radial direction x_i itself: (-y,x,0) . (x,y,z) = 0), so
it is a clean, everywhere-valid example of a Randers drift that never
points off the manifold. It degenerates (direction undefined, though
magnitude still -> alpha after normalisation) only exactly at the two
poles (x=y=0) -- a measure-zero set under continuous sampling, guarded
with a small epsilon in the normalisation below.

Because this field is tangent to the sphere, it is ALIGNED with the local
displacement direction between nearby neighbours (x_j - x_i for j near i
is itself dominantly tangential to the sphere) -- so <omega_i, x_j-x_i> is
generically large in magnitude, and we expect a strong, clean asymmetry
signal in D_asym and a strong recovered drift. Contrast with
run_sphere_radial.py's field, which is expected to do the opposite.

Usage
-----
    python run_sphere_tangential.py
    python run_sphere_tangential.py --n 2000 --epochs 500
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

from run_swiss_roll import run_located_drift


def make_sphere_points(n, seed=42, radius=10.0):
    """
    n points sampled uniformly on the surface of a sphere of the given
    radius (via normalised isotropic Gaussians -- NOT uniform theta/phi,
    which would over-sample the poles).

    Returns
    -------
    X     : (n,3) ambient points, ||x_i|| == radius
    theta : (n,) colatitude, angle from +z axis, in [0, pi] (0 = north pole)
    phi   : (n,) azimuth, angle in the xy-plane, in [-pi, pi]
    """
    rng = np.random.RandomState(seed)
    vec = rng.normal(size=(n, 3))
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    X = vec * radius

    theta = np.arccos(np.clip(X[:, 2] / radius, -1.0, 1.0))
    phi = np.arctan2(X[:, 1], X[:, 0])
    return X, theta, phi


def make_sphere_tangential_randers(n, seed=42, radius=10.0, alpha=0.5):
    """
    Sphere point cloud + the tangential (azimuthal) Randers field described
    in the module docstring.

    Returns
    -------
    X     : (n,3) ambient points
    omega : (n,3) Randers drift vectors, ||omega_i|| == alpha (constant
            magnitude everywhere -- unlike swiss roll/mammoth, whose fields
            are magnitude-modulated; here only the DIRECTION type
            (tangential vs radial) is the experimental variable)
    theta : (n,) colatitude -- used purely for colouring plots
    """
    X, theta, phi = make_sphere_points(n, seed=seed, radius=radius)

    tangent = np.column_stack([-X[:, 1], X[:, 0], np.zeros(n)])
    tnorm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = tangent / np.maximum(tnorm, 1e-9)
    omega = tangent * alpha

    return X, omega, theta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n",      type=int, default=1000)
    p.add_argument("--radius", type=float, default=10.0)
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
                    help="locate step's placement method. 'isomap' (default) = classical_mds. "
                         "'umap' = fuzzy_simplicial_set+spectral_layout.")
    p.add_argument("--alpha", type=float, default=0.5, help="||omega|| for the tangential drift field (constant)")
    p.add_argument("--proj-dim", type=int, default=2, choices=[2, 3],
                    help="[OURS 2026-08-20, per explicit user request] embedding "
                         "dimension for the locate step's placement AND the apply "
                         "step's force-directed training -- see run_swiss_roll.py's "
                         "--proj-dim help for the full explanation. 3 = full 3D "
                         "layout, main scatter plot switches to 3D axes automatically.")
    p.add_argument("--adjacency", choices=["threshold", "knn"], default="threshold",
                    help="[OURS 2026-08-20, per explicit user request] how "
                         "compute_dist_matrix builds its base adjacency graph -- see "
                         "run_swiss_roll.py's --adjacency help / randers_bridge."
                         "compute_dist_matrix's adjacency docstring for the full "
                         "explanation. 'threshold' (default) = unchanged. 'knn' = TRUE "
                         "per-point k-nearest-neighbours membership, asymmetric in general.")
    p.add_argument("--seed",   type=int, default=0)
    p.add_argument("--out",    default="sphere_tangential_embedding")
    p.add_argument("--quiet",  action="store_true")
    args = p.parse_args()

    if not args.quiet:
        print(f"Generating sphere (tangential field): n={args.n}, radius={args.radius}")
    X, omega, theta = make_sphere_tangential_randers(args.n, seed=42, radius=args.radius, alpha=args.alpha)
    n = args.n

    # ---- 3D plot of the ambient sphere with the omega (Randers) field -------
    fig3d = plt.figure(figsize=(11, 9))
    ax3d = fig3d.add_subplot(111, projection="3d")
    sc3d = ax3d.scatter(X[:, 0], X[:, 1], X[:, 2], c=theta, cmap="viridis", s=8,
                        alpha=0.85, linewidths=0)
    fig3d.colorbar(sc3d, ax=ax3d, label="theta (colatitude, 0=north pole)", shrink=0.6, pad=0.08)
    rng3d = np.random.RandomState(0)
    idx3d = rng3d.choice(n, size=min(200, n), replace=False)
    span = X.max(axis=0) - X.min(axis=0)
    omega_norm_max = np.linalg.norm(omega, axis=1).max()
    scale3d = 0.35 * span.max() / max(omega_norm_max, 1e-12)
    ax3d.quiver(X[idx3d, 0], X[idx3d, 1], X[idx3d, 2],
                omega[idx3d, 0] * scale3d, omega[idx3d, 1] * scale3d, omega[idx3d, 2] * scale3d,
                color="k", alpha=0.7, linewidth=1.0, arrow_length_ratio=0.3)
    ax3d.set_title(f"Sphere, TANGENTIAL field (ambient X, n={n}, radius={args.radius})", fontsize=11)
    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
    fig3d.tight_layout()
    fig3d.savefig(f"{args.out}_3d_field.png", dpi=150)
    if not args.quiet:
        print(f"wrote {args.out}_3d_field.png")

    # ---- 3D plot of the initial data with the drift ATTACHED (x_i -> x_i+omega_i), ----
    # ---- exaggerated to be visible, drawn as crimson quiver arrows -- same -----
    # ---- convention as run_mammoth.py's drift-attached plot. ------------------
    X_virtual_display = X + omega * scale3d
    fig3d_v = plt.figure(figsize=(11, 9))
    ax3d_v = fig3d_v.add_subplot(111, projection="3d")
    sc3d_v = ax3d_v.scatter(X[:, 0], X[:, 1], X[:, 2], c=theta, cmap="viridis", s=8,
                             alpha=0.85, linewidths=0)
    fig3d_v.colorbar(sc3d_v, ax=ax3d_v, label="theta (colatitude, 0=north pole)", shrink=0.6, pad=0.08)
    ax3d_v.quiver(X[idx3d, 0], X[idx3d, 1], X[idx3d, 2],
                  X_virtual_display[idx3d, 0] - X[idx3d, 0],
                  X_virtual_display[idx3d, 1] - X[idx3d, 1],
                  X_virtual_display[idx3d, 2] - X[idx3d, 2],
                  color="crimson", alpha=0.9, linewidth=1.2, arrow_length_ratio=0.25)
    ax3d_v.set_title(f"Sphere, TANGENTIAL: drift attached (x_i -> x_i+omega_i) "
                      f"(n={n}, omega exaggerated x{scale3d:.1f} for visibility)", fontsize=10)
    ax3d_v.set_xlabel("x"); ax3d_v.set_ylabel("y"); ax3d_v.set_zlabel("z")
    fig3d_v.tight_layout()
    fig3d_v.savefig(f"{args.out}_3d_drift_attached.png", dpi=150)
    if not args.quiet:
        print(f"wrote {args.out}_3d_drift_attached.png")

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

    if args.init_only:
        ax.set_title(f"Randers-UMAP sphere (tangential), LOCATED INIT ONLY ({args.init_method}, no training)  (n={n})", fontsize=11)
    else:
        ax.set_title(f"Randers-UMAP sphere (tangential), located-drift init ({args.init_method})  (n={n})", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)

    np.savez(f"{args.out}.npz", Y=Y, B=B, theta=theta, X=X, omega=omega,
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

        fig2.suptitle(f"Randers-UMAP sphere (tangential), apply-step trajectory  (n={n}, "
                      f"snapshot_every={args.snapshot_every})", fontsize=11)
        if sc2 is not None:
            fig2.colorbar(sc2, ax=fig2.get_axes(), label="theta (colatitude)",
                          fraction=0.02, pad=0.01)
        fig2.savefig(f"{args.out}_snapshots.png", dpi=150, bbox_inches="tight")

        if not args.quiet:
            print(f"wrote {args.out}_snapshots.png ({n_snap} snapshots)")


if __name__ == "__main__":
    main()
