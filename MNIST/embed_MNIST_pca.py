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
from randers_umap import (randers_umap_fit, arrow_scale, _compute_N,
                           compute_drift, knn_mask_from_distance_matrix)
from isumap_bridge import build_isumap_dist_matrix, isumap_style_init
from sphere_view import stereographic_project


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
    p.add_argument("--no-virtual-neighbor", action="store_true",
                    help="[OURS 2026-09-10, default ON] each node's own virtual point "
                         "xi_i=y_i+b_i is, BY DEFAULT, an unconditional (k+1)-th attractive "
                         "neighbour, pulled with UMAP's own attraction curve -- see "
                         "randers_umap.py's use_virtual_neighbor docstring, and "
                         "run_mammoth_isumap.py's own --no-virtual-neighbor flag (same "
                         "mechanism, same default) for the full explanation. Previously this "
                         "file never passed use_virtual_neighbor at all, silently falling back "
                         "to randers_umap_fit's own default (False) instead of matching the "
                         "rest of the isumap family -- fixed. Pass this flag to DISABLE it.")
    p.add_argument("--fixed-drift", action="store_true",
                    help="[OURS 2026-09-10] derive B ONCE from D_asym's own asymmetry at "
                         "Y_init and FREEZE it for the whole run, instead of the default live "
                         "mechanism (B recomputed from the CURRENT Y every epoch) -- see "
                         "run_swiss_roll_isumap.py's --fixed-drift help for the full "
                         "explanation. Off by default.")
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
    p.add_argument("--force-model", choices=["fr_gravity", "umap"], default="fr_gravity",
                    help="[OURS 2026-09-02] attraction/repulsion law passed to randers_umap_fit "
                         "-- 'fr_gravity' (default) = Bannister et al.'s spring/inverse-square "
                         "law, 'umap' = UMAP's own fitted (a,b)-curve. Was previously only "
                         "exposed on the swiss_roll/mammoth/sphere/isumap scripts, not here.")
    p.add_argument("--fr-k", type=float, default=None,
                    help="[OURS 2026-09-02] natural edge-length constant for force_model="
                         "fr_gravity (default None -> 1/sqrt(n)). Ignored for force_model=umap.")
    p.add_argument("--neg-sampling", action="store_true",
                    help="[OURS 2026-09-02] use TRUE stochastic negative sampling for repulsion "
                         "(n_negative_samples random points per node, drawn fresh every epoch) "
                         "instead of the dense/exact sum. See randers_umap_fit's own "
                         "negative_sampling docstring for the exact mechanism and rescaling.")
    p.add_argument("--sphere-view", action="store_true",
                    help="[OURS 2026-09-10] also save <out>_sphere.png: the trained 2D embedding "
                         "mapped onto a unit sphere via inverse stereographic projection (same "
                         "map-on-paper vs. map-on-a-globe idea) -- purely a post-training "
                         "visualization, does not change training in any way. See "
                         "randers_umap.stereographic_project's own docstring for the formula. "
                         "Off by default; the normal flat 2D plot is always still produced.")
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
    # [OURS 2026-08-26]
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

    # ---- IsUMap's own asymmetric distance -- SPARSE (pre-Dijkstra), fed
    # directly into randers_umap_fit's force computation. [OURS 2026-09-09]
    # Previously this file ran the i==j-filtered reconstruction + Dijkstra
    # itself right here, producing a DENSE D_asym and feeding THAT to
    # randers_umap_fit -- meaning the force computation's own k-NN selection
    # ran on the POST-Dijkstra geodesic graph (any of the n-1 other points
    # could end up "closest"), not the raw pre-Dijkstra star-graph structure
    # isumap's own ~k directly-listed neighbours actually encode. Empirically
    # verified (swiss_roll, n=500, k=15): 0/500 rows had the same k-NN SET
    # between the sparse and Dijkstra-completed versions of the same D_asym
    # -- a real, not cosmetic, difference. Now uses isumap_bridge.py's
    # build_isumap_dist_matrix() (which already carries the i==j-key fix
    # this file's own comment above used to explain by hand), byte-for-byte
    # the same function run_mammoth_isumap.py/run_swiss_roll_isumap.py/
    # run_sphere_isumap.py feed their own force computations -- this file's
    # pipeline is now structurally identical to those, just with MNIST's
    # PCA-reduced X.
    n = X_pca.shape[0]
    D_asym = build_isumap_dist_matrix(X_pca, k=args.k, verbose=verbose)

    # emb_k: some rows can have fewer real (finite) neighbours in the sparse
    # D_asym than --emb-k requests -- clip to the worst-case row, same guard
    # the isumap family scripts use.
    min_real_neighbors = int(np.isfinite(D_asym).sum(axis=1).min() - 1)
    emb_k = min(args.emb_k, max(min_real_neighbors, 1))
    if verbose:
        is_symmetric = np.allclose(D_asym, D_asym.T)
        print(f"D_asym: {D_asym.shape}  symmetric={is_symmetric}  (should be False)  "
              f"min real neighbours/row={min_real_neighbors}  emb_k used={emb_k}")

    np.save(os.path.join(save_dir, "asymm_matrix_pca.npy"), D_asym)
    np.save(os.path.join(save_dir, "labels_pca.npy"), y)
    np.save(os.path.join(save_dir, "X_pca.npy"), X_pca)

    # Y_init: IsUMap's own cMDS choice (real IsUMap's default -- see
    # isumap_style_init's own docstring), same as every other isumap-family
    # script. Built on a SEPARATE, throwaway Dijkstra-completed dense copy,
    # purely for this init -- does not replace the sparse D_asym fed to
    # randers_umap_fit above.
    Y_init = isumap_style_init(D_asym, d=2, seed=args.seed)

    # [OURS 2026-09-10] --fixed-drift: derive B ONCE from D_asym's own
    # asymmetry at Y_init, then freeze it for the whole run -- see the flag's
    # own help text above. B_fixed=None (default) keeps the live mechanism.
    if args.fixed_drift:
        if verbose:
            print(f"\nDeriving B ONCE from D_asym's own asymmetry at Y_init, then freezing it "
                  f"for the whole run (--fixed-drift)...")
        knn_mask_fixed = knn_mask_from_distance_matrix(D_asym, emb_k)
        N_fixed = _compute_N(D_asym)
        B_fixed = compute_drift(N_fixed, knn_mask_fixed, emb_k, Y_init, clip_delta=0.01)
    else:
        B_fixed = None

    # D_geo: the Euclidean-consistent weight source for randers_umap_fit's
    # weight-consistency fix (see run_swiss_roll_isumap.py's own docstring
    # for the full rationale and the both-finite-fallback bug fix -- a plain
    # (D_asym+D_asym.T)/2 average would be sparser than D_asym itself here).
    # [OURS 2026-09-09] Previously never wired through for MNIST at all.
    both_finite = np.isfinite(D_asym) & np.isfinite(D_asym.T)
    D_geo = np.where(both_finite, (D_asym + D_asym.T) / 2.0,
                      np.where(np.isfinite(D_asym), D_asym, D_asym.T))

    # ---- embed with our own randers_umap_fit -------------------------------
    # [OURS 2026-08-26] some pairs remain unreachable (inf) in the sparse
    # D_asym -- an inherent property of a directed k-NN graph, not a bug.
    # randers_umap_fit's own N computation already handles this safely
    # (np.where(isfinite(N), N, 0.0) zeroes out undefined pairs -- see its
    # own comment), but numpy still prints a RuntimeWarning for the
    # inf-inf/inf/inf arithmetic that produces those NaNs before they get
    # zeroed. Suppressed here (verified harmless via direct testing) purely
    # to keep the console output readable.
    with np.errstate(invalid="ignore", divide="ignore"):
        out = randers_umap_fit(D_asym, n_neighbors=emb_k, n_negative_samples=args.neg,
                                n_epochs=args.epochs, use_drift=True, B_fixed=B_fixed,
                                Y_init_override=Y_init, D_geo=D_geo,
                                snapshot_every=args.snapshot_every,
                                use_gravity=args.gravity,
                                gravity_strength=args.gravity_strength,
                                gravity_neighbor_weight=not args.no_gravity_neighbor_weight,
                                use_virtual_neighbor=not args.no_virtual_neighbor,
                                ramp=args.ramp,
                                force_model=args.force_model, fr_k=args.fr_k,
                                negative_sampling=args.neg_sampling,
                                seed=args.seed, verbose=verbose)
    Y, B = out["Y"], out["B"]

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
    ax.set_title(f"Randers-UMAP on MNIST (PCA {X.shape[1]}->{pca_dim}D first, "
                 f"n={n}, explained var={explained:.3f})", fontsize=10)
    fig.tight_layout()
    out_path = os.path.join(save_dir, f"{args.out}.png")
    fig.savefig(out_path, dpi=150)

    np.savez(os.path.join(save_dir, f"{args.out}.npz"), Y=Y, B=B, labels=y,
             pca_dim=pca_dim, explained_variance=explained)

    if verbose:
        print(f"\nwrote {out_path} and {args.out}.npz (in {save_dir})")

    # ---- sphere view: same trained Y, mapped onto a unit sphere via inverse
    # stereographic projection (map-on-paper vs. map-on-a-globe) -- purely a
    # post-training visualization, --sphere-view opt-in, training itself is
    # completely untouched. [OURS 2026-09-10]
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
        ax_s.set_title(f"Randers-UMAP on MNIST (PCA {X.shape[1]}->{pca_dim}D, sphere view, "
                        f"n={n})", fontsize=10)
        fig_s.tight_layout()
        sphere_path = os.path.join(save_dir, f"{args.out}_sphere.png")
        fig_s.savefig(sphere_path, dpi=150)
        if verbose:
            print(f"wrote {sphere_path}")

    # ---- snapshot grid: init -> every N epochs -> final, side by side -----
    # [OURS 2026-08-26]
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
            bigi = np.argsort(bni)[::-1][:100]
            if bni.max() > 0:
                sc_scale_i = arrow_scale(Yi, bni)
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
