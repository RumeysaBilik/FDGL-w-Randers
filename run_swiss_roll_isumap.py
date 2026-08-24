#!/usr/bin/env python3
"""
run_swiss_roll_isumap.py -- swiss roll, but the distance matrix comes from
distance_graph_generation() (the isumap method used in asymm_dist_MNIST.py),
NOT from randers_bridge.compute_dist_matrix().

Deliberately pulls isumap_dist[0] (data_D, the raw pre-t-conorm/pre-Dijkstra
neighbourhood distances from comp_graph()) exactly like asymm_dist_MNIST.py
does -- NOT the fully-processed D. This is intentional (per user), not the
"bug" discussed earlier: applying the t-conorm graph-merge symmetrises the
result, which is exactly what should be avoided here. distance_graph_generation.py
itself is untouched.

isumap's own D_asym construction (distance_graph_generation) has no notion
of the swiss roll's ground-truth Randers field (omega) -- it derives
asymmetry purely from the directed k-NN/star-graph/t-conorm structure of X,
not from an injected vector field. That part is unchanged and is the whole
point of comparison: how does isumap's own asymmetric graph behave under
the shared force-directed update.

[OURS 2026-08-18, per explicit user/advisor feedback -- "iki drift
mekanizması çakışıyor"] B's source was changed AGAIN, this time to remove
a genuine methodological conflict, not just for style. Every earlier
version of this file (see git history) located B via a virtual-point
mechanism that peeked at the TRUE omega field directly (X_virtual =
X + omega, embedded once via spectral_layout, B := y_virtual - y_real).
That meant two independent channels carried the same underlying drift
information into the pipeline at once: (1) D_asym itself, whose asymmetry
already encodes omega (baked in by distance_graph_generation), and (2) the
separately-located B, which peeked at omega a SECOND time via the virtual
points. This defeats the actual point of using isumap's own asymmetric
graph: isumap's whole premise is that drift can be inferred purely from an
observed asymmetric dissimilarity matrix, with no privileged access to the
ground-truth field that produced it -- exactly what a real (non-synthetic)
application would require, since in practice you only ever observe D_asym,
never omega itself.

The fix: stop locating B via virtual points entirely -- but ALSO don't swing
all the way to recomputing B every epoch. An earlier, separately-agreed
design principle (from a different discussion, well before the advisor
feedback above) was that B should be computed ONCE and FROZEN, attached to
each node for the whole apply-step training ("ai+bi, bi sabit": a_i moves
every epoch, b_i does not) -- the same reasoning that motivated replacing
the old locate_epochs-epoch trained locate step with a single deterministic
spectral_layout call in the first place. A first attempt at fixing the
omega-double-injection problem (see git history) violated this by calling
compute_drift() live, every epoch, on the current evolving Y -- reintroducing
exactly the kind of SGD-entangled, non-frozen B that principle was meant to
rule out, just from a different (omega-free) source this time. [OURS
2026-08-18, per explicit user request -- "ikisini birleştirelim"] The two
principles are reconciled in locate_B_from_D_asym() (defined below):
B is derived ENTIRELY from D_asym's own asymmetry (compute_drift(N,
knn_mask, k, Y_init, clip_delta), N = (D_asym-D_asym.T)/(D_asym+D_asym.T),
no omega anywhere), but the compute_drift() call happens exactly ONCE, on
Y_init (the same untrained spectral_layout initialisation randers_umap_fit
would build internally) -- not per epoch. The result is frozen and passed
as B_fixed, exactly like the old virtual-point mechanism used to do, just
sourced from D_asym's asymmetry instead of from omega.

This script no longer imports anything from run_swiss_roll_svd_radar.py or
svd_radar.py -- the SVD-of-Delta B variant is REMOVED entirely, and the
virtual-point locate function (locate_B_isumap, formerly defined below) is
now also removed rather than just unused.

t and alpha(t) (ground truth) are still known for swiss roll, so
test.py's direction_accuracy_swiss metric can still be run against this
output to see whether the isumap-derived asymmetry (used now for BOTH
D_asym and B) correlates with the true field -- that comparison is still
the point of this script, but it is now an honest test of whether D_asym's
asymmetry alone is enough to recover the true drift direction, rather than
a test that was quietly cheating by re-injecting omega a second time
through a separately located B.

Usage
-----
    python3 run_swiss_roll_isumap.py
    python3 run_swiss_roll_isumap.py --n 2000 --epochs 500
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

from run_swiss_roll import make_swiss_roll_randers
from distance_graph_generation import distance_graph_generation
from randers_umap import randers_umap_fit, fuzzy_simplicial_set, spectral_layout, compute_drift


def build_isumap_dist_matrix(X, k=30, verbose=True):
    """Same recipe as asymm_dist_MNIST.py: pull data_D (isumap_dist[0]),
    NOT D -- reconstruct the (i,j,k) dict into a dense (n,n) matrix.

    [OURS 2026-08-07 bug fix] data_D only populates ~k entries per row
    (the epm=True "pure star graph" default -- see distance_graph_generation.py
    docstring). asymm_dist_MNIST.py's original reconstruction used
    np.zeros(), leaving every UN-populated (i,j) pair at exactly 0.0 --
    indistinguishable from a genuine zero distance. randers_umap.py's
    _knn_from_distance_matrix() picks the k SMALLEST values per row via
    argsort, so on a mostly-zero-filled row it was picking ~k phantom
    "distance-0" non-edges as the nearest neighbours instead of the ~k
    REAL populated ones (verified empirically: for k=20, all 20 selected
    neighbours were phantom zeros, 0/20 real). Filling the unpopulated
    entries with np.inf instead fixes this -- inf can never win an
    argsort-smallest selection, and downstream smooth_knn_dist/mu
    computations already handle inf gracefully (exp(-inf/sigma) = 0, so
    an inf "neighbour" that leaks into the top-k for an under-populated
    row just gets zero weight instead of corrupting the graph).
    """
    n = X.shape[0]
    isumap_dist = distance_graph_generation(
        X, k=k, normalize=True, distBeyondNN=True, verbose=verbose,
        dataIsDistMatrix=False, dataIsGeodesicDistMatrix=False, saveDistMatrix=False,
    )
    data_D = isumap_dist[0]
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0.0)
    for key, value in data_D.items():
        i, j, k_ = key
        D[i, j] = value
    return D


def locate_B_from_D_asym(D_asym, emb_k, clip_delta=0.01, seed=0, verbose=True):
    """
    [OURS 2026-08-18, per explicit user request -- reconciles two design
    principles that were previously in tension] B originates ENTIRELY from
    D_asym's own asymmetry (no omega involved anywhere -- this is the
    advisor-driven fix from earlier the same day: isumap's whole premise is
    that drift can be inferred purely from an observed asymmetric
    dissimilarity matrix), but is computed ONCE and FROZEN, then attached
    to each node for the whole apply-step training (this is the earlier,
    separately-agreed design principle -- "ai+bi, bi sabit": a_i moves every
    epoch, b_i does not).

    Mechanism: build Y_init exactly the way randers_umap_fit would build it
    internally (spectral_layout on D_asym's own fuzzy graph, same emb_k and
    seed -- so this call reproduces that Y_init deterministically, no
    Y_init_override needed downstream), then call compute_drift() ONCE on
    this Y_init (not per-epoch/live) to get B. Contrast with the (rejected)
    live-B mechanism this replaces, which called compute_drift() every
    epoch on the CURRENT, evolving Y -- this version calls it exactly once,
    on the untrained initial layout, and freezes the result.
    """
    n = D_asym.shape[0]
    mu, knn_mask = fuzzy_simplicial_set(D_asym, emb_k)
    Y_init = spectral_layout(mu, d=2, seed=seed)

    N = (D_asym - D_asym.T) / (D_asym + D_asym.T + 1e-12)
    N = np.where(np.isfinite(N), N, 0.0)

    B_located = compute_drift(N, knn_mask, emb_k, Y_init, clip_delta=clip_delta)

    if verbose:
        bn = np.linalg.norm(B_located, axis=1)
        limit = 1.0 - clip_delta
        print(f"B located (from D_asym asymmetry only, frozen): mean||b||={bn.mean():.4f}  "
              f"max||b||={bn.max():.4f}  clipped={(bn >= limit - 1e-9).sum()}/{n}")

    return B_located


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
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
    p.add_argument("--ramp", action="store_true",
                    help="[OURS 2026-08-12] ramp B's magnitude 0->1 over the first 70%% of "
                         "epochs instead of applying it at full strength from epoch 0 "
                         "(default: off, matching run_swiss_roll.py -- B is located/computed "
                         "once and attached at full strength from the start).")
    p.add_argument("--init-only", action="store_true",
                    help="[OURS 2026-08-13] stop before force-directed training -- isumap has no "
                         "explicit separate Y_init step (randers_umap_fit computes its own spectral "
                         "init on D_asym internally), so this runs a single epoch with an internal "
                         "epoch-0 snapshot and returns that pre-training state instead of out['Y']/"
                         "out['B']. Ignores --epochs, --ramp, --gravity.")
    p.add_argument("--snapshot-every", type=int, default=None,
                    help="[OURS 2026-08-14] if given, also save <out>_snapshots.png: the apply-step "
                         "embedding (with drift-vector arrows) every N epochs, side by side. "
                         "Ignored if --init-only is also given.")
    p.add_argument("--proj-dim", type=int, default=2, choices=[2, 3],
                    help="[OURS 2026-08-20, per explicit user request] embedding "
                         "dimension for randers_umap_fit's own internal spectral "
                         "init AND the apply-step training -- see run_swiss_roll.py's "
                         "--proj-dim help for the full explanation. 3 = full 3D "
                         "layout, main scatter plot switches to 3D axes automatically.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="swiss_embedding_isumap")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    # [OURS 2026-08-13] init-only: run just 1 epoch with a snapshot_every=1 so
    # randers_umap_fit's guaranteed pre-loop epoch-0 capture gives us the raw
    # spectral Y_init + B (at ramp's epoch-0 value) without any real training.
    apply_epochs = 1 if args.init_only else args.epochs
    apply_snapshot_every = 1 if args.init_only else args.snapshot_every

    if not args.quiet:
        print(f"Generating swiss roll: n={args.n}")
    X, omega, t = make_swiss_roll_randers(args.n, seed=42)

    if not args.quiet:
        print(f"\nBuilding distance matrix via distance_graph_generation (data_D, unfixed)...")
    D_asym = build_isumap_dist_matrix(X, k=args.k, verbose=not args.quiet)

    # [OURS 2026-08-07] data_D only populates ~k-1 real entries per row (the
    # rest are the np.inf fill from the bug fix above). If emb_k requests
    # more neighbours than a row actually has, _knn_from_distance_matrix's
    # argsort is forced to include a phantom inf-distance "neighbour" to
    # fill the quota -- and N=0.5*(D_asym-D_asym.T) can be inf/nan exactly
    # there. Capping emb_k to the worst-case row's real neighbour count
    # guarantees every selected neighbour is real.
    min_real_neighbors = int(np.isfinite(D_asym).sum(axis=1).min() - 1)  # -1 excludes the diagonal
    emb_k = min(args.k, max(min_real_neighbors, 1))
    if not args.quiet:
        print(f"D_asym: {D_asym.shape}  symmetric={np.allclose(D_asym, D_asym.T)}  "
              f"min real neighbours/row={min_real_neighbors}  emb_k used={emb_k}")

    # [OURS 2026-08-19, per explicit user request -- reverted back to the
    # live mechanism] B is derived live, every epoch, purely from D_asym's
    # own asymmetry (compute_drift on N=(D_asym-D_asym.T)/(D_asym+D_asym.T),
    # no omega anywhere) and the CURRENT embedding Y -- B_fixed=None.
    # locate_B_from_D_asym() (still defined above) computes the same thing
    # but ONCE and frozen; that hybrid was tried and then explicitly
    # reverted in favour of this live version. The epoch-0 snapshot
    # (--init-only reads this) is NOT an empty placeholder despite B_fixed
    # being None: see the 2026-08-19 fix in randers_umap.py's snapshot
    # capture, which computes the real epoch-0 compute_drift(...) value
    # there specifically so --init-only still shows a meaningful drift.
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
                            snapshot_every=apply_snapshot_every, verbose=not args.quiet)

    if args.init_only:
        # true pre-training state, captured before any epoch update
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

    drift_label = "live B (from D_asym asymmetry only)"
    init_suffix = ", INIT ONLY (no training)" if args.init_only else ""
    ax.set_title(f"Randers-UMAP, isumap-derived D, {drift_label}{init_suffix}  (n={args.n})", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)

    np.savez(f"{args.out}.npz", Y=Y, B=B, t=t, X=X, omega=omega)

    if not args.quiet:
        print(f"\nwrote {args.out}.png and {args.out}.npz")

    # ---- snapshot grid: init -> every N epochs -> final, side by side,
    # each panel with its own drift-vector arrows [OURS 2026-08-14] --------
    if args.snapshot_every is not None and not args.init_only:
        snaps = out["snapshots"]
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

        fig2.suptitle(f"Randers-UMAP, isumap-derived D, {drift_label}, trajectory  "
                      f"(n={args.n}, snapshot_every={args.snapshot_every})", fontsize=11)
        if sc2 is not None:
            fig2.colorbar(sc2, ax=fig2.get_axes(), label="t (intrinsic coordinate)",
                          fraction=0.02, pad=0.01)
        fig2.savefig(f"{args.out}_snapshots.png", dpi=150, bbox_inches="tight")

        if not args.quiet:
            print(f"wrote {args.out}_snapshots.png ({n_snap} snapshots)")


if __name__ == "__main__":
    main()
