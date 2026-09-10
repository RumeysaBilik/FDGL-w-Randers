#!/usr/bin/env python3
"""
run_swiss_roll_isumap.py -- swiss roll, but the distance matrix comes from
distance_graph_generation() (the isumap method used in asymm_dist_MNIST.py),
NOT from randers_bridge.compute_dist_matrix().

Deliberately pulls isumap_dist[0] (data_D, the raw pre-t-conorm/pre-Dijkstra
neighbourhood distances from comp_graph()) exactly like asymm_dist_MNIST.py
does -- NOT the fully-processed D. This is intentional, not the
"bug" discussed earlier: applying the t-conorm graph-merge symmetrises the
result, which is exactly what should be avoided here. distance_graph_generation.py
itself is untouched.

isumap's own D_asym construction (distance_graph_generation) has no notion
of the swiss roll's ground-truth Randers field (omega) -- it derives
asymmetry purely from the directed k-NN/star-graph/t-conorm structure of X,
not from an injected vector field. That part is unchanged and is the whole
point of comparison: how does isumap's own asymmetric graph behave under
the shared force-directed update.

[OURS 2026-08-18] B's source was changed AGAIN, this time to remove
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
each node for the whole apply-step training ("ai+bi, bi fixed": a_i moves
every epoch, b_i does not) -- the same reasoning that motivated replacing
the old locate_epochs-epoch trained locate step with a single deterministic
spectral_layout call in the first place. A first attempt at fixing the
omega-double-injection problem (see git history) violated this by calling
compute_drift() live, every epoch, on the current evolving Y -- reintroducing
exactly the kind of SGD-entangled, non-frozen B that principle was meant to
rule out, just from a different (omega-free) source this time. [OURS
2026-08-18] The two
principles are reconciled in locate_B_from_D_asym() (defined below):
B is derived ENTIRELY from D_asym's own asymmetry (compute_drift(N,
knn_mask, k, Y_init, clip_delta), N = (D_asym-D_asym.T)/(D_asym+D_asym.T),
no omega anywhere), but the compute_drift() call happens exactly ONCE, on
Y_init (the same untrained spectral_layout initialisation randers_umap_fit
would build internally) -- not per epoch. The result is frozen and passed
as B_fixed, exactly like the old virtual-point mechanism used to do, just
sourced from D_asym's asymmetry instead of from omega.

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
from randers_umap import (randers_umap_fit, arrow_scale, _compute_N,
                           compute_drift, knn_mask_from_distance_matrix)
from isumap_bridge import build_isumap_dist_matrix, isumap_style_init

# [OURS 2026-09-08] locate_B_from_D_asym() (the frozen-B alternative to the
# live mechanism main() actually runs) removed -- confirmed dead code, never
# called anywhere in this file, run_mammoth_isumap.py, run_sphere_isumap.py,
# or asymmetry_k_sweep_isumap.py (only imported by run_mammoth_isumap.py,
# itself never invoked there either). The live/frozen comparison this
# function was for still exists and is actively used, just as its own
# separate copy in MNIST/compare_live_vs_frozen_direction.py -- that one is
# unaffected by this removal.


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
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
    p.add_argument("--fixed-drift", action="store_true",
                    help="[OURS 2026-09-10] instead of the default live mechanism (B "
                         "recomputed from D_asym's own asymmetry every epoch, tracking the "
                         "CURRENT/evolving Y), derive B ONCE from D_asym's asymmetry at "
                         "Y_init -- the same untrained isumap_style_init position -- and "
                         "FREEZE it there for the whole run, exactly like the 'generated' "
                         "family's B_located mechanism (randers_bridge.py). This revives the "
                         "old locate_B_from_D_asym() design (removed as dead code on "
                         "2026-09-08, still alive as a standalone copy in "
                         "MNIST/compare_live_vs_frozen_direction.py) as an opt-in flag here, "
                         "using the CURRENT _compute_N (with the existence-asymmetry "
                         "diameter-substitution fix) rather than that old copy's stale inline "
                         "N formula. Off by default -- live drift, exact prior behaviour.")
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
                    help="[OURS 2026-08-20] embedding "
                         "dimension for randers_umap_fit's own internal spectral "
                         "init AND the apply-step training -- see run_swiss_roll.py's "
                         "--proj-dim help for the full explanation. 3 = full 3D "
                         "layout, main scatter plot switches to 3D axes automatically.")
    p.add_argument("--force-model", choices=["fr_gravity", "umap"], default="fr_gravity",
                    help="[OURS 2026-08-28] see run_swiss_roll.py's "
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

    # [OURS 2026-09-07, fixed 2026-09-07] D_geo -- the Euclidean-consistent
    # weight source for randers_umap_fit's A_geo (see its own weight-
    # consistency-fix docstring). In the isomap-style pipeline
    # (run_swiss_roll.py) D_geo is "the same D_asym construction with
    # randers_field=None" -- but distance_graph_generation has no field
    # parameter at all: this pipeline's asymmetry isn't injected by a
    # field, it's the raw k-NN structural asymmetry that real IsUMap's own
    # t-conorm+"(D+D.T)/2" step would normally erase. So the natural analog
    # here is to do exactly that erasure ourselves: symmetrize D_asym
    # directly.
    #
    # [BUG FIX] A plain average, (D_asym+D_asym.T)/2, requires BOTH
    # directions to be finite (inf+anything=inf) -- and isumap's raw D_asym
    # very often has i list j as a neighbour without j listing i back (no
    # symmetry guarantee at all, epm=True "pure star graph"). At small k
    # this made D_geo STRICTLY SPARSER than D_asym itself (empirically: at
    # n=2000, k=5, 1115/2000 rows ended up with FEWER than emb_k real
    # entries in D_geo, 14 rows with ZERO -- corrupting A_geo's rho/sigma
    # calibration with inf, cascading into the trained embedding as NaN).
    # Fix: average where both directions exist, otherwise fall back to
    # whichever single direction is finite -- this guarantees D_geo is
    # never sparser than D_asym in any row, restoring the "emb_k, already
    # sized to D_asym's worst-case row, stays safely valid for D_geo too"
    # property this was originally meant to have.
    both_finite = np.isfinite(D_asym) & np.isfinite(D_asym.T)
    D_geo = np.where(both_finite, (D_asym + D_asym.T) / 2.0,
                      np.where(np.isfinite(D_asym), D_asym, D_asym.T))

    # [OURS 2026-08-19] B is derived live, every epoch, purely from D_asym's
    # own asymmetry (compute_drift on N=(D_asym-D_asym.T)/(D_asym+D_asym.T),
    # no omega anywhere) and the CURRENT embedding Y -- B_fixed=None.
    # locate_B_from_D_asym() (still defined above) computes the same thing
    # but ONCE and frozen; that hybrid was tried and then explicitly
    # reverted in favour of this live version. The epoch-0 snapshot
    # (--init-only reads this) is NOT an empty placeholder despite B_fixed
    # being None: see the 2026-08-19 fix in randers_umap.py's snapshot
    # capture, which computes the real epoch-0 compute_drift(...) value
    # there specifically so --init-only still shows a meaningful drift.
    # [OURS 2026-09-03] init: real IsUMap's own cMDS choice
    # (isumap_style_init, see its docstring above), NOT randers_umap_fit's
    # internal UMAP-style spectral_layout default -- so the ONLY thing this
    # script still borrows from UMAP is the attractive/repulsive force
    # computation itself, per the advisor-facing goal of this "_isumap"
    # comparison family (build_isumap_dist_matrix's own D_asym is untouched
    # by this -- only the starting position changes).
    if not args.quiet:
        print(f"\nInitialising Y via IsUMap's own classical MDS (not UMAP spectral_layout)...")
    Y_init = isumap_style_init(D_asym, d=args.proj_dim, seed=args.seed)

    # [OURS 2026-09-10] --fixed-drift: derive B ONCE from D_asym's own
    # asymmetry at Y_init, then freeze it for the whole run -- see the flag's
    # own help text above. B_fixed=None (default) keeps the live mechanism
    # (randers_umap_fit recomputes B every epoch from the CURRENT Y).
    if args.fixed_drift:
        if not args.quiet:
            print(f"\nDeriving B ONCE from D_asym's own asymmetry at Y_init, then freezing it "
                  f"for the whole run (--fixed-drift)...")
        knn_mask_fixed = knn_mask_from_distance_matrix(D_asym, emb_k)
        N_fixed = _compute_N(D_asym)
        B_fixed = compute_drift(N_fixed, knn_mask_fixed, emb_k, Y_init, clip_delta=args.clip_delta)
    else:
        if not args.quiet:
            print(f"\nDeriving B live from D_asym's own asymmetry (no omega used) each epoch...")
        B_fixed = None

    out = randers_umap_fit(D_asym, n_neighbors=emb_k, n_negative_samples=args.neg,
                            n_epochs=apply_epochs, use_drift=True, B_fixed=B_fixed,
                            d=args.proj_dim, Y_init_override=Y_init,
                            clip_delta=args.clip_delta,
                            use_gravity=args.gravity, gravity_strength=args.gravity_strength,
                            gravity_neighbor_weight=not args.no_gravity_neighbor_weight,
                            use_virtual_neighbor=not args.no_virtual_neighbor,
                            ramp=args.ramp, seed=args.seed,
                            snapshot_every=apply_snapshot_every, verbose=not args.quiet,
                            force_model=args.force_model, fr_k=args.fr_k,
                            negative_sampling=args.neg_sampling, D_geo=D_geo)

    if args.init_only:
        # true pre-training state, captured before any epoch update
        Y, B = out["snapshots"][0]["Y"], out["snapshots"][0]["B"]
    else:
        Y, B = out["Y"], out["B"]

    # ---- plot --------------------------------------------------------
    # [OURS 2026-08-20] proj_dim==3 -> 3D scatter
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
            sc_scale = arrow_scale(Y, bn)
            ax.quiver(Y[big, 0], Y[big, 1], Y[big, 2],
                      B[big, 0] * sc_scale, B[big, 1] * sc_scale, B[big, 2] * sc_scale,
                      color="k", alpha=0.6, linewidth=1.0, arrow_length_ratio=0.3)
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2"); ax.set_zlabel("dim 3")
    else:
        fig, ax = plt.subplots(figsize=(9, 8))
        sc = ax.scatter(Y[:, 0], Y[:, 1], c=t, cmap="viridis", s=10, alpha=0.85, linewidths=0)
        plt.colorbar(sc, ax=ax, label="t (intrinsic coordinate)")
        if bn.max() > 0:
            sc_scale = arrow_scale(Y, bn)
            ax.quiver(Y[big, 0], Y[big, 1], B[big, 0] * sc_scale, B[big, 1] * sc_scale,
                      color="k", alpha=0.6, width=0.004, scale=1, scale_units="xy")
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")

    drift_label = ("frozen B (from D_asym asymmetry at Y_init only)" if args.fixed_drift
                   else "live B (from D_asym asymmetry only)")
    init_suffix = ", INIT ONLY (no training)" if args.init_only else f", epochs={args.epochs}"
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
                    sc_scale_i = arrow_scale(Yi, bni)
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
                    sc_scale_i = arrow_scale(Yi, bni)
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
