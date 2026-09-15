#!/usr/bin/env python3
"""
asymmetry_k_sweep_isumap.py -- [OURS 2026-09-07] isumap-flavored counterpart
of asymmetry_k_sweep.py. Computes EXACTLY the same two quantities (global
asymmetry_score vs. k, and per-node % preservation at a fixed k), but built
on the isumap pipeline (run_swiss_roll_calculated.py's build_isumap_dist_matrix
+ isumap_style_init + live-drift fdgl_low_dim) instead of
randers_bridge.fdgl_pipeline.

This is a SEPARATE file, not a --isumap flag bolted onto the original,
because the two pipelines diverge in ways that don't collapse into shared
plumbing without real duplication anyway: no --adjacency choice (isumap's
own distance_graph_generation has no knn/threshold switch), D_asym can be
sparse/inf-filled with a k that gets clipped down to emb_k per-dataset
(build_isumap_dist_matrix's own row-sparsity guard), and B is always the
LIVE mechanism (B_fixed=None, use_drift=True) -- matching what
run_swiss_roll_calculated.py's own main() actually runs (see that file's
docstring: the frozen locate_B_from_D_asym() alternative is defined there
but never called).

--mode sweep (default): sweeps k (the distance_graph_generation k passed to
    build_isumap_dist_matrix; the k actually used for the embedding, emb_k,
    is printed alongside since it can be clipped down) and plots the GLOBAL
    asymmetry_score (initial/target vs. final/trained) as k varies.

--mode distribution: for a SINGLE k, runs the pipeline once and plots a
    histogram of per-node "% of target asymmetry preserved" -- same
    definition as asymmetry_k_sweep.py's own --mode distribution.

Usage
-----
    python asymmetry_k_sweep_isumap.py
    python asymmetry_k_sweep_isumap.py --n 1000 --k-min 5 --k-max 60 --k-step 5
    python asymmetry_k_sweep_isumap.py --dataset mammoth --epochs 300
    python asymmetry_k_sweep_isumap.py --mode distribution --k 20 --epochs 300
    python asymmetry_k_sweep_isumap.py --mode distribution --dataset sphere --k 30
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
# [OURS 2026-09-15] this file now lives in test/, not flat in the FDGL
# root where randers_bridge.py/run_swiss_roll_generated.py/etc. actually live --
# added ROOT explicitly.
sys.path.insert(0, str(ROOT))

from randers_bridge import asymmetry_score, reconstruct_rho
from randers_fdgl import fdgl_low_dim
from isumap_bridge import build_isumap_dist_matrix, isumap_style_init
from run_swiss_roll_generated import make_swiss_roll_randers
from run_mammoth_generated import make_mammoth_randers
from run_sphere_tangential_generated import make_sphere_points

# [OURS 2026-09-07] unlike asymmetry_k_sweep.py's DATASET_GENERATORS, these
# don't need to share a return signature -- only the first element (X) is
# ever used, since isumap's own D_asym construction (distance_graph_
# generation) has no field/vector-field parameter at all; whatever else
# each generator returns (omega, theta/phi, ...) is irrelevant here.
DATASET_GENERATORS = {
    "swiss_roll": make_swiss_roll_randers,
    "mammoth": make_mammoth_randers,
    "sphere": make_sphere_points,
}


def _alignment(per_node_initial, per_node_final):
    """[OURS 2026-09-07] identical to asymmetry_k_sweep.py's own _alignment --
    Pearson correlation between the two per-node asymmetry vectors, nan-pairs
    dropped first."""
    valid = np.isfinite(per_node_initial) & np.isfinite(per_node_final)
    if valid.sum() < 2:
        return float("nan")
    a, b = per_node_initial[valid], per_node_final[valid]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run_isumap_asymmetry(X, k, epochs, neg, seed, proj_dim=2,
                          force_model="fr_gravity", fr_k=None,
                          negative_sampling=False, ramp=False, verbose=False):
    """
    [OURS 2026-09-07] The isumap-pipeline counterpart of
    randers_bridge.fdgl_pipeline's asymmetry-relevant computation --
    builds D_asym the isumap way, trains via the LIVE drift mechanism (the
    same one run_swiss_roll_calculated.py's main() actually runs), and computes
    the same four asymmetry quantities via the same asymmetry_score()/
    reconstruct_rho() functions the main pipeline uses, so the numbers are
    directly comparable across the two pipelines.

    ramp defaults to False here (not fdgl_low_dim's own internal default
    of True), matching every other run_*.py/embed_*.py script's own
    --ramp convention (off by default, explicit opt-in) and randers_bridge.
    fdgl_pipeline's own default -- this file previously omitted `ramp`
    entirely from the fdgl_low_dim call below, which silently pulled in
    fdgl_low_dim's internal ramp=True (zeroing out B/asymmetry for the
    first 30% of epochs, full strength only after 70%) with no way to turn
    it off from the CLI. [OURS 2026-09-08, fixed]

    Returns
    -------
    dict: {"asymmetry_score", "asymmetry_per_node", "asymmetry_score_final",
           "asymmetry_per_node_final", "emb_k"}
    """
    D_asym = build_isumap_dist_matrix(X, k=k, verbose=False)

    # [OURS, ported from run_swiss_roll_calculated.py's main()] some rows can
    # have fewer real (finite) neighbours than the requested k -- clip.
    min_real_neighbors = int(np.isfinite(D_asym).sum(axis=1).min() - 1)
    emb_k = min(k, max(min_real_neighbors, 1))

    # [OURS 2026-09-07, fixed 2026-09-07] D_geo -- see
    # run_swiss_roll_calculated.py's own docstring for the full rationale AND
    # the bug-fix note: a plain (D_asym+D_asym.T)/2 average requires BOTH
    # directions finite, which made D_geo STRICTLY SPARSER than D_asym for
    # isumap's asymmetric-existence graphs (some rows ending up with zero
    # real neighbours, corrupting A_geo's calibration with inf/NaN). Fix:
    # average where both directions exist, fall back to whichever single
    # direction is finite otherwise.
    both_finite = np.isfinite(D_asym) & np.isfinite(D_asym.T)
    D_geo = np.where(both_finite, (D_asym + D_asym.T) / 2.0,
                      np.where(np.isfinite(D_asym), D_asym, D_asym.T))

    # [OURS 2026-09-07, merged into asymmetry_score() 2026-09-15] call
    # asymmetry_score(D_asym) with bln OMITTED (not asymmetry_score(D_asym,
    # bln_asym) with it given) -- D_asym is sparse (~k entries/row), and a
    # real edge (i,j) with D_asym[i,j] finite can easily have D_asym[j,i]=inf
    # (j is one of i's k-NN without i being one of j's), which would blow
    # |D[i,j]-D[j,i]| up to inf if scored directly. Leaving bln=None makes
    # asymmetry_score take its raw/sparse path: derive bln from D_asym's own
    # isfinite mask, complete D_asym via a directed shortest_path internally
    # (purely for scoring, without touching what's actually fed to
    # training), then score. bln reconstructed the same way below, so the
    # "final" score (on the always-dense rho_final) uses the identical
    # real-edge set.
    asym_per_node, asym_global = asymmetry_score(D_asym)
    n = D_asym.shape[0]
    bln_asym = np.isfinite(D_asym) & ~np.eye(n, dtype=bool)

    Y_init = isumap_style_init(D_asym, d=proj_dim, seed=seed)

    out = fdgl_low_dim(D_asym, n_neighbors=emb_k, n_negative_samples=neg,
                            n_epochs=epochs, use_drift=True, B_fixed=None,
                            d=proj_dim, Y_init_override=Y_init,
                            seed=seed, verbose=verbose,
                            force_model=force_model, fr_k=fr_k,
                            negative_sampling=negative_sampling, ramp=ramp,
                            D_geo=D_geo)
    Y, B = out["Y"], out["B"]

    rho_final = reconstruct_rho(Y, B)
    asym_per_node_final, asym_global_final = asymmetry_score(rho_final, bln_asym)

    return {"asymmetry_score": asym_global, "asymmetry_per_node": asym_per_node,
            "asymmetry_score_final": asym_global_final,
            "asymmetry_per_node_final": asym_per_node_final, "emb_k": emb_k}


def sweep_asymmetry_vs_k(X, k_values, epochs, neg, seed, verbose=True,
                          force_model="fr_gravity", fr_k=None, negative_sampling=False,
                          ramp=False):
    """isumap counterpart of asymmetry_k_sweep.py's sweep_asymmetry_vs_k."""
    out = {"initial": [], "final": [], "alignment": [], "emb_k": []}
    for k in k_values:
        result = run_isumap_asymmetry(X, k, epochs, neg, seed,
                                       force_model=force_model, fr_k=fr_k,
                                       negative_sampling=negative_sampling, ramp=ramp,
                                       verbose=False)
        initial = result["asymmetry_score"]
        final = result["asymmetry_score_final"]
        align = _alignment(result["asymmetry_per_node"], result["asymmetry_per_node_final"])
        out["initial"].append(initial)
        out["final"].append(final)
        out["alignment"].append(align)
        out["emb_k"].append(result["emb_k"])
        if verbose:
            pct = 100.0 * final / max(initial, 1e-12)
            print(f"  k={k:3d} (emb_k={result['emb_k']:3d})  initial={initial:.4f}  "
                  f"final={final:.4f}  ({pct:5.1f}% preserved)  alignment={align:.4f}")
    return out


def per_node_preservation(X, k, epochs, neg, seed, min_initial=1e-3,
                           force_model="fr_gravity", fr_k=None, negative_sampling=False,
                           ramp=False):
    """isumap counterpart of asymmetry_k_sweep.py's per_node_preservation."""
    result = run_isumap_asymmetry(X, k, epochs, neg, seed,
                                   force_model=force_model, fr_k=fr_k,
                                   negative_sampling=negative_sampling, ramp=ramp,
                                   verbose=False)
    initial = result["asymmetry_per_node"]
    final = result["asymmetry_per_node_final"]
    n_total = initial.shape[0]

    finite = np.isfinite(initial) & np.isfinite(final)
    n_excluded_nan = int((~finite).sum())

    near_zero = finite & (initial < min_initial)
    n_excluded_near_zero = int(near_zero.sum())

    valid = finite & ~near_zero
    pct = 100.0 * final[valid] / initial[valid]

    return {"pct": pct, "initial": initial[valid], "final": final[valid],
            "n_total": n_total, "n_valid": int(valid.sum()),
            "n_excluded_nan": n_excluded_nan,
            "n_excluded_near_zero": n_excluded_near_zero,
            "emb_k": result["emb_k"]}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["sweep", "distribution"], default="sweep")
    p.add_argument("--dataset", choices=list(DATASET_GENERATORS.keys()), default="swiss_roll")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--k-min", type=int, default=5, help="[--mode sweep only]")
    p.add_argument("--k-max", type=int, default=60, help="[--mode sweep only]")
    p.add_argument("--k-step", type=int, default=5, help="[--mode sweep only]")
    p.add_argument("--k", type=int, default=20,
                    help="[--mode distribution only] the single k to run at.")
    p.add_argument("--bins", type=int, default=30,
                    help="[--mode distribution only] number of histogram bins.")
    p.add_argument("--min-initial", type=float, default=1e-3,
                    help="[--mode distribution only] nodes with initial per-node "
                         "asymmetry below this are excluded from the %% preserved histogram.")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--neg", type=int, default=10)
    p.add_argument("--force-model", choices=["fr_gravity", "umap"], default="fr_gravity")
    p.add_argument("--fr-k", type=float, default=None)
    p.add_argument("--neg-sampling", action="store_true")
    p.add_argument("--ramp", action="store_true",
                    help="[OURS 2026-09-08] ramp B's magnitude 0->1 over the first 70%% of "
                         "epochs (0 for the first 30%%, linear 30-70%%, full strength after) "
                         "instead of applying it at full strength from epoch 0. Off by "
                         "default -- matches every other run_*.py/embed_*.py script's own "
                         "--ramp convention, deliberately overriding fdgl_low_dim's own "
                         "internal default of ramp=True.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=None,
                    help="output PNG filename -- defaults to "
                         "asymmetry_k_sweep_isumap_<dataset>.png (--mode sweep) or "
                         "asymmetry_distribution_isumap_<dataset>_k<k>.png (--mode distribution).")
    args = p.parse_args()

    print(f"Generating {args.dataset} (isumap D_asym, no field used), n={args.n}, seed={args.seed}...")
    generator = DATASET_GENERATORS[args.dataset]
    X = generator(args.n, seed=args.seed)[0]

    if args.mode == "distribution":
        if args.out is None:
            args.out = f"asymmetry_distribution_isumap_{args.dataset}_k{args.k}.png"

        print(f"Running once at k={args.k}, epochs={args.epochs} "
              f"(single training run, per-node preservation)...")
        d = per_node_preservation(X, args.k, args.epochs, args.neg, args.seed,
                                   min_initial=args.min_initial,
                                   force_model=args.force_model, fr_k=args.fr_k,
                                   negative_sampling=args.neg_sampling, ramp=args.ramp)

        edges = [0, 25, 50, 75, 100, 125, 150, float("inf")]
        labels = ["0-25%", "25-50%", "50-75%", "75-100%", "100-125%", "125-150%", "150%+"]
        print(f"\n--- per-node % preserved distribution (isumap {args.dataset}, k={args.k} "
              f"(emb_k={d['emb_k']}), n={args.n}, epochs={args.epochs}) ---")
        print(f"total nodes = {d['n_total']}  |  valid = {d['n_valid']}  |  "
              f"excluded (isolated/nan) = {d['n_excluded_nan']}  |  "
              f"excluded (initial < {args.min_initial}) = {d['n_excluded_near_zero']}")
        for lo, hi, label in zip(edges[:-1], edges[1:], labels):
            count = int(((d["pct"] >= lo) & (d["pct"] < hi)).sum())
            frac = 100.0 * count / max(d["n_valid"], 1)
            print(f"  {label:>10s} : {count:5d} nodes  ({frac:5.1f}% of valid)")
        print(f"\nmean={d['pct'].mean():.1f}%  median={np.median(d['pct']):.1f}%  "
              f"std={d['pct'].std():.1f}%  min={d['pct'].min():.1f}%  max={d['pct'].max():.1f}%")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(d["pct"], bins=args.bins, color="tab:purple", edgecolor="white", alpha=0.85)
        ax.axvline(np.median(d["pct"]), color="k", linestyle="--", linewidth=1.2,
                   label=f"median = {np.median(d['pct']):.1f}%")
        ax.axvline(100.0, color="tab:red", linestyle=":", linewidth=1.2,
                   label="100% (fully preserved)")
        ax.set_xlabel("% of node's target (initial) asymmetry preserved in final embedding")
        ax.set_ylabel("number of nodes")
        ax.set_title(f"Per-node asymmetry preservation distribution (isumap D_asym)\n"
                     f"({args.dataset}, n={args.n}, k={args.k}, emb_k={d['emb_k']}, "
                     f"epochs={args.epochs})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_png = HERE / args.out
        fig.savefig(out_png, dpi=150)
        print(f"\nSaved distribution plot to {out_png}")

        out_npz = out_png.with_suffix(".npz")
        np.savez(out_npz, pct=d["pct"], initial=d["initial"], final=d["final"])
        print(f"Saved raw per-node data to {out_npz}")
        return

    # ---- mode == "sweep" ----
    if args.out is None:
        args.out = f"asymmetry_k_sweep_isumap_{args.dataset}.png"

    k_values = list(range(args.k_min, args.k_max + 1, args.k_step))

    print(f"Sweeping k over {k_values}, epochs={args.epochs} "
          f"(training a real embedding at every k)...")
    results = sweep_asymmetry_vs_k(X, k_values, epochs=args.epochs, neg=args.neg,
                                    seed=args.seed, force_model=args.force_model,
                                    fr_k=args.fr_k, negative_sampling=args.neg_sampling,
                                    ramp=args.ramp)

    print(f"\n--- asymmetry_k_sweep_isumap results ({args.dataset}, n={args.n}, "
          f"epochs={args.epochs}) ---")
    print(f"{'k':>4}  {'emb_k':>5}  {'initial':>8}  {'final':>8}  {'% preserved':>12}  {'alignment':>9}")
    for k, emb_k, initial, final, align in zip(k_values, results["emb_k"], results["initial"],
                                                 results["final"], results["alignment"]):
        pct = 100.0 * final / max(initial, 1e-12)
        print(f"{k:4d}  {emb_k:5d}  {initial:8.4f}  {final:8.4f}  {pct:11.1f}%  {align:9.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_values, results["initial"], marker="o", color="tab:blue", label="initial (target, D_asym)")
    ax.plot(k_values, results["final"], marker="s", color="tab:red", label="final (trained embedding)")
    ax.set_xlabel("k (distance_graph_generation k; emb_k may be clipped lower)")
    ax.set_ylabel("asymmetry_score (global)")
    ax.set_title(f"Asymmetry score vs. k, isumap D_asym  ({args.dataset}, n={args.n}, "
                 f"epochs={args.epochs})")
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_png = HERE / args.out
    fig.savefig(out_png, dpi=150)
    print(f"\nSaved main plot to {out_png}")

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(k_values, results["alignment"], marker="^", color="tab:green")
    ax2.axhline(0.0, color="k", linestyle=":", linewidth=1)
    ax2.set_xlabel("k (distance_graph_generation k)")
    ax2.set_ylabel("alignment (Pearson corr., per-node initial vs. final)")
    ax2.set_title(f"Per-node asymmetry alignment vs. k, isumap D_asym  ({args.dataset}, "
                 f"n={args.n}, epochs={args.epochs})")
    ax2.set_ylim(-1, 1)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

    out_align_png = out_png.with_name(out_png.stem + "_alignment" + out_png.suffix)
    fig2.savefig(out_align_png, dpi=150)
    print(f"Saved alignment plot to {out_align_png}")

    out_npz = out_png.with_suffix(".npz")
    np.savez(out_npz, k_values=np.array(k_values), emb_k=np.array(results["emb_k"]),
             initial=np.array(results["initial"]), final=np.array(results["final"]),
             alignment=np.array(results["alignment"]))
    print(f"Saved raw sweep data to {out_npz}")


if __name__ == "__main__":
    main()
