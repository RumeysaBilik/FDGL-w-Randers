#!/usr/bin/env python3
"""
asymmetry_k_sweep.py -- two modes, both built on run_located_drift's
"initial" (target, D_asym) vs. "final" (trained embedding) asymmetry_score
(randers_bridge.asymmetry_score/asymmetry_score_final), on any of this
project's 4 Randers-field datasets (swiss_roll, mammoth, sphere_tangential,
sphere_radial) -- see --dataset below.

--mode sweep (default, unchanged from the 2026-09-01 rewrite): sweeps k
    (n_neighbors) for ONE chosen adjacency construction (--adjacency
    knn|threshold) and plots how the GLOBAL asymmetry_score (mean over all
    nodes) behaves as k varies -- one point per k.

--mode distribution : [OURS 2026-09-01] For a SINGLE chosen k (--k, not a
    k-min/k-max/k-step range), runs run_located_drift ONCE and looks at
    asymmetry_per_node / asymmetry_per_node_final (n,) directly, instead of
    their means (asymmetry_score / asymmetry_score_final). For each node i,
    computes
        pct_i = 100 * asymmetry_per_node_final[i] / asymmetry_per_node[i]
    -- i.e. the SAME "% of target asymmetry preserved" idea the sweep mode
    already reports GLOBALLY, just computed per node instead of once over
    the mean. Plots a HISTOGRAM of pct_i across all n nodes (how many nodes
    are ~50% preserved, how many ~100%, etc.), plus a printed summary table
    binned into ranges. Nodes with near-zero initial asymmetry (no
    meaningful "% preserved" -- division by ~0) are excluded and reported
    separately, not silently dropped.

Usage
-----
    python asymmetry_k_sweep.py
    python asymmetry_k_sweep.py --n 1000 --k-min 5 --k-max 60 --k-step 5
    python asymmetry_k_sweep.py --adjacency threshold --epochs 300
    python asymmetry_k_sweep.py --dataset mammoth --adjacency knn
    python asymmetry_k_sweep.py --mode distribution --k 20 --epochs 300
    python asymmetry_k_sweep.py --mode distribution --dataset mammoth --k 30
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

from randers_bridge import run_located_drift
from run_swiss_roll import make_swiss_roll_randers
from run_mammoth import make_mammoth_randers
from run_sphere_tangential import make_sphere_tangential_randers
from run_sphere_radial import make_sphere_radial_randers

# [OURS 2026-08-25] each generator shares the same (X, omega, label) return
# signature (label is only used for plot colouring elsewhere -- unused
# here), so a single dispatch dict is enough, no per-dataset branching
# needed beyond this.
DATASET_GENERATORS = {
    "swiss_roll": make_swiss_roll_randers,
    "mammoth": make_mammoth_randers,
    "sphere_tangential": make_sphere_tangential_randers,
    "sphere_radial": make_sphere_radial_randers,
}


def _alignment(per_node_initial, per_node_final):
    """
    [OURS 2026-09-01] Pearson correlation between the two per-node
    asymmetry vectors, nan-pairs (isolated nodes) dropped first. Returns
    nan if fewer than 2 valid pairs remain (can't correlate). Used by
    --mode sweep's second plot (ONE scalar per k, needs the whole node
    population to correlate against) -- NOT the same question as
    --mode distribution's per-node pct_i (see that mode's own docstring
    above for why a per-node "correlation" isn't a meaningful quantity).
    """
    valid = np.isfinite(per_node_initial) & np.isfinite(per_node_final)
    if valid.sum() < 2:
        return float("nan")
    a, b = per_node_initial[valid], per_node_final[valid]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")  # constant vector -- correlation undefined
    return float(np.corrcoef(a, b)[0, 1])


def sweep_asymmetry_vs_k(X, omega, k_values, adjacency, epochs, neg, seed, verbose=True,
                          force_model="fr_gravity", fr_k=None, negative_sampling=False):
    """
    For each k in k_values, runs the FULL located-drift pipeline
    (run_located_drift, apply_step=True -- real force-directed training,
    not just a D_asym rebuild) with the given adjacency mode, and reads off
    four numbers: the target ("initial") and trained ("final") global
    asymmetry_score, plus the per-node alignment between them.

    Returns
    -------
    dict of lists, all same length as k_values:
        {"initial": [...], "final": [...], "alignment": [...]}
    """
    out = {"initial": [], "final": [], "alignment": []}
    for k in k_values:
        result = run_located_drift(X, omega, k=k, emb_k=k, neg=neg, epochs=epochs,
                                    adjacency=adjacency, apply_step=True,
                                    force_model=force_model, fr_k=fr_k,
                                    negative_sampling=negative_sampling,
                                    seed=seed, verbose=False)
        initial = result["asymmetry_score"]
        final = result["asymmetry_score_final"]
        align = _alignment(result["asymmetry_per_node"], result["asymmetry_per_node_final"])
        out["initial"].append(initial)
        out["final"].append(final)
        out["alignment"].append(align)
        if verbose:
            pct = 100.0 * final / max(initial, 1e-12)
            print(f"  adjacency={adjacency:9s} k={k:3d}  initial={initial:.4f}  "
                  f"final={final:.4f}  ({pct:5.1f}% preserved)  alignment={align:.4f}")
    return out


def per_node_preservation(X, omega, k, adjacency, epochs, neg, seed, min_initial=1e-3,
                           force_model="fr_gravity", fr_k=None, negative_sampling=False):
    """
    [OURS 2026-09-01] --mode distribution's core computation: ONE
    run_located_drift call at a single k, then a per-node "% of target
    asymmetry preserved" ratio -- pct_i = 100 * final_i / initial_i --
    instead of the sweep's single global mean-based ratio.

    Nodes are split into two groups:
      - "valid": both initial_i and final_i finite, AND initial_i >=
        min_initial (default 1e-3) -- below that, dividing by an
        almost-zero target asymmetry makes pct_i blow up/become
        meaningless (a node with essentially no target asymmetry to begin
        with can't meaningfully "preserve X% of nothing").
      - "excluded": everything else (isolated nodes with no direct
        neighbours -> nan from asymmetry_score's own per-node computation,
        or nodes whose initial_i is below min_initial).

    Returns
    -------
    dict: {"pct": (n_valid,) ndarray, "initial": (n_valid,), "final": (n_valid,),
           "n_total": int, "n_valid": int, "n_excluded_nan": int,
           "n_excluded_near_zero": int}
    """
    result = run_located_drift(X, omega, k=k, emb_k=k, neg=neg, epochs=epochs,
                                adjacency=adjacency, apply_step=True,
                                force_model=force_model, fr_k=fr_k,
                                negative_sampling=negative_sampling,
                                seed=seed, verbose=False)
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
            "n_excluded_near_zero": n_excluded_near_zero}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["sweep", "distribution"], default="sweep",
                    help="[OURS 2026-09-01] 'sweep' (default) = global asymmetry_score vs. k, "
                         "one point per k. 'distribution' = per-node % preserved histogram at "
                         "ONE fixed k (--k).")
    p.add_argument("--dataset", choices=list(DATASET_GENERATORS.keys()), default="swiss_roll",
                    help="which of this project's 4 Randers-field datasets to use")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--k-min", type=int, default=5, help="[--mode sweep only]")
    p.add_argument("--k-max", type=int, default=60, help="[--mode sweep only]")
    p.add_argument("--k-step", type=int, default=5, help="[--mode sweep only]")
    p.add_argument("--k", type=int, default=20,
                    help="[--mode distribution only] the single k (n_neighbors) to run at.")
    p.add_argument("--bins", type=int, default=30,
                    help="[--mode distribution only] number of histogram bins.")
    p.add_argument("--min-initial", type=float, default=1e-3,
                    help="[--mode distribution only] nodes with initial (target) per-node "
                         "asymmetry below this are excluded from the % preserved histogram "
                         "(dividing by ~0 target asymmetry is not meaningful) -- reported "
                         "separately, not silently dropped.")
    p.add_argument("--adjacency", choices=["knn", "threshold"], default="knn",
                    help="which SINGLE adjacency construction to use -- shown in the plot "
                         "title.")
    p.add_argument("--epochs", type=int, default=300,
                    help="training epochs for each run_located_drift call -- --mode sweep "
                         "trains once PER k (n_k_values total runs); --mode distribution "
                         "trains once, total.")
    p.add_argument("--neg", type=int, default=10)
    p.add_argument("--force-model", choices=["fr_gravity", "umap"], default="fr_gravity",
                    help="[OURS 2026-09-02] attraction/repulsion law passed to "
                         "run_located_drift/randers_umap_fit -- 'fr_gravity' (default) = "
                         "Bannister et al.'s spring/inverse-square law, 'umap' = UMAP's own "
                         "fitted (a,b)-curve. Applied at every k in --mode sweep, and at the "
                         "single --k in --mode distribution.")
    p.add_argument("--fr-k", type=float, default=None,
                    help="[OURS 2026-09-02] natural edge-length constant for force_model="
                         "fr_gravity (default None -> 1/sqrt(n)). Ignored for force_model=umap.")
    p.add_argument("--neg-sampling", action="store_true",
                    help="[OURS 2026-09-02] use TRUE stochastic negative sampling for repulsion "
                         "instead of the dense/exact sum.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=None,
                    help="output PNG filename for the main plot -- defaults to "
                         "asymmetry_k_sweep_<dataset>.png (--mode sweep) or "
                         "asymmetry_distribution_<dataset>_k<k>.png (--mode distribution).")
    args = p.parse_args()

    print(f"Generating {args.dataset} + Randers field, n={args.n}, seed={args.seed}...")
    generator = DATASET_GENERATORS[args.dataset]
    X, omega, _label = generator(args.n, seed=args.seed)

    if args.mode == "distribution":
        if args.out is None:
            args.out = f"asymmetry_distribution_{args.dataset}_k{args.k}.png"

        print(f"Running once at k={args.k}, adjacency={args.adjacency}, epochs={args.epochs} "
              f"(single training run, per-node preservation)...")
        d = per_node_preservation(X, omega, args.k, args.adjacency, args.epochs, args.neg,
                                   args.seed, min_initial=args.min_initial,
                                   force_model=args.force_model, fr_k=args.fr_k,
                                   negative_sampling=args.neg_sampling)

        # ---- summary table (binned) ----
        edges = [0, 25, 50, 75, 100, 125, 150, float("inf")]
        labels = ["0-25%", "25-50%", "50-75%", "75-100%", "100-125%", "125-150%", "150%+"]
        print(f"\n--- per-node % preserved distribution ({args.dataset}, k={args.k}, "
              f"adjacency={args.adjacency}, n={args.n}, epochs={args.epochs}) ---")
        print(f"total nodes = {d['n_total']}  |  valid = {d['n_valid']}  |  "
              f"excluded (isolated/nan) = {d['n_excluded_nan']}  |  "
              f"excluded (initial < {args.min_initial}) = {d['n_excluded_near_zero']}")
        for lo, hi, label in zip(edges[:-1], edges[1:], labels):
            count = int(((d["pct"] >= lo) & (d["pct"] < hi)).sum())
            frac = 100.0 * count / max(d["n_valid"], 1)
            print(f"  {label:>10s} : {count:5d} nodes  ({frac:5.1f}% of valid)")
        print(f"\nmean={d['pct'].mean():.1f}%  median={np.median(d['pct']):.1f}%  "
              f"std={d['pct'].std():.1f}%  min={d['pct'].min():.1f}%  max={d['pct'].max():.1f}%")

        # ---- histogram plot ----
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(d["pct"], bins=args.bins, color="tab:purple", edgecolor="white", alpha=0.85)
        ax.axvline(np.median(d["pct"]), color="k", linestyle="--", linewidth=1.2,
                   label=f"median = {np.median(d['pct']):.1f}%")
        ax.axvline(100.0, color="tab:red", linestyle=":", linewidth=1.2,
                   label="100% (fully preserved)")
        ax.set_xlabel("% of node's target (initial) asymmetry preserved in final embedding")
        ax.set_ylabel("number of nodes")
        ax.set_title(f"Per-node asymmetry preservation distribution\n"
                     f"({args.dataset}, n={args.n}, k={args.k}, epochs={args.epochs}, "
                     f"adjacency={args.adjacency})")
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

    # ---- mode == "sweep" (default, unchanged) ----
    if args.out is None:
        args.out = f"asymmetry_k_sweep_{args.dataset}.png"

    k_values = list(range(args.k_min, args.k_max + 1, args.k_step))

    print(f"Sweeping k over {k_values}, adjacency={args.adjacency}, epochs={args.epochs} "
          f"(training a real embedding at every k)...")
    results = sweep_asymmetry_vs_k(X, omega, k_values, adjacency=args.adjacency,
                                    epochs=args.epochs, neg=args.neg, seed=args.seed,
                                    force_model=args.force_model, fr_k=args.fr_k,
                                    negative_sampling=args.neg_sampling)

    # ---- results table ----
    print(f"\n--- asymmetry_k_sweep results ({args.dataset}, adjacency={args.adjacency}, "
          f"n={args.n}, epochs={args.epochs}) ---")
    print(f"{'k':>4}  {'initial':>8}  {'final':>8}  {'% preserved':>12}  {'alignment':>9}")
    for k, initial, final, align in zip(k_values, results["initial"], results["final"],
                                         results["alignment"]):
        pct = 100.0 * final / max(initial, 1e-12)
        print(f"{k:4d}  {initial:8.4f}  {final:8.4f}  {pct:11.1f}%  {align:9.4f}")

    # ---- plot 1: initial vs. final asymmetry_score ----
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_values, results["initial"], marker="o", color="tab:blue", label="initial (target, D_asym)")
    ax.plot(k_values, results["final"], marker="s", color="tab:red", label="final (trained embedding)")
    ax.set_xlabel("k (n_neighbors)")
    ax.set_ylabel("asymmetry_score (global)")
    ax.set_title(f"Asymmetry score vs. k  ({args.dataset}, n={args.n}, epochs={args.epochs})\n"
                 f"adjacency = {args.adjacency}")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_png = HERE / args.out
    fig.savefig(out_png, dpi=150)
    print(f"\nSaved main plot to {out_png}")

    # ---- plot 2: alignment (Pearson corr of per-node initial vs. final) ----
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(k_values, results["alignment"], marker="^", color="tab:green")
    ax2.axhline(0.0, color="k", linestyle=":", linewidth=1)
    ax2.set_xlabel("k (n_neighbors)")
    ax2.set_ylabel("alignment (Pearson corr., per-node initial vs. final)")
    ax2.set_title(f"Per-node asymmetry alignment vs. k  ({args.dataset}, n={args.n}, "
                 f"epochs={args.epochs})\nadjacency = {args.adjacency}")
    ax2.set_ylim(-1, 1)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

    out_align_png = out_png.with_name(out_png.stem + "_alignment" + out_png.suffix)
    fig2.savefig(out_align_png, dpi=150)
    print(f"Saved alignment plot to {out_align_png}")

    out_npz = out_png.with_suffix(".npz")
    np.savez(out_npz, k_values=np.array(k_values),
             initial=np.array(results["initial"]), final=np.array(results["final"]),
             alignment=np.array(results["alignment"]))
    print(f"Saved raw sweep data to {out_npz}")


if __name__ == "__main__":
    main()
