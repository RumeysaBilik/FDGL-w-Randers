#!/usr/bin/env python3
"""
asymmetry_k_sweep.py -- sweeps k (n_neighbors) for a chosen adjacency
construction (knn / threshold / both) and plots how asymmetry_score
(randers_bridge.asymmetry_score) responds, on any of this project's 4
Randers-field datasets (swiss_roll, mammoth, sphere_tangential,
sphere_radial) -- see --dataset below.

Only the "apply" step's D_asym build is needed to get asymmetry_score --
see run_swiss_roll.py's run_located_drift(), where asymmetry_score is
computed right after compute_dist_matrix(..., return_adjacency=True) and
BEFORE randers_umap_fit() is ever called. So this sweep does NOT need to
run any force-directed training at all: X, omega are generated once, then
for each k we just rebuild the D_asym graph and read off the score --
cheap, no epochs involved.

Usage
-----
    python asymmetry_k_sweep.py
    python asymmetry_k_sweep.py --n 2000 --k-min 5 --k-max 80 --k-step 5
    python asymmetry_k_sweep.py --adjacency both --out k_sweep.png
    python asymmetry_k_sweep.py --dataset mammoth --adjacency both
    python asymmetry_k_sweep.py --dataset sphere_tangential
    python asymmetry_k_sweep.py --dataset sphere_radial
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

from randers_bridge import compute_dist_matrix, asymmetry_score
from run_swiss_roll import make_swiss_roll_randers
from run_mammoth import make_mammoth_randers
from run_sphere_tangential import make_sphere_tangential_randers
from run_sphere_radial import make_sphere_radial_randers

# [OURS 2026-08-25, per explicit user request -- "başka data setleri de
# olsun"] each generator shares the same (X, omega, label) return
# signature (label is only used for plot colouring elsewhere -- unused
# here), so a single dispatch dict is enough, no per-dataset branching
# needed beyond this.
DATASET_GENERATORS = {
    "swiss_roll": make_swiss_roll_randers,
    "mammoth": make_mammoth_randers,
    "sphere_tangential": make_sphere_tangential_randers,
    "sphere_radial": make_sphere_radial_randers,
}


def sweep_asymmetry_vs_k(X, omega, k_values, adjacency="knn", verbose=True):
    """
    For each k in k_values, rebuild D_asym with the given adjacency mode
    and return the resulting global asymmetry_score. No training involved
    -- this mirrors exactly the D_asym build in run_located_drift()'s
    "apply" step (run_swiss_roll.py lines ~204-214), just swept over k.

    Returns
    -------
    scores : list of float, same length as k_values
    """
    scores = []
    for k in k_values:
        D_asym, _, bln_asym = compute_dist_matrix(
            X, n_neighbors=k, path_method="auto",
            randers_field=omega, adjacency=adjacency, return_adjacency=True,
        )
        _, global_score = asymmetry_score(D_asym, bln_asym)
        scores.append(global_score)
        if verbose:
            print(f"  adjacency={adjacency:9s} k={k:3d}  asymmetry_score={global_score:.4f}")
    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASET_GENERATORS.keys()), default="swiss_roll",
                    help="which of this project's 4 Randers-field datasets to sweep")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--k-min", type=int, default=5)
    p.add_argument("--k-max", type=int, default=60)
    p.add_argument("--k-step", type=int, default=5)
    p.add_argument("--adjacency", choices=["knn", "threshold", "both"], default="knn",
                    help="which adjacency construction(s) to sweep -- 'both' plots "
                         "knn and threshold as two lines for side-by-side comparison")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=None,
                    help="output PNG filename -- defaults to "
                         "asymmetry_k_sweep_<dataset>.png")
    args = p.parse_args()

    if args.out is None:
        args.out = f"asymmetry_k_sweep_{args.dataset}.png"

    k_values = list(range(args.k_min, args.k_max + 1, args.k_step))
    modes = ["knn", "threshold"] if args.adjacency == "both" else [args.adjacency]

    print(f"Generating {args.dataset} + Randers field, n={args.n}, seed={args.seed}...")
    generator = DATASET_GENERATORS[args.dataset]
    X, omega, _label = generator(args.n, seed=args.seed)

    print(f"Sweeping k over {k_values} for adjacency mode(s) {modes}...")
    results = {}
    for mode in modes:
        results[mode] = sweep_asymmetry_vs_k(X, omega, k_values, adjacency=mode)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"knn": "tab:blue", "threshold": "tab:orange"}
    markers = {"knn": "o", "threshold": "s"}
    for mode in modes:
        ax.plot(k_values, results[mode], marker=markers[mode], color=colors[mode],
                 label=f"adjacency={mode}")
    ax.set_xlabel("k (n_neighbors)")
    ax.set_ylabel("asymmetry_score (global)")
    ax.set_title(f"Asymmetry score vs. k  ({args.dataset}, n={args.n})")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    if len(modes) > 1:
        ax.legend()
    fig.tight_layout()

    out_png = HERE / args.out
    fig.savefig(out_png, dpi=150)
    print(f"\nSaved plot to {out_png}")

    out_npz = out_png.with_suffix(".npz")
    save_kwargs = {"k_values": np.array(k_values)}
    for mode in modes:
        save_kwargs[f"scores_{mode}"] = np.array(results[mode])
    np.savez(out_npz, **save_kwargs)
    print(f"Saved raw sweep data to {out_npz}")


if __name__ == "__main__":
    main()
