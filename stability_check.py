#!/usr/bin/env python3
"""
stability_check.py -- [OURS 2026-08-14, per UMAP paper Section 5.3
"Embedding Stability"] measures how much our located-drift swiss-roll
embedding changes as a function of sample size, using normalized
Procrustes distance (randers_umap.py::procrustes_align).

Recipe (mirrors the paper's Figure 8 exactly, adapted to our setting):
    1. Generate the swiss roll ONCE at n_full points (X_full, omega_full, t_full).
    2. Run the full located-drift pipeline (run_swiss_roll.py::run_located_drift)
       on the full n_full points -> Y_full.
    3. For each n_sub in --sub-sizes: take the FIRST n_sub points of
       X_full/omega_full (exact slice, not a fresh generation -- see note
       below) and run the SAME pipeline independently -> Y_sub.
    4. Procrustes-align Y_sub onto Y_full[:n_sub] (the corresponding points'
       positions from the big run) and record the normalized distance.
    5. Plot distance vs n_sub/n_full, mirroring the paper's Figure 8.

Note on point correspondence: make_swiss_roll_randers(n, seed=42) does NOT
have a clean nested-prefix property across different n for X's height/y
coordinate. t and omega happen to nest correctly across n (verified
empirically -- they only depend on the FIRST n draws of the RNG stream),
but `height` is drawn from the SAME RandomState right after t, so its
starting offset in the stream depends on n (e.g. offset 2000 when n=2000,
offset 500 when n=500) -- this breaks the nesting for X's y-column
specifically (X_full[:500,1] != X_sub(n=500)[:,1], even though the x/z
columns and t/omega DO match). To sidestep this entirely, this script
generates X/omega/t ONCE at n_full and slices down for every n_sub, rather
than calling make_swiss_roll_randers(n_sub) again for each subsample --
this guarantees identical ambient points by construction, independent of
any RNG-stream subtlety.

Usage
-----
    python3 stability_check.py --n-full 2000 --sub-sizes 100 300 600 1000 1500 --ramp
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

from run_swiss_roll import make_swiss_roll_randers, run_located_drift
from randers_umap import procrustes_align


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-full", type=int, default=2000)
    p.add_argument("--sub-sizes", type=int, nargs="+", default=[100, 300, 600, 1000, 1500])
    p.add_argument("--k", type=int, default=20, help="k-NN for the geodesic backbone")
    p.add_argument("--neg", type=int, default=10)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--locate-epochs", type=int, default=500)
    p.add_argument("--ramp", action="store_true",
                    help="ramp B_fixed's magnitude 0->1 (recommended, matches run_swiss_roll.py "
                         "default recommendation for large/noisy n)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="stability_check")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if not args.quiet:
        print(f"Generating swiss roll ONCE at n_full={args.n_full} (sub-samples are exact "
              f"prefixes of this, not fresh generations -- see module docstring)...")
    X_full, omega_full, t_full = make_swiss_roll_randers(args.n_full, seed=42)

    if not args.quiet:
        print(f"\n=== full run, n={args.n_full} ===")
    result_full = run_located_drift(X_full, omega_full, k=args.k, emb_k=args.k,
                                     neg=args.neg, locate_epochs=args.locate_epochs,
                                     epochs=args.epochs, ramp=args.ramp,
                                     seed=args.seed, verbose=not args.quiet)
    Y_full = result_full["Y"]

    sub_sizes = sorted(s for s in args.sub_sizes if s < args.n_full)
    distances = []
    fractions = []

    for n_sub in sub_sizes:
        if not args.quiet:
            print(f"\n=== sub run, n={n_sub} ({n_sub/args.n_full:.1%} of full) ===")
        X_sub = X_full[:n_sub]
        omega_sub = omega_full[:n_sub]
        result_sub = run_located_drift(X_sub, omega_sub, k=args.k, emb_k=args.k,
                                        neg=args.neg, locate_epochs=args.locate_epochs,
                                        epochs=args.epochs, ramp=args.ramp,
                                        seed=args.seed, verbose=not args.quiet)
        Y_sub = result_sub["Y"]

        _, dist = procrustes_align(Y_full[:n_sub], Y_sub)
        distances.append(dist)
        fractions.append(n_sub / args.n_full)
        if not args.quiet:
            print(f"  normalized Procrustes distance vs full run: {dist:.4f}")

    # ---- plot: distance vs subsample fraction, mirrors UMAP paper Fig 8 ----
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([f * 100 for f in fractions], distances, "o-", color="k")
    ax.set_xlabel("subsample size (% of n_full)")
    ax.set_ylabel("normalized Procrustes distance vs. full-run embedding")
    ax.set_title(f"Embedding stability under subsampling (n_full={args.n_full})", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)

    np.savez(f"{args.out}.npz", sub_sizes=np.array(sub_sizes),
             fractions=np.array(fractions), distances=np.array(distances))

    if not args.quiet:
        print(f"\nwrote {args.out}.png and {args.out}.npz")
        print(f"\nsummary:")
        for n_sub, frac, dist in zip(sub_sizes, fractions, distances):
            print(f"  n={n_sub:5d} ({frac:5.1%})  distance={dist:.4f}")


if __name__ == "__main__":
    main()
