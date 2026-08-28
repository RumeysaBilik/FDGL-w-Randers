"""
drift_magnitude_test.py -- per-epoch drift-magnitude monitor / sanity check.

[OURS 2026-08-27, per explicit user request -- "sence her epochta driftlerin
boyunu olcen bir test ve kontrol mekanizmasi kurabilir miyiz kodumuza ek bi
dosyada"; generalised across datasets 2026-08-28 per "bunu sadece swiss roll
icin degil de her data seti icin uygulanabilir hale getirir misin"]
Standalone diagnostic (does not modify randers_umap.py, run_swiss_roll.py,
run_mammoth.py, run_sphere_radial.py, run_sphere_tangential.py, or test.py)
that tracks ||b_i|| epoch by epoch during training and checks it against the
bound the pipeline is supposed to respect, for BOTH the frozen-B mechanism
(--normalize off, the original clip_delta-based B_located) and the
live-rescaled mechanism (--normalize on, the divide-by-kth-nn-distance rule
from 2026-08-27 -- see fdgl_report.tex's "Drift Magnitude Normalization"
section for the full derivation of both).

Works across all four datasets that share the located-drift mechanism
(swiss_roll, mammoth, sphere_radial, sphere_tangential -- run_mammoth.py/
run_sphere_radial.py/run_sphere_tangential.py all import run_located_drift
straight from run_swiss_roll.py rather than reimplementing it, and every
make_*_randers() generator returns the same (X, omega, extra) triple, so a
single small registry below is enough to dispatch to any of them; no
per-dataset logic is duplicated). Not applicable to MNIST/BreastCancer --
those use the live-drift mechanism (B_fixed=None, use_drift=True,
compute_drift's own norm_mode="relative" bound), which has no locate step
and no --normalize flag to compare against in the first place.

Mechanism: runs run_swiss_roll.py's own run_located_drift() with
snapshot_every=N (reusing the SAME snapshot machinery randers_umap_fit
already has -- no new capture logic needed), reads ||b_i|| from every
captured snapshot's B, and reports:
  - a per-epoch trajectory (mean/max ||b_i||), plotted for both mechanisms
    side by side against the theoretical bound 1-clip_delta
  - an explicit PASS/VIOLATION check: does max||b_i|| ever exceed the bound,
    and in how many of the sampled epochs

Expected result, if the divide-based fix from 2026-08-27 is working as
intended: normalize=False should be flat and always within the bound (B is
frozen at locate time, untouched by training); normalize=True should mostly
stay within the same bound too, except possibly a brief transient at the
very start of training (before the embedding has spread past unit scale,
i.e. before kth_dist(t) >= ||B_located||) -- this transient, if present, is
visible directly in the plotted trajectory, not just asserted. This should
hold regardless of dataset, since the bound comes from clip_delta/the divide
rule itself, not from any dataset-specific geometry.

Usage
-----
    python3 drift_magnitude_test.py
    python3 drift_magnitude_test.py --dataset mammoth --n 1000 --epochs 300
    python3 drift_magnitude_test.py --dataset sphere_radial --epochs 300
    python3 drift_magnitude_test.py --dataset sphere_tangential --epochs 300
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
from run_mammoth import make_mammoth_randers
from run_sphere_radial import make_sphere_radial_randers
from run_sphere_tangential import make_sphere_tangential_randers

# [OURS 2026-08-28] every make_*_randers() below has the same (n, seed=...)
# -> (X, omega, extra) interface, and every run_*.py's own run_located_drift
# is the SAME function (imported from run_swiss_roll.py, not reimplemented)
# -- so dispatching by name is enough, no per-dataset branching needed
# anywhere else in this file.
DATASET_MAKERS = {
    "swiss_roll": make_swiss_roll_randers,
    "mammoth": make_mammoth_randers,
    "sphere_radial": make_sphere_radial_randers,
    "sphere_tangential": make_sphere_tangential_randers,
}


def collect_norm_trajectory(X, omega, normalize, epochs, snapshot_every, k, emb_k,
                             clip_delta, seed, verbose):
    """
    Runs the located-drift pipeline once, returns the ||b_i|| trajectory
    (mean/max/min per captured epoch) read straight from randers_umap_fit's
    own snapshot_every mechanism -- no separate training loop, no
    duplicated logic.
    """
    out = run_located_drift(X, omega, k=k, emb_k=emb_k, epochs=epochs,
                             clip_delta=clip_delta, snapshot_every=snapshot_every,
                             normalize_drift_by_asymmetry=normalize,
                             seed=seed, verbose=verbose)
    snaps = out["snapshots"]
    epoch_list, mean_norms, max_norms, min_norms = [], [], [], []
    for snap in snaps:
        bn = np.linalg.norm(snap["B"], axis=1)
        epoch_list.append(snap["epoch"])
        mean_norms.append(bn.mean())
        max_norms.append(bn.max())
        min_norms.append(bn.min())
    return {
        "epoch": np.array(epoch_list),
        "mean": np.array(mean_norms),
        "max": np.array(max_norms),
        "min": np.array(min_norms),
    }


def check_bounded(traj, limit, label):
    """
    PASS/VIOLATION check: does max||b_i|| ever exceed `limit` in any
    sampled epoch? Returns a dict summarising the result (not just a bool)
    so the caller can print/report specifics, not just ok/not-ok.
    """
    violating = traj["max"] > (limit + 1e-6)
    n_violations = int(violating.sum())
    worst_epoch = int(traj["epoch"][np.argmax(traj["max"])])
    return {
        "label": label,
        "limit": float(limit),
        "global_max": float(traj["max"].max()),
        "worst_epoch": worst_epoch,
        "n_violating_epochs": n_violations,
        "n_epochs_checked": len(traj["epoch"]),
        "ok": n_violations == 0,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=sorted(DATASET_MAKERS.keys()), default="swiss_roll",
                    help="[OURS 2026-08-28] which dataset's make_*_randers() to run this "
                         "check against -- all four share the exact same located-drift "
                         "mechanism (run_located_drift, imported from run_swiss_roll.py), "
                         "so the same check applies unchanged to any of them.")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--emb-k", type=int, default=20)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--snapshot-every", type=int, default=5,
                    help="how often (in epochs) to sample ||b_i|| -- every epoch (1) is the "
                         "most precise but slowest; 5 is a reasonable default for a quick check.")
    p.add_argument("--clip-delta", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,
                    help="defaults to drift_magnitude_test_<dataset>.png")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    verbose = not args.quiet
    out_name = args.out or f"drift_magnitude_test_{args.dataset}.png"

    make_randers = DATASET_MAKERS[args.dataset]
    if verbose:
        print(f"Generating '{args.dataset}' data (n={args.n})...")
    X, omega, _extra = make_randers(args.n, seed=args.seed)
    limit = 1.0 - args.clip_delta

    if verbose:
        print("Running WITHOUT --normalize (frozen B_located, clip_delta-bounded)...")
    traj_off = collect_norm_trajectory(X, omega, False, args.epochs, args.snapshot_every,
                                        args.k, args.emb_k, args.clip_delta, args.seed, verbose)

    if verbose:
        print("\nRunning WITH --normalize (||B_located||/kth_dist(t), 2026-08-27 divide rule)...")
    traj_on = collect_norm_trajectory(X, omega, True, args.epochs, args.snapshot_every,
                                       args.k, args.emb_k, args.clip_delta, args.seed, verbose)

    check_off = check_bounded(traj_off, limit, "normalize=False (frozen)")
    check_on = check_bounded(traj_on, limit, "normalize=True (divide)")

    print("\n--- drift magnitude bound check (limit = 1 - clip_delta = "
          f"{limit:.4f}) ---")
    for c in (check_off, check_on):
        status = "PASS" if c["ok"] else "VIOLATION"
        print(f"[{status}] {c['label']}: global max||b_i|| = {c['global_max']:.4f} "
              f"(at epoch {c['worst_epoch']}), violating epochs = "
              f"{c['n_violating_epochs']}/{c['n_epochs_checked']}")

    # ---- plot: mean/max ||b_i|| vs epoch, both mechanisms + the bound ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(traj_off["epoch"], traj_off["mean"], color="tab:blue",
            label="mean||b_i||  (normalize=False)")
    ax.plot(traj_off["epoch"], traj_off["max"], color="tab:blue", linestyle="--",
            label="max||b_i||  (normalize=False)")
    ax.plot(traj_on["epoch"], traj_on["mean"], color="tab:red",
            label="mean||b_i||  (normalize=True)")
    ax.plot(traj_on["epoch"], traj_on["max"], color="tab:red", linestyle="--",
            label="max||b_i||  (normalize=True)")
    ax.axhline(limit, color="k", linestyle=":", linewidth=1,
               label=f"clip bound (1-delta={limit:.3f})")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\|b_i\|$")
    ax.set_title(f"Drift magnitude over training ({args.dataset}, n={args.n})")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path = HERE / out_name
    fig.savefig(out_path, dpi=150)
    if verbose:
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
