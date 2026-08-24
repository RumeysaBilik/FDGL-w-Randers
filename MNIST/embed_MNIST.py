#!/usr/bin/env python3
"""
embed_MNIST.py -- embed the asymmetric MNIST distance matrix (produced by
asymm_dist_MNIST.py) with Randers-UMAP, coloured by digit label.

Usage
-----
    python3 asymm_dist_MNIST.py     # writes MNIST/asymm_matrix.npy + MNIST/labels.npy
    python3 embed_MNIST.py          # reads those, writes mnist_embedding.png/.npz
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

from randers_umap import randers_umap_fit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=20, help="n_neighbors for randers_umap_fit")
    p.add_argument("--neg", type=int, default=10)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--gravity", action="store_true",
                    help="add per-node gravity = b_i (no extra scaling)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="mnist_embedding")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    D_asym = np.load(HERE / "MNIST" / "asymm_matrix.npy")
    labels = np.load(HERE / "MNIST" / "labels.npy")

    if not args.quiet:
        print(f"D_asym: {D_asym.shape}  symmetric={np.allclose(D_asym, D_asym.T)}")

    out = randers_umap_fit(D_asym, n_neighbors=args.k, n_negative_samples=args.neg,
                            n_epochs=args.epochs, use_drift=True,
                            use_gravity=args.gravity, seed=args.seed,
                            verbose=not args.quiet)
    Y, B = out["Y"], out["B"]

    # ---- plot ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(Y[:, 0], Y[:, 1], c=labels, cmap="tab10", s=6, alpha=0.85, linewidths=0)
    cbar = plt.colorbar(sc, ax=ax, label="digit", ticks=range(10))

    bn = np.linalg.norm(B, axis=1)
    big = np.argsort(bn)[::-1][:25]
    if bn.max() > 0:
        sc_scale = 0.12 * (Y.max() - Y.min()) / bn.max()
        ax.quiver(Y[big, 0], Y[big, 1], B[big, 0] * sc_scale, B[big, 1] * sc_scale,
                  color="k", alpha=0.6, width=0.004, scale=1, scale_units="xy")

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Randers-UMAP on MNIST asymmetric distances (n={D_asym.shape[0]})", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)

    np.savez(f"{args.out}.npz", Y=Y, B=B, labels=labels)

    if not args.quiet:
        print(f"\nwrote {args.out}.png and {args.out}.npz")


if __name__ == "__main__":
    main()
