"""
test.py -- diagnostics for the swiss-roll located-drift pipeline (run_swiss_roll.py).

Pulls the swiss-roll + Randers-field data itself (no separate data file
needed), runs the located-drift mechanism (gravity off, then on, for
comparison), and reports three checks:

  1. randers_validity_check -- ||b_i|| <= 1-clip_delta, generic, ported
     unchanged from the migration project's own test.py.
  2. stress -- reconstruction quality of rho from (Y, B) against the true
     D_asym, generic, same formula as the migration project's test.py
     (that file relied on finsler_mds.py's finsler_distances_freeB for the
     reconstruction step; reconstruct_rho() below is the same formula,
     just inlined since randers_umap.py has no equivalent standalone
     function).
  3. direction_accuracy_swiss -- NEW, swiss-roll-specific. The migration
     project's direction_accuracy() needed ground truth F_true, an (n,n)
     flow matrix, and a carefully-derived sign convention (documented at
     length in that function's docstring, and got it backwards once). Swiss
     roll's ground truth is stronger and doesn't need any of that: alpha(t)
     is an exact, closed-form, per-point SIGNED scalar (the true drift
     strength along the roll's length direction) -- no proxy, no derivation
     needed. We estimate the local "increasing t" direction inside the
     EMBEDDING via a k-NN-weighted finite difference, then correlate
     <b_i, that direction> against alpha(t_i). No sign convention is
     assumed up front -- the sign of the correlation tells us the
     convention empirically, avoiding the kind of bug the migration
     project's docstring describes.

Usage
-----
    python3 test.py
    python3 test.py --n 1000 --locate-epochs 200 --epochs 300
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_swiss_roll import make_swiss_roll_randers
from randers_bridge import run_located_drift


# ─────────────────────────────────────────────────────────────────────────
# 1. Reconstruction stress -- same formula as the migration project's test.py
# ─────────────────────────────────────────────────────────────────────────
def stress(D_target: np.ndarray, D_reconstructed: np.ndarray) -> float:
    """
    Normalised reconstruction stress over off-diagonal entries:
        sqrt(sum((D_target - D_rec)^2)) / sqrt(sum(D_target^2))
    Lower is better; 0 = perfect reconstruction.
    """
    mask = ~np.eye(len(D_target), dtype=bool)
    num = ((D_target[mask] - D_reconstructed[mask]) ** 2).sum()
    den = (D_target[mask] ** 2).sum() + 1e-12
    return float(np.sqrt(num) / np.sqrt(den))


def reconstruct_rho(Y: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    rho_{i->j} = ||y_i-y_j|| + b_i . (y_j-y_i)   -- same formula randers_umap.py's
    own training loop uses, standalone here so stress() can compare it
    against D_asym after training.
    """
    diff = Y[np.newaxis, :, :] - Y[:, np.newaxis, :]
    d = np.sqrt((diff ** 2).sum(-1))
    proj = (B[:, np.newaxis, :] * diff).sum(-1)
    rho = d + proj
    np.fill_diagonal(rho, 0.0)
    return rho


# ─────────────────────────────────────────────────────────────────────────
# 2. Randers validity check -- ||b_i|| <= 1 - clip_delta  [DAGES]
#    (unchanged from the migration project's test.py -- fully generic)
# ─────────────────────────────────────────────────────────────────────────
def randers_validity_check(B: np.ndarray, clip_delta: float = 0.01) -> dict:
    norms = np.linalg.norm(B, axis=1)
    limit = 1.0 - clip_delta
    n_violations = int((norms > limit + 1e-9).sum())
    return {
        "max_norm": float(norms.max()),
        "mean_norm": float(norms.mean()),
        "limit": limit,
        "n_violations": n_violations,
        "ok": n_violations == 0,
    }


# ─────────────────────────────────────────────────────────────────────────
# 3. Direction-accuracy diagnostic -- swiss-roll-specific, exact ground truth
# ─────────────────────────────────────────────────────────────────────────
def direction_accuracy_swiss(Y: np.ndarray, B: np.ndarray, t: np.ndarray,
                              alpha: np.ndarray, k: int = 10) -> dict:
    """
    Does b_i point the way the TRUE Randers field (alpha(t_i)) says it should?

    For each point i, estimate the local "increasing t" direction inside
    the embedding Y via a k-NN-weighted finite difference:
        that_i  propto  sum_{j in kNN(i)} (t_j - t_i) * (y_j - y_i)
    then look at proj_i = <b_i, that_i>. If the method is capturing the
    true field, proj_i should correlate with alpha(t_i) (whose sign flips
    along the roll, exactly as the drift's should).

    No sign convention is hard-coded: the sign of `correlation` itself
    tells you empirically whether "b_i aligned with that_i" means "toward
    increasing alpha" or "toward decreasing alpha" for this construction.
    A near-zero correlation means the method captured no directional
    signal at all, regardless of convention.

    Returns
    -------
    dict: correlation (Pearson, proj vs alpha), pct_sign_match (naive,
          assumes matching sign = correct -- read alongside correlation's
          own sign), n_valid (points with both a well-defined that_i and
          nonzero b_i)
    """
    n = Y.shape[0]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Y)
    _, idx = nn.kneighbors(Y)
    idx = idx[:, 1:]  # drop self

    t_hat = np.zeros_like(Y)
    for i in range(n):
        j = idx[i]
        diff = Y[j] - Y[i]
        dt = t[j] - t[i]
        v = (dt[:, None] * diff).sum(axis=0)
        nrm = np.linalg.norm(v)
        if nrm > 1e-10:
            t_hat[i] = v / nrm

    proj = (B * t_hat).sum(axis=1)
    valid = (np.linalg.norm(t_hat, axis=1) > 1e-10) & (np.linalg.norm(B, axis=1) > 1e-10)

    if valid.sum() < 3:
        return {"correlation": 0.0, "pct_sign_match": 0.0, "n_valid": int(valid.sum())}

    corr = float(np.corrcoef(proj[valid], alpha[valid])[0, 1])
    sign_match = float((np.sign(proj[valid]) == np.sign(alpha[valid])).mean() * 100)
    return {"correlation": corr, "pct_sign_match": sign_match, "n_valid": int(valid.sum())}


# ─────────────────────────────────────────────────────────────────────────
# 4. Standalone report
# ─────────────────────────────────────────────────────────────────────────
def evaluate(label, X, omega, t, alpha, args, use_gravity):
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    result = run_located_drift(X, omega, k=args.k, emb_k=args.k, neg=args.neg,
                               locate_epochs=args.locate_epochs, epochs=args.epochs,
                               clip_delta=args.clip_delta, use_gravity=use_gravity,
                               seed=args.seed, verbose=not args.quiet)
    Y, B, D_asym = result["Y"], result["B"], result["D_asym"]

    print(f"\n[1/3] Randers validity")
    rc = randers_validity_check(B, args.clip_delta)
    status = "OK" if rc["ok"] else "FAIL"
    print(f"      max||b_i||={rc['max_norm']:.4f}  mean||b_i||={rc['mean_norm']:.4f}  "
          f"(limit {rc['limit']:.2f})  violations={rc['n_violations']}  ({status})")

    print(f"\n[2/3] Reconstruction stress")
    rho = reconstruct_rho(Y, B)
    s = stress(D_asym, rho)
    print(f"      stress={s:.4f}  (lower is better, 0=perfect)")

    print(f"\n[3/3] Direction accuracy (ground truth: exact alpha(t))")
    da = direction_accuracy_swiss(Y, B, t, alpha, k=args.dir_k)
    print(f"      correlation(<b_i,that_i>, alpha_i) = {da['correlation']:+.4f}")
    print(f"      pct_sign_match (naive)             = {da['pct_sign_match']:.1f}%  "
          f"(n_valid={da['n_valid']}/{X.shape[0]})")

    bn = np.linalg.norm(B, axis=1)
    print(f"\nextent={Y.max()-Y.min():.2f}  mean||b||={bn.mean():.4f}")
    return {"stress": s, "correlation": da["correlation"],
            "pct_sign_match": da["pct_sign_match"], "extent": float(Y.max() - Y.min())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n",      type=int, default=1000)
    p.add_argument("--k",      type=int, default=15, help="k-NN for the geodesic backbone")
    p.add_argument("--neg",    type=int, default=10)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--locate-epochs", type=int, default=500)
    p.add_argument("--clip-delta", type=float, default=0.01)
    p.add_argument("--dir-k",  type=int, default=10, help="k-NN for direction_accuracy_swiss")
    p.add_argument("--seed",   type=int, default=0)
    p.add_argument("--quiet",  action="store_true")
    args = p.parse_args()

    print(f"Generating swiss roll: n={args.n}")
    X, omega, t = make_swiss_roll_randers(args.n, seed=42)
    alpha = -0.5 * np.cos(t) + 0.3 * np.sin(t)   # exact ground truth, same formula as data gen

    r_off = evaluate("gravity OFF (located-drift only)", X, omega, t, alpha, args, use_gravity=False)
    r_on  = evaluate("gravity ON  (located-drift + b_i gravity)", X, omega, t, alpha, args, use_gravity=True)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"{'':20} {'stress':>10} {'correlation':>12} {'sign_match':>11} {'extent':>10}")
    print(f"{'gravity OFF':20} {r_off['stress']:10.4f} {r_off['correlation']:+12.4f} "
          f"{r_off['pct_sign_match']:10.1f}% {r_off['extent']:10.2f}")
    print(f"{'gravity ON':20} {r_on['stress']:10.4f} {r_on['correlation']:+12.4f} "
          f"{r_on['pct_sign_match']:10.1f}% {r_on['extent']:10.2f}")


if __name__ == "__main__":
    main()
