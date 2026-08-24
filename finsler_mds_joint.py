"""
finsler_mds_joint.py
=====================
[Ported from asymmetricfinsler-mds_demo/finsler_mds.py, then adapted.
Default mode (train_B=False): B is computed ONCE from the SVD of the
skew-symmetric part of D_asym at init, clipped to the Randers bound, and
then held FIXED ("attached to the point") for the whole run -- only Y is
trained by Adam, using that frozen B inside the Finsler forward pass every
epoch. This is the "D mentality" analogue of randers_umap_fit's B_fixed
(located-drift) path in the g-mentality/UMAP-force pipeline. Optionally
(train_B=True) B can also be trained jointly with Y (the original file's
free-B design) -- see finsler_mds_freeB's own docstring. svd_init() itself
is NOT redefined here -- imported from svd_radar.py so there's a single
source of truth for it.]

SOURCE ATTRIBUTION
------------------
[DAGES]  Dages et al., "Finsler Multi-Dimensional Scaling", CVPR 2025,
         arXiv:2503.18010 -- original Finsler MDS formulation (Eq. 1,
         standard MDS gradient w.r.t. Y, Randers validity clipping).
[OURS]   pair_mask (exclude zero-both/incomparable pairs from loss),
         free-B (B is a free (n,d) parameter jointly optimised with Y,
         not recomputed post-hoc from Y), per-node gravity term.
         See asymmetricfinsler-mds_demo/finsler_mds.py's own docstring
         for the full derivation history (frozen-B vs free-B, etc).

Loss (masked -- observed pairs only):
  L(Y, B) = sum_{(i,j) observed} (F(i->j) - D_asym[i,j])^2
  F(i->j) = ||y_j - y_i|| + <b_i, e_ij>          [DAGES Eq. 1]

Gradients:
  dL/db_i = 2 * sum_j res[i,j] * e_ij             [OURS]
  dL/dy_i = source + target contributions          [DAGES]

Randers validity: ||b_i|| <= 1 - clip_delta        [DAGES constraint]
"""

import numpy as np

from svd_radar import svd_init


# ─────────────────────────────────────────────────────────────────────────
# Optimiser -- standard Adam, extended to update [Y, B] simultaneously
# ─────────────────────────────────────────────────────────────────────────
class _Adam:
    def __init__(self, shapes, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.t = 0
        self.m = [np.zeros(s) for s in shapes]
        self.v = [np.zeros(s) for s in shapes]

    def step(self, params, grads):
        self.t += 1
        out = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g ** 2
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            out.append(p - self.lr * mh / (np.sqrt(vh) + self.eps))
        return out


# ─────────────────────────────────────────────────────────────────────────
# Loss and gradient computation  [DAGES forward pass + OURS free-B grad]
# ─────────────────────────────────────────────────────────────────────────
def finsler_loss_grad_freeB(Y, B, D_asym, loss_mask, eps=1e-8):
    """
    Compute L(Y, B) and gradients dL/dY, dL/dB.

    F(i->j) = ||y_j-y_i|| + <b_i, e_ij>     [DAGES Eq. 1]
    Returns: loss (float), grad_Y (n,d), grad_B (n,d)
    """
    diff = Y[np.newaxis, :, :] - Y[:, np.newaxis, :]   # (n,n,d): diff[i,j] = y_j-y_i
    r = np.sqrt((diff ** 2).sum(-1))
    r_safe = np.maximum(r, eps)
    e = diff / r_safe[:, :, np.newaxis]                 # (n,n,d): e_ij
    proj = (B[:, np.newaxis, :] * e).sum(-1)             # (n,n): <b_i, e_ij>
    F_mat = r + proj

    res = (F_mat - D_asym) * loss_mask
    loss = float((res ** 2).sum())

    # grad w.r.t. Y -- node i is SOURCE in (i,j) and TARGET in (j,i)
    perp = B[:, np.newaxis, :] - proj[:, :, np.newaxis] * e

    # [OURS 2026-08-11 perf fix] res.T / e.transpose(1,0,2) / perp.transpose(1,0,2)
    # / r_safe.T are all VIEWS with non-contiguous (transposed) memory layout --
    # elementwise-multiply + .sum(axis=1) on them forces numpy to walk memory in
    # a cache-unfriendly stride pattern every single epoch. On some numpy/BLAS
    # builds (and especially on loaded shared servers where cache/memory
    # bandwidth is already contended) this has been observed to be dramatically
    # slower than the equivalent contiguous computation -- not a small constant
    # factor, but potentially the difference between ~0.1s/epoch and minutes/
    # epoch. Materialising contiguous copies once per epoch is strictly cheap
    # (O(n^2 d), same order as everything else here) and removes the bad
    # access pattern entirely.
    res_T = np.ascontiguousarray(res.T)
    e_T = np.ascontiguousarray(e.transpose(1, 0, 2))
    perp_T = np.ascontiguousarray(perp.transpose(1, 0, 2))
    r_safe_T = np.ascontiguousarray(r_safe.T)

    grad_Y = 2.0 * (res[:, :, np.newaxis] *
                    (-e - perp / r_safe[:, :, np.newaxis])).sum(axis=1)
    grad_Y += 2.0 * (res_T[:, :, np.newaxis] *
                     (e_T + perp_T / r_safe_T[:, :, np.newaxis])).sum(axis=1)

    # grad w.r.t. B  [OURS -- does not exist in frozen-B]
    grad_B = 2.0 * (res[:, :, np.newaxis] * e).sum(axis=1)

    return loss, grad_Y, grad_B


def finsler_distances_freeB(Y: np.ndarray, B: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Reconstruct asymmetric Finsler distances from an (Y,B) embedding."""
    diff = Y[np.newaxis, :, :] - Y[:, np.newaxis, :]
    r_safe = np.maximum(np.sqrt((diff ** 2).sum(-1)), eps)
    e = diff / r_safe[:, :, np.newaxis]
    F = r_safe + (B[:, np.newaxis, :] * e).sum(-1)
    np.fill_diagonal(F, 0.0)
    return F


# ─────────────────────────────────────────────────────────────────────────
# Main training loop: SVD is ONLY the initialisation (Y_init, B_init),
# Adam then jointly updates Y and B every epoch to minimise the Finsler
# reconstruction loss against D_asym.
# ─────────────────────────────────────────────────────────────────────────
def finsler_mds_freeB(
    D_asym: np.ndarray,
    d: int = 2,
    n_epochs: int = 3000,
    lr: float = 5e-3,
    clip_delta: float = 0.01,
    pair_mask: np.ndarray = None,
    max_epochs_no_improvement: int = 300,
    min_epochs: int = 200,
    svd_k: int = 10,
    verbose: bool = True,
    snapshot_every: int = None,
    gravity_strength: float = 0.0,
    node_mass: np.ndarray = None,
    normalize_D: bool = True,
    train_B: bool = False,
) -> dict:
    """
    Finsler MDS, SVD-RADAR init, Adam training.

    Y_init = svd_init(D_asym, k=svd_k)[:, :d]
    B_init = svd_init(D_asym.T - D_asym, k=svd_k)[:, :d], clipped

    train_B : [OURS 2026-08-10] bool. This is the actual point of the
        "attach B to the point at init, then fix it" design we agreed on
        -- default is now False:
          * train_B=False (default): B is computed ONCE from the SVD of
            the skew-symmetric part (Delta = D.T-D) at init, clipped to
            the Randers bound, and then held FIXED for the entire run.
            Only Y is optimised by Adam. B still enters the forward pass
            and Y's gradient every epoch (F=r+proj, proj depends on the
            frozen B) -- it just never gets updated itself.
          * train_B=True: the old free-B behaviour (ported directly from
            asymmetricfinsler-mds_demo/finsler_mds.py::finsler_mds_freeB)
            -- B is ALSO a live Adam parameter, jointly optimised with Y.
            [Note: I had defaulted to this the first time I ported this
            file, which was a mistake -- the plan was always fixed-B,
            same spirit as randers_umap_fit's B_fixed for the g-mentality
            pipeline.]

    This is the difference from svd_radar.py's svd_init() used standalone:
    there, Y and B never see each other after the one-shot SVD (B is just
    drawn on top, doesn't affect Y at all); here (train_B=False), B still
    shapes Y's gradient every epoch through the Finsler forward pass --
    it's attached and load-bearing, just not itself trainable.

    normalize_D : [OURS 2026-08-10] bool. If True (default), D_asym is
        divided ONCE up front by its own std (scale-only, mean NOT
        subtracted) and that single rescaled matrix is used for BOTH the
        SVD init AND the loss target. Previously svd_init() z-scored its
        own copy of D internally (for the SVD step only), while
        finsler_loss_grad_freeB compared F(Y,B) against the RAW,
        un-normalised D_asym -- a scale mismatch (Y_init's natural scale
        vs D_asym's raw scale can differ by 10-30x on this dataset) that
        forced Adam to spend most of its budget on a global rescale
        before it could do anything else, and empirically led the
        *converged* (not just early/partial) solution to fragment the
        manifold into disconnected same-t clusters rather than one
        coherent unrolled sheet -- confirmed reproducible at both
        lr=5e-3 (the default) run to full completion and lr=0.5 run
        briefly. Scale-only (not full z-score) matters: subtracting the
        mean would push near-neighbour pairs (small raw D_asym) to
        strongly negative targets that F=r+proj (r>=0) can never reach,
        which empirically collapsed those pairs to r~0 instead of fixing
        anything.
        result["D_mu"]/result["D_sigma"] hold the removed scale so
        reconstructed distances can be converted back to D_asym's
        original units if needed.

    Returns
    -------
    dict: Y, B, Y_init, B_init, loss_history, b_norms, D_mu, D_sigma, (snapshots)
    """
    D_asym = np.array(D_asym, dtype=np.float64)
    n = D_asym.shape[0]

    mask = pair_mask.astype(np.float64) if pair_mask is not None else 1.0 - np.eye(n)
    M = np.ones(n) if node_mass is None else np.asarray(node_mass, dtype=np.float64)

    # [OURS 2026-08-10] normalise D_asym ONCE, use the SAME normalised
    # matrix for both svd_init (init only) and the loss (training target)
    # -- see normalize_D docstring above.
    #
    # [FIX] scale-only, NOT z-score: subtracting the mean (D-mu)/sigma makes
    # near-neighbour pairs (small raw D_asym, well below the mean) map to
    # strongly NEGATIVE targets -- but F=r+proj can never go below -0.99
    # (r=||y_i-y_j||>=0, proj in [-0.99,0.99]), so those pairs become
    # unreachable and the optimiser collapses them to r~0 instead (observed:
    # dense central clump + a few flung-out points, not a clean unroll).
    # Dividing by a scale factor only (no mean subtraction) keeps D_train
    # >= 0, same "starts at 0" structure as the raw distances, just at a
    # magnitude F=r+proj can actually reach.
    if normalize_D:
        D_mu = 0.0
        D_sigma = float(np.std(D_asym)) + 1e-8
        D_train = D_asym / D_sigma
    else:
        D_mu, D_sigma = 0.0, 1.0
        D_train = D_asym

    # ---- SVD-RADAR initialisation only (not the final embedding) ----
    # normalize=False here -- D_train is already normalised above (or the
    # caller explicitly opted out via normalize_D=False); avoids double
    # z-scoring which svd_init's own normalize=True would otherwise do.
    X_svd = svd_init(D_train, k=svd_k, normalize=False)
    Y = X_svd[:, :d]
    Y_init = Y.copy()

    Delta = D_train.T - D_train                        # skew-symmetric, bounded
    B = svd_init(Delta, k=svd_k, normalize=False)[:, :d]

    limit = 1.0 - clip_delta
    b_norms_init = np.linalg.norm(B, axis=1, keepdims=True)
    B = B * np.where(b_norms_init > limit, limit / np.maximum(b_norms_init, 1e-12), 1.0)
    B_init_saved = B.copy()

    # [OURS 2026-08-10] train_B=False (default): Adam only ever sees Y's
    # shape -- B is never among opt's params, so it physically cannot be
    # updated by opt.step() below. train_B=True: Adam gets both shapes,
    # same as the original free-B loop.
    opt = _Adam([Y.shape, B.shape], lr=lr) if train_B else _Adam([Y.shape], lr=lr)

    best_loss, best_Y, best_B = np.inf, Y.copy(), B.copy()
    no_imp, history = 0, []
    snapshots = []

    if verbose:
        n_pairs = int(mask.sum())
        mode = "free-B (Y,B both trained)" if train_B else "fixed-B (B frozen at SVD init, only Y trained)"
        print(f"\n-- Finsler MDS, SVD-RADAR init, {mode}  (n={n}, d={d}) --")
        print(f"   pair_mask: {n_pairs}/{n*(n-1)} pairs ({n_pairs/n/(n-1)*100:.1f}%)")
        print(f"   lr={lr}  clip_delta={clip_delta}  svd_k={svd_k}  max_epochs={n_epochs}")

    for epoch in range(n_epochs):
        # [OURS 2026-08-11] snapshot BEFORE this epoch's update -- so epoch=0
        # captures the true pre-training state (Y_init/B_init), matching
        # randers_umap_fit's snapshot_every convention (epoch N label =
        # state after N epochs of training have completed).
        if snapshot_every is not None and epoch % snapshot_every == 0:
            snapshots.append({"epoch": epoch, "Y": Y.copy(), "B": B.copy()})

        # B still enters the forward pass (F=r+proj) and grad_Y (via perp)
        # every epoch even when frozen -- gB is simply computed and, if
        # train_B=False, discarded right below instead of applied.
        loss, gY, gB = finsler_loss_grad_freeB(Y, B, D_train, mask)

        if gravity_strength > 0:
            gY = gY - gravity_strength * M[:, np.newaxis] * B

        if train_B:
            [Y, B] = opt.step([Y, B], [gY, gB])
            norms = np.linalg.norm(B, axis=1, keepdims=True)
            B = B * np.where(norms > limit, limit / np.maximum(norms, 1e-12), 1.0)
        else:
            [Y] = opt.step([Y], [gY])
            # B untouched -- stays exactly at B_init_saved for the whole run

        history.append(float(loss))

        if loss < best_loss:
            best_loss = loss
            best_Y, best_B = Y.copy(), B.copy()
            no_imp = 0
        else:
            no_imp += 1

        if verbose and epoch % 200 == 0:
            bn = np.linalg.norm(B, axis=1)
            print(f"    epoch {epoch:4d}  loss={loss:.4f}  "
                  f"||B||_F={np.linalg.norm(B):.3f}  "
                  f"mean||bi||={bn.mean():.3f}  max||bi||={bn.max():.3f}")

        if no_imp >= max_epochs_no_improvement and epoch >= min_epochs:
            if verbose:
                print(f"  Early stop epoch {epoch}  best={best_loss:.4f}")
            break

    # [OURS 2026-08-11] guarantee the final (post-training) state is always
    # captured, even if n_epochs isn't a multiple of snapshot_every or the
    # loop early-stopped -- matches randers_umap_fit's snapshot_every.
    if snapshot_every is not None:
        final_label = epoch + 1
        if not snapshots or snapshots[-1]["epoch"] != final_label:
            snapshots.append({"epoch": final_label, "Y": Y.copy(), "B": B.copy()})

    b_norms = np.linalg.norm(best_B, axis=1)
    if verbose:
        print(f"  Final ||B||_F={np.linalg.norm(best_B):.3f}  "
              f"mean||bi||={b_norms.mean():.3f}  max||bi||={b_norms.max():.3f}  "
              f"invalid={(b_norms>=1).sum()}")

    result = {
        "Y": best_Y,
        "B": best_B,
        "Y_init": Y_init,
        "B_init": B_init_saved,
        "loss_history": history,
        "b_norms": b_norms,
        "D_mu": D_mu,
        "D_sigma": D_sigma,
    }
    if snapshot_every is not None:
        result["snapshots"] = snapshots
    return result
