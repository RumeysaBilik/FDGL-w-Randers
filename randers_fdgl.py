"""
randers_fdgl.py
================
Embeds an asymmetric distance matrix D_asym into 2D/3D. Each point gets
its own drift vector b_i, folded directly into the embedding's own
distance so the asymmetry survives the embedding instead of being
averaged away:
    rho(i->j) = ||y_i-y_j|| + b_i . (y_j-y_i)
The layout mechanics are UMAP's. b_i=0 everywhere recovers
plain UMAP exactly.

SOURCE ATTRIBUTION
══════════════════
[UMAP]   McInnes, Healy, Melville, "UMAP: Uniform Manifold Approximation
         and Projection for Dimension Reduction", arXiv:1802.03426.

[DAGES]  Dagès et al., "Finsler Multi-Dimensional Scaling", CVPR 2025.
"""

import numpy as np
from scipy.optimize import curve_fit


# ─────────────────────────────────────────────────────────────────────────────
# 1. Calibrated k-NN graph construction
# ─────────────────────────────────────────────────────────────────────────────
def _knn_from_distance_matrix(D: np.ndarray, k: int):
    """
    Per-row k nearest neighbours (by D[i,:], excluding self).
    
    Returns
    -------
    knn_idx : (n, k) int   -- neighbour indices, sorted nearest-first
    knn_dist: (n, k) float -- corresponding distances
    """
    n = D.shape[0]
    D_work = D.copy()
    np.fill_diagonal(D_work, np.inf)
    knn_idx = np.argsort(D_work, axis=1)[:, :k]
    knn_dist = np.take_along_axis(D_work, knn_idx, axis=1)
    return knn_idx, knn_dist


def smooth_knn_dist(knn_dist: np.ndarray, k: int, n_iter: int = 64,
                     local_connectivity: float = 1.0, tol: float = 1e-5,
                     min_sigma: float = 1e-3):
    """
    Per-point rho_i (nearest-neighbour distance, local_connectivity=1) and
    sigma_i, binary-search calibrated so
        sum_j exp(-max(0, d_ij - rho_i) / sigma_i) = log2(k)
    -- every node ends up with the same total "soft neighbourhood weight".

    knn_dist : (n, k) sorted-ascending neighbour distances (self excluded)
    """
    n = knn_dist.shape[0]
    target = np.log2(k)
    rho = knn_dist[:, 0].copy()

    sigma = np.zeros(n)
    for i in range(n):
        lo, hi = 0.0, np.inf
        mid = 1.0
        d_i = knn_dist[i]
        for _ in range(n_iter):
            psum = np.exp(-np.maximum(0.0, d_i - rho[i]) / mid).sum()
            if abs(psum - target) < tol:
                break
            if psum > target:
                hi = mid
                mid = (lo + mid) / 2.0
            else:
                lo = mid
                mid = mid * 2.0 if hi == np.inf else (lo + hi) / 2.0
        sigma[i] = max(mid, min_sigma)
    return rho, sigma


def knn_mask_from_distance_matrix(D: np.ndarray, k: int) -> np.ndarray:
    """
    Boolean k-NN mask only -- for callers that don't need _knn_weights'
    full calibrated weight matrix.

    Returns knn_mask : (n, n) bool, i's k nearest neighbours (row-wise).
    """
    n = D.shape[0]
    knn_idx, _ = _knn_from_distance_matrix(D, k)
    knn_mask = np.zeros((n, n), dtype=bool)
    rows = np.repeat(np.arange(n), k)
    knn_mask[rows, knn_idx.ravel()] = True
    return knn_mask


def _knn_weights(D: np.ndarray, k: int):
    """
    UMAP-style calibrated membership weight from a (possibly asymmetric)
    dissimilarity matrix D. Shared by fdgl_low_dim's two call sites
    (A from D_asym, A_geo from D_geo).

    Returns
    -------
    A        : (n, n) raw, directed weighted adjacency (A[i,j] need not
               equal A[j,i]), A_ii = 0.
    knn_mask : (n, n) bool -- i's k nearest neighbours (row-wise)
    """
    n = D.shape[0]
    knn_idx, knn_dist = _knn_from_distance_matrix(D, k)
    rho, sigma = smooth_knn_dist(knn_dist, k)

    A = np.zeros((n, n))
    for i in range(n):
        w = np.exp(-np.maximum(0.0, knn_dist[i] - rho[i]) / sigma[i])
        A[i, knn_idx[i]] = w
    np.fill_diagonal(A, 0.0)

    knn_mask = np.zeros((n, n), dtype=bool)
    rows = np.repeat(np.arange(n), k)
    knn_mask[rows, knn_idx.ravel()] = True

    return A, knn_mask


# ─────────────────────────────────────────────────────────────────────────────
# 2. (a, b) curve fit to min_dist
# ─────────────────────────────────────────────────────────────────────────────
def find_ab_params(spread: float = 1.0, min_dist: float = 0.1):
    """
    Fit a, b in  1 / (1 + a*x^(2b))  to the piecewise target curve
        f(x) = 1                          if x <= min_dist
        f(x) = exp(-(x-min_dist)/spread)  otherwise
    """
    xv = np.linspace(0, spread * 3, 300)
    yv = np.where(xv < min_dist, 1.0, np.exp(-(xv - min_dist) / spread))

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    (a, b), _ = curve_fit(curve, xv, yv, p0=(1.0, 1.0))
    return float(a), float(b)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Classical/Torgerson MDS initialisation
# ─────────────────────────────────────────────────────────────────────────────
def classical_mds(D: np.ndarray, d: int, seed: int = 0) -> np.ndarray:
    """
    Classical/Torgerson MDS: eigendecompose the double-centered squared
    distance matrix, Y = U * sqrt(v). Cheap, unrolls manifolds cleanly.

    """
    n = D.shape[0]
    finite = np.isfinite(D)
    if not finite.all():
        fallback = D[finite].max() if finite.any() else 1.0
        D = np.where(finite, D, fallback)
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    vals, vecs = np.linalg.eigh(B)
    idx = np.argsort(vals)[::-1][:d]
    vals_d = np.clip(vals[idx], 0.0, None)
    Y = vecs[:, idx] * np.sqrt(vals_d)[None, :]

    # tiny jitter to break residual degeneracy
    rng = np.random.default_rng(seed)
    Y = Y + rng.normal(scale=1e-4, size=Y.shape)

    # rescale to extent ~10, matching UMAP's own spectral-init scale
    Y = 10.0 * Y / (np.abs(Y).max() + 1e-12)
    return Y


def arrow_scale(Y: np.ndarray, bn: np.ndarray, frac: float = 0.12) -> float:
    """Multiplier so the longest drift arrow spans `frac` of the embedding's visual extent (0.0 if bn is empty/all-zero)."""
    if Y.size == 0:
        return 0.0
    spans = Y.max(axis=0) - Y.min(axis=0)
    extent = float(spans.max()) if spans.size else 0.0
    bn_max = float(bn.max()) if bn.size else 0.0
    return frac * extent / bn_max if bn_max > 0 else 0.0


def plot_caption(dataset: str, family: str, n: int, k: int, B_fixed: bool,
                  epochs: int = None, init_only: bool = False) -> str:
    """
    Standard two-line title for a trained-embedding plot, shared across every
    run_*.py / embed_*.py driver script so all result plots report the same
    fields in the same order.

    Line 1: "<dataset> (<family>), n=<n>"
    Line 2: "B=frozen|live | epochs=<epochs>|init-only | k=<k> | N=(D-D')/(D+D'+eps)"

    The N formula shown matches _compute_N()'s current, active formula
    above: the bounded [-1,1] ratio (D-D')/(D+D'+eps). Keep this string in
    sync if _compute_N() is ever changed back to the squared-numerator
    variant.
    """
    line1 = f"{dataset} ({family}), n={n}"
    b_str = "live" if not B_fixed else "frozen"
    epoch_str = "init-only" if init_only else f"{epochs}"
    line2 = (f"B={b_str} | epochs={epoch_str} | k={k} | "
             f"N=(D-D')/(D+D'+ε)")
    return line1 + "\n" + line2


# ─────────────────────────────────────────────────────────────────────────────
# 4. Drift vector: recomputed from the live embedding every epoch
# ─────────────────────────────────────────────────────────────────────────────
def _compute_N(D_asym: np.ndarray, knn_mask: np.ndarray = None) -> np.ndarray:
    """
    Per-pair asymmetry signal feeding compute_drift(), bounded in [-1,1]:
        N[i,j] = (D[i,j] - D[j,i]) / (D[i,j] + D[j,i] + eps)
    Fixed, computed once from D_asym (not re-derived per epoch).

    Fill value for a missing (inf) directed entry D_asym[i,j]: node i's own
    farthest finite k-NN neighbour distance (row i's max over knn_mask[i,:]
    & finite), NOT a single global diameter shared by the whole matrix.
    Per-node local scale, since a global diameter fill (the old behaviour)
    can be orders of magnitude larger than a node's typical neighbour
    distance, letting one missing pair dominate compute_drift's per-node
    average and saturate its clip. Falls back to the global finite max for
    any row with no finite knn_mask entry, or if knn_mask isn't given.
    """
    n = D_asym.shape[0]
    finite = np.isfinite(D_asym)
    global_fallback = D_asym[finite].max() if finite.any() else 1.0

    if knn_mask is not None:
        local_finite = finite & knn_mask
        has_local = local_finite.any(axis=1)
        masked = np.where(local_finite, D_asym, -np.inf)
        row_max = masked.max(axis=1)
        row_max = np.where(has_local, row_max, global_fallback)
    else:
        row_max = np.full(n, global_fallback)

    row_fill = np.where(row_max > 0, row_max * (1.0 + 1e-6), 1e-6)
    fill_matrix = np.broadcast_to(row_fill[:, np.newaxis], D_asym.shape)
    D_filled = np.where(finite, D_asym, fill_matrix)

    N = (D_filled - D_filled.T) / (D_filled + D_filled.T + 1e-12)
    both_missing = ~finite & ~finite.T
    return np.where(both_missing, 0.0, N)


def compute_drift(N: np.ndarray, knn_mask: np.ndarray, k: int,
                   Y: np.ndarray, clip_delta: float = 0.01,
                   magnitude_target: np.ndarray = None,
                   eps: float = 1e-8) -> np.ndarray:
    """
    b_i = (1/k) * sum_{j in N_k(i)} N[i,j] * e_ij(Y)

    N        : (n,n) the per-pair asymmetry signal from _compute_N(D_asym),
               fixed, computed once outside
    knn_mask : (n,n) bool, i's k nearest neighbours (fixed, from D_asym)
    Y        : (n,d) CURRENT embedding -- this is what makes b_i "live"

    magnitude_target : (n,) array or None. If given (typically
        asymmetry_score()'s fixed, Y-independent per-node output), b_i's
        magnitude is replaced with it every call while direction stays
        live -- avoids the live-alignment magnitude self-reinforcing as
        training reshapes Y. None (default) is a no-op.
    """
    diff = Y[np.newaxis, :, :] - Y[:, np.newaxis, :]        # (n,n,d) y_j-y_i
    r = np.maximum(np.sqrt((diff ** 2).sum(-1)), eps)
    e = diff / r[:, :, np.newaxis]                           # e_ij(Y)

    weight = np.where(knn_mask, N, 0.0)                      # (n,n)
    b = (1.0 / k) * (weight[:, :, np.newaxis] * e).sum(axis=1)   # (n,d)

    limit = 1.0 - clip_delta
    norms = np.linalg.norm(b, axis=1, keepdims=True)
    b = b * np.where(norms > limit, limit / np.maximum(norms, 1e-12), 1.0)

    if magnitude_target is not None:
        bnorm = np.linalg.norm(b, axis=1, keepdims=True)
        direction = b / np.maximum(bnorm, eps)
        b = magnitude_target[:, np.newaxis] * direction
    return b


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main fit: force-directed layout with the drift folded into rho
# ─────────────────────────────────────────────────────────────────────────────
def fdgl_low_dim(
    D_asym: np.ndarray,
    d: int = 2,
    n_neighbors: int = 20,
    n_epochs: int = 500,
    lr: float = 1.0,
    min_dist: float = 0.1,
    spread: float = 1.0,
    use_drift: bool = True,
    clip_delta: float = 0.01,
    grad_clip: float = 4.0,
    n_negative_samples: int = 10,
    B_fixed: np.ndarray = None,
    Y_init_override: np.ndarray = None,
    use_gravity: bool = False,
    node_mass: np.ndarray = None,
    gravity_strength: float = 1.0,
    gravity_neighbor_weight: bool = True,
    randers_attractive: bool = True,
    randers_repulsive: bool = False,
    D_geo: np.ndarray = None,
    randers_gravity: bool = False,
    drift_magnitude_target: np.ndarray = None,
    scale_B_fixed_by_knn_distance: bool = False,
    use_virtual_neighbor: bool = False,
    ramp: bool = True,
    snapshot_every: int = None,
    seed: int = 0,
    verbose: bool = True,
    force_model: str = "fr_gravity",
    fr_k: float = None,
    negative_sampling: bool = False,
) -> dict:
    """
    Part A:

        rho(i->j) = ||y_i-y_j|| + b_i . (y_j-y_i)
        grad_{y_i} rho = -(e_ij + b_i) =: -g_ij


    force_model : 
        "fr_gravity":
            f_r(i,j) = k^2/rho(i,j)^2 * (-g(i,j))   [repulsion, ALL pairs]
            f_a(i,j) = rho(i,j)/k     * ( g(i,j))   [attraction, EDGES only]
        "umap":
            attr_coeff(rho) = 2*a*b*rho^(2b-1) / (1+a*rho^(2b))
            rep_coeff(rho)  = 2*b*rho / (rho^2*(1+a*rho^(2b)))

    fr_k : float or None. Only for force_model="fr_gravity" -- the
        "natural edge length" k in the formula above. None uses
        sqrt(1/n); override to tune the attraction/repulsion balance.

    negative_sampling : bool, default False. If True, repulsion is
        estimated from n_negative_samples random points per node per
        epoch instead of the full dense sum 

    B_fixed : (n,d) array, or None. If given, b_i is frozen at this value
        for the whole run

    Y_init_override : (n,d) array, or None. Used instead of
        classical_mds() as the starting position if given.

    use_gravity : bool, default False. Additive force pulling each
        node i toward xi_i = y_i + b_i (social-gravity style):
            f_g(v) = gamma_t * M[v] * (xi - P[v])  =  gamma * M[i] * b_i
        gravity_strength is gamma_t; gravity_neighbor_weight scales the
        pull by plausibility.

    gravity_strength : float, default 1.0. Multiplier gamma_t above.
        Only active when use_gravity=True.

    gravity_neighbor_weight : bool, default True. If True, scales gravity
        on node i by w_i = exp(-max(0, ||b_i||-rho_i)/sigma_i), a
        smooth_knn_dist-style plausibility weight computed live from Y
        each epoch (cheap mean-based sigma, not full calibration). If
        False, w_i=1 always.

    node_mass : (n,) per-node mass M[i] for gravity. None = uniform (1).

    randers_attractive, randers_repulsive : bool, default True/False.
        Independent switches for whether the main force's attraction and
        repulsion terms are evaluated at the Randers-perturbed (rho_r,
        g_r) or plain Euclidean (rho_e, g_e) pair.

    D_geo : (n,n) ndarray or None. Drift-free/Euclidean counterpart of
        D_asym (same graph, no Randers perturbation) -- only used when
        randers_attractive or randers_repulsive is False.

    randers_gravity : bool, default False. If True, gravity's pseudo-edge
        (y_i, xi_i=y_i+b_i) is run through the same rho/g/attr_coeff
        construction as the main force (pure attraction, mu=1) instead of
        a plain linear pull. False: gravity_term_i = b_i (linear).

    drift_magnitude_target : (n,) array or None. If given, b_i's
        magnitude is replaced with this value every epoch while direction
        stays live -- see compute_drift(). Typically
        asymmetry_score()'s fixed per-node output, avoiding the
        self-reinforcing live-alignment magnitude. Ignored if B_fixed set.

    scale_B_fixed_by_knn_distance : bool, default False. Only with
        B_fixed: keeps its direction frozen but replaces magnitude each
        epoch with node i's live k-th-nearest-neighbour distance in Y, so
        drift length tracks local scale instead of staying fixed.

    use_virtual_neighbor : bool, default False. Independent of
        use_gravity. Each node's xi_i=y_i+b_i is an unconditional
        (k+1)-th attractive neighbour (mu=1), using UMAP's own attraction
        curve at rho_v=||b_i||:
            attr_coeff_v = 2*a*b*rho_v^(2b-1) / (1+a*rho_v^(2b))
            force_virtual_i = attr_coeff_v * b_i

    Drift: b_i = (1/k) sum over j in N_k(i) -- see compute_drift().

    n_negative_samples : the dense repulsion sum is rescaled by
        n_negative_samples/(n-1-k).

    Returns
    -------
    dict: Y, Y_init, B (final drift), mu / mu_directed (directed
          membership used by the force computation), knn_mask, a,
          b_param, history (mean |rho-d| per 100 epochs)
    """
    D_asym = np.asarray(D_asym, dtype=np.float64)
    n = D_asym.shape[0]
    rng = np.random.default_rng(seed)

    # A = directed membership used by the force computation's weighting.
    A, knn_mask = _knn_weights(D_asym, n_neighbors)

    # A_geo is the Euclidean-weight counterpart for a term whose randers_*
    A_geo = None
    if not (randers_attractive and randers_repulsive):
        if D_geo is not None:
            D_geo = np.asarray(D_geo, dtype=np.float64)
            A_geo, _ = _knn_weights(D_geo, n_neighbors)
        else:
            A_geo = A  
            if verbose:
                print("WARNING: randers_attractive/randers_repulsive requests a "
                      "Euclidean-weighted term but D_geo was not supplied -- "
                      "falling back to the Randers-derived A as its weight too "
                      "(drift may leak into the 'Euclidean' term's weight).")
    A_attr = A if randers_attractive else A_geo
    A_rep = A if randers_repulsive else A_geo

    a, b_param = find_ab_params(spread=spread, min_dist=min_dist)
    N = _compute_N(D_asym, knn_mask=knn_mask)

    # force_model="fr_gravity" setup -- fixed edge set/k, computed once.

    if force_model not in ("umap", "fr_gravity"):
        raise ValueError("force_model must be 'umap' or 'fr_gravity'")
    if force_model == "fr_gravity":
        A_edges = (knn_mask | knn_mask.T).astype(np.float64)
        np.fill_diagonal(A_edges, 0.0)
        k_nat = float(fr_k) if fr_k is not None else 1.0 / np.sqrt(n)

    if Y_init_override is not None:
        Y = np.asarray(Y_init_override, dtype=np.float64).copy()
    else:
        Y = classical_mds(D_asym, d, seed=seed)  # inf-safe internally
    Y_init = Y.copy()

    B = np.zeros((n, d)) if B_fixed is None else np.asarray(B_fixed, dtype=np.float64).copy()
    M = np.ones(n) if node_mass is None else np.asarray(node_mass, dtype=np.float64)
    eps = 1e-8

    # B_fixed's unit direction, precomputed once -- only magnitude gets
    # replaced per-epoch when scale_B_fixed_by_knn_distance=True.
    B_fixed_direction = None
    if B_fixed is not None and scale_B_fixed_by_knn_distance:
        B_fixed_arr = np.asarray(B_fixed, dtype=np.float64)
        bn_fixed = np.linalg.norm(B_fixed_arr, axis=1, keepdims=True)
        B_fixed_direction = B_fixed_arr / np.maximum(bn_fixed, eps)

    # Snapshot capture
    snapshots = []
    if snapshot_every is not None:
        if B_fixed is None and use_drift:
            s0 = 0.0 if ramp else 1.0
            B_snap0 = s0 * compute_drift(N, knn_mask, n_neighbors, Y_init, clip_delta,
                                         magnitude_target=drift_magnitude_target)
        elif B_fixed is not None and scale_B_fixed_by_knn_distance:
            # k-th-nn-distance-scaled magnitude at Y_init, same as below.
            s0 = 0.0 if ramp else 1.0
            k_local0 = min(n_neighbors, n - 1)
            diff0 = Y_init[np.newaxis, :, :] - Y_init[:, np.newaxis, :]
            d_mat0 = np.maximum(np.sqrt((diff0 ** 2).sum(-1)), eps)
            d_self_excl0 = d_mat0.copy()
            np.fill_diagonal(d_self_excl0, np.inf)
            nearest_k0 = np.partition(d_self_excl0, k_local0 - 1, axis=1)[:, :k_local0]
            kth_dist0 = nearest_k0.max(axis=1)
            B_snap0 = s0 * (bn_fixed / np.maximum(kth_dist0[:, np.newaxis], eps)) * B_fixed_direction
        else:
            B_snap0 = B.copy()
        snapshots.append({"epoch": 0, "Y": Y_init.copy(), "B": B_snap0})
    history = []

    # rescale dense repulsion sum
    n_non_neighbors = max(n - 1 - n_neighbors, 1)
    rep_scale = n_negative_samples / n_non_neighbors

    if verbose:
        mode_str = "B_fixed (frozen)" if B_fixed is not None else f"use_drift={use_drift}"
        grav_str = (f"  +gravity(gamma={gravity_strength}, "
                    f"nbr_weight={gravity_neighbor_weight}, "
                    f"randers={randers_gravity})" if use_gravity else "")
        vn_str = "  +virtual_neighbor(k+1, UMAP curve)" if use_virtual_neighbor else ""
        rf_str = "" if (randers_attractive and randers_repulsive) else \
            f"  [main force: attr={'randers' if randers_attractive else 'euclid'}, " \
            f"rep={'randers' if randers_repulsive else 'euclid'}]"
        print(f"\n-- Randers Force-Directed Layout (Part A)  n={n} d={d} k={n_neighbors}  "
              f"force_model={force_model}  {mode_str}{grav_str}{vn_str}{rf_str} --")
        if force_model == "umap":
            print(f"   a={a:.4f} b={b_param:.4f}  (min_dist={min_dist}, spread={spread})")
        else:
            print(f"   fr_k={k_nat:.4f}  (n_edges={int(A_edges.sum())//2})")

    for epoch in range(n_epochs):
        # ramp: drift off for first 30% of epochs, linear 0->1 over next
        # 40%, full strength for last 30%. s=1.0 always if ramp=False.
        if ramp:
            t_prog = epoch / max(n_epochs - 1, 1)
            s = 0.0 if t_prog <= 0.30 else (1.0 if t_prog >= 0.70 else (t_prog - 0.30) / 0.40)
        else:
            s = 1.0

        diff = Y[np.newaxis, :, :] - Y[:, np.newaxis, :]     # (n,n,d) y_j-y_i
        d_mat = np.maximum(np.sqrt((diff ** 2).sum(-1)), eps)
        e = diff / d_mat[:, :, np.newaxis]

        if B_fixed is not None:
            if scale_B_fixed_by_knn_distance:
                # live k-th-nearest-neighbour distance
                k_local = min(n_neighbors, n - 1)
                d_self_excl = d_mat.copy()
                np.fill_diagonal(d_self_excl, np.inf)
                nearest_k = np.partition(d_self_excl, k_local - 1, axis=1)[:, :k_local]
                kth_dist = nearest_k.max(axis=1)
                # magnitude = ||B_fixed|| / kth_dist, direction untouched;
                # bounded by 1-clip_delta once kth_dist >= 1
                B = s * (bn_fixed / np.maximum(kth_dist[:, np.newaxis], eps)) * B_fixed_direction
            else:
                B = s * np.asarray(B_fixed, dtype=np.float64)
        elif use_drift:
            B = s * compute_drift(N, knn_mask, n_neighbors, Y, clip_delta,
                                          magnitude_target=drift_magnitude_target)
        # force calculation
        raw_dot = (B[:, np.newaxis, :] * diff).sum(-1)        # b_i . (y_j-y_i)
        rho_r = np.maximum(d_mat + raw_dot, eps)               # Randers rho
        g_r = e + B[:, np.newaxis, :]                          # Randers g
        rho_e = d_mat                                          # vanilla UMAP
        g_e = e

        rho_attr, g_attr = (rho_r, g_r) if randers_attractive else (rho_e, g_e)
        rho_rep, g_rep = (rho_r, g_r) if randers_repulsive else (rho_e, g_e)

        rho = rho_r 

        if force_model == "umap":
            rho2b_attr = rho_attr ** (2 * b_param)
            attr_coeff = (2 * a * b_param * rho_attr ** (2 * b_param - 1)) / (1.0 + a * rho2b_attr)

            if negative_sampling:
                # repulsion handled by the sampled block after
                # step=force.sum(axis=1) below -- attraction only here.
                force = (A_attr * attr_coeff)[:, :, np.newaxis] * g_attr
            else:
                rho2b_rep = rho_rep ** (2 * b_param)
                rep_coeff = (2 * b_param * rho_rep) / ((eps + rho_rep ** 2) * (1.0 + a * rho2b_rep))

                force = (A_attr * attr_coeff)[:, :, np.newaxis] * g_attr \
                        - rep_scale * ((1.0 - A_rep) * rep_coeff)[:, :, np.newaxis] * g_rep
        else:  # force_model == "fr_gravity"
            attr_coeff_fr = rho_attr / k_nat

            if negative_sampling:
                # repulsion handled by the sampled block after
                # step=force.sum(axis=1) below -- attraction only here.
                force = (A_edges * attr_coeff_fr)[:, :, np.newaxis] * g_attr
            else:
                rep_coeff_fr = (k_nat * k_nat) / (eps + rho_rep ** 2)
                force = (A_edges * attr_coeff_fr)[:, :, np.newaxis] * g_attr \
                        - rep_coeff_fr[:, :, np.newaxis] * g_rep
        np.fill_diagonal(force[:, :, 0], 0.0)
        if d > 1:
            for dd in range(1, d):
                np.fill_diagonal(force[:, :, dd], 0.0)

        step = force.sum(axis=1)                              # (n,d) net force on y_i

        # True stochastic negative sampling: draws n_negative_samples
        # random points per node, fresh every epoch, additive to `step`.
        if negative_sampling:
            neg_idx = rng.integers(0, n, size=(n, n_negative_samples))
            self_hit = neg_idx == np.arange(n)[:, np.newaxis]
            neg_idx = np.where(self_hit, (neg_idx + 1) % n, neg_idx)   # deflect self-samples

            Y_neg = Y[neg_idx]                                          # (n, S, d)
            diff_neg = Y_neg - Y[:, np.newaxis, :]                       # y_neg - y_i
            d_neg = np.maximum(np.sqrt((diff_neg ** 2).sum(-1)), eps)    # (n, S)
            e_neg = diff_neg / d_neg[:, :, np.newaxis]

            if randers_repulsive:
                raw_dot_neg = (B[:, np.newaxis, :] * diff_neg).sum(-1)
                rho_neg = np.maximum(d_neg + raw_dot_neg, eps)
                g_neg = e_neg + B[:, np.newaxis, :]
            else:
                rho_neg = d_neg
                g_neg = e_neg

            if force_model == "umap":
                rho2b_neg = rho_neg ** (2 * b_param)
                rep_coeff_neg = (2 * b_param * rho_neg) / ((eps + rho_neg ** 2) * (1.0 + a * rho2b_neg))
                step = step - (rep_coeff_neg[:, :, np.newaxis] * g_neg).sum(axis=1)
            else:  # force_model == "fr_gravity"
                rep_coeff_neg_fr = (k_nat * k_nat) / (eps + rho_neg ** 2)
                fr_neg_scale = (n - 1) / n_negative_samples
                step = step - fr_neg_scale * (rep_coeff_neg_fr[:, :, np.newaxis] * g_neg).sum(axis=1)

        # Per-node gravity -- separate additive force pulling y_i toward
        # xi_i = y_i + b_i, i.e. gamma * M[i] * b_i. 
        if use_gravity:
            if gravity_neighbor_weight:
                k_local = min(n_neighbors, n - 1)
                d_self_excl = d_mat.copy()
                np.fill_diagonal(d_self_excl, np.inf)
                nearest_k = np.partition(d_self_excl, k_local - 1, axis=1)[:, :k_local]
                rho_local = nearest_k.min(axis=1)
                sigma_local = np.maximum(nearest_k.mean(axis=1) - rho_local, 1e-6)
                bnorm = np.linalg.norm(B, axis=1)
                w_grav = np.exp(-np.maximum(0.0, bnorm - rho_local) / sigma_local)
            else:
                w_grav = np.ones(n)

            if randers_gravity:
                # gravity's pseudo-edge (i, xi_i=y_i+b_i) through the same
                # rho/g/attr_coeff construction as the main force
                bnorm_g = np.linalg.norm(B, axis=1)
                d_v = np.maximum(bnorm_g, eps)
                e_v = B / d_v[:, np.newaxis]
                raw_dot_v = bnorm_g ** 2                      # b_i . diff_v = b_i.b_i
                rho_v = np.maximum(d_v + raw_dot_v, eps)      # same form as rho above
                g_v = e_v + B                                  # same form as g above
                rho2b_v = rho_v ** (2 * b_param)
                attr_coeff_v_g = (2 * a * b_param * rho_v ** (2 * b_param - 1)) / (1.0 + a * rho2b_v)
                gravity_term = attr_coeff_v_g[:, np.newaxis] * g_v
            else:
                gravity_term = B  # plain linear pull along b_i

            step = step + gravity_strength * w_grav[:, np.newaxis] * M[:, np.newaxis] * gravity_term

        # Virtual-neighbor: xi_i=y_i+b_i as an unconditional (k+1)-th attractive neighbour
        if use_virtual_neighbor:
            bnorm_v = np.linalg.norm(B, axis=1)
            rho_v = np.maximum(bnorm_v, eps)
            attr_coeff_v = (2 * a * b_param * rho_v ** (2 * b_param - 1)) / (1.0 + a * rho_v ** (2 * b_param))
            step = step + attr_coeff_v[:, np.newaxis] * B

        step = np.clip(step, -grad_clip, grad_clip)

        # decaying learning rate, mirrors UMAP's "slowly decreasing forces"
        cur_lr = lr * (1.0 - epoch / n_epochs)
        Y = Y + cur_lr * step

        if snapshot_every is not None and (epoch + 1) % snapshot_every == 0:
            snapshots.append({"epoch": epoch + 1, "Y": Y.copy(), "B": B.copy()})

        if verbose and epoch % 100 == 0:
            resid = float(np.abs(rho - d_mat).mean())
            history.append(resid)
            bn = np.linalg.norm(B, axis=1)
            print(f"    epoch {epoch:4d}  mean|rho-d|={resid:.4f}  "
                  f"mean||b_i||={bn.mean():.4f}  max||b_i||={bn.max():.4f}  "
                  f"lr={cur_lr:.4f}")

    if B_fixed is None and use_drift:
        B = compute_drift(N, knn_mask, n_neighbors, Y, clip_delta,
                                  magnitude_target=drift_magnitude_target)
    elif B_fixed is not None and scale_B_fixed_by_knn_distance:
        # recompute against the true final Y so (Y, B) stays consistent
        # (the in-loop B lags one step behind the returned Y otherwise)
        k_local = min(n_neighbors, n - 1)
        d_mat_final = np.maximum(np.sqrt(((Y[np.newaxis, :, :] - Y[:, np.newaxis, :]) ** 2).sum(-1)), eps)
        d_self_excl = d_mat_final.copy()
        np.fill_diagonal(d_self_excl, np.inf)
        nearest_k = np.partition(d_self_excl, k_local - 1, axis=1)[:, :k_local]
        kth_dist = nearest_k.max(axis=1)
        B = kth_dist[:, np.newaxis] * B_fixed_direction

    if snapshot_every is not None and snapshots[-1]["epoch"] != n_epochs:
        snapshots.append({"epoch": n_epochs, "Y": Y.copy(), "B": B.copy()})

    result = {
        # "mu"/"mu_directed": same array (A); "mu" kept for old callers
        "Y": Y, "Y_init": Y_init, "B": B, "mu": A, "mu_directed": A,
        "knn_mask": knn_mask,
        "N": N, "a": a, "b_param": b_param, "history": history,
    }
    if snapshot_every is not None:
        result["snapshots"] = snapshots
    return result
