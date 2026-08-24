"""
randers_umap.py
================
Part A of the "drift inside UMAP" direction: the Randers drift vector
lives INSIDE the UMAP low-dimensional distance itself, rather than being
a separate additive force. Replaces every previous B-update method in
this project (free-B/Adam+MSE, Option A frozen-direction, per-node
gravity) with a UMAP-style force-directed graph layout instead.

SOURCE ATTRIBUTION
══════════════════
[UMAP]   McInnes, Healy, Melville, "UMAP: Uniform Manifold Approximation
         and Projection for Dimension Reduction", arXiv:1802.03426.
         Fuzzy simplicial set graph construction (smooth_knn_dist,
         rho_i/sigma_i calibration), spectral initialisation, the
         (a,b) curve fit to min_dist, and the attractive/repulsive
         force-directed layout with negative sampling.

[DAGES]  Dagès et al., "Finsler Multi-Dimensional Scaling", CVPR 2025.
         The Randers metric F(x,v) = ||v|| + omega_x^T v this whole
         construction is trying to embed a discrete, per-node version of.

[OURS]   The actual new idea in this file, matching the working report
         "Asymmetric Embedding: Two Placements of the Drift Vector",
         Part A:
           - the input "distance" fed to smooth_knn_dist/fuzzy graph
             construction is our own D_asym (log-ratio + ISUMAP row-
             normalised migration distance), not a raw Euclidean k-NN
             distance -- so the graph already encodes the asymmetry
             before any drift is added.
           - the drift vector is folded directly into UMAP's own
             low-dimensional distance:
                 rho(i->j) = ||y_i-y_j|| + b_i . (y_j-y_i)
             instead of being a separate, additively bolted-on term.
             This is the RAW/unnormalised Randers form (b_i . (y_j-y_i),
             not the unit-vector form b_i . e_ij used elsewhere in this
             project) -- see grad rho below.
           - setting b_i = 0 identically reproduces vanilla UMAP (no
             drift term at all) -- used below as a sanity check.
           - DENSE full-batch adaptation: real UMAP uses per-edge SGD
             with negative sampling because its graphs are sparse and
             huge (thousands-millions of points). Our n=232 is small
             enough that computing every pair's attractive/repulsive
             contribution densely (as finsler_mds.py already does) is
             both tractable and more stable than stochastic negative
             sampling, so that is what this file does instead --
             weighted by mu_ij (attractive) and (1-mu_ij) (repulsive)
             for EVERY pair, not just a random subsample. This is a
             full-batch version of exactly the same forces, not a
             different method.

Randers validity: ||b_i|| <= 1 - clip_delta, same convention as
finsler_mds.py [DAGES].
"""

import numpy as np
from scipy.optimize import curve_fit


# ─────────────────────────────────────────────────────────────────────────────
# 1. Fuzzy simplicial set graph construction  [UMAP Section 3.1]
# ─────────────────────────────────────────────────────────────────────────────
def _knn_from_distance_matrix(D: np.ndarray, k: int):
    """
    Per-row k nearest neighbours (by D[i,:], excluding self).
    D need not be symmetric -- each row uses its own ordering, exactly
    as UMAP's own k-NN step would if D were the true input metric.

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
    [UMAP Section 3.1] Per-point rho_i, sigma_i.

    rho_i = distance to the nearest neighbour (local_connectivity=1):
        ensures node i connects to at least one neighbour with weight 1,
        regardless of local density -- the local-connectivity guarantee.

    sigma_i: binary-search calibrated so that
        sum_j exp(-max(0, d_ij - rho_i) / sigma_i) = log2(k)
        i.e. every node's total "soft neighbourhood weight" is the same,
        which is how UMAP realises the "data is uniformly distributed
        w.r.t. its own local Riemannian metric" assumption.

    Parameters
    ----------
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


def fuzzy_simplicial_set(D: np.ndarray, k: int):
    """
    [UMAP Section 3.1] Build the symmetric fuzzy graph mu (n,n) from an
    input dissimilarity matrix D (may be asymmetric -- our D_asym).

    Returns
    -------
    mu       : (n, n) symmetric weighted adjacency, mu_ii = 0
    knn_mask : (n, n) bool -- i's k nearest neighbours (row-wise, from D)
               used later to restrict the drift sum to N_k(i)
    """
    n = D.shape[0]
    knn_idx, knn_dist = _knn_from_distance_matrix(D, k)
    rho, sigma = smooth_knn_dist(knn_dist, k)

    A = np.zeros((n, n))
    for i in range(n):
        w = np.exp(-np.maximum(0.0, knn_dist[i] - rho[i]) / sigma[i])
        A[i, knn_idx[i]] = w

    # [OURS 2026-08-14] re-enabled -- this was commented out, leaving mu as
    # the raw asymmetric row-wise adjacency A. That silently broke
    # spectral_layout(): np.linalg.eigh() only reads the lower triangle of
    # its input by default, so an asymmetric mu fed into the normalised
    # Laplacian there was throwing away half the graph's information,
    # producing a degenerate (collapsed-onto-two-lines) spectral init
    # instead of a meaningful one. Confirmed side-by-side against real
    # umap-learn's own fuzzy_simplicial_set+spectral_layout on identical
    # input: real graph was symmetric, ours wasn't, and only the symmetric
    # one gave a coherent swiss-roll-shaped init at epoch 0.
    mu = A + A.T - A * A.T   # probabilistic t-conorm  [UMAP Eq. Section 3.1]
    np.fill_diagonal(mu, 0.0)

    knn_mask = np.zeros((n, n), dtype=bool)
    rows = np.repeat(np.arange(n), k)
    knn_mask[rows, knn_idx.ravel()] = True

    return mu, knn_mask


# ─────────────────────────────────────────────────────────────────────────────
# 2. (a, b) curve fit to min_dist  [UMAP Section 3.2 / Definition 11]
# ─────────────────────────────────────────────────────────────────────────────
def find_ab_params(spread: float = 1.0, min_dist: float = 0.1):
    """
    Fit a, b in  1 / (1 + a*x^(2b))  to the piecewise target curve
        f(x) = 1                          if x <= min_dist
        f(x) = exp(-(x-min_dist)/spread)  otherwise
    [UMAP Definition 11 / Section 3.2 -- "a and b are hyper-parameters"]
    """
    xv = np.linspace(0, spread * 3, 300)
    yv = np.where(xv < min_dist, 1.0, np.exp(-(xv - min_dist) / spread))

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    (a, b), _ = curve_fit(curve, xv, yv, p0=(1.0, 1.0))
    return float(a), float(b)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Spectral initialisation  [UMAP Section 3.2, "spectral layout"]
# ─────────────────────────────────────────────────────────────────────────────
def spectral_layout(mu: np.ndarray, d: int, seed: int = 0) -> np.ndarray:
    """
    Graph-Laplacian spectral embedding of mu, used as the SGD starting
    point instead of random init [UMAP: "faster convergence and greater
    stability"]. n=232 is small enough for a dense eigh.
    """
    n = mu.shape[0]
    deg = mu.sum(axis=1)
    deg_safe = np.maximum(deg, 1e-12)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg_safe))
    L_sym = np.eye(n) - D_inv_sqrt @ mu @ D_inv_sqrt   # normalised Laplacian

    vals, vecs = np.linalg.eigh(L_sym)
    # skip the trivial (near-zero) eigenvalue/eigenvector
    idx = np.argsort(vals)[1:d + 1]
    Y = vecs[:, idx].astype(np.float64)

    # tiny random jitter to break any residual degeneracy, matches UMAP's
    # own practice of adding small noise to the spectral embedding
    rng = np.random.default_rng(seed)
    Y = Y + rng.normal(scale=1e-4, size=Y.shape)

    # rescale to a sane initial extent (UMAP scales spectral init to ~10)
    Y = 10.0 * Y / (np.abs(Y).max() + 1e-12)
    return Y


# ─────────────────────────────────────────────────────────────────────────────
# 3b. Classical/Torgerson MDS initialisation  [Isomap's own finishing step]
# ─────────────────────────────────────────────────────────────────────────────
def classical_mds(D: np.ndarray, d: int, seed: int = 0) -> np.ndarray:
    """
    Classical/Torgerson MDS:
    eigendecompose the double-centered squared distance matrix,
    Y = U * sqrt(v). This is Isomap's OWN finishing step (confirmed via the
    IsUMap paper's own words this session: "when we forgo local
    modifications... our approach essentially becomes an implementation of
    the Isomap algorithm" -- and sklearn's Isomap._fit_transform: k-NN +
    Dijkstra geodesics, then KernelPCA(kernel="precomputed") on
    -0.5*dist_matrix**2, which is exactly this).

    Alternative to spectral_layout() for run_located_drift's locate step
    (--init-method isomap): D is already built Isomap-style (k-NN +
    Dijkstra via randers_bridge.compute_dist_matrix, fully dense/complete
    -- no sparse/inf issue like isumap's data_D), so this makes the WHOLE
    pipeline consistently Isomap (distances AND embedding), not just the
    distance construction. Motivated by this session's empirical finding
    that spectral/force-directed init tends to fragment this kind of
    geodesic-distance data, while Isomap-style classical MDS is
    specifically known (Tenenbaum et al. 2000) to unroll a swiss roll
    cleanly, and DAGES's own Isomap init (checked directly this session)
    was clean/non-fragmented.

    Requires D to be dense/complete -- unlike spectral_layout, which
    tolerates a sparse fuzzy graph, classical MDS's double-centering sums
    whole rows/columns, so missing/inf entries would contaminate every
    row/column.
    """
    n = D.shape[0]
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    vals, vecs = np.linalg.eigh(B)
    idx = np.argsort(vals)[::-1][:d]
    vals_d = np.clip(vals[idx], 0.0, None)
    Y = vecs[:, idx] * np.sqrt(vals_d)[None, :]

    # match spectral_layout's scale convention (UMAP scales its own init to
    # ~10) so downstream min_dist/spread-tuned force-directed training sees
    # a comparable starting extent regardless of which init method was used
    Y = 10.0 * Y / (np.abs(Y).max() + 1e-12)
    return Y


# ─────────────────────────────────────────────────────────────────────────────
# 4. Drift vector: recomputed from the live embedding every epoch  [OURS]
# ─────────────────────────────────────────────────────────────────────────────
def compute_drift(N: np.ndarray, knn_mask: np.ndarray, k: int,
                   Y: np.ndarray, clip_delta: float = 0.01,
                   eps: float = 1e-8) -> np.ndarray:
    """
    b_i = (1/k) * sum_{j in N_k(i)} N[i,j] * e_ij(Y)

    N        : (n,n) = 1/2 (D_asym - D_asym^T), fixed, computed once outside
    knn_mask : (n,n) bool, i's k nearest neighbours (fixed, from D_asym)
    Y        : (n,d) CURRENT embedding -- this is what makes b_i "live"
    """
    diff = Y[np.newaxis, :, :] - Y[:, np.newaxis, :]        # (n,n,d) y_j-y_i
    r = np.maximum(np.sqrt((diff ** 2).sum(-1)), eps)
    e = diff / r[:, :, np.newaxis]                           # e_ij(Y)

    weight = np.where(knn_mask, N, 0.0)                      # (n,n)
    b = (1.0 / k) * (weight[:, :, np.newaxis] * e).sum(axis=1)   # (n,d)

    limit = 1.0 - clip_delta
    norms = np.linalg.norm(b, axis=1, keepdims=True)
    b = b * np.where(norms > limit, limit / np.maximum(norms, 1e-12), 1.0)
    return b


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main fit: force-directed layout with the drift folded into rho
#    [UMAP Section 3.2 forces, OURS: rho/g_ij replace UMAP's plain d/e_ij]
# ─────────────────────────────────────────────────────────────────────────────
def randers_umap_fit(
    D_asym: np.ndarray,
    d: int = 2,
    n_neighbors: int = 20,
    n_epochs: int = 500,
    lr: float = 1.0,
    min_dist: float = 0.1,
    spread: float = 1.0,
    use_drift: bool = True,
    drift_mode: str = "knn",
    clip_delta: float = 0.01,
    grad_clip: float = 4.0,
    n_negative_samples: int = 10,
    B_fixed: np.ndarray = None,
    force_edges: np.ndarray = None,
    Y_init_override: np.ndarray = None,
    use_gravity: bool = False,
    node_mass: np.ndarray = None,
    gravity_strength: float = 1.0,
    gravity_neighbor_weight: bool = True,
    use_virtual_neighbor: bool = False,
    norm_mode: str = "relative",
    ramp: bool = True,
    snapshot_every: int = None,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Part A: drift folded into the UMAP metric itself.

        rho(i->j) = ||y_i-y_j|| + b_i . (y_j-y_i)      [OURS/report Eq. 2]
        grad_{y_i} rho = -(e_ij + b_i) =: -g_ij         [report Eq. 3]

    UMAP's own attractive/repulsive forces (Section 3.2) are evaluated at
    rho, g instead of d, e. b_i=0 for every i recovers exactly vanilla
    UMAP (use_drift=False), since then rho=d, g=e.

    B_fixed : [OURS 2026-08-05] (n,d) array, or None. If given, b_i is NOT
        recomputed every epoch (the whole "live drift" mechanism -- see
        module docstring point 3 -- is bypassed entirely). Instead every
        b_i is frozen at this value for the whole run: it still enters
        rho/g exactly as before and so still shapes the forces on Y every
        epoch, but the vector itself never changes length or direction --
        it is rigidly "attached" to node i and just rides along as y_i
        moves. See frozen_drift.py for how B_fixed itself gets built (a
        barycentric double-embedding of each node's own data-derived
        drift target). use_drift/drift_mode are ignored when B_fixed is
        given.

    force_edges : [OURS 2026-08-05] (m,2) int array of (i,j) index pairs,
        or None. For each pair, mu[i,j]=mu[j,i]=1.0 is set AFTER the
        normal k-NN fuzzy graph construction -- i.e. these edges get a
        guaranteed maximal attractive weight regardless of whether they
        would have ranked in the top n_neighbors on distance alone. Used
        by frozen_drift.py's joint (2n)-point embedding: a real node and
        its own virtual target must stay attractively linked, but generic
        k-NN competition among 2n-1 other points does not guarantee this
        (verified empirically: only ~36% of nodes had their own target
        rank in the top 20 without this override, leaving the rest with
        pure repulsion for the whole run and a badly degenerate result).

    Y_init_override : [OURS 2026-08-05] (n,d) array, or None. If given,
        used instead of spectral_layout() as the starting position. Also
        needed by frozen_drift.py's joint embedding: even WITH
        force_edges guaranteeing an attractive edge, the UMAP force curve
        weakens at large distance (attr_coeff ~ 1/rho as rho -> infinity,
        unlike a linear spring) -- verified empirically that generic
        spectral positions for the 2n augmented system can start a target
        tens of units from its own source, and 500 epochs of weak-at-
        range attraction cannot close that gap against constant ambient
        repulsion from every other point. Seeding each target AT (a small
        jitter away from) its own source's position fixes this: the
        force-directed loop only has to find the right LOCAL offset, not
        travel cross-embedding first.

    use_gravity : [OURS 2026-08-06, reworked 2026-08-17 per explicit user
        request] bool, default False (disabled, reproduces prior behaviour
        exactly). If True, an extra SEPARATE additive force pulls each node i
        toward its own per-node target xi_i = y_i + b_i -- Bannister et al.'s
        social-gravity force (arXiv:1209.0748):

            f_g(v) = gamma_t * M[v] * (xi - P[v])

        adapted to a per-node target xi_i instead of one shared center.
        Since xi_i - y_i = b_i by construction, this collapses to
        f_g(i) = gamma * M[i] * b_i -- see gravity_strength below for gamma_t,
        and gravity_neighbor_weight for the second 2026-08-17 change.

        [OURS 2026-08-17, per explicit user request] Previously this force
        was applied UNCONDITIONALLY at every epoch, regardless of whether
        ||b_i|| was small (a plausible, nearby target) or large (an implausible
        one relative to i's actual local neighbour scale) -- a constant push
        with no relation to UMAP's own attraction curve, which weakens as
        points get close. The user's proposed fix: treat the virtual point
        xi_i as a genuine k-NN CANDIDATE for node i, so gravity only pulls
        with real strength when xi_i would plausibly BE one of i's neighbours
        -- if b_i is small enough (relative to i's current real neighbour
        distances) to look like an actual nearby point, the pull is strong;
        if b_i is large (implausible drift), the pull fades toward zero,
        exactly like a UMAP edge weight decays for a distant point. See
        gravity_neighbor_weight below for the exact mechanism.

        This whole idea (a per-node gravity pull equal to the node's own
        drift vector, unconditional, gamma tunable) was previously tried on
        finsler_mds.py's free-B model (see Gravity_Report.tex): once that
        model's B-init bug was fixed, no gamma>0 showed a measurable benefit
        over the gamma=0 baseline there. Per explicit user instruction
        (2026-08-17), that earlier null result is NOT treated as a reason to
        skip re-testing here -- this is a structurally different force basis
        (UMAP's Phi-curve vs. plain Adam-optimised distance-fit loss) AND a
        structurally different force (neighbour-weighted, not unconditional),
        so the earlier result does not obviously carry over.

    gravity_strength : [OURS 2026-08-17, per explicit user request] float,
        default 1.0. This IS gamma_t from Bannister et al.'s formula above --
        previously absent entirely (force was hardcoded to exactly M[i]*b_i,
        gamma implicitly 1, no separate knob). Now an explicit, sweepable
        multiplier, matching the paper's own notation. Only has any effect
        when use_gravity=True.

    gravity_neighbor_weight : [OURS 2026-08-17, per explicit user request]
        bool, default True. If True, the gravity force on node i is scaled by
        a per-node weight w_i, computed as an UMAP-style exponential
        membership weight --
        NOT the k-NN augmentation itself (no extra points added to the
        system, no extra O(n^2) cost), just a per-node scalar computed each
        epoch from the CURRENT embedding:

            rho_i   = i's own nearest real-neighbour distance in Y (this epoch)
            sigma_i = local scale estimate (mean of i's k nearest real
                      neighbour distances in Y, minus rho_i)
            w_i     = exp(-max(0, ||b_i|| - rho_i) / sigma_i)

        This is the exact same smooth_knn_dist-style decay UMAP uses to turn
        a raw distance into an edge weight (Section 3.1) -- here applied to
        ||b_i|| (the fixed distance from y_i to its own virtual target xi_i)
        against i's LIVE local neighbourhood scale, so w_i in (0,1] answers
        "how plausible is it that xi_i is actually one of i's neighbours,
        right now, in the current embedding". If False, w_i=1 always (the
        old unconditional behaviour). sigma_i uses a cheap mean-based
        estimate rather than full binary-search calibration (recomputing the
        exact smooth_knn_dist calibration for every node every epoch would be
        the dominant cost of the whole loop).

    node_mass : [OURS 2026-08-06] (n,) per-node mass M[i], used only if
        use_gravity=True. None (default) uses M[i]=1 for all nodes (uniform).

    use_virtual_neighbor : [OURS 2026-08-20, per explicit user request] bool,
        default False. A DIFFERENT mechanism from use_gravity, not a
        variant of it -- both may be independently enabled, though they are
        not expected to be combined in practice. Each node i's own virtual
        point xi_i = y_i + b_i is treated as an UNCONDITIONAL (k+1)-th
        neighbour: it always participates in the attractive force, using
        UMAP's OWN attraction curve (the exact same attr_coeff formula real
        k-NN edges use, built from the same (a, b_param) fit to
        min_dist/spread) evaluated at rho_v = ||b_i|| -- NOT a separate,
        ad-hoc force basis like gravity's gamma*M[i]*b_i, and NOT gated by
        any plausibility/local-density weight (contrast with
        gravity_neighbor_weight's smooth w_i in (0,1] -- here the
        equivalent weight is always exactly 1, i.e. mu_virtual=1.0 for
        every node, every epoch, matching how force_edges already forces
        mu[i,j]=1.0 for a guaranteed edge elsewhere in this project):

            rho_v_i       = ||b_i||
            attr_coeff_v  = (2*a*b_param*rho_v^(2*b_param-1)) / (1 + a*rho_v^(2*b_param))
            force_virtual_i = attr_coeff_v * b_i

        No repulsive counterpart is needed (a permanent/unconditional
        neighbour has mu=1 everywhere, so (1-mu)=0 -- exactly like
        force_edges' guaranteed-attractive pairs). Since b_i is already the
        direction FROM y_i TO xi_i by construction, force_virtual_i needs
        no separate unit-vector term the way real edges' g=e_ij+b_i does.
        Works with both frozen (B_fixed) and live (use_drift) B.

    drift_mode : "knn" (adopted, report's verdict) -- b_i = (1/k) sum over
                 j in N_k(i) only, i.e. same k restricted neighbour set
                 used for the UMAP graph itself. Report Table 1: mean||b||
                 ~0.05, no clipping, best trust/dir-acc/alignment combo.
                 "all_j" -- b_i = (1/k) sum over EVERY j != i (still
                 divided by k, not n -- that is what makes it ~2.5x
                 LARGER, not smaller, than "knn"). Report's cautionary
                 variant: extent explodes (near-degenerate collapse) while
                 trust@10 barely moves, because trust@10 is a purely local
                 measure -- reproduced empirically below (see project
                 report for the numbers this was validated against).

    Full dense pairwise forces every epoch (all i,j, not sampled edges +
    negative samples) -- see module docstring for why, given n=232.

    n_negative_samples : the DENSE repulsion sum below (over every non-
        neighbour, ~n-1-k terms per node) is rescaled by
        n_negative_samples / (n-1-k) so its expected magnitude matches
        real UMAP's stochastic negative sampling (n_negative_samples
        random negatives per edge) exactly, instead of being ~10x too
        strong from summing every non-edge unweighted. This is the
        exact expectation of the stochastic estimator (zero variance),
        not an approximation of it -- see module docstring, point 5.

    Returns
    -------
    dict: Y, Y_init, B (final drift, zeros if use_drift=False), mu,
          knn_mask, a, b_param, history (list of mean |rho-mu-implied-d|
          per epoch, for a rough convergence trace)
    """
    D_asym = np.asarray(D_asym, dtype=np.float64)
    n = D_asym.shape[0]
    rng = np.random.default_rng(seed)

    mu, knn_mask = fuzzy_simplicial_set(D_asym, n_neighbors)
    if force_edges is not None and len(force_edges) > 0:
        fi, fj = np.asarray(force_edges)[:, 0], np.asarray(force_edges)[:, 1]
        mu[fi, fj] = 1.0
        mu[fj, fi] = 1.0
        np.fill_diagonal(mu, 0.0)
    a, b_param = find_ab_params(spread=spread, min_dist=min_dist)
    # [OURS 2026-08-10] norm_mode -- how N's raw units get removed before
    # it drives the drift sum. Ported from isomap_randers_umap.py (see
    # that file's docstring, "normalise_N" switch): "raw" (default, our
    # original behaviour) leaves N in D_asym's own units, which can
    # saturate/need heavy clipping if D isn't already ~[0,1]. "relative"
    # is dimensionless PER PAIR and bounded (|N|<=1 by construction), so
    # ||b_i||<=1 holds without clipping ever doing real work.
    if norm_mode == "relative":
        N = (D_asym - D_asym.T) / (D_asym + D_asym.T + 1e-12)
    elif norm_mode == "rowmax":
        Dr = D_asym.copy(); np.fill_diagonal(Dr, 0.0)
        Dr = Dr / np.maximum(Dr.max(axis=1, keepdims=True), 1e-12)
        N = 0.5 * (Dr - Dr.T)
    elif norm_mode == "maxabs":
        N = 0.5 * (D_asym - D_asym.T)
        N = N / max(np.abs(N).max(), 1e-12)
    elif norm_mode == "raw":
        N = 0.5 * (D_asym - D_asym.T)          # frozen asymmetric weighting [OURS]
    else:
        raise ValueError("norm_mode must be 'raw', 'relative', 'rowmax' or 'maxabs'")
    # [OURS 2026-08-07] D_asym is not guaranteed dense (e.g. a sparse,
    # pre-Dijkstra isumap data_D reconstruction has ~k defined entries per
    # row, the rest np.inf): inf-inf=nan and finite-inf=+-inf can appear in
    # N wherever a pair's distance is only known in one direction or
    # neither. Undefined pairs carry no directional information, so they
    # contribute nothing to the drift sum -- 0, not nan/inf.
    N = np.where(np.isfinite(N), N, 0.0)

    if drift_mode == "knn":
        drift_mask = knn_mask
    elif drift_mode == "all_j":
        drift_mask = ~np.eye(n, dtype=bool)
    else:
        raise ValueError("drift_mode must be 'knn' or 'all_j'")

    if Y_init_override is not None:
        Y = np.asarray(Y_init_override, dtype=np.float64).copy()
    else:
        Y = spectral_layout(mu, d, seed=seed)
    Y_init = Y.copy()

    B = np.zeros((n, d)) if B_fixed is None else np.asarray(B_fixed, dtype=np.float64).copy()
    M = np.ones(n) if node_mass is None else np.asarray(node_mass, dtype=np.float64)
    eps = 1e-8

    # [OURS 2026-08-11] snapshot capture -- mirrors finsler_mds_joint.py's
    # snapshot_every: records the embedding every N epochs so the whole
    # training trajectory (init -> ... -> final) can be plotted side by
    # side. epoch 0 entry is Y_init itself (pre-training).
    #
    # [FIX 2026-08-19, per explicit user request] When B_fixed is None and
    # use_drift=True (the "live" mechanism, B recomputed every epoch from
    # N=D_asym's own asymmetry and the current Y), B here is still the
    # np.zeros((n,d)) placeholder from line 520 -- the real B doesn't exist
    # until the training loop's first iteration computes it. That meant
    # --init-only (which reads exactly this epoch-0 snapshot) always showed
    # an empty (all-zero) drift, even though a genuine "epoch 0 drift" value
    # is well-defined and computable: it's exactly what the loop's first
    # iteration would compute, compute_drift(N, ..., Y_init), scaled by
    # ramp's own epoch-0 factor (0.0 if ramp=True -- ramp intentionally
    # holds drift off initially, so zero is correct THERE -- else 1.0).
    # This does NOT change live training itself (the loop still recomputes
    # B fresh every epoch); it only makes the pre-loop snapshot show a
    # meaningful value instead of a placeholder.
    snapshots = []
    if snapshot_every is not None:
        if B_fixed is None and use_drift:
            s0 = 0.0 if ramp else 1.0
            B_snap0 = s0 * compute_drift(N, drift_mask, n_neighbors, Y_init, clip_delta=clip_delta)
        else:
            B_snap0 = B.copy()
        snapshots.append({"epoch": 0, "Y": Y_init.copy(), "B": B_snap0})
    history = []

    # [OURS] expectation-matching correction for going dense instead of
    # sampling: real UMAP repels each point from n_negative_samples random
    # non-neighbours per edge; we sum over every non-neighbour instead
    # (~n-1-n_neighbors terms), so scale down by the ratio to reproduce the
    # same expected total repulsion magnitude.
    n_non_neighbors = max(n - 1 - n_neighbors, 1)
    rep_scale = n_negative_samples / n_non_neighbors

    if verbose:
        mode_str = "B_fixed (frozen)" if B_fixed is not None else f"use_drift={use_drift}"
        grav_str = (f"  +gravity(gamma={gravity_strength}, "
                    f"nbr_weight={gravity_neighbor_weight})" if use_gravity else "")
        vn_str = "  +virtual_neighbor(k+1, UMAP curve)" if use_virtual_neighbor else ""
        print(f"\n-- Randers-UMAP (Part A)  n={n} d={d} k={n_neighbors}  {mode_str}{grav_str}{vn_str} --")
        print(f"   a={a:.4f} b={b_param:.4f}  (min_dist={min_dist}, spread={spread})")

    for epoch in range(n_epochs):
        # [OURS 2026-08-10] ramp -- ported from isomap_randers_umap.py:
        # hold drift off for the first 30% of epochs (pure vanilla-UMAP
        # forces, lets the topology untangle from init before any
        # asymmetric pull), linearly ramp 0->1 over the next 40%, full
        # strength for the last 30%. s=1.0 always when ramp=False
        # (reproduces prior behaviour exactly).
        if ramp:
            t_prog = epoch / max(n_epochs - 1, 1)
            s = 0.0 if t_prog <= 0.30 else (1.0 if t_prog >= 0.70 else (t_prog - 0.30) / 0.40)
        else:
            s = 1.0

        if B_fixed is not None:
            B = s * np.asarray(B_fixed, dtype=np.float64)
        elif use_drift:
            B = s * compute_drift(N, drift_mask, n_neighbors, Y, clip_delta=clip_delta)
        # else: use_drift=False -> B stays all-zeros -> rho=d, g=e -> vanilla UMAP

        diff = Y[np.newaxis, :, :] - Y[:, np.newaxis, :]     # (n,n,d) y_j-y_i
        d_mat = np.maximum(np.sqrt((diff ** 2).sum(-1)), eps)
        e = diff / d_mat[:, :, np.newaxis]

        raw_dot = (B[:, np.newaxis, :] * diff).sum(-1)       # b_i . (y_j-y_i)
        rho = np.maximum(d_mat + raw_dot, eps)                # [OURS Eq. 2]
        g = e + B[:, np.newaxis, :]                           # [OURS Eq. 3 (sign folded into force below)]

        rho2b = rho ** (2 * b_param)
        denom = 1.0 + a * rho2b

        attr_coeff = (2 * a * b_param * rho ** (2 * b_param - 1)) / denom
        rep_coeff = (2 * b_param * rho) / ((eps + rho ** 2) * denom)

        force = (mu * attr_coeff)[:, :, np.newaxis] * g \
                - rep_scale * ((1.0 - mu) * rep_coeff)[:, :, np.newaxis] * g
        np.fill_diagonal(force[:, :, 0], 0.0)
        if d > 1:
            for dd in range(1, d):
                np.fill_diagonal(force[:, :, dd], 0.0)

        step = force.sum(axis=1)                              # (n,d) net force on y_i

        # [OURS 2026-08-06, reworked 2026-08-17 per explicit user request]
        # per-node gravity -- separate additive force, ported from
        # finsler_mds.py's gravity experiment (Gravity_Report.tex). Pulls y_i
        # toward xi_i = y_i + b_i; since xi_i - y_i = b_i by construction,
        # Bannister et al.'s f_g(v) = gamma_t*M[v]*(xi-P[v]) collapses to
        # gamma * M[i] * b_i. Does NOT touch rho/g/B above -- B is a fixed
        # source for this term (no gradient flows back onto B).
        if use_gravity:
            if gravity_neighbor_weight:
                # [OURS 2026-08-17] "is xi_i plausibly one of i's real
                # neighbours right now" weight -- reuses d_mat (already
                # computed above, this epoch's live pairwise Y distances).
                # Cheap mean-based local-scale estimate (NOT the full
                # smooth_knn_dist binary search -- redoing that per-node,
                # per-epoch, would dominate the loop's runtime).
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
            step = step + gravity_strength * w_grav[:, np.newaxis] * M[:, np.newaxis] * B

        # [OURS 2026-08-20, per explicit user request] virtual-neighbor --
        # each node's own xi_i = y_i + b_i is an UNCONDITIONAL (k+1)-th
        # attractive neighbour (mu_virtual=1.0 always, no plausibility
        # gating), pulled with UMAP's OWN attraction curve instead of
        # gravity's separate gamma*M*b_i basis. See module docstring for
        # the exact formula and why no repulsive counterpart is needed.
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
        B = compute_drift(N, drift_mask, n_neighbors, Y, clip_delta=clip_delta)
    # else B_fixed: leave B exactly as given -- it was never touched by the loop

    if snapshot_every is not None and snapshots[-1]["epoch"] != n_epochs:
        snapshots.append({"epoch": n_epochs, "Y": Y.copy(), "B": B.copy()})

    result = {
        "Y": Y, "Y_init": Y_init, "B": B, "mu": mu, "knn_mask": knn_mask,
        "N": N, "a": a, "b_param": b_param, "history": history,
    }
    if snapshot_every is not None:
        result["snapshots"] = snapshots
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6. Evaluation helpers -- mirror the report's Part A metrics table
#    (alignment, dir-acc, trust@10, extent). dir-acc reuses test.py's
#    direction_accuracy() directly rather than reinventing it [OURS].
# ─────────────────────────────────────────────────────────────────────────────
def extent(Y: np.ndarray) -> float:
    """Max distance from the embedding's own centroid -- a simple, honest
    proxy for how spread out/degenerate the layout is."""
    return float(np.linalg.norm(Y - Y.mean(axis=0), axis=1).max())


def alignment(Y: np.ndarray, Y_baseline: np.ndarray, B: np.ndarray) -> float:
    """
    Mean cosine similarity between each node's displacement relative to
    the drift-off (use_drift=False) baseline and its own final b_i.
    Matches the report's own definition and its own caveat (Section 4.2,
    "Other limitations"): this measures "was a force applied", not
    "is the embedding better" -- b both pushes and is being measured
    against, so a positive number confirms the mechanism is doing what
    it was derived to do, it is not by itself evidence of a better map.
    """
    disp = Y - Y_baseline
    disp_norm = np.linalg.norm(disp, axis=1, keepdims=True)
    b_norm = np.linalg.norm(B, axis=1, keepdims=True)
    valid = (disp_norm[:, 0] > 1e-9) & (b_norm[:, 0] > 1e-9)
    if not valid.any():
        return float("nan")
    cos = (disp[valid] * B[valid]).sum(1) / (disp_norm[valid, 0] * b_norm[valid, 0])
    return float(cos.mean())


def procrustes_align(X: np.ndarray, Y: np.ndarray):
    """
    [OURS 2026-08-14, per UMAP paper Section 5.3 "Embedding Stability"]
    Optimal translation + uniform scaling + rotation of Y onto X, where X
    and Y are (n,d) point sets with a known correspondence (point i in X
    <-> point i in Y) -- e.g. the same n_sub real points, embedded once as
    part of a larger n_full run (X) and once as a standalone n_sub run (Y).

    Procedure (orthogonal Procrustes, same recipe the paper describes):
      1. Center both point sets on their own centroid.
      2. Find the optimal ROTATION R (via SVD of Y_c^T X_c) that best
         aligns Y_c onto X_c -- this is the classic orthogonal Procrustes
         solution, minimising ||X_c - Y_c R||^2 over rotations R.
      3. Find the optimal uniform SCALE s that minimises ||X_c - s*(Y_c R)||^2
         (closed form: s = <X_c, Y_c R> / ||Y_c R||^2).
      4. Re-add X's centroid (translation) to get Y_aligned.
      5. Report the residual RMS distance, normalised by X's own average
         point-norm about its centroid -- a scale-free number, comparable
         across runs/datasets, matching the paper's own normalisation
         ("dividing by the average norm of the embedded dataset").

    Returns
    -------
    Y_aligned : (n,d) -- Y after the optimal translation/scale/rotation
    distance  : float -- normalised Procrustes distance, 0 = identical shape
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X_mean = X.mean(axis=0)
    Xc = X - X_mean
    Yc = Y - Y.mean(axis=0)

    U, _, Vt = np.linalg.svd(Yc.T @ Xc)
    R = U @ Vt                                    # optimal rotation

    Y_rot = Yc @ R
    denom = float((Y_rot ** 2).sum())
    s = float((Xc * Y_rot).sum() / denom) if denom > 1e-12 else 1.0  # optimal scale

    Y_aligned = s * Y_rot + X_mean

    resid = np.linalg.norm(X - Y_aligned, axis=1)
    norm = float(np.linalg.norm(Xc, axis=1).mean()) + 1e-12
    distance = float(np.sqrt((resid ** 2).mean()) / norm)
    return Y_aligned, distance


if __name__ == "__main__":
    # Standalone demo/report, mirrors test.py's `python test.py` entry point.
    from load_asymmetric_graph import load_migration_graph
    from test import direction_accuracy
    from sklearn.manifold import trustworthiness

    graph = load_migration_graph("network.csv/edges.csv", year=2015, verbose=True)
    D, D_sym, F_true = graph["D_asym"], graph["D_sym"], graph["F"]
    pair_mask = (D > 0) | (D.T > 0)

    print("\n[1/3] Training vanilla (use_drift=False) baseline...")
    res_van = randers_umap_fit(D, use_drift=False, seed=0, verbose=False)

    print("[2/3] Training Part A, drift_mode='knn' (adopted)...")
    res_knn = randers_umap_fit(D, use_drift=True, drift_mode="knn", seed=0, verbose=False)

    print("[3/3] Training Part A, drift_mode='all_j' (cautionary variant)...\n")
    res_allj = randers_umap_fit(D, use_drift=True, drift_mode="all_j", seed=0, verbose=False)

    print(f"{'config':28s} {'mean||b||':>10s} {'align':>8s} {'pair_agree':>11s} "
          f"{'w_pair_agree':>13s} {'trust@10':>9s} {'extent':>8s}")
    for name, res in [("vanilla (no drift)", res_van),
                       ("Part A: knn (adopted)", res_knn),
                       ("Part A: all_j (caution)", res_allj)]:
        bn = np.linalg.norm(res["B"], axis=1)
        trust10 = trustworthiness(D_sym, res["Y"], n_neighbors=10, metric="precomputed")
        da = direction_accuracy(res["Y"], res["B"], F_true, pair_mask)
        al = alignment(res["Y"], res_van["Y"], res["B"]) if bn.sum() > 0 else float("nan")
        print(f"{name:28s} {bn.mean():10.4f} {al:8.3f} {da['pair_agree']:10.1f}% "
              f"{da['w_pair_agree']:12.1f}% {trust10:9.4f} {extent(res['Y']):8.2f}")
