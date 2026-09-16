"""
randers_bridge.py
==================
Converts a raw point cloud X (n, m) plus a per-point Randers drift field
omega (n, m) into a full (n, n) asymmetric geodesic distance matrix D_asym,
suitable as direct input to fdgl_low_dim().

Construction (compute_dist_matrix)
-----------------------------------
    1. Build an adjacency graph on X: true k-NN membership
       (adjacency="knn", the default) or an eps-threshold rule
       (adjacency="threshold", eps auto-derived from n_neighbors if not
       given) -- see compute_dist_matrix's own docstring for the difference.
    2. For every edge (i, j), perturb its weight with the Randers field:
           d(i, j) <- d(i, j) + <omega_i, x_j - x_i>
       the discrete version of the continuous Randers metric
       F(x, v) = ||v|| + <omega_x, v>.
    3. Directed shortest-path (Dijkstra) over the resulting asymmetric
       weighted graph:
           D_asym[i, j] = geodesic distance i -> j,   in general != D_asym[j, i]
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial.distance import cdist

from randers_fdgl import fdgl_low_dim, classical_mds


def compute_dist_matrix(
        X,
        n_neighbors=5,
        metric="euclidean",
        randers_field=None,
        directed=None,
        adjacency="knn",
        return_adjacency=False,
):
    """
    Shortest-path step always uses Dijkstra: every caller in this repo
    builds a sparse, non-negative-weighted graph (clip_delta/the Randers
    field's own ||omega||<1 constraint keeps every perturbed edge weight
    positive), exactly what Dijkstra is for.

    Parameters
    ----------
    X             : (n, m) raw coordinates
    n_neighbors   : target neighbour count (adjacency="threshold" derives
                    an eps from this so every point has >= n_neighbors
                    neighbours; adjacency="knn" uses it as the actual
                    per-point neighbour count)
    adjacency     : "knn" (default) -- each node connects to its
                    n_neighbors nearest points; asymmetric in general
                    (edge (i,j) doesn't imply edge (j,i)), which is the
                    asymmetric-existence structure the drift signal
                    (_compute_N in randers_fdgl.py) is meant to pick up on.
                    "threshold" -- connect i,j iff dist(i,j) < eps;
                    symmetric by construction.
    randers_field : (n, m) per-point drift vector omega_i, or None for the
                    plain Isomap-style geodesic distance
    directed      : bool or None. None (default): directed iff
                    randers_field is given, undirected otherwise.
    return_adjacency : bool, default False. If True, also return `bln`,
                    the (n, n) boolean direct-neighbour mask (needed by
                    asymmetry_score(), since dense D_asym alone doesn't
                    show which pairs were real edges).

    Returns
    -------
    dist_matrix_ : (n, n) dense ndarray -- this is D_asym
    preds_       : (n, n) shortest-path predecessor matrix
    bln          : (n, n) bool ndarray, only returned if return_adjacency=True
    """
    n = X.shape[0]
    dist = cdist(X, X, metric=metric)
    np.fill_diagonal(dist, np.inf)  # exclude self so eps auto-derivation below ignores it

    if adjacency == "knn":
        # row i keeps exactly its n_neighbors smallest entries, via
        # argpartition (O(n) per row instead of a full O(n log n) sort).
        # Asymmetric in general: bln[i,j] can be True while bln[j,i] is False.
        def _sparse_from_knn(k_val):
            k_val = min(k_val, n - 1)
            idx = np.argpartition(dist, k_val - 1, axis=1)[:, :k_val]
            rows = np.repeat(np.arange(n), k_val)
            cols = idx.ravel()
            vals = dist[rows, cols]
            bln_ = np.zeros((n, n), dtype=bool)
            bln_[rows, cols] = True
            return csr_matrix((vals, (rows, cols)), shape=(n, n)), bln_

        nbg, bln = _sparse_from_knn(n_neighbors)

        # Disconnected graph -> shortest_path would leave some pairs at
        # inf, silently breaking every downstream step. Error out with the
        # requested n_neighbors rather than silently connecting the graph
        # with a larger k than the caller asked for.
        n_components, _ = connected_components(nbg)
        if n_components > 1:
            raise ValueError(
                f"adjacency='knn' with n_neighbors={n_neighbors} gives a disconnected "
                f"graph ({n_components} components). Increase n_neighbors, or pass "
                f"adjacency='threshold'.")

    elif adjacency == "threshold":
        # smallest per-point n_neighbors-th nearest distance, maxed over all
        # points -- guarantees every node has >= n_neighbors neighbours
        # within the threshold.
        kth = np.sort(dist, axis=1)[:, n_neighbors - 1]
        eps_ = kth.max()

        def _sparse_from_threshold(eps_val):
            # bln[i, j] True iff dist(i, j) < eps_val.
            bln_ = dist < eps_val
            rows, cols = np.nonzero(bln_)
            vals = dist[rows, cols]
            return csr_matrix((vals, (rows, cols)), shape=(n, n)), bln_

        nbg, bln = _sparse_from_threshold(eps_)

        # Same reasoning as the knn branch above -- error instead of
        # silently widening eps past what n_neighbors implied.
        n_components, _ = connected_components(nbg)
        if n_components > 1:
            raise ValueError(
                f"adjacency='threshold' with n_neighbors={n_neighbors} (eps={eps_:.4g}) "
                f"gives a disconnected graph ({n_components} components). Increase "
                f"n_neighbors.")
    else:
        raise ValueError(f"adjacency must be 'threshold' or 'knn', got {adjacency!r}")

    # ── Randers injection ──
    if randers_field is not None:
        rows, cols = np.nonzero(bln)
        randers_update = np.einsum("ij,ij->i", X[cols] - X[rows], randers_field[rows])
        vals = dist[rows, cols] + randers_update
        nbg = csr_matrix((vals, (rows, cols)), shape=(n, n))
        directed_ = True if directed is None else directed
    else:
        directed_ = False if directed is None else directed

    dist_matrix_, preds_ = shortest_path(nbg, method="D", directed=directed_,
                                          return_predecessors=True)

    if X.dtype == np.float32:
        dist_matrix_ = dist_matrix_.astype(X.dtype, copy=False)

    if return_adjacency:
        return dist_matrix_, preds_, bln
    return dist_matrix_, preds_


def asymmetry_score(D, bln=None):
    """
    For each node's direct graph neighbours, how much do D[i,j] and D[j,i]
    disagree?
        asymm_ij = |D[i,j] - D[j,i]|
        per_node[i] = mean_j(asymm_ij)   (j in i's direct neighbours)
        global_score = mean_i(per_node[i])
    In D's own units, not a ratio.

    Pass bln when D is already dense -- bln says which pairs are 
    real edges. Leave bln=None when D is sparse -- it's derived from
    D's own finite entries and D is Dijkstra-completed internally first, so
    scoring doesn't blow up comparing a finite entry against a raw inf.

    Returns (per_node, global_score).
    """
    if bln is None:
        n = D.shape[0]
        bln = np.isfinite(D) & ~np.eye(n, dtype=bool)
        rows, cols = np.nonzero(bln)
        nbg = csr_matrix((D[rows, cols], (rows, cols)), shape=(n, n))
        D_complete, _ = shortest_path(nbg, method="auto", directed=True,
                                       return_predecessors=True)
        D_complete = np.where(np.isfinite(D_complete), D_complete, D_complete.T)
        D_complete = np.where(np.isfinite(D_complete), D_complete, 0.0)
        D = D_complete

    n = D.shape[0]
    per_node = np.full(n, np.nan)
    for i in range(n):
        js = np.nonzero(bln[i])[0]
        js = js[js != i]
        if len(js) == 0:
            continue
        asymm = np.abs(D[i, js] - D[js, i])
        per_node[i] = np.mean(asymm)
    global_score = float(np.nanmean(per_node))
    return per_node, global_score


def reconstruct_rho(Y, B):
    """
    rho(i->j) = ||y_i-y_j|| + b_i.(y_j-y_i) -- same formula
    fdgl_low_dim's training loop uses, standalone here so
    asymmetry_score() can be run on the trained embedding's own
    reconstructed distances, not just the raw input D_asym.
    """
    diff = Y[np.newaxis, :, :] - Y[:, np.newaxis, :]
    d = np.sqrt((diff ** 2).sum(-1))
    proj = (B[:, np.newaxis, :] * diff).sum(-1)
    rho = d + proj
    np.fill_diagonal(rho, 0.0)
    return rho


def fdgl_pipeline(X, omega, k=15, emb_k=20, neg=10, locate_epochs=500,
                      epochs=500, clip_delta=0.01, use_gravity=False,
                      gravity_strength=1.0, gravity_neighbor_weight=True,
                      use_virtual_neighbor=False, proj_dim=2, adjacency="knn",
                      snapshot_every=None, ramp=False, seed=0, verbose=True,
                      apply_step=True,
                      normalize_drift_by_asymmetry=False,
                      force_model="fr_gravity", fr_k=None, negative_sampling=False,
                      randers_attractive=True, randers_repulsive=False):
    """
    The full two-step "located drift" pipeline: builds D_asym from (X, omega)
    and trains an embedding with a frozen drift vector B.

    adjacency : "knn" or "threshold", forwarded to both
        compute_dist_matrix calls below (locate step's D_sym_aug and apply
        step's D_asym).

    randers_attractive, randers_repulsive : bool, forwarded to
        fdgl_low_dim's same-named parameters (default: attractive=True,
        repulsive=False). D_geo (computed below from
        the same X/adjacency, randers_field=None) is always forwarded too,
        so switching either flag to False gets a correctly Euclidean-
        weighted force term instead of reusing the Randers-derived one.

    proj_dim : int, default 2. Embedding dimension for both the locate
        step's placement (classical_mds) and the apply step's
        fdgl_low_dim call. 

    STEP 1 locate : place n real + n virtual (x_i+omega_i) points with one
                    deterministic classical_mds call on D_sym_aug (no
                    training), read off B := Y_virtual - Y_real.
    STEP 2 apply  : embed the n real points on the true D_asym, with that
                    B frozen + attached (B_fixed), optionally +gravity --
                    only a_i (each real point's own position) trains each
                    epoch, b_i is never touched.

    snapshot_every : int or None. If given, forwarded to the apply step's
        fdgl_low_dim call only -- captures Y every snapshot_every
        epochs, from Y_real0 through the final embedding, for a training
        trajectory plot.

    normalize_drift_by_asymmetry : bool, default False. If False, B_located
        is used as-is (frozen direction and magnitude for the whole run).
        If True, B_located's direction stays fixed but its magnitude is
        replaced every epoch with that node's live distance to its own
        k-th nearest neighbour in the current embedding Y -- forwarded to
        fdgl_low_dim as scale_B_fixed_by_knn_distance (see its
        docstring for the exact mechanism).

    apply_step : bool, default True. If False, skip STEP 2 entirely -- no
        D_asym build, no training -- and return "Y"/"B" set to the raw
        locate-step output (Y_real0/B_located). Used for a quick
        "--init-only" mode.

    Returns
    -------
    dict: Y, B, D_asym, Y_real0, Y_virtual0, B_located (pre-apply, for
          diagnostics that want to inspect the locate step in isolation),
          snapshots (list of {"epoch", "Y", "B"}, only if snapshot_every given)
    """
    n = X.shape[0]

    # ---- locate: embed n real + n virtual (x_i+omega_i) points together ----
    X_virtual = X + omega
    X_aug = np.vstack([X, X_virtual])

    if verbose:
        print(f"\nLocate: building symmetric geodesic D on {2*n} augmented points...")
    D_sym_aug, _ = compute_dist_matrix(X_aug, n_neighbors=k,
                                       randers_field=None, adjacency=adjacency)

    # [OURS 2026-09-16] (real,real) block used to be plain Euclidean (via
    # X_aug above, randers_field=None). Swap it for the symmetrized Randers
    # geodesic instead -- 0.5*(D_asym+D_asym^T), D_asym = the same directed
    # distance the apply step below uses -- so the locate step actually
    # reflects the Randers structure on the real points, not just on the
    # real-virtual/virtual-virtual links. Only the real-real quadrant
    # changes; the rest of D_sym_aug (built above) is untouched.
    if verbose:
        print(f"Locate: (real,real) block -> symmetrized Randers geodesic "
              f"0.5*(D_asym+D_asym^T)...")
    D_real_asym, _ = compute_dist_matrix(X, n_neighbors=k,
                                         randers_field=omega, adjacency=adjacency)
    D_sym_aug[:n, :n] = 0.5 * (D_real_asym + D_real_asym.T)
    np.fill_diagonal(D_sym_aug, 0.0)

    # One deterministic placement call, no training. locate_epochs is
    # unused -- kept only so existing callers/CLI flags don't break.
    if verbose:
        print(f"Locate: classical_mds on the augmented graph (no training)...")
    Y_aug0 = classical_mds(D_sym_aug, d=proj_dim, seed=seed)
    Y_real0, Y_virtual0 = Y_aug0[:n], Y_aug0[n:]

    B_located = Y_virtual0 - Y_real0
    limit = 1.0 - clip_delta
    bn0 = np.linalg.norm(B_located, axis=1, keepdims=True)
    B_located = B_located * np.minimum(1.0, limit / np.maximum(bn0, 1e-12))

    if verbose:
        bn = np.linalg.norm(B_located, axis=1)
        print(f"B located: mean||b||={bn.mean():.4f}  max||b||={bn.max():.4f}  "
              f"clipped={(bn >= limit - 1e-9).sum()}/{n}")

    if not apply_step:
        # Stop here and hand back the raw located embedding/drift as
        # "Y"/"B" so callers can treat this like a normal result.
        if verbose:
            print("\napply_step=False -- skipping STEP 2, returning located init only.")
        return {"Y": Y_real0, "B": B_located, "D_asym": None,
                "Y_real0": Y_real0, "Y_virtual0": Y_virtual0, "B_located": B_located}

    # ---- apply: real D_asym, B frozen + attached ----------------------------
    if verbose:
        print(f"\nApply: building asymmetric D_asym on the {n} real points...")
    D_asym, _, bln_asym = compute_dist_matrix(X, n_neighbors=k,
                                    randers_field=omega, adjacency=adjacency,
                                    return_adjacency=True)

    # D_geo -- drift-free counterpart of D_asym, used by fdgl_low_dim
    # when randers_attractive/randers_repulsive is False (see docstring).
    D_geo, _ = compute_dist_matrix(X, n_neighbors=k,
                                    randers_field=None, adjacency=adjacency)

    asym_per_node, asym_global = asymmetry_score(D_asym, bln_asym)
    if verbose:
        print(f"asymmetry_score: global={asym_global:.4f}  "
              f"(mean |d_ij-d_ji|, in D_asym's own units, averaged over each "
              f"node's real neighbours, then over all nodes -- 0 = fully symmetric)")

    out2 = fdgl_low_dim(D_asym, n_neighbors=emb_k, n_negative_samples=neg,
                            n_epochs=epochs, use_drift=True, d=proj_dim,
                            B_fixed=B_located, Y_init_override=Y_real0,
                            use_gravity=use_gravity, gravity_strength=gravity_strength,
                            gravity_neighbor_weight=gravity_neighbor_weight,
                            use_virtual_neighbor=use_virtual_neighbor, ramp=ramp,
                            snapshot_every=snapshot_every,
                            scale_B_fixed_by_knn_distance=normalize_drift_by_asymmetry,
                            clip_delta=clip_delta, seed=seed, verbose=verbose,
                            force_model=force_model, fr_k=fr_k,
                            negative_sampling=negative_sampling,
                            D_geo=D_geo, randers_attractive=randers_attractive,
                            randers_repulsive=randers_repulsive)
    Y, B = out2["Y"], out2["B"]

    # Same score on the trained embedding's reconstructed distances --
    # asym_global_final vs asym_global = how much asymmetry survived training.
    rho_final = reconstruct_rho(Y, B)
    asym_per_node_final, asym_global_final = asymmetry_score(rho_final, bln_asym)

    if verbose:
        bn = np.linalg.norm(B, axis=1)
        print(f"\nextent={Y.max()-Y.min():.2f}  mean||b||={bn.mean():.4f}  "
              f"max||b||={bn.max():.4f}  clipped={(bn >= limit - 1e-9).sum()}/{n}")
        pct = 100.0 * asym_global_final / max(asym_global, 1e-12)
        print(f"asymmetry_score (final embedding): {asym_global_final:.4f}  "
              f"vs target (D_asym) {asym_global:.4f}  -- {pct:.1f}% of target asymmetry preserved")

    result = {"Y": Y, "B": B, "D_asym": D_asym,
              "Y_real0": Y_real0, "Y_virtual0": Y_virtual0, "B_located": B_located,
              "asymmetry_score": asym_global, "asymmetry_per_node": asym_per_node,
              "asymmetry_score_final": asym_global_final,
              "asymmetry_per_node_final": asym_per_node_final}
    if snapshot_every is not None:
        result["snapshots"] = out2["snapshots"]
    return result
