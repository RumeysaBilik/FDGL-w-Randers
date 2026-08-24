"""
randers_bridge.py
==================
Converts a raw point cloud X (n, m) plus a per-point Randers drift field
omega (n, m) into a full (n, n) asymmetric geodesic distance matrix D_asym,
suitable as direct input to randers_umap_fit() -- exactly the role that
load_migration_graph()'s D_asym plays for the migration dataset.

[OURS 2026-08-11] Adjacency construction rewritten to match
github.com/lwileczek/isomap's make_adjacency() (README "Step 1 Adjacency &
Distance Matrices", threshold variant) instead of sklearn's kneighbors_graph:

    dist = cdist(X, X)                     # full (n,n) pairwise distance
    adj  = inf everywhere
    adj[dist < eps] = dist[dist < eps]      # threshold, not k-NN membership
    D    = shortest_path(adj)

i.e. two points are an edge iff their Euclidean distance is below a
threshold `eps`, not iff one is among the other's k nearest neighbours.
Function name/signature kept as compute_dist_matrix() and n_neighbors kept
as the public knob (every caller in this repo passes n_neighbors=k) -- eps
is auto-derived from n_neighbors so nothing downstream has to change: eps
is set to the smallest radius such that every point has >= n_neighbors
neighbours within it (max, over all i, of point i's n_neighbors-th nearest
distance). Pass eps explicitly to bypass that and use lwileczek's raw
threshold knob directly.

Construction
------------
    1. Full pairwise Euclidean distance (scipy cdist), thresholded at eps
       -> adjacency matrix (lwileczek/isomap's method, not sklearn's k-NN
       graph).
    2. For every surviving edge (i, j):
           d(i, j) <- d(i, j) + <omega_i, x_j - x_i>
       (Randers-perturbed edge weight -- the discrete version of the
       continuous Randers metric F(x, v) = ||v|| + <omega_x, v>. This step
       has no counterpart in lwileczek/isomap -- it's the Finsler/Randers
       extension specific to this project.)
    3. Directed shortest-path (Dijkstra, via scipy's shortest_path -- the
       modern equivalent of the old sklearn.utils.graph_shortest_path that
       lwileczek/isomap's own code calls) over the resulting asymmetric
       weighted graph:
           D_asym[i, j] = geodesic distance i -> j,   in general != D_asym[j, i]
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial.distance import cdist


def compute_dist_matrix(
        X,
        n_neighbors=5,
        eps=None,
        path_method="auto",
        metric="euclidean",
        randers_field=None,
        directed=None,
        adjacency="threshold",
        return_adjacency=False,
):
    """
    Parameters
    ----------
    X             : (n, m) raw coordinates
    n_neighbors   : used only to auto-derive eps when eps=None (see module
                    docstring) -- kept for call-site compatibility with the
                    rest of this repo, which all pass n_neighbors=k. This
                    is independent of randers_umap_fit's own n_neighbors,
                    which builds a *second*, UMAP-style fuzzy graph on top
                    of the D_asym this function returns. When
                    adjacency="knn", this IS the actual per-point neighbour
                    count (see adjacency below), not just an eps-derivation
                    input.
    eps           : [OURS] lwileczek/isomap's actual threshold knob -- two
                    points are connected iff their Euclidean distance is
                    < eps. None (default) auto-derives eps from n_neighbors
                    so every point ends up with >= n_neighbors neighbours.
                    Pass a float to control the threshold directly instead.
                    Ignored when adjacency="knn".
    adjacency     : [OURS 2026-08-20, per explicit user request -- "bu
                    mantığı direkt k-nearest'a çevirip denemek istiyorum"]
                    "threshold" (default, unchanged) = the eps-threshold
                    rule above (lwileczek/isomap style, symmetric by
                    construction: dist(i,j)==dist(j,i) so i~j iff j~i).
                    "knn" = TRUE per-point k-nearest-neighbours membership
                    (sklearn.kneighbors_graph style, the pre-2026-08-11
                    method this file used to use): node i connects to
                    EXACTLY its n_neighbors nearest points, no more, no
                    fewer, regardless of local density. This is NOT
                    symmetric in general -- j being among i's k nearest
                    does not imply i is among j's k nearest -- so with
                    randers_field=None this reintroduces a second,
                    topology-driven source of directionality alongside the
                    Randers term (the exact asymmetry the 2026-08-11
                    threshold rewrite was meant to isolate away). Kept as
                    an explicit opt-in for side-by-side comparison, not a
                    replacement for the default.
    randers_field : (n, m) per-point drift vector omega_i (e.g. the `omega`
                    array from generated_swiss_roll-2.py), or None for the
                    plain Isomap-style geodesic distance
    directed      : bool or None. None (default): directed shortest-path
                    iff randers_field is given, undirected (symmetric
                    result) otherwise -- same semantics as before. Note the
                    eps-threshold adjacency (unlike sklearn's k-NN graph)
                    is symmetric by construction (dist(i,j)==dist(j,i)), so
                    with directed=True and randers_field=None there is no
                    longer any k-NN-membership asymmetry to isolate -- the
                    graph is symmetric until the Randers step perturbs it.
                    With adjacency="knn", this symmetry guarantee no longer
                    holds even before the Randers step (see adjacency above).
    path_method   : passed to scipy.sparse.csgraph.shortest_path
    return_adjacency : [OURS 2026-08-24, per explicit user request -- building
                    an asymmetry control metric] bool, default False. If True,
                    also return `bln`, the (n, n) boolean DIRECT-neighbour mask
                    (the raw graph edges used to build the geodesic distance,
                    before shortest_path) -- needed by asymmetry_score() below,
                    since D_asym itself is a DENSE all-pairs matrix (every pair
                    is finite once the graph is connected) and doesn't on its
                    own tell you which pairs were actual constructed edges.

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
        # [OURS 2026-08-20] true k-NN membership: row i keeps EXACTLY its
        # n_neighbors smallest entries, via argpartition (no full sort
        # needed -- O(n) per row instead of O(n log n)). Asymmetric in
        # general: bln[i,j] can be True while bln[j,i] is False.
        def _sparse_from_knn(k_val):
            k_val = min(k_val, n - 1)
            idx = np.argpartition(dist, k_val - 1, axis=1)[:, :k_val]
            rows = np.repeat(np.arange(n), k_val)
            cols = idx.ravel()
            vals = dist[rows, cols]
            bln_ = np.zeros((n, n), dtype=bool)
            bln_[rows, cols] = True
            return csr_matrix((vals, (rows, cols)), shape=(n, n)), bln_

        k_ = n_neighbors
        nbg, bln = _sparse_from_knn(k_)

        # [OURS] connectivity safety net, mirroring the eps-widening loop
        # below -- grow k by 1 (rather than eps *= 1.5) until the directed
        # k-NN graph's underlying connectivity is a single component.
        n_components, _ = connected_components(nbg)
        while n_components > 1 and k_ < n - 1:
            k_ += 1
            nbg, bln = _sparse_from_knn(k_)
            n_components, _ = connected_components(nbg)

    elif adjacency == "threshold":
        if eps is None:
            # smallest per-point n_neighbors-th nearest distance, maxed over
            # all points -- guarantees every node has >= n_neighbors neighbours
            # within the threshold (this is lwileczek/isomap's own suggestion:
            # "tune your threshold so that each node has some minimum number
            # of connections").
            kth = np.sort(dist, axis=1)[:, n_neighbors - 1]
            eps_ = kth.max()
        else:
            eps_ = eps

        def _sparse_from_threshold(eps_val):
            # bln[i, j] True iff dist(i, j) < eps_val -- lwileczek/isomap's
            # edge rule. Only True entries are stored (true sparsity, unlike
            # a dense inf-filled array), matching what kneighbors_graph gave
            # us before.
            bln_ = dist < eps_val
            rows, cols = np.nonzero(bln_)
            vals = dist[rows, cols]
            return csr_matrix((vals, (rows, cols)), shape=(n, n)), bln_

        nbg, bln = _sparse_from_threshold(eps_)

        # [OURS] connectivity safety net -- lwileczek/isomap's own code has no
        # fallback for a disconnected graph (shortest_path just leaves
        # unreachable pairs at inf). We widen eps until connected, since a
        # graph full of infs breaks every downstream step (SVD, Adam loss,
        # UMAP fuzzy graph) rather than just degrading gracefully.
        n_components, _ = connected_components(nbg)
        while n_components > 1:
            eps_ *= 1.5
            nbg, bln = _sparse_from_threshold(eps_)
            n_components, _ = connected_components(nbg)
    else:
        raise ValueError(f"adjacency must be 'threshold' or 'knn', got {adjacency!r}")

    # ── the actual Randers injection (no counterpart in lwileczek/isomap) ──
    if randers_field is not None:
        rows, cols = np.nonzero(bln)
        randers_update = np.einsum("ij,ij->i", X[cols] - X[rows], randers_field[rows])
        vals = dist[rows, cols] + randers_update
        nbg = csr_matrix((vals, (rows, cols)), shape=(n, n))
        directed_ = True if directed is None else directed
    else:
        directed_ = False if directed is None else directed

    dist_matrix_, preds_ = shortest_path(nbg, method=path_method, directed=directed_,
                                          return_predecessors=True)

    if X.dtype == np.float32:
        dist_matrix_ = dist_matrix_.astype(X.dtype, copy=False)

    if return_adjacency:
        return dist_matrix_, preds_, bln
    return dist_matrix_, preds_


def asymmetry_score(D, bln):
    """
    [OURS 2026-08-24, per explicit user request -- "asymmetriyi ne kadar
    kaybettiğimizi ne kadar sakladığımızı ölçecek bir kontrol parametresi"]
    Per-node, then global, control metric for how much of a distance matrix's
    magnitude is asymmetric (direction-dependent) vs symmetric, restricted to
    each point's DIRECT graph neighbours (bln) -- NOT all n-1 pairs, since
    D (e.g. D_asym from compute_dist_matrix) is dense: every pair is finite
    once the graph is connected, so without bln you'd be averaging over
    shortest-path-reachable pairs that were never actually adjacent edges.

    For each node i and each of its direct neighbours j (bln[i, j] True,
    i != j):
        asymm_ij = abs(D[i, j] - D[j, i])
        ratio_ij = asymm_ij / (D[i, j] + D[j, i])

    [OURS 2026-08-24, per explicit user request -- "0-1 arası çıksın"]
    ratio_ij is in [0, 1): 0 = this edge is perfectly symmetric
    (D[i,j]==D[j,i]); approaches 1 only in the extreme case where one
    direction's distance collapses to ~0 relative to the other (never
    exactly reaches 1 unless one direction is exactly 0). This is the
    earlier asymm_ij/symm_ij formula (symm_ij=(D[i,j]+D[j,i])/2) divided by
    2 -- same relative, scale-free ratio, just rescaled to land in [0,1]
    instead of [0,2) so it reads like a normalised "how asymmetric" score.

    Same function works on ANY (n, n) distance matrix + neighbour mask -- call
    it on the input D_asym (the Randers field's own, "target" asymmetry) and,
    separately, on a distance matrix built from a trained embedding (e.g. via
    the same canonical Randers formula evaluated on Y, B) to compare the two
    scores and see how much of the original asymmetry survived the embedding
    -- that comparison is the "ne kadar kaybettik / ne kadar sakladık" number,
    this function itself just computes the score for ONE given matrix.

    Parameters
    ----------
    D   : (n, n) dense distance matrix (e.g. D_asym from compute_dist_matrix,
          or any other (n, n) array of pairwise distances/dissimilarities)
    bln : (n, n) bool direct-neighbour mask (from compute_dist_matrix's
          return_adjacency=True) -- bln[i, j] True iff j is one of i's
          DIRECT constructed neighbours (not just shortest-path reachable)

    Returns
    -------
    per_node : (n,) ndarray -- mean ratio_ij over j in neighbours(i);
               np.nan for any node with zero direct neighbours (shouldn't
               happen for a connected graph with n_neighbors>=1, but guarded)
    global_score : float -- mean of per_node over all nodes (nanmean, so any
               isolated node doesn't skew/crash the average)
    """
    n = D.shape[0]
    per_node = np.full(n, np.nan)
    for i in range(n):
        js = np.nonzero(bln[i])[0]
        js = js[js != i]
        if len(js) == 0:
            continue
        asymm = np.abs(D[i, js] - D[js, i])
        denom = np.maximum(D[i, js] + D[js, i], 1e-12)  # avoid /0 for coincident points
        per_node[i] = np.mean(asymm / denom)
    global_score = float(np.nanmean(per_node))
    return per_node, global_score


def asymmetry_score_from_raw(D_raw):
    """
    [OURS 2026-08-24, per explicit user request -- "isumapler için nasıl
    uygulayabiliriz"] asymmetry_score() wrapper for RAW, sparse/inf-filled
    distance matrices -- the kind build_isumap_dist_matrix() (in
    run_swiss_roll_isumap.py / run_mammoth_isumap.py / run_sphere_isumap.py)
    hands back, via distance_graph_generation(), rather than
    compute_dist_matrix()'s own already-Dijkstra-completed D_asym.

    Why this needs a separate function, not just asymmetry_score(D_raw, bln)
    directly: isumap's raw D only populates ~k entries per row -- everywhere
    else is np.inf, INCLUDING, in general, the reverse of a real edge (j may
    be one of i's k nearest without i being one of j's -- the exact same
    per-point-asymmetric-membership issue as our own adjacency="knn" mode).
    So for a real edge (i,j) with D_raw[i,j] finite, D_raw[j,i] can easily
    still be inf -- and asymmetry_score's |D[i,j]-D[j,i]| would then blow up
    to inf instead of giving a meaningful ratio. Fix: complete D_raw via
    directed shortest_path FIRST (so every reachable pair has a finite
    distance in both directions), but keep bln (which pairs were REAL,
    directly-constructed edges) from the ORIGINAL raw matrix -- exactly
    mirroring what compute_dist_matrix() itself already does internally
    (raw finite edges decide adjacency/bln, then shortest_path densifies).

    Parameters
    ----------
    D_raw : (n, n) ndarray, np.inf for non-edges, 0 on the diagonal (the
            direct return of build_isumap_dist_matrix(), BEFORE any use by
            randers_umap_fit -- this function does not mutate D_raw or
            affect what actually gets fed to training, it only builds an
            internal completed COPY for scoring purposes)

    Returns
    -------
    per_node : (n,) ndarray, same semantics as asymmetry_score()
    global_score : float
    """
    n = D_raw.shape[0]
    bln = np.isfinite(D_raw) & ~np.eye(n, dtype=bool)
    rows, cols = np.nonzero(bln)
    nbg = csr_matrix((D_raw[rows, cols], (rows, cols)), shape=(n, n))
    D_complete, _ = shortest_path(nbg, method="auto", directed=True,
                                   return_predecessors=True)
    return asymmetry_score(D_complete, bln)
