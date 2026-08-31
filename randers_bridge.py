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

# [OURS 2026-08-28, per explicit user request -- "run_located_drift'i
# randers_bridge'e tasimak istiyorum, butun diger dosyalar oradan ceksin"]
# run_located_drift() itself (moved here from run_swiss_roll.py, see that
# function's own docstring below) needs randers_umap_fit's own init/fitting
# machinery -- imported here, not the other way around, so there's no
# circular import (randers_umap.py itself never imports from this module,
# only mentions it in comments/docstrings).
from randers_umap import randers_umap_fit, fuzzy_simplicial_set, spectral_layout, classical_mds


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
    D_complete = np.where(np.isfinite(D_complete), D_complete, D_complete.T)
    D_complete = np.where(np.isfinite(D_complete), D_complete, 0.0)
    return asymmetry_score(D_complete, bln)


def run_located_drift(X, omega, k=15, emb_k=20, neg=10, locate_epochs=500,
                      epochs=500, clip_delta=0.01, use_gravity=False,
                      gravity_strength=1.0, gravity_neighbor_weight=True,
                      use_virtual_neighbor=False, proj_dim=2, adjacency="threshold",
                      snapshot_every=None, ramp=False, seed=0, verbose=True,
                      apply_step=True, init_method="isomap",
                      normalize_drift_by_asymmetry=False,
                      force_model="fr_gravity", fr_k=None):
    """
    [OURS 2026-08-28, per explicit user request -- "run_located_drift
    fonksiyonunu randers_bridge'e veya randers_umap'e tasimak istiyorum,
    butun diger runladigimiz dosyalar oradan cekse"] Moved here verbatim
    from run_swiss_roll.py (where it was originally defined and where every
    other run_*.py/test.py/drift_magnitude_test.py/stability_check.py
    imported it from) -- randers_bridge.py is the natural home since this
    function's whole job is exactly what this module's own docstring
    describes (X, omega -> D_asym), just with the embedding step (STEP 2,
    randers_umap_fit) folded in too. run_swiss_roll.py now imports this
    function FROM here instead of defining it, so there is exactly one
    copy, matching every other caller.

    The full two-step pipeline, factored out so main() (CLI/plotting) in
    each run_*.py script and test.py/drift_magnitude_test.py/
    stability_check.py's own diagnostics all call the exact same code -- no
    duplication.

    adjacency : [OURS 2026-08-20, per explicit user request -- "bu mantığı
        direkt k-nearest'a çevirip denemek istiyorum"] "threshold" (default,
        unchanged) or "knn", forwarded to BOTH compute_dist_matrix calls
        below (locate step's D_sym_aug AND apply step's D_asym). See
        compute_dist_matrix's own adjacency docstring above for the full
        explanation of the difference.

    proj_dim : [OURS 2026-08-20, per explicit user request] int, default 2.
        Embedding dimension for BOTH the locate step's placement
        (classical_mds/spectral_layout) and the apply step's
        randers_umap_fit call. randers_umap_fit's own `d` parameter
        already supports arbitrary dimension -- this was previously
        hardcoded to 2 at every call site in this file. Pass 3 for a full
        3D force-directed layout, mirroring FinslerMDS's --proj-dim
        convention (main_sphere_tangential.py etc.).

    STEP 1 locate : place n real + n virtual (x_i+omega_i) points with ONE
                    deterministic placement call (no training), read off
                    B := Y_virtual - Y_real from that placement.
    STEP 2 apply  : embed the n real points on the true D_asym, with that
                    B frozen + attached (B_fixed), optionally +gravity --
                    only a_i (each real point's own position) trains each
                    epoch, b_i is never touched.

    init_method : [OURS 2026-08-16, per explicit user request] "umap"
        (default, unchanged) uses fuzzy_simplicial_set+spectral_layout --
        UMAP's own Laplacian-eigenmap init. "isomap" uses classical_mds
        instead -- Isomap's own finishing step (D_sym_aug here is already
        Isomap-style k-NN+Dijkstra via compute_dist_matrix, fully dense, so
        this makes the WHOLE pipeline consistently Isomap, not just the
        distance construction). Only affects STEP 1's placement method;
        everything else (B extraction, apply-step training) is identical.

    snapshot_every : [OURS 2026-08-11] int or None. If given, forwarded to
        the APPLY step's randers_umap_fit call only (the locate step's
        result isn't what we usually want to watch evolve) -- captures Y
        every snapshot_every epochs, from the initial Y_real0 through to
        the final embedding, for a "training trajectory" plot.

    normalize_drift_by_asymmetry : [OURS 2026-08-25, per explicit user
        request -- "default eskisi gibi normalize edilmeden olsun, --normalize
        flagi verirsek normalize etsin", later changed to "direkt kth nearest
        neighbouruna gore yapicaz"] bool, default False (off -- exact prior
        behaviour, B_located used as-is, frozen direction AND magnitude for
        the whole run). If True: B_located's DIRECTION stays exactly as
        located, but its MAGNITUDE is replaced every epoch with that node's
        LIVE distance to its own k-th nearest neighbour in the current
        embedding Y -- forwarded to randers_umap_fit as
        scale_B_fixed_by_knn_distance (see its docstring there for the
        exact mechanism; this needs the live, epoch-by-epoch Y, so it can't
        be precomputed here before training starts, unlike the earlier
        asymmetry_score-based version this replaced). asym_per_node/
        asym_global (below) remain always-computed diagnostics either way,
        just no longer used to set B's length when this flag is off... or
        on, now -- they're purely informational now regardless.

    apply_step : [OURS 2026-08-13] bool, default True. If False, skip STEP 2
        entirely -- no D_asym build, no force-directed training -- and
        return with "Y"/"B" set to the raw locate-step output (Y_real0 /
        B_located). Lets callers get just the initial (located) embedding
        with its drift vectors, e.g. for a quick "--init-only" mode.

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
    D_sym_aug, _ = compute_dist_matrix(X_aug, n_neighbors=k, path_method="auto",
                                       randers_field=None, adjacency=adjacency)

    # [OURS 2026-08-16] init-only: ONE deterministic placement call on the
    # augmented graph, no force-directed training. locate_epochs is
    # intentionally unused now -- kept as a parameter only so existing
    # callers/CLI flags don't break.
    if init_method == "isomap":
        if verbose:
            print(f"Locate: classical_mds on the augmented graph (no training)...")
        Y_aug0 = classical_mds(D_sym_aug, d=proj_dim, seed=seed)
    elif init_method == "umap":
        if verbose:
            print(f"Locate: spectral_layout on the augmented graph (no training)...")
        mu_aug, _ = fuzzy_simplicial_set(D_sym_aug, emb_k)
        Y_aug0 = spectral_layout(mu_aug, d=proj_dim, seed=seed)
    else:
        raise ValueError(f"init_method must be 'umap' or 'isomap', got {init_method!r}")
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
        # [OURS 2026-08-13] init-only: stop here, hand back the raw located
        # embedding/drift as "Y"/"B" so callers (main()'s plotting code) can
        # treat this exactly like a normal result -- no D_asym built, no
        # training run.
        if verbose:
            print("\napply_step=False -- skipping STEP 2, returning located init only.")
        return {"Y": Y_real0, "B": B_located, "D_asym": None,
                "Y_real0": Y_real0, "Y_virtual0": Y_virtual0, "B_located": B_located}

    # ---- apply: real D_asym, B frozen + attached ----------------------------
    if verbose:
        print(f"\nApply: building asymmetric D_asym on the {n} real points...")
    D_asym, _, bln_asym = compute_dist_matrix(X, n_neighbors=k, path_method="auto",
                                    randers_field=omega, adjacency=adjacency,
                                    return_adjacency=True)

    # [OURS 2026-08-24, per explicit user request] control metric -- how much
    # of D_asym's magnitude, on average over each node's real graph
    # neighbours, is asymmetric (direction-dependent) vs symmetric. See
    # asymmetry_score's own docstring above for the exact formula.
    # Always computed (cheap, O(n*k)) and stashed in the result dict; printed
    # automatically when verbose, same as the other summary stats below.
    asym_per_node, asym_global = asymmetry_score(D_asym, bln_asym)
    if verbose:
        print(f"asymmetry_score: global={asym_global:.4f}  "
              f"(mean |d_ij-d_ji| / mean-dist, averaged over each node's real "
              f"neighbours, then over all nodes -- 0 = fully symmetric)")

    # [OURS 2026-08-25, per explicit user request -- "normalizasyonu
    # degistirip direkt kth nearest neighbouruna gore yapicaz"] default
    # False = exact prior behaviour, B_located used as-is (frozen
    # direction AND magnitude for the whole run). True: B_located's
    # DIRECTION stays exactly as located, but its MAGNITUDE is replaced,
    # EVERY EPOCH, with that node's live distance to its own k-th nearest
    # neighbour in the CURRENT embedding Y -- handled entirely inside
    # randers_umap_fit's own loop via scale_B_fixed_by_knn_distance (see
    # its docstring there for the exact mechanism), since this needs the
    # LIVE, epoch-by-epoch Y, not a one-time value computable here before
    # training even starts. This REPLACES the previous asymmetry_score-
    # based magnitude target (kept only as the always-computed diagnostic
    # asym_per_node/asym_global below, no longer used to set B's length).
    out2 = randers_umap_fit(D_asym, n_neighbors=emb_k, n_negative_samples=neg,
                            n_epochs=epochs, use_drift=True, d=proj_dim,
                            B_fixed=B_located, Y_init_override=Y_real0,
                            use_gravity=use_gravity, gravity_strength=gravity_strength,
                            gravity_neighbor_weight=gravity_neighbor_weight,
                            use_virtual_neighbor=use_virtual_neighbor, ramp=ramp,
                            snapshot_every=snapshot_every,
                            scale_B_fixed_by_knn_distance=normalize_drift_by_asymmetry,
                            clip_delta=clip_delta, seed=seed, verbose=verbose,
                            force_model=force_model, fr_k=fr_k)
    Y, B = out2["Y"], out2["B"]

    if verbose:
        bn = np.linalg.norm(B, axis=1)
        print(f"\nextent={Y.max()-Y.min():.2f}  mean||b||={bn.mean():.4f}  "
              f"max||b||={bn.max():.4f}  clipped={(bn >= limit - 1e-9).sum()}/{n}")

    # [OURS 2026-08-28] key names kept EXACTLY as run_swiss_roll.py's
    # original ("asymmetry_score"/"asymmetry_per_node", not "asym_global"/
    # "asym_per_node") -- every run_*.py's own main() reads these back via
    # result.get("asymmetry_score", ...)/result.get("asymmetry_per_node", ...)
    # when saving its .npz, so the key names are part of this function's
    # real interface, not just internal naming.
    result = {"Y": Y, "B": B, "D_asym": D_asym,
              "Y_real0": Y_real0, "Y_virtual0": Y_virtual0, "B_located": B_located,
              "asymmetry_score": asym_global, "asymmetry_per_node": asym_per_node}
    if snapshot_every is not None:
        result["snapshots"] = out2["snapshots"]
    return result
    return asymmetry_score(D_complete, bln)
