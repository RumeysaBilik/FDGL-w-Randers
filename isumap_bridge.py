#!/usr/bin/env python3
"""
isumap_bridge.py -- [OURS 2026-09-08] shared D_asym-construction + init
infrastructure for the isumap family (run_swiss_roll_isumap.py,
run_mammoth_isumap.py, run_sphere_isumap.py, asymmetry_k_sweep_isumap.py).

This is the isumap-pipeline counterpart of randers_bridge.py -- but the two
are NOT interchangeable: randers_bridge.py builds D_asym from a Randers
VECTOR FIELD (X, omega) via compute_dist_matrix()/run_located_drift(), while
this module builds D_asym purely from distance_graph_generation()'s own
directed k-NN/star-graph structure -- no field, no omega, ever. Kept as a
separate file (rather than added to randers_bridge.py) specifically to avoid
implying these are the same kind of D_asym construction.

Moved here 2026-09-08 from run_swiss_roll_isumap.py, where these two
functions used to live even though run_mammoth_isumap.py, run_sphere_isumap.py
and asymmetry_k_sweep_isumap.py all imported them from there too -- i.e. every
non-swiss_roll isumap script secretly depended on the swiss_roll-named file
for its own core D_asym/init machinery. Purely a rename/relocation for
honesty (matches the fuzzy_simplicial_set -> knn_mask_from_distance_matrix/
_knn_weights split from the same day): no behaviour change, verified via
diff-equivalent bodies.
"""

import numpy as np

from distance_graph_generation import distance_graph_generation
from randers_umap import classical_mds


def build_isumap_dist_matrix(X, k=20, verbose=True):
    """Same recipe as asymm_dist_MNIST.py: pull data_D (isumap_dist[0]),
    NOT D -- reconstruct the (i,j,k) dict into a dense (n,n) matrix.

    [OURS 2026-08-07 bug fix] data_D only populates ~k entries per row
    (the epm=True "pure star graph" default -- see distance_graph_generation.py
    docstring). asymm_dist_MNIST.py's original reconstruction used
    np.zeros(), leaving every UN-populated (i,j) pair at exactly 0.0 --
    indistinguishable from a genuine zero distance. randers_umap.py's
    _knn_from_distance_matrix() picks the k SMALLEST values per row via
    argsort, so on a mostly-zero-filled row it was picking ~k phantom
    "distance-0" non-edges as the nearest neighbours instead of the ~k
    REAL populated ones (verified empirically: for k=20, all 20 selected
    neighbours were phantom zeros, 0/20 real). Filling the unpopulated
    entries with np.inf instead fixes this -- inf can never win an
    argsort-smallest selection, and downstream smooth_knn_dist/mu
    computations already handle inf gracefully (exp(-inf/sigma) = 0, so
    an inf "neighbour" that leaks into the top-k for an under-populated
    row just gets zero weight instead of corrupting the graph).
    """
    n = X.shape[0]
    isumap_dist = distance_graph_generation(
        X, k=k, normalize=True, distBeyondNN=True, verbose=verbose,
        dataIsDistMatrix=False, dataIsGeodesicDistMatrix=False, saveDistMatrix=False,
    )
    data_D = isumap_dist[0]
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0.0)
    for key, value in data_D.items():
        i, j, k_ = key
        D[i, j] = value
    return D


def isumap_style_init(D_asym, d=2, seed=0):
    """
    [OURS 2026-09-03] Real IsUMap (github.com/LUK4S-B/IsUMap, src/isumap.py,
    `initialization="cMDS"` default) initialises its embedding with
    classical/Torgerson MDS, NOT a UMAP-style spectral (Laplacian-eigenmap)
    layout -- confirmed directly against IsUMap's own source. The
    "_isumap"-suffixed scripts in this project (run_swiss_roll_isumap.py,
    run_mammoth_isumap.py, run_sphere_isumap.py) exist specifically to test
    how IsUMap's own asymmetric D_asym behaves under our shared
    force-directed machinery, so their init should match IsUMap's own choice
    -- the ONLY thing actually borrowed from UMAP in these scripts should be
    the attractive/repulsive force computation itself, not the init method.

    classical_mds() requires a dense/complete distance matrix (it
    double-centers whole rows/columns), but this project's own D_asym here
    (build_isumap_dist_matrix's reconstruction of data_D, the raw
    pre-t-conorm/pre-Dijkstra neighbourhood distances) is deliberately
    SPARSE (np.inf outside each point's own ~k-NN row) -- see
    build_isumap_dist_matrix's own docstring for why that raw, unmerged
    matrix is what's used for the force computation and for extracting B's
    asymmetry. So this helper builds a SEPARATE, directed-Dijkstra-completed
    dense copy (same recipe as MNIST/embed_MNIST_raw.py's own D_asym
    construction) purely to get a valid input for classical_mds -- it does
    NOT replace the sparse D_asym fed to randers_umap_fit/_knn_weights
    elsewhere, so the force computation and the drift/asymmetry extraction
    are both completely unaffected by this change; only the starting
    position Y_init changes.
    """
    n = D_asym.shape[0]
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    rows, cols = np.nonzero(np.isfinite(D_asym) & (D_asym > 0))
    nbg = csr_matrix((D_asym[rows, cols], (rows, cols)), shape=(n, n))
    D_dense, _ = shortest_path(nbg, method="auto", directed=True, return_predecessors=True)

    # [OURS 2026-09-03] a directed k-NN graph is not guaranteed strongly
    # connected -- some (i,j) pairs can stay unreachable (inf) even after
    # Dijkstra, especially at small n/k or for point clouds with thin
    # regions (observed empirically on mammoth at n=150). classical_mds's
    # double-centering (D**2, then J@D2@J) turns any remaining inf into nan
    # across the WHOLE matrix (inf*0 / inf-inf during centering), not just
    # the unreachable entries -- so this has to be resolved before calling
    # it. Standard MDS practice: replace unreachable pairs with a large-but-
    # finite fallback (here, the graph's own observed diameter), since "far
    # apart" is a reasonable stand-in for "no directed path found" and keeps
    # every entry finite without distorting the reachable geometry.
    finite = np.isfinite(D_dense)
    if not finite.all():
        fallback = D_dense[finite].max() if finite.any() else 1.0
        D_dense = np.where(finite, D_dense, fallback)
        n_unreachable = int((~finite).sum())
        print(f"isumap_style_init: {n_unreachable} directed pair(s) unreachable after "
              f"Dijkstra -- filled with the graph's own max finite distance "
              f"({fallback:.4f}) before classical_mds.")

    return classical_mds(D_dense, d=d, seed=seed)
