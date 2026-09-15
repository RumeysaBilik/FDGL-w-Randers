#!/usr/bin/env python3
"""
isumap_bridge.py -- shared D_asym-construction + init infrastructure for
the isumap ("calculated") family: run_swiss_roll_calculated.py,
run_mammoth_calculated.py, run_sphere_calculated.py, asymmetry_k_sweep_calculated.py.

Counterpart of randers_bridge.py, but not interchangeable: randers_bridge.py
builds D_asym from a Randers vector field (X, omega); this module builds
D_asym purely from distance_graph_generation()'s own directed k-NN/star-graph
structure -- no field, no omega, ever.
"""

import os
import sys

# vendored isumap library lives in isumap/, not flat in this directory
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "isumap"))

import numpy as np

from distance_graph_generation import find_nn, normalization, comp_graph, canonical_dist
from randers_fdgl import classical_mds


def build_isumap_dist_matrix(X, k=20, verbose=True):
    """
    Builds data_D via find_nn -> normalization -> comp_graph directly
    (the steps distance_graph_generation() itself runs to produce data_D),
    skipping its t-conorm/Dijkstra passes since nothing here uses them.

    """
    n = X.shape[0]
    knn_inds, knn_distances = find_nn(X, k)
    knn_distances = normalization(knn_distances, normalize=True, distBeyondNN=True)
    data_D = comp_graph(knn_inds, knn_distances, X, canonical_dist,
                         epm=True, directedDistances=False)
    if verbose:
        print(f"build_isumap_dist_matrix: {n} points, k={k}, "
              f"{len(data_D)} raw (i,j,k) entries (t-conorm/Dijkstra skipped -- unused by this project)")
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0.0)
    for key, value in data_D.items():
        i, j, k_ = key
        D[i, j] = value
    return D


def isumap_style_init(D_asym, d=2, seed=0):
    """
    classical_mds() needs a dense distance matrix, but D_asym here
    is deliberately sparse. So this builds a SEPARATE, Dijkstra-completed
    dense copy purely for the init
    """
    n = D_asym.shape[0]
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    rows, cols = np.nonzero(np.isfinite(D_asym) & (D_asym > 0))
    nbg = csr_matrix((D_asym[rows, cols], (rows, cols)), shape=(n, n))
    D_dense, _ = shortest_path(nbg, method="auto", directed=True, return_predecessors=True)

    finite = np.isfinite(D_dense)
    if not finite.all():
        fallback = D_dense[finite].max() if finite.any() else 1.0
        D_dense = np.where(finite, D_dense, fallback)
        n_unreachable = int((~finite).sum())
        print(f"isumap_style_init: {n_unreachable} directed pair(s) unreachable after "
              f"Dijkstra -- filled with the graph's own max finite distance "
              f"({fallback:.4f}) before classical_mds.")

    return classical_mds(D_dense, d=d, seed=seed)
