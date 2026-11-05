"""
fdgl_pipeline_lowdim_B.py -- fdgl_pipeline (randers_bridge.py) with ONE change:
the drift B is computed in the low-dimensional embedding, the way it was in
the first commit's randers_umap_fit, instead of being located from a
2n-point (real + virtual) MDS.

Everything else is exactly fdgl_pipeline's own apply step:
    - D_real_asym = compute_dist_matrix(X, randers_field=omega)   (global Randers)
      is the training matrix passed to fdgl_low_dim,
    - D_geo = compute_dist_matrix(X, randers_field=None) is passed too,
    - force model, gravity, virtual neighbour, negative sampling, clipping,
      asymmetry diagnostics: same arguments, same defaults.

What is NOT done: no locate step (no X_virtual, no D_sym_aug, no 2n-point MDS,
no B_located).

Initialization (low-dim):
    Y_init = classical_mds(D_real_asym) on the n real points
             (D_real_asym symmetrized first by 0.5*(D+D^T) unless
             symmetrize_init=False, as the locate step does before its MDS)
    B at epoch 0 = compute_drift(N, mask, k, Y_init)
    B_fixed=False (default): B recomputed every epoch from the current Y
                             (N and the neighbour mask stay fixed).
    B_fixed=True           : B computed once from Y_init and frozen.
N and the mask come from the training matrix D_real_asym, i.e. from
fdgl_low_dim's own _compute_N / knn_mask on the matrix it receives.
"""

import numpy as np

from randers_fdgl import (fdgl_low_dim, classical_mds, _compute_N, compute_drift,
                           knn_mask_from_distance_matrix)
from randers_bridge import compute_dist_matrix, asymmetry_score, reconstruct_rho


def fdgl_pipeline_lowdim_B(X, omega, k=15, emb_k=20, neg=10,
                           epochs=500, clip_delta=0.01, use_gravity=False,
                           gravity_strength=1.0, gravity_neighbor_weight=True,
                           use_virtual_neighbor=False, proj_dim=2, adjacency="knn",
                           snapshot_every=None, ramp=False, seed=0, verbose=True,
                           normalize_drift_by_asymmetry=False,
                           force_model="umap", fr_k=None, negative_sampling=False,
                           isumap_perturb=True,
                           randers_attractive=True, randers_repulsive=False,
                           B_fixed=False, symmetrize_init=True):
    """
    Same arguments as randers_bridge.fdgl_pipeline, minus locate_epochs and
    apply_step (there is no locate step), plus:

    B_fixed : bool, default False.
        False -> live low-dim drift: B starts as compute_drift(N, mask, k,
                 Y_init) and is recomputed from the current Y every epoch.
        True  -> B = compute_drift(N, mask, k, Y_init) computed once, frozen.
    symmetrize_init : bool, default True. classical_mds is run on
        0.5*(D_real_asym + D_real_asym^T) (what the locate step does before
        its MDS). False -> D_real_asym is passed as is.

    Returns
    -------
    dict: Y, B, D_asym, Y_init, asymmetry_score, asymmetry_per_node,
          asymmetry_score_final, asymmetry_per_node_final,
          snapshots (only if snapshot_every given)
    """
    n = X.shape[0]
    limit = 1.0 - clip_delta

    if verbose:
        print("\nTraining matrix: global Randers (directed Dijkstra over Randers "
              "weights built from omega)...")
    D_asym, _, bln_asym = compute_dist_matrix(X, n_neighbors=k,
                                              randers_field=omega, adjacency=adjacency,
                                              return_adjacency=True)

    D_geo, _ = compute_dist_matrix(X, n_neighbors=k,
                                   randers_field=None, adjacency=adjacency,
                                   isumap_perturb=isumap_perturb)

    if verbose:
        print("Init: classical_mds on the n real points only (no virtual points, "
              "no 2n-point system)...")
    D_init = 0.5 * (D_asym + D_asym.T) if symmetrize_init else D_asym
    Y_init = classical_mds(D_init, d=proj_dim, seed=seed)

    B_fixed_arr = None
    if B_fixed:
        knn_mask0 = knn_mask_from_distance_matrix(D_asym, emb_k)
        N0 = _compute_N(D_asym, knn_mask=knn_mask0)
        B_fixed_arr = compute_drift(N0, knn_mask0, emb_k, Y_init, clip_delta)
        if verbose:
            bn0 = np.linalg.norm(B_fixed_arr, axis=1)
            print(f"B computed once from Y_init (frozen): mean||b||={bn0.mean():.4f}  "
                  f"max||b||={bn0.max():.4f}")

    asym_per_node, asym_global = asymmetry_score(D_asym, bln_asym)
    if verbose:
        print(f"asymmetry_score: global={asym_global:.4f}")
        print(f"B mode: {'frozen (computed once from Y_init)' if B_fixed else 'live (recomputed every epoch from the current Y)'}")

    out = fdgl_low_dim(D_asym, n_neighbors=emb_k, n_negative_samples=neg,
                       n_epochs=epochs, use_drift=True, d=proj_dim,
                       B_fixed=B_fixed_arr, Y_init_override=Y_init,
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
    Y, B = out["Y"], out["B"]

    rho_final = reconstruct_rho(Y, B)
    asym_per_node_final, asym_global_final = asymmetry_score(rho_final, bln_asym)

    if verbose:
        bn = np.linalg.norm(B, axis=1)
        print(f"\nextent={Y.max()-Y.min():.2f}  mean||b||={bn.mean():.4f}  "
              f"max||b||={bn.max():.4f}  clipped={(bn >= limit - 1e-9).sum()}/{n}")
        pct = 100.0 * asym_global_final / max(asym_global, 1e-12)
        print(f"asymmetry_score (final embedding): {asym_global_final:.4f}  "
              f"vs target (D_asym) {asym_global:.4f}  -- {pct:.1f}% of target asymmetry preserved")

    result = {"Y": Y, "B": B, "D_asym": D_asym, "Y_init": Y_init,
              "asymmetry_score": asym_global, "asymmetry_per_node": asym_per_node,
              "asymmetry_score_final": asym_global_final,
              "asymmetry_per_node_final": asym_per_node_final}
    if snapshot_every is not None:
        result["snapshots"] = out["snapshots"]
    return result
