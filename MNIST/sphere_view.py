#!/usr/bin/env python3
"""
sphere_view.py -- [OURS 2026-09-10] MNIST-only visualization helper, used
by embed_MNIST_raw.py's and embed_MNIST_pca.py's --sphere-view flag.

Deliberately NOT in randers_fdgl.py: this is a pure post-training
VISUALIZATION transform (map a 2D embedding onto a unit sphere, same
map-on-paper vs. map-on-a-globe idea), not part of the embedding
algorithm itself, and currently only used by the two MNIST scripts -- so
it lives here rather than in the shared, dataset-agnostic engine.
"""

import numpy as np


def stereographic_project(Y2d: np.ndarray, center: np.ndarray = None,
                           scale: float = None):
    """
    Map a 2D embedding onto the unit sphere via inverse stereographic
    projection -- the same map-on-paper vs. map-on-a-globe transform,
    applied purely for VISUALIZATION after training. Does not touch
    fdgl_low_dim or the force computation in any way; Y2d is whatever
    a normal 2D run already produced.

        d^2 = x^2 + y^2                     (after centering + scaling)
        X = 2x / (d^2+1)
        Y = 2y / (d^2+1)
        Z = (d^2-1) / (d^2+1)

    Points near the plane's origin land near the sphere's south pole
    (Z=-1); points far from the origin land near the north pole (Z=+1) --
    conformal (angle-preserving), so local neighbourhood shapes are not
    distorted, only where they sit on the sphere.

    Centering (subtract center, default Y2d's own mean) and scaling
    (divide by scale, default the median distance from that center) are
    necessary before applying the raw formula above: an un-rescaled UMAP
    embedding's coordinates are typically >> 1, which would crush nearly
    every point up against the north pole. Scaling by the median radius
    instead spreads the typical point out near the equator, using the
    sphere's whole surface.

    Parameters
    ----------
    Y2d    : (n, 2) embedding coordinates
    center : (2,) or None -- pass the SAME center used for the main point
             cloud when projecting a second batch of points (e.g. drift
             arrow heads) so they land in the same frame; None re-derives
             it from Y2d itself.
    scale  : float or None -- same idea as center, for the radius scale.

    Returns
    -------
    Y3d    : (n, 3) points on the unit sphere
    center : (2,) the center actually used (re-use for consistent calls)
    scale  : float the scale actually used (re-use for consistent calls)
    """
    if center is None:
        center = Y2d.mean(axis=0)
    Yc = Y2d - center
    if scale is None:
        r = np.linalg.norm(Yc, axis=1)
        med = float(np.median(r)) if r.size else 0.0
        scale = med if med > 0 else 1.0
    Ys = Yc / scale
    d2 = (Ys ** 2).sum(axis=1)
    denom = d2 + 1.0
    X = 2.0 * Ys[:, 0] / denom
    Yout = 2.0 * Ys[:, 1] / denom
    Z = (d2 - 1.0) / denom
    return np.stack([X, Yout, Z], axis=1), center, scale
