# FDGL — Force-Directed Graph Layout with a Randers Metric

Force-directed graph layout (FDGL) built on UMAP's optimization scheme, extended with an asymmetric **Randers (Finsler) metric**: each point carries a local drift vector `b_i` that makes the effective distance direction-dependent (`rho(x,y) != rho(y,x)`). Two ways of obtaining that asymmetry are supported:

- **generated** — a Randers vector field `omega` is hand-crafted per dataset and injected into the distance computation (`randers_bridge.py`); asymmetry is synthetic with known ground truth.
- **calculated** — no field is hand-crafted; instead, `omega` is *derived* from the raw point cloud's own directed k-NN/shortest-path asymmetry (`randers_bridge.py`'s `compute_highdim_drift`, in the ambient/high-dimensional space), and that derived `(X, omega)` pair is fed into the SAME `fdgl_pipeline` the **generated** family uses. Used for all real-world datasets (MNIST, scRNA, BreastCancer) and the three synthetic datasets' own calculated variant, none of which have a hand-craftable ground-truth field. [OURS 2026-09-17] No longer touches isumap in any way -- the earlier isumap-based construction (`isumap_bridge.py`, the vendored `isumap/` library) has been removed entirely from this repo.

Mathematical derivations (Randers metric, drift bound, `D_asym`/`D_geo`, drift-magnitude normalization) live in `reports/` as PDFs.

## Install

```bash
pip install numpy scipy scikit-learn matplotlib pandas tqdm numba pylanczos plotly
```

[OURS 2026-09-17] `torch`/`torchvision` dropped from this list -- nothing in this repo uses either anymore. They were only ever pulled in by the vendored `isumap/` library (via its own unrelated `load_CIFAR_10`), which has been removed entirely, along with `isumap_bridge.py`. `MNIST/embed_MNIST_pca.py`/`embed_MNIST_raw.py` load MNIST via their own `MNIST/mnist_loader.py`, and `compute_highdim_drift` (used by every **calculated**-family script) lives in `randers_bridge.py`.

## Quick start

`main.py` is a thin dispatcher over the individual `run_*.py`/`embed_*.py` scripts — pick a dataset and a drift mechanism, and it forwards the rest of the arguments unchanged:

```bash
python3 main.py --dataset swiss_roll --drift generated --n 1000
python3 main.py --dataset mammoth    --drift calculated
```

Each underlying script also runs standalone, e.g.:

```bash
python3 run_swiss_roll_generated.py --n 1000 --k 15 --epochs 500
```

Output is an `.npz` (embedding + metadata) and a `.png` plot.

## Repository structure

| Path | Contents |
|---|---|
| `randers_fdgl.py` | Core UMAP-style optimizer extended with the Randers drift (`compute_drift`, `_compute_N`, `fdgl_low_dim`, `procrustes_align`, ...). |
| `randers_bridge.py` | Both families' shared core: `compute_dist_matrix` (builds `D_asym` from `(X, omega)`, `omega=None` for a plain geodesic), `fdgl_pipeline` (the locate+apply located-drift pipeline), and `compute_highdim_drift` (moved here 2026-09-17 from the now-removed `isumap_bridge.py`) -- the **calculated** family's own `omega`-derivation: `compute_highdim_drift(X, D_asym, k, clip_delta)` derives `omega` from an ambient-space `D_asym`'s own asymmetry, so `(X, omega)` can be fed straight into `fdgl_pipeline`, same as the **generated** family's hand-crafted `omega`. |
| `main.py` | Single entry point dispatching to the right `run_*.py`/`embed_*.py` by `--dataset`/`--drift`. |
| `run_swiss_roll_generated.py`, `run_mammoth_generated.py`, `run_sphere_radial_generated.py`, `run_sphere_tangential_generated.py` | **generated**-family scripts, one per synthetic dataset/field. |
| `run_swiss_roll_calculated.py`, `run_mammoth_calculated.py`, `run_sphere_calculated.py` | **calculated**-family counterparts for the synthetic datasets — `compute_highdim_drift` + `fdgl_pipeline`, same pipeline as **generated**, just with a derived instead of hand-crafted `omega`. |
| `MNIST/`, `scRNA/`, `BreastCancer/` | Real-world datasets — **calculated** family only (`embed_*.py`: `embed_MNIST_pca.py`, `embed_MNIST_raw.py`, `embed_scRNA.py`, `embed_BreastCancer.py`, all on the same `compute_highdim_drift` + `fdgl_pipeline` mechanism as the synthetic-dataset scripts above). `MNIST/mnist_loader.py` holds a self-contained, torchvision-free `load_MNIST` (see Install above). |
| `test/` | Standalone diagnostic/sanity-check scripts (see below); none of them modify the core pipeline files. |
| `reports/` | Dated PDFs with the mathematical derivations behind the pipeline. |

## Diagnostics (`test/`)

| Script | Checks |
|---|---|
| `test.py` | Randers-validity (`\|\|b_i\|\| <= 1-delta`), reconstruction stress, direction accuracy on the swiss roll. |
| `asymmetry_k_sweep_generated.py` | How asymmetry strength varies with `k`, for the generated family. [OURS 2026-09-17] Its `_calculated` counterpart, which swept the older isumap-based `D_asym` construction, has been removed along with `isumap_bridge.py`/`isumap/`. |
| `drift_magnitude_test.py` | Tracks `\|\|b_i\|\|` per epoch against its theoretical bound, across all four located-drift datasets. |
| `stability_check.py` | Embedding stability vs. subsample size (Procrustes distance), mirroring the UMAP paper's Fig. 8. |
| `compare_force_models.py` | Compares force models (e.g. Fruchterman-Reingold-style gravity vs. UMAP's own attraction/repulsion). |

## Notes

- Every script follows the same `sys.path` convention: `HERE = os.path.dirname(os.path.abspath(__file__))` (plus `ROOT` for scripts one level down, e.g. in `test/` or `MNIST/`) inserted at the top, so sibling/parent modules resolve regardless of which directory you run from.
- [OURS 2026-09-17] The vendored `isumap/` library and `isumap_bridge.py` have been removed entirely -- every script in this repo now goes through `randers_bridge.py`'s own `compute_dist_matrix`/`fdgl_pipeline`/`compute_highdim_drift`, no third-party asymmetric-distance code involved anywhere.
