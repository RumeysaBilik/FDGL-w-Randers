# FDGL — Force-Directed Graph Layout with a Randers Metric

Force-directed graph layout (FDGL) built on UMAP's optimization scheme, extended with an asymmetric **Randers (Finsler) metric**: each point carries a local drift vector `b_i` that makes the effective distance direction-dependent (`rho(x,y) != rho(y,x)`). Two ways of obtaining that asymmetry are supported:

- **generated** — a Randers vector field `omega` is hand-crafted per dataset and injected into the distance computation (`randers_bridge.py`); asymmetry is synthetic with known ground truth.
- **calculated** — no field is injected; asymmetry emerges from the raw point cloud's own directed k-NN/shortest-path geometry (`isumap_bridge.py`, based on the vendored `isumap/` library). Used for all real-world datasets (MNIST, scRNA, BreastCancer), which have no hand-craftable ground-truth field.

Mathematical derivations (Randers metric, drift bound, `D_asym`/`D_geo`, drift-magnitude normalization) live in `reports/` as PDFs.

## Install

```bash
pip install numpy scipy scikit-learn matplotlib pandas torch tqdm numba pylanczos plotly
```

`torch`/`torchvision` are only needed for the MNIST scripts (`torchvision` is used solely for loading the MNIST dataset in `isumap/data_and_plots.py`).

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
| `randers_bridge.py` | **generated** family: builds `omega`-driven asymmetric distance matrices and runs the located-drift pipeline (`fdgl_pipeline`). |
| `isumap_bridge.py` | **calculated** family: bridges to the vendored `isumap/` library to build `D_asym` from raw point-cloud geometry, no injected field. |
| `isumap/` | Vendored third-party isumap library (`isumap.py`, `distance_graph_generation.py`, `dimension_reduction_schemes.py`, `metric_mds.py`, `data_and_plots.py`) — left unmodified. |
| `main.py` | Single entry point dispatching to the right `run_*.py`/`embed_*.py` by `--dataset`/`--drift`. |
| `run_swiss_roll_generated.py`, `run_mammoth_generated.py`, `run_sphere_radial_generated.py`, `run_sphere_tangential_generated.py` | **generated**-family scripts, one per synthetic dataset/field. |
| `run_swiss_roll_calculated.py`, `run_mammoth_calculated.py`, `run_sphere_calculated.py` | **calculated**-family counterparts for the synthetic datasets. |
| `MNIST/`, `scRNA/`, `BreastCancer/` | Real-world datasets — **calculated** family only (`embed_*.py`, plus dataset-specific diagnostics). |
| `test/` | Standalone diagnostic/sanity-check scripts (see below); none of them modify the core pipeline files. |
| `reports/` | Dated PDFs with the mathematical derivations behind the pipeline. |

## Diagnostics (`test/`)

| Script | Checks |
|---|---|
| `test.py` | Randers-validity (`\|\|b_i\|\| <= 1-delta`), reconstruction stress, direction accuracy on the swiss roll. |
| `asymmetry_k_sweep_generated.py` / `asymmetry_k_sweep_calculated.py` | How asymmetry strength varies with `k`, for the generated / calculated families respectively. |
| `drift_magnitude_test.py` | Tracks `\|\|b_i\|\|` per epoch against its theoretical bound, across all four located-drift datasets. |
| `stability_check.py` | Embedding stability vs. subsample size (Procrustes distance), mirroring the UMAP paper's Fig. 8. |
| `compare_force_models.py` | Compares force models (e.g. Fruchterman-Reingold-style gravity vs. UMAP's own attraction/repulsion). |

## Notes

- Every script follows the same `sys.path` convention: `HERE = os.path.dirname(os.path.abspath(__file__))` (plus `ROOT` for scripts one level down, e.g. in `test/` or `MNIST/`) inserted at the top, so sibling/parent modules resolve regardless of which directory you run from.
- `isumap/` is third-party and intentionally left untouched; all project-specific logic lives in the root-level `randers_*`/`isumap_bridge`/`run_*`/`embed_*` files.
