#!/usr/bin/env python3
"""
mnist_loader.py -- self-contained MNIST loading, decoupled from the vendored
isumap/ library.

[OURS 2026-09-17] Previously embed_MNIST_pca.py/embed_MNIST_raw.py imported
load_MNIST from isumap/data_and_plots.py. That file works fine for loading
MNIST (sklearn.datasets.fetch_openml under the hood, cached to a .pkl), but
it also unconditionally does `import torchvision` at module level -- needed
ONLY by data_and_plots.py's own load_CIFAR_10 (unused anywhere in this
project), never by load_MNIST itself. That one unrelated import forced every
MNIST script to have torchvision installed just to load MNIST. This file
copies load_MNIST's own logic verbatim (no CIFAR10/torch_dataset branch,
since nothing here ever used it) so the MNIST scripts no longer depend on
isumap/ at all.
"""

import os
import pickle
import random

import numpy as np
from sklearn import datasets


def load_and_store_mnist(N, filename="mnist_784", filetype=".pkl",
                          normalize=True, datasetPath="../Dataset_files/"):
    """
    Downloads (via sklearn.datasets.fetch_openml) and caches `filename` as
    datasetPath/filename+filetype on first use, then loads from that cache
    on every subsequent call -- same behaviour as isumap/data_and_plots.py's
    load_and_store_data_file, minus the torch_dataset/CIFAR10 branch (never
    used in this project).
    """
    if not os.path.exists(datasetPath + filename + filetype):
        print(f"\nDownloading '{filename}' data.")
        os.makedirs(datasetPath, exist_ok=True)
        ddata = datasets.fetch_openml(filename)
        data = np.array(ddata["data"])
        labels = np.array(ddata["target"])
        with open(datasetPath + filename + filetype, "wb") as f:
            pickle.dump((data, labels), f)
        print(f"Download successful. The files are stored in "
              f"'{datasetPath}{filename}{filetype}' and are directly loaded "
              f"from there in case you run this script a second time.")
    print(f"\nLoading '{filename}' data from file")
    with open(datasetPath + filename + filetype, "rb") as f:
        data, labels = pickle.load(f)
    print("Selecting subset of N = ", N)
    indices = random.sample(range(len(data)), N)
    data = np.array(data[indices], dtype=np.float32)
    if normalize:
        data = data / np.max(data)
    labels = np.array(labels[indices], dtype=np.int64)
    return data, labels


def load_MNIST(N, datasetPath="../Dataset_files/"):
    return load_and_store_mnist(N, "mnist_784", datasetPath=datasetPath)
