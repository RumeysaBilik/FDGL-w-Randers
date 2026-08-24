import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
from data_and_plots import load_MNIST
from distance_graph_generation import distance_graph_generation

save_dir_isumap = './MNIST/'
os.makedirs(save_dir_isumap, exist_ok=True)

# MNIST -- fetched automatically via sklearn (fetch_openml), cached locally
# by load_MNIST/load_and_store_data_file. No CSV files needed.
X, y = load_MNIST(5000, datasetPath='./Dataset_files/')

print(type(X))
print(X.shape)


# Run isumap
isumap_dist = distance_graph_generation(X, k=30,
                                    normalize=True, distBeyondNN=True, verbose=True,
                                    dataIsDistMatrix=False, dataIsGeodesicDistMatrix=False, saveDistMatrix=False,
                                    )


asymm_distance = isumap_dist[0]



#for key, value in asy_distance.items():
#    print("key:", key)
 #   print("type:", type(value))
  #  
  #  if hasattr(value, "shape"):
  #      print("shape:", value.shape)
        
n = X.shape[0]

asymm_matrix = np.zeros((n, n))

for key, value in asymm_distance.items():
    i, j, k = key
    asymm_matrix[i, j] = value       
print(asymm_matrix.shape)
print(np.allclose(asymm_matrix, asymm_matrix.T)) #should be false

# save asymetric distance matrix computed for MNIST dataset
np.save(os.path.join(save_dir_isumap, 'asymm_matrix.npy'), asymm_matrix)

# [OURS 2026-08-07] also save the labels for these SAME 5000 rows, so a
# separate embedding script can colour by digit without re-sampling
# (load_MNIST's random.sample has no fixed seed -- a second call would
# pick a different subset and misalign with asymm_matrix.npy's rows).
np.save(os.path.join(save_dir_isumap, 'labels.npy'), y)
