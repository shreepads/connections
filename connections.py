# Copyright (c) 2024 Shreepad Shukla
# SPDX-License-Identifier: MIT

# Based on the guide at https://medium.com/predict/a-simple-comprehensive-guide-to-running-large-language-models-locally-on-cpu-and-or-gpu-using-c0c2a8483eee and
# the document at https://docs.mistral.ai/capabilities/embeddings/
# Input strings from NYT's Connections game (https://www.nytimes.com/games/connections) published on 20th Apr 2024

# Imports
from huggingface_hub import hf_hub_download
from llama_cpp import Llama                     # Needs pip install llama-cpp-python==0.2.55 for create_embedding
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import HDBSCAN

# Orca retraining of Mistral 7B, runs in 8GB
model_name = "TheBloke/Mistral-7B-OpenOrca-GGUF"
model_file = "mistral-7b-openorca.Q5_K_M.gguf"

# Download the model
model_path = hf_hub_download(model_name, filename=model_file)

# Setup model parameters
model_kwargs = {
  "n_ctx"        : 4096,  # Context length to use
  "n_threads"    : 4,     # Number of CPU threads to use
  "n_gpu_layers" : 0,     # Number of model layers to offload to GPU. Set to 0 as only using CPU
  "embedding"    : True,  # Include embeddings for classification
}

## Instantiate model from downloaded file
llm = Llama(model_path=model_path, **model_kwargs)

# Setup input list of strings

# Reduced set of 2
# input_strs = ["Bunk", "Crock", "Hogwash", "Baton", "Hammer", "Hurdle"]
# input_strs = ["bunk", "crock", "hogwash", "baton", "hammer", "hurdle"]

# Full set of 2
# input_strs = ["bunk", "crock", "hogwash", "horsefeathers", "baton", "hammer", "hurdle", "pole"]
# input_strs = ["bunk", "baton", "horsefeathers", "crock", "hammer", "hogwash", "hurdle", "pole"]

# Full set of 2 with labels
# input_strs = ["bunk", "baton", "horsefeathers", "crock", "balderdash", "hammer", "hogwash", "hurdle", "pole", "track and field equipment"]
# clusters = 2

# Full set of 3
#input_strs = ["bunk", "crock", "hogwash", "horsefeathers", "baton", "hammer", "hurdle", "pole", "goatee", "horns", "pitchfork", "tail"]
#clusters = 3

# Full set of 4, the whole problem
input_strs = ["bunk", "crock", "hogwash", "horsefeathers", "baton", "hammer", "hurdle", "pole", "goatee", "horns", "pitchfork", "tail", "bend", "bowline", "hitch", "sheepshank"]
clusters = 4


# Generate embeddings
#embeddings = llm.create_embedding(input_strs)       # doesn't work, generates NaNs for more than 3 strings

embeddings = [llm.create_embedding(i)["data"][0]["embedding"] for i in input_strs]    
vecs = []
    
# List embeddings data/types and convert to numpy arrays
for embedding in embeddings:
    # print(type(embedding))
    # print(type(embedding)[0])
    # print(len(embedding))
    vecs.append(np.array(embedding))
    
# Calculate Euclidean distance
#dist_01 = np.linalg.norm(vecs[0] - vecs[1])
#dist_02 = np.linalg.norm(vecs[0] - vecs[2])
#dist_12 = np.linalg.norm(vecs[1] - vecs[2])
#dist_03 = np.linalg.norm(vecs[0] - vecs[3])
#dist_34 = np.linalg.norm(vecs[3] - vecs[4])

# Cluster by KMeans
kmeans = KMeans(n_clusters=clusters, n_init=10)
kmeans.fit(vecs)

# Cluster by Spectral Clustering
spectral = SpectralClustering(n_clusters=clusters, assign_labels='discretize', random_state=0)
spectral.fit(vecs)

# Cluster by HDBSCAN
# hdb = HDBSCAN(min_cluster_size=4, max_cluster_size=4)      # Fails to cluster
# hdb = HDBSCAN(min_cluster_size=4)                          # Fails to cluster
hdb = HDBSCAN(min_cluster_size=3)                            # Partly fails to cluster
hdb.fit(vecs) 


print("KMeans clustering")
print(input_strs)
print(kmeans.labels_)

print("Centers: ", len(kmeans.cluster_centers_) , ", iterations: ", kmeans.n_iter_ , ", features in: ", kmeans.n_features_in_)

print()

print("Spectral clustering")
print(input_strs)
print(spectral.labels_)

print()

print("HDBSCAN clustering")
print(input_strs)
print(hdb.labels_)


