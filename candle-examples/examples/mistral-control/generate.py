import torch
from safetensors import safe_open
from safetensors.torch import save_file

import numpy as np
from sklearn.decomposition import PCA
import torch

path = "last_hidden_states.safetensors"
print(f"Loading tensors from {path}")

tensors = {}
with safe_open(path, framework="pt", device="cpu") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)


rows, layers, elems = tensors["last_hidden_states"].shape

# split the data into positive and negative examples
pos = tensors["last_hidden_states"][: int(rows / 2), :, :]
neg = tensors["last_hidden_states"][int(rows / 2) :, :, :]

# compute the difference
diff = (pos - neg).to(torch.float32)

# compute the PCA for each layer
count = 0
all_directions = torch.zeros((diff.shape[1], diff.shape[2]), dtype=torch.float32)
for i in range(diff.shape[1]):
    data = diff[:, i, :].numpy()
    print(f"Computing PCA for layer {i} of {diff.shape[1]}")
    pca_model = PCA(n_components=1, whiten=False).fit(data)
    directions = pca_model.components_.astype(np.float32).squeeze(axis=0)
    all_directions[i] = torch.from_numpy(directions)
    count += 1


print(f"Computed {count} PCA directions")
path = "directions.safetensors"

# save as safetensors file
tensors = {"all_directions": all_directions}
save_file(tensors, path)

print(f"Saved all_directions to {path}")
