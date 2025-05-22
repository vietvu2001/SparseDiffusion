import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def sample_data(N, n, k, device):
    """
    Parameters:
    - N: number of samples
    - n: dimension of each sample
    - k: number of nonzero elements in each non-zero sample
    
    Returns:
    - Tensor of shape (N, n)
    """
    
    # Decide for each row whether it's zero or non-zero (Bernoulli with p=0.5)
    is_nonzero_row = 1 * (torch.rand(N, device=device) < 0.5)  # boolean mask
    #is_nonzero_row = torch.ones(N, device=device)

    # Preallocate the output tensor
    sampled_vectors = torch.zeros(N, n, device=device)

    # Indices of the non-zero rows
    nonzero_indices = torch.nonzero(is_nonzero_row).squeeze()

    if nonzero_indices.numel() > 0:
        # Number of non-zero rows
        N_nonzero = nonzero_indices.shape[0]

        # Step 1: Generate k random indices per selected row
        indices = torch.stack([
            torch.randperm(n, device=device)[:k].sort().values for _ in range(N_nonzero)
        ])

        # Step 2: Create binary ±1 vectors
        signs = (torch.randint(0, 2, (N_nonzero, k), device=device) * 2 - 1).float()  # random ±1

        # Step 3: Scatter into the correct rows
        temp = torch.zeros(N_nonzero, n, device=device)
        temp.scatter_(1, indices, signs)
        sampled_vectors[nonzero_indices] = temp

        # Step 4: Normalize
        sampled_vectors[nonzero_indices] /= k ** 0.5

    return sampled_vectors


# Create a custom dataset class
class SubmatrixDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __len__(self):
        return self.data_tensor.shape[0]

    def __getitem__(self, i):
        # Return a single sample (in this case, a 2D tensor of shape [n, n])
        return self.data_tensor[i]