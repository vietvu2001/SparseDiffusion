import os
import torch
import numpy as np
from datetime import datetime
import gc
import time
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import itertools
from copy import deepcopy

# Set default data type
torch.set_default_dtype(torch.float32)

# Write loss function
def mc_loss_batch_simul(model, batch, time_array, n, k, time_threshold, p_bad, device):
    """
    Parameters:
    - model: diffusion denoiser
    - batch: data to train model of size [B, n, n]
    - time_array: an array of time points to sample from
    """
    
    # Initialize loss
    loss = 0
    
    # Batch is of size [batch_size, n, n]

    # T1: Use fixed time
    #time_array = torch.ones((batch.shape[0], 1, 1), device=batch.device) * t
    
    # T2: Generate an array of random times, one for each observation in the batch
    #time_array = torch.distributions.Exponential(0.001).sample((batch.shape[0], 1, 1)).to(device)
    #random_indices = torch.randint(low=0, high=len(time_array), size=(batch.shape[0], 1, 1), device=device)
    #random_times = time_array[random_indices]

    # T3: Non-uniform sampling
    T = time_array.size(0)
    B = batch.shape[0]
    
    # 1) build a probs vector so that
    #    sum_{t <= threshold} probs[t] = 0.2
    #    sum_{t > threshold}  probs[t] = 0.8
    mask_small = time_array <= time_threshold        # Boolean mask, shape (T,)
    mask_large = ~mask_small                         # the rest
    
    n_small = mask_small.sum().item()     # number of times <= time_threshold
    n_large = mask_large.sum().item()     # number of times > time_threshold
    
    probs = torch.empty(T, device=time_array.device, dtype=torch.float)
    probs[mask_small] = p_bad / n_small
    probs[mask_large] = (1 - p_bad) / n_large
    # now probs.sum() == 1.0
    
    # 2) sample B indices with replacement
    idx = torch.multinomial(probs, num_samples=B, replacement=True)
    # idx is (B,)
    
    # 3) reshape to your desired (B,1,1) and index
    random_indices = idx.view(B, 1, 1)
    random_times   = time_array[random_indices]  # shape (B,1,1)

    # Create noisy input to the denoiser
    batch_outer = batch[:, :, None] * batch[:, None, :]
    y_t = batch_outer + torch.randn(batch_outer.shape, device=device) * (random_times ** (-1/2))
    y_t = (1/2) * (y_t + y_t.transpose(-1, -2))

    # Get random node embeddings and normalize them
    x = torch.randn((batch.shape[0], n), device=device)
    x = x / x.norm(dim=1, keepdim=True)
    
    # Match dimensions
    x_expand = x.unsqueeze(-1)
    y_t_expand = y_t.unsqueeze(-1)
    
    # Get output from the denoiser
    out = model(x_expand, y_t_expand, random_times)

    # Get loss by rank-1 matrices
    loss = torch.sum((out - batch_outer) ** 2)

    # Return memory
    torch.cuda.empty_cache()
        
    return loss / batch.shape[0]


# Write loss function
def mc_loss_batch_fixed(model, batch, t, n, k, predict, device):
    """
    Parameters:
    - model: diffusion denoiser
    - batch: data to train model of size [B, n, n]
    - time_points: an array of time points to sample from
    - predict: True/False
    """
    
    # Initialize loss
    loss = 0
    
    # Batch is of size [batch_size, n, n]

    # T1: Use fixed time
    time_array = torch.ones((batch.shape[0], 1, 1), device=batch.device) * t
    
    # T2: Generate an array of random times, one for each observation in the batch
    #time_array = torch.distributions.Exponential(0.001).sample((batch.shape[0], 1, 1)).to(device)

    # Create noisy input to the denoiser
    batch_outer = batch[:, :, None] * batch[:, None, :]
    y_t = batch_outer + torch.randn(batch_outer.shape, device=device) * (time_array ** (-1/2))
    y_t = (1/2) * (y_t + y_t.transpose(-1, -2))

    # Get random node embeddings and normalize them
    x = torch.randn((batch.shape[0], n), device=device)
    x = x / x.norm(dim=1, keepdim=True)
    
    # Match dimensions
    x_expand = x.unsqueeze(-1)
    y_t_expand = y_t.unsqueeze(-1)
    
    # Get output from the denoiser
    if not predict:
        out = model(x_expand, y_t_expand, time_array, noise_free=False)
    else:
        out = model.predict(x_expand, y_t_expand, time_array, noise_free=False)

    # Get loss by rank-1 matrices
    loss = torch.sum((out - batch_outer) ** 2)

    # Return memory
    torch.cuda.empty_cache()
        
    return loss / batch.shape[0]


# Technically we can make this function with <mc_loss_batch_fixed> with another Boolean, but there are too many parameters already
def mc_loss_batch_noise_free(model, batch, n, k, predict, device):
    """
    Parameters:
    - model: diffusion denoiser
    - batch: data to train model of size [B, n, n]
    - time_points: an array of time points to sample from
    - predict: True/False
    """
    
    # Initialize loss
    loss = 0
    
    # Batch is of size [batch_size, n, n]

    # This time is null
    time_array = torch.ones((batch.shape[0], 1, 1), device=batch.device)

    # Create noisy input to the denoiser
    batch_outer = batch[:, :, None] * batch[:, None, :]
    y_t = batch_outer
    
    # Get random node embeddings and normalize them
    x = torch.randn((batch.shape[0], n), device=device)
    x = x / x.norm(dim=1, keepdim=True)
    
    # Match dimensions
    x_expand = x.unsqueeze(-1)
    y_t_expand = y_t.unsqueeze(-1)
    
    # Get output from the denoiser
    if not predict:
        out = model(x_expand, y_t_expand, time_array, noise_free=True)
    else:
        out = model.predict(x_expand, y_t_expand, time_array, noise_free=True)

    # Get loss by rank-1 matrices
    loss = torch.sum((out - batch_outer) ** 2)

    # Return memory
    torch.cuda.empty_cache()
        
    return loss / batch.shape[0]