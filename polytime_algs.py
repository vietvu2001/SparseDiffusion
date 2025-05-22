import torch
import math

# Thresholded eigenvector algorithm
def optimal_polynomial_estimator(Y, k):
    """
    Optimal polynomial estimator for rank-1 sparse recovery:
      1. Symmetrize the input: (Y + Y.T)/2
      2. Compute top eigenvector of the symmetric matrix
      3. Keep the k entries of largest magnitude in that eigenvector,
         set them to ±1/√k based on their sign, zero out others
      4. Reconstruct the rank-1 sparse matrix u u^T

    Args:
        Y (torch.Tensor): input matrix of shape (..., n, n)
        k (int): sparsity level
        t (real): time point
    Returns:
        torch.Tensor: estimated rank-1 sparse matrix of shape (..., n, n)
    """
    # Symmetrize
    Y_sym = 0.5 * (Y + Y.transpose(-1, -2))
    # Eigen-decomposition
    e_vals, e_vecs = torch.linalg.eigh(Y_sym)
    # Extract top eigenvector (last column)
    u = e_vecs[..., :, -1]  # shape (..., n)
    # Enforce k-sparsity
    abs_u = u.abs()
    topk = torch.topk(abs_u, k, dim=-1)
    topk_idx = topk.indices  # shape (..., k)
    mask = torch.zeros_like(u)
    mask.scatter_(dim=-1, index=topk_idx, value=1.0)
    # Build sparse vector ±1/sqrt(k)
    u_sparse = u.sign() * mask / math.sqrt(k)
    # Reconstruct rank-1 matrix
    # Unsqueeze to align dims: u_sparse[..., i] u_sparse[..., j]
    X_hat = u_sparse.unsqueeze(-1) * u_sparse.unsqueeze(-2)

    # Conduct hypothesis test
    C = torch.bmm(Y_sym, X_hat)
    traces = torch.einsum('bii->b', C)
    test_vector = 1.0 * (traces >= 0.8)
    X_hat = X_hat * test_vector.view(X_hat.shape[0], 1, 1)
    
    return X_hat


# Power iteration approximation
def power_iteration_estimator(Y, k, num_iters=10):
    """
    Optimal polynomial estimator for rank-1 sparse recovery:
      1. Symmetrize the input: (Y + Y.T)/2
      2. Compute top eigenvector of the symmetric matrix
      3. Keep the k entries of largest magnitude in that eigenvector,
         set them to ±1/√k based on their sign, zero out others
      4. Reconstruct the rank-1 sparse matrix u u^T

    Args:
        Y (torch.Tensor): input matrix of shape (..., n, n)
        k (int): sparsity level
        t (real): time point
    Returns:
        torch.Tensor: estimated rank-1 sparse matrix of shape (..., n, n)
    """
    # Symmetrize
    Y_sym = 0.5 * (Y + Y.transpose(-1, -2))

    # Initialize normal vectors for output and normalize appropriately
    u_current = torch.randn((Y_sym.shape[0], Y_sym.shape[1]), device=device)
    u_current = u_current / u_current.norm(dim=1, keepdim=True)

    # Power-iteration loop
    for _ in range(num_iters):
        u_current = torch.bmm(Y_sym, u_current.unsqueeze(2))
        u_current = u_current.squeeze(2)

        """
        # Truncate
        abs_u = u.abs()
        topk = torch.topk(abs_u, k, dim=-1)
        topk_idx = topk.indices  # shape (..., k)
        mask = torch.zeros_like(u)
        mask.scatter_(dim=-1, index=topk_idx, value=1.0)

        # Update
        u_current = u.sign() * mask / math.sqrt(k)
        """

    # Truncate
    abs_u = u_current.abs()
    topk = torch.topk(abs_u, k, dim=-1)
    topk_idx = topk.indices  # shape (..., k)
    mask = torch.zeros_like(u_current)
    mask.scatter_(dim=-1, index=topk_idx, value=1.0)

    # Update
    u_current = u_current.sign() * mask / math.sqrt(k)
        
    # Reconstruct rank-1 matrix
    # Unsqueeze to align dims: u_sparse[..., i] u_sparse[..., j]
    X_hat = u_current.unsqueeze(-1) * u_current.unsqueeze(-2)

    # Conduct hypothesis test
    if t is not None:
        C = torch.bmm(Y_sym, X_hat)
        traces = torch.einsum('bii->b', C)
        test_vector = 1.0 * (traces >= 0.7)
        X_hat = X_hat * test_vector.view(X_hat.shape[0], 1, 1)
    
    return X_hat


# Evaluate polytime algs
def evaluate_estimator(fn, dataloader, t, n, k, device, num_iters=None):
    loss = 0

    # Set batch counter
    counter = 0
    
    for batch in dataloader: 
        time_array = torch.ones((batch.shape[0], 1, 1), device=batch.device) * t
    
        # Create noisy input to the denoiser
        batch_outer = batch[:, :, None] * batch[:, None, :]
        y_t = batch_outer + torch.randn(batch_outer.shape, device=device) * (time_array ** (-1/2))
    
        # Get output from the denoiser
        if not num_iters:
            out = fn(y_t, k)

        else:
            out = fn(y_t, k, num_iters=num_iters)
    
        # Get loss by rank-1 matrices
        loss += torch.sum((out - batch_outer) ** 2)

        # Print statement
        print("Batch {} finished!".format(counter))

        # Update counter
        counter += 1

    return loss / len(dataloader.dataset)