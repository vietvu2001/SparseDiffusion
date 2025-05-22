import torch

def generation(model, num_samples, n, time_step, terminal_time, device):
    # Create current values
    y = torch.zeros(num_samples, n, n, device=device)
    diff_sample = torch.zeros(num_samples, n, n, device=device)
    t = 0

    with torch.no_grad():
        while t <= terminal_time:
            if t == 0:
               t += time_step
    
            else:
                # Update y
                y = y + time_step * diff_sample + torch.randn_like(y, device=device) * (time_step ** (1/2))
    
                # Symmetrize y as input
                y_symm = (1/2) * (y + y.transpose(-1, -2))
                y_symm_expand = y_symm.unsqueeze(-1)
    
                # Update diffusion sample
                # Generate random node embedding
                x_init = torch.randn((num_samples, n), device=device)
                x_init = x_init / x_init.norm(dim=1, keepdim=True)
                x_init_expand = x_init.unsqueeze(-1)
    
                # Create time array
                time_array = torch.ones((num_samples, 1, 1), device=device) * t
    
                # Evaluate the diffusion sample
                diff_sample = model.predict(x_init_expand, y_symm_expand / t, time_array)
    
                # Increment t
                t += time_step

            # Print out t
            print("Time {} updated".format(t))

    return diff_sample