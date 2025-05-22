import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKStraightThrough(nn.Module):
    """
    Top-k sparsification for vectors.
    """
    def __init__(self, k, temp=0.1):
        super().__init__()
        self.k = k
        self.temp = temp

    def forward(self, v):
        abs_v = v.abs()
        mask_prob = torch.softmax(abs_v / self.temp, dim=1)
        topk_vals, topk_indices = torch.topk(mask_prob, self.k, dim=1)
        mask_hard = torch.zeros_like(mask_prob).scatter_(1, topk_indices, 1.0)      # [B, N]
        return mask_hard + (mask_prob - mask_prob.detach())


class TestMPNN_3(nn.Module):
    def __init__(self, k, hidden_dim_1=32, hidden_dim_2=16, hidden_dim_3=4, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        # Create distinct message and update MLPs for each layer
        self.msg_mlps = nn.ModuleList()
        self.upd_mlps = nn.ModuleList()
        for _ in range(num_layers):
            self.msg_mlps.append(nn.Sequential(
                nn.Linear(3, hidden_dim_1),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim_1, 1),
            ))
            self.upd_mlps.append(nn.Sequential(
                nn.Linear(2, hidden_dim_2),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim_2, 1),
            ))
        
        # Sparsifier remains the same
        self.sparsifier = TopKStraightThrough(k, temp=0.2)

        # Parameter that governs how much node embeddings are damped
        #self.alpha = alpha
        self.logit_alpha = nn.ParameterList(
            nn.Parameter(torch.zeros(1)) for _ in range(self.num_layers)
        )
        self.alpha_max = 0.85

        # Number of power iterations at initialization
        self.soft_iters = 3

        # Scale for time embeddings
        self.log_scale = nn.Parameter(torch.zeros(1))

        # Tiny decoder
        self.tiny_decoder = nn.Sequential(
            nn.Linear(2, hidden_dim_3),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim_3, 1),
        )

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Initialize weight
                nn.init.xavier_uniform_(m.weight, gain=0.2)
                # now scale the bias
                if m.bias is not None:
                    m.bias.data.mul_(0.2)


        # Coefficients
        self.readout_coefs = nn.Parameter(torch.ones(1))

    
    def forward(self, X, E, time_array, noise_free=False):
        """
        X: [B, N, 1]    initial node features
        E: [B, N, N, 1] dense edge weights
        time_array: [B, 1, 1]
        """
        B, N, _ = X.shape
        v = X  # [B, N, 1]

        # Soft power-iteration
        for _ in range(self.soft_iters):
            v = torch.bmm(E.squeeze(-1), v)
            v = v.squeeze(-1)

            # Do not divide by zero in case E is close to 0 (x = 0 and large t)
            v = v / v.norm(dim=1, keepdim=True).clamp(min=1e-5)
            v = v.unsqueeze(-1)

        for l in range(self.num_layers):
            # message passing for layer l
            xi = v.unsqueeze(2).expand(B, N, N, 1)  # [B, N, N, 1]
            xj = v.unsqueeze(1).expand(B, N, N, 1)  # [B, N, N, 1]

            # message
            m_in = torch.cat([xi, xj, E], dim=-1)   # [B, N, N, 3]
            m_ij = self.msg_mlps[l](m_in)           # [B, N, N, 1]

            # aggregate (mean)
            agg = m_ij.mean(dim=2)    # [B, N, 1]

            # Get new output
            upd_in = torch.cat([v, agg], dim=-1)    # [B, N, 2]
            v_new = self.upd_mlps[l](upd_in)            # [B, N, 1]

            # Update
            alpha_l = torch.sigmoid(self.logit_alpha[l]) * self.alpha_max
            v = alpha_l * v + (1 - alpha_l) * v_new

        # Create time embeddings
        scale = torch.exp(self.log_scale)
        time_emb = torch.tanh(scale * time_array / N) if not noise_free else torch.ones((X.shape[0], 1, 1), device=X.device)
        time_emb_rep = time_emb.expand(-1, N, -1)

        # Concatenate with <v>
        v_concat = torch.cat([v, time_emb_rep], dim=-1)

        # Get v
        #v = self.tiny_decoder(v_concat)
        v = self.readout_coefs[0] * (v * time_emb_rep) + self.tiny_decoder(v_concat)

        # final sparsify + normalize
        scores = v.squeeze(-1)                     # [B, N]
        mask = self.sparsifier(scores)             # [B, N]
        scores = scores * mask                     # [B, N]
        #scores = scores / (scores.norm(p=2, dim=1, keepdim=True).clamp(min=1))

        # build rank-1 output
        X_hat = torch.einsum('bi,bj->bij', scores, scores)  # [B, N, N]

        return X_hat

    
    @torch.no_grad()
    def predict(self, X, E, time_array, noise_free=False):
        """
        X: [B, N, 1]    initial node features
        E: [B, N, N, 1] dense edge weights
        time_array: [B, 1, 1]
        """
        B, N, _ = X.shape
        v = X  # [B, N, 1]

        # Soft power-iteration
        for _ in range(self.soft_iters):
            v = torch.bmm(E.squeeze(-1), v)
            v = v.squeeze(-1)
            v = v / v.norm(dim=1, keepdim=True)
            v = v.unsqueeze(-1)

        for l in range(self.num_layers):
            # message passing for layer l
            xi = v.unsqueeze(2).expand(B, N, N, 1)  # [B, N, N, 1]
            xj = v.unsqueeze(1).expand(B, N, N, 1)  # [B, N, N, 1]

            # message
            m_in = torch.cat([xi, xj, E], dim=-1)   # [B, N, N, 3]
            m_ij = self.msg_mlps[l](m_in)           # [B, N, N, 1]

            # aggregate (mean)
            agg = m_ij.mean(dim=2)    # [B, N, 1]

            # Get new output
            upd_in = torch.cat([v, agg], dim=-1)    # [B, N, 2]
            v_new = self.upd_mlps[l](upd_in)            # [B, N, 1]

            # Update
            alpha_l = torch.sigmoid(self.logit_alpha[l]) * self.alpha_max
            v = alpha_l * v + (1 - alpha_l) * v_new

        # Create time embeddings
        scale = torch.exp(self.log_scale)
        time_emb = torch.tanh(scale * time_array / N) if not noise_free else torch.ones((X.shape[0], 1, 1), device=X.device)
        time_emb_rep = time_emb.expand(-1, N, -1)

        # Concatenate with <v>
        v_concat = torch.cat([v, time_emb_rep], dim=-1)

        # Get v
        #v = self.tiny_decoder(v_concat)
        v = self.readout_coefs[0] * (v * time_emb_rep) + self.tiny_decoder(v_concat)

        # final sparsify + normalize
        scores = v.squeeze(-1)                     # [B, N]
        mask = self.sparsifier(scores)             # [B, N]
        scores = scores * mask                     # [B, N]

        # Have to project point onto unit sphere when testing
        scores = scores / (scores.norm(p=2, dim=1, keepdim=True).clamp(min=1))

        # build rank-1 output
        X_hat = torch.einsum('bi,bj->bij', scores, scores)  # [B, N, N]

        return X_hat


def evaluate_model(model):
    # Make test dataset
    N = 15000
    n = 350
    k = 20
    
    # Create dataset for training
    test_data = sample_data(N, n, k, device=device)
    test_dataset = SubmatrixDataset(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=250, shuffle=True)
    
    # Draw a loss curve
    loss_pts = []
    
    # Make an array of time points
    time_array = torch.linspace(20, 1400, 70, device=device)
    
    with torch.no_grad():
        for it in range(len(time_array)):
            loss = 0
            
            # Set time
            t = time_array[it].item()
        
            # Evaluate model
            for batch in test_dataloader:
                loss_b = mc_loss_batch_fixed(model, batch, t, n, k, predict=True, device=device)
                loss += loss_b.item() * batch.shape[0]
    
            # Store loss
            loss_pts.append(loss / len(test_dataloader.dataset))
            print("Iteration {} finished!".format(it))

    return loss_pts