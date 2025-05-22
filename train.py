# Import functions from helper files
import torch
from sample_data import sample_data, SubmatrixDataset
from loss import mc_loss_batch_simul, mc_loss_batch_fixed
from denoiser import TopKStraightThrough, TestMPNN_3
import os
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader


def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_part_1(model, n, k, train_dataloader, num_epochs, device):
    # Make an array of time points
    time_array = torch.linspace(0.5, 700, 1400, device=device)
    
    # Initialize optimal model
    best_weights = None
    min_loss = float("inf")
    
    # Specify the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    
    for it in range(num_epochs):
        counter = 0
        loss_total = 0
        
        # If iteration is small, train with 9 layers
        if it < 150:
            model.num_layers = 7
        elif it < 250:
            model.num_layers = 9
        else:
            model.num_layers = 10
    
        # Iterate through batches
        for batch in train_dataloader:
            # Get loss from model
            loss = mc_loss_batch_simul(model, batch, time_array, n, k, time_threshold=400, p_bad=0.05, device=device)
            
            optimizer.zero_grad()
            loss.backward()
        
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
            # Backpropagate
            optimizer.step()
    
            # Update counter
            counter += 1
            loss_total += loss.item() * batch.shape[0]
    
        # Update best model
        with torch.no_grad():
            avg = loss_total / len(train_dataloader.dataset)
            print("Average loss per iteration {}: {}".format(it, avg))

            # Only start updating when model is fully training
            if avg < min_loss and model.num_layers == 10:
                min_loss = avg
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # Return optimal model
    return best_weights, min_loss


def train_part_2(model, n, k, train_dataloader, num_epochs, device):
    # Make an array of time points
    time_array = torch.linspace(0.5, 700, 1400, device=device)
    
    # Specify the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # When fine-tune, have to reset min_loss to take into account the new points
    best_weights = None
    min_loss = float("inf")
    
    for it in range(num_epochs):
        counter = 0
        loss_total = 0
        for batch in train_dataloader:
            # Get loss from model
            loss = mc_loss_batch_simul(model, batch, time_array, n, k, time_threshold=325, p_bad=0.05, device=device)
            
            optimizer.zero_grad()
            loss.backward()
        
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
            # Backpropagate
            optimizer.step()
    
            # Update counter
            counter += 1
            loss_total += loss.item() * batch.shape[0]
    
        # Update best model
        with torch.no_grad():
            avg = loss_total / len(train_dataloader.dataset)
            print("Average loss per iteration {}: {}".format(it, avg))
            if avg < min_loss:
                min_loss = avg
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # Save optimal model to save time for generation
    return best_weights, min_loss


def train_part_3(model, n, k, train_dataloader, num_epochs, device):    
    # Make an array of time points
    time_array = torch.linspace(0.5, 700, 1400, device=device)
    
    # Specify the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4)
    
    # When fine-tune, have to reset min_loss to take into account the new points
    best_weights = None
    min_loss = float("inf")

    # Fishing pole at large time to make sure that the model does not overfit to the time region
    t_large = 10000
    
    for it in range(num_epochs):
        counter = 0
        loss_total = 0
        if it % 6 < 5:
            for batch in train_dataloader:
                # Get loss from model
                loss = mc_loss_batch_simul(model, batch, time_array, n, k, time_threshold=205, p_bad=0.05, device=device)
                
                optimizer.zero_grad()
                loss.backward()
            
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.9)
            
                # Backpropagate
                optimizer.step()
        
                # Update counter
                counter += 1
                loss_total += loss.item() * batch.shape[0]
        
            # Update best model
            with torch.no_grad():
                avg = loss_total / len(train_dataloader.dataset)
                print("Average loss per iteration {}: {}".format(it, avg))
                if avg < min_loss and it >= 12:
                    min_loss = avg
                    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}


        else:
            for batch in train_dataloader:
                # Get loss from model
                loss = mc_loss_batch_fixed(model, batch, t_large, n, k, predict=False, device=device)
                
                optimizer.zero_grad()
                loss.backward()
            
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.9)
            
                # Backpropagate
                optimizer.step()
        
                # Update counter
                counter += 1
                loss_total += loss.item() * batch.shape[0]

            with torch.no_grad():
                avg = loss_total / len(train_dataloader.dataset)
                print("Average loss per fishing-pole iteration {}: {}".format(it, avg))
            
    
    return best_weights, min_loss


def train(n, k, device, num_epochs_ls = [600, 300, 300], seed=None):
    if seed is not None:
        set_global_seed(seed)

    # Part 1
    # Sample data
    N = 10000
    
    # Create dataset for training
    train_data = sample_data(N, n, k, device=device)
    train_dataset = SubmatrixDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=200, shuffle=True)

    # Initialize model
    model = TestMPNN_3(k, hidden_dim_1=32, hidden_dim_2=16, hidden_dim_3=4, num_layers=10)
    model.to(device)

    # Get back optimal model from part 1
    weights_ckpt1, min_loss_1 = train_part_1(model, n, k, train_dataloader, num_epochs_ls[0], device)
    print("Rep loss: {}".format(min_loss_1))

    # Reload model
    model.load_state_dict(weights_ckpt1)

    # Train part 2
    # Re-sample data
    train_data = sample_data(N, n, k, device=device)
    train_dataset = SubmatrixDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=200, shuffle=True)

    # Get best weights
    weights_ckpt2, min_loss_2 = train_part_2(model, n, k, train_dataloader, num_epochs_ls[1], device)

    # Reload model
    model.load_state_dict(weights_ckpt2)

    # Train part 3
    # Re-sample data
    train_data = sample_data(N, n, k, device=device)
    train_dataset = SubmatrixDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=200, shuffle=True)

    # Get best weights
    weights_ckpt3, min_loss_3 = train_part_3(model, n, k, train_dataloader, num_epochs_ls[2], device)

    # Reload model
    model.load_state_dict(weights_ckpt3)

    # Return model
    return model