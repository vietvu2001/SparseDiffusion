# SparseDiffusion

This repository is the official implementation of the paper "Computational bottlenecks for denoising diffusions".

# Hardware details and dependencies

The models were trained on a single NVIDIA A100 GPU with 84.97 GB memory. The only dependency is PyTorch, as specified in `requirements.txt`.

# Train a new model

Functions to train a new model are specified in `train.py`, and use the architecture specified in `denoiser.py`. 

<pre># Modify this seed
seed = 1234
n = 350
k = 20
model = train(n, k, num_epochs_ls = [600, 300, 300], seed)</pre>

# Load in pre-trained models

The pre-trained models are contained in the `pretrained_models` folder. The main figures in the paper were made using an old version of our current architecture. To load in this model, please use

<pre># Reload the deprecated neural network
n = 350
k = 20
model = TestMPNN_3(k, hidden_dim_1=32, hidden_dim_2=16, hidden_dim_3=4, num_layers=10)
model.to(device)
model.load_state_dict(torch.load("pretrained_models/model_weights_opt_350.pth", weights_only=False), strict=False)
model.readout_coefs.data.zero_()
model.eval()</pre>

Alternatively, the newer models can be loaded in a simpler way. For instance:

<pre>model = TestMPNN_3(k, hidden_dim_1=32, hidden_dim_2=16, hidden_dim_3=4, num_layers=10)
model.to(device)
model.load_state_dict(torch.load("model_4173183967.pth", weights_only=False))</pre>