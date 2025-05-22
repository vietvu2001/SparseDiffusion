# SparseDiffusion

This repository is the official implementation of the paper "Computational bottlenecks for denoising diffusions".

# Hardware details and dependencies

The models were trained on a single NVIDIA A100 GPU with 84.97 GB memory. The only dependency is PyTorch, as specified in `requirements.txt`.

# Load in pre-trained models

The main figures in the paper were made using an old version of our current architecture. To load in this model, please use

<pre># Reload the deprecated neural network
n = 350
k = 20
model = TestMPNN_3(k, hidden_dim_1=32, hidden_dim_2=16, hidden_dim_3=4, num_layers=10)
model.to(device)
model.load_state_dict(torch.load("pretrained_models/model_weights_opt_350.pth", weights_only=False), strict=False)
model.readout_coefs.data.zero_()
model.eval()</pre>