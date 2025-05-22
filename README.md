# SparseDiffusion

This repository is the official implementation of the paper "Computational bottlenecks for denoising diffusions".

# Hardware details and dependencies

The models were trained on a single NVIDIA A100 GPU with 84.97 GB memory. The only dependency is PyTorch, as specified in `dependencies/requirements.txt`.

# Data distribution

The data distribution is given in `sample_data.py`, with the function `sample_data`. This distribution is described in Equation 16, Section 5 of our paper. 

# Train a new model

Functions to train a new model are specified in `train.py`, and use the architecture specified in `denoiser.py`. 

<pre>
from train import train
# Modify this seed
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

# Evaluate score-matching loss

The function to evaluate a trained model is `evaluate_model` in the file `denoiser.py`. This assumes an equispaced time array from 0.5 to 700 with time step 0.5. 

<pre>
from denoiser import evaluate_model

# Get losses
loss_ls = evaluate_model(model)
</pre>

A few loss lists with this time array are given in the `scoring_losses` folder, named by the convention `numbers_{seed}.txt` format. For the seeded models not given in pre-trained models, one can retrain them using the training code provided above.

# Results

Our paper provides evidence that under computational bottlenecks, the samples provided by polytime algorithms will be different from that of the target distribution. In addition to the histogram provided in the main text of our paper, we have made two new histograms, where `histogram_1.pdf` corresponds to `model_4173183967.pth` and `histogram_2.pdf` corresponds to `model_564395852.pth`. These histograms are provided in the `images` folder.