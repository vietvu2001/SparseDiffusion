# SparseDiffusion

This repository is the official implementation of the paper "Computational bottlenecks for denoising diffusions".

# Hardware details and dependencies

The models were trained on a single NVIDIA A100 GPU with 84.97 GB memory. The only dependency is PyTorch, as specified in `dependencies/requirements.txt`, version 2.5.1. The CUDA version is 12.4. The Python version is 3.12.8.

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

# Generate diffusion samples

The function to generate diffusion samples is `generation` in the file `generation.py`. To generate samples using a pre-trained model and make a corresponding histogram, please see for instance

<pre>
model = TestMPNN_3(k, hidden_dim_1=32, hidden_dim_2=16, hidden_dim_3=4, num_layers=10)
model.to(device)
model.load_state_dict(torch.load("model_4173183967.pth", weights_only=False))

# Generate
diff_sample_1 = generation(model, 250, n, 0.5, 4 * n, device=device)
diff_sample_2 = generation(model, 250, n, 0.5, 4 * n, device=device)

diff_sample = torch.concat([diff_sample_1, diff_sample_2], dim=0)
fn = torch.norm(diff_sample, p='fro', dim=(1, 2))

fig, ax = plt.subplots()  # width, height in inches
# Plot histogram
ax.hist(fn.detach().cpu().numpy(),
        bins='fd',                     # Freedman–Diaconis rule
        density=True,                  # or density=True for area=1
        facecolor='lightblue',                # light gray
        edgecolor='black', 
        alpha=0.6,
        linewidth=0.5, label="Diffusion samples")

# Clean up spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

# Ticks outside
ax.tick_params(direction='out', width=0.5)

# Labels (match your manuscript font & size)
ax.set_xlabel('Frobenius norm', fontsize=9)
ax.set_ylabel('density', fontsize=9)

plt.tight_layout()
plt.legend()
plt.savefig('histogram_test.pdf')
plt.show()
</pre>

We cannot generate 500 samples within one single batch due to hardware constraints, so the generation code is split into two. Alternatively, one can modify the `generation` code to generate mini-batches at a time.

# Results

Our paper provides evidence that under computational bottlenecks, the samples provided by polytime algorithms will be different from that of the target distribution. In addition to the histogram provided in the main text of our paper, we have made two new histograms, where `histogram_1.pdf` corresponds to `model_4173183967.pth` and `histogram_2.pdf` corresponds to `model_564395852.pth`. These histograms are provided in the `images` folder.