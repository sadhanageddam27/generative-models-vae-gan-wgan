# Generative Models — VAE, GAN, and WGAN

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?logo=pytorch)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Dataset](https://img.shields.io/badge/Dataset-MNIST%20%26%20Pokémon-purple)

Three-part deep learning project implementing Autoencoders, Variational
Autoencoders, Deep Convolutional GANs trained on a Pokémon image dataset,
and Wasserstein GANs on MNIST — all built from scratch in PyTorch with
full training progression visualizations.

## Live Project Report
Full results, generated image progression, and architecture breakdown:
**https://sadhanageddam27.github.io/generative-models-vae-gan-wgan/**

---

## Project Structure

    generative-models-vae-gan-wgan/
    ├── Part1_VAE/
    │   ├── Part1_AE_Task1.ipynb       # Basic Autoencoder, latent dim 2/8/32
    │   └── Part1_VAE_Task2.ipynb      # Variational Autoencoder + sampling
    ├── Part2_GAN/
    │   ├── Part2_GAN_Task1.ipynb      # DCGAN on Pokémon dataset
    │   ├── Part2_GAN_Task2.ipynb      # ReLU vs LeakyReLU comparison
    │   └── dcgan_images/              # Generated images saved per epoch
    └── Part3_WGAN/
        ├── Part3_WGAN_Task1.ipynb     # WGAN on MNIST
        └── Part2_WGAN_Task2.ipynb     # Extended WGAN experiments

---

## What This Project Covers

### Part 1 — Autoencoder and VAE on MNIST

Encoder-Decoder architecture trained on MNIST across three latent
dimensions to study the effect of bottleneck size on reconstruction.

| Latent Dim | Epoch 1 Loss | Final Loss | Quality |
|-----------|-------------|------------|---------|
| 2 | 0.2386 | 0.1805 | Blurry, ambiguous |
| 8 | 0.1893 | 0.1141 | Clear, minor blur |
| **32** | 0.1774 | **0.0759** | Sharp, detailed |

VAE adds KL divergence regularization using the reparameterization trick:

    z = μ + ε · exp(0.5 · logvar),  ε ~ N(0,1)
    L = E[log p(x|z)] − KL(q(z|x) ‖ p(z))

VAE with latent_dim=2 produced smooth, diverse generative samples from
N(0,1) — outperforming all AE configurations at sampling because KL
regularization forces a continuous, structured latent manifold.

### Part 2 — DCGAN on Pokémon Dataset

Generator uses transposed convolutions (nz=100 → 512 → 256 → 128 →
64 → RGB). Discriminator uses strided convolutions. Trained 30 epochs
with Adam (lr=0.0002, β₁=0.5).

Visual progression from training:
- Epoch 1 — random color blobs, no structure
- Epoch 5 — color clusters begin forming
- Epoch 15 — Pokémon silhouettes recognizable
- Epoch 30 — structured characters, D(x) = 0.5144 (near Nash equilibrium)

Task 2 compares ReLU vs LeakyReLU in the discriminator — LeakyReLU
produced sharper outlines by preventing dead neurons.

### Part 3 — Wasserstein GAN on MNIST

Replaced binary cross-entropy with Wasserstein distance, discriminator
with a Critic (linear output, no sigmoid), and enforced the Lipschitz
constraint via weight clipping [-0.01, 0.01].

| Aspect | Standard GAN | WGAN |
|--------|-------------|------|
| Loss | Binary CE | Wasserstein distance |
| Output | Sigmoid [0,1] | Linear (unbounded) |
| Optimizer | Adam | RMSProp |
| Stability | Unstable | **Stable, no mode collapse** |
| Best epochs | 30 | **35–45** |

---

## Results Summary

| Model | Dataset | Key Result |
|-------|---------|-----------|
| AE (dim=32) | MNIST | Reconstruction loss 0.076 |
| VAE (dim=2) | MNIST | Smooth latent space, coherent samples |
| DCGAN | Pokémon | Stable at epoch 30, D(x)=0.51 |
| WGAN | MNIST | Monotonic convergence, no mode collapse |

---

## Setup and Usage

    git clone https://github.com/sadhanageddam27/generative-models-vae-gan-wgan.git
    cd generative-models-vae-gan-wgan
    pip install torch torchvision matplotlib jupyter
    jupyter notebook

---

## Tech Stack
Python · PyTorch · torchvision · Jupyter · Matplotlib

## Topics
`deep-learning` `pytorch` `vae` `gan` `dcgan` `wgan`
`generative-models` `mnist` `pokemon` `python`
