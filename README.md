# infoCatVAE

This repository gives an implementation of InfoCatVAE.

InfoCatVAE is a variational autoencoder framework that enables categorical and continuous interpretable representation with three main specifities:

- A multimodal fixed prior distribution
- A soft-clustering shaped objective function
- An information maximization layer that:
  - requires no additional network
  - improves conditional generation
  - gives a natural framework to overpass discrete sampling backpropagation problem

# Modification of VAEs for representation learning

#### Objective:

Enforce categorical readable information in the latent code representation with the following categorical VAE (CatVAE):


### Three main contributions:
- Choosing the form of the latent prior $$p(z,c)$$, then choosing $$p(c)$$ and $$p(z \vert c)$$
- Back-propagating the gradient through the categorical sampling layer
- Keeping the generative power despite the structure of the latent space

# Illustrative results

### MNIST
<img src="https://github.com/edouardpineau/infoCatVAE/raw/master/images/InfoCatVAE_MNIST_interp.png" width="400">

### Fashion MNISTR
<img src="https://github.com/edouardpineau/infoCatVAE/raw/master/images/InfoCatVAE_inter_centroids.png" width="400">









