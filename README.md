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

### Modifying VAEs for representation learning

# Objective:

Enforce categorical readable information in the latent code representation with the following categorical VAE (CatVAE):


- Generative model: $$ p(x,z,c)=p(c)p(z \vert c)p(x \vert z,c)$ with $c\in \{1...K \} $$
- Inference model: $$ q_\phi (c \vert x) q_\phi (z, x,c) $$
- New ELBO: $$ \max_\phi \EE_{p_d(x)}\left[\EE_{q(z|x)}\left[\log p_\theta(x|z)\right] -\EE_{q_\phi(c|x)} \left[\KL\left(q_\phi(z|c,x)||p(z|c) \right) \right] -\KL\left( q_\phi(c|x)||p(c)\right) \right] $$
