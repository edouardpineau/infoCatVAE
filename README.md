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

### New architecture

