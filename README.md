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

### New model and adapted ELBO

- Generative model: $$p(x,c,z)=p(c)p(z|x)p(x|c,z)$$
- Inference model: $$q_\phi(c|x)q_\phi(z,c,x)$$
- New ELBO:

max Epd(x) Eq(z|x) [log pθ(x|z)] − Eqφ(c|x) [KL (qφ(z|c, x)||p(z|c))] − KL (qφ(c|x)||p(c)

<img src="https://github.com/edouardpineau/infoCatVAE/raw/master/images/figure1.png" width="400">

This architecture offers a natural new ELBO that has the following propoerties:

- The mapping of a datapoint to a cluster is done with a softclutering framework
- All the distances between the code and all clusters are explicitly computed and used in the backpropagation algorithm
- An entropy term prevents trivial solution where all datapoints are mapped to one cluster


### Three main contributions:
- Choosing the form of the latent prior $$p(z,c)$$, then choosing $$p(c)$$ and $$p(z \vert c)$$
- Dealing with the impossibility to back-propagate the gradient through the categorical sampling layer
- Keeping the generative power despite the structure of the latent space


# Choice of the prior

Let d be the dimension of the latent space such that ∃ δ ∈ N s.t. d = K.δ.




# Illustrative results

### MNIST
<img src="https://github.com/edouardpineau/infoCatVAE/raw/master/images/InfoCatVAE_MNIST_interp.png" width="400">

### Fashion MNISTR
<img src="https://github.com/edouardpineau/infoCatVAE/raw/master/images/InfoCatVAE_inter_centroids.png" width="400">









