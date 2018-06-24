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

- Generative model: p(x,c,z)=p(c)p(z|x)p(x|c,z)
- Inference model: qφ(c|x)qφ(z, x, c)
- New ELBO:

max Epd(x) Eq(z|x) [log pθ(x|z)] − Eqφ(c|x) [KL (qφ(z|c, x)||p(z|c))] − KL (qφ(c|x)||p(c)

<img src="https://github.com/edouardpineau/infoCatVAE/raw/master/images/CatVAE_architecture.png" width="1000">

Figure 1: CatVAE: square blocks represent neural networks, oval-shaped blocks represent sampling

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

Assumption: data with K categories should be encoded with a K-modal distri- bution respectively modeled with N (z; μc, 1), with {μc}^K_{c=1} ∈ Rd and μc.μc′ = 0. We propose ∀c ∈ {1...K} we propose:

\mu_c=\{\lambda.\mathds{1}_{j \in \llbracket {c\times \delta:(c+1) \times \delta} \llbracket} \}_{j=1}^{d}

- Each categories lives mainly in a δ−dimensional subspace of Z
- The categorical variable is modeled by p(c) = U({1...K})
- This prior form encourage the network to find discriminative representation of
the data according to its most salient attribute, like a in a clustering framework


# InfoCatVAE: categorical VAE with information maximization

Objective: improve generation and regularize representation learning InfoCatVAE uses learned classifier as to evaluate the quality of the generation:

The higher the mutual information between the sample and its category is, the better the generation should be

<img src="https://github.com/edouardpineau/infoCatVAE/raw/master/images/InfoCatVAE_architecture.png" width="1000">

Figure 2: square blocks represent neural networks, oval-shaped blocks represent sam- pling. Encoding and decoding blocks are shared with CatVAE presented in figure 1.


Mutual information has a tractable lower bound (see Chen's InfoGAN) whose exact algorithmic transcription is described by the figure 2. Main idea: each conditionally generated data should be classified in its original cluster. The mutual information lower bound term is added to the CatVAE ELBO.


# Optimization with categorical sampling layer

Objective: optimize the square blocks of the figure 1 and 2. No natural sampling optimization for categorical distributions. Usually, Jang's Gumbel-Softmax trick is used. We propose to use the information maximization brick as an alternative two-step method:

- In the inference learning, no categorical sampling: for each x all qφ(c|x) are computed and two by two confronted
- Categorical sampling in InfoCatVAE learning is done in the information maximization part


# Illustrative results

### MNIST
<img src="https://github.com/edouardpineau/infoCatVAE/raw/master/images/InfoCatVAE_MNIST_interp.png" width="400">

### Fashion MNISTR
<img src="https://github.com/edouardpineau/infoCatVAE/raw/master/images/InfoCatVAE_inter_centroids.png" width="400">









