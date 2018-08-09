# InfoCatVAE

This repository gives an implementation of InfoCatVAE: https://arxiv.org/pdf/1806.08240.pdf

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

### Mixture model

<img src="https://github.com/edouardpineau/infoCatVAE/raw/master/images/CatVAE_architecture.png" width="1000">

Figure 1: CatVAE: square blocks represent neural networks, oval-shaped blocks represent sampling

This architecture offers a natural new ELBO that has the following propoerties:

- The mapping of a datapoint to a cluster is done with a softclutering framework
- All the distances between the code and all clusters are explicitly computed and used in the backpropagation algorithm
- An entropy term prevents trivial solution where all datapoints are mapped to one cluster


### Contributions:

- Exploring how standard mixture models could be modified to overpass robustness by using fixed subspace-clustering-type of prior
- Lowering the negative trade-off between expressiveness and robustness in mixture models by using information maximization trick and 
- Leveraging information maximization architecture to enable to network to naturally optimize categorical sampling layer


# Choice of the prior

Let d be the dimension of the latent space such that ∃ δ ∈ N s.t. d = K.δ.

Assumption: data with K categories should be encoded with a K-modal distribution modeled with N (z; μc, 1) such that μc ∈ Rd and μc.μc′ = 0. We inspire from subspace clustering assumptions and propose ∀c ∈ {1...K} we propose a μc such that:

- Each categories lives mainly in a δ−dimensional subspace of Z
- The categorical variable is modeled by p(c) = U({1...K})
- This prior shape encourages the network to find discriminative representation of the data according to its most salient attribute


# InfoCatVAE: categorical VAE with information maximization

Objective: improve generation and regularize representation learning InfoCatVAE uses learned classifier as to evaluate the quality of the generation:

The higher the mutual information between the sample and its category is, the better the generation should be

<img src="https://github.com/edouardpineau/infoCatVAE/raw/master/images/InfoCatVAE_architecture.png" width="1000">

Figure 2: square blocks represent neural networks, oval-shaped blocks represent sam- pling. Encoding and decoding blocks are shared with CatVAE presented in figure 1.


Mutual information has a tractable lower bound (see Chen's InfoGAN) whose exact algorithmic transcription is described by the figure 2. Main idea: each conditionally generated data should be classified in its original cluster. The mutual information lower bound term is added to the CatVAE ELBO. 

This term is not arbitrary. With our particular form of prior, all clusters in InfoCatVAE are equidistant in term of euclidean norm, and orthogonal. This particular geometry imposes to propose a non-euclidean distance metric between our clusters. The node pair sampling ratio is a probabilistic symmetric and non-negative divergence between two distinct clusters, that we can adapt to our problem. It is shown that in InfoCatVAE we implicitely maximize pairwise cross entropy between distinct clusters (using samples from these clusters). Note that this is completely equivalent to say that information that defines a cluster should be as exclusive as possible to this cluster's samples: this is the intra-cluster information maximization.


# Optimization with categorical sampling layer

Gumbel-softmax trick is a standard continuous relaxation for categorical sampling optimization. Several discrete optimization trick for back-propagating the gradient have been developped.

In InfoCatVAE: 
- Categorical representation is let deterministic instead of random: for each x all qφ(c|x) are computed and two by two confronted
- The categorical sampling is made in parallel during optimization in the information maximization part

It means that categorical representation is deterministic (catVAE) conditionally to the fact that representation is coherent when sampled randomly (infoCatVAE). It is a form of gradient-free (Monte-Carlo) optimization.


# Illustrative results

### MNIST
<img src="https://github.com/edouardpineau/infoCatVAE/raw/master/images/InfoCatVAE_MNIST_interp.png" width="400">

### Fashion MNISTR
<img src="https://github.com/edouardpineau/infoCatVAE/raw/master/images/InfoCatVAE_inter_centroids.png" width="400">



