# https://github.com/pytorch/examples/blob/master/vae/main.py

from utils import *

class InfoCatVAE(nn.Module):
    def __init__(self, in_dim, num_class, sub_dim, z_dim, h_dim, lmbda, is_cuda):
        """
        :param in_dim: input dimension
        :param num_class: latent categorical dimension
        :param sub_dim: dimension of the subspaces
        :param z_dim: latent continuous dimension
        :param h_dim: intermediate layer dimension
        :param lmbda: subspace mean axis values
        :param is_cuda: instruction about the cuda encoding of the Pytorch variables
        """
        
        super(InfoCatVAE, self).__init__()

        self.in_dim = in_dim
        self.num_class = num_class
        self.sub_dim = sub_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.lmbda = lmbda

        self.fc1 = nn.Linear(in_dim, h_dim)
        self.fc21 = nn.Linear(h_dim + num_class, z_dim)
        self.fc22 = nn.Linear(h_dim + num_class, z_dim)
        self.fca = nn.Linear(h_dim, num_class)
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, in_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.is_cuda = is_cuda

    def encode(self, x):
        """
        Encodes the input x into a subspace clustering type of mixture distribution
        :param x: input
        :return: parameters of the latent distribution for the specific input associated with each latent category
                 + probabilities of being in each category
        """
        h1 = F.relu(F.dropout(self.fc1(x), p=0.25))
        a = F.softmax(self.fca(h1), dim=1)
        idt = torch.eye(self.num_class)

        if self.is_cuda:
            allmu = torch.stack([self.fc21(torch.cat((h1, Variable(idt[i, :].repeat(h1.size(0), 1)).cuda()), 1))
                                 for i in range(self.num_class)])
            allvar = torch.stack([self.fc22(torch.cat((h1, Variable(idt[i, :].repeat(h1.size(0), 1)).cuda()), 1))
                                  for i in range(self.num_class)])
        else:
            allmu = torch.stack([self.fc21(torch.cat((h1, Variable(idt[i, :].repeat(h1.size(0), 1))), 1))
                                 for i in range(self.num_class)])
            allvar = torch.stack([self.fc22(torch.cat((h1, Variable(idt[i, :].repeat(h1.size(0), 1))), 1))
                                  for i in range(self.num_class)])

        return self.fc21(torch.cat((h1, a), 1)), self.fc22(torch.cat((h1, a), 1)), a, allmu, allvar

    def reparameterize(self, mu, logvar):
        """
        Model the reparametrization trick for gaussian sampling back-propagation
        
        :param mu: mean of the gaussian
        :param logvar: logvariance of the gaussian
        :return: samples from the gaussian
        """
        
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            z = eps.mul(std).add_(mu)
            return z
        else:
            return mu

    def decode(self, z):
        """
        MLP decoder
        :param z: latent sample before reconstruction / generation
        :return: reconstruction / generation
        """
        
        h3 = F.relu(F.dropout(self.fc3(z), p=0.25))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        """
        Complete mixture autoencoder
        :param x: total mixture autoencoder process
        :return: reconstruction + latent parameters
        """
        
        mu, logvar, a, allmu, allvar = self.encode(x.view(-1, self.in_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, a, allmu, allvar
