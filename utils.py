from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.utils import save_image
import numpy as np
import pandas as pd



def KL_gaussian(mu1, mu2, logvar):
    return -0.5 * (1 + logvar - (mu1 - mu2).pow(2) - logvar.exp())


def prior_construction(model):
    num_class, sub_dim, lmbda, is_cuda = model.num_class, model.sub_dim, model.lmbda, model.is_cuda
    muprior = np.zeros((model.num_class*model.sub_dim, model.num_class))

    for i in range(model.num_class):
        for j in range(model.sub_dim):
            muprior[model.sub_dim * i + j, i] = model.lmbda

    if model.is_cuda:
        mupriorT = Variable(torch.from_numpy(muprior).type(torch.FloatTensor)).cuda()
    else:
        mupriorT = Variable(torch.from_numpy(muprior).type(torch.FloatTensor))

    return mupriorT


def loss_function(model, recon_x, x, a, allmu, allvar, mupriorT):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, model.in_dim), size_average=False)
    MSE = F.mse_loss(recon_x, x.view(-1, model.in_dim))

    KLD = 0
    for i in range(model.num_class):
        KLD += torch.sum(torch.t(a[:, i].repeat(model.z_dim, 1)) * KL_gaussian(allmu[i, :, :],
                                                                               mupriorT[:, i].repeat(allmu.size(1), 1),
                                                                               allvar[i, :, :]))

    negH = torch.sum(a.mul(torch.log(a + 0.001)))

    return BCE + 10*negH + 10*KLD, BCE, negH, KLD, MSE


def sampling(model, k, mupriorT):
    allz = []
    if model.is_cuda:
        for i in range(k):
            mu = mupriorT.t().data
            logvar = torch.FloatTensor(np.zeros((model.num_class, model.z_dim)))
            z = model.reparameterize(Variable(mu).cuda(), Variable(logvar).cuda())
            allz.append(z)
        sample = torch.cat(allz, dim=0)  # Variable(torch.randn(64, 20))
        sample = model.decode(sample)
    else:
        for i in range(k):
            mu = mupriorT.t().data
            logvar = torch.FloatTensor(np.zeros((model.num_class, model.z_dim)))
            z = model.reparameterize(Variable(mu), Variable(logvar))
            allz.append(z)
        sample = torch.cat(allz, dim=0)  # Variable(torch.randn(64, 20))
        sample = model.decode(sample)

    return sample, Variable(torch.arange(0, model.num_class).repeat(k).long())




