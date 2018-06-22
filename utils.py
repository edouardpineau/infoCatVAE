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


def prior_construction(lmbda, is_cuda):
    muprior = np.zeros((z_dim, num_class))

    for i in range(num_class):
        for j in range(sub_dim):
            muprior[sub_dim * i + j, i] = lmbda

    if is_cuda:
        mupriorT = Variable(torch.from_numpy(muprior).type(torch.FloatTensor)).cuda()
    else:
        mupriorT = Variable(torch.from_numpy(muprior).type(torch.FloatTensor))

    return mupriorT


def loss_function(recon_x, x, mu, logvar, a, allmu, allvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, in_dim), size_average=False)
    MSE = F.mse_loss(recon_x, x.view(-1, in_dim))

    KLD = 0
    for i in range(num_class):
        KLD += torch.sum(torch.t(a[:, i].repeat(z_dim, 1)) * KL_gaussian(allmu[i, :, :], mupriorT[:, i].repeat(allmu.size(1), 1),
                                                               allvar[i, :, :]))

    negH = torch.sum(a.mul(torch.log(a + 0.001)))

    return BCE + 100*negH + 100*KLD, BCE, negH, KLD, MSE


def sampling(k, is_cuda):
    allz = []
    if is_cuda:
        for i in range(k):
            mu = torch.FloatTensor(muprior.T)
            logvar = torch.FloatTensor(np.zeros((num_class, model.z_dim)))
            z = model.reparameterize(Variable(mu).cuda(), Variable(logvar).cuda())
            allz.append(z)
        sample = torch.cat(allz, dim=0)  # Variable(torch.randn(64, 20))
        sample = model.decode(sample)
    else:
        for i in range(k):
            mu = torch.FloatTensor(muprior.T)
            logvar = torch.FloatTensor(np.zeros((num_class, model.z_dim)))
            z = model.reparameterize(Variable(mu), Variable(logvar))
            allz.append(z)
        sample = torch.cat(allz, dim=0)  # Variable(torch.randn(64, 20))
        sample = model.decode(sample)

    return sample, Variable(torch.arange(0, num_class).repeat(k).long())


def train(epoch):
    model.train()
    train_loss, train_reco_loss, train_negH, train_KLD = 0, 0, 0, 0
    class_loss = []
    for batch_idx, data in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar, a, allmu, allvar = model(data)
        loss, la, lb, lc, mse_loss = loss_function(recon_batch, data, mu, logvar, a, allmu, allvar)

        loss.backward(retain_graph=True)

        # Adversarial learning of classes

        sample, labels = sampling(10)
        if args.cuda:
            sample = sample.cuda()
            labels = labels.cuda()
        _, _, a, _, _ = model.encode(sample)
        adv_loss = F.cross_entropy(a, labels)

        adv_loss.backward()

        train_loss += loss.data[0]
        train_reco_loss += la.data[0]
        train_negH += lb.data[0]
        train_KLD += lc.data[0]

        if args.cuda:
            del sample
            del labels

            torch.cuda.empty_cache()

        class_loss.append(adv_loss.data[0])
        optimizer.step()
        _, preds = torch.max(a, 1)

    print('====> Epoch: {}, Reco: {:.2f}, negH: {:.2f}, KLD: {:.2f}, Class: {:.2f}'.format(
        epoch,
        train_reco_loss / len(train_loader.dataset),
        train_negH / len(train_loader.dataset),
        train_KLD / len(train_loader.dataset),
        np.mean(class_loss)))


def test(epoch):
    global test_lost_list
    model.eval()
    test_loss = 0

    for i, data in enumerate(test_loader):
        data = Variable(data, volatile=True)
        if args.cuda:
            data = data.cuda()
        recon_batch, mu, logvar, a, allmu, allvar = model(data)
        _, preds = torch.max(a, 1)
        loss, la, lb, lc, mse_loss = loss_function(recon_batch, data, mu, logvar, a, allmu, allvar)
        test_loss += loss.data[0]
        if i == 0:
            n = min(data.size(0), 10)
            comparison = torch.cat([data.view(data.size(0), 1, 28, 28)[:n],
                                    recon_batch.view(recon_batch.size(0), 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(),
                       '../InfoCatVAE' + str(epoch) + '.png', nrow=n)

    if args.cuda:
        del data
        torch.cuda.empty_cache()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    test_lost_list.append(test_loss)

