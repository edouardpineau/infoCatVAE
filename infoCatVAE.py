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


########## Learning parameters ##########

parser = argparse.ArgumentParser(description='GMSVAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
# args = parser.parse_args()
args, unknown = parser.parse_known_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


########## Data ##########

train_loader = torch.utils.data.DataLoader(train_data,
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(test_data,
    batch_size=args.batch_size, shuffle=True, **kwargs)



########## Model ##########


class InfoCatVAE(nn.Module):
    def __init__(self, in_dim, num_class, sub_dim, z_dim, h_dim):
        super(InfoCatVAE, self).__init__()

        self.in_dim = in_dim
        self.num_class = num_class
        self.sub_dim = sub_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.fc1 = nn.Linear(in_dim, h_dim)
        self.fc21 = nn.Linear(h_dim + num_class, z_dim)
        self.fc22 = nn.Linear(h_dim + num_class, z_dim)
        self.fca = nn.Linear(h_dim, num_class)
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, in_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = F.relu(F.dropout(self.fc1(x), p=0.25))
        a = F.softmax(self.fca(h1), dim=1)
        idt = torch.eye(self.num_class)

        if args.cuda:
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
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            z = eps.mul(std).add_(mu)
            return z
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(F.dropout(self.fc3(z), p=0.25))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar, a, allmu, allvar = self.encode(x.view(-1, self.in_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, a, allmu, allvar


model = InfoCatVAE(in_dim, num_class, sub_dim, z_dim, h_dim)

if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def KLDG(mu1, mu2, logvar):
    return -0.5 * (1 + logvar - (mu1 - mu2).pow(2) - logvar.exp())


muprior = np.zeros((z_dim, num_class))

for i in range(num_class):
    for j in range(sub_dim):
        muprior[sub_dim * i + j, i] = lambda

if args.cuda:
    mupriorT = Variable(torch.from_numpy(muprior).type(torch.FloatTensor)).cuda()
else:
    mupriorT = Variable(torch.from_numpy(muprior).type(torch.FloatTensor))


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, a, allmu, allvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, in_dim), size_average=False)
    MSE = F.mse_loss(recon_x, x.view(-1, in_dim))

    KLD = 0
    for i in range(num_class):
        KLD += torch.sum(torch.t(a[:, i].repeat(z_dim, 1)) * KLDG(allmu[i, :, :], mupriorT[:, i].repeat(allmu.size(1), 1),
                                                               allvar[i, :, :]))
    # KLD = KLDG(mu, 0, logvar)

    negH = torch.sum(a.mul(torch.log(a + 0.001)))

    return BCE + 5*negH + 5 * KLD, BCE, negH, KLD, MSE


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

        # Adversarial learning of classes

        sample, labels = sampling(10)
        if args.cuda:
            sample = sample.cuda()
            labels = labels.cuda()
        _, _, a, _, _ = model.encode(sample)
        adv_loss = F.cross_entropy(a, labels)

        (loss + 100*adv_loss).backward()

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


def sampling(k):
    allz = []
    if args.cuda:
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




########## Model parameters ##########

in_dim = 784
num_class = 40
sub_dim = 2
z_dim = num_class*sub_dim
h_dim = 400
lambda = 2

n_epochs = 5000



########## Learning ##########

test_class_perf = []
test_lost_list = []


for epoch in range(1, n_epochs + 1):
    train(epoch)
    test(epoch)

    if epoch % 10 == 0:
        torch.save(model.state_dict(), '../InfoCatVAE' + str(epoch) + '.pt')
        sample, labels = sampling(10)
        save_image(sample.view(10*num_class, 1, 28, 28).data,
                   '../InfoCatVAE' + str(epoch) + '.png', nrow=num_class)



torch.save(model.state_dict(), '../InfoCatVAE.pt')







