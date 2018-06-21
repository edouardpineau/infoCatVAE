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
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=True,
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

class InfoCatRVAE(nn.Module):
    def __init__(self, in_dim, h_dim, z_dim, num_class, n_layers, bidirectional_encoder=True):
        super(InfoCatRVAE, self).__init__()

        self.in_dim = in_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.num_class = num_class
        self.bidirectional_encoder = bidirectional_encoder
        self.n_layers = n_layers

        self.encoder = nn.GRU(in_dim, h_dim, n_layers, bidirectional=bidirectional_encoder)
        self.decoder = nn.GRU(in_dim, h_dim, n_layers, bidirectional=False)

        self.fc21 = nn.Linear(h_dim * 2 + num_class, z_dim)
        self.fc22 = nn.Linear(h_dim * 2 + num_class, z_dim)
        self.fca = nn.Linear(h_dim * 2, num_class)
        self.fc3 = nn.Linear(z_dim, h_dim * n_layers)
        self.fc4 = nn.Linear(h_dim, in_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def encode(self, x):
        _, h = self.encoder(x)
        h_fb = torch.cat([h[-1], h[-2]], dim=1) # since budurectional rnn
        a = F.softmax(self.fca(h_fb), dim=1)
        idt = torch.eye(self.num_class)
        if args.cuda:
            idt = idt.cuda()
        allmu = torch.stack(
            [self.fc21(torch.cat((h_fb, Variable(idt[i, :].repeat(x.size(1), 1))), 1)) for i in range(self.num_class)])
        allvar = torch.stack(
            [self.fc22(torch.cat((h_fb, Variable(idt[i, :].repeat(x.size(1), 1))), 1)) for i in range(self.num_class)])

        return self.fc21(torch.cat((h_fb, a), 1)), self.fc22(torch.cat((h_fb, a), 1)), a, allmu, allvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            z = eps.mul(std).add_(mu)
            return z
        else:
            return mu

    def decode(self, z, batch_size, len_sample, target=None):
        h_temp = self.tanh(self.fc3(z)).unsqueeze(0)
        h3 = torch.cat([h_temp[:, :, int(self.h_dim * (i - 1)):int(self.h_dim * i)] for i in
                       np.arange(1, 1 + self.n_layers)], dim=0)

        if args.cuda:
            o = Variable(torch.zeros(1, batch_size, self.in_dim)).cuda()
        else:
            o = Variable(torch.zeros(1, batch_size, self.in_dim))
        x_reco = [o]
        loss = 0

        try:
            if target==None:
                for t in np.arange(len_sample):
                    o, h3 = self.decoder(o, h3)
                    o = self.sigmoid(self.fc4(o))
                    x_reco.append(o)
                X_reco = torch.cat(x_reco, dim=0)

                return X_reco
        except:
            for t in np.arange(len_sample - 1):
                o, h3 = self.decoder(o, h3)
                o = self.sigmoid(self.fc4(o))
                x_reco.append(o)
                loss += F.mse_loss(o.squeeze(0), target[t + 1])
            X_reco = torch.cat(x_reco, dim=0)

            return X_reco, loss

    def forward(self, x):
        mu, logvar, a, allmu, allvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reco, loss = self.decode(z, x.size(1), x.size(0), x)
        return x_reco, mu, logvar, a, allmu, allvar, loss

model = InfoCatRVAE(in_dim, h_dim, z_dim, num_class, n_layers, bidirectional_encoder=True)

if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def KLDG(mu1, mu2, logvar):
    return -0.5 * (1 + logvar - (mu1 - mu2).pow(2) - logvar.exp())

muprior = np.zeros((z_dim, num_class))

for i in range(num_class):
    for j in range(sub_dim):
        muprior[sub_dim * i + j, i] = 2

if args.cuda:
    mupriorT = Variable(torch.from_numpy(muprior).type(torch.FloatTensor)).cuda()
else:
    mupriorT = Variable(torch.from_numpy(muprior).type(torch.FloatTensor))


# Reconstruction + KL divergence losses summed over all elements and batch

def loss_function(recon_x, x, mu, logvar, a, allmu, allvar):
    BCE = F.binary_cross_entropy(recon_x.view(-1, recon_x.size(0) * recon_x.size(2))
                                 , x.view(-1, x.size(0) * x.size(2)), size_average=False)
    KLD = 0
    # print(a.size())
    for i in range(num_class):
        KLD += torch.sum(torch.t(a[:, i].repeat(z_dim, 1)) * KLDG(allmu[i, :, :], mupriorT[:, i].repeat(allmu.size(1), 1),
                                                               allvar[i, :, :]))

    negH = torch.sum(a.mul(torch.log(a + 0.001)))

    return BCE + 100 * negH + KLD, BCE, negH, KLD


def sampling(k):
    samples = []
    if args.cuda:
        for i in range(k):
            mu = mupriorT.t()
            logvar = Variable(torch.FloatTensor(np.zeros((num_class, z_dim)))).cuda()
            z = model.reparameterize(mu, logvar)
            samples.append(model.decode(z, num_class, 300))
        sample = torch.cat(samples, 1)
        labels = Variable(torch.arange(0, num_class).repeat(k).long()).cuda()
    else:
        for i in range(k):
            mu = Variable(torch.FloatTensor(muprior.T))
            logvar = Variable(torch.FloatTensor(np.zeros((num_class, z_dim))))
            z = model.reparameterize(mu, logvar)
            samples.append(model.decode(z, num_class, 300))
        sample = torch.cat(samples, 1)
        labels = Variable(torch.arange(0, num_class).repeat(k).long())
    return sample, labels


def train(epoch):
    model.train()
    train_loss = 0
    class_loss = []

    for batch_idx, data in enumerate(train_loader):

        data = Variable(data)
        data = torch.transpose(data, 0, 1).float()
        data = torch.cat([Variable(torch.zeros(1, data.size(1), data.size(2))), data])

        if args.cuda:
            data = data.cuda()

        optimizer.zero_grad()
        recon_batch, mu, logvar, a, allmu, allvar, loss = model(data)
        _, la, lb, lc = loss_function(recon_batch, data, mu, logvar, a, allmu, allvar)

        # Adversarial learning of classes

        sample, labels = sampling(10)
        _, _, a, _, _ = model.encode(sample)
        adv_loss = F.cross_entropy(a, labels)

        (loss + 10*lb + 10*lc + 100*adv_loss).backward()

        train_loss += loss.data[0]
        class_loss.append(adv_loss.data[0])
        optimizer.step()
        _, preds = torch.max(a, 1)

    print('====> Epoch: {} Average loss: {:.4f}\tInfoLoss: {:.2f}'.format(
        epoch, train_loss / len(train_loader.dataset), np.mean(class_loss)))

test_class_perf = []
test_lost_list = []

def test(epoch):
    global test_lost_list
    model.eval()
    test_loss = 0

    for i, data in enumerate(test_loader):
        data = Variable(data, volatile=True)
        if args.cuda:
            data = data.cuda()
        data = torch.transpose(data, 0, 1).float()
        if args.cuda:
            data = torch.cat([Variable(torch.zeros(1, data.size(1), data.size(2))).cuda(), data])
        else:
            data = torch.cat([Variable(torch.zeros(1, data.size(1), data.size(2))), data])

        recon_batch, mu, logvar, a, allmu, allvar, loss = model(data)
        _, preds = torch.max(a, 1)
        _, la, lb, lc = loss_function(recon_batch, data, mu, logvar, a, allmu, allvar)
        test_loss += loss.data[0]
        if i == 0:
            n = min(data.size(0), 10)
            comparison = torch.cat([data.t().unsqueeze(1)[:n],
                                    recon_batch.t().unsqueeze(1)[:n]])
            save_image(comparison.data.cpu(),
                       '../InfoCatRVAE_rec.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    test_lost_list.append(test_loss)



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
        torch.save(model.state_dict(), '../InfoCatRVAE_rec.pt')
        delta = np.ones((num_class, z_dim))
        model.train()
        samples = []
        for i in range(num_class):
            mu = Variable(torch.FloatTensor(muprior.T))
            logvar = Variable(torch.FloatTensor(np.zeros((num_class, z_dim))))
            z = model.reparameterize(mu, logvar)
            if args.cuda:
                z = z.cuda()
            samples.append(model.decode(z, num_class, 300))

        sample = torch.cat(samples, 1) # Variable(torch.randn(64, 20))
        if args.cuda:
            sample = sample.cuda()
        # save_image(sample.t().unsqueeze(1).data,
        #            '../InfoCatRVAE_rec.png', nrow=10)

        # d = sample[-1, 0].cpu().data.numpy()
        # plt.scatter(d[np.arange(0, len(d), 2)], d[np.arange(1, len(d), 2)])
        # plt.savefig('../InfoCatRVAE_rec.png')
        # plt.close()

        fig, ax1 = plt.plot(sample[:, :, 0].data.numpy())
        plt.savefig('../' + name_img + '.png')
        plt.close()






