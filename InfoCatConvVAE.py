# https://github.com/pytorch/examples/blob/master/vae/main.py
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

from safrantech.loaddata.MNIST import extract_mnist_data, extract_mnist_labels
from safrantech.loaddata.FASHION import extract_fashion_mnist_data, extract_fashion_mnist_labels


########## Parameters ##########

parser = argparse.ArgumentParser(description='GMSVAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
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


def data_builder():

    # MNIST

    filename_data_train = '/data/tsi/analyse_de_donnees/MNIST/train-images-idx3-ubyte.gz'
    filename_labels_train = '/data/tsi/analyse_de_donnees/MNIST/train-labels-idx1-ubyte.gz'
    filename_data_test = '/data/tsi/analyse_de_donnees/MNIST/t10k-images-idx3-ubyte.gz'
    filename_labels_test = '/data/tsi/analyse_de_donnees/MNIST/t10k-labels-idx1-ubyte.gz'
    num_images_train = 60000
    num_images_test = 10000
    data_train = extract_mnist_data(filename_data_train, num_images_train)
    data_train = data_train / 255
    data_test = extract_mnist_data(filename_data_test, num_images_test)
    data_test = data_test / 255

    mnist_train, labels_train = [], []
    mnist_test, labels_test = [], []

    labels_train = extract_mnist_labels(filename_labels_train, num_images_train)
    labels_test = extract_mnist_labels(filename_labels_test, num_images_test)

    for f in data_train:
        mnist_train.append(f.squeeze())
    for f in data_test:
        mnist_test.append(f.squeeze())

    # FashionMNIST

    filename_data_train = '/data/tsi/analyse_de_donnees/FASHION/train-images-idx3-ubyte.gz'
    filename_labels_train = '/data/tsi/analyse_de_donnees/FASHION/train-labels-idx1-ubyte.gz'
    filename_data_test = '/data/tsi/analyse_de_donnees/FASHION/t10k-images-idx3-ubyte.gz'
    filename_labels_test = '/data/tsi/analyse_de_donnees/FASHION/t10k-labels-idx1-ubyte.gz'
    num_images_train = 60000
    num_images_test = 10000
    data_train = extract_fashion_mnist_data(filename_data_train, num_images_train)
    data_train = data_train / 255
    data_test = extract_fashion_mnist_data(filename_data_test, num_images_test)
    data_test = data_test / 255

    fashion_mnist_train, fashion_labels_train = [], []
    fashion_mnist_test, fashion_labels_test = [], []

    fashion_labels_train = extract_fashion_mnist_labels(filename_labels_train, num_images_train)
    fashion_labels_test = extract_fashion_mnist_labels(filename_labels_test, num_images_test)

    for f in data_train:
        fashion_mnist_train.append(f.squeeze())
    for f in data_test:
        fashion_mnist_test.append(f.squeeze())

    return mnist_train, labels_train, mnist_test, labels_test, \
           fashion_mnist_train, fashion_labels_train, fashion_mnist_test, fashion_labels_test


mnist_train, labels_train, mnist_test, labels_test, \
           fashion_mnist_train, fashion_labels_train, fashion_mnist_test, fashion_labels_test = data_builder()

train_loader = torch.utils.data.DataLoader(mnist_train,
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(mnist_test,
    batch_size=args.batch_size, shuffle=True, **kwargs)



########## Model ##########

loss_classifier = nn.NLLLoss()
n_epochs = 1000
batch_size = 64

in_dim = 784
num_class = 10
sub_dim = 2
z_dim = sub_dim * num_class
h_dim = 400


class InfoCatConvVAE(nn.Module):
    def __init__(self, in_dim, num_class, sub_dim, z_dim, h_dim):
        super(InfoCatConvVAE, self).__init__()

        self.in_dim = in_dim
        self.num_class = num_class
        self.sub_dim = sub_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=3)
        self.fc1 = nn.Linear(64*5*5, h_dim)

        self.fc20 = nn.Linear(self.h_dim, self.num_class)
        self.fc21 = nn.Linear(self.h_dim + self.num_class, self.z_dim)
        self.fc22 = nn.Linear(self.h_dim + self.num_class, self.z_dim)

        self.fc3 = nn.Linear(self.z_dim, h_dim)
        self.fc4 = nn.Linear(self.h_dim, 28*28*32)

        self.deconv1 = nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 28, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(28, 1, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(F.dropout(self.conv1(x), p=0.25))
        h1 = self.relu(F.dropout(self.conv2(h1), p=0.25))
        h1 = self.relu(F.dropout(self.conv3(h1), p=0.25))
        h1 = self.relu(F.dropout(self.conv4(h1), p=0.25))
        h1 = h1.view(h1.size(0), -1)
        h1 = self.fc1(h1)

        a = F.softmax(self.fc20(h1), dim=1)
        mu = self.fc21(torch.cat((h1, a), 1))
        logvar = self.fc22(torch.cat((h1, a), 1))

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
        return mu, logvar, a, allmu, allvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            z = eps.mul(std).add_(mu)
            return z
        else:
            return mu

    def decode(self, z):

        h3 = self.relu(F.dropout(self.fc3(z), p=0.25))
        out = self.relu(F.dropout(self.fc4(h3), p=0.25))
        # import pdb; pdb.set_trace()
        out = out.view(out.size(0), 32, 28, 28)
        out = self.relu(self.deconv1(out))
        out = self.relu(self.deconv2(out))
        out = self.relu(self.deconv3(out))
        out = self.sigmoid(self.conv5(out))

        return out

    def forward(self, x):
        mu, logvar, a, allmu, allvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, a, allmu, allvar


model = InfoCatConvVAE(in_dim, num_class, sub_dim, z_dim, h_dim)

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

def loss_function(coef, recon_x, x, mu, logvar, a, allmu, allvar):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), size_average=False)

    KLD = 0
    if args.cuda:
        for i in range(10):
            KLD += torch.sum(
                torch.t(a[:, i].repeat(model.z_dim, 1)) * KLDG(allmu[i, :, :], mupriorT[:, i].repeat(allmu.size(1), 1),
                                                               allvar[i, :, :]))
    else:
        for i in range(10):
            KLD += torch.sum(torch.t(a[:, i].repeat(model.z_dim, 1)) * KLDG(allmu[i, :, :],
                                                                            mupriorT.cpu()[:, i].repeat(allmu.size(1),
                                                                                                        1),
                                                                            allvar[i, :, :]))
    # KLD = KLDG(mu, 0, logvar)

    negH = torch.sum(a.mul(torch.log(a + 0.001)))

    return BCE + 2*negH + 2*KLD, BCE, negH, KLD


def sampling(k):
    allz = []
    if args.cuda:
        for i in range(k):
            mu = torch.FloatTensor(muprior.T)
            logvar = torch.FloatTensor(np.zeros((10, model.z_dim)))
            z = model.reparameterize(Variable(mu).cuda(), Variable(logvar).cuda())
            allz.append(z)
        sample = torch.cat(allz, dim=0)  # Variable(torch.randn(64, 20))
        sample = model.decode(sample)
    else:
        for i in range(k):
            mu = torch.FloatTensor(muprior.T)
            logvar = torch.FloatTensor(np.zeros((10, model.z_dim)))
            z = model.reparameterize(Variable(mu), Variable(logvar))
            allz.append(z)
        sample = torch.cat(allz, dim=0)  # Variable(torch.randn(64, 20))
        sample = model.decode(sample)

    return sample, Variable(torch.arange(0, 10).repeat(k).long())


def train(epoch):
    model.train()
    train_loss, train_reco_loss, train_negH, train_KLD = 0, 0, 0, 0
    class_loss = []
    for batch_idx, data in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar, a, allmu, allvar = model(data.unsqueeze(1))
        loss, la, lb, lc = loss_function(10, recon_batch, data, mu, logvar, a, allmu, allvar)

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

    torch.cuda.empty_cache()
    print('====> Epoch: {}, Reco: {:.2f}, negH: {:.2f}, KLD: {:.2f}, Class: {:.2f}'.format(
        epoch,
        train_reco_loss / len(train_loader.dataset),
        train_negH / len(train_loader.dataset),
        train_KLD / len(train_loader.dataset),
        np.mean(class_loss)))


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
        recon_batch, mu, logvar, a, allmu, allvar = model(data.unsqueeze(1))
        _, preds = torch.max(a, 1)
        loss, la, lb, lc = loss_function(10, recon_batch, data, mu, logvar, a, allmu, allvar)
        test_loss += loss.data[0]
        if i == 0:
            n = min(data.size(0), 10)
            comparison = torch.cat([data.view(data.size(0), 1, 28, 28)[:n],
                                    recon_batch.view(recon_batch.size(0), 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(),
                       '../InfoCatConvVAE_reco.png', nrow=n)

    if args.cuda:
        del data
        torch.cuda.empty_cache()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    test_lost_list.append(test_loss)


for epoch in range(1, n_epochs + 1):
    train(epoch)
    test(epoch)

    if epoch % 10 == 0:
        torch.save(model.state_dict(), '../InfoCatConvVAE.pt')
        delta = np.ones((10, model.z_dim))
        allz = []
        model.train()
        sample, labels = sampling(10)
        save_image(sample.view(100, 1, 28, 28).data,
                   '../InfoCatConvVAE_sample.png', nrow=10)
        if args.cuda:
            del sample
            torch.cuda.empty_cache()

# model.load_state_dict(torch.load('../InfoCatConvVAE.pt'))
torch.save(model.state_dict(), '../InfoCatConvVAE.pt')


# Kernel density estimation

from sklearn.neighbors import KernelDensity

mnkde_train = [m.flatten() for m in mnist_train]
mnkde_train = np.array(mnkde_train)[:10000]

mnkde_test = [m.flatten() for m in mnist_test]
mnkde_test = np.array(mnkde_test)

bandwidth = 0.31622776601683794
kd = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
kd.fit(mnkde_train)

sample, labels = sampling(1000)

lll_train = kd.score_samples(mnkde_train)
lll_test = kd.score_samples(mnkde_test)
lll_sample = kd.score_samples(sample.view(10000, 784).data.numpy())

print(np.mean(lll_train), np.mean(lll_test))
print(np.mean(lll_sample))


# Clustering metrics

from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, jaccard_similarity_score

test_data = torch.cat([Variable(torch.FloatTensor(m)).unsqueeze(0) for m in mnist_test], dim=0)

cuda_temp = args.cuda
args.cuda = False

_, _, a_test, _, _ = model.cpu().encode(test_data.unsqueeze(1))

args.cuda = cuda_temp

ars = adjusted_rand_score(labels_test, a_test.topk(1)[-1].data.numpy().flatten())
cs = completeness_score(labels_test, a_test.topk(1)[-1].data.numpy().flatten())
hs = homogeneity_score(labels_test, a_test.topk(1)[-1].data.numpy().flatten())
vms = v_measure_score(labels_test, a_test.topk(1)[-1].data.numpy().flatten())
jss = jaccard_similarity_score(labels_test, a_test.topk(1)[-1].data.numpy().flatten())

labelss = pd.DataFrame(np.array([labels_test, a_test.topk(1)[-1].data.numpy().flatten()]).T)
pred_labels = pd.DataFrame(np.zeros(labels_test.shape))

for i in np.unique(labels_test):
    l = labelss.loc[labelss[0] == i][1].value_counts().keys()[0]
    pred_labels[labelss[1] == l] = i

labelss = pd.DataFrame(np.array([labels_test, pred_labels.values.flatten()]).T)
print((labelss[0] == labelss[1]).sum()/10000)





######### Dessins utiles ##########

# Interpolation entre les centroïds #

allz = []
k = 10

for c in np.arange(num_class):
    if c == num_class - 1:
        for i in range(k):
            mu = (mupriorT.cpu()[:, c] * (9 - i) + mupriorT.cpu()[:, 0] * i) / 9 * 1.5
            allz.append(mu)
    else:
        for i in range(k):
            mu = (mupriorT.cpu()[:, c] * (9 - i) + mupriorT.cpu()[:, c+1] * i) / 9 * 1.5
            allz.append(mu)

sample = torch.stack(allz)  # Variable(torch.randn(64, 20))
sample = model.decode(sample).cpu()

save_image(sample.view(num_class*num_class, 1, 28, 28).data,
           '../InfoCatVAE_inter_centroids.png', nrow=num_class)


# Extension sur les axes #

allz = []
k = 10

for c in np.arange(num_class):
    for i in range(k):
        mu = mupriorT.cpu()[:, c] * i
        allz.append(mu)

sample = torch.stack(allz)  # Variable(torch.randn(64, 20))
sample = model.decode(sample).cpu()

save_image(sample.view(num_class * num_class, 1, 28, 28).data,
           '../InfoCatVAE_dim.png', nrow=num_class)

















