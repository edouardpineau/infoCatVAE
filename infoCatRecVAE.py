from utils import *


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
