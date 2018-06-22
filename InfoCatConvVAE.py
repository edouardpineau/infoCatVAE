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

