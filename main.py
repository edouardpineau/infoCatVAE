from InfoCatVAE import InfoCatVAE
from utils import *


########## Parameters ##########

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
	
	

########## Train and test functions ##########


def train(epoch, model):
    model.train()
    train_loss, train_reco_loss, train_negH, train_KLD = 0, 0, 0, 0
    class_loss = []
    for batch_idx, data in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar, a, allmu, allvar = model(data)
        loss, la, lb, lc, mse_loss = loss_function(model, recon_batch, data, a, allmu, allvar, mupriorT)

        loss.backward(retain_graph=True)

        # Adversarial learning of classes

        sample, labels = sampling(model, 10, mupriorT)
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


########## Model parameters ##########

loss_classifier = nn.NLLLoss()
n_epochs = 5000
batch_size = 64

in_dim = 784
num_class = 10
sub_dim = 2
z_dim = num_class*sub_dim
h_dim = 400

lmbda = 2



test_class_perf = []
test_lost_list = []

model = InfoCatVAE(in_dim, num_class, sub_dim, z_dim, h_dim, lmbda, args.cuda)


########## Prior ##########

mupriorT = prior_construction(model)

if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, n_epochs + 1):
    train(epoch)
    test(epoch)

    if epoch % 10 == 0:
        torch.save(model.state_dict(), '../InfoCatVAE' + str(epoch) + '.pt')
        sample, labels = sampling(10)
        save_image(sample.view(10*num_class, 1, 28, 28).data,
                   '../InfoCatVAE' + str(epoch) + '.png', nrow=num_class)