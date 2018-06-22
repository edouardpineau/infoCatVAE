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


########## Model ##########

loss_classifier = nn.NLLLoss()
n_epochs = 5000
batch_size = 64

in_dim = 784
num_class = 40
sub_dim = 2
z_dim = num_class*sub_dim
h_dim = 400

test_class_perf = []
test_lost_list = []

model = InfoCatVAE(in_dim, num_class, sub_dim, z_dim, h_dim)

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
