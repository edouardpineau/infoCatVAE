from utils import *

def train(epoch, model, train_loader, optimizer, args):
    """
    :param epoch: current epoch 
    :param model: infoCatVAE model being trained
    :param args: args associated to current infoCatVAE
    :return: 
    """
    model.train()
    train_loss, train_reco_loss, train_negH, train_KLD = 0, 0, 0, 0
    class_loss = []
    for batch_idx, (data,_) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        
        # Categorical VAE part
        
        if args.is_convolutional:
            recon_batch, mu, logvar, a, allmu, allvar = model(data.unsqueeze(1))
        else:
            recon_batch, mu, logvar, a, allmu, allvar = model(data)

        # Information maximisation part

        sample, labels = sampling(model, 10, mupriorT)
        if args.cuda:
            sample = sample.cuda()
            labels = labels.cuda()
        _, _, a, _, _ = model.encode(sample)
        
        # Loss computation + backpropagation
        
        loss, la, lb, lc, mse_loss = loss_function(model, recon_batch.view(recon_batch.size(0), -1),
                                                   data.view(data.size(0), -1), a, allmu, allvar, mupriorT)
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


def test(epoch, model, test_loader, args):
    """
    :param epoch: current epoch 
    :param model: infoCatVAE model being trained
    :param args: args associated to current infoCatVAE
    :return: 
    """
    global test_lost_list
    model.eval()
    test_loss = 0

    for i, (data,_) in enumerate(test_loader):
        data = Variable(data, volatile=True)
        if args.cuda:
            data = data.cuda()
        
        if args.is_convolutional:
            recon_batch, mu, logvar, a, allmu, allvar = model(data.unsqueeze(1))
        else:
            recon_batch, mu, logvar, a, allmu, allvar = model(data)
            
        _, preds = torch.max(a, 1)
        loss, la, lb, lc, mse_loss = loss_function(model, recon_batch.view(recon_batch.size(0), -1),
                                                   data.view(data.size(0), -1), a, allmu, allvar, mupriorT)
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
