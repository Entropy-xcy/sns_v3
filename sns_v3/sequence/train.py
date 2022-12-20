import torch
from tqdm import tqdm


def train(net, device, num_epoch, optimizer, train_loader, dataset, dataset_length, checkpt=1):
    net.train()
    loss_list = []
    for epoch in range(num_epoch):
        print('Epoch: ', epoch + 1)
        train_loss = 0
        for batch in tqdm(train_loader):
            X, X_mask, y, y_mask = batch
            X, X_mask, y, y_mask = X.to(device), X_mask.to(device), y.to(device), y_mask.to(device)
            loss = net(input_ids=X, attention_mask=X_mask, labels=y).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        print('Loss: ', train_loss)
        loss_list.append(train_loss)
        if checkpt != 0:
            if (epoch + 1) == checkpt * num_epoch:  # save model checkpoint every (checkpt)% epochs; 0% means no check point saved
                print('Saving model at epoch ', epoch + 1)
                model_checkpoint = 'seq2seq_' + dataset + '_length=' + str(dataset_length) + '_epoch=' + str(
                    epoch + 1) + '.pt'
                torch.save(net.state_dict(), model_checkpoint)

    return loss_list

def test(net, test_loader):
    net.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            X, X_mask, y, y_mask = batch
            loss = net(input_ids=X, attention_mask=X_mask, labels=y).loss
            test_loss += loss.item()
            test_loss /= len(test_loader)
            print('Test Loss: ', test_loss)

    return test_loss



