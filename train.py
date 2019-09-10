import os
import torch
import argparse
import utils
from model import seq


def main(opt):
    # Set device
    device = torch.device("cuda:0")
    # Prepare data
    train_loader = utils.default_dataloader(batch_size=opt.bs)
    # Prepare model
    f_lookup = "/home/dataset/NLP/ChinesePhrase/lookup.pt"
    # net = seq.BiLSTM(num_classes=len(train_loader.dataset.alphabet), f_lookup_ts=f_lookup)
    # net = seq.AttentionNet(input_size=201, hidden_size=512, depth=3, head=5,
    #                        num_classes=len(train_loader.dataset.alphabet), f_lookup_ts=f_lookup)
    net = seq.AttentionNet(input_size=200, hidden_size=512, depth=3, head=5,
                           num_classes=len(train_loader.dataset.alphabet), f_lookup_ts=f_lookup, k=8)
    net = net.to(device)
    if opt.pretrain:
        net.load_state_dict(torch.load(opt.pretrain), strict=False)
    # Prepare back-propagation tools
    loss_fn = utils.NegSamplingCELoss(num_neg=opt.num_neg)
    lr = opt.lr
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = utils.LrController(threshold=10, decay_rate=0.3, init_lr=1e-4, min_lr=8e-7)
    # Other preparations to control the training process
    best_loss = 10000.0
    loss_list = []
    # train
    print("Training...")
    for i in range(opt.epoch):
        running_loss = 0.0
        for j, (seqs, labels, ids_insert) in enumerate(train_loader):
            seqs, labels, ids_insert = seqs.to(device), labels.to(device), ids_insert.to(device)
            # forward propagation
            net.train()
            y = net(seqs, ids_insert)
            loss = loss_fn(y, labels)
            # backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Periodical check
            if j % opt.interval == opt.interval - 1:
                loss_avg = running_loss / opt.interval
                running_loss = 0.0
                # Save the best model
                if best_loss > loss_avg:
                    print("saving...", end=' ')
                    torch.save(net.state_dict(), opt.sav)
                    best_loss = loss_avg
                print("[{:3d}-{:4d}] lr: {:.6f}, loss: {:.4f}, best_loss: {:.4f}".format(
                    i+1, j+1, lr, loss_avg, best_loss))
                # Update lr according to the loss list
                loss_list.append(loss_avg)
                lr, loss_list = lr_scheduler.update(lr, loss_list)
                if len(loss_list) == 0:
                    optimizer.param_groups[0]['lr'] = lr
    print("Training process done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--idc", type=str, default='1')
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=100)
    parser.add_argument("--num_neg", type=int, default=199)
    parser.add_argument("--interval", type=int, default=500)
    parser.add_argument("--sav", type=str, default='model.pkl')
    parser.add_argument("--pretrain", type=str, default='')
    opt = parser.parse_args()
    print(opt)
    os.system("mkdir param")
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.idc
    main(opt)
