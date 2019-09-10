import torch
import numpy as np
from dataset import getdata


def default_dataloader(batch_size=100):
    lmdb_dir = "/home/dataset/NLP/ChinesePhrase/word_emb_lmdb/"
    f_alphabet = "/home/dataset/NLP/ChinesePhrase/alphabet.json"
    dataloader = getdata.getloader(lmdb_dir, f_alphabet, batch_size=batch_size)
    return dataloader


class LrController(object):
    def __init__(self, threshold=10, eps=1e-8, decay_rate=0.3, init_lr=1e-2, min_lr=1e-5):
        self.threshold = threshold
        self.eps = eps
        self.decay_rate = decay_rate
        self.init_lr = init_lr
        self.min_lr = min_lr

    def cannot_drop_loss(self, loss_list):
        losses = np.array(loss_list)
        count = np.sum(np.diff(losses) > -self.eps)
        if count >= self.threshold:
            return True
        else:
            return False

    def update(self, lr, loss_list):
        if self.cannot_drop_loss(loss_list):
            lr = lr * self.decay_rate
            if lr < self.min_lr:
                lr = self.init_lr
            loss_list = []
        return lr, loss_list


class NegSamplingCELoss(object):
    def __init__(self, num_neg):
        self.num_neg = num_neg
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')

    def _sampling(self, b, nc, label):
        label = label - 1
        device = label.device
        # generate indexes without about positive samples
        idx = torch.arange(nc).repeat(b, 1).to(device)  # [b, nc]
        id_neg = torch.nonzero(idx != label.unsqueeze(1))
        id_neg = id_neg[:, 1].view(b, -1)  # [b, nc-1]
        # generate indexes of id_neg
        id_id = torch.randperm(nc-1)[:self.num_neg]
        id_id = id_id.to(device)
        # indexes to be used on the pred
        idx_sample = torch.cat((label.unsqueeze(1), id_neg[:, id_id]), dim=1)
        return idx_sample

    def __call__(self, pred, label):
        """
        Randomly sample a limited number of negative samples with the unique positive sample to feed the CE loss.
        :param pred: [b, nc]
        :param label: [b], exclude 0
        :return:
        """
        b, nc = pred.size()
        assert self.num_neg < nc, "The number of negative samples must be less than the numbere of classes!"
        idx = self._sampling(b, nc, label)  # [b, 1+num_neg]
        score_sampled = pred.gather(1, idx)
        label_modified = torch.zeros(b).long().to(label.device)
        loss = self.loss(score_sampled, label_modified)
        return loss


if __name__ == '__main__':
    loss_fn = NegSamplingCELoss(9)
    device = torch.device("cuda:1")
    pred = torch.rand(5, 20).to(device)
    label = torch.LongTensor([13, 12, 4, 1, 17]).to(device)
    loss = loss_fn(pred, label)
    print(loss)
