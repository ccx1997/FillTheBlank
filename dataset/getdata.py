import torch
from torch.utils.data import Dataset, DataLoader, sampler
import lmdb
import json
import numpy as np
import random
import math


def get_alphabet(f_alphabet):
    with open(f_alphabet, 'r', encoding='utf-8') as f:
        alphabet = sorted(json.load(f))
    return alphabet


class WordData(Dataset):
    def __init__(self, word_lmdb_dir, f_alphabet_json):
        super(WordData, self).__init__()
        self.env = lmdb.open(word_lmdb_dir, readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.nSamples = int(txn.get('nSamples'.encode()).decode())
            self.nSub = []
            for i in range(3, 15):
                key_sub = 'num_' + str(i)
                self.nSub.append(int(txn.get(key_sub.encode()).decode()))
        self.alphabet = get_alphabet(f_alphabet_json)
        print('Loaded {0} words'.format(self.nSamples))

    def __len__(self):
        return self.nSamples

    def word2indexes(self, word):
        indexes = torch.tensor([self.alphabet.index(c) for c in word]) + 1  # 0: blank_token
        return indexes

    def random_drop(self, sequence, idx, T, drop_proportion=0.4, drop_rate=0.5):
        y = sequence[idx].item()
        sequence[idx] = 0
        if random.random() < drop_rate:
            index = list(range(T))
            random.shuffle(index)
            num_drop = random.randint(1, math.ceil(T * drop_proportion))
            sequence[index[:num_drop]] = 0
        return sequence, y

    def __getitem__(self, idx):
        key_word = "%d_0_label" % idx
        with self.env.begin(write=False) as txn:
            word = txn.get(key_word.encode()).decode()
        seq = self.word2indexes(word)
        length = len(word)
        id_blank = random.randint(0, length-1)
        seq, label = self.random_drop(seq, id_blank, length)
        return seq, label, id_blank


class DoubleShuffleSampler(sampler.Sampler):
    """
    A sampler which could be used to dataset where the sequence data has several groups with different length.
    Dataset is like this: [...n1...], [...n2...], [...n3...], ...
    where each group has their size different from others.
    We want a single batch to be from a single group, while keeping the batches shuffled.
    So we just follow the 2 steps: randomly shuffle indexes with each group; shuffle all batches.
    """
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        all_indexes = []    # index of the items in dataset
        all_indexes2 = []   # index of the above index value
        start_index = 0
        for i, n_sub in enumerate(self.data_source.nSub):
            sub_indexes = list(range(start_index, start_index + n_sub))
            random.shuffle(sub_indexes)     # the first shuffle within group
            all_indexes.extend(sub_indexes)
            num_sub_batches = n_sub // self.batch_size
            num_remain = n_sub % self.batch_size
            sub_indexes2 = start_index + np.arange(num_sub_batches) * self.batch_size
            sub_indexes2 = sub_indexes2.tolist()
            if num_remain != 0:
                sub_indexes2.append(start_index + n_sub - self.batch_size)  # store the start index of all batches
            all_indexes2.extend(sub_indexes2)
            start_index += n_sub
        random.shuffle(all_indexes2)    # the second shuffle about batches(denoted by the start index)
        # complete other indexes besides the start ones. Use numpy to accelerate
        tmp_all_indexes2 = np.array(all_indexes2)
        tmp_expand = tmp_all_indexes2.repeat(self.batch_size).reshape(len(all_indexes2), self.batch_size)
        tmp_other = np.arange(self.batch_size).reshape(1, -1)
        tmp = tmp_expand + tmp_other
        tmp = tmp.ravel().tolist()
        all_indexes = np.array(all_indexes)
        final_indexes = all_indexes[tmp]
        return iter(final_indexes.tolist())


def getloader(word_lmdb_dir, f_alphabet_json, batch_size):
    word_data = WordData(word_lmdb_dir, f_alphabet_json)
    dataloader = DataLoader(word_data, batch_size=batch_size, sampler=DoubleShuffleSampler(word_data, batch_size),
                            num_workers=4)
    return dataloader


def wrap_a_word(word, alphabet):
    indexes = []
    for c in word:
        indexes.append(0 if c == "?" else alphabet.index(c)+1)
    indexes = torch.tensor(indexes)
    return indexes


if __name__ == "__main__":
    lmdb_dir = "/home/dataset/NLP/ChinesePhrase/word_emb_lmdb/"
    f_alphabet = "/home/dataset/NLP/ChinesePhrase/alphabet.json"
    # test dataset
    mydataset = WordData(lmdb_dir, f_alphabet)
    seq, label, id_blank = mydataset[1200000]
    word = ''.join([mydataset.alphabet[k-1] for k in seq])
    char_to_fill = mydataset.alphabet[label-1]
    print(seq, label)
    print(word, char_to_fill, id_blank)
    # test dataloader
    myloader = getloader(lmdb_dir, f_alphabet, batch_size=8)
    # display some results
    print(len(myloader))
    dataiter = iter(myloader)
    seqs, labels, id_blanks = dataiter.next()
    print(seqs, labels, id_blanks)  # [8, T], [8]
