import os
import torch
import argparse
from model import seq
from dataset import getdata


def one_pass(opt):
    # Set device
    device = torch.device("cuda:0")
    # alphabet
    f_alphabet = "/home/dataset/NLP/ChinesePhrase/alphabet.json"
    alphabet = getdata.get_alphabet(f_alphabet)
    # Prepare model
    # net = seq.BiLSTM(num_classes=len(alphabet))
    # net = seq.AttentionNet(input_size=201, hidden_size=512, depth=3, head=5, num_classes=len(alphabet))
    net = seq.AttentionNet(input_size=200, hidden_size=512, depth=3, head=5, num_classes=len(alphabet), k=8)
    net = net.to(device)
    net.load_state_dict(torch.load(opt.param))
    net.eval()
    # Prepare data
    sequence = getdata.wrap_a_word(opt.word, alphabet)
    sequence.unsqueeze_(0)
    sequence = sequence.to(device)
    # forward
    with torch.no_grad():
        score = net(sequence)
    prob = torch.nn.functional.softmax(score, dim=1)
    # Show the top5-likelihood candidate characters
    v5, i5 = prob[0].topk(10)
    v5, i5 = v5.tolist(), i5.tolist()
    w5 = [alphabet[ii] for ii in i5]
    for item in zip(w5, v5):
        print(item, end='  ')
    print('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idc", type=str, default='1')
    parser.add_argument("--param", type=str, default='param/lstm.pkl')
    parser.add_argument("--word", type=str, default="新闻媒?", help="A word with ? to be filled.")
    opt = parser.parse_args()
    print(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.idc
    one_pass(opt)

