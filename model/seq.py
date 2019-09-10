import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM(nn.Module):
    def __init__(self, num_classes, f_lookup_ts=None, hidden_size=512):
        super(BiLSTM, self).__init__()
        if f_lookup_ts is not None:
            lookup_ts = torch.load(f_lookup_ts)
            lookup_ts = torch.cat((torch.zeros(1, lookup_ts.size(1)), lookup_ts), dim=0)
            self.register_buffer("lookup_ts", lookup_ts)
        else:
            self.register_buffer("lookup_ts", torch.zeros(num_classes+1, 200))
            print("Randomly initializing the lookup table. It is to be fed with pretrained model parameters.")
        self.rnn = nn.LSTM(input_size=200, hidden_size=hidden_size, num_layers=3, dropout=0.3, bidirectional=True)
        self.classifier = nn.Linear(hidden_size*2, num_classes)

    def idx2embedding(self, indexes):
        return self.lookup_ts[indexes]

    def forward(self, x, idx):
        """
        :param x: [b, T]
        :param idx: [b]
        :return:
        """
        embedding = self.idx2embedding(x)   # [b, T, d_emb]
        rnn_out, _ = self.rnn(embedding.permute(1, 0, 2))
        rnn_out = rnn_out.permute(1, 0, 2)  # [b, T, 2*hidden_size]
        b, T, c = rnn_out.size()
        # pad at both sides to deal with circumstances where the blank spot is at the beginning or end of a word.
        device = rnn_out.device
        rows = torch.arange(b).to(device)
        padding = torch.zeros(b, 1, c).to(device)
        rnn_out = torch.cat((padding, rnn_out, padding), dim=1)  # [b, T+2, c]
        idx = idx + 1
        context = torch.cat((rnn_out[rows, idx-1, :c//2],
                             rnn_out[rows, idx+1, c//2:]), dim=1)   # [b, c]
        output = self.classifier(context)
        return output


class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, depth=1):
        super(SelfAttention, self).__init__()
        self.fc0 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.linear_q = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.3),
        )
        self.linear_k = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.3),
        )
        self.linear_v = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.3),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.depth = depth

    def forward(self, x):
        """
        self attention.
        :param x: [b, T, input_size]
        :return: hs: [b, T, hidden_size]
        """
        hs = self.fc0(x)  # [b, T, hidden_size]
        for _ in range(self.depth):
            q = self.linear_q(hs)
            k = self.linear_k(hs)
            v = self.linear_v(hs)
            a = q.matmul(k.permute(0, 2, 1))  # [b, T, T]
            a = F.softmax(a, dim=2)
            hs = self.fc1(a.matmul(v))  # [b, T, hidden_size]
        return hs


class SelfAttentionRPR(SelfAttention):
    def __init__(self, input_size, hidden_size, k, pos_size=None, depth=1):
        """
        :param k: distant we would consider
        :param pos_size: size of embedding for every relative position
        :param depth: number of iterations
        """
        super(SelfAttentionRPR, self).__init__(input_size, hidden_size, depth)
        if pos_size is None:
            pos_size = hidden_size
        self.pos_embedding_key = nn.Parameter(torch.randn(k * 2 + 1, pos_size))
        self.pos_embedding_value = nn.Parameter(torch.randn(k * 2 + 1, pos_size))
        nn.init.kaiming_normal_(self.pos_embedding_key)
        nn.init.kaiming_normal_(self.pos_embedding_value)
        self.k = k

    def relative_pos_emb(self, T, pe):
        # Every row denotes a position
        base = torch.arange(self.k, self.k + T).repeat(T, 1)
        minus_d = torch.arange(T).unsqueeze(1)
        relative_mat_id = torch.clamp(base - minus_d, min=0, max=2 * self.k).to(pe.device)
        return pe[relative_mat_id.view(-1)].view(T, T, -1)

    def forward(self, x):
        """
        self attention.
        :param x: [b, T, input_size]
        :return: hs: [b, T, hidden_size]
        """
        T = x.size(1)
        rpr_key = self.relative_pos_emb(T, self.pos_embedding_key)
        rpr_value = self.relative_pos_emb(T, self.pos_embedding_value)
        hs = self.fc0(x)  # [b, T, hidden_size]
        for _ in range(self.depth):
            q = self.linear_q(hs)
            k = self.linear_k(hs)
            v = self.linear_v(hs)
            # query-key
            a = q.matmul(k.permute(0, 2, 1))  # [b, T, T]
            a_pos = rpr_key.matmul(q.unsqueeze(3)).squeeze(3)  # [b, T, T]
            a = a + a_pos
            a = F.softmax(a, dim=2)  # [b, T, T]
            # attention-value
            c = a.matmul(v)
            c_pos = a.unsqueeze(2).matmul(rpr_value).squeeze(2)  # [b, T, hidden_size]
            c = c + c_pos
            hs = self.fc1(c)  # [b, T, hidden_size]
        return hs


class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, depth, head, k=None):
        """
        :param depth: number of iterations
        :param head: number of heads
        :param k: distant we would consider in rpr
        """
        super(MultiHeadAttention, self).__init__()
        if k is not None:
            self.attention = nn.ModuleList([SelfAttentionRPR(
                input_size, hidden_size, k, depth=depth) for _ in range(head)])
        else:
            self.attention = nn.ModuleList([SelfAttention(input_size, hidden_size, depth) for _ in range(head)])

    def forward(self, x):
        ys = []
        for m in self.attention:
            ys.append(m(x))
        return torch.cat(ys, dim=2)


class AttentionNet(nn.Module):
    def __init__(self, input_size, hidden_size, depth, head, num_classes, f_lookup_ts=None, k=None):
        super(AttentionNet, self).__init__()
        if f_lookup_ts is not None:
            lookup_ts = torch.load(f_lookup_ts)
            lookup_ts = torch.cat((torch.zeros(1, lookup_ts.size(1)), lookup_ts), dim=0)
            self.register_buffer("lookup_ts", lookup_ts)
        else:
            self.register_buffer("lookup_ts", torch.zeros(num_classes+1, 200))
            print("Randomly initializing the lookup table. It is to be fed with pretrained model parameters.")
        self.mha = MultiHeadAttention(input_size, hidden_size, depth, head, k)
        self.cls = nn.Sequential(
            nn.Linear(hidden_size*head, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )
        self.use_rpr = True if k is not None else False

    def idx2embedding(self, indexes):
        return self.lookup_ts[indexes]

    def forward(self, x, idx):
        """
        self attention.
        :param x: [b, T]
        :param idx: [b]
        :return:
        """
        b, T = x.size()
        device = x.device
        rows = torch.arange(b).to(device)
        embedding = self.idx2embedding(x)  # [b, T, d_emb]
        if not self.use_rpr:
            pos_emb = torch.arange(T).float().repeat(b, 1).unsqueeze(2) / T  # [b, T, 1]
            embedding = torch.cat((embedding, pos_emb.to(device)), dim=2)  # [b, T, d_emb + 1]
        hs = self.mha(embedding)
        y = self.cls(hs[rows, idx, :])  # [b, num_classes]
        return y


if __name__ == '__main__':
    f_lookup = "/home/dataset/NLP/ChinesePhrase/lookup.pt"
    num_classes = 6405
    device = torch.device("cuda:1")
    # net = BiLSTM(num_classes, f_lookup).to(device)
    # net = SelfAttention(num_classes, f_lookup).to(device)
    net = AttentionNet(
        input_size=200, hidden_size=512, depth=3, head=5, num_classes=num_classes, f_lookup_ts=f_lookup, k=6).to(device)
    x = torch.tensor([[3, 0, 21, 0, 33], [0, 87, 0, 10, 97]]).to(device)
    idx = torch.tensor([3, 2]).to(device)
    y = net(x, idx)
    print(y.size())
