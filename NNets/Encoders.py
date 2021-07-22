import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout):
        super().__init__()
        #         self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(input_dim, enc_hid_dim, n_layers, bidirectional=True)
        self.linear = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.hid_dim = enc_hid_dim
        self.n_layers = n_layers

    def forward(self, src):
        #         embedded = self.embedding(src)
        #         embedded = self.dropout(embedded)
        #         print(embedded.shape)
        outputs, hidden = self.rnn(src)
        #         print('enc hidden={}'.format(hidden.shape))
        concat_hiddens = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        concat_hiddens = self.linear(concat_hiddens)
        concat_hiddens = torch.tanh(concat_hiddens)
        return outputs, concat_hiddens
