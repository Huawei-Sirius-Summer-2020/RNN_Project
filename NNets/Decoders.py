import torch
import torch.nn as nn
from .Attentions import Attention
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.attention = Attention(enc_hid_dim, dec_hid_dim).to(device)
        #         self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim) + output_dim + emb_dim, dec_hid_dim, n_layers)
        self.out = nn.Linear((enc_hid_dim * 2) + output_dim + dec_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.hid_dim = enc_hid_dim
        self.n_layers = n_layers

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        #         print('inp={}'.format(input))
        #         embedded = self.embedding(input)
        #         embedded = self.dropout(embedded)

        att = self.attention(hidden, encoder_outputs)
        #         print('hid={},enc_outputs={},input={},att={}'.format(hidden.shape,encoder_outputs.shape,input.shape,att.shape))

        att = att.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(att, encoder_outputs)
        #         print('dec enc_out={}, weighted={}, att={}'.format(encoder_outputs.shape,weighted.shape,att.shape))
        weighted = weighted.permute(1, 0, 2)
        #         print('weight={}'.format(torch.unique(weighted)))
        rnn_input = torch.cat((input, weighted), dim=2)
        #         print('dec weighted={}, rnn_inp={}'.format(weighted.shape,rnn_input.shape))

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        #         print(torch.unique(rnn_input))
        #         print(torch.unique(weighted))
        #         print('kjnfrkjernfkjen')
        #         embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        input = input.squeeze(0)
        #         print('dec weighted={}, output={}'.format(weighted.shape,output.shape))

        #         print('out_shape={}'.format(torch.cat((output, weighted, input), dim = 1).shape))
        output = self.out(torch.cat((output, weighted, input), dim=1))
        #         print('dec_out={}'.format(output))

        return output, hidden.squeeze(0)