import torch
import torch.nn as nn
import torch.optim as optim

# import torchtext
# from torchtext.datasets import TranslationDataset, Multi30k
# from torchtext.data import Field, BucketIterator

import random
import math
import time
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout):
        super().__init__()
        #self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, n_layers, bidirectional = True)
        self.linear = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.hid_dim=enc_hid_dim
        self.n_layers=n_layers
        
    def forward(self, src):
        #embedded = self.embedding(src)
        #embedded = self.dropout(embedded)     
        outputs, hidden = self.rnn(embedded)
        concat_hiddens = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        concat_hiddens = self.linear(concat_hiddens)
        concat_hiddens = torch.tanh(concat_hiddens)
        return outputs, concat_hiddens

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        # print(hidden.shape)
        hidden = hidden.unsqueeze(1)
        # print(hidden.shape)
        hidden = hidden.repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        attention = torch.bmm(v, energy).squeeze(1)
        attention = self.softmax(attention)
        return attention

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim,n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.attention = Attention(enc_hid_dim, dec_hid_dim).to(device)
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim,n_layers)
        self.out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.hid_dim=enc_hid_dim
        self.n_layers=n_layers
        
    def forward(self, input, hidden, encoder_outputs):
        
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        
        att = self.attention(hidden, encoder_outputs)
        att = att.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)        
        weighted = torch.bmm(att, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim = 2)            
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        output = self.out(torch.cat((output, weighted, embedded), dim = 1))
        
        return output, hidden.squeeze(0)



# class Seq2Seq(nn.Module):
#     def __init__(self, encoder, decoder, device):
#         super().__init__()
        
#         self.encoder = encoder
#         self.decoder = decoder
#         self.device = device
        
#         assert encoder.hid_dim == decoder.hid_dim, \
#             "Hidden dimensions of encoder and decoder must be equal!"
#         assert encoder.n_layers == decoder.n_layers, \
#             "Encoder and decoder must have equal number of layers!"
        
#     def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
#         #src = [src sent len, batch size]
#         #trg = [trg sent len, batch size]
#         #teacher_forcing_ratio is probability to use teacher forcing
#         #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
#         # Again, now batch is the first dimention instead of zero
#         batch_size = trg.shape[1]
#         max_len = trg.shape[0]
#         trg_vocab_size = self.decoder.output_dim
        
#         #tensor to store decoder outputs
#         outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
#         #last hidden state of the encoder is used as the initial hidden state of the decoder
#         hidden, cell = self.encoder(src)
        
#         #first input to the decoder is the <sos> tokens
#         input = trg[0,:]
        
#         for t in range(1, max_len):
            
#             output, hidden, cell = self.decoder(input, hidden, cell)
#             outputs[t] = output
#             teacher_force = random.random() < teacher_forcing_ratio
#             top1 = output.max(1)[1]
#             input = (trg[t] if teacher_force else top1)
        
#         return outputs

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
 
        encoder_outputs, hidden = self.encoder(src)
                
        input = trg[0,:]
        
        for t in range(1, max_len):
            
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            
            outputs[t] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            
            top1 = output.argmax(1) 

            input = trg[t] if teacher_force else top1

        return outputs