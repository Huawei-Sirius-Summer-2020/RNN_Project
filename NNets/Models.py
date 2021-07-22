import torch
import torch.nn as nn
import random

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        max_len = trg.shape[0]

        outputs = torch.zeros_like(trg).to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        input = trg[0]

        for t in range(1, max_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t, :, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output
            input = trg[t] if teacher_force else top1

        return outputs
