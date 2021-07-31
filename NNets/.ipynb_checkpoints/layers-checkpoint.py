import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch import Tensor

class Layer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(Layer, self).__init__()
        self.cell = cell(*cell_args)
    
    def forward(self, inp, state):
        inputs = inp.unbind(0)
        outputs = []
        for i in range(len(inputs)):
            state = self.cell(inputs[i], state)
            outputs += [state]
        return torch.stack(outputs), state


class ReverseLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(ReverseLayer, self).__init__()
        self.cell = cell(*cell_args)
    
    def forward(self, inp, state):
        inputs = inp.unbind(0)
        outputs = []
        l_inputs = len(inputs)
        for i in range(l_inputs):
            j = l_inputs - i - 1
            state = self.cell(inputs[j], state)
            outputs = [state] + outputs
        return torch.stack(outputs), state


class BidirLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(BidirLayer, self).__init__()
        self.directions = nn.ModuleList([
            Layer(cell, *cell_args),
            ReverseLayer(cell, *cell_args)
        ])
    
    def forward(self, inp, states):
        outputs = []
        output_states = []
        for i, direction in enumerate(self.directions):
            state = states[i]
            out, out_state = direction(inp, state)
            outputs += [out]
            output_states += [out_state]
        return torch.cat(outputs, -1), output_states