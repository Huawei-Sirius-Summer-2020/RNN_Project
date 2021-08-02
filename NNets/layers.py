import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple
import torch.jit as jit


def RNN_constructor(cell, input_size, hidden_size, num_layers, bias=True,
                    batch_first=False, bidirectional=False):
    '''Returns a ScriptModule that mimics a PyTorch native LSTM.'''

    # The following are not implemented.
    assert bias
    assert not batch_first

    if bidirectional:
        stack_type = StackedLayers2
        layer_type = BidirLayer
        dirs = 2
    else:
        stack_type = StackedLayers
        layer_type = Layer
        dirs = 1

    return stack_type(num_layers, layer_type,
                      first_layer_args=[cell, input_size, hidden_size],
                      other_layer_args=[cell, hidden_size * dirs,
                                        hidden_size]
                      )


class Layer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(Layer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(self, inp, state):
        BATCH_SIZE = 10
        HID_DIM = 5
        inputs = inp.unbind(0)
        # outputs = []
        outputs = torch.zeros(size=(1, BATCH_SIZE, HID_DIM))
        for i in range(len(inputs)):
            state = self.cell(inputs[i], state)
            # outputs += [state]
            outputs = torch.cat((outputs, torch.unsqueeze(state, 0)), dim=0)
        # return torch.stack(outputs), state
        return outputs[1:], state


class ReverseLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(ReverseLayer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(self, inp, state):
        BATCH_SIZE = 10
        HID_DIM = 5
        inputs = inp.unbind(0)
        # outputs = []
        outputs = torch.zeros(size=(1, BATCH_SIZE, HID_DIM))
        l_inputs = len(inputs)
        for i in range(l_inputs):
            j = l_inputs - i - 1
            state = self.cell(inputs[j], state)
            # outputs = [state] + outputs
            outputs = torch.cat((torch.unsqueeze(state, 0), outputs), dim=0)
        # return torch.stack(outputs), state
        return outputs[:-1], state


class BidirLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(BidirLayer, self).__init__()
        self.directions = nn.ModuleList([
            Layer(cell, *cell_args),
            ReverseLayer(cell, *cell_args)
        ])

    def forward(self, inp, states):
        BATCH_SIZE = 10
        HID_DIM = 5
        SEQ_LEN = 10
        outputs = torch.zeros(size=(1, SEQ_LEN + 1, BATCH_SIZE, HID_DIM))
        # outputs = []
        output_states = torch.zeros(size=(1, BATCH_SIZE, HID_DIM))
        # output_states = []
        for i, direction in enumerate(self.directions):
            state = states[i]
            out, out_state = direction(inp, state)
            # outputs += [out]
            outputs = torch.cat((outputs, torch.unsqueeze(out, 0)), dim=0)
            output_states = torch.cat((output_states, torch.unsqueeze(out_state, 0)), dim=0)
            # output_states += [out_state]
        # return torch.cat(outputs, -1), output_states
        return torch.cat((outputs[1], outputs[2]), dim=-1), output_states[1:]


def init_stacked_cells(num_layers, layer, first_layer_args, other_layer_args):
    layers = [layer(*first_layer_args)] + [layer(*other_layer_args)
                                           for _ in range(num_layers - 1)]
    return nn.ModuleList(layers)


class StackedLayers(jit.ScriptModule):
    __constants__ = ['layers']  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super(StackedLayers, self).__init__()
        self.layers = init_stacked_cells(num_layers, layer, first_layer_args,
                                         other_layer_args)

    @jit.script_method
    # def forward(self, input: Tensor, states: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
    def forward(self, input, states):
        # List[LSTMState]: One state per layer
        # output_states = jit.annotate(List[Tensor], [])
        BATCH_SIZE = 10
        HID_DIM = 5
        output_states = torch.zeros(size=(1, 1, BATCH_SIZE, HID_DIM))  # здесь возможно (1, 2, BATCH_SIZE, HID_DIM)?
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            # output_states += [out_state]
            output_states = torch.cat((output_states, torch.unsqueeze(out_state, 0)), dim=0)
            i += 1

        return output, output_states[1:]


class StackedLayers2(jit.ScriptModule):
    __constants__ = ['layers']  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super(StackedLayers2, self).__init__()
        self.layers = init_stacked_cells(num_layers, layer, first_layer_args,
                                         other_layer_args)

    # @jit.script_method
    # def forward(self, input: Tensor, states: List[List[Tensor]]) -> Tuple[Tensor, List[List[Tensor]]]:
    def forward(self, input, states):
        BATCH_SIZE = 10
        HID_DIM = 5
        # output_states = jit.annotate(List[List[Tensor]], [])
        output_states = torch.zeros(size=(1, 2, BATCH_SIZE, HID_DIM))
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            # output_states += [out_state]
            output_states = torch.cat((output_states, torch.unsqueeze(out_state, 0)), dim=0)
            i += 1
        return output, output_states[1:]
