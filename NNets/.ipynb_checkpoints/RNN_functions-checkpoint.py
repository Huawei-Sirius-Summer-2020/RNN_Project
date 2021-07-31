import torch.nn as nn


class IGRNNCell(nn.Module):



    def __init__(self, input_size, hidden_size, bias=True):
        super(IGRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 1 * hidden_size, bias=bias)
        self.reset_parameters()

    
    def forward(self, x, hidden):
        
        x = x.view(-1, x.size(1))
        
        gate_x = self.x2h(x) 
        gate_h = self.h2h(hidden)
        
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        
        i_r, i_i, i_n = gate_x.chunk(3, 1)
#         h_r, h_i, h_n = gate_h.chunk(3, 1)
        
        
        resetgate = F.sigmoid(i_r)
        inputgate = F.sigmoid(i_i)
        newgate = F.tanh(i_n + (resetgate * gate_h))
        
        hy = newgate + inputgate * (hidden - newgate)
        
        
        return hy
    
class IGIRNNCell(nn.Module):


    def __init__(self, input_size, hidden_size, bias=True):
        super(IGIRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 2 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i = gate_x.chunk(2, 1)
        #         h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = F.sigmoid(i_r)
        inputgate = F.sigmoid(i_i)
        newgate = F.tanh(resetgate * gate_h)

        hy = newgate + inputgate * (hidden - newgate)

        return hy

class IGIRNN(nn.Module):
    def __init__(self,input_size,hidden_size,bias=True):