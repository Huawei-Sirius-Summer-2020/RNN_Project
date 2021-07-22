import torch
import torch.nn as nn
import numpy as np
import scipy.io
import torch.utils.tensorboard
import os
import matplotlib.pyplot as plt
import tensorboardX
import time
import random

##### Parameters of model
M=3      # size of secondary dim
PATH_TO_EXPERIMENT='./experiment_data/seq2seq_att_hid_50/'
INPUT_DIM = 2
OUTPUT_DIM = 2
ENC_EMB_DIM = 2
DEC_EMB_DIM = 50
HID_DIM = 50
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
N_EPOCHS = 10
BATCH_SIZE = 10
CLIP = 1
device=torch.device('cuda:0')
######################

class IGIRNNCell(nn.Module):
    """
    An implementation of GRUCell.

    """

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

class IGIRNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(IGIRNNModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.gru_cell = IGIRNNCell(input_dim, hidden_dim, layer_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        # print(x.shape,"x.shape")100, 28, 28
        #         if torch.cuda.is_available():
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.fc.weights.device)
        #         else:
        #             h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        outs = []

        hn = h0[0, :, :]

        for seq in range(x.size(1)):
            hn = self.gru_cell(x[:, seq, :], hn)
            outs.append(hn)

        out = outs[-1].squeeze()

        out = self.fc(out)
        # out.size() --> 100, 10
        return out

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout):
        super().__init__()
#         self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(input_dim, enc_hid_dim, n_layers, bidirectional = True)
        self.linear = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.hid_dim=enc_hid_dim
        self.n_layers=n_layers
        
    def forward(self, src):
#         embedded = self.embedding(src)
#         embedded = self.dropout(embedded)     
#         print(embedded.shape)
        outputs, hidden = self.rnn(src)
#         print('enc hidden={}'.format(hidden.shape))
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
#         self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim)+output_dim + emb_dim, dec_hid_dim,n_layers)
        self.out = nn.Linear((enc_hid_dim*2) + output_dim + dec_hid_dim , output_dim)
        self.dropout = nn.Dropout(dropout)
        self.hid_dim=enc_hid_dim
        self.n_layers=n_layers
        
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
        rnn_input = torch.cat((input, weighted), dim = 2)        
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
        output = self.out(torch.cat((output, weighted, input), dim = 1))
#         print('dec_out={}'.format(output))
        
        return output, hidden.squeeze(0)

def batch_generator(X,D,count_of_delays_elements,batch_size=10):
    for j in range(0,X.shape[0],batch_size):
        yield torch.cat(([X[i-count_of_delays_elements:i+1,:,:] if i > count_of_delays_elements else \
        torch.cat((torch.zeros(count_of_delays_elements-i,1,2),X[:i+1,:,:]),dim=0) 
            for i in range(j,batch_size+j)]),dim=1),\
        torch.cat(([D[i-count_of_delays_elements:i+1,:,:] if i > count_of_delays_elements else \
        torch.cat((torch.zeros(count_of_delays_elements-i,1,2),D[:i+1,:,:]),dim=0) 
            for i in range(j,batch_size+j)]),dim=1)
#         D[:,i:i+1,:]


def NMSE(X, E):
    return 10 * torch.log10((torch.pow((E).norm(dim=0), 2)).sum() / (torch.pow((X).norm(dim=0), 2)).sum())

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
        
        outputs = torch.zeros_like( trg).to(self.device)
 
        encoder_outputs, hidden = self.encoder(src)
#         print('seq2seq enc_outputs={} hidden={}'.format(encoder_outputs.shape,hidden.shape))
                
        input = trg[0]
        
        for t in range(1, max_len):
#             print('seq2seq_input={}'.format(input.shape))
            output, hidden = self.decoder(input, hidden, encoder_outputs)
#             print(output.shape)
            outputs[t,:,:] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            
            top1 = output
#             print('seq2seq_top1={}'.format(top1))

            input = trg[t] if teacher_force else top1

        return outputs

def plot_to_tensorboard(writer, outputs, step,x,d):
    """
    Takes a matplotlib figure handle and converts it using
    canvas and string-casts to a numpy array that can be
    visualized in TensorBoard using the add_image function

    Parameters:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        fig (matplotlib.pyplot.fig): Matplotlib figure handle.
        step (int): counter usually specifying steps/epochs/time.
    """
#     no=[]
#     for o in range(len(outputs)):
#         no.append(outputs[o][0].item()+1j*outputs[o][1].item())

    fig=plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.psd(x.reshape(-1),NFFT=2048,label='X')
    ax.psd(d.reshape(-1),NFFT=2048,label='D')
#     outputs=torch.complex(outputs[0,:,:],outputs[1,:,:]).detach().cpu().view(-1)
    ax.psd(outputs,NFFT=2048,label='output')
    ax.psd(d.reshape(-1)-outputs,NFFT=2048,label='eOut')
    ax.legend()
    ax.grid()
    # Draw figure on canvas
    fig.canvas.draw()

    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
    img = img / 255.0
    # img = np.swapaxes(img, 0, 2) # if your TensorFlow + TensorBoard version are >= 1.8

    # Add figure in numpy "image" to TensorBoard writer
#     print(img)
    writer.add_image('PSD_plot', img.transpose(2,0,1), step)
    
    plt.close(fig)
    
def init_weights(m):
    # <YOUR CODE HERE>
    for name, param in m.named_parameters():
        nn.init.uniform_(param, -0.008, 0.008)

def train(model, iterator, optimizer, criterion, clip, train_history=None, valid_history=None):
    model.train()
    
    epoch_loss = 0
    history = []
    output_sig=[]
    output_sig_for_acc=torch.zeros((1,2)).to(device)
    i=1
    for i, batch in enumerate(iterator):
        
        src = batch[0].to(device)
        trg = batch[1].to(device)
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        output_sig=np.hstack((output_sig,
                              np.apply_along_axis(lambda args: [complex(*args)],
                                                  1, (output[-1,:,:].data.cpu().numpy())).reshape(-1)))
        output_sig_for_acc=torch.cat((output_sig_for_acc,output[-1,:,:].data))
        output = output.view(-1)
#         print(output.shape)
        trg = trg.view(-1)
#         print(trg.shape)
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
        history.append(loss.item())
#         print('i={}'.format(i))

    return epoch_loss / i, output_sig, output_sig_for_acc

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# best_valid_loss = float('inf')

# PAD_IDX = TRG.vocab.stoi['<pad>']
# criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)
    
def NMSE(X, E):
    return 10 * torch.log10((torch.pow((E).norm(dim=2), 2)).sum() / (torch.pow((X).norm(dim=2), 2)).sum())

if __name__ == '__main__':
    if not os.path.isdir('./experiment_data/'):
        os.mkdir('./experiment_data/')
    if not os.path.isdir(PATH_TO_EXPERIMENT):
        os.mkdir(PATH_TO_EXPERIMENT)
    name = 'Data/BlackBoxData_80'
    # name = 'BlackBoxData'
    # name = '../BlackBoxData/data1'
    mat = scipy.io.loadmat(name)
    x = np.array(mat['x']).reshape(-1,1)[:]/2**15
    d = np.array(mat['y']).reshape(-1,1)[:]/2**15
    # x = np.array(mat['xE']).reshape(-1,1)/2**15
    # d = np.array(mat['d']).reshape(-1,1)/2**15
    # x, d = mat['xE'], mat['d']
    x_real, x_imag = torch.from_numpy(np.real(x)), torch.from_numpy(np.imag(x))
    d_real, d_imag = torch.from_numpy(np.real(d)), torch.from_numpy(np.imag(d))
    X = torch.DoubleTensor(torch.cat((x_real, x_imag))).reshape(2,-1,1).type(torch.FloatTensor).permute(1,2,0)
    D = torch.DoubleTensor(torch.cat((d_real, d_imag))).reshape(2,-1,1).type(torch.FloatTensor).permute(1,2,0)
    writer = tensorboardX.SummaryWriter(PATH_TO_EXPERIMENT)

    torch.manual_seed(10)

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, HID_DIM,1, DEC_DROPOUT)

    # dont forget to put the model to the right device
    model = Seq2Seq(enc, dec, device).to(device)
    train_dataloader=batch_generator(X,D,10,6)
# src,v=train_dataloader.__next__()
# model(src, v, teacher_forcing_ratio = 0.9)

#     model=GMP(POLYNOM_ORDER,COUNT_OF_DELAYS_ELEMENTS,passthrough=PASSTHROUGH,)
    model.to(device)
#     dataloader=batch_generator(X,D)

    
    model.apply(init_weights)
    train_history = []
    valid_history = []
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,10,12], gamma=0.8)
    loss_fn=nn.MSELoss()

    accuracy=[]
    train_dataloader=batch_generator(X,D,10,BATCH_SIZE)
    best_valid_loss=0
    for epoch in range(N_EPOCHS):

        train_dataloader = batch_generator(X, D, 10, BATCH_SIZE)

        start_time = time.time()

        train_loss , signal_for_drawing,signal= train(model, train_dataloader, optimizer, loss_fn, CLIP, train_history, valid_history)
    #     valid_loss = evaluate(model, valid_iterator, criterion)
        scheduler.step()
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        signal = signal.unsqueeze(1)
        signal = signal[1:,:,:].detach().cpu()
        accuracy.append(NMSE(X.cpu(),D.cpu()-signal).item())
        # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

        # clear_output(True)
#         ax[0].plot(accuracy, label='train NMSE')
#         ax[0].set_xlabel('Batch')
#         ax[0].set_title('Train loss')
#         ax[1].psd(x.reshape(-1),NFFT=2048,label='X')
#         ax[1].psd(d.reshape(-1),NFFT=2048,label='D')
#     #     outputs=torch.complex(outputs[0,:,:],outputs[1,:,:]).detach().cpu().view(-1)
#         ax[1].psd(no,NFFT=2048,label='output')
#         ax[1].psd(d.reshape(-1)-no,NFFT=2048,label='eOut')
#         ax[1].legend()
#         ax[1].grid()
#         # Draw figure on canvas
#     #     fig.canvas.draw()
#     #     if train_history is not None:
#     #         ax[1].plot(train_history, label='general train history')
#     #         ax[1].set_xlabel('Epoch')
#     #     if valid_history is not None:
#     #         ax[1].plot(valid_history, label='general valid history')
#     #     plt.legend()

#     #     plt.show()

        if accuracy[-1] < best_valid_loss:
            best_valid_loss = accuracy[-1] 
            torch.save(model.state_dict(), PATH_TO_EXPERIMENT+'/tut1-model.pt')

        train_history.append(train_loss)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {accuracy[-1]:.3f}')
        plot_to_tensorboard(writer,signal_for_drawing,epoch,x,d)




#     break