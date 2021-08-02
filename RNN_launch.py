from NNets import IGIRNNCell,IGRNNCell
from NNets import RNN_constructor
from necessary_tools import get_matlab_data,data_prepare,batch_generator
import torch
import os

###############################################
PATH_TO_EXPERIMENT='./experiment_data/test_exp/'
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
SEQ_LENGTH=10 #length of seq in
BIDIRECTIONAL=False
DATA_NAME='BlackBoxData_80.mat'
device=torch.device('cuda:0')
###################################################
if __name__ == '__main__':
    seq_len, batch, input_size,num_layers,hidden_size=5,2,3,4,7
    if not os.path.isdir('./experiment_data/'):
        os.mkdir('./experiment_data/')
    if not os.path.isdir(PATH_TO_EXPERIMENT):
        os.mkdir(PATH_TO_EXPERIMENT)
    x,d = get_matlab_data(DATA_NAME)
    X,D = data_prepare(x,d)
    # inp = torch.randn(seq_len, BATCH_SIZE, INPUT_DIM)
    if BIDIRECTIONAL:
        states = [[torch.randn(BATCH_SIZE, HID_DIM)
               for _ in range(2)]
              for _ in range(N_LAYERS)]
    else:
        states = [torch.randn(BATCH_SIZE, HID_DIM)
                  for _ in range(N_LAYERS)]
    train_dataloader=batch_generator(X,D,SEQ_LENGTH,BATCH_SIZE)

    rnn = RNN_constructor(IGRNNCell,INPUT_DIM, HID_DIM, N_LAYERS, bidirectional=BIDIRECTIONAL)
    out, out_state = rnn(train_dataloader.__next__()[0], states)
    a = torch.nn.GRU(INPUT_DIM, HID_DIM, N_LAYERS, bidirectional=True)
    # custom_state = double_flatten_states(out_state)