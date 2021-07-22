import os
import tensorboardX
import time
from necessary_tools import *
from NNets import *

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
DATA_NAME='BlackBoxData_80.mat'
device=torch.device('cuda:0')
######################




def train(model, iterator, optimizer, criterion, clip, train_history=None, valid_history=None):
    model.train()

    epoch_loss = 0
    history = []
    output_sig = []
    output_sig_for_acc = torch.zeros((1, 2)).to(device)
    i = 1
    for i, batch in enumerate(iterator):
        src = batch[0].to(device)
        trg = batch[1].to(device)

        optimizer.zero_grad()

        output = model(src, trg)
        output_sig = np.hstack((output_sig,
                                np.apply_along_axis(lambda args: [complex(*args)],
                                                    1, (output[-1, :, :].data.cpu().numpy())).reshape(-1)))
        output_sig_for_acc = torch.cat((output_sig_for_acc, output[-1, :, :].data))
        output = output.view(-1)
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


if __name__ == '__main__':
    if not os.path.isdir('./experiment_data/'):
        os.mkdir('./experiment_data/')
    if not os.path.isdir(PATH_TO_EXPERIMENT):
        os.mkdir(PATH_TO_EXPERIMENT)
    x,d = get_matlab_data(DATA_NAME)
    X,D = data_prepare(x,d)
    writer = tensorboardX.SummaryWriter(PATH_TO_EXPERIMENT)

    torch.manual_seed(10)

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, HID_DIM,1, DEC_DROPOUT)

    # dont forget to put the model to the right device
    model = Seq2Seq(enc, dec, device).to(device)
    train_dataloader=batch_generator(X,D,10,6)

    model.to(device)

    
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
        scheduler.step()
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        signal = signal.unsqueeze(1)
        signal = signal[1:,:,:].detach().cpu()
        accuracy.append(NMSE(X.cpu(),D.cpu()-signal).item())

        if accuracy[-1] < best_valid_loss:
            best_valid_loss = accuracy[-1] 
            torch.save(model.state_dict(), PATH_TO_EXPERIMENT+'/tut1-model.pt')

        train_history.append(train_loss)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {accuracy[-1]:.3f}')
        plot_to_tensorboard(writer,signal_for_drawing,epoch,x,d)




#     break