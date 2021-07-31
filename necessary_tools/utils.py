import scipy.io
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
###data preparing
def get_matlab_data(data_name='BlackBoxData_80.mat'):
    name = 'Data/'+data_name

    mat = scipy.io.loadmat(name)
    x = np.array(mat['x']).reshape(-1,1)[:]/2**15
    d = np.array(mat['y']).reshape(-1,1)[:]/2**15
    return x,d
def data_prepare(x,d):
    x_real, x_imag = torch.from_numpy(np.real(x)), torch.from_numpy(np.imag(x))
    d_real, d_imag = torch.from_numpy(np.real(d)), torch.from_numpy(np.imag(d))
    X = torch.DoubleTensor(torch.cat((x_real, x_imag))).reshape(2,-1,1).type(torch.FloatTensor).permute(1,2,0)
    D = torch.DoubleTensor(torch.cat((d_real, d_imag))).reshape(2,-1,1).type(torch.FloatTensor).permute(1,2,0)
    return X,D

def batch_generator(X, D, count_of_delays_elements, batch_size=10):
    for j in range(0, X.shape[0], batch_size):
        yield torch.cat(([X[i - count_of_delays_elements:i + 1, :, :] if i > count_of_delays_elements else \
                              torch.cat((torch.zeros(count_of_delays_elements - i, 1, 2), X[:i + 1, :, :]), dim=0)
                          for i in range(j, batch_size + j)]), dim=1), \
              torch.cat(([D[i - count_of_delays_elements:i + 1, :, :] if i > count_of_delays_elements else \
                              torch.cat((torch.zeros(count_of_delays_elements - i, 1, 2), D[:i + 1, :, :]), dim=0)
                          for i in range(j, batch_size + j)]), dim=1)
###ACCURACY metrics
def NMSE(X, E):
    return 10 * torch.log10((torch.pow((E).norm(dim=0), 2)).sum() / (torch.pow((X).norm(dim=0), 2)).sum())
### model initialization
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param, -0.008, 0.008)

### tensorboard data
def plot_to_tensorboard(writer, outputs, step, x, d):
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

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.psd(x.reshape(-1), NFFT=2048, label='X')
    ax.psd(d.reshape(-1), NFFT=2048, label='D')
    #     outputs=torch.complex(outputs[0,:,:],outputs[1,:,:]).detach().cpu().view(-1)
    ax.psd(outputs, NFFT=2048, label='output')
    ax.psd(d.reshape(-1) - outputs, NFFT=2048, label='eOut')
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
    writer.add_image('PSD_plot', img.transpose(2, 0, 1), step)

    plt.close(fig)





def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
