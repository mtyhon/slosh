import torch, sklearn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os, re, glob, h5py, matplotlib, shutil, math
import torch.utils.data as utils
print(torch.__version__)

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.autograd import Variable
from torch.distributions import Categorical
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic, norm
from copy import deepcopy

import math
import warnings
warnings.filterwarnings("ignore")


def squeeze(arr, minval, maxval, axis=0):
    """
    Returns version of 1D arr with values squeezed to range [minval,maxval]
    """
    #array is 1D
    minvals = np.ones(arr.shape)*minval
    maxvals = np.ones(arr.shape)*maxval

    #assure above minval first
    squeezed = np.max(np.vstack((arr,minvals)),axis=0)
    squeezed = np.min(np.vstack((squeezed,maxvals)),axis=0)

    return squeezed

def ps_to_array(freq, power, nbins=128, supersample=1,
                minfreq=3., maxfreq=283., minpow=3., maxpow=3e7, scale=False, fix=False):
    """
    Produce 2D array representation of power spectrum that is similar to Marc Hon's 2D images
    Written by Keaton Bell (bell@mps.mpg.de)
    This should be faster and more precise than writing plots to images
    Returns nbin x nbins image-like representation of the data
    freq and power are from power spectrum
    min/max freqs/powers define the array edges in same units as input spectrum
    if supersample == 1, result is strictly black and white (1s and 0s)
    if supersample > 1, returns grayscale image represented spectrum "image" density
    """
    # make sure integer inputs are integers
    nbins = int(nbins)
    supersample = int(supersample)
    # Set up array for output
    output = np.zeros((nbins, nbins))
    if supersample > 1:  # SUPERSAMPLE
        # Call yourself and flip orientation again
        supersampled = 1. - ps_to_array(freq, power, nbins=nbins * supersample, supersample=1,
                                        minfreq=minfreq, maxfreq=maxfreq, minpow=minpow, maxpow=maxpow, scale=scale)[::-1]
        for i in range(supersample):
            for j in range(supersample):
                output += supersampled[i::supersample, j::supersample]
        output = output / (supersample ** 2.)
    else:  # don't supersample
        # Do everything in log space
        freq_min_index = np.argmin(np.abs(freq - minfreq))
        freq_max_index = np.argmin(np.abs(freq - maxfreq))
        logfreq = np.log10(freq[freq_min_index:freq_max_index])
        logpower = np.log10(power[freq_min_index:freq_max_index])
   
        minlogfreq = np.log10(minfreq)
        maxlogfreq = np.log10(maxfreq)
        
        if scale:
            mean_logpower = np.mean(logpower)
            std_logpower = np.std(logpower)
            logpower = (logpower - mean_logpower) / std_logpower
            minlogpow = -5
            maxlogpow = 5
        else:
            minlogpow = np.log10(minpow)
            maxlogpow = np.log10(maxpow)

        # Define bins

        if not fix:
            xbinedges = np.linspace(np.log10(minfreq), np.log10(maxfreq), nbins + 1)
            xbinwidth = xbinedges[1] - xbinedges[0]
            ybinedges = np.linspace(np.log10(minpow), np.log10(maxpow), nbins + 1)
            ybinwidth = ybinedges[1] - ybinedges[0]
        else: # fixing the scale..
            xbinedges = np.linspace(minlogfreq, maxlogfreq, nbins + 1)
            xbinwidth = xbinedges[1] - xbinedges[0]
            ybinedges = np.linspace(minlogpow, maxlogpow, nbins + 1)
            ybinwidth = ybinedges[1] - ybinedges[0] 

        # resample at/near edges of bins and at original frequencies

        smalloffset = xbinwidth / (10. * supersample)  # to get included in lower-freq bin
        interpps = interp1d(logfreq, logpower, fill_value=(0,0), bounds_error=False)
        poweratedges = interpps(xbinedges)
        logfreqsamples = np.concatenate((logfreq, xbinedges, xbinedges - smalloffset))
        powersamples = np.concatenate((logpower, poweratedges, poweratedges))

        sort = np.argsort(logfreqsamples)
        logfreqsamples = logfreqsamples[sort]
        powersamples = powersamples[sort]

        # Get maximum and minimum of power in each frequency bin
        maxpow = binned_statistic(logfreqsamples, powersamples, statistic='max', bins=xbinedges)[0]
        minpow = binned_statistic(logfreqsamples, powersamples, statistic='min', bins=xbinedges)[0]
        # Convert to indices of binned power

        # Fix to fall within power range
        minpowinds = np.floor((minpow - minlogpow) / ybinwidth)
        minpowinds = squeeze(minpowinds, 0, nbins).astype('int')
        maxpowinds = np.ceil((maxpow - minlogpow) / ybinwidth)
        maxpowinds = squeeze(maxpowinds, 0, nbins).astype('int')

        # populate output array

        for i in range(nbins):
            output[minpowinds[i]:maxpowinds[i], i] = 1.
            if maxpowinds[i] - minpowinds[i] != np.sum(output[minpowinds[i]:maxpowinds[i], i]):
                print(i, "!!!!!!")
                print(minpowinds[i])
                print(maxpowinds[i])
    # return result, flipped to match orientation of Marc's images
    return output[::-1]

class NPZ_Dataset_OnTheFly(data.Dataset):
    # filenames = train_filenames,kic=train_kic,mag=train_mags,numax_sigma = train_numax_sigma, dim=(128,128), random_draws=True
    # On the fly freqs generation for training. This version is repurposed for MDN
    # This method randomizes the sigma for auxiliary variables, perturbes the variables, and returns sigma as an input
    def __init__(self, filenames, kic, dim, scale=False, validate=False):
    
        self.filenames = filenames
        self.file_kic = kic
        self.dim=dim # image/2D array dimensions
        self.scale = scale
        self.indexes = np.arange(len(self.filenames))
        self.validate = validate       

        if self.scale:
            print('Standard Scaling Log-log Power Spectrum!')
        if self.validate:
            print('Validate Mode. Batch includes Mag and H/B')     

    def __len__(self):
        'Total number of samples'
        return len(self.indexes)

    def __getitem__(self, index):
        'Generates ONE sample of data'

        batch_filenames = self.filenames[index]
        batch_kic = self.file_kic[index]

        # Generate data
        if not self.validate:
            X, y = self.__data_generation(batch_filenames)
            #plt.imshow(X.squeeze(), cmap='gray')
            #plt.savefig('/scratch/test_images/%s.png' %batch_kic)
            #plt.close()        
            return X.copy(), y, batch_filenames
        else:
            X, y, mag, hb, idx  = self.__data_generation(batch_filenames)
        
            return X.copy(), y, mag, hb, idx 

  
    def __data_generation(self, batch_filenames):
        data = np.load(batch_filenames)
        freq = data['freq']
        pow = data['pow']
        file_numax = data['numax']
        im = ps_to_array(freq, pow, minfreq=3., scale=self.scale, fix=True)
        im = np.expand_dims(im, 0)
        X = im
        y = file_numax


        if not self.validate:
            return X, y
        else:
            return X, y, data['mag'], data['hb'], np.array(data['id'])
  
  

ONEOVERSQRT2PI = 1.0 / math.sqrt(2*math.pi)

class MDN(nn.Module):
    """A mixture density network layer
    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.
    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions
    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.
    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """
    def __init__(self, in_features, out_features, num_gaussians, temperature=1):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.temperature = temperature
        self.pi = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256,num_gaussians),
        )
        self.sigma = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256,out_features*num_gaussians),
            nn.Softplus()
        )
        
        self.mu = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256,out_features*num_gaussians),
            nn.Softplus(threshold=5)  
        )
        #self.elu = nn.ELU()

    def forward(self, minibatch):
        pi = self.pi(minibatch)
        pi = F.softmax(pi/self.temperature, dim=1)
        #sigma = self.elu(self.sigma(minibatch)) + 1
        sigma  = self.sigma(minibatch)
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu


def gaussian_probability(sigma, mu, target):
    """Returns the probability of `data` given MoG parameters `sigma` and `mu`.
    
    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        data (BxI): A batch of data. B is the batch size and I is the number of
            input dimensions.
    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """
    if len(sigma.size()) == 2:
        sigma = sigma.unsqueeze(-1) # need tensors to be 3d
        mu = mu.unsqueeze(-1)
    if len(target.size()) == 1:
        target = target.unsqueeze(-1)
    data = target.unsqueeze(1).expand_as(sigma)
#     print('Sigma Nan: ', torch.sum(torch.isnan(sigma)))
#     print('Mu Nan: ', torch.sum(torch.isnan(mu)))
#     print('Target Nan: ', torch.sum(torch.isnan(data)))

    ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((data - mu) / (sigma+1e-6))**2) / (sigma+1e-6)
#     print('Ret: ', torch.sum(torch.isnan(ret)))
#     print('Final: ', torch.sum(torch.isnan(torch.prod(ret, 2))))
    return torch.prod(ret, 2)


def mdn_loss(pi, sigma, mu, target):
    """Calculates the error, given the MoG parameters and the target
    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    prob = pi * gaussian_probability(sigma, mu, target)
    nll = -torch.log(torch.sum(prob, dim=1)+1e-6)
#     print('-------')
#     print('NLL: ', torch.sum(torch.isnan(nll)))
#     print('Sum: ', torch.sum(prob, dim=1)+1e-6)
#     print('Log: ', torch.log(torch.sum(prob, dim=1)+1e-6))
    return torch.mean(nll)


def sample(pi, sigma, mu):
    """Draw samples from a MoG.
    """
    categorical = Categorical(pi)
    pis = list(categorical.sample().data)
    sample = Variable(sigma.data.new(sigma.size(0), sigma.size(2)).normal_())
    for i, idx in enumerate(pis):
        sample[i] = sample[i].mul(sigma[i,idx]).add(mu[i,idx])
    return sample

def dist_mu(pi, mu):
    """Calculate the mean of a mixture.
    """
    if pi.size() != mu.size():
        pi = pi.unsqueeze(2)
    return torch.sum(pi*mu, dim=1)

def dist_var_npy(pi, mu, mixture_mu, sigma):
    """Calculate the second moment (variance) of a bimodal distribution
    mu is the tensor while mixture_mu is the mean of the entire mixture as calculated by dist_mu
    """
    if pi.shape != mu.shape:
        pi = np.expand_dims(pi, 2)
    if mixture_mu.shape != mu.shape:
        mixture_mu = np.expand_dims(mixture_mu, -1)
    delta_square =(mu-mixture_mu)* (mu-mixture_mu)
    summation = sigma*sigma + delta_square
    return np.sum(pi*summation, 1)


def dist_mu_npy(pi, mu):
    """Calculate the mean of a mixture.
    """
    if pi.shape != mu.shape:
        pi = np.expand_dims(pi, 2)
    return np.sum(pi*mu, 1)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true-y_pred))


def initialization(model):
    for name, param in model.named_parameters():  # initializing model weights
        if 'bias' in name:
            nn.init.constant_(param, 0.00)
        elif 'weight' in name:
            nn.init.xavier_uniform_(param)


class SLOSH_Regressor(nn.Module):
    def __init__(self, num_gaussians):
        super(SLOSH_Regressor, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, padding=2) # same padding 2P = K-1
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1) # same padding 2P = K-1
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1) # same padding 2P = K-1
        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2)
        self.pool3 = nn.MaxPool2d(2,2)
        self.drop1 = nn.Dropout(0.1)
        self.linear1 = nn.Linear(16*16*16, 128)
        self.mdn = MDN(in_features=128, out_features=1, num_gaussians=num_gaussians)

    def print_instance_name(self):
        print (self.__class__.__name__)

    def forward(self, input_image):
        conv1 = F.leaky_relu(self.conv1(input_image), negative_slope=0.1) # (N, C, H, W)
        conv1 = self.pool1(conv1)
        conv2 = F.leaky_relu(self.conv2(conv1), negative_slope=0.1)
        conv2 = self.pool2(conv2)
        conv3 = F.leaky_relu(self.conv3(conv2), negative_slope=0.1)
        conv3 = self.pool3(conv3)
        conv3 = self.drop1(conv3)

        linear1 = F.relu(self.linear1(conv3.view(conv3.size()[0], -1)))
        pi, sigma, mu = self.mdn(linear1)
        return pi, sigma, mu


model = SLOSH_Regressor(num_gaussians = 4)
model.cuda()
torch.backends.cudnn.benchmark=True
print('CUDNN Enabled? ', torch.backends.cudnn.enabled)
print(str(model))
initialization(model)

learning_rate = 0.0001
model_optimizer = optim.Adam(model.parameters(), lr = learning_rate)
scheduler = ReduceLROnPlateau(model_optimizer, mode='min', factor=0.5, patience=2, verbose=True, min_lr = 1E-6)


#root_folder = '/data/marc/TESS/QLP/celerite_tess_sim_train/PSD/v1/'#npz_PSD_numax or npz_PSD_xd_ or test_numax
root_folder = '/scratch/celerite/v1/'
print('Parsing files in folder... ')



file_kic_vec = []  
filename_vec = []

count = 0
for dirpath, dirnames, filenames in os.walk(root_folder): # Getting the mags, KICS, and numax sigma for all stars in folder
    for q, file in tqdm(enumerate(filenames), total = len(filenames)):
        #if q > 100000:
        #    break
        if file.endswith('.npz'): 
            d = np.load(os.path.join(dirpath, file))
            if not ( (d['numax'] < 25) and (d['mag'] > 7) and (d['hb'] > 1000) ):
                count += 1
                continue

            file_kic = int(re.search(r'\d+', file).group())
            filename_vec.append(os.path.join(dirpath, file))
            file_kic_vec.append(file_kic)

print('Original File Number in Folder: ', len(os.listdir(root_folder)))
print('Final File Number: ', len(filename_vec))
print('Removed Samples: ', count)

file_kic = np.array(file_kic_vec)
filenames = np.array(filename_vec)

unique_id = np.unique(file_kic)
train_ids, test_ids = train_test_split(unique_id, test_size =0.15, random_state = 137)
train_ids, val_ids = train_test_split(train_ids, test_size =0.1765, random_state = 137)

train_kic = file_kic[np.in1d(file_kic, train_ids)]
val_kic = file_kic[np.in1d(file_kic, val_ids)] 
test_kic = file_kic[np.in1d(file_kic, test_ids)]

train_filenames = filenames[np.in1d(file_kic, train_ids)]
val_filenames = filenames[np.in1d(file_kic, val_ids)]
test_filenames = filenames[np.in1d(file_kic, test_ids)]

  
print('Total Files: ', len(file_kic))
print('Total Unique IDs: ', len(unique_id))
print('Total Train Files: ', len(train_kic))
print('Train Unique IDs: ', len(train_ids))
print('Total Val Files: ', len(val_kic))
print('Test Unique IDs: ', len(test_ids))
print('Total Test Files: ', len(test_kic))   
print('Number of Training IDs in Test IDs: ', np.sum(np.in1d(train_ids, test_ids)))
print('Number of Training IDs in Val IDs: ', np.sum(np.in1d(train_ids, val_ids))) 
print('Test IDs: ', test_ids)
print('Setting up generators... ')


def validate(model, val_dataloader):
    model.eval() # set to evaluate mode
    val_loss = 0
    val_cum_mape = 0
    val_cum_mae = 0   
    val_batches = 0

    for j, val_data in enumerate(val_dataloader, 0): #indices,scaled_indices, numax, teff, fe_h, age, tams_age
        #if j > 32:
        #    break
        val_image = val_data[0].cuda()
        val_numax = val_data[1].cuda()
    
        if len(val_image.size()) <= 2:
            print('Insufficient Batch!')
            continue
        with torch.no_grad():
            pi, sigma, mu = model(input_image=torch.squeeze(val_image, -1).float())
        
            val_batch_loss = mdn_loss(pi, sigma, mu, target=val_numax.float()) # log-likelihood loss
            val_pred_npy = dist_mu(pi, mu).data.cpu().numpy().reshape(-1,1)
            val_truth_npy = val_numax.data.cpu().numpy()

            val_mape = mean_absolute_percentage_error(y_true=val_truth_npy.squeeze(),y_pred=val_pred_npy.squeeze())
            val_mae = mean_absolute_error(y_true=val_truth_npy.squeeze(),y_pred=val_pred_npy.squeeze())
       

            val_loss += val_batch_loss.item()
            val_cum_mape += val_mape
            val_cum_mae += val_mae
            val_batches += 1

    print('Sample Numax: ', val_numax[-1])

    print('Sample Truth: ', val_truth_npy[-1])
    print('Sample Pred: ', val_pred_npy[-1])
    print('Sample Age Pi: ', pi[-1])
    print('Sample Sigma: ', sigma[-1])
    print('Sample Mu: ', mu[-1])


    return (val_loss/val_batches), (val_cum_mape/val_batches), (val_cum_mae/val_batches)


def train(model, model_optimizer, input_image, input_numax):

    model_optimizer.zero_grad()

    # Combined forward pass


    pi, sigma, mu = model(input_image = torch.squeeze(input_image, -1).float())

    # Calculate loss and backpropagate

    loss = mdn_loss(pi, sigma, mu, target=input_numax.float()) #log-likelihood optimization
    #loss = compute_full_crps(pi, mu, sigma, label=input_numax.float(), label_err=input_numax_sigma.float(), loss=True) # CRPS optim
    #loss = compute_closed_crps(pi, mu, sigma, label=input_numax.float())
    loss.backward()
    
    pred_mean = dist_mu(pi, mu).data.cpu().numpy().reshape(-1,1)
    truth_npy = input_numax.data.cpu().numpy()

    mape = mean_absolute_percentage_error(y_true=truth_npy.squeeze(), y_pred=pred_mean.squeeze())
    mae = mean_absolute_error(y_true=truth_npy.squeeze(),y_pred=pred_mean.squeeze())

    
    #Clipnorm?
    #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # Update parameters
    model_optimizer.step()

    return loss.item(), mape, mae


def train_model():

    train_gen = NPZ_Dataset_OnTheFly(filenames = train_filenames,kic=train_kic, dim=(128,128), scale=True)
    train_dataloader = utils.DataLoader(train_gen, shuffle=True, batch_size=32, num_workers=5)

    val_gen = NPZ_Dataset_OnTheFly(filenames = val_filenames,kic=val_kic, dim=(128,128), scale=True)
    val_dataloader = utils.DataLoader(val_gen, shuffle=True, batch_size=32, num_workers=5)
    print('Limited range!')

    n_epochs=1000
    model_checkpoint = True
    best_loss = 9999
    for epoch in range(1, n_epochs + 1):
        print('---------------------')
        print('Epoch: ', epoch)
        train_loss = 0
        train_batches = 0
        mape_cum = 0
        mae_cum = 0

        model.train()  # set to training mode
    
        for i, data in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader), unit='batches'):
            #if i > 32:
            #    break
            train_batches += 1
    
            image = data[0].cuda()
            numax = data[1].cuda()

            
            if len(image.size()) <= 2:
                print('Insufficient Batch!')
                continue
    
            loss, mape, mae = train(model, model_optimizer, image, numax)
            train_loss += loss  # Summing losses across all batches, so if you want the mean for EACH sample, divide by number of batches
            mape_cum += mape
            mae_cum += mae

    
        train_loss = train_loss / train_batches
        train_mape = mape_cum / train_batches
        train_mae = mae_cum / train_batches

        val_loss, val_mape, val_mae = validate(model, val_dataloader)
        scheduler.step(val_loss)  # reduce LR on loss plateau

        print('\n\nTrain Loss: ', train_loss)
        print('Train Mape: ', train_mape)
        print('Train Mae: ', train_mae)
    
        print('Val Loss: ', val_loss)
        print('Val Mape: ', val_mape)
        print('Val Mae: ', val_mae)

        model.print_instance_name()

        for param_group in model_optimizer.param_groups:
            print('Current Learning Rate: ', param_group['lr'])
        is_best = val_mape < best_loss
        best_loss = min(val_mape, best_loss)
        print('Current Best Metric: ', best_loss)

        if model_checkpoint:

            if is_best:
                filename = 'CELERITE_1-100_LIMITEDFIXEDSCALE-MAPE:%.2f-MAE:%.2f_NETWORK5' % (
                val_mape, val_mae)
                filepath = '/scratch/'
                torch.save(model.state_dict(), os.path.join(filepath, filename))
                print('Model saved to %s' %os.path.join(filepath, filename))
                
            else:
                print('No improvement over the best of %.4f' % best_loss)

 
def test_model():

    val_gen = NPZ_Dataset_OnTheFly(filenames = val_filenames,kic=val_kic, dim=(128,128), scale=True, validate=True)
    val_dataloader = utils.DataLoader(val_gen, shuffle=True, batch_size=32, num_workers=5)

    model = SLOSH_Regressor(num_gaussians = 4)
    model.cuda()
    model.eval()
    torch.backends.cudnn.benchmark=True
    print('CUDNN Enabled? ', torch.backends.cudnn.enabled)
    print(str(model))

    #saved_model = torch.load('/home/z3384751/K2Detection/ClassifyModels/2D_classifier/Pytorch_MDN_SLOSH/CELERITE_SCALED_Network1_Drop=0.1-MAPE:4.72-MAE:1.88')
    saved_model = torch.load('/home/z3384751/K2Detection/ClassifyModels/2D_classifier/Pytorch_MDN_SLOSH/STANDARDSCALED_SLOSCFilter_Drop=0.1_NoAugment-MAPE:17.05-MAE:6.86')
    model.load_state_dict(saved_model)
    
    val_pred_vec, val_var_vec, val_label_vec, val_mag_vec, val_hb_vec, val_id_vec = [], [], [], [], [], []
    for j, val_data in tqdm(enumerate(val_dataloader, 0), total = len(val_dataloader)): #indices,scaled_indices, numax, teff, fe_h, age, tams_age
      
        val_image = val_data[0].cuda()
        val_numax = val_data[1].data.cpu().numpy()
        val_mag = val_data[2].data.cpu().numpy()
        val_hb = val_data[3].data.cpu().numpy()
        val_ids = val_data[4].data.cpu().numpy()
    
        if len(val_image.size()) <= 2:
            print('Insufficient Batch!')
            continue
        with torch.no_grad():
            pi, sigma, mu = model(input_image=torch.squeeze(val_image, -1).float())
        
            val_pred_npy = dist_mu(pi, mu).data.cpu().numpy().reshape(-1,1)
            val_var = dist_var_npy(pi.data.cpu().numpy(), mu.data.cpu().numpy(), val_pred_npy, sigma.data.cpu().numpy())
            val_sigma = np.sqrt(val_var)
            val_pred_vec.append(deepcopy(val_pred_npy.squeeze()))
            val_var_vec.append(deepcopy(val_sigma.squeeze()))
            val_label_vec.append(deepcopy(val_numax.squeeze()))
            val_mag_vec.append(deepcopy(val_mag))
            val_hb_vec.append(deepcopy(val_hb))
            val_id_vec.append(deepcopy(val_ids))

    val_pred_vec = np.concatenate(val_pred_vec)
    val_var_vec = np.concatenate(val_var_vec)
    val_mag_vec = np.concatenate(val_mag_vec)
    val_label_vec = np.concatenate(val_label_vec)
    val_hb_vec = np.concatenate(val_hb_vec)
    val_id_vec = np.concatenate(val_id_vec)

    np.savez_compressed('SLOSC_SCALED_REGRESSOR_NETWORK1_VALDATA', id=val_id_vec, label=val_label_vec, pred=val_pred_vec, var=val_var_vec, mag=val_mag_vec, hb=val_hb_vec)


def test_model_jie():

    val_gen = NPZ_Dataset_OnTheFly(filenames = val_filenames,kic=val_kic, dim=(128,128), scale=True, validate=False) # turn validate false to not get all the additional variables
    val_dataloader = utils.DataLoader(val_gen, shuffle=True, batch_size=32, num_workers=5)

    model = SLOSH_Regressor(num_gaussians = 4)
    model.cuda()
    model.eval()
    torch.backends.cudnn.benchmark=True
    print('CUDNN Enabled? ', torch.backends.cudnn.enabled)
    print(str(model))

    saved_model = torch.load('/home/z3384751/K2Detection/ClassifyModels/2D_classifier/Pytorch_MDN_SLOSH/CELERITE_SCALED_Network1_Drop=0.1-MAPE:4.72-MAE:1.88')
    #saved_model = torch.load('/home/z3384751/K2Detection/ClassifyModels/2D_classifier/Pytorch_MDN_SLOSH/STANDARDSCALED_SLOSCFilter_Drop=0.1_NoAugment-MAPE:17.05-MAE:6.86')
    model.load_state_dict(saved_model)
    
    val_pred_vec, val_var_vec, val_label_vec, val_mag_vec, val_hb_vec, val_id_vec = [], [], [], [], [], []
    for j, val_data in tqdm(enumerate(val_dataloader, 0), total = len(val_dataloader)): #indices,scaled_indices, numax, teff, fe_h, age, tams_age
      
        val_image = val_data[0].cuda()
        val_numax = val_data[1].data.cpu().numpy()
        val_ids = [idx for idx in val_data[2]]#val_data[2].data.cpu().numpy()
    
        if len(val_image.size()) <= 2:
            print('Insufficient Batch!')
            continue
        with torch.no_grad():
            pi, sigma, mu = model(input_image=torch.squeeze(val_image, -1).float())
        
            val_pred_npy = dist_mu(pi, mu).data.cpu().numpy().reshape(-1,1)
            val_var = dist_var_npy(pi.data.cpu().numpy(), mu.data.cpu().numpy(), val_pred_npy, sigma.data.cpu().numpy())
            val_sigma = np.sqrt(val_var)
            val_pred_vec.append(deepcopy(val_pred_npy.squeeze()))
            val_var_vec.append(deepcopy(val_sigma.squeeze()))
            val_label_vec.append(deepcopy(val_numax.squeeze()))
            val_id_vec.append(deepcopy(val_ids))

    val_pred_vec = np.concatenate(val_pred_vec)
    val_var_vec = np.concatenate(val_var_vec)
    val_label_vec = np.concatenate(val_label_vec)
    val_id_vec = np.concatenate(val_id_vec)

    np.savez_compressed('JIEPRED_CELERITE_SCALED_REGRESSOR_NETWORK1_VALDATA', id=val_id_vec, label=val_label_vec, pred=val_pred_vec, var=val_var_vec)


train_model()
#test_model()
#test_model_jie()
