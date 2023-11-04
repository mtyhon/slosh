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
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic, norm
from copy import deepcopy
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from astropy.stats import LombScargle

import math
import warnings
warnings.filterwarnings("ignore")
#torch.multiprocessing.set_sharing_strategy('file_system') # fix for 0 items of ancdata runtime error?
#import resource
#rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
#resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

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
    def __init__(self, filenames, kic, mag, numax, labels, dim, add_noise=False,random_draws=False, scale=False, validate=False):
    
        self.filenames = filenames
        self.file_kic = kic
        self.mags = mag
        self.numax = numax
        self.dim=dim # image/2D array dimensions
        self.random_draws = random_draws
        self.subfolder_labels = labels # for binary classification
        self.add_noise = add_noise
        self.scale = scale
        self.tess_nfactor = 1.0
        self.cadence =1764.        
        elat = 30.0  # 0 brings forward, 90 backwards
        V = 23.345 - 1.148 * ((np.abs(elat) - 90) / 90.0) ** 2
        self.tess_cc1 = 2.56e-3 * np.power(10.0, -0.4 * (V - 22.8)) * 69 * 21.1 * 21.1 * self.cadence * 4  # pixel scale is 21.1 arcsec and effective collecting area is 69cm^2, 4 CCDs
        self.tess_cc2 = 2.56e-3 * np.power(10.0, -0.4 * (23.345 - 22.8)) * 69 * 21.1 * 21.1 * self.cadence * 4

        self.indexes = np.arange(len(self.filenames))     
        self.validate = validate
        assert len(self.indexes) == len(self.file_kic) == len(self.numax) == len(self.filenames) == len(self.mags)
        if self.scale:
            print('Standard Scaling Log-log Power Spectrum!')

    def __len__(self):
        'Total number of samples'
        return len(self.indexes)

    def __getitem__(self, index):
        'Generates ONE sample of data'

        batch_filenames = self.filenames[index]
        batch_kic = self.file_kic[index]
        batch_mag = self.mags[index]
        batch_numax = self.numax[index]
        batch_label = self.subfolder_labels[index]

        # Generate data
        X, flag = self.__data_generation(batch_filenames, batch_mag, batch_numax)
        if not self.validate:
            return X.copy(), batch_label, flag
        else:
            return X.copy(), batch_label, flag, batch_numax, batch_mag

  
    def __data_generation(self, batch_filenames, batch_mag, batch_numax):
        #with np.load(batch_filenames) as data:
        try:
            data = np.load(batch_filenames)
        except:
            data = np.load(batch_filenames, allow_pickle=True)       
        freq = data['freq']
        power = data['pow']
 
        if self.add_noise:
            noised_pow, image_flag = self.add_noise_level(power,batch_mag, batch_numax)
            if len(noised_pow.shape) > 1:
                noised_pow = noised_pow.squeeze(0)
            power = noised_pow
        else:
            image_flag = 1.

        im = ps_to_array(freq, power, minfreq=3., scale=self.scale, fix=True) 
        im = np.expand_dims(im, 0)
        return im, image_flag

    def tess_noise(self, vkic):

        # 0.75 to match Fig14 of Sullivan et al. 2015
        cc = 0.75 * 69 * 1.514e6 * self.cadence * np.power(10.0, -0.4 * (vkic))  # vkic is mag, the photon flux at T = 0 is 1.514e6 photons /s /cm^2
        # 47-137 e per sec per pixel, 76 at b=30, 4 pixels

        sig = (1e6 / cc) * np.sqrt(cc + self.tess_cc1 + self.tess_cc2) * self.tess_nfactor  # 60 is ppm hr^1/2, div to remove hr^1/2
        return sig, sig + 60.0 / np.sqrt(self.cadence / 3600.0)


    def kepler_noise(self, vkic):
        cc = 3.46 * np.power(10.0, 0.4 * (12.0 - vkic) + 8.0)
        cc2 = 0.0
        temp = vkic.clip(min=14.0) / 14.0
        sig = ((1e6 / cc) * np.sqrt(cc + 7e6 * np.power(temp, 4) + cc2))
        return sig

    def add_noise_level(self, batch_power, batch_mag, batch_numax): # single values

        assert np.sum(np.isnan(batch_mag)) == 0
        assert np.sum(np.isnan(batch_numax)) == 0

        batch_mag_tess = batch_mag - 5 # the apparent magnitude in TESS
        image_flag = 0
        if batch_numax == 0:
            batch_upper_limit_mag = 14.
        else:
            func1 = np.log2(batch_numax/3.5E3)/-0.5
            func2 = np.log2(batch_numax/7.5E5)/-1.1
            heaviside1 = func1*np.heaviside(batch_numax-40, 0.5*(func1+func2))
            heaviside2 = func2*np.heaviside(40-batch_numax, 0.5*(func1+func2))
            mag_boundary = heaviside1 + heaviside2
            batch_upper_limit_mag= min(mag_boundary-0.5, 14)

        if batch_mag_tess < batch_upper_limit_mag: # if there is still room to add more noise
            random_draw = np.random.uniform(low=0.0, high=1.0, size=batch_numax.shape)
            if (self.random_draws) and (random_draw < 0.5):
                image_flag = 2
                return batch_power,image_flag # 50% chance to not augment

            #random_draw = np.random.uniform(low=0.0, high=1.0, size=batch_numax.shape)
            #if (self.random_draws) and (batch_numax != 0) and (random_draw < 0.6) and (batch_upper_limit_mag > 11.): # push above faint (11th mag) with 60% chance
            #    simulated_magnitude = np.random.uniform(low=11., high=batch_upper_limit_mag, size=len(batch_mag_tess))
            #    image_flag = simulated_magnitude
            #else:
            simulated_magnitude = np.random.uniform(low=batch_mag_tess, high=batch_upper_limit_mag)

            #sigma_ppm_kepler = self.kepler_noise(batch_mag_tess+5) #calculate its actual noise
            sigma_ppm_tess, sigma_ppm_tess_sys = self.tess_noise(simulated_magnitude) #calculate the noise from simulations
            tess_floor_noise = 2e-6*self.cadence*np.power(sigma_ppm_tess,2) # this is in ppm^2/uHz
            data_floor_noise = np.median(batch_power[-500:])
            floor_difference = tess_floor_noise - data_floor_noise

            if floor_difference > 0:
                chi_square_noise = 0.5*np.random.chisquare(df=2, size=len(batch_power))*floor_difference
                #chi_square_noise = -1*np.log(np.random.uniform(0, 1, len(batch_power)))*mean_noise # Andersen, Duvall, Jefferies 1990
                return_power = batch_power + chi_square_noise
                #mean_noise = 2e-6*self.cadence*np.power(sigma_ppm_tess-sigma_ppm_kepler,2)
                image_flag = 1
            else:
                return_power = batch_power
        else:
            return_power = batch_power

        return return_power, image_flag



class NPZ_Dataset_OnTheFly_Magnitude_Variation(data.Dataset):
    def __init__(self, filenames, kic, mag, numax, labels, dim, fixed_magnitude=15, do_test=False):
    
        self.filenames = filenames
        self.file_kic = kic
        self.mags = mag
        self.numax = numax
        self.dim=dim # image/2D array dimensions
        self.subfolder_labels = labels # for binary classification
        self.tess_nfactor = 1.0
        self.cadence =1764.
        self.do_test = do_test  
        self.fixed_magnitude = fixed_magnitude      
        elat = 30.0  # 0 brings forward, 90 backwards
        V = 23.345 - 1.148 * ((np.abs(elat) - 90) / 90.0) ** 2
        self.tess_cc1 = 2.56e-3 * np.power(10.0, -0.4 * (V - 22.8)) * 69 * 21.1 * 21.1 * self.cadence * 4  # pixel scale is 21.1 arcsec and effective collecting area is 69cm^2, 4 CCDs
        self.tess_cc2 = 2.56e-3 * np.power(10.0, -0.4 * (23.345 - 22.8)) * 69 * 21.1 * 21.1 * self.cadence * 4

        self.indexes = np.arange(len(self.filenames))     

        assert len(self.indexes) == len(self.file_kic) == len(self.numax) == len(self.filenames) == len(self.mags)

    def __len__(self):
        'Total number of samples'
        return len(self.indexes)

    def __getitem__(self, index):
        'Generates ONE sample of data'

        batch_filenames = self.filenames[index]
        batch_kic = self.file_kic[index]
        batch_mag = self.mags[index]
        batch_numax = self.numax[index]
        batch_label = self.subfolder_labels[index]

        # Generate data
        X, flag = self.__data_generation(batch_filenames, batch_mag, batch_numax)

        return X.copy(), batch_label, batch_kic, flag

  
    def __data_generation(self, batch_filenames, batch_mag, batch_numax):
        #with np.load(batch_filenames) as data:
        data = np.load(batch_filenames)
        freq = data['freq']
        power = data['pow']

        noised_pow, image_flag = self.add_noise_level(power,batch_mag, batch_numax)
        if len(noised_pow.shape) > 1:
            noised_pow = noised_pow.squeeze(0)

        im = ps_to_array(freq, noised_pow, minfreq=3.)

        if self.do_test:
            if (image_flag < 5) :
                ori_im = ps_to_array(freq, power, minfreq=3.)
                fig = plt.figure(figsize=(12,6))
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                ax1.set_title('Mag: %.1f, Numax: %d' %(batch_mag, batch_numax))
                ax2.set_title('Flag: %d' %(image_flag))
                ax1.imshow(ori_im, cmap='gray')
                ax2.imshow(im, cmap='gray')
                plt.savefig('/data/marc/K2Detection/test_images/%.2f.png' %(batch_numax))
                plt.close()
                
      
        return im, image_flag

    def tess_noise(self, vkic):

        # 0.75 to match Fig14 of Sullivan et al. 2015
        cc = 0.75 * 69 * 1.514e6 * self.cadence * np.power(10.0, -0.4 * (vkic))  # vkic is mag, the photon flux at T = 0 is 1.514e6 photons /s /cm^2
        # 47-137 e per sec per pixel, 76 at b=30, 4 pixels

        sig = (1e6 / cc) * np.sqrt(cc + self.tess_cc1 + self.tess_cc2) * self.tess_nfactor  # 60 is ppm hr^1/2, div to remove hr^1/2
        return sig, sig + 60.0 / np.sqrt(self.cadence / 3600.0)


    def kepler_noise(self, vkic):
        cc = 3.46 * np.power(10.0, 0.4 * (12.0 - vkic) + 8.0)
        cc2 = 0.0
        temp = vkic.clip(min=14.0) / 14.0
        sig = ((1e6 / cc) * np.sqrt(cc + 7e6 * np.power(temp, 4) + cc2))
        return sig

    def add_noise_level(self, batch_power, batch_mag, batch_numax): # single values
        assert np.sum(np.isnan(batch_mag)) == 0
        assert np.sum(np.isnan(batch_numax)) == 0
        batch_mag_tess = batch_mag - 5 # the apparent magnitude in TESS
        image_flag = 0

        if batch_numax == 0:
            batch_upper_limit_mag = max(13,self.fixed_magnitude) # seems too artificial, maybe just dont augment non-detections? 
            #return batch_power, 1
        else:
            func1 = np.log2(batch_numax/3.5E3)/-0.5
            func2 = np.log2(batch_numax/7.5E5)/-1.1
            heaviside1 = func1*np.heaviside(batch_numax-40, 0.5*(func1+func2))
            heaviside2 = func2*np.heaviside(40-batch_numax, 0.5*(func1+func2))
            mag_boundary = heaviside1 + heaviside2
            batch_upper_limit_mag= mag_boundary

        if batch_mag_tess > self.fixed_magnitude: # in the event that the noise is already ABOVE the fixed mag
            return batch_power, 1 # 0 = drop, 1 = keep
        elif batch_mag_tess < batch_upper_limit_mag: # if there is still room to add more noise and is BELOW the upper limit magnitude
            if batch_upper_limit_mag < self.fixed_magnitude: # if can add noise, but can't reach the fixed magnitude
                return batch_power, 0
            else: # This must mean that the upper bound is greater or equal to the fixed magnitude
                sigma_ppm_tess, sigma_ppm_tess_sys = self.tess_noise(self.fixed_magnitude) #calculate the expected noise from fixed magnitude
                tess_floor_noise = 2e-6*self.cadence*np.power(sigma_ppm_tess,2) # this is in ppm^2/uHz
                data_floor_noise = np.median(batch_power[-500:])
                floor_difference = tess_floor_noise - data_floor_noise
                #print(floor_difference)
                if floor_difference >= 0:
                    chi_square_noise = 0.5*np.random.chisquare(df=2, size=len(batch_power))*floor_difference
                    #chi_square_noise = -1*np.log(np.random.uniform(0, 1, len(batch_power)))*mean_noise # Andersen, Duvall, Jefferies 1990
                    return_power = batch_power + chi_square_noise
                    return return_power, 1
                else: # If for some reason the simulated noise is smaller than original noise
                    return batch_power, 2
        else: # No room to add more noise
            return batch_power, 0


class SLOSH_Classifier(nn.Module):
    def __init__(self):
        super(SLOSH_Classifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, padding=2) # same padding 2P = K-1
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1) # same padding 2P = K-1
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1) # same padding 2P = K-1
        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2)
        self.pool3 = nn.MaxPool2d(2,2)
        self.drop1 = nn.Dropout(0.1)
        self.linear1 = nn.Linear(16*16*16, 128)
        self.linear2 = nn.Linear(128, 2)

    def print_instance_name(self):
        print (self.__class__.__name__)

    def forward(self, input_image):
        conv1 = F.leaky_relu(self.conv1(input_image), negative_slope=0.1) # (N, C, H, W) input_image.unsqueeze(1) if single channel 
        conv1 = self.pool1(conv1)
        conv2 = F.leaky_relu(self.conv2(conv1), negative_slope=0.1)
        conv2 = self.pool2(conv2)
        conv3 = F.leaky_relu(self.conv3(conv2), negative_slope=0.1)
        conv3 = self.pool3(conv3)
        conv3 = self.drop1(conv3)

        linear1 = F.relu(self.linear1(conv3.view(conv3.size()[0], -1)))
        linear2 = self.linear2(linear1)
        return linear2

def initialization(model):
    for name, param in model.named_parameters():  # initializing model weights
        if 'bias' in name:
            nn.init.constant_(param, 0.00)
        elif 'weight' in name:
            nn.init.xavier_uniform_(param)

def train(model, model_optimizer, input_image, input_label, loss_function):
    model_optimizer.zero_grad()

    # Combined forward pass
    outputs = model(input_image)

    # Calculate loss and backpropagate
    loss = loss_function(outputs, input_label)  # (input, target)
    loss.backward()

    # Update parameters
    model_optimizer.step()
    pred = torch.max(outputs, dim=1)[1]
    pos_pred = pred[pred == 1]
    neg_pred = pred[pred == 0]
    correct = torch.sum(pred.eq(input_label)).item()
    if torch.sum(input_label == 1) != 0:
        pos_recall = (torch.sum(pos_pred.eq(input_label[pred == 1])).float()/(torch.sum(input_label == 1)).float()).item()
    else:
        pos_recall = 0
    if torch.sum(input_label == 0) != 0:
        neg_recall = (torch.sum(neg_pred.eq(input_label[pred == 0])).float()/(torch.sum(input_label == 0)).float()).item()
    else:
        neg_recall = 0

    if len(pos_pred) != 0: 
        pos_precision = (torch.sum(pos_pred.eq(input_label[pred == 1])).float()/(float(pos_pred.size()[0]))).item()
    else:
        pos_precision = 0
    if len(neg_pred) != 0:
        neg_precision = (torch.sum(neg_pred.eq(input_label[pred == 0])).float()/(float(neg_pred.size()[0]))).item()
    else:
        neg_precision = 0
    total = input_label.numel()
    acc = 100. * correct / total
    return loss.item(), acc, pos_recall, pos_precision, neg_recall, neg_precision


def validate(model, val_dataloader, loss_function):
    model.eval()  # set to evaluate mode
    val_loss = 0
    val_batch_acc = 0
    val_pos_recall = []
    val_pos_precision = []
    val_neg_recall = []
    val_neg_precision = []
    val_flag_array = []
    val_batches = 0

    for batch_idy, val_data in enumerate(val_dataloader, 0):  # indices,scaled_indices, numax, teff, fe_h, age, tams_age

        val_image = val_data[0].cuda().float()
        input_label = val_data[1].cuda().long()
        val_flag = val_data[2].data.cpu().numpy()
        val_flag_array.append(deepcopy(val_flag))
        with torch.no_grad():
            outputs= model(val_image)
            val_batch_loss = loss_function(outputs, input_label)

        pred = torch.max(outputs, dim=1)[1]
        pos_pred = pred[pred == 1]
        neg_pred = pred[pred == 0]
        correct = torch.sum(pred.eq(input_label)).item()
        if torch.sum(input_label == 1) != 0:
            pos_recall = (torch.sum(pos_pred.eq(input_label[pred == 1])).float()/(torch.sum(input_label == 1)).float()).item()
        else:
            pos_recall = 0
        if torch.sum(input_label == 0) != 0:
            neg_recall = (torch.sum(neg_pred.eq(input_label[pred == 0])).float()/(torch.sum(input_label == 0)).float()).item()
        else:
            neg_recall = 0

        if len(pos_pred) != 0: 
            pos_precision = (torch.sum(pos_pred.eq(input_label[pred == 1])).float()/(float(pos_pred.size()[0]))).item()
        else:
            pos_precision = 0
        if len(neg_pred) != 0:
            neg_precision = (torch.sum(neg_pred.eq(input_label[pred == 0])).float()/(float(neg_pred.size()[0]))).item()
        else:
            neg_precision = 0
        total = input_label.numel()
        acc = 100. * correct / total

        val_loss += val_batch_loss.item()
        val_batch_acc += 100. * correct / total
        val_batches += 1
        if pos_recall != 0:
            val_pos_recall.append(deepcopy(pos_recall))
        if pos_precision != 0:
            val_pos_precision.append(deepcopy(pos_precision))
        if neg_recall != 0:
            val_neg_recall.append(deepcopy(neg_recall))
        if neg_precision != 0:
            val_neg_precision.append(deepcopy(neg_precision))    


    return (val_loss / val_batches), (val_batch_acc / val_batches), np.mean(np.array(val_pos_recall)), np.mean(np.array(val_pos_precision)), np.mean(np.array(val_neg_recall)), np.mean(np.array(val_neg_precision)), np.concatenate(val_flag_array)



def train_model_bell_array():
    model_checkpoint = True
    load_state=False

    model = SLOSH_Classifier()
    model.cuda()
    torch.backends.cudnn.benchmark=True
    print('CUDNN Enabled? ', torch.backends.cudnn.enabled)
    print(str(model))

    if load_state:
        check_path = ''
        checkpoint = torch.load(check_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        print('Loaded From Checkpoint!')
    else:
        initialization(model)

    loss_function = nn.CrossEntropyLoss(weight=torch.Tensor([1., 7.]).float().cuda())

    learning_rate = 0.001
    model_optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = ReduceLROnPlateau(model_optimizer, mode='min', factor=0.5, patience=2, verbose=True, min_lr = 1E-6)

    root_folder = '/data/marc/KEPSEISMIC/LC_Data/PSD/npz_PSD_all/'#npz_PSD_all/'#test_run_images/
    file_count = 0
    kepmag_data = pd.read_csv('/home/z3384751/K2Detection/DR25_KIC_RADec.dat', delim_whitespace=True, header=0)

    numax_data = pd.read_csv('/home/z3384751/K2Detection/Table_1_Extend_Probabilities_V2.dat', header=0, delimiter='|',comment='#')
    numax_values = numax_data['nu_max'].values
    numax_kic = numax_data['KIC'].values



    ####### STATISTICAL TEST #######
    slosc_data = np.load('slosc_result.npz', allow_pickle=True)
    slosc_filename = slosc_data['filename']
    slosc_kic = np.array([int(filex.split('-')[0]) for filex in slosc_filename])

    h1 = slosc_data['h1']
    slosc_mag = slosc_data['mag']
    slosc_numax = slosc_data['numax'].squeeze() 

    #print('H1 Shape: ', h1.shape)
    #print('slosc numax shape: ', slosc_numax.shape)
    #print('slosc filename shape: ', slosc_filename.shape)
    #print('slosc mag shape: ', slosc_mag.shape)

    slosc_false_positive = slosc_filename[(slosc_numax > 40) & (h1 < 0.99)]


    ####### MANUAL BOUNDARY EXCLUSION #######   THIS IS TOO HARSH

    below_boundary_kic = pd.read_csv('/data/marc/K2Detection/TESS-Kepler_Below_Boundary_Hon_Sample.dat').iloc[:].values


    id_vec = [] #KICs
    subfolder_labels = []
    folder_filenames = []
    mags = []
    file_kic = []
    count = 0
    boundary_exclude = 0
    original_filecount = 0
    catalogue_mag = kepmag_data['Kep_Mag'].values
    catalogue_kic = kepmag_data['KIC'].values

    unique_dirpath = []
    print('Parsing files in folder... ')

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for i, filex in tqdm(enumerate(filenames), total=len(filenames), unit='files'):
            original_filecount += 1
            
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for i, filex in tqdm(enumerate(filenames), total=len(filenames), unit='files'): # Getting the mags, KICS, and numax sigma for all stars in catalog
            if filex.endswith('.npz') & dirpath[-1].isdigit(): # I infer the class label '0' or '1' according to subfolder names
                if filex in slosc_false_positive:
                    count += 1
                    continue

                kicx = int(re.search(r'\d+', filex).group())

                #if (dirpath[-1] == '1'):
                #    if kicx not in below_boundary_kic:
                #        boundary_exclude += 1
                #        continue
                    


                folder_filenames.append(deepcopy(os.path.join(dirpath, filex)))
                subfolder_labels.append(deepcopy(int(dirpath[-1])))
                candidate_mag = catalogue_mag[np.where(catalogue_kic == kicx)[0]]
                try:
                    mags.append(deepcopy(candidate_mag[0]))
                except:                   
                    mags.append(-99)
                file_kic.append(deepcopy(kicx))

    print('Counted False Positives: ', count)
    print('Counted Boundary Excluded: ', boundary_exclude)
    print('Initial File Number: ', original_filecount)
    print('Filtered File Number: ', len(folder_filenames))

    mags = np.array(mags)
    file_kic = np.array(file_kic)
    subfolder_labels = np.array(subfolder_labels)
    numax = np.zeros(len(file_kic))

    for i, kicz in enumerate(file_kic):
        indez = np.where(numax_kic == kicz)[0]
        if len(indez) == 0:
            continue
        numax[i] = numax_values[indez]

    file_kic = file_kic[mags != -99] # these are for ENTIRE FOLDER
    subfolder_labels = subfolder_labels[mags != -99]
    numax = numax[mags != -99]
    filenames = np.array(folder_filenames)[mags != -99]
    mags = mags[mags != -99]

    print('Total NaNs in Mag (First Filter): ', np.sum(np.isnan(mags)))

    file_kic = file_kic[~np.isnan(mags)] # these are for ENTIRE FOLDER
    subfolder_labels = subfolder_labels[~np.isnan(mags)]
    numax = numax[~np.isnan(mags)]
    filenames = filenames[~np.isnan(mags)]
    mags = mags[~np.isnan(mags)]

    print('Total NaNs in Mag (Second Filter): ', np.sum(np.isnan(mags)))
    unique_posdet_kic = file_kic[subfolder_labels == 1]
    print('Number of Unique PosDet KIC: ', len(unique_posdet_kic))
    print('Number of Unique PosDet KIC in Detection Catalogue: ', np.sum(np.in1d(unique_posdet_kic, numax_kic)))
    print('Number of Unique PosDet KIC NOT in Detection Catalogue: ', np.sum(~np.in1d(unique_posdet_kic, numax_kic)))

    unique_id, unique_indices = np.unique(file_kic, return_index=True)

    print('Unique Labels Nb NonDet: ', np.sum(subfolder_labels[unique_indices] == 0))
    print('Unique Labels Nb Det: ', np.sum(subfolder_labels[unique_indices] == 1))

    train_ids, test_ids,train_unique_labels, test_unique_labels = train_test_split(unique_id, subfolder_labels[unique_indices], test_size =0.15, random_state = 137, stratify = subfolder_labels[unique_indices])
    train_ids, val_ids = train_test_split(train_ids, test_size =0.1765, random_state = 137, stratify=train_unique_labels)

    train_kic = file_kic[np.in1d(file_kic, train_ids)]
    val_kic = file_kic[np.in1d(file_kic, val_ids)]
    test_kic = file_kic[np.in1d(file_kic, test_ids)]

    train_numax = numax[np.in1d(file_kic, train_ids)]
    val_numax = numax[np.in1d(file_kic, val_ids)]
    test_numax = numax[np.in1d(file_kic, test_ids)]

    train_labels = subfolder_labels[np.in1d(file_kic, train_ids)]
    val_labels = subfolder_labels[np.in1d(file_kic, val_ids)]
    test_labels = subfolder_labels[np.in1d(file_kic, test_ids)]

    train_filenames = filenames[np.in1d(file_kic, train_ids)]
    val_filenames = filenames[np.in1d(file_kic, val_ids)]
    test_filenames = filenames[np.in1d(file_kic, test_ids)]

    train_mags = mags[np.in1d(file_kic, train_ids)]
    val_mags = mags[np.in1d(file_kic, val_ids)]
    test_mags = mags[np.in1d(file_kic, test_ids)]


    print('Total Files: ', len(file_kic))
    print('Total Unique IDs: ', len(unique_id))
    print('Total Train Files: ', len(train_kic))
    print('Train Unique IDs: ', len(train_ids))
    print('Total Val Files: ', len(val_kic))
    print('Test Unique IDs: ', len(test_ids))
    print('Total Test Files: ', len(test_kic))

    print('Number of Training IDs in Test IDs: ', np.sum(np.in1d(train_ids, test_ids)))
    print('Number of Training IDs in Val IDs: ', np.sum(np.in1d(train_ids, val_ids)))
    print('Nb of Neg to Pos in Train: %d/%d' %(np.sum(train_labels == 0), np.sum(train_labels == 1)))
    print('Nb of Neg to Pos in Val: %d/%d' %(np.sum(val_labels == 0), np.sum(val_labels == 1)))
    print('Nb of Neg to Pos in Test: %d/%d' %(np.sum(test_labels == 0), np.sum(test_labels == 1)))

    print('Setting up generators... ')
    print(train_filenames)
    train_gen = NPZ_Dataset_OnTheFly(filenames = train_filenames,kic=train_kic,mag=train_mags,numax = train_numax,labels=train_labels, dim=(128,128), add_noise=False, random_draws=False, scale=True)
    train_dataloader = utils.DataLoader(train_gen, shuffle=True, batch_size=32, num_workers=3)

    val_gen = NPZ_Dataset_OnTheFly(filenames = val_filenames,kic=val_kic,mag=val_mags,numax = val_numax,labels=val_labels, dim=(128,128), add_noise=False, random_draws=False,  scale=True)
    val_dataloader = utils.DataLoader(val_gen, shuffle=True, batch_size=32, num_workers=3)

    n_epochs=100
    best_loss = 9999
    for epoch in range(1, n_epochs + 1):
        print('---------------------')
        print('Epoch: ', epoch)
        train_loss = 0
        train_batches = 0
        train_acc = 0
        train_pos_precision = []
        train_pos_recall = []
        train_neg_precision = []
        train_neg_recall = []
        train_flag = []
    
        model.train()  # set to training mode
    
        for i, data in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader), unit='batches'):
            train_batches += 1
    
            image = data[0].cuda().float()
            label = data[1].cuda().long()
            flag = data[2].data.cpu().numpy()
            train_flag.append(deepcopy(flag))
            loss, acc, pos_recall, pos_precision, neg_recall, neg_precision = train(model, model_optimizer, image, label, loss_function)
            train_loss += loss  # Summing losses across all batches, so if you want the mean for EACH sample, divide by number of batches
            train_acc += acc
            if pos_recall != 0:
                train_pos_recall.append(deepcopy(pos_recall))
            if pos_precision != 0:
                train_pos_precision.append(deepcopy(pos_precision))
            if neg_recall != 0:
                train_neg_recall.append(deepcopy(neg_recall))
            if neg_precision != 0:
                train_neg_precision.append(deepcopy(neg_precision))

        train_loss = train_loss / train_batches
        train_acc = train_acc / train_batches
        train_pos_precision = np.mean(np.array(train_pos_precision))
        train_pos_recall = np.mean(np.array(train_pos_recall))
        train_neg_precision = np.mean(np.array(train_neg_precision))
        train_neg_recall = np.mean(np.array(train_neg_recall))
        train_flag = np.concatenate(train_flag, axis=0)

        val_loss, val_acc, val_pos_recall, val_pos_precision, val_neg_recall, val_neg_precision, val_flag = validate(model, val_dataloader, loss_function)
        scheduler.step(train_loss)  # reduce LR on loss plateau
    
        print('\n\nTrain Loss: ', train_loss)
        print('Train Acc: ', train_acc)
        print('Train Pos Precision: ', train_pos_precision)
        print('Train Pos Recall: ', train_pos_recall)
        print('Train Neg Precision: ', train_neg_precision)
        print('Train Neg Recall: ', train_neg_recall)
        print('Nb 0 Train Flags (Could Not Augment): ', np.sum(train_flag == 0))
        print('Nb 1 Train Flags (Augmented Draw): ', np.sum(train_flag == 1))
        print('Nb 2 Train Flags (Unaugmented Draw): ', np.sum(train_flag == 2))
    
        print('Val Loss: ', val_loss)
        print('Val Acc: ', val_acc)
        print('Val Pos Precision: ', val_pos_precision)
        print('Val Pos Recall: ', val_pos_recall)
        print('Val Neg Precision: ', val_neg_precision)
        print('Val Neg Recall: ', val_neg_recall)
        print('Nb 0 Val Flags (Could Not Augment): ', np.sum(val_flag == 0))
        print('Nb 1 Val Flags (Augmented Draw): ', np.sum(val_flag == 1))
        print('Nb 2 Val Flags (Unaugmented Draw): ', np.sum(val_flag == 2))
    

        for param_group in model_optimizer.param_groups:
            print('Current Learning Rate: ', param_group['lr'])
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        print('Current Best Metric: ', best_loss)

        if model_checkpoint:

            if is_best:
                filename = '27d_SCALED_NETWORK3_SLOSCFilter_Drop=0.1_NoAugment-Loss:%.2f-Acc:%.2f' % (
                val_loss, val_acc)
                filepath = '/data/marc/K2Detection/ClassifyModels/'
                torch.save(model.state_dict(), os.path.join(filepath, filename))
                print('Model saved to %s' %os.path.join(filepath, filename))

                torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': model_optimizer.state_dict(),'best_loss': best_loss}, os.path.join(filepath, filename+'.checkpoint'))
                print('Checkpoint saved at %s' %os.path.join(filepath, filename+'.checkpoint'))


            else:
                print('No improvement over the best of %.4f' % best_loss)

def test_model_bell_array():
    model = SLOSH_Classifier()
    model.cuda()

    #saved_model = torch.load('/data/marc/K2Detection/ClassifyModels/CELERITE_SCALED_Drop=0.1-Loss:0.02-Acc:99.21')
    saved_model = torch.load('/data/marc/K2Detection/ClassifyModels/27d_FIXEDSCALE_SLOSCFilter_Drop=0.1_NoAugment-Loss:0.09-Acc:98.20')
    model.load_state_dict(saved_model)

    torch.backends.cudnn.benchmark=True
    print('CUDNN Enabled? ', torch.backends.cudnn.enabled)
    print(str(model))
    model.eval()


    root_folder = '/data/marc/KEPSEISMIC/LC_Data/PSD/npz_PSD_all/'#npz_PSD_all/'#test_run_images/
    file_count = 0
    kepmag_data = pd.read_csv('/home/z3384751/K2Detection/DR25_KIC_RADec.dat', delim_whitespace=True, header=0)

    #numax_data = pd.read_csv('/home/z3384751/K2Detection/Table_1_Extend_Probabilities_V2.dat', header=0, delimiter='|',comment='#')
    #numax_values = numax_data['nu_max'].values
    #numax_kic = numax_data['KIC'].values


    numax_data = pd.read_csv('/home/z3384751/K2Detection/JieData_Full2018.txt', delimiter='|', header=0)
    numax_values = numax_data['numax'].values
    numax_kic = numax_data['KICID'].values

    ####### STATISTICAL TEST #######
    slosc_data = np.load('slosc_result.npz', allow_pickle=True)
    slosc_filename = slosc_data['filename']
    slosc_kic = np.array([int(filex.split('-')[0]) for filex in slosc_filename])

    h1 = slosc_data['h1']
    slosc_mag = slosc_data['mag']
    slosc_numax = slosc_data['numax'].squeeze() 
    slosc_false_positive = slosc_filename[(slosc_numax > 40) & (h1 < 0.99)]

    id_vec = [] #KICs
    subfolder_labels = []
    folder_filenames = []
    mags = []
    file_kic = []
    count = 0
    boundary_exclude = 0
    original_filecount = 0
    catalogue_mag = kepmag_data['Kep_Mag'].values
    catalogue_kic = kepmag_data['KIC'].values
 
    unique_dirpath = []
    print('Parsing files in folder... ')

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for i, filex in tqdm(enumerate(filenames), total=len(filenames), unit='files'):
            original_filecount += 1
            
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for i, filex in tqdm(enumerate(filenames), total=len(filenames), unit='files'): # Getting the mags, KICS, and numax sigma for all stars in catalog
            if filex.endswith('.npz') & dirpath[-1].isdigit(): # I infer the class label '0' or '1' according to subfolder names
                if filex in slosc_false_positive:
                    count += 1
                    continue

                kicx = int(re.search(r'\d+', filex).group())

                folder_filenames.append(deepcopy(os.path.join(dirpath, filex)))
                subfolder_labels.append(deepcopy(int(dirpath[-1])))
                candidate_mag = catalogue_mag[np.where(catalogue_kic == kicx)[0]]
                try:
                    mags.append(deepcopy(candidate_mag[0]))
                except:                   
                    mags.append(-99)
                file_kic.append(deepcopy(kicx))

    print('Counted False Positives: ', count)
    print('Counted Boundary Excluded: ', boundary_exclude)
    print('Initial File Number: ', original_filecount)
    print('Filtered File Number: ', len(folder_filenames))

    mags = np.array(mags)
    file_kic = np.array(file_kic)
    subfolder_labels = np.array(subfolder_labels)
    numax = np.zeros(len(file_kic))

    for i, kicz in tqdm(enumerate(file_kic), total=len(file_kic)):
        indez = np.where(numax_kic == kicz)[0]
        if len(indez) == 0:
            continue
        numax[i] = numax_values[indez]


    file_kic = file_kic[mags != -99] # these are for ENTIRE FOLDER
    subfolder_labels = subfolder_labels[mags != -99]
    numax = numax[mags != -99]
    filenames = np.array(folder_filenames)[mags != -99]
    mags = mags[mags != -99]

    print('Total NaNs in Mag (First Filter): ', np.sum(np.isnan(mags)))

    file_kic = file_kic[~np.isnan(mags)] # these are for ENTIRE FOLDER
    subfolder_labels = subfolder_labels[~np.isnan(mags)]
    numax = numax[~np.isnan(mags)]
    filenames = filenames[~np.isnan(mags)]
    mags = mags[~np.isnan(mags)]

    print('Total NaNs in Mag (Second Filter): ', np.sum(np.isnan(mags)))
    unique_posdet_kic = file_kic[subfolder_labels == 1]
    print('Number of Unique PosDet KIC: ', len(unique_posdet_kic))
    print('Number of Unique PosDet KIC in Detection Catalogue: ', np.sum(np.in1d(unique_posdet_kic, numax_kic)))
    print('Number of Unique PosDet KIC NOT in Detection Catalogue: ', np.sum(~np.in1d(unique_posdet_kic, numax_kic)))

    unique_id, unique_indices = np.unique(file_kic, return_index=True)

    print('Unique Labels Nb NonDet: ', np.sum(subfolder_labels[unique_indices] == 0))
    print('Unique Labels Nb Det: ', np.sum(subfolder_labels[unique_indices] == 1))

    train_ids, test_ids,train_unique_labels, test_unique_labels = train_test_split(unique_id, subfolder_labels[unique_indices], test_size =0.15, random_state = 137, stratify = subfolder_labels[unique_indices])
    train_ids, val_ids = train_test_split(train_ids, test_size =0.1765, random_state = 137, stratify=train_unique_labels)

    train_kic = file_kic[np.in1d(file_kic, train_ids)]
    val_kic = file_kic[np.in1d(file_kic, val_ids)]
    test_kic = file_kic[np.in1d(file_kic, test_ids)]

    train_numax = numax[np.in1d(file_kic, train_ids)]
    val_numax = numax[np.in1d(file_kic, val_ids)]
    test_numax = numax[np.in1d(file_kic, test_ids)]

    train_labels = subfolder_labels[np.in1d(file_kic, train_ids)]
    val_labels = subfolder_labels[np.in1d(file_kic, val_ids)]
    test_labels = subfolder_labels[np.in1d(file_kic, test_ids)]

    train_filenames = filenames[np.in1d(file_kic, train_ids)]
    val_filenames = filenames[np.in1d(file_kic, val_ids)]
    test_filenames = filenames[np.in1d(file_kic, test_ids)]

    train_mags = mags[np.in1d(file_kic, train_ids)]
    val_mags = mags[np.in1d(file_kic, val_ids)]
    test_mags = mags[np.in1d(file_kic, test_ids)]


    print('Total Files: ', len(file_kic))
    print('Total Unique IDs: ', len(unique_id))
    print('Total Train Files: ', len(train_kic))
    print('Train Unique IDs: ', len(train_ids))
    print('Total Val Files: ', len(val_kic))
    print('Test Unique IDs: ', len(test_ids))
    print('Total Test Files: ', len(test_kic))

    print('Number of Training IDs in Test IDs: ', np.sum(np.in1d(train_ids, test_ids)))
    print('Number of Training IDs in Val IDs: ', np.sum(np.in1d(train_ids, val_ids)))
    print('Nb of Neg to Pos in Train: %d/%d' %(np.sum(train_labels == 0), np.sum(train_labels == 1)))
    print('Nb of Neg to Pos in Val: %d/%d' %(np.sum(val_labels == 0), np.sum(val_labels == 1)))
    print('Nb of Neg to Pos in Test: %d/%d' %(np.sum(test_labels == 0), np.sum(test_labels == 1)))

    print('Setting up generators... ')
    

    test_gen = NPZ_Dataset_OnTheFly(filenames = test_filenames,kic=test_kic,mag=test_mags,numax = test_numax,labels=test_labels, dim=(128,128), add_noise=False, random_draws=False,  scale=True, validate=True)
    test_dataloader = utils.DataLoader(test_gen, shuffle=True, batch_size=32, num_workers=5)

    label_vec, pred_vec, prob_vec, numax_vec, mag_vec = [], [], [], [], []

    for batch_idy, val_data in tqdm(enumerate(test_dataloader, 0), total = len(test_dataloader)):
  
        val_image = val_data[0].cuda().float()
        input_label = val_data[1].data.cpu().numpy()
        val_flag = val_data[2].data.cpu().numpy()
        val_numax = val_data[3].data.cpu().numpy()
        val_mag = val_data[4].data.cpu().numpy()


        with torch.no_grad():
            outputs= model(val_image)
            class_pred = torch.max(outputs, dim=1)[1].data.cpu().numpy()
            class_prob = F.softmax(outputs, dim=1)[:,1].data.cpu().numpy()
        label_vec.append(deepcopy(input_label))
        prob_vec.append(deepcopy(class_prob))
        pred_vec.append(deepcopy(class_pred))
        numax_vec.append(deepcopy(val_numax))
        mag_vec.append(deepcopy(val_mag))

    label_vec = np.concatenate(label_vec)
    pred_vec = np.concatenate(pred_vec)
    prob_vec = np.concatenate(prob_vec)
    numax_vec = np.concatenate(numax_vec)
    mag_vec = np.concatenate(mag_vec)
    
        
    np.savez_compressed('REALLYFIXED_JIEPRED_SLOSC_SCALED_CLASSIFIER_NETWORK_VALDATA', label=label_vec, prob=prob_vec, pred=pred_vec, numax=numax_vec, mag=mag_vec)



#train_model_bell_array()
test_model_bell_array()

