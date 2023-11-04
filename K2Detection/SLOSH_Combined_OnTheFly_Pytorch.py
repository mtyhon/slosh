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

import math
import warnings
warnings.filterwarnings("ignore")
#torch.multiprocessing.set_sharing_strategy('file_system') # fix for 0 items of ancdata runtime error?
#import resource
#rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
#print('BUDDEH THIS IS MY RLIMIT: ', rlimit)
#resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
#print('BUDDEH THIS IS MY RLIMIT NOW: ', rlimit)
images_path = '/home/z3384751/K2Detection/LCDetection/Kepler/28d_classification_full_LC'
numax_images_path = '/home/z3384751/K2Detection/LCDetection/Kepler/28days_numax'



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
                minfreq=3., maxfreq=283., minpow=3., maxpow=3e7):
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
                                        minfreq=minfreq, maxfreq=maxfreq, minpow=minpow, maxpow=maxpow)[::-1]
        for i in range(supersample):
            for j in range(supersample):
                output += supersampled[i::supersample, j::supersample]
        output = output / (supersample ** 2.)
    else:  # don't supersample
        # Do everything in log space
   
        logfreq = np.log10(freq)
        logpower = np.log10(power)
        minlogfreq = np.log10(minfreq)
        maxlogfreq = np.log10(maxfreq)
        minlogpow = np.log10(minpow)
        maxlogpow = np.log10(maxpow)

        # Define bins


        xbinedges = np.linspace(np.log10(minfreq), np.log10(maxfreq), nbins + 1)
        xbinwidth = xbinedges[1] - xbinedges[0]
        ybinedges = np.linspace(np.log10(minpow), np.log10(maxpow), nbins + 1)
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
    def __init__(self, filenames, kic, mag, numax, labels, dim, add_noise,random_draws=False):
    
        self.filenames = filenames
        self.file_kic = kic
        self.mags = mag
        self.numax = numax
        self.dim=dim # image/2D array dimensions
        self.random_draws = random_draws
        self.subfolder_labels = labels # for binary classification
        self.add_noise = add_noise
        self.tess_nfactor = 1.0
        self.cadence =1764.        
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
        return X.copy(), batch_label, batch_numax, flag

  
    def __data_generation(self, batch_filenames, batch_mag, batch_numax):
        #with np.load(batch_filenames) as data:
        data = np.load(batch_filenames)
        freq = data['freq']
        power = data['pow']

        noised_pow, image_flag = self.add_noise_level(power,batch_mag, batch_numax)
        if len(noised_pow.shape) > 1:
            noised_pow = noised_pow.squeeze(0)
        if self.add_noise:
            im = ps_to_array(freq, noised_pow, minfreq=3.)
        else:
            im = ps_to_array(freq, power, minfreq=3.)
      
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
        try:
            target = target.unsqueeze(-1)
        except:
            print(target)
            target = torch.Tensor([target]).float().cuda()
    data = target.unsqueeze(1).expand_as(sigma)
#     print('Sigma Nan: ', torch.sum(torch.isnan(sigma)))
#     print('Mu Nan: ', torch.sum(torch.isnan(mu)))
#     print('Target Nan: ', torch.sum(torch.isnan(data)))

    ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((data - mu) / (sigma+1e-6))**2) / (sigma+1e-6)
#     print('Ret: ', torch.sum(torch.isnan(ret)))
#     print('Final: ', torch.sum(torch.isnan(torch.prod(ret, 2))))
    return torch.prod(ret, 2)


def mdn_loss(pi, sigma, mu, target, pred):
    """Calculates the error, given the MoG parameters and the target
    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    pi = pi[pred == 1]
    sigma = sigma[pred == 1]
    mu = mu[pred == 1]
    target = target[pred == 1]

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

class SLOSH_Hybrid(nn.Module):
    def __init__(self):
        super(SLOSH_Hybrid, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2) # same padding 2P = K-1
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1) # same padding 2P = K-1
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1) # same padding 2P = K-1
        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2)
        self.pool3 = nn.MaxPool2d(2,2)
        self.drop1 = nn.Dropout(0.1)
        self.linear1 = nn.Linear(16*16*16, 128)
        self.linear2 = nn.Linear(128, 2)
        self.mdn = MDN(in_features=128, out_features=1, num_gaussians=4)

    def print_instance_name(self):
        print (self.__class__.__name__)

    def forward(self, input_image):
        conv1 = F.leaky_relu(self.conv1(input_image.unsqueeze(1)), negative_slope=0.1) # (N, C, H, W)
        conv1 = self.pool1(conv1)
        conv2 = F.leaky_relu(self.conv2(conv1), negative_slope=0.1)
        conv2 = self.pool2(conv2)
        conv3 = F.leaky_relu(self.conv3(conv2), negative_slope=0.1)
        conv3 = self.pool3(conv3)
        conv3 = self.drop1(conv3)

        linear1 = F.relu(self.linear1(conv3.view(conv3.size()[0], -1)))
        pi, sigma, mu = self.mdn(linear1)
        logits = self.linear2(linear1)
        return logits, pi, sigma, mu

def initialization(model):
    for name, param in model.named_parameters():  # initializing model weights
        if 'bias' in name:
            nn.init.constant_(param, 0.00)
        elif 'weight' in name:
            nn.init.xavier_uniform_(param)

def train(model, model_optimizer, input_image, input_numax, input_label, classifier_loss_function):
    model_optimizer.zero_grad()

    # Combined forward pass
    logits, pi, sigma, mu = model(input_image)
    pred = torch.max(logits, dim=1)[1]

    # Calculate loss and backpropagate
    class_loss = classifier_loss_function(logits, input_label)  # (input, target)
    if torch.sum(pred == 1) == 0:
        regress_loss = 0
    else:
        regress_loss = mdn_loss(pi, sigma, mu, target=input_numax.float(), pred=pred) #log-likelihood optimization
    loss = class_loss + regress_loss
    loss.backward()

    if torch.sum(pred == 1) != 0:
        regress_loss = regress_loss.item()

    # Update parameters
    model_optimizer.step()
    
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

    pred_mean = dist_mu(pi, mu).data.cpu().numpy().reshape(-1,1)
    truth_npy = input_numax.data.cpu().numpy()

    if len(truth_npy[truth_npy != 0]) > 0:
        mape = mean_absolute_percentage_error(y_true=truth_npy[truth_npy != 0].squeeze(), y_pred=pred_mean[truth_npy != 0].squeeze())
    else:
        mape = 0
    mae = mean_absolute_error(y_true=truth_npy.squeeze(),y_pred=pred_mean.squeeze())

    return class_loss.item(), regress_loss, acc, pos_recall, pos_precision, neg_recall, neg_precision, mape, mae


def validate(model, val_dataloader, classifier_loss_function):
    model.eval()  # set to evaluate mode
    val_loss = 0
    val_regress_loss = []
    val_batch_acc = 0
    val_pos_recall = []
    val_pos_precision = []
    val_neg_recall = []
    val_neg_precision = []
    val_flag_array = []
    val_mape = []
    val_mae = []
    val_batches = 0

    for batch_idy, val_data in enumerate(val_dataloader, 0):  # indices,scaled_indices, numax, teff, fe_h, age, tams_age

        val_image = val_data[0].cuda().float()
        input_label = val_data[1].cuda().long()
        input_numax = val_data[2].cuda().float()
        val_flag = val_data[3].data.cpu().numpy()
        val_flag_array.append(deepcopy(val_flag))
        if len(input_numax) == 0:
            continue
        with torch.no_grad():
            logits, pi, sigma, mu= model(val_image)
            pred = torch.max(logits, dim=1)[1]
            val_batch_loss = classifier_loss_function(logits, input_label)
            if torch.sum(pred == 1) == 0:
                val_batch_regress_loss = 0
            else:
                val_batch_regress_loss = mdn_loss(pi, sigma, mu, target=input_numax.float(), pred=pred).item() #log-likelihood optimization

        
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
        val_regress_loss.append(val_batch_regress_loss)
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

        pred_mean = dist_mu(pi, mu).data.cpu().numpy().reshape(-1,1)
        truth_npy = input_numax.data.cpu().numpy()
        if len(truth_npy[truth_npy != 0]) > 0:
            mape = mean_absolute_percentage_error(y_true=truth_npy[truth_npy != 0].squeeze(), y_pred=pred_mean[truth_npy != 0].squeeze())
        else:
            mape = 0
        mae = mean_absolute_error(y_true=truth_npy.squeeze(),y_pred=pred_mean.squeeze())    
        val_mape.append(mape)
        val_mae.append(mae)

    return (val_loss / val_batches), np.mean(np.array(val_regress_loss)), (val_batch_acc / val_batches), np.mean(np.array(val_pos_recall)), np.mean(np.array(val_pos_precision)), np.mean(np.array(val_neg_recall)), np.mean(np.array(val_neg_precision)), np.mean(val_mape), np.mean(val_mae), np.concatenate(val_flag_array)



def train_model_bell_array():

    model = SLOSH_Hybrid()
    model.cuda()
    torch.backends.cudnn.benchmark=True
    print('CUDNN Enabled? ', torch.backends.cudnn.enabled)
    print(str(model))
    initialization(model)

    loss_function = nn.CrossEntropyLoss(weight=torch.Tensor([1., 10.]).float().cuda())

    learning_rate = 0.001
    model_optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = ReduceLROnPlateau(model_optimizer, mode='min', factor=0.5, patience=2, verbose=True, min_lr = 1E-6)

    root_folder = '/data/marc/KEPSEISMIC/LC_Data/PSD/npz_PSD_all/'#npz_PSD_all/'#test_run_images/
    file_count = 0
    kepmag_data = pd.read_csv('/home/z3384751/K2Detection/DR25_KIC_RADec.dat', delim_whitespace=True, header=0)
    numax_data = pd.read_csv('/home/z3384751/K2Detection/Table_1_Extend_Probabilities_V2.dat', header=0, delimiter='|',comment='#')
    id_vec = [] #KICs
    subfolder_labels = []
    folder_filenames = []
    mags = []
    file_kic = []
    catalogue_mag = kepmag_data['Kep_Mag'].values
    catalogue_kic = kepmag_data['KIC'].values
    numax_values = numax_data['nu_max'].values
    numax_kic = numax_data['KIC'].values
    numax_flag = numax_data['flag'].values
 
    print('Parsing files in folder... ')
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for i, filex in tqdm(enumerate(filenames), total=len(filenames), unit='files'): # Getting the mags, KICS, and numax sigma for all stars in catalog
            if filex.endswith('.npz') & dirpath[-1].isdigit(): # I infer the class label '0' or '1' according to subfolder names
                folder_filenames.append(deepcopy(os.path.join(dirpath, filex)))
                kicx = int(re.search(r'\d+', filex).group())
                subfolder_labels.append(deepcopy(int(dirpath[-1])))
                candidate_mag = catalogue_mag[np.where(catalogue_kic == kicx)[0]]
                try:
                    mags.append(deepcopy(candidate_mag[0]))
                except:                   
                    mags.append(-99)
                file_kic.append(deepcopy(kicx))

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
    train_gen = NPZ_Dataset_OnTheFly(filenames = train_filenames,kic=train_kic,mag=train_mags,numax = train_numax,labels=train_labels, dim=(128,128), add_noise=True, random_draws=True)
    train_dataloader = utils.DataLoader(train_gen, shuffle=True, batch_size=32, num_workers=2)

    val_gen = NPZ_Dataset_OnTheFly(filenames = val_filenames,kic=val_kic,mag=val_mags,numax = val_numax,labels=val_labels, dim=(128,128), add_noise=False, random_draws=True)
    val_dataloader = utils.DataLoader(val_gen, shuffle=True, batch_size=32, num_workers=1)

    n_epochs=100
    model_checkpoint = False
    best_loss = 9999
    for epoch in range(1, n_epochs + 1):
        print('---------------------')
        print('Epoch: ', epoch)
        train_loss = 0
        train_regress_loss = 0
        train_batches = 0
        train_acc = 0
        train_pos_precision = []
        train_pos_recall = []
        train_neg_precision = []
        train_neg_recall = []
        train_mape = []
        train_mae = []
        train_flag = []
    
        model.train()  # set to training mode
    
        for i, data in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader), unit='batches'):
            train_batches += 1
    
            image = data[0].cuda().float()
            label = data[1].cuda().long()
            numax = data[2].cuda().float()
            flag = data[3].data.cpu().numpy()
            train_flag.append(deepcopy(flag)) 
            if len(numax) == 0:
                continue
            loss, regress_loss, acc, pos_recall, pos_precision, neg_recall, neg_precision, mape, mae = train(model, model_optimizer, image, numax, label, loss_function)
            train_loss += loss  # Summing losses across all batches, so if you want the mean for EACH sample, divide by number of batches
            train_regress_loss.append(regress_loss)
            train_acc += acc
            if pos_recall != 0:
                train_pos_recall.append(deepcopy(pos_recall))
            if pos_precision != 0:
                train_pos_precision.append(deepcopy(pos_precision))
            if neg_recall != 0:
                train_neg_recall.append(deepcopy(neg_recall))
            if neg_precision != 0:
                train_neg_precision.append(deepcopy(neg_precision))
            train_mape.append(mape)
            train_mae.append(mae)

        train_loss = train_loss / train_batches
        train_regress_loss = np.mean(np.array(train_regress_loss))
        train_acc = train_acc / train_batches
        train_pos_precision = np.mean(np.array(train_pos_precision))
        train_pos_recall = np.mean(np.array(train_pos_recall))
        train_neg_precision = np.mean(np.array(train_neg_precision))
        train_neg_recall = np.mean(np.array(train_neg_recall))
        train_flag = np.concatenate(train_flag, axis=0)
        train_mape = np.mean(np.array(train_mape))
        train_mae = np.mean(np.array(train_mae))
        val_loss, val_regress_loss,  val_acc, val_pos_recall, val_pos_precision, val_neg_recall, val_neg_precision, val_mape, val_mae, val_flag = validate(model, val_dataloader, loss_function)
        scheduler.step(train_loss)  # reduce LR on loss plateau

        print('---TRAINING---\n')    
        print('Train Loss: ', train_loss)
        print('Train Regress Loss: ', train_regress_loss)
        print('Train Acc: ', train_acc)
        print('Train Pos Precision: ', train_pos_precision)
        print('Train Pos Recall: ', train_pos_recall)
        print('Train Neg Precision: ', train_neg_precision)
        print('Train Neg Recall: ', train_neg_recall)
        print('Train Mape (Osc Only): ', train_mape)
        print('Train Mae: ', train_mae)
        #print('Nb 0 Train Flags (Could Not Augment): ', np.sum(train_flag == 0))
        #print('Nb 1 Train Flags (Augmented Draw): ', np.sum(train_flag == 1))
        #print('Nb 2 Train Flags (Unaugmented Draw): ', np.sum(train_flag == 2))
        print('\n')
        print('---VALIDATION---\n')        
        print('Val Loss: ', val_loss)
        print('Val Regress_Loss: ', val_regress_loss)
        print('Val Acc: ', val_acc)
        print('Val Pos Precision: ', val_pos_precision)
        print('Val Pos Recall: ', val_pos_recall)
        print('Val Neg Precision: ', val_neg_precision)
        print('Val Neg Recall: ', val_neg_recall)
        print('Val Mape (Osc Only): ', val_mape)
        print('Val Mae: ', val_mae)
        #print('Nb 0 Val Flags (Could Not Augment): ', np.sum(val_flag == 0))
        #print('Nb 1 Val Flags (Augmented Draw): ', np.sum(val_flag == 1))
        #print('Nb 2 Val Flags (Unaugmented Draw): ', np.sum(val_flag == 2))
    

        for param_group in model_optimizer.param_groups:
            print('Current Learning Rate: ', param_group['lr'])
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        print('Current Best Metric: ', best_loss)

        if model_checkpoint:

            if is_best:
                filename = 'OTF_FIXED_Augto14_50percent-Loss:%.2f-Acc:%.2f' % (
                val_loss, val_acc)
                filepath = '/data/marc/K2Detection/ClassifyModels/'
                torch.save(model.state_dict(), os.path.join(filepath, filename))
                print('Model saved to %s' %os.path.join(filepath, filename))

            else:
                print('No improvement over the best of %.4f' % best_loss)


def test_model_magnitude_variation():
    model = SLOSH_Classifier()
    saved_model_dict = '/data/marc/K2Detection/ClassifyModels/OTF_FIXED_Augto14_50percent-Loss:0.10-Acc:98.10'
    model.load_state_dict(torch.load(saved_model_dict))
    model.cuda()
    model.eval()

    torch.backends.cudnn.benchmark=True
    print('CUDNN Enabled? ', torch.backends.cudnn.enabled)
    print(str(model))

    root_folder = '/data/marc/KEPSEISMIC/LC_Data/PSD/npz_PSD_all/'#npz_PSD_all/'#test_run_images/
    file_count = 0
    kepmag_data = pd.read_csv('/home/z3384751/K2Detection/DR25_KIC_RADec.dat', delim_whitespace=True, header=0)
    numax_data = pd.read_csv('/home/z3384751/K2Detection/Table_1_Extend_Probabilities_V2.dat', header=0, delimiter='|',comment='#')
    id_vec = [] #KICs
    subfolder_labels = []
    folder_filenames = []
    mags = []
    file_kic = []
    catalogue_mag = kepmag_data['Kep_Mag'].values
    catalogue_kic = kepmag_data['KIC'].values
    numax_values = numax_data['nu_max'].values
    numax_kic = numax_data['KIC'].values
    numax_flag = numax_data['flag'].values
 
    print('Parsing files in folder... ')
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for i, filex in tqdm(enumerate(filenames), total=len(filenames), unit='files'): # Getting the mags, KICS, and numax sigma for all stars in catalog
            if filex.endswith('.npz') & dirpath[-1].isdigit(): # I infer the class label '0' or '1' according to subfolder names
                folder_filenames.append(deepcopy(os.path.join(dirpath, filex)))
                kicx = int(re.search(r'\d+', filex).group())
                subfolder_labels.append(deepcopy(int(dirpath[-1])))
                candidate_mag = catalogue_mag[np.where(catalogue_kic == kicx)[0]]
                try:
                    mags.append(deepcopy(candidate_mag[0]))
                except:                   
                    mags.append(-99)
                file_kic.append(deepcopy(kicx))

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

    #fixed_magnitudes = np.linspace(9, 14, 26)#[13.2, 13.4, 13.6, 13.8, 14]
    fixed_magnitudes = np.linspace(14.2, 15, 5)
    for magz in fixed_magnitudes:
        print('Predicting on Mag: %.1f' %magz)
        test_gen = NPZ_Dataset_OnTheFly_Magnitude_Variation(filenames = test_filenames,kic=test_kic,mag=test_mags,numax = test_numax,labels=test_labels, dim=(128,128), fixed_magnitude = magz, do_test=False)
        test_dataloader = utils.DataLoader(test_gen, shuffle=False, batch_size=32, num_workers=2)

        flag_array = []
        pred_array = []
        kic_array = []
        label_array = []
        saveflag_array = []
        model.train()  # set to training mode
    
        for i, data in tqdm(enumerate(test_dataloader, 1), total=len(test_dataloader), unit='batches'):
            image = data[0].cuda().float()
            label = data[1].data.cpu().numpy()
            kic = data[2].data.cpu().numpy()
            flag = data[3].cuda()
            image = image[flag == 1]
            flag = flag.data.cpu().numpy()
            flag_array.append(deepcopy(flag))

            kic = kic[flag == 1]
            label = label[flag == 1]
            saveflag = flag[flag == 1]

            if image.size()[0] <= 1:
                print('Insufficient Batch!')
                continue

            with torch.no_grad():
                outputs= model(image)

                pred = torch.max(outputs, dim=1)[1]

                pred_array.append(deepcopy(pred.data.cpu().numpy()))
                kic_array.append(deepcopy(kic))

                label_array.append(deepcopy(label))
                saveflag_array.append(deepcopy(saveflag))

        pred_array = np.concatenate(pred_array, axis=0)
        kic_array = np.concatenate(kic_array, axis=0)
        flag_array = np.concatenate(flag_array, axis=0)
        label_array = np.concatenate(label_array, axis = 0)
        saveflag_array = np.concatenate(saveflag_array, axis=0)

        print('Flag 0 : ', np.sum(flag_array == 0))
        print('Flag 1 : ', np.sum(flag_array == 1))
        print('Flag 2 : ', np.sum(flag_array == 2))

        assert len(kic_array) == len(pred_array)

        np.savez_compressed('/data/marc/Classifier_Pytorch_Prediction-Indices(FIXED_Augto14)-Mag%.1f-NoAugNonDet' %magz, pred = pred_array, label = label_array, kic = kic_array, flag=saveflag_array)

train_model_bell_array()
#test_model_magnitude_variation()

