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
from Ranger_Optimizer import Ranger

import math
import warnings
warnings.filterwarnings("ignore")

print('Sci-kit Version {}.'.format(sklearn.__version__))


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
    def __init__(self, filenames, kic, mag, numax_sigma, dim, random_draws=False, add_noise=False, reverse=False, scale=False):
    
        self.filenames = filenames
        self.file_kic = kic
        self.mags = mag
        self.numax_sigma = numax_sigma
        self.dim=dim # image/2D array dimensions
        self.random_draws = random_draws
        self.add_noise = add_noise
        self.reverse= reverse
        self.scale = scale
        self.tess_nfactor = 1.0
        self.cadence =1764.        
        elat = 30.0  # 0 brings forward, 90 backwards
        V = 23.345 - 1.148 * ((np.abs(elat) - 90) / 90.0) ** 2
        self.tess_cc1 = 2.56e-3 * np.power(10.0, -0.4 * (V - 22.8)) * 69 * 21.1 * 21.1 * self.cadence * 4  # pixel scale is 21.1 arcsec and effective collecting area is 69cm^2, 4 CCDs
        self.tess_cc2 = 2.56e-3 * np.power(10.0, -0.4 * (23.345 - 22.8)) * 69 * 21.1 * 21.1 * self.cadence * 4

        self.indexes = np.arange(len(self.filenames))

        print('Adding Noise? : ', self.add_noise)
        if self.scale:
            print('Standard Scaling Log-log Power Spectrum!')     

        assert len(self.indexes) == len(self.file_kic) == len(self.numax_sigma) == len(self.filenames) == len(self.mags)

    def __len__(self):
        'Total number of samples'
        return len(self.indexes)

    def __getitem__(self, index):
        'Generates ONE sample of data'

        batch_filenames = self.filenames[index]
        batch_kic = self.file_kic[index]
        batch_mag = self.mags[index]

        # Generate data
        X, y, flag = self.__data_generation(batch_filenames, batch_mag, batch_kic)
        
        batch_numax_sigma = self.numax_sigma[index]
        return X.copy(), y, batch_numax_sigma.squeeze(), flag

  
    def __data_generation(self, batch_filenames, batch_mag, batch_kic):
        data = np.load(batch_filenames)
        freq = data['freq']
        pow = data['pow']
        file_numax = data['numax'][0]
        if self.add_noise:
            noised_pow, image_flag = self.add_noise_level(pow,np.array(batch_mag), np.array(file_numax))
            if len(noised_pow.shape) > 1:
                noised_pow = noised_pow.squeeze(0)
            im = ps_to_array(freq, noised_pow, minfreq=3., scale=self.scale)

             #if image_flag > 0:
             #    plt.imshow(im, cmap='gray')   
             #    plt.savefig('/data/marc/KEPSEISMIC/LC_Data/PSD/test_images/' + str(batch_kic) + '-%.1f_noise.png' %(image_flag))     
             #    plt.close()  

        else:
            im = ps_to_array(freq, pow, minfreq=3., scale=self.scale)
            im = np.expand_dims(im, 0)
            image_flag = 1.

        if self.reverse:
            im_reverse = ps_to_array(283-freq, pow, minfreq=3., maxfreq=283, minpow=3, maxpow=3e4)
            im = np.concatenate((im, np.expand_dims(im_reverse, 0)))
  
        X = im
        y = file_numax

        return X, y, image_flag

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
            batch_upper_limit_mag = 13.
        else:
            func1 = np.log2(batch_numax/3.5E3)/-0.5
            func2 = np.log2(batch_numax/7.5E5)/-1.1
            heaviside1 = func1*np.heaviside(batch_numax-40, 0.5*(func1+func2))
            heaviside2 = func2*np.heaviside(40-batch_numax, 0.5*(func1+func2))
            mag_boundary = heaviside1 + heaviside2
            batch_upper_limit_mag= min(mag_boundary-0.5, 13)

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
                return_power = batch_power + chi_square_noise
                image_flag = 1
            else:
                return_power = batch_power
        else:
            return_power = batch_power

        return return_power, image_flag
  


class NPZ_Dataset_Magnitude_Variation(data.Dataset):
    # On the fly freqs generation for training. This version is repurposed for MDN
    # This method randomizes the sigma for auxiliary variables, perturbes the variables, and returns sigma as an input
    def __init__(self, root, dim, catalogue_kic, catalogue_mag, extension = '.npz', indices=[], label_uncertainty=False, obs_len=82., fixed_magnitude=15):
        self.root = root # root folder containing subfolders
        self.extension = extension # file extension
        self.filenames = []
        self.file_kic = []
        self.mags = []
        self.numax_sigma = []
        self.dim=dim # image/2D array dimensions
        self.label_uncertainty = label_uncertainty
        self.fixed_magnitude = fixed_magnitude # fix magnitude to this, if it can't reach here, it is dropped
        self.uncertainty_factor = np.sqrt(365.*4./obs_len)
        print('Label Uncertainty Included? ', self.label_uncertainty)
        print('Uncertainty Factor: ', self.uncertainty_factor)
        self.tess_nfactor = 1.0
        self.cadence =1764        
        elat = 30.0  # 0 brings forward, 90 backwards
        V = 23.345 - 1.148 * ((np.abs(elat) - 90) / 90.0) ** 2
        self.tess_cc1 = 2.56e-3 * np.power(10.0, -0.4 * (V - 22.8)) * 69 * 21.1 * 21.1 * self.cadence * 4  # pixel scale is 21.1 arcsec and effective collecting area is 69cm^2, 4 CCDs
        self.tess_cc2 = 2.56e-3 * np.power(10.0, -0.4 * (23.345 - 22.8)) * 69 * 21.1 * 21.1 * self.cadence * 4

        if self.label_uncertainty:
            numax_data = pd.read_csv('/home/z3384751/K2Detection/JieData_Full2018.txt', delimiter='|', header=0)
            cat_numax = numax_data['numax'].values
            cat_numax_sig = numax_data['numax_err'].values
            cat_kic = numax_data['KICID'].values

        for dirpath, dirnames, filenames in os.walk(root):
            for file in filenames:
                if file.endswith(extension) & dirpath[-1].isdigit(): # I infer the class label '0' or '1' according to subfolder names
                    self.filenames.append(os.path.join(dirpath, file))
                    file_kic = int(re.search(r'\d+', file).group())
                    self.mags.append(catalogue_mag[np.where(catalogue_kic == file_kic)[0]])
                    self.file_kic.append(file_kic)
                    if self.label_uncertainty:
                        self.numax_sigma.append(cat_numax_sig[np.where(cat_kic == file_kic)[0]])

        if len(indices) == 0: # otherwise pass a list of training/validation indices
            self.indexes = np.arange(len(self.filenames))
        else:
            self.indexes = np.array(indices)
        self.mags = np.array(self.mags)
        self.file_kic = np.array(self.file_kic)
        if self.label_uncertainty:
            self.numax_sigma = np.array(self.numax_sigma)*self.uncertainty_factor

    def __len__(self):
        'Total number of samples'
        return len(self.indexes)

    def __getitem__(self, index):
        'Generates ONE sample of data'
        batch_filenames = self.filenames[index]
        batch_mag = self.mags[index]
        batch_indexes = self.indexes[index]
        # Generate data
        X, y,keep_flag = self.__data_generation(batch_filenames, batch_mag)
        
        if self.label_uncertainty:
            batch_numax_sigma = self.numax_sigma[index]
            return X.copy(), y, batch_numax_sigma.squeeze(), keep_flag, batch_indexes
        else:
            return X.copy(), y, keep_flag, batch_indexes
  
    def __data_generation(self, batch_filenames, batch_mag):
        data = np.load(batch_filenames)
        freq = data['freq']
        pow = data['pow']
        file_numax = data['numax'][0]
        noised_pow, keep_flag = self.add_noise_level(pow,np.array(batch_mag), np.array(file_numax))
        if len(noised_pow.shape) > 1:
            noised_pow = noised_pow.squeeze(0)
        im = ps_to_array(freq, noised_pow, minfreq=3.)         
        X = im
        y = file_numax

        return np.expand_dims(X,-1), y, keep_flag

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
        batch_mag_tess = batch_mag - 5 # the apparent magnitude in TESS
        batch_upper_limit_mag = self.fixed_magnitude

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
                #sigma_ppm_kepler = self.kepler_noise(batch_mag_tess+5) #calculate its actual noise
                sigma_ppm_tess, sigma_ppm_tess_sys = self.tess_noise(self.fixed_magnitude) #calculate the expected noise from fixed magnitude
                tess_floor_noise = 2e-6*self.cadence*np.power(sigma_ppm_tess,2) # this is in ppm^2/uHz
                data_floor_noise = np.median(batch_power[-500:])
                floor_difference = tess_floor_noise - data_floor_noise
                if floor_difference >= 0:
                    #mean_noise = 2e-6*self.cadence*np.power(sigma_ppm_tess-sigma_ppm_kepler,2)
                    #chi_square_noise = np.random.chisquare(df=2, size=batch_power.shape)*mean_noise
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



def compute_full_crps(pi, mu, sigma, label, label_err, loss=True):

    # Create initial numax array
    vals = torch.linspace(0, 300, 10000).unsqueeze(1).unsqueeze(2).cuda() # [10000, 1, 1]

    # Label cdfs
    y = torch.distributions.normal.Normal(label.cuda(), label_err.cuda())

    cdf_y = y.cdf(vals).squeeze()

    # Create grid
    vals = vals.repeat(1, pi.size()[0], 1) # [10000, 32, 1]

    # Create mixture distribution cdfs
    x = torch.distributions.normal.Normal(mu.squeeze(), sigma.squeeze())
    cdf_x = x.cdf(vals)
    cdf_x = torch.sum(cdf_x*pi, 2)

    if loss:
        loss = torch.sum((cdf_x - cdf_y)**2, 0)*(vals[1]-vals[0]).squeeze()
        return torch.mean(loss)
    else:
        return torch.sum((cdf_x - cdf_y)**2, 0)*(vals[1]-vals[0]).squeeze()


def compute_pit(pi, mu, sigma, label):
    # Calculates the CDF of the predicted PDF evaluated at the observation
    x = torch.distributions.normal.Normal(mu.squeeze(), sigma.squeeze())

    pit = x.cdf(label.float().unsqueeze(1))
    pit = torch.sum(pit*pi, 1)
    return pit.squeeze()
 


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

class SLOSH_Regressor_No_Drop(nn.Module):
    def __init__(self, num_gaussians):
        super(SLOSH_Regressor_No_Drop, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, padding=2) # same padding 2P = K-1
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1) # same padding 2P = K-1
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1) # same padding 2P = K-1
        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2)
        self.pool3 = nn.MaxPool2d(2,2)
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
#model_optimizer = Ranger(model.parameters(), lr = learning_rate)
scheduler = ReduceLROnPlateau(model_optimizer, mode='min', factor=0.5, patience=2, verbose=True, min_lr = 1E-6)


root_im_folder = '/home/z3384751/K2Detection/LCDetection/Kepler/Bell_Arrays_27d_numax/'
root_folder = '/data/marc/KEPSEISMIC/LC_Data/PSD/npz_PSD_numax/'#npz_PSD_numax or npz_PSD_xd_ or test_numax
kepmag_data = pd.read_csv('/home/z3384751/K2Detection/DR25_KIC_RADec.dat', delim_whitespace=True, header=0)
catalogue_kic = kepmag_data['KIC'].values
catalogue_mag = kepmag_data['Kep_Mag'].values

print('Parsing files in folder... ')


numax_data = pd.read_csv('/home/z3384751/K2Detection/JieData_Full2018.txt', delimiter='|', header=0)
cat_numax = numax_data['numax'].values
cat_numax_sig = numax_data['numax_err'].values
cat_kic = numax_data['KICID'].values


####### STATISTICAL TEST #######
slosc_data = np.load('slosc_result.npz', allow_pickle=True)
slosc_filename = slosc_data['filename']
slosc_kic = np.array([int(filex.split('-')[0]) for filex in slosc_filename])

h1 = slosc_data['h1']
slosc_mag = slosc_data['mag']
slosc_numax = slosc_data['numax'].squeeze()

slosc_false_positive = slosc_filename[(slosc_numax > 40) & (h1 < 0.99)]


####### MANUAL BOUNDARY EXCLUSION ####### THIS IS TOO HARSH

below_boundary_kic = pd.read_csv('/data/marc/K2Detection/TESS-Kepler_Below_Boundary_Jie_Sample.dat').iloc[:].values



mags = []
file_kic_vec = []  
filename_vec = []
numax_sigma = []       
count = 0
boundary_exclude = 0
original_filecount = 0

for dirpath, dirnames, filenames in os.walk(root_folder):
    for i, filex in tqdm(enumerate(filenames), total=len(filenames), unit='files'):
        original_filecount += 1

for dirpath, dirnames, filenames in os.walk(root_folder): # Getting the mags, KICS, and numax sigma for all stars in folder
    for file in filenames:
        if file.endswith('.npz') & dirpath[-1].isdigit(): 
            if file in slosc_false_positive:
                count += 1
                continue

            file_kic = int(re.search(r'\d+', file).group())
            #if file_kic not in below_boundary_kic:
            #    boundary_exclude += 1
            #    continue

            filename_vec.append(os.path.join(dirpath, file))
            candidate_mag = catalogue_mag[np.where(catalogue_kic == file_kic)[0]]
            try:
                mags.append(candidate_mag[0])
            except:
                print(candidate_mag)
                print(file_kic)
                mags.append(-99)
            file_kic_vec.append(file_kic)
            numax_sigma.append(cat_numax_sig[np.where(cat_kic == file_kic)[0]])

print('Counted False Positives: ', count)
print('Counted Boundary Excluded: ', boundary_exclude)
print('Initial File Number: ', original_filecount)
print('Filtered File Number: ', len(filename_vec))

mags = np.array(mags)
file_kic = np.array(file_kic_vec)
filenames = np.array(filename_vec)
numax_sigma = np.array(numax_sigma)
print('Mags Length: ', len(mags))
print('File KIC Length (Unfiltered 572,377): ', len(file_kic))
file_kic = file_kic[mags != -99] # these are for ENTIRE FOLDER
numax_sigma = numax_sigma[mags != -99]
filenames = filenames[mags != -99]
mags = mags[mags != -99]

print('Total NaNs in Mag (First Filter): ', np.sum(np.isnan(mags)))

file_kic = file_kic[~np.isnan(mags)]
numax_sigma = numax_sigma[~np.isnan(mags)]
filenames = filenames[~np.isnan(mags)]
mags = mags[~np.isnan(mags)]

print('Total NaNs in Mag (Second Filter): ', np.sum(np.isnan(mags)))


obs_len=27 #365.*4
uncertainty_factor = np.sqrt(365.*4./obs_len)
numax_sigma = np.array(numax_sigma)*uncertainty_factor

unique_id = np.unique(file_kic)
train_ids, test_ids = train_test_split(unique_id, test_size =0.15, random_state = 137)
train_ids, val_ids = train_test_split(train_ids, test_size =0.1765, random_state = 137)

train_kic = file_kic[np.in1d(file_kic, train_ids)]
val_kic = file_kic[np.in1d(file_kic, val_ids)] 
test_kic = file_kic[np.in1d(file_kic, test_ids)]

train_filenames = filenames[np.in1d(file_kic, train_ids)]
val_filenames = filenames[np.in1d(file_kic, val_ids)]
test_filenames = filenames[np.in1d(file_kic, test_ids)]

train_numax_sigma = numax_sigma[np.in1d(file_kic, train_ids)]
val_numax_sigma = numax_sigma[np.in1d(file_kic, val_ids)]
test_numax_sigma = numax_sigma[np.in1d(file_kic, test_ids)]

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
print('Test IDs: ', test_ids)
print('Setting up generators... ')


def validate(model, val_dataloader):
    model.eval() # set to evaluate mode
    val_loss = 0
    val_cum_mape = 0
    val_cum_mae = 0   
    val_batches = 0
    val_flag_array = []
    for j, val_data in enumerate(val_dataloader, 0): #indices,scaled_indices, numax, teff, fe_h, age, tams_age

        val_image = val_data[0].cuda()
        val_numax = val_data[1].cuda()
        val_flag = val_data[3].cuda()
        
        try:
            val_numax_sigma = val_data[2].cuda()
            val_flag = val_data[3].cuda()
        except:
            val_numax_sigma = torch.zeros_like(val_numax).cuda()
            val_flag = torch.zeros_like(val_numax).cuda()

        if len(val_image.size()) <= 2:
            print('Insufficient Batch!')
            continue
        with torch.no_grad():
            pi, sigma, mu = model(input_image=torch.squeeze(val_image, -1).float())
        
            val_batch_loss = mdn_loss(pi, sigma, mu, target=val_numax.float()) # log-likelihood loss
            #val_batch_loss = compute_full_crps(pi, mu, sigma, label=val_numax.float(), label_err=val_numax_sigma.float(), loss=True) # CRPS loss
            #val_batch_loss = compute_closed_crps(pi, mu, sigma, label=val_numax.float())
            val_pred_npy = dist_mu(pi, mu).data.cpu().numpy().reshape(-1,1)
            val_truth_npy = val_numax.data.cpu().numpy()

            val_mape = mean_absolute_percentage_error(y_true=val_truth_npy.squeeze(),y_pred=val_pred_npy.squeeze())
            val_mae = mean_absolute_error(y_true=val_truth_npy.squeeze(),y_pred=val_pred_npy.squeeze())
       

            val_loss += val_batch_loss.item()
            val_cum_mape += val_mape
            val_cum_mae += val_mae
            val_flag_array.append(val_flag.data.cpu().numpy())
            val_batches += 1

    print('Sample Numax: ', val_numax[-1])

    print('Sample Truth: ', val_truth_npy[-1])
    print('Sample Pred: ', val_pred_npy[-1])
    print('Sample Age Pi: ', pi[-1])
    print('Sample Sigma: ', sigma[-1])
    print('Sample Mu: ', mu[-1])
    val_flag_array = np.concatenate(val_flag_array, axis=0)
    print('Nb 0 Val Flags (Could Not Augment): ', np.sum(val_flag_array == 0))
    print('Nb 1 Val Flags (Augmented Draw): ', np.sum(val_flag_array == 1))
    print('Nb 2 Val Flags (Unaugmented Draw): ', np.sum(val_flag_array == 2))

    return (val_loss/val_batches), (val_cum_mape/val_batches), (val_cum_mae/val_batches)


def train(model, model_optimizer, input_image, input_numax, input_numax_sigma):

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

    #train_gen = NPZ_Dataset(root=root_folder, dim=(128,128), extension='npz', indices = train_indices)
    train_gen = NPZ_Dataset_OnTheFly(filenames = train_filenames,kic=train_kic,mag=train_mags,numax_sigma = train_numax_sigma, dim=(128,128), random_draws=False, add_noise=False, reverse=False, scale=True)
    train_dataloader = utils.DataLoader(train_gen, shuffle=True, batch_size=32, num_workers=3)

    #val_gen = NPZ_Dataset(root=root_folder, dim=(128,128), extension='npz', indices = val_indices)
    val_gen = NPZ_Dataset_OnTheFly(filenames = val_filenames,kic=val_kic,mag=val_mags,numax_sigma = val_numax_sigma,dim=(128,128), random_draws=False, add_noise=False, reverse=False, scale=True)
    val_dataloader = utils.DataLoader(val_gen, shuffle=True, batch_size=32, num_workers=3)


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
        flag_vec = []
        model.train()  # set to training mode
    
        for i, data in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader), unit='batches'):
            train_batches += 1
    
            image = data[0].cuda()
            numax = data[1].cuda()

            try:
                numax_sigma = data[2].cuda()
                flag = data[3].cuda()
            except:
                numax_sigma = torch.zeros_like(numax).cuda()
                flag = torch.zeros_like(numax).cuda()
            
            if len(image.size()) <= 2:
                print('Insufficient Batch!')
                continue
    
            loss, mape, mae = train(model, model_optimizer, image, numax, numax_sigma)
            train_loss += loss  # Summing losses across all batches, so if you want the mean for EACH sample, divide by number of batches
            mape_cum += mape
            mae_cum += mae
            flag_vec.append(flag.data.cpu().numpy())
    
        train_loss = train_loss / train_batches
        train_mape = mape_cum / train_batches
        train_mae = mae_cum / train_batches
        flag_vec = np.concatenate(flag_vec,axis=0)
        val_loss, val_mape, val_mae = validate(model, val_dataloader)
        scheduler.step(train_loss)  # reduce LR on loss plateau

        print('Nb 0 Train Flags (Could Not Augment): ', np.sum(flag_vec == 0))
        print('Nb 1 Train Flags (Augmented Draw): ', np.sum(flag_vec == 1))
        print('Nb 2 Train Flags (Unaugmented Draw): ', np.sum(flag_vec == 2))

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
                filename = '27d_FIXEDSCALE_SLOSCFilter_Drop=0.1_NoAugment-MAPE:%.2f-MAE:%.2f' % (
                val_mape, val_mae)
                filepath = '/home/z3384751/K2Detection/ClassifyModels/2D_classifier/Pytorch_MDN_SLOSH/'
                alt_filepath = '/data/marc/Pytorch_MDN_SLOSH/'
                try:
                    torch.save(model.state_dict(), os.path.join(filepath, filename))
                    print('Model saved to %s' %os.path.join(filepath, filename))
                except:
                    torch.save(model.state_dict(), os.path.join(alt_filepath, filename))
                    print('Model saved to %s' %os.path.join(alt_filepath, filename))
                # to load models:
                # the_model = TheModelClass(*args, **kwargs)
                # the_model.load_state_dict(torch.load(PATH))
            else:
                print('No improvement over the best of %.4f' % best_loss)



def test_model():
    test_model = SLOSH_Regressor(num_gaussians = 4)
    saved_model_dict = '/home/z3384751/K2Detection/ClassifyModels/2D_classifier/Pytorch_MDN_SLOSH/V6-UpperMag13-OTF_FIXED_TrainMore-CRPS_4yrUncertainty-MDN_WITH_Drop-50percentAug_to_13_Softplus-MAPE:3.86-MAE:2.34'
    test_model.load_state_dict(torch.load(saved_model_dict))
    test_model.cuda()
    test_model.eval()
    torch.backends.cudnn.benchmark=True

    #test_gen = NPZ_Dataset(root=root_im_folder, dim=(128,128), extension='npz', indices = test_indices) # no perturbations
    test_gen = NPZ_Dataset_OnTheFly(filenames = test_filenames,kic=test_kic,mag=test_mags,numax_sigma = test_numax_sigma,dim=(128,128), random_draws=True)
    test_dataloader = utils.DataLoader(test_gen, shuffle=True, batch_size=32, num_workers=10)

    test_cum_mape = 0
    test_cum_mae = 0
    
    test_pi, test_mu, test_sigma, test_numax_array = [], [], [], []
    test_image_array = []
    pit_array = []
  
    for k, test_data in tqdm(enumerate(test_dataloader, 0), total=len(test_dataloader)): #indices,scaled_indices, numax, teff, fe_h, age, tams_age

        test_image = test_data[0].cuda()
        test_numax = test_data[1].cuda()
        
        if len(test_image.size()) <= 2:
            print('Insufficient Batch!')
            continue
        with torch.no_grad():
            pi, sigma, mu = test_model(input_image=torch.squeeze(test_image).float())
            pit = compute_pit(pi, mu, sigma, label=test_numax)
            test_pred_npy = dist_mu(pi, mu).data.cpu().numpy().reshape(-1,1)
            test_truth_npy = test_numax.data.cpu().numpy()

        test_mape = mean_absolute_percentage_error(y_true=test_truth_npy.squeeze(),y_pred=test_pred_npy.squeeze())
        test_mae = mean_absolute_error(y_true=test_truth_npy.squeeze(),y_pred=test_pred_npy.squeeze())

        test_pi.append(pi.data.cpu().numpy())
        test_sigma.append(sigma.data.cpu().numpy())
        test_mu.append(mu.data.cpu().numpy())
        test_numax_array.append(test_numax.data.cpu().numpy())
        #test_image_array.append(test_image.data.cpu().numpy())
        pit_array.append(pit.data.cpu().numpy())

    test_pi = np.concatenate(test_pi, axis=0)
    test_sigma = np.concatenate(test_sigma, axis=0)
    test_mu = np.concatenate(test_mu, axis=0)
    test_numax_array = np.concatenate(test_numax_array, axis=0)
    #test_image_array = np.concatenate(test_image_array, axis=0)
    test_pit = np.concatenate(pit_array, axis=0)

    numax_pred_mean = dist_mu_npy(test_pi, test_mu[:,:,0])
    numax_pred_var = dist_var_npy(pi=test_pi, mu=test_mu[:,:,0], mixture_mu = dist_mu_npy(test_pi, test_mu[:,:,0]), 
                          sigma=test_sigma[:,:,0])
    numax_pred_sigma = np.sqrt(numax_pred_var)
    numax_truth_npy = test_numax_array
    numax_pred_sigma_fractional = numax_pred_sigma*100/numax_pred_mean

    leading_coefficient_value = np.max(test_pi, axis=1)
    print('Numax Truth NPY Size: ', numax_truth_npy.shape)
    print('Numax Pred Mean Size: ', numax_pred_mean.shape)
    print('Minimum Sigma predicted: ', np.min(numax_pred_sigma))
    print('wtf predicted negative values? : ', numax_pred_mean[numax_pred_mean <= 0])
    print('wtf number of predicted zero sigmas? ',len(numax_pred_sigma[numax_pred_sigma == 0]))
    print('wtf pred numax for these zero sigmas ',numax_pred_mean[numax_pred_sigma == 0])
    print('wtf true numax for these zero sigmas ',numax_truth_npy[numax_pred_sigma == 0])

    #for i, im in enumerate(test_image_array[numax_pred_sigma > 50]):
    #    plt.imshow(im.squeeze(), cmap='gray')
    #    plt.waitforbuttonpress()
    #    plt.close()

    #print('Fitering Out Those With Fractional Sigmas Larger than 50%...')
    #numax_pred_mean = numax_pred_mean[numax_pred_sigma < 50]
    #numax_truth_npy = numax_truth_npy[numax_pred_sigma < 50]
    #numax_pred_sigma = numax_pred_sigma[numax_pred_sigma < 50]
    numax_pred_sigma[numax_pred_sigma == 0] += 1e-3

    numax_mape = mean_absolute_percentage_error(y_true=numax_truth_npy.squeeze(),y_pred=numax_pred_mean.squeeze())
    numax_mae = mean_absolute_error(y_true=numax_truth_npy.squeeze(),y_pred=numax_pred_mean.squeeze())
    print('Numax Mape: ', numax_mape)
    print('Numax Mae: ', numax_mae)

    abs_deviation = np.abs(numax_pred_mean.squeeze() - numax_truth_npy.squeeze())*100/numax_truth_npy.squeeze()

    plt.hist((100*numax_pred_sigma/numax_pred_mean).squeeze(), bins=100)
    plt.title('Distribution of Fractional Sigmas')
    plt.show()
    plt.close()

    import matplotlib as mpl
    mpl.rcParams.update({'font.size': 9})
    fig = plt.figure(figsize=(6, 18))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)


    ax1.errorbar(numax_truth_npy.squeeze(), numax_pred_mean.squeeze()
             , yerr=numax_pred_sigma.squeeze(), fmt='ro', ecolor='k', markersize=2)
    ax1.plot([0,290], [0,290], c='lime', ls='--',lw=2, zorder=3)
    ax1.set_title('MAPE: %.2f%%, MAE: %.2f $\\mu$Hz' %(numax_mape, numax_mae), fontsize=6)
    ax1.set_xlabel('$\\nu_{\\mathrm{max, true}}$ ($\\mu$Hz)', labelpad=-1)
    ax1.set_ylabel('$\\nu_{\\mathrm{max, pred}}$ ($\\mu$Hz)')

    ax2.errorbar(numax_pred_mean.squeeze(), numax_pred_mean.squeeze()-numax_truth_npy.squeeze(), yerr=numax_pred_sigma.squeeze(), fmt='ro',ecolor='k', markersize=2)
    ax2.plot([0, 290], [0,0], c='lime', ls='--',lw=2, zorder=3)
    ax2.set_ylabel('$\\nu_{\\mathrm{max, pred}}$ - $\\nu_{\\mathrm{max, true}}$ ($\\mu$Hz)')
    ax2.set_xlabel('$\\nu_{\\mathrm{max, pred}}$ ($\\mu$Hz)', labelpad=-1)

    numax_zscore = (numax_pred_mean.squeeze()-numax_truth_npy.squeeze())**2/(numax_pred_sigma.squeeze())**2
    ax3.hist(numax_zscore, density=True, bins=1000, histtype='step', color='k')
    ax3.set_ylabel('pdf')
    ax3.set_xlim([-4, 4])
    ax3.set_xlabel('$\\nu_{\\mathrm{max}}$ z-statistic squared', labelpad=-1)
    ax3.set_title('Z-Median: %.2f, Z-Mean: %.2f, Z-Sigma: %.2f, Median $\\sigma$: %.2f%%' %(np.median(numax_zscore),
                                                       np.mean(numax_zscore),
                                                       np.std(numax_zscore), np.median(numax_pred_sigma_fractional)), fontsize=6)
    plt.tight_layout(h_pad=7.5, pad=3)
    plt.show()
    plt.close()
    from matplotlib.colors import LogNorm

    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    print('Abs Deviation: ', abs_deviation)
    print('Numax Pred Sigma Fractional: ', numax_pred_sigma_fractional)
    im1= ax1.scatter(numax_pred_mean, leading_coefficient_value, c=numax_pred_sigma_fractional, norm=LogNorm())
    ax1.set_xlabel('Predicted $\\nu_{\\mathrm{max}}$ ($\\mu$Hz)')
    ax1.set_ylabel('Leading mixing coefficient')
    fig.colorbar(im1, ax=ax1, label='Fractional $\\nu_{\\mathrm{max}}$ Uncertainty (%)')
    im2 = ax2.scatter(numax_pred_mean, leading_coefficient_value, c=abs_deviation, norm=LogNorm())
    ax2.set_xlabel('Predicted $\\nu_{\\mathrm{max}}$ ($\\mu$Hz)')
    ax2.set_ylabel('Leading mixing coefficient')
    fig.colorbar(im2, ax=ax2, label='Fractional $\\nu_{\\mathrm{max}}$ Abs. Deviation (%)')

    plt.show()
    plt.close()

    plt.hist(test_pit, bins=50, histtype='step')
    plt.xlabel('Probability Integral Transform')
    plt.ylabel('Count Frequency')
    plt.show()


    jie_numax_data = pd.read_csv('/home/z3384751/K2Detection/JieData_Full2018.txt', delimiter='|', header=0)
    jie_numax = jie_numax_data['numax'].values
    jie_numax_sig = jie_numax_data['numax_err'].values

    plt.scatter(numax_pred_mean.squeeze(),numax_pred_sigma.squeeze()/numax_pred_mean.squeeze(), s=4, label='SLOSH 27 day')
    plt.scatter(jie_numax, jie_numax_sig/jie_numax, s=4, label='Yu et al. (2018)')
    plt.xlabel('$\\nu_{\\mathrm{max, pred}}$ ($\\mu$Hz)')
    plt.ylabel('$\\sigma/\\nu_{\\mathrm{max, pred}}$')
    plt.yscale('log')
    plt.legend()
    plt.ylim([0.001, 10])
    plt.show()


def test_model_magnitude_variation():

  
    test_model = SLOSH_Regressor_No_Drop(num_gaussians = 4)
    saved_model_dict = '/home/z3384751/K2Detection/ClassifyModels/2D_classifier/Pytorch_MDN_SLOSH/V6-UpperMag13-OTF_FIXED_TrainMore-CRPS_4yrUncertainty-MDN_WITH_Drop-50percentAug_to_13_Softplus-MAPE:3.86-MAE:2.34'
    test_model.load_state_dict(torch.load(saved_model_dict))
    test_model.cuda()
    test_model.eval()
    torch.backends.cudnn.benchmark=True
    root_test_folder = '/data/marc/KEPSEISMIC/LC_Data/PSD/npz_PSD_numax/'#npz_PSD_numax or test_numax
    file_count = 0
    id_vec = []
    print('Parsing files in folder... ')
    for dirpath, dirnames, filenames in os.walk(root_test_folder):
        for filename in filenames:
            if filename.endswith('.npz') & dirpath[-1].isdigit():
                id_vec.append(int(filename.split('-')[0]))
                file_count += 1
    id_vec = np.array(id_vec)
 
    file_indices = np.arange(file_count)
    unique_id = np.unique(id_vec)
    train_ids, test_ids = train_test_split(unique_id, test_size =0.15, random_state = 137)
    train_ids, val_ids = train_test_split(train_ids, test_size =0.1765, random_state = 137)
    train_indices = file_indices[np.in1d(id_vec, train_ids)]  
    val_indices = file_indices[np.in1d(id_vec, val_ids)]  
    test_indices = file_indices[np.in1d(id_vec, test_ids)]
  
    print('Total Files: ', len(id_vec))
    print('Total Unique IDs: ', len(unique_id))
    print('Total Train Files: ', len(train_indices))
    print('Train Unique IDs: ', len(train_ids))
    print('Total Val Files: ', len(val_indices))
    print('Test Unique IDs: ', len(test_ids))
    print('Total Test Files: ', len(test_indices))   
    print('Number of Training IDs in Test IDs: ', np.sum(np.in1d(train_ids, test_ids)))
    print('Number of Training IDs in Val IDs: ', np.sum(np.in1d(train_ids, val_ids))) 


    fixed_magnitudes = [2,4,6,8,10,11,12]
    for magz in fixed_magnitudes:
        print('Predicting Using Magnitude %d' %magz)
        print('Test Indices Length: ', len(test_indices))
        test_gen = NPZ_Dataset_Magnitude_Variation(root=root_test_folder, dim=(128,128), extension='npz', indices = test_indices,catalogue_kic=kepmag_data['KIC'].values, catalogue_mag=kepmag_data['Kep_Mag'].values, label_uncertainty=True, obs_len= 365*4., fixed_magnitude= magz)
        test_dataloader = utils.DataLoader(test_gen, shuffle=True, batch_size=32, num_workers=5)
        print('Number of Samples with This Magnitude: ', len(test_gen))
        test_cum_mape = 0
        test_cum_mae = 0
        skip_batch = 0
        test_pi, test_mu, test_sigma, test_numax_array, test_indices_array, test_flag_array  = [], [], [], [], [], []
  
        for k, test_data in tqdm(enumerate(test_dataloader, 0), total=len(test_dataloader)): #indices,scaled_indices, numax, teff, fe_h, age, tams_age

            test_image = test_data[0].cuda()
            test_numax = test_data[1].cuda()
            test_flag = test_data[-2].cuda()
            test_index = test_data[-1].cuda()
            test_index = test_index[test_flag == 1]
            test_image = test_image[test_flag == 1]
            test_numax = test_numax[test_flag == 1]
    
            if test_image.size()[0] <= 1:
                print('Insufficient Batch!')
                skip_batch += 1
                continue

            with torch.no_grad():
                pi, sigma, mu = test_model(input_image=torch.squeeze(test_image).float())
                test_pred_npy = dist_mu(pi, mu).data.cpu().numpy().reshape(-1,1)
                test_truth_npy = test_numax.data.cpu().numpy()

            test_mape = mean_absolute_percentage_error(y_true=test_truth_npy.squeeze(),y_pred=test_pred_npy.squeeze())
            test_mae = mean_absolute_error(y_true=test_truth_npy.squeeze(),y_pred=test_pred_npy.squeeze())

            test_pi.append(pi.data.cpu().numpy())
            test_sigma.append(sigma.data.cpu().numpy())
            test_mu.append(mu.data.cpu().numpy())
            test_numax_array.append(test_numax.data.cpu().numpy())
            test_indices_array.append(test_index.data.cpu().numpy())
            test_flag_array.append(test_flag.data.cpu().numpy())

        test_pi = np.concatenate(test_pi, axis=0)
        test_sigma = np.concatenate(test_sigma, axis=0)
        test_mu = np.concatenate(test_mu, axis=0)
        test_numax_array = np.concatenate(test_numax_array, axis=0)
        test_indices_array = np.concatenate(test_indices_array, axis=0)
        test_flag_array = np.concatenate(test_flag_array, axis=0)

        numax_pred_mean = dist_mu_npy(test_pi, test_mu[:,:,0])
        numax_pred_var = dist_var_npy(pi=test_pi, mu=test_mu[:,:,0], mixture_mu = dist_mu_npy(test_pi, test_mu[:,:,0]), 
                          sigma=test_sigma[:,:,0])
        numax_pred_sigma = np.sqrt(numax_pred_var)
        numax_truth_npy = test_numax_array
        numax_pred_sigma_fractional = numax_pred_sigma*100/numax_pred_mean
        print('Batches Skipped: ', skip_batch)
        print('Prediction Length: ', len(numax_truth_npy))
        print('Test Flag Length: ', len(test_flag_array))
        print('Test Flag 0 : ', np.sum(test_flag_array == 0))
        print('Test Flag 1 : ', np.sum(test_flag_array == 1))
        print('Test Flag 2 : ', np.sum(test_flag_array == 2))
        np.savez_compressed('Prediction-Mag+Indices(FIXED_Augto13)-%d' %magz, numax_pred_mean = numax_pred_mean, numax_pred_sigma = numax_pred_sigma, numax_truth_npy = numax_truth_npy, indices = test_indices_array)


def test_individual_star():
    test_model = SLOSH_Regressor_No_Drop(num_gaussians = 4)
    saved_model_dict = '/home/z3384751/K2Detection/ClassifyModels/2D_classifier/Pytorch_MDN_SLOSH/V6-70percentfloorOTF_TrainMore-CRPS_4yrUncertainty-MDN_No_Drop-Softplus-MAPE:1.64-MAE:0.79'
    test_model.load_state_dict(torch.load(saved_model_dict))
    test_model.cuda()
    test_model.eval()

    test_data = '/home/z3384751/K2Detection/kplr002568888_473_COR_PSD_filt_inp.fits'
    #data = np.load(test_data)
    with fits.open(test_data) as data_psd:
        df_p = pd.DataFrame(data_psd[0].data)
        freq = df_p.iloc[:, 0].values
        power = df_p.iloc[:, 1].values
        freq *= 1e6
    
    #X = np.expand_dims(data['im'], 0)# shape should be (1,128,128)
    X = np.expand_dims(ps_to_array(freq, power, minfreq=3.),0)

    X = torch.from_numpy(X.copy()).float().cuda()
    with torch.no_grad():
        pi, sigma, mu = test_model(input_image=X)
    pi = pi.data.cpu().numpy()
    sigma = sigma.data.cpu().numpy()
    mu = mu.data.cpu().numpy()
    print('Pi: ', pi)
    print('Sigma: ', sigma)
    print('Mu: ', mu)
    select_pi = pi[pi > 0.01]
    select_mu = mu[pi>0.01]
    select_sigma = sigma[pi>0.01]
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(X.squeeze(), cmap='gray')
    ax1.set_title('KIC 2568888 Image Input to SLOSH')
    ax2.plot(freq, power)
    ax2.set_xlim([3, 20])
    ax2a = ax2.twinx()
    c=['orange','purple']
    for i in range(len(select_pi)):
        ax2.axvline(x=select_mu[i], c=c[i],ls='--')
        ax2.set_ylabel('Power Density')
        ax2.set_xlabel('Frequency (uHz)')
        ax2a.set_ylabel('pdf')
        ax2a.plot(freq, select_pi[i]*(norm.pdf(freq, loc=select_mu[i], scale=select_sigma[i])), c=c[i], label='$p=%.2f$'%select_pi[i])
        ax2a.legend(loc='best')
    ax2.set_title('Prediction on Linear-Scale PSD')
    plt.tight_layout(w_pad=4)
    plt.show()
    

train_model()
#test_model()
#test_individual_star()
#test_model_magnitude_variation()
