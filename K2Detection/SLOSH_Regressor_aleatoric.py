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

import math
import warnings
warnings.filterwarnings("ignore")

print('Sci-kit Version {}.'.format(sklearn.__version__))

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
                minfreq=3., maxfreq=283., minpow=3., maxpow=3e7, scale=False):
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
    def __init__(self, filenames, kic, mag, numax_sigma, dim, random_draws=False, add_noise=False, reverse=False, scale=False, scale_pixel=False):
    
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
        self.scale_pixel = scale_pixel
        self.conversion_a = 3
        self.conversion_b = (1. / 128.) * np.log(283. / 3.)

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
        if self.scale_pixel:
            pix = (1/self.conversion_b)*np.log(y/conversion_a) 
            y = pix
       
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
 

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true-y_pred))

def aleatoric_loss(y_true, y_pred, log_var): #here pred_var should be [prediction, variance], y_true is true numax
    loss = (torch.abs(y_true - y_pred)/y_true)*(torch.exp(-log_var)) + log_var
    return torch.mean(loss) # K.mean loss?

def NLL(y_true, y_pred, var):
    """ Compute the negative log likelihood """
    diff = torch.sub(y_true, y_pred)
    # Compute loss 
    loss = torch.div(diff**2, 2*var) + 0.5*torch.log(var)
    return torch.mean(loss)

def initialization(model):
    for name, param in model.named_parameters():  # initializing model weights
        if 'bias' in name:
            nn.init.constant_(param, 0.00)
        elif 'weight' in name:
            nn.init.xavier_uniform_(param)


class SLOSH_Regressor(nn.Module):
    ## Your plain old regressor ##

    def __init__(self):
        super(SLOSH_Regressor, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, padding=2) # same padding 2P = K-1
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1) # same padding 2P = K-1
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1) # same padding 2P = K-1
        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2)
        self.pool3 = nn.MaxPool2d(2,2)
        self.drop1 = nn.Dropout(0.1)
        self.linear1 = nn.Linear(16*16*16, 128)
        self.output = nn.Linear(128, 1)

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

        return self.output(linear1)


class SLOSH_Regressor_Aleatoric(nn.Module):
    ## Regressor version with additional output to express aleatoric uncertainty ##

    def __init__(self):
        super(SLOSH_Regressor_Aleatoric, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, padding=2) # same padding 2P = K-1
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1) # same padding 2P = K-1
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1) # same padding 2P = K-1
        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2)
        self.pool3 = nn.MaxPool2d(2,2)
        self.drop1 = nn.Dropout(0.1)
        self.linear1 = nn.Linear(16*16*16, 128)
        self.output = nn.Linear(128, 1)
        self.var = nn.Linear(128, 1)
        self.softplus = nn.Softplus()

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
        # Avoid infinities due to taking the exponent
        varx = self.var(linear1)
        softplus = self.softplus(varx) + 1e-6
        softplus = torch.where(softplus==float('inf'), varx, softplus)

        return self.output(linear1), softplus


aleatoric = True # do we want to include aleatoric uncertainty in our estimates?
scale_pixel = False # do we want to use the pixel scaling form like the previous Keras version used?
weight_pixel_mse = False # only used if scale_pixel is True. Weights the low and high numax more.



if aleatoric:
    model = SLOSH_Regressor_Aleatoric()
else:
    model = SLOSH_Regressor()
model.cuda()
torch.backends.cudnn.benchmark=True
print('CUDNN Enabled? ', torch.backends.cudnn.enabled)
print(str(model))
initialization(model)

learning_rate = 0.001
model_optimizer = optim.Adam(model.parameters(), lr = learning_rate)
scheduler = ReduceLROnPlateau(model_optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr = 1E-6)
conversion_a = 3
conversion_b = (1. / 128.) * np.log(283. / 3.)


root_folder = '/data/marc/KEPSEISMIC/LC_Data/PSD/npz_PSD_numax_365d/'#npz_PSD_numax or test_numax
kepmag_data = pd.read_csv('/home/z3384751/K2Detection/DR25_KIC_RADec.dat', delim_whitespace=True, header=0)
catalogue_kic = kepmag_data['KIC'].values
catalogue_mag = kepmag_data['Kep_Mag'].values

print('Parsing files in folder... ')


numax_data = pd.read_csv('/home/z3384751/K2Detection/JieData_Full2018.txt', delimiter='|', header=0)
cat_numax = numax_data['numax'].values
cat_numax_sig = numax_data['numax_err'].values
cat_kic = numax_data['KICID'].values


####### STATISTICAL TEST #######
slosc_data = np.load('slosc_result_365d.npz', allow_pickle=True)
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


obs_len=365 #365.*4
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


def validate(model, val_dataloader, aleatoric, scale_pixel, weight_pixel_mse):
    model.eval() # set to evaluate mode
    val_loss = 0
    val_cum_mape = 0
    val_cum_mae = 0   
    val_batches = 0
    val_frac_uncertainty = 0
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
            if aleatoric:
                pred_numax, var = model(input_image = torch.squeeze(val_image, -1).float())
                val_batch_loss = NLL(y_true=val_numax.float(), y_pred=pred_numax, var=var)
                val_frac_uncertainty += torch.mean(torch.sqrt(var)*100./pred_numax).item() #average fractional uncertainty
            else:
                pred_numax = model(input_image = torch.squeeze(val_image, -1).float())
                val_batch_loss = torch.nn.MSELoss()(pred_numax.squeeze(1), val_numax.float())

            ## Convert to pixel and weight data like Keras version??
            if scale_pixel:
                pred_numax_npy = conversion_a * (np.exp((pred_numax.data.cpu().numpy()) * conversion_b))
                val_numax_npy = conversion_a * (np.exp((val_numax.data.cpu().numpy()) * conversion_b))
            else:
                pred_numax_npy = pred_numax.data.cpu().numpy()
                val_numax_npy = val_numax.data.cpu().numpy()

            val_pred_npy = pred_numax_npy
            val_truth_npy = val_numax_npy

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

    val_flag_array = np.concatenate(val_flag_array, axis=0)
    #print('Nb 0 Val Flags (Could Not Augment): ', np.sum(val_flag_array == 0))
    #print('Nb 1 Val Flags (Augmented Draw): ', np.sum(val_flag_array == 1))
    #print('Nb 2 Val Flags (Unaugmented Draw): ', np.sum(val_flag_array == 2))

    return (val_loss/val_batches), (val_cum_mape/val_batches), (val_cum_mae/val_batches), (val_frac_uncertainty/val_batches)


def train(model, model_optimizer, input_image, input_numax, input_numax_sigma, aleatoric, scale_pixel, weight_pixel_mse):

    model_optimizer.zero_grad()

    # Combined forward pass
    if aleatoric:
        pred_numax, var = model(input_image = torch.squeeze(input_image, -1).float())
        loss = NLL(y_true=input_numax.float(), y_pred=pred_numax, var=var)
        frac_uncertainty = torch.mean(torch.sqrt(var)*100./pred_numax).item()
    else:
        pred_numax = model(input_image = torch.squeeze(input_image, -1).float())
        loss = torch.nn.MSELoss()(pred_numax.squeeze(1), input_numax.float())
        frac_uncertainty = 0

    loss.backward()

    ## Convert to pixel and weight data like Keras version??
    if scale_pixel:
        pred_numax_npy = conversion_a * (np.exp((pred_numax.data.cpu().numpy()) * conversion_b))
        input_numax_npy = conversion_a * (np.exp((input_numax.data.cpu().numpy()) * conversion_b))
    else:
        pred_numax_npy = pred_numax.data.cpu().numpy()
        input_numax_npy = input_numax.data.cpu().numpy()

    # Calculate loss and backpropagate

    mape = mean_absolute_percentage_error(y_true=input_numax_npy.squeeze(), y_pred=pred_numax_npy.squeeze())
    mae = mean_absolute_error(y_true=input_numax_npy.squeeze(), y_pred=pred_numax_npy.squeeze())

    model_optimizer.step()

    return loss.item(), mape, mae, frac_uncertainty


def train_model():

    train_gen = NPZ_Dataset_OnTheFly(filenames = train_filenames,kic=train_kic,mag=train_mags,numax_sigma = train_numax_sigma, dim=(128,128), random_draws=False, add_noise=False, reverse=False, scale=True, scale_pixel=scale_pixel)
    train_dataloader = utils.DataLoader(train_gen, shuffle=True, batch_size=32, num_workers=3)

    val_gen = NPZ_Dataset_OnTheFly(filenames = val_filenames,kic=val_kic,mag=val_mags,numax_sigma = val_numax_sigma,dim=(128,128), random_draws=False, add_noise=False, reverse=False, scale=True, scale_pixel=scale_pixel)
    val_dataloader = utils.DataLoader(val_gen, shuffle=True, batch_size=32, num_workers=3)


    n_epochs=1000
    model_checkpoint = False #True
    best_loss = 9999
    for epoch in range(1, n_epochs + 1):
        print('---------------------')
        print('Epoch: ', epoch)
        train_loss = 0
        train_frac_uncertainty = 0
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
    
            loss, mape, mae, frac_uncertainty = train(model, model_optimizer, image, numax, numax_sigma, aleatoric, scale_pixel, weight_pixel_mse)
            train_loss += loss  # Summing losses across all batches, so if you want the mean for EACH sample, divide by number of batches
            train_frac_uncertainty += frac_uncertainty
            mape_cum += mape
            mae_cum += mae
            flag_vec.append(flag.data.cpu().numpy())
    
        train_loss = train_loss / train_batches
        train_mape = mape_cum / train_batches
        train_mae = mae_cum / train_batches
        flag_vec = np.concatenate(flag_vec,axis=0)
        val_loss, val_mape, val_mae, val_frac_uncertainty = validate(model, val_dataloader,aleatoric, scale_pixel, weight_pixel_mse)
        scheduler.step(train_loss)  # reduce LR on loss plateau

        print('Nb 0 Train Flags (Could Not Augment): ', np.sum(flag_vec == 0))
        print('Nb 1 Train Flags (Augmented Draw): ', np.sum(flag_vec == 1))
        print('Nb 2 Train Flags (Unaugmented Draw): ', np.sum(flag_vec == 2))

        print('\n\nTrain Loss: ', train_loss)
        print('Train Mape: ', train_mape)
        print('Train Mae: ', train_mae)
        print('Train Average Frac Uncertainty: ', train_frac_uncertainty)
    
        print('Val Loss: ', val_loss)
        print('Val Mape: ', val_mape)
        print('Val Mae: ', val_mae)
        print('Val Average Frac Uncertainty: ', val_frac_uncertainty)
    
        print('Aleatoric? ', aleatoric)
        print('Scale Pixel? ', scale_pixel)
        print('Weighted Pixel MSE? ', weight_pixel_mse)

        model.print_instance_name()

        for param_group in model_optimizer.param_groups:
            print('Current Learning Rate: ', param_group['lr'])
        is_best = val_mape < best_loss
        best_loss = min(val_mape, best_loss)
        print('Current Best Metric: ', best_loss)

        if model_checkpoint:

            if is_best:
                filename = 'STANDARDSCALED_SLOSCFilter_Drop=0.1_NoAugment-MAPE:%.2f-MAE:%.2f' % (
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


    

train_model()
#test_model()
