import numpy as np
seed = 7500
np.random.seed(seed)

from keras.layers import Input, Dropout, Activation, MaxPooling2D, Flatten, BatchNormalization, LSTM, Conv2D, Cropping2D, LeakyReLU, ZeroPadding2D
from keras.models import Model, Sequential, load_model
from keras.layers.core import Dense, Reshape
from keras.constraints import max_norm
from keras.optimizers import SGD, Adam
from keras.regularizers import l1, l2, l1_l2
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from keras.preprocessing.image import ImageDataGenerator
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from sklearn.model_selection import cross_val_score, GridSearchCV, learning_curve, KFold, StratifiedKFold, train_test_split
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, History, TensorBoard
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, brier_score_loss, auc, roc_curve, log_loss, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from keras.layers.merge import concatenate, multiply,add, average
from keras import backend as K
from keras.initializers import glorot_normal, glorot_uniform
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from keras.layers.advanced_activations import LeakyReLU, ELU,PReLU
#from read_activations import get_activations, display_activations
from vis.visualization import visualize_saliency, visualize_activation, visualize_activation_with_losses, get_num_filters
from vis.utils import utils
from vis.input_modifiers import Jitter
from keras import activations
from keras.utils.generic_utils import get_custom_objects
from PIL import Image as pil_image
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic
sc = StandardScaler()

import pandas as pd
import seaborn as sns
import sklearn, os, re, glob, matplotlib, h5py, math, pylab, copy, keras, csv
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf # This next few lines makes it such that only 25% of the Titan XP's memory is used for each training
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config)) # 

print('Sci-kit Version {}.'.format(sklearn.__version__))
print('Keras Version {}.'.format(keras.__version__))
print('Tensorflow Version {}.'.format(tf.__version__))

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

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def weighted_mean_squared_error(y_true, y_pred):
    return K.mean((K.square(y_pred - y_true))*K.square(y_true-64), axis=-1)

def aleatoric_loss(y_true, pred_var): #here pred_var should be [prediction, variance], y_true is true numax
    y_pred = pred_var[:,0]
    log_var = pred_var[:, 1]
    loss = (K.abs(y_true - y_pred)/y_true)*(K.exp(-log_var)) + log_var
    print('LOG VAR: ', log_var)
    return K.mean(loss, -1) # K.mean loss?


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

class npy_generator(keras.utils.Sequence):
    def __init__(self, filenames, kic, numax, labels, mags, batch_size, dim, extension = '.npz', shuffle = True, random_draws=False, draw_probability=0.5, init_shuffle=False):
        # filenames = train_filenames, kic=train_kic, numax=train_numax, labels = train_labels, mags=train_mags

        self.batch_size = batch_size
        self.extension = extension # file extension
        self.filenames = filenames
        self.file_kic = kic
        self.mags = mags
        self.numax = numax
        self.draw_probability = draw_probability # likelihood of not augmenting data at all      
        self.subfolder_labels = labels # for binary classification
        self.shuffle = shuffle # shuffles data after every epoch
        self.dim=dim # image/2D array dimensions
        self.random_draws = random_draws # chance to boost noise floor to 11th Magnitude
        self.tess_nfactor = 1.0
        self.init_shuffle=init_shuffle
        self.cadence =1764        
        elat = 30.0  # 0 brings forward, 90 backwards
        V = 23.345 - 1.148 * ((np.abs(elat) - 90) / 90.0) ** 2
        self.tess_cc1 = 2.56e-3 * np.power(10.0, -0.4 * (V - 22.8)) * 69 * 21.1 * 21.1 * self.cadence * 4  # pixel scale is 21.1 arcsec and effective collecting area is 69cm^2, 4 CCDs
        self.tess_cc2 = 2.56e-3 * np.power(10.0, -0.4 * (23.345 - 22.8)) * 69 * 21.1 * 21.1 * self.cadence * 4

        self.indexes = np.arange(len(self.filenames))
        print('Max Indexes: ', np.max(self.indexes))
        print('Filename Length: ', len(self.filenames))
     
        assert len(self.indexes) == len(self.file_kic) == len(self.subfolder_labels) == len(self.filenames) == len(self.mags) == len(self.numax)
        if self.init_shuffle:
            np.random.shuffle(self.indexes) # initial shuffle
    def __len__(self):
        return int(np.ceil(len(self.indexes) / float(self.batch_size)))

    def __getitem__(self, index):
        # Generate indexes of the batch
        batch_indices = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Get a list of filenames of the batch

        batch_filenames = [self.filenames[k] for k in batch_indices]
        batch_labels = [self.subfolder_labels[k] for k in batch_indices]
        batch_mag = [self.mags[k] for k in batch_indices]
        batch_numax = [self.numax[k] for k in batch_indices]
        batch_kic = [self.file_kic[k] for k in batch_indices]
        # Generate data
        X, y = self.__data_generation(batch_filenames, batch_labels, batch_mag, batch_numax, batch_kic)
        return X, keras.utils.to_categorical(y, num_classes=2)

    def on_epoch_end(self):
        # Shuffles indices after every epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_filenames, batch_labels, batch_mag, batch_numax, batch_kic):
        # Generates data - this example is repurposed for .npy files
        X = np.empty((len(batch_filenames), *self.dim))
        y = np.empty((len(batch_filenames)), dtype=int)
        for i, ID in enumerate(batch_filenames): # have to do and add noise here
            data = np.load(ID)
            freq = data['freq']
            pow = data['pow']

            noised_pow, image_flag, simulated_magnitude = self.add_noise_level(pow,batch_mag[i], batch_numax[i])
            if len(noised_pow.shape) > 1:
                noised_pow = noised_pow.squeeze(0)

            im = ps_to_array(freq, noised_pow, minfreq=3.)
            #if (image_flag > 0) and (simulated_magnitude > 11): # sanity-check
            #    im_ori = ps_to_array(freq, pow, minfreq=3.)
                #fig = plt.figure(figsize=(12,6))
                #ax1 = fig.add_subplot(121)
                #ax2 = fig.add_subplot(122)
                #ax1.imshow(im, cmap='gray')
                #ax2.imshow(im_ori, cmap='gray')                   
                #plt.savefig('/data/marc/KEPSEISMIC/LC_Data/PSD/test_images/sanity_check/' + str(batch_kic[i]) + '-%.1f_noise-%d.png' %(simulated_magnitude, batch_labels[i]))     
                #plt.close()    
            X[i, :] = im
            y[i] = batch_labels[i]
        return np.expand_dims(X,-1), y


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
        batch_mag = batch_mag.squeeze()

        assert np.sum(np.isnan(batch_mag)) == 0
        assert np.sum(np.isnan(batch_numax)) == 0
        assert np.sum(batch_power) != np.nan
        assert batch_mag > 0


        #print('Batch Mag: ', batch_mag)
        batch_mag_tess = batch_mag - 5 # the apparent magnitude in TESS
        image_flag = 0
        simulated_magnitude = 0
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
            if (self.random_draws) and (random_draw < self.draw_probability):
                image_flag = 2
                return batch_power,image_flag, simulated_magnitude # 50% chance to not augment
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
                image_flag = 1
            else:
                return_power = batch_power
        else:
            return_power = batch_power

        return return_power, image_flag, simulated_magnitude



def weighted_mean_squared_error(y_true, y_pred):
    return K.mean((K.square(y_pred - y_true))*K.square(y_true-64), axis=-1)



def altered_lenet():
    init = glorot_uniform(seed=seed)
    sgd = SGD(lr=0.01)
    elu_alpha = 0.25
    reg = l2(1E-6)
    adam = Adam(clipnorm=1., lr=0.0005)
    input1 = Input(shape=(128, 128, 1))
    drop0 = Dropout(0.25)(input1)
    conv1 = Conv2D(4, kernel_size=(5, 5), padding='same', kernel_initializer=init,
                       kernel_regularizer=reg)(drop0)
    lrelu1 = LeakyReLU(0.1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='valid')(lrelu1)
    conv2 = Conv2D(8, kernel_size=(4, 4), padding='same', kernel_initializer=init,
                       kernel_regularizer=reg)(pool1)
    lrelu2 = LeakyReLU(0.1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='valid')(lrelu2)
    conv3 = Conv2D(16, kernel_size=(4, 4), padding='same', kernel_initializer=init,
                       kernel_regularizer=reg)(pool2)
    lrelu3 = LeakyReLU(0.1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), padding='valid')(lrelu3)

    flat = Flatten()(pool3)
    drop1 = Dropout(0.1)(flat)
    dense1 = Dense(128, kernel_initializer=init, activation='relu', kernel_regularizer=reg)(drop1)
    output = Dense(2, kernel_initializer=init, activation='softmax')(dense1)
    model = Model(input1, output)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def altered_lenet_regressor():
    init = glorot_uniform(seed=seed)
    reg = l2(7.5E-4)
    adam = Adam(clipnorm=1.)
    input1 = Input(shape=(128, 128, 1))
    drop0 = Dropout(0.25)(input1)
    conv1 = Conv2D(4, kernel_size=(5, 5), padding='same', kernel_initializer=init,
               kernel_regularizer=reg)(drop0)
    lrelu1 = LeakyReLU(0.1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='valid')(lrelu1)
    conv2 = Conv2D(8, kernel_size=(3, 3), padding='same', kernel_initializer=init,
               kernel_regularizer=reg)(pool1)
    lrelu2 = LeakyReLU(0.1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='valid')(lrelu2)
    conv3 = Conv2D(16, kernel_size=(2, 2), padding='same', kernel_initializer=init,
               kernel_regularizer=reg)(pool2)
    lrelu3 = LeakyReLU(0.1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), padding='valid')(lrelu3)
    flat = Flatten()(pool3)
    drop1 = Dropout(0.5)(flat)

    dense1 = Dense(1024, kernel_initializer=init, activation='relu', kernel_regularizer=reg)(drop1)
    dense2 = Dense(128, kernel_regularizer=reg, kernel_initializer=init, activation='relu')(dense1)
    output = Dense(1, kernel_initializer=init,name='prediction')(dense2)
    output_var = Dense(1, kernel_initializer=init, name='variance')(dense2)
    pred_var = concatenate([output, output_var], name='pred_var')

    model = Model(input1, [output, pred_var])
    model.compile(optimizer=adam, loss={'prediction': weighted_mean_squared_error,'pred_var': aleatoric_loss}, metrics={'prediction': 'mae'}, loss_weights={'prediction': 1., 'pred_var': .2})
    return model


#root, batch_size, dim, catalogue_kic, catalogue_mag,numax_kic, numax_values, extension = '.npz', shuffle = True, indices=[])

def train_model_bell_array():
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
    # TO TRY: Set flag = 0    

    #fig = plt.figure(figsize=(12,6))
    #jie_kic = numax_data['KIC'].values
    #cat_kic = kepmag_data['KIC'].values
    #cat_mag = kepmag_data['Kep_Mag'].values
    #nan_kics = cat_kic[np.isnan(cat_mag)]
    #print('NaN KICs: ', nan_kics)
    #print('Which of these NaN KICs are in Jie KIC?: ', nan_kics[np.in1d(nan_kics, jie_kic)])
    #plot_kic = cat_kic[~np.isnan(cat_mag)]
    #plot_mag = cat_mag[~np.isnan(cat_mag)]
    #plt.hist(plot_mag[np.in1d(plot_kic,jie_kic)], color='red', histtype='step', label='RG', bins=50)
    #plt.hist(plot_mag[~np.in1d(plot_kic,jie_kic)], color='blue', histtype='step', label='Everything Else', bins=50)
    #plt.show()

    print('Parsing files in folder... ')
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for i, filex in enumerate(filenames): # Getting the mags, KICS, and numax sigma for all stars in catalog
            if filex.endswith('.npz') & dirpath[-1].isdigit(): # I infer the class label '0' or '1' according to subfolder names
                folder_filenames.append(os.path.join(dirpath, filex))
                kicx = int(re.search(r'\d+', filex).group())
                subfolder_labels.append(int(dirpath[-1]))
                candidate_mag = catalogue_mag[np.where(catalogue_kic == kicx)[0]]
                try:
                    mags.append(candidate_mag[0])
                except:                   
                    mags.append(-99)
                file_kic.append(kicx)

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
    train_gen = npy_generator(filenames = train_filenames, kic=train_kic, numax=train_numax, labels = train_labels, mags=train_mags, batch_size=32, dim=(128,128), extension='npz', random_draws=True, draw_probability=0.5, init_shuffle=True) # 50 chance of not augmenting data
    val_gen = npy_generator(filenames = val_filenames, kic=val_kic, numax=val_numax, labels = val_labels, mags=val_mags, batch_size=32, dim=(128,128), extension='npz', random_draws=True, draw_probability = 0, init_shuffle=False)# 0 = all data is augmented, 1.0 =all data is NOT AUGMENTED

 
    model = altered_lenet()  #### choose model here ####
    print("Output Size: ", model.output_shape)
    model.summary()
    print('Train generator length: ', len(train_gen))
    reduce_lr = ReduceLROnPlateau(factor=0.5, patience = 3, verbose=1, monitor='loss')
    early_stop = EarlyStopping(patience=10, verbose = 1, monitor='loss')
    checkpoint = ModelCheckpoint(filepath='/home/z3384751/K2Detection/ClassifyModels/2D_classifier/LC_Classifier_V6/SLOSH_OnTheFly/FIXED_Classifier-2-{val_acc:.4f}-Testing-Max-50percent13Mag-Sampling.h5', monitor='val_acc',
                                       verbose=1, save_best_only=True)
    model.fit_generator(train_gen, epochs = 100, validation_data = val_gen, callbacks=[reduce_lr, early_stop, checkpoint], class_weight={0:1., 1:10.}, workers=15, use_multiprocessing=True, max_queue_size=15)



def test_model_bell_array():
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
        for i, filex in enumerate(filenames): # Getting the mags, KICS, and numax sigma for all stars in catalog
            if filex.endswith('.npz') & dirpath[-1].isdigit(): # I infer the class label '0' or '1' according to subfolder names
                folder_filenames.append(os.path.join(dirpath, filex))
                kicx = int(re.search(r'\d+', filex).group())
                subfolder_labels.append(int(dirpath[-1]))
                candidate_mag = catalogue_mag[np.where(catalogue_kic == kicx)[0]]
                try:
                    mags.append(candidate_mag[0])
                except:                   
                    mags.append(-99)
                file_kic.append(kicx)

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
    test_gen = npy_generator(filenames = val_filenames, kic=val_kic, numax=val_numax, labels = val_labels, mags=val_mags, batch_size=32, dim=(128,128), extension='npz', random_draws=True, draw_probability = 0.0, init_shuffle=False) # 0 = all data is augmented, 1.0 =all data is NOT AUGMENTED

    model = load_model('/home/z3384751/K2Detection/ClassifyModels/2D_classifier/LC_Classifier_V6/SLOSH_OnTheFly/FIXED_Classifier-1-0.9862-Testing-Max-50percent13Mag-Sampling.h5')  #### choose model here ####
    print("Output Size: ", model.output_shape)
    model.summary()
    print('Generator length: ', len(test_gen))
    pred = model.predict_generator(test_gen, workers=15, use_multiprocessing=True, max_queue_size=15, verbose = 1)
    pos_pred = np.empty(len(pred))
    prob = pred[:, 1]
    for i in range(len(pos_pred)):
        if prob[i] >= 0.5:
            pos_pred[i] = 1
        else:
            pos_pred[i] = 0

    print('Accuracy: ', accuracy_score(y_true=val_labels, y_pred=pos_pred))
    print('Confusion Matrix: \n', confusion_matrix(y_true=val_labels, y_pred=pos_pred))
    print('Prob: ', prob)
    print('Val Labels: ', val_labels)
    print('Log Loss: ', log_loss(y_true=val_labels, y_pred=prob-1e-9))
    print('data all augmented, no shuffle')

def train_regressor_bell_array():
    root_folder = '/data/marc/KEPSEISMIC/LC_Data/PSD/npz_PSD_numax'
    kepmag_data = pd.read_csv('/home/z3384751/K2Detection/DR25_KIC_RADec.dat', delim_whitespace=True, header=0)
    file_count = 0

    print('Parsing files in folder... ')
    for dirpath, dirnames, filenames in os.walk(root_folder):
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
    print('Setting up generators... ')

    train_gen = npy_generator_regressor(root=root_folder, batch_size=32, dim=(128,128), extension='npz', indices = train_indices,catalogue_kic=kepmag_data['KIC'].values, catalogue_mag=kepmag_data['Kep_Mag'].values, random_draws=True)
    val_gen = npy_generator_regressor(root=root_folder, batch_size=32, dim=(128,128), extension='npz', indices = val_indices,catalogue_kic=kepmag_data['KIC'].values, catalogue_mag=kepmag_data['Kep_Mag'].values, random_draws=True)

    model = altered_lenet_regressor()  #### choose model here ####
    print("Output Size: ", model.output_shape)
    model.summary()
    reduce_lr = ReduceLROnPlateau(factor=0.5, patience = 5, verbose=1, monitor='val_prediction_mean_absolute_error')
    early_stop = EarlyStopping(patience=10, verbose = 1, monitor='val_prediction_mean_absolute_error')
    checkpoint = ModelCheckpoint(filepath='/home/z3384751/K2Detection/ClassifyModels/2D_classifier/28d_RG_numax/Bell_Arrays/OnTheFly/Regressor-{val_prediction_mean_absolute_error:.4f}.h5', monitor='val_prediction_mean_absolute_error', verbose=1, save_best_only=True)
    
    model.fit_generator(train_gen, epochs = 200, callbacks=[reduce_lr, early_stop, checkpoint], validation_data = val_gen, workers=15, use_multiprocessing=True, max_queue_size=15) # validation_data = val_gen LossHistory callback



train_model_bell_array()
#test_model_bell_array()
#train_regressor_bell_array()
