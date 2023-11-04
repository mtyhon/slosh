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

    def __init__(self, filenames, kic, labels, dim, scale=False, validate = False):
    
        self.filenames = filenames
        self.file_kic = kic
        self.dim=dim # image/2D array dimensions
        self.subfolder_labels = labels # for binary classification
        self.scale = scale
        self.indexes = np.arange(len(self.filenames))     
        self.validate = validate
        self.kepmag_data =  pd.read_csv('/home/z3384751/K2Detection/DR25_KIC_RADec.dat', delim_whitespace=True, header=0)
        self.numax_data = pd.read_csv('/home/z3384751/K2Detection/Table_1_Extend_Probabilities_V2.dat', header=0, delimiter='|',comment='#')

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
        batch_label = self.subfolder_labels[index]

        # Generate data

        if not self.validate:
            X = self.__data_generation(batch_filenames, batch_kic)
            return X.copy(), batch_label
        else:
            X, mag, hb, idx, nmx = self.__data_generation(batch_filenames, batch_kic)
            #print(batch_label, mag, hb, idx, nmx)
            return X.copy(), batch_label, mag, hb, idx, nmx

  
    def __data_generation(self, batch_filenames, batch_kic):
        try:
            data = np.load(batch_filenames)
        except:
            data = np.load(batch_filenames, allow_pickle=True)       
        freq = data['freq']
        power = data['pow']
 

        im = ps_to_array(freq, power, minfreq=3., scale=self.scale, fix=True)
        im = np.expand_dims(im, 0)
        if not self.validate:
            return im
        else:
            try:
                hb = data['hb'] # only simulated positives have HB
                return im, data['mag'].astype(float), data['hb'].astype(float), np.array([data['id'].astype(int)]), data['numax'].astype(float) 
            except:
                magx = self.kepmag_data['Kep_Mag'].values[np.where(self.kepmag_data['KIC'].values == batch_kic)[0]]
                if len(magx) == 0:
                    magx = [-99.0] 

                return im, np.array(magx).astype(float), np.array([-99.]).astype(float), np.array([batch_kic]), np.array([-99.]).astype(float)


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
    val_batches = 0

    for batch_idy, val_data in enumerate(val_dataloader, 0):  # indices,scaled_indices, numax, teff, fe_h, age, tams_age
        #if batch_idy > 32:
         #   break
        val_image = val_data[0].cuda().float()
        input_label = val_data[1].cuda().long()

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


    return (val_loss / val_batches), (val_batch_acc / val_batches), np.mean(np.array(val_pos_recall)), np.mean(np.array(val_pos_precision)), np.mean(np.array(val_neg_recall)), np.mean(np.array(val_neg_precision))



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

    pos_folder = '/data/marc/TESS/QLP/celerite_tess_sim_train/PSD/v1/'#npz_PSD_all/'#test_run_images/
    neg_folder = '/data/marc/KEPSEISMIC/LC_Data/PSD/npz_PSD_all/0/'#npz_PSD_all/'#test_run_images/


    id_vec = [] #KICs
    subfolder_labels = []
    folder_filenames = []
    file_kic_vec = []
    
    print('Parsing files in folder... ')
    for filex in tqdm(os.listdir(pos_folder)):
        if filex.endswith('.npz'):
            file_kic = int(re.search(r'\d+', filex).group())
            folder_filenames.append(os.path.join(pos_folder, filex))
            file_kic_vec.append(file_kic)
            subfolder_labels.append(1)

    for filex in tqdm(os.listdir(neg_folder)):
        if filex.endswith('.npz'):
            file_kic = int(re.search(r'\d+', filex).group())
            folder_filenames.append(os.path.join(neg_folder, filex))
            file_kic_vec.append(file_kic)
            subfolder_labels.append(0)

    print('Total File Number: ', len(folder_filenames))

    file_kic = np.array(file_kic_vec)
    subfolder_labels = np.array(subfolder_labels)
    folder_filenames = np.array(folder_filenames)

    unique_posdet_kic = file_kic[subfolder_labels == 1]
    print('Number of Unique PosDet KIC: ', len(unique_posdet_kic))

    unique_id, unique_indices = np.unique(file_kic, return_index=True)

    print('Unique Labels Nb NonDet: ', np.sum(subfolder_labels[unique_indices] == 0))
    print('Unique Labels Nb Det: ', np.sum(subfolder_labels[unique_indices] == 1))

    train_ids, test_ids,train_unique_labels, test_unique_labels = train_test_split(unique_id, subfolder_labels[unique_indices], test_size =0.15, random_state = 137, stratify = subfolder_labels[unique_indices])
    train_ids, val_ids = train_test_split(train_ids, test_size =0.1765, random_state = 137, stratify=train_unique_labels)

    train_kic = file_kic[np.in1d(file_kic, train_ids)]
    val_kic = file_kic[np.in1d(file_kic, val_ids)]
    test_kic = file_kic[np.in1d(file_kic, test_ids)]

    train_labels = subfolder_labels[np.in1d(file_kic, train_ids)]
    val_labels = subfolder_labels[np.in1d(file_kic, val_ids)]
    test_labels = subfolder_labels[np.in1d(file_kic, test_ids)]

    train_filenames = folder_filenames[np.in1d(file_kic, train_ids)]
    val_filenames = folder_filenames[np.in1d(file_kic, val_ids)]
    test_filenames = folder_filenames[np.in1d(file_kic, test_ids)]


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
    train_gen = NPZ_Dataset_OnTheFly(filenames = train_filenames,kic=train_kic ,labels=train_labels, dim=(128,128), scale=False)
    train_dataloader = utils.DataLoader(train_gen, shuffle=True, batch_size=32, num_workers=5)

    val_gen = NPZ_Dataset_OnTheFly(filenames = val_filenames,kic=val_kic,labels=val_labels, dim=(128,128), scale=False)
    val_dataloader = utils.DataLoader(val_gen, shuffle=True, batch_size=32, num_workers=5)

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

    
        model.train()  # set to training mode
    
        for i, data in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader), unit='batches'):
            #if i > 32:
            #    break
            train_batches += 1
    
            image = data[0].cuda().float()
            label = data[1].cuda().long()

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


        val_loss, val_acc, val_pos_recall, val_pos_precision, val_neg_recall, val_neg_precision = validate(model, val_dataloader, loss_function)
        scheduler.step(train_loss)  # reduce LR on loss plateau
    
        print('\n\nTrain Loss: ', train_loss)
        print('Train Acc: ', train_acc)
        print('Train Pos Precision: ', train_pos_precision)
        print('Train Pos Recall: ', train_pos_recall)
        print('Train Neg Precision: ', train_neg_precision)
        print('Train Neg Recall: ', train_neg_recall)
    
        print('Val Loss: ', val_loss)
        print('Val Acc: ', val_acc)
        print('Val Pos Precision: ', val_pos_precision)
        print('Val Pos Recall: ', val_pos_recall)
        print('Val Neg Precision: ', val_neg_precision)
        print('Val Neg Recall: ', val_neg_recall)
    

        for param_group in model_optimizer.param_groups:
            print('Current Learning Rate: ', param_group['lr'])
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        print('Current Best Metric: ', best_loss)

        if model_checkpoint:

            if is_best:
                filename = 'CELERITE_NOSCALED_Drop=0.1_Network1-Loss:%.2f-Acc:%.2f' % (
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
    model.eval()
    torch.backends.cudnn.benchmark=True
    print('CUDNN Enabled? ', torch.backends.cudnn.enabled)
    print(str(model))

    #saved_model = torch.load('/data/marc/K2Detection/ClassifyModels/CELERITE_SCALED_Drop=0.1-Loss:0.02-Acc:99.21')
    #saved_model = torch.load('/data/marc/K2Detection/ClassifyModels/STANDARDSCALED_SLOSCFilter_Drop=0.1_NoAugment-Loss:0.15-Acc:94.12')
    saved_model = torch.load('/data/marc/K2Detection/ClassifyModels/SLOSCFilter_Drop=0.1_NoAugment-Loss:0.09-Acc:98.09')
    #saved_model = torch.load('/data/marc/K2Detection/ClassifyModels/27d_FIXEDSCALE_SLOSCFilter_Drop=0.1_NoAugment-Loss:0.09-Acc:98.20')
    model.load_state_dict(saved_model)

    pos_folder = '/data/marc/TESS/QLP/celerite_tess_sim_train/PSD/v1/'#npz_PSD_all/'#test_run_images/
    neg_folder = '/data/marc/KEPSEISMIC/LC_Data/PSD/npz_PSD_all/0/'#npz_PSD_all/'#test_run_images/


    id_vec = [] #KICs
    subfolder_labels = []
    folder_filenames = []
    file_kic_vec = []
    
    print('Parsing files in folder... ')
    for filex in tqdm(os.listdir(pos_folder)):
        if filex.endswith('.npz'):
            file_kic = int(re.search(r'\d+', filex).group())
            folder_filenames.append(os.path.join(pos_folder, filex))
            file_kic_vec.append(file_kic)
            subfolder_labels.append(1)

    for filex in tqdm(os.listdir(neg_folder)):
        if filex.endswith('.npz'):
            file_kic = int(re.search(r'\d+', filex).group())
            folder_filenames.append(os.path.join(neg_folder, filex))
            file_kic_vec.append(file_kic)
            subfolder_labels.append(0)

    print('Total File Number: ', len(folder_filenames))

    file_kic = np.array(file_kic_vec)
    subfolder_labels = np.array(subfolder_labels)
    folder_filenames = np.array(folder_filenames)

    unique_posdet_kic = file_kic[subfolder_labels == 1]
    print('Number of Unique PosDet KIC: ', len(unique_posdet_kic))

    unique_id, unique_indices = np.unique(file_kic, return_index=True)

    print('Unique Labels Nb NonDet: ', np.sum(subfolder_labels[unique_indices] == 0))
    print('Unique Labels Nb Det: ', np.sum(subfolder_labels[unique_indices] == 1))

    train_ids, test_ids,train_unique_labels, test_unique_labels = train_test_split(unique_id, subfolder_labels[unique_indices], test_size =0.15, random_state = 137, stratify = subfolder_labels[unique_indices])
    train_ids, val_ids = train_test_split(train_ids, test_size =0.1765, random_state = 137, stratify=train_unique_labels)

    train_kic = file_kic[np.in1d(file_kic, train_ids)]
    val_kic = file_kic[np.in1d(file_kic, val_ids)]
    test_kic = file_kic[np.in1d(file_kic, test_ids)]

    train_labels = subfolder_labels[np.in1d(file_kic, train_ids)]
    val_labels = subfolder_labels[np.in1d(file_kic, val_ids)]
    test_labels = subfolder_labels[np.in1d(file_kic, test_ids)]

    train_filenames = folder_filenames[np.in1d(file_kic, train_ids)]
    val_filenames = folder_filenames[np.in1d(file_kic, val_ids)]
    test_filenames = folder_filenames[np.in1d(file_kic, test_ids)]


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

    print('Setting up generator... ')

    val_gen = NPZ_Dataset_OnTheFly(filenames = val_filenames,kic=val_kic,labels=val_labels, dim=(128,128), scale=False, validate=True)
    val_dataloader = utils.DataLoader(val_gen, shuffle=True, batch_size=32, num_workers=5)

    label_vec, pred_vec, prob_vec, mag_vec, hb_vec, id_vec, nmx_vec = [], [], [], [], [], [], []

    for batch_idy, val_data in tqdm(enumerate(val_dataloader, 0), total = len(val_dataloader)):
  
        val_image = val_data[0].cuda().float()
        input_label = val_data[1].data.cpu().numpy()
        val_mag = val_data[2].data.cpu().numpy()
        val_hb = val_data[3].data.cpu().numpy()
        val_idx = val_data[4].data.cpu().numpy()
        val_nmx = val_data[5].data.cpu().numpy()

        with torch.no_grad():
            outputs= model(val_image)
            class_pred = torch.max(outputs, dim=1)[1].data.cpu().numpy()
            class_prob = F.softmax(outputs, dim=1)[:,1].data.cpu().numpy()
        label_vec.append(deepcopy(input_label))
        prob_vec.append(deepcopy(class_prob))
        pred_vec.append(deepcopy(class_pred))
        mag_vec.append(deepcopy(val_mag))
        hb_vec.append(deepcopy(val_hb))
        id_vec.append(deepcopy(val_idx))
        nmx_vec.append(deepcopy(val_nmx))

    label_vec = np.concatenate(label_vec)
    pred_vec = np.concatenate(pred_vec)
    prob_vec = np.concatenate(prob_vec)
    mag_vec = np.concatenate(mag_vec)
    hb_vec = np.concatenate(hb_vec)
    id_vec = np.concatenate(id_vec)
    nmx_vec = np.concatenate(nmx_vec)
        
    np.savez_compressed('OLD_SLOSC_FIXEDSCALED2_CLASSIFIER_NETWORK1_VALDATA', id=id_vec, label=label_vec, prob=prob_vec, pred=pred_vec, mag=mag_vec, hb=hb_vec, numax=nmx_vec)



#train_model_bell_array()
test_model_bell_array()
