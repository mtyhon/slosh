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
from triplet_utils import AllTripletSelector,HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector # Strategies for selecting triplets within a minibatch
from torch.utils.data.sampler import BatchSampler



import math
import warnings
warnings.filterwarnings("ignore")

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
classes = ['NON-DET', 'DET']
def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    for i in range(10):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(classes)
    plt.show()
    plt.close()

def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        filenames = []
        k = 0
        for images, target, filename in dataloader:
            images = images.cuda().float()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
            filenames.append(filename)
    return embeddings, labels, np.concatenate(filenames, axis=0)

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """
    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - Samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """
    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == int(label))[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


class NPZ_Dataset(data.Dataset):

    def __init__(self, kic, filenames, mode='classification', return_kic=False, labels=None):
    
        self.filenames = filenames
        self.file_kic = kic
        self.indexes = np.arange(len(self.filenames))
        self.mode = mode
        self.return_kic = return_kic
        self.labels =labels
  
        if mode not in ['classification', 'regression', 'prediction', 'metric']:
            raise ValueError     

        assert len(self.indexes) == len(self.file_kic) == len(self.filenames)

    def __len__(self):
        'Total number of samples'
        return len(self.indexes)

    def __getitem__(self, index):
        'Generates ONE sample of data'

        batch_filenames = self.filenames[index]
        batch_kic = self.file_kic[index]
        batch_labels = self.labels[index]

        # Generate data
        X, y, y_sigma = self.__data_generation(batch_filenames, batch_labels)
        if np.isnan(y):
            print('Nan Numax %s with sigma %s for star %d: ' %(y, y_sigma, batch_kic))
        #print('y: ', y)
        #print('y_sigma: ', y_sigma)
        if self.return_kic:
            return X.copy(), y, y_sigma, batch_kic
        else:
            return X.copy(), y, y_sigma

    def __data_generation(self, batch_filenames, batch_labels):
        data = np.load(batch_filenames)
        im = data['im']

        if self.mode == 'classification':
            y = data['det']  
            y_sigma = batch_filenames
        elif self.mode == 'metric':
            y = batch_labels 
            y_sigma = batch_filenames
        elif self.mode == 'regression':      
            y = data['numax']
            y_sigma = data['numax_sigma']
        else:
            y = 0
            y_sigma = 0
        return im, y, y_sigma


class SLOSH_Classifier(nn.Module):
    def __init__(self):
        super(SLOSH_Classifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, padding=2) # same padding 2P = K-1
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1) # same padding 2P = K-1
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1) # same padding 2P = K-1
        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2)
        self.pool3 = nn.MaxPool2d(2,2)
        self.drop1 = nn.Dropout(0.5)
        self.linear1 = nn.Linear(16*16*16, 128)
        self.linear2 = nn.Linear(128, 2)

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
        linear2 = self.linear2(linear1)
        return linear2

    def get_embedding(self, x):
        return self.forward(x)


class SLOSH_Embedding(nn.Module):
    def __init__(self):
        super(SLOSH_Embedding, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, padding=2)  # same padding 2P = K-1
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)  # same padding 2P = K-1
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # same padding 2P = K-1
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.5)
        self.linear1 = nn.Linear(16*16*16, 128)
        self.linear2 = nn.Linear(128, 2)

    def forward(self, input_image):
        # input_image = self.drop0(input_image)
        conv1 = self.conv1(input_image.unsqueeze(1)) # (N, C, H, W)
        conv1 = F.leaky_relu(conv1)
        conv1 = self.pool1(conv1)
        conv2 = F.leaky_relu(self.conv2(conv1))
        conv2 = self.pool2(conv2)
        conv3 = F.leaky_relu(self.conv3(conv2))
        conv3 = self.pool3(conv3)
        conv3 = self.drop1(conv3)

        linear1 = F.relu(self.linear1(conv3.view(conv3.size()[0], -1)))
        linear2 = self.linear2(linear1)
        return linear2

    def get_embedding(self, x):
        return self.forward(x)

class SLOSH_Large_Embedding(nn.Module):
    def __init__(self):
        super(SLOSH_Large_Embedding, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, padding=2)  # same padding 2P = K-1
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)  # same padding 2P = K-1
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # same padding 2P = K-1
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.5)
        self.linear1 = nn.Linear(16*16*16, 128)


    def forward(self, input_image):
        # input_image = self.drop0(input_image)
        conv1 = self.conv1(input_image.unsqueeze(1)) # (N, C, H, W)
        conv1 = F.leaky_relu(conv1)
        conv1 = self.pool1(conv1)
        conv2 = F.leaky_relu(self.conv2(conv1))
        conv2 = self.pool2(conv2)
        conv3 = F.leaky_relu(self.conv3(conv2))
        conv3 = self.pool3(conv3)
        conv3 = self.drop1(conv3)

        linear1 = self.linear1(conv3.view(conv3.size()[0], -1))
        return linear1

    def get_embedding(self, x):
        return self.forward(x)

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
            nn.Softplus(threshold=0.5)  
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

def weighted_mdn_loss(pi, sigma, mu, target):
    """Calculates the error, given the MoG parameters and the target
    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    prob = pi * gaussian_probability(sigma, mu, target)
    nll = -torch.log(torch.sum(prob, dim=1)+1e-6)*(torch.abs(target-100))**2
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

def dist_var(pi, mu, mixture_mu, sigma):
    """Calculate the second moment (variance) of a bimodal distribution
    mu is the tensor while mixture_mu is the mean of the entire mixture as calculated by dist_mu
    """
    if pi.size() != mu.size():
        pi = pi.unsqueeze(2)
    if mixture_mu.size() != mu.size():
        mixture_mu = mixture_mu.unsqueeze(-1)
    delta_square =torch.mul(mu-mixture_mu, mu-mixture_mu)
    summation = torch.mul(sigma, sigma) + delta_square
    return torch.sum(pi*summation, dim=1)

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


class SLOSH_Regressor(nn.Module):
    def __init__(self, num_gaussians):
        super(SLOSH_Regressor, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, padding=2) # same padding 2P = K-1
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1) # same padding 2P = K-1
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1) # same padding 2P = K-1
        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2)
        self.pool3 = nn.MaxPool2d(2,2)
        self.drop1 = nn.Dropout(0.5)
        self.linear1 = nn.Linear(16*16*16, 128)
        self.mdn = MDN(in_features=128, out_features=1, num_gaussians=num_gaussians)

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
        return pi, sigma, mu

class SLOSH_Regressor_Vanilla(nn.Module):
    def __init__(self, num_gaussians):
        super(SLOSH_Regressor_Vanilla, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, padding=2) # same padding 2P = K-1
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1) # same padding 2P = K-1
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1) # same padding 2P = K-1
        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2)
        self.pool3 = nn.MaxPool2d(2,2)
        self.drop1 = nn.Dropout(0.5)
        self.linear1 = nn.Linear(16*16*16, 128)
        self.output = nn.Linear(128,1)

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
        output = self.output(linear1)
        return output


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

def weighted_mean_squared_error(y_true, y_pred):
    return torch.mean(((y_pred - y_true)**2)*((y_true-50)**2), dim=0)


def train_classifier(model, model_optimizer, input_image, input_label, loss_function):
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


def train_regressor(model, model_optimizer, input_image, input_numax, input_numax_sigma):

    model_optimizer.zero_grad()

    # Combined forward pass
    #print('Input Numax: ', input_numax)
    #print('Input Numax Sigma: ', input_numax_sigma)

    pi, sigma, mu = model(input_image = input_image.float())
    #pred = model(input_image = input_image.float())
    
    # Calculate loss and backpropagate

    #loss = mdn_loss(pi, sigma, mu, target=input_numax.float()) #log-likelihood optimization
    #loss = weighted_mdn_loss(pi, sigma, mu, target=input_numax.float()) #log-likelihood optimization
    #loss = weighted_mean_squared_error(y_true=input_numax.float(), y_pred=pred) # weighted_mse
    loss = compute_full_crps(pi, mu, sigma, label=input_numax.float(), label_err=input_numax_sigma.float(), loss=True) # CRPS optim

    loss.backward()
    
    pred_mean = dist_mu(pi, mu).data.cpu().numpy().reshape(-1,1) #pred.data.cpu().numpy()
    truth_npy = input_numax.data.cpu().numpy()

    mape = mean_absolute_percentage_error(y_true=truth_npy.squeeze(), y_pred=pred_mean.squeeze())
    mae = mean_absolute_error(y_true=truth_npy.squeeze(),y_pred=pred_mean.squeeze())

    
    #Clipnorm?
    #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # Update parameters
    model_optimizer.step()

    return loss.item(), mape, mae


def validate_classifier(model, val_dataloader, loss_function):
    model.eval()  # set to evaluate mode
    val_loss = 0
    val_batch_acc = 0
    val_pos_recall = []
    val_pos_precision = []
    val_neg_recall = []
    val_neg_precision = []
    val_batches = 0

    for batch_idy, val_data in enumerate(val_dataloader, 0):  # indices,scaled_indices, numax, teff, fe_h, age, tams_age

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
            val_pos_recall.append(pos_recall)
        if pos_precision != 0:
            val_pos_precision.append(pos_precision)
        if neg_recall != 0:
            val_neg_recall.append(neg_recall)
        if neg_precision != 0:
            val_neg_precision.append(neg_precision)    


    return (val_loss / val_batches), (val_batch_acc / val_batches), np.mean(np.array(val_pos_recall)), np.mean(np.array(val_pos_precision)), np.mean(np.array(val_neg_recall)), np.mean(np.array(val_neg_precision))


def validate_regressor(model, val_dataloader):
    model.eval() # set to evaluate mode
    val_loss = 0
    val_cum_mape = 0
    val_cum_mae = 0   
    val_batches = 0
    val_num = 0
    val_flag_array = []
    for j, val_data in enumerate(val_dataloader, 0): #indices,scaled_indices, numax, teff, fe_h, age, tams_age

        val_image = val_data[0].cuda()
        val_numax = val_data[1].cuda()
        val_numax_sigma = val_data[2].cuda()
        val_num += val_numax.size()[0]
        if len(val_image.size()) <= 1:
            print('Insufficient Batch!')
            continue
        with torch.no_grad():
            pi, sigma, mu = model(input_image=torch.squeeze(val_image, -1).float())
            #pred = model(input_image=torch.squeeze(val_image, -1).float())
        
            #val_batch_loss = mdn_loss(pi, sigma, mu, target=val_numax.float()) # log-likelihood loss
            #val_batch_loss = weighted_mdn_loss(pi, sigma, mu, target=val_numax.float()) # log-likelihood loss
            #val_batch_loss = weighted_mean_squared_error(y_true=val_numax.float(), y_pred=pred) # weighted_mse
            val_batch_loss = compute_full_crps(pi, mu, sigma, label=val_numax.float(), label_err=val_numax_sigma.float(), loss=True) # CRPS loss

            val_pred_npy = dist_mu(pi, mu).data.cpu().numpy().reshape(-1,1) #pred.data.cpu().numpy()
            val_truth_npy = val_numax.data.cpu().numpy()

            val_mape = mean_absolute_percentage_error(y_true=val_truth_npy.squeeze(),y_pred=val_pred_npy.squeeze())
            val_mae = mean_absolute_error(y_true=val_truth_npy.squeeze(),y_pred=val_pred_npy.squeeze())
       

        val_loss += val_batch_loss.item()
        val_cum_mape += val_mape
        val_cum_mae += val_mae
        val_batches += 1

    print('Sample Numax: ', val_numax[-1])
    print('Val Num: ', val_num)
    print('Sample Truth: ', val_truth_npy[-1])
    print('Sample Pred: ', val_pred_npy[-1])
    #print('Sample Age Pi: ', pi[-1])
    #print('Sample Sigma: ', sigma[-1])
    #rint('Sample Mu: ', mu[-1])


    return (val_loss/val_batches), (val_cum_mape/val_batches), (val_cum_mae/val_batches)


def classification():

    model = SLOSH_Classifier()
    model.cuda()
    torch.backends.cudnn.benchmark=True
    print('CUDNN Enabled? ', torch.backends.cudnn.enabled)
    print(str(model))
    initialization(model)

    loss_function = nn.CrossEntropyLoss(weight=torch.Tensor([1., 10.]).float().cuda())

    learning_rate = 0.001
    model_optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = ReduceLROnPlateau(model_optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr = 1E-6)

    root_folder = '/data/marc/KEPSEISMIC/Bell_Arrays_4year_Superluminous/Classification/'#npz_PSD_all/'#test_run_images/
    file_count = 0


    subfolder_labels = []
    folder_filenames = []
    file_kic = []

    print('Parsing files in folder... ')
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for i, filex in enumerate(filenames): # Getting the mags, KICS, and numax sigma for all stars in catalog
            if filex.endswith('.npz') & dirpath[-1].isdigit(): # I infer the class label '0' or '1' according to subfolder names
                folder_filenames.append(os.path.join(dirpath, filex))
                kicx = int(re.search(r'\d+', filex).group())
                subfolder_labels.append(int(dirpath[-1]))
                file_kic.append(kicx)

    file_kic = np.array(file_kic)
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
    train_gen = NPZ_Dataset(kic=train_kic, filenames=train_filenames, mode='classification')
    train_dataloader = utils.DataLoader(train_gen, shuffle=True, batch_size=32, num_workers=10)

    val_gen = NPZ_Dataset(kic=val_kic, filenames=val_filenames, mode='classification')
    val_dataloader = utils.DataLoader(val_gen, shuffle=True, batch_size=32, num_workers=10)

    n_epochs=100
    model_checkpoint = True
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
            train_batches += 1
    
            image = data[0].cuda().float()
            label = data[1].cuda().long()
 
            loss, acc, pos_recall, pos_precision, neg_recall, neg_precision = train_classifier(model, model_optimizer, image, label, loss_function)
            train_loss += loss  # Summing losses across all batches, so if you want the mean for EACH sample, divide by number of batches
            train_acc += acc
            if pos_recall != 0:
                train_pos_recall.append(pos_recall)
            if pos_precision != 0:
                train_pos_precision.append(pos_precision)
            if neg_recall != 0:
                train_neg_recall.append(neg_recall)
            if neg_precision != 0:
                train_neg_precision.append(neg_precision)

    
        train_loss = train_loss / train_batches
        train_acc = train_acc / train_batches
        train_pos_precision = np.mean(np.array(train_pos_precision))
        train_pos_recall = np.mean(np.array(train_pos_recall))
        train_neg_precision = np.mean(np.array(train_neg_precision))
        train_neg_recall = np.mean(np.array(train_neg_recall))

        val_loss, val_acc, val_pos_recall, val_pos_precision, val_neg_recall, val_neg_precision = validate_classifier(model, val_dataloader, loss_function)
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
                filename = 'SLOSH_4yr_Superluminous_Classifier-Loss:%.2f-Acc:%.2f.torchmodel' % (
                val_loss, val_acc)
                filepath = '/home/z3384751/K2Detection/ClassifyModels/2D_classifier/SLOSH_Superluminous_4yr/Classification/'
                try:
                    torch.save(model.state_dict(), os.path.join(filepath, filename))
                    print('Model saved to %s' %os.path.join(filepath, filename))     
                except:
                    pass
            else:
                print('No improvement over the best of %.4f' % best_loss)



def regression():

    model = SLOSH_Regressor(num_gaussians = 4)
    model.cuda()
    torch.backends.cudnn.benchmark=True
    print('CUDNN Enabled? ', torch.backends.cudnn.enabled)
    print(str(model))
    initialization(model)

    learning_rate = 0.001
    model_optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = ReduceLROnPlateau(model_optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr = 1E-6)

    root_folder = '/data/marc/KEPSEISMIC/Bell_Arrays_4year_Superluminous/Regression_LLRGB_Added/'#npz_PSD_all/'#test_run_images/
    file_count = 0

    folder_filenames = []
    file_kic = []

    print('Parsing files in folder... ')
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for i, filex in enumerate(filenames): # Getting the mags, KICS, and numax sigma for all stars in catalog
            if filex.endswith('.npz'): # I infer the class label '0' or '1' according to subfolder names
                folder_filenames.append(os.path.join(dirpath, filex))
                kicx = int(re.search(r'\d+', filex).group())
                file_kic.append(kicx)

    file_kic = np.array(file_kic)
    folder_filenames = np.array(folder_filenames)
    unique_id = np.unique(file_kic)
    #train_ids, test_ids = train_test_split(unique_id, test_size =0.15, random_state = 137)
    train_ids, val_ids = train_test_split(unique_id, test_size =0.15, random_state = 137)

    train_kic = file_kic[np.in1d(file_kic, train_ids)]
    val_kic = file_kic[np.in1d(file_kic, val_ids)] 
    #test_kic = file_kic[np.in1d(file_kic, test_ids)]

    train_filenames = folder_filenames[np.in1d(file_kic, train_ids)]
    val_filenames = folder_filenames[np.in1d(file_kic, val_ids)]
    #test_filenames = folder_filenames[np.in1d(file_kic, test_ids)]

    train_gen = NPZ_Dataset(kic=train_kic, filenames=train_filenames, mode='regression')
    train_dataloader = utils.DataLoader(train_gen, shuffle=True, batch_size=32, num_workers=10)

    val_gen = NPZ_Dataset(kic=val_kic, filenames=val_filenames, mode='regression')
    val_dataloader = utils.DataLoader(val_gen, shuffle=True, batch_size=32, num_workers=10)

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
        num_stars = 0 

        model.train()  # set to training mode
    
        for i, data in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader), unit='batches'):
            train_batches += 1
    
            image = data[0].cuda()
            numax = data[1].cuda()
            numax_sigma = data[2].cuda()
            if torch.sum(torch.isnan(numax)) > 0:
                print('Numax Nan!')        
                raise ValueError
            elif torch.sum(torch.isnan(numax_sigma)) > 0:
                print('Numax Sigma Nan!')
                raise ValueError

            num_stars += numax.size()[0]
            if len(image.size()) <= 1:
                print('Insufficient Batch!')
                continue
    
            loss, mape, mae = train_regressor(model, model_optimizer, image, numax, numax_sigma)
            train_loss += loss  # Summing losses across all batches, so if you want the mean for EACH sample, divide by number of batches
            mape_cum += mape
            mae_cum += mae

    
        train_loss = train_loss / train_batches
        train_mape = mape_cum / train_batches
        train_mae = mae_cum / train_batches

        val_loss, val_mape, val_mae = validate_regressor(model, val_dataloader)
        scheduler.step(train_loss)  # reduce LR on loss plateau

        print('Num Stars: ', num_stars)
        print('\n\nTrain Loss: ', train_loss)
        print('Train Mape: ', train_mape)
        print('Train Mae: ', train_mae)
    
        print('Val Loss: ', val_loss)
        print('Val Mape: ', val_mape)
        print('Val Mae: ', val_mae)

        model.print_instance_name()
        print('Using CRPS, WITH Dropout, FIXED 50% noise?')
        for param_group in model_optimizer.param_groups:
            print('Current Learning Rate: ', param_group['lr'])
        is_best = val_mape < best_loss
        best_loss = min(val_mape, best_loss)
        print('Current Best Metric: ', best_loss)

        if model_checkpoint:

            if is_best:
                filename = 'SLOSH_4yr_LLRGB-ADDED_MDN_Superluminous_Regression-MAPE:%.2f-MAE:%.2f' % (
                val_mape, val_mae)
                filepath = '/home/z3384751/K2Detection/ClassifyModels/2D_classifier/SLOSH_Superluminous_4yr/'

                try:
                    torch.save(model.state_dict(), os.path.join(filepath, filename))
                    print('Model saved to %s' %os.path.join(filepath, filename))
                except:
                    pass
  
            else:
                print('No improvement over the best of %.4f' % best_loss)


def full_prediction():
   
    root_folder = '/home/z3384751/K2Detection/ClassifyModels/2D_classifier/SLOSH_Superluminous_4yr/New_Images+Superluminous/'#npz_PSD_all/'#test_run_images/
    file_count = 0
    saved_classifier = '/home/z3384751/K2Detection/ClassifyModels/2D_classifier/SLOSH_Superluminous_4yr/Classification/SLOSH_4yr_Superluminous_Classifier-Loss:0.04-Acc:98.66.torchmodel'
    classifier = SLOSH_Classifier()
    classifier.load_state_dict(torch.load(saved_classifier))
    classifier.cuda()
    saved_regressor = '/home/z3384751/K2Detection/ClassifyModels/2D_classifier/SLOSH_Superluminous_4yr/Regression/SLOSH_4yr_LLRGB-ADDED_MDN_Superluminous_Regression-MAPE:3.88-MAE:2.14'
    regressor = SLOSH_Regressor(num_gaussians = 4)
    regressor.load_state_dict(torch.load(saved_regressor))
    regressor.cuda()
    regressor.eval()
    torch.backends.cudnn.benchmark=True


    ######## EVALUATING TRAINING/VALIDATION SET ########
    root_folder = '/data/marc/KEPSEISMIC/Bell_Arrays_4year_Superluminous/Classification/'
    file_count = 0
    subfolder_labels = []
    folder_filenames = []
    file_kic = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for i, filex in enumerate(filenames): # Getting the mags, KICS, and numax sigma for all stars in catalog
            if filex.endswith('.npz') & dirpath[-1].isdigit(): # I infer the class label '0' or '1' according to subfolder names
                folder_filenames.append(os.path.join(dirpath, filex))
                kicx = int(re.search(r'\d+', filex).group())
                subfolder_labels.append(int(dirpath[-1]))
                file_kic.append(kicx)
    file_kic = np.array(file_kic)
    subfolder_labels = np.array(subfolder_labels)
    folder_filenames = np.array(folder_filenames)
    unique_posdet_kic = file_kic[subfolder_labels == 1]
    unique_id, unique_indices = np.unique(file_kic, return_index=True)
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

    train_gen = NPZ_Dataset(kic=train_kic, filenames=train_filenames, mode='prediction')
    train_dataloader = utils.DataLoader(train_gen, shuffle=True, batch_size=32, num_workers=10)

    val_gen = NPZ_Dataset(kic=val_kic, filenames=val_filenames, mode='prediction')
    val_dataloader = utils.DataLoader(val_gen, shuffle=True, batch_size=32, num_workers=10)
    ######## ######################## ########


    '''######## OR EVALUATE ON REAL DATA ########
    folder_filenames = []
    file_kic = []
    print('Parsing files in folder... ')
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for i, filex in enumerate(filenames):
            if filex.endswith('.npz'): 
                folder_filenames.append(os.path.join(dirpath, filex))
                kicx = int(re.search(r'\d+', filex).group())
                file_kic.append(kicx)

    file_kic = np.array(file_kic)
    folder_filenames = np.array(folder_filenames)
    data_gen = NPZ_Dataset(kic=file_kic, filenames=folder_filenames, mode='prediction', return_kic=True)
    ######## ######################## ########'''


    dataloader = utils.DataLoader(train_gen, shuffle=False, batch_size=32, num_workers=10)
    
    classifier_pred_array = []
    classifier_std_array = []
    regressor_pred_array = []
    regressor_std_array = []
    kic_array = []
    print('Running Dataloader...')
    for i, data in tqdm(enumerate(dataloader, 0), total=len(dataloader), unit='batches'):
        image = data[0].cuda().float()
        kic = data[-1]
        kic = kic.data.cpu().numpy()
        mc_iterations = 10

        with torch.no_grad():
            pred_grid = np.empty((mc_iterations, image.size()[0], 2))
            for i in range(mc_iterations):
                pred_grid[i, :] = F.softmax(classifier(image), dim=1).data.cpu().numpy()
            pred_mean = np.mean(pred_grid, axis=0)
            epistemic = np.mean(pred_grid ** 2, axis=0) - np.mean(pred_grid, axis=0) ** 2
            aleatoric = np.mean(pred_grid * (1 - pred_grid), axis=0)

            #print(epistemic.shape)
            #print(pred_mean.shape)
            #print(pred_mean[:,1])
            #print(np.sqrt(epistemic[:,1] + aleatoric[:,1]))
            #print(F.softmax(classifier(image), dim=1).data.cpu().numpy())
            #print(classifier(image).data.cpu().numpy())
            #print('---')

            classifier_pred_array.append(pred_mean[:,1])
            classifier_std_array.append(np.sqrt(epistemic[:,1] + aleatoric[:,1]))
            kic_array.append(kic)
            pi, sigma, mu = regressor(input_image=image.float())
            #pred = regressor(input_image=image.float())
            
            numax_pred_mean = dist_mu(pi, mu[:,:,0]).data.cpu().numpy()#pred.squeeze()
            numax_pred_var = dist_var(pi=pi, mu=mu[:,:,0], mixture_mu = dist_mu(pi, mu[:,:,0]), sigma=sigma[:,:,0]).data.cpu().numpy()#1
            numax_pred_sigma = np.sqrt(numax_pred_var)
            regressor_pred_array.append(numax_pred_mean)
            #print('Classifier Pred: ', np.mean(classifier_pred_mc_array, axis=1))
            #print('Classifier Std: ', np.std(classifier_pred_mc_array, axis=1))
            #print('Numax Pred Mean: ', numax_pred_mean)
            #print('Numax Pred Sigma: ', numax_pred_sigma)
            regressor_std_array.append(numax_pred_sigma)
    
    classifier_pred_array = np.concatenate(classifier_pred_array, axis=0)
    classifier_std_array = np.concatenate(classifier_std_array, axis=0)
    regressor_pred_array = np.concatenate(regressor_pred_array, axis=0)
    regressor_std_array = np.concatenate(regressor_std_array, axis=0)# np.zeros(len(regressor_pred_array)) # 
    label_array = []
    kic_array = np.concatenate(kic_array, axis=0)
    #print('KIC Array: ', kic_array)
    for i in range(len(kic_array)):
        if classifier_pred_array[i] >= 0.5:
            label_array.append(1)
        else:
            label_array.append(0)
    label_array = np.array(label_array)
    with open('Superluminous_Pred_Training_Set_LLRGB_MDN.dat', 'a') as writer:
        writer.write('KIC Pred Pred_Conf Label Numax Numax_Std\n')
        for i in range(len(label_array)):
            writer.write(str(int(kic_array[i])) + ' ' + str(np.round(classifier_pred_array[i], 3)) + ' ' + str(np.round(classifier_std_array[i], 3)) + ' ' +str(int(label_array[i])) + ' ' + str(np.round(regressor_pred_array[i], 2)) + ' ' + str(np.round(regressor_std_array[i], 2)) + '\n')


def online_metric_learning(transfer):

    if transfer:
        #### Transfer Learning ####
        saved_classifier = '/home/z3384751/K2Detection/ClassifyModels/2D_classifier/SLOSH_Superluminous_4yr/Classification/SLOSH_4yr_Superluminous_Classifier-Loss:0.04-Acc:98.66.torchmodel'
        embedding_net = SLOSH_Classifier()
        embedding_net.load_state_dict(torch.load(saved_classifier))
        model = embedding_net
        model.conv1.weight.requires_grad = False
        model.conv1.bias.requires_grad = False
        model.conv2.weight.requires_grad = False
        model.conv2.bias.requires_grad = False
        model.conv3.weight.requires_grad = False
        model.conv3.bias.requires_grad = False
        model.cuda()


    else:
        #### Learn from scratch ####
        embedding_net = SLOSH_Embedding()
        model = embedding_net
        model.cuda()
        initialization(model)
    torch.backends.cudnn.benchmark = True
    print('CUDNN Enabled? ', torch.backends.cudnn.enabled)
    print(str(model))

    margin = 1.

    loss_function = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))

    learning_rate = 0.001
    model_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = ReduceLROnPlateau(model_optimizer, mode='min', factor=0.5, patience=10, verbose=True,
                                  min_lr=1E-6)

    iic_data = np.load('/data/marc/AFFINE_CROPOTHER_10CLASSES_Best_Clustering_KeplerSuperluminous_ACC:0.36_Epoch-49_Loss:-1.65.npz')
    iic_pred = iic_data['pred']
    iic_unique_pred = np.unique(iic_pred)
    print('IIC Unique Pred: ', iic_unique_pred)

    iic_filename = iic_data['filename']
    iic_id = np.array([iic_filename[d].split('/')[-1].split('.')[0].split('_')[0] for d in range(len(iic_filename))]).astype(int)
    print('IIC ID: ', iic_id)
    root_folder = '/data/marc/KEPSEISMIC/Bell_Arrays_4year_Superluminous/Classification/'#npz_PSD_all/'#test_run_images/
    file_count = 0

    subfolder_labels = []
    folder_filenames = []
    file_kic = []

    print('Parsing files in folder... ')
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for i, filex in enumerate(filenames): # Getting the mags, KICS, and numax sigma for all stars in catalog
            if filex.endswith('.npz') & dirpath[-1].isdigit(): # I infer the class label '0' or '1' according to subfolder names
                folder_filenames.append(os.path.join(dirpath, filex))
                kicx = int(re.search(r'\d+', filex).group())
                #subfolder_labels.append(int(dirpath[-1]))
                iic_label = iic_pred[np.where(iic_id == kicx)[0]]
                subfolder_labels.append(np.where(iic_unique_pred == iic_label)[0][0])
                file_kic.append(kicx)

    file_kic = np.array(file_kic)
    subfolder_labels = np.array(subfolder_labels)
    folder_filenames = np.array(folder_filenames)
    assert len(file_kic) == len(subfolder_labels)
    print('Unique Subfolder Labels: ', np.unique(subfolder_labels))
    print('Nb Classes: ', len(np.unique(subfolder_labels)))
    print('File KIC: ', (file_kic))
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
    train_gen = NPZ_Dataset(kic=train_kic, filenames=train_filenames, mode='metric', labels=train_labels)
    train_dataloader = utils.DataLoader(train_gen, shuffle=True, batch_size=32, num_workers=10)
    train_batch_sampler = BalancedBatchSampler(train_labels, n_classes=len(np.unique(subfolder_labels)), n_samples=50)
    train_dataloader_online = utils.DataLoader(train_gen, num_workers=10, batch_sampler=train_batch_sampler)

    val_gen = NPZ_Dataset(kic=val_kic, filenames=val_filenames, mode='metric', labels=val_labels)
    val_dataloader = utils.DataLoader(val_gen, shuffle=True, batch_size=32, num_workers=10)
    val_batch_sampler = BalancedBatchSampler(val_labels, n_classes=len(np.unique(subfolder_labels)), n_samples=50)
    val_dataloader_online = utils.DataLoader(val_gen, num_workers=10, batch_sampler=val_batch_sampler)
   

    train_loader = train_dataloader_online
    val_loader = val_dataloader_online

    n_epochs = 500
    best_loss = 9999
    model_checkpoint=False
    for epoch in range(1, n_epochs + 1):
        print('---------------------')
        print('Epoch: ', epoch)
        total_loss = 0
        train_batches = 0

        model.train()  # set to training mode
        losses = []
        for i, (data, target, _) in tqdm(enumerate(train_loader, 0), total=len(train_loader), unit='batches'):
            train_batches += 1
            data = data.float().cuda()
            target = target.long().cuda()
            # data = tuple(d.float().cuda() for d in data)
            # target = tuple(t.long().cuda() for t in target) if len(target) > 0 else None

            model_optimizer.zero_grad()
            # Combined forward pass
            outputs = model(data) # unsqueeze here is you rebuild as a sequential
            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            # Calculate loss and backpropagate

            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_function(*loss_inputs)

            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

            losses.append(loss.item())
            total_loss += loss.item()
            loss.backward()
            model_optimizer.step()

        train_loss = total_loss / train_batches

        val_loss = 0
        val_batches = 0
        model.eval()
        with torch.no_grad():
            for i, (data, target, _) in tqdm(enumerate(val_loader, 0), total=len(val_loader), unit='batches'):
                val_batches += 1
                data = data.float().cuda()
                target = target.long().cuda()
                # target = tuple(t.long().cuda() for t in target) if len(target) > 0 else None
                # if not type(data) in (tuple, list):
                #     data = (data,)
                # data = tuple(d.float().cuda() for d in data)

                outputs = model(data)

                if type(outputs) not in (tuple, list):
                    outputs = (outputs,)
                loss_inputs = outputs
                if target is not None:
                    target = (target,)
                    loss_inputs += target

                loss_outputs = loss_function(*loss_inputs)
                loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
                val_loss += loss.item()
            val_loss = val_loss / val_batches

        print('\n\nTrain Loss: ', train_loss)
        print('Val Loss: ', val_loss)
        scheduler.step(train_loss)  # reduce LR on loss plateau

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        print('Current Best Metric: ', best_loss)

        if model_checkpoint:
            if is_best:
                filename = '/home/z3384751/K2Detection/Superluminous_SLOSH_Metric_Learning/SLOSH_Large_Embedding_128_Loss:%.2f' % (val_loss)
                torch.save(model.state_dict(), filename)
                print('Model saved to %s' %filename)     
     
            else:
                print('No improvement over the best of %.4f' % best_loss)

        if epoch % 50 == 0:
            train_embeddings_baseline, train_labels_baseline, train_embed_filenames = extract_embeddings(
                train_dataloader, model)
            print('Does the Conv layer requires a gradient? ', model.conv1.weight.requires_grad) # this indexing for non-sequential
            np.savez_compressed('IIC_Superluminous_Embed2_OnlineTripletTrain', embedding=train_embeddings_baseline, label=train_labels_baseline, filename=train_embed_filenames)
            val_embeddings_baseline, val_labels_baseline, val_embed_filenames = extract_embeddings(
                val_dataloader, model)
            np.savez_compressed('IIC_Superluminous_Embed2_OnlineTripletVal', embedding=val_embeddings_baseline, label=val_labels_baseline, filename=val_embed_filenames)

            plot_embeddings(train_embeddings_baseline, train_labels_baseline)
            plot_embeddings(val_embeddings_baseline, val_labels_baseline)



def npz_to_image():
    src_folder = '/data/marc/KEPSEISMIC/Bell_Arrays_4year_Superluminous/Classification/'
    image_folder = '/data/marc/KEPSEISMIC/Bell_Arrays_4year_Superluminous/Classification_Images/'
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    for dirpath, dirnames, filenames in os.walk(src_folder):
        for i, filex in enumerate(filenames): # Getting the mags, KICS, and numax sigma for all stars in catalog
            if filex.endswith('.npz') & dirpath[-1].isdigit(): # I infer the class label '0' or '1' according to subfolder names
                base_filename = os.path.splitext(os.path.basename(filex))[0]
                data = np.load(os.path.join(dirpath, filex))
                fig = Figure(figsize=(256 / 85, 256 / 85), dpi=96)
                canvas = FigureCanvas(fig)
                ax = fig.gca()
                ax.imshow(data['im'], cmap='gray')
                ax.axis('off')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                canvas.draw()  # draw the canvas, cache the renderer
                canvas.print_figure(os.path.join(image_folder,base_filename)+ '.png', bbox_inches='tight', pad_inches=0, facecolor='black')
                # plt.savefig(os.path.join(image_folder,base_filename)+ '.png')
                #plt.show()
   
#classification()
#regression()
#full_prediction()
online_metric_learning(transfer=False)
#npz_to_image()
