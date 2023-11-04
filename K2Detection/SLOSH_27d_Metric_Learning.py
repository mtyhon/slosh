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

print('Sci-kit Version {}.'.format(sklearn.__version__))


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

    def __init__(self, kic, filenames,labels, mode='classification', return_kic=False):
    
        self.filenames = filenames
        self.file_kic = kic
        self.indexes = np.arange(len(self.filenames))
        self.mode = mode
        self.labels=labels
        self.return_kic = return_kic
  
        if mode not in ['classification', 'regression', 'prediction']:
            raise ValueError     

        assert len(self.indexes) == len(self.file_kic) == len(self.filenames)

    def __len__(self):
        'Total number of samples'
        return len(self.indexes)

    def __getitem__(self, index):
        'Generates ONE sample of data'

        batch_filenames = self.filenames[index]
        batch_labels = self.labels[index]
        batch_kic = self.file_kic[index]

        # Generate data
        X = self.__data_generation(batch_filenames)
        y = batch_labels
        if self.return_kic:
            return X.copy(), y, batch_kic
        else:
            return X.copy(), y, batch_filenames

    def __data_generation(self, batch_filenames):
        data = np.load(batch_filenames)
        try:    
            im = data['im']
        except:    
            im = data['det']

        return im

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(x.size()[0], -1)
  


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
        conv1 = F.leaky_relu(self.conv1(input_image.unsqueeze(1)), negative_slope=0.1) # (N, C, H, W)
        conv1 = self.pool1(conv1)
        conv2 = F.leaky_relu(self.conv2(conv1), negative_slope=0.1)
        conv2 = self.pool2(conv2)
        conv3 = F.leaky_relu(self.conv3(conv2), negative_slope=0.1)
        conv3 = self.pool3(conv3)

        linear1 = F.relu(self.linear1(conv3.view(conv3.size()[0], -1)))
        pi, sigma, mu = self.mdn(linear1)
        return pi, sigma, mu

class SLOSH_Embedding(nn.Module):
    def __init__(self, embed_size=2):
        super(SLOSH_Embedding, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=7, padding=3)  # same padding 2P = K-1
        self.conv2 = nn.Conv2d(4, 8, kernel_size=5, padding=2)  # same padding 2P = K-1
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # same padding 2P = K-1
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.5)
        self.linear1 = nn.Linear(16 * 16 * 16, embed_size)

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

def online_metric_learning(transfer):

    if transfer:
        #### Transfer Learning ####
        embedding_net = SLOSH_Regressor(num_gaussians = 4)
        saved_classifier = '/home/z3384751/K2Detection/ClassifyModels/2D_classifier/Pytorch_MDN_SLOSH/V6-UpperMag13-OTF_FIXED_TrainMore-CRPS_4yrUncertainty-MDN_WITH_Drop-50percentAug_to_13_Softplus-MAPE:3.86-MAE:2.34'
        embedding_net.load_state_dict(torch.load(saved_classifier))
        pre_model = embedding_net
        pre_model_dict = pre_model.state_dict()
        model = SLOSH_Embedding()
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pre_model_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        model.conv1.weight.requires_grad = False
        model.conv1.bias.requires_grad = False
        model.conv2.weight.requires_grad = False
        model.conv2.bias.requires_grad = False
        model.conv3.weight.requires_grad = True
        model.conv3.bias.requires_grad = True
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
    root_folder = '/home/z3384751/K2Detection/Bell_Arrays_27d/Bell_Arrays_27d_Full/'#Bell_Arrays_27d_Full/'#Sample/
    file_count = 0

    subfolder_labels = []
    folder_filenames = []
    file_kic = []

    print('Parsing files in folder... ')
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for i, filex in tqdm(enumerate(filenames), total=len(filenames), unit='files'): # Getting the mags, KICS, and numax sigma for all stars in catalog
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
    train_gen = NPZ_Dataset(kic=train_kic, filenames=train_filenames, labels = train_labels, mode='classification')
    train_dataloader = utils.DataLoader(train_gen, shuffle=True, batch_size=32, num_workers=10)
    train_batch_sampler = BalancedBatchSampler(train_labels, n_classes=2, n_samples=25)
    train_dataloader_online = utils.DataLoader(train_gen, num_workers=10, batch_sampler=train_batch_sampler)

    val_gen = NPZ_Dataset(kic=val_kic, filenames=val_filenames, labels=val_labels, mode='classification')
    val_dataloader = utils.DataLoader(val_gen, shuffle=True, batch_size=32, num_workers=10)
    val_batch_sampler = BalancedBatchSampler(val_labels, n_classes=2, n_samples=25)
    val_dataloader_online = utils.DataLoader(val_gen, num_workers=10, batch_sampler=val_batch_sampler)
   

    train_loader = train_dataloader_online
    val_loader = val_dataloader_online

    n_epochs = 500
    best_loss = 9999
    model_checkpoint=True
    for epoch in range(1, n_epochs + 1):
        print('---------------------')
        print('Epoch: ', epoch)
        total_loss = 0
        train_batches = 0

        model.train()  # set to training mode
        losses = []
        for i, (data, target,_) in tqdm(enumerate(train_loader, 0), total=len(train_loader), unit='batches'):
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
                filename = '/data/marc/SLOSH_Metric_Learning/NONFROZEN_SLOSH_27d_Embedding_2_Loss:%.2f' % (val_loss)
                torch.save(model.state_dict(), filename)
                print('Model saved to %s' %filename)     
     
            else:
                print('No improvement over the best of %.4f' % best_loss)

        if epoch % 50 == 0:
            train_embeddings_baseline, train_labels_baseline, train_embed_filenames = extract_embeddings(
                train_dataloader, model)
            print('Does the Conv layer requires a gradient? ', model.conv1.weight.requires_grad) # this indexing for non-sequential
            np.savez_compressed('/data/marc/SLOSH_Metric_Learning/NONFROZEN_SLOSH_27d_Embed2_OnlineTripletTrain', embedding=train_embeddings_baseline, label=train_labels_baseline, filename=train_embed_filenames)
            val_embeddings_baseline, val_labels_baseline, val_embed_filenames = extract_embeddings(
                val_dataloader, model)
            np.savez_compressed('/data/marc/SLOSH_Metric_Learning/NONFROZEN_SLOSH_27d_Embed2_OnlineTripletVal', embedding=val_embeddings_baseline, label=val_labels_baseline, filename=val_embed_filenames)

            plot_embeddings(train_embeddings_baseline, train_labels_baseline)
            plot_embeddings(val_embeddings_baseline, val_labels_baseline)


def predict_embeddings():
    embedding_net = SLOSH_Embedding()
    model = embedding_net
    model.cuda()
    model.load_state_dict(torch.load('/home/z3384751/Echelle_DAGMM/saved_models/kepler_q9/MULTISLOSH_Embedding_2_Epoch-300'))
    torch.backends.cudnn.benchmark = True
    print('CUDNN Enabled? ', torch.backends.cudnn.enabled)
    print(str(model))

    root_folder = '/home/z3384751/K2Detection/Bell_Arrays_27d/Bell_Arrays_27d_Full/' 

    folder_filenames = []
    file_kic = []
    labels = []

    print('Parsing files in folder... ')
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for i, filex in tqdm(enumerate(filenames), total=len(filenames)): 
            if filex.endswith('.npz'):
                kicx = int(re.search(r'\d+', filex).group())
                if kicx in file_kic: # get unique only
                    continue
                file_kic.append(kicx) 
                folder_filenames.append(os.path.join(dirpath, filex))
                labels.append(0)# labels.append(np.load(os.path.join(dirpath, filex))['det'])

    file_kic = np.array(file_kic)
    folder_filenames = np.array(folder_filenames)
    labels = np.array(labels)


    print('Total Files: ', len(file_kic))
    print('Setting up generators... ')

    train_gen = NPZ_Dataset(filenames=folder_filenames, labels=labels, kic=file_kic)
    train_dataloader = utils.DataLoader(train_gen, shuffle=True, batch_size=32, num_workers=8)

    train_embeddings_baseline, train_labels_baseline, train_embed_filenames = extract_embeddings(
        train_dataloader, model)
    np.savez_compressed('27d_MULTISLOSH_Kepler_Embed_Epoch300', embedding=train_embeddings_baseline,
                        label=train_labels_baseline, filename=train_embed_filenames)


online_metric_learning(transfer=False)
#predict_embeddings()
