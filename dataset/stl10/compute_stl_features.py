import os, argparse
import sys
import scipy
import timeit
import gzip
import torch
from numpy import save
import numpy as np
from sys import stdout
import pickle as pkl
import scipy.io as scio
import torchvision
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

def to_numpy(x):
	if x.is_cuda:
		return x.data.cpu().numpy()
	else:
		return x.data.numpy()

path = 'stl10_matlab/'
data=scio.loadmat(path+'train.mat')
train_X = data['X']
train_Y = data['y'].squeeze()
data=scio.loadmat(path+'test.mat')
test_X = data['X']
test_Y = data['y'].squeeze()
X = []
Y = []
X.append(train_X)
X.append(test_X)
Y.append(train_Y)
Y.append(test_Y)
X = np.concatenate(X,axis=0)
Y = np.concatenate(Y,axis=0)
X = np.reshape(X,(-1,3,96,96))
X = np.transpose(X,(0,1,3,2))
image_train = X.astype('float32')/255
image_train[:,0,:,:] = (image_train[:,0,:,:] - 0.485)/0.229
image_train[:,1,:,:] = (image_train[:,1,:,:] - 0.456)/0.224
image_train[:,2,:,:] = (image_train[:,2,:,:] - 0.406)/0.225
label_train = Y.astype('float32')-1

res50_model = torchvision.models.resnet50(pretrained=True)
res50_conv = nn.Sequential(*list(res50_model.children())[:-2])
res50_conv.eval()
data = torch.from_numpy(image_train)
dataloader = DataLoader(TensorDataset(data),batch_size=200,shuffle=False)
res50_conv = res50_conv.cuda()
total_output = []
for batch_idx, batch in enumerate(dataloader):
    inputs = batch[0].cuda()
    output = res50_conv(inputs)
    total_output.append(output.data)
total_output = torch.cat(total_output,dim=0)

feature_train = torch.sum(torch.sum(total_output,dim=-1),dim=-1)/9

image_train = to_numpy(feature_train)

save('stl_features.npy', image_train)
save('stl_label.npy', label_train)
