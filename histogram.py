# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 22:39:21 2022

@author: farha
@author: Abigail Broscius awb9691@rit.edu
"""
import pandas as pd
import os
import sys
import torch
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from torch.autograd import Variable
import matplotlib.pyplot as plt
from modelArchitecture_3LSTM_4FC import TorchModel

"""
Command line inputs:
1) datapath (str) - path to directory containing all .npy data files for model
2) modelname (str) - name of directory to store outputs in ./outputs_from_models/
3) resume_name (str) - the filename to resume from in /outputs_from_models/<modelname>/model/
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} \n")

modelname = sys.argv[2]
parent_dir = "outputs_from_models"
location = os.path.join(parent_dir, modelname)

# load in data
datapath = sys.argv[1]

Time_series_X_train = torch.tensor(np.load(os.path.join(datapath, r'T_s_X_train.npy')), dtype=torch.float32).to(device)
Time_series_Y_train = torch.tensor(np.load(os.path.join(datapath, r'T_s_Y_train_flattened.npy')), dtype=torch.float32).to(device)
Time_series_X_test = torch.tensor(np.load(os.path.join(datapath, r'T_s_X_test.npy')), dtype=torch.float32).to(device)
Time_series_Y_test = torch.tensor(np.load(os.path.join(datapath, r'T_s_Y_test_flattened.npy')), dtype=torch.float32).to(device)
idx_y = np.load(os.path.join(datapath, 'y_test_idx.npy'))


# boot up model from saved training
seq_len = Time_series_X_train.shape[1]  # number of timestamps in 1 sample
num_lines = Time_series_X_train.shape[2]  # number of lines in 1 sample
learning_rate = 0.001


def resume(model, filename):
    checkpath = os.path.join(location, 'model', filename)
    model.load_state_dict(torch.load(checkpath, weights_only=True))


# Instantiate the model
torchmodel = TorchModel(num_lines=num_lines, seq_len=seq_len, num_layers=1)
torchmodel.to(device)
print(torchmodel)

print("\n Using model from: ", sys.argv[3])
# assume resume file is in format "epoch-##.pth"
resume(torchmodel, sys.argv[3],)

# Define the loss function and optimizer
criterion = torch.nn.MSELoss()  # mean squared error
optimizer = torch.optim.Adam(torchmodel.parameters(),
                             lr=learning_rate,
                             betas=(0.9, 0.99),
                             eps=1e-07)

# use trained model to make predictions
torchmodel.eval()
predictions = torchmodel(Time_series_X_test)


# add predictions info to histogram
print("Nonzero predictions: ", np.count_nonzero(predictions.detach().numpy()))

print("max prediction: ", np.max(predictions.detach().numpy()))
print("min prediction: ", np.min(predictions.detach().numpy()))

predflat = predictions.detach().numpy().flatten()

print("created predflat, size: ", predflat.shape)
print("Creating histogram....")
# plt.hist(predflat, label="Predictions", alpha=0.7)
plt.title(f"Histogram of {modelname} predictions")
plt.savefig(os.path.join(location, f'hist_{modelname}_Predictions.png'))

# add ground test info
ytest = np.array(Time_series_Y_test)
print("Nonzero Y test: ", np.count_nonzero(ytest))

print("max prediction: ", np.max(ytest))
print("min prediction: ", np.min(ytest))

ytestflat = ytest.flatten()

print("created ytestflat, size: ", ytestflat.shape)
print("Creating histogram....")
plt.hist(abs(ytestflat), label="Ytest", alpha=0.7)
plt.title(f"Histogram of {modelname} predictions vs test values")

plt.legend()
# plt.savefig(f'hist_{modelname}_pred_vs_ground.png')

# add training set info
ytrain = np.array(Time_series_Y_train)
print("Nonzero Y train: ", np.count_nonzero(ytrain))

print("max y train: ", np.max(ytrain))
print("min y train: ", np.min(ytrain))

ytrainflat = ytrain.flatten()

print("created ytrainflat, size: ", ytrainflat.shape)
print("Creating histogram....")
plt.hist(abs(ytrainflat), label="Ytrain", alpha=0.5)

# add legend and save figure
plt.legend()
plt.savefig(f'hist_{modelname}_test_train.png')


exit()