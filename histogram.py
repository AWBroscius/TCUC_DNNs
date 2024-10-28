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
Time_series_Y_train = torch.tensor(np.load(os.path.join(datapath, r'T_s_Y_train_flattened.npy')),
                                   dtype=torch.float32).to(device)
Time_series_X_test = torch.tensor(np.load(os.path.join(datapath, r'T_s_X_test.npy')), dtype=torch.float32).to(device)
Time_series_Y_test = torch.tensor(np.load(os.path.join(datapath, r'T_s_Y_test_flattened.npy')), dtype=torch.float32).to(
    device)
idx_y = np.load(os.path.join(datapath, 'y_test_idx.npy'))

print("Nonzero Y test: ", np.count_nonzero(Time_series_Y_test))

print("max Y test: ", np.max(Time_series_Y_test))
print("min Y test: ", np.min(Time_series_Y_test))

print("Creating histogram for testing data....")
plt.hist(Time_series_Y_test)
plt.savefig('histYtest.png')

print("Nonzero Y train: ", np.count_nonzero(Time_series_Y_train))

print("max Y train: ", np.max(Time_series_Y_train))
print("min Y train: ", np.min(Time_series_Y_train))

print("Creating histogram for training data....")
plt.figure(2)
plt.hist(Time_series_Y_train)
plt.savefig('histYtrain.png')

exit()