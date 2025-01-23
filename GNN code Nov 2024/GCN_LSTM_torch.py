#!/usr/bin/env python
# coding: utf-8
# Author: farha, awb9691
# file: GCN_LSTM_torch.py
# summary: a translation of the TC-UC Graph Convolution Network into Pytorch from TF

import sys
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Lambda, Layer
from tensorflow.keras import backend as K
import pandas as pd
import tensorflow as tf
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

print(tf.__version__)

"""
Command line inputs:
1) num_epochs (int) - the number of epochs to train for
"""

# Check for GPU architecture/compatibility
print("CUDA version:", torch.version.cuda)
print("GPU available? (pytorch):", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

# %%%%%%%%%%%%%%%%%%%%%%%%%% LOAD IN DATA  %%%%%%%%%%%%%%%%%%%%%%%%
Time_series_X_train = torch.tensor(np.load(r'T_s_X_train.npy'), dtype=torch.float32).to(device)
Time_series_Y_train = torch.tensor(np.load(r'T_s_Y_train_flattened.npy'), dtype=torch.float32).to(device)
Time_series_X_test = torch.tensor(np.load(r'T_s_X_test.npy'), dtype=torch.float32).to(device)
Time_series_Y_test = torch.tensor(np.load(r'T_s_Y_test_flattened.npy'), dtype=torch.float32).to(device)

print('Train X shape =', Time_series_X_train.shape)
print('Train Y shape =', Time_series_Y_train.shape)
print('Test X shape =', Time_series_X_test.shape)
print('Test Y shape =', Time_series_Y_test.shape)

# y_norm = np.load('adj_matrix_with_identity.npy')
# adj_matrix = np.load('adj_matrix.npy')
normalized_adj =torch.tensor(np.load('normalized_adj_matrix.npy'), dtype=torch.float32).to(device)
print('Adjacency matrix shape =', normalized_adj.shape)


class LinesDataset(Dataset):
    def __init__(self, xdata, ydata):
        self.labels = ydata
        self.data = xdata

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        datum = self.data[idx]
        return datum, label

# Convert to pytorch Datasets
training_data = LinesDataset(
    xdata=Time_series_X_train,
    ydata=Time_series_Y_train)

testing_data = LinesDataset(
    xdata=Time_series_X_test,
    ydata=Time_series_Y_test)

# Create DataLoaders to batch data
train_dataloader = DataLoader(training_data, batch_size=200, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=200, shuffle=True)



# %%%%%%%%%%%%%%%%%%%%%%%% DEFINE THE MODEL %%%%%%%%%%%%%%%%%%%%%%%
# 1 sample = num timestamps x num lines
seq_len = Time_series_X_train.shape[1]  # num timestamps in 1 sample
num_lines = Time_series_X_train.shape[2]  # num lines in 1 sample
num_epochs = int(sys.argv[1])
learning_rate = 0.0001  # changed from 2 to 3 zeroes

class torchGCN(nn.Module):
    def __init__(self, adjacency, hidden_units, **kwargs):
        super(torchGCN, self).__init__()
        self.adjacency = adjacency
        self.hidden_units = hidden_units
        self.activation = nn.ReLU()  # activation function
        self.transform = nn.Linear(in_features=hidden_units, out_features=hidden_units)  # passing (input . adjacency) through a fully connected layer
        # number of hidden units = 120 (Must be equal to number of features (lines))

    def forward(self, inputs):
        dot_product = torch.matmul(inputs, self.adjacency)  # Dot product of (input . adjacency)

        seq_fts = self.transform(dot_product)  # passing it through a fully connected layer
        ret_fts = self.activation(seq_fts)  # activation

        return ret_fts

class TorchModel(nn.Module):
    def __init__(self, seq_len, num_lines, num_layers, adj):
        super(TorchModel, self).__init__()
        # dataset dependencies:
        self.num_lines = num_lines
        self.seq_length = seq_len
        self.num_layers = num_layers

        self.gcn = torchGCN(adjacency=adj, hidden_units=num_lines)
        # LSTM layer 1
        self.lstm_1 = nn.LSTM(input_size=num_lines, hidden_size=1000, batch_first=True)
        # LSTM layer 2
        self.lstm_2 = nn.LSTM(input_size=1000, hidden_size=500, batch_first=True)
        # Rest of the Neural Net
        self.fc_1 = nn.Linear(500, 3000)
        self.fc_2 = nn.Linear(3000, 1000)
        self.fc_3 = nn.Linear(1000, 3000)
        self.op_layer = nn.Linear(3000, 24 * num_lines)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # Propagate input through LSTM
        output = self.gcn(x)
        output, (hn, cn) = self.lstm_1(output)  # first lstm layer
        output = self.sigmoid(output)
        # hn = hn.view(-1, 500)
        out, (hn, cn) = self.lstm_2(output)  # second lstm layer
        out = self.tanh(out)
        hn = hn.view(-1, 500)
        out = self.relu(hn)
        out = self.fc_1(out)  # first fc layer
        out = self.relu(out)
        out = self.fc_2(out)  # second fc layer
        out = self.relu(out)
        out = self.fc_3(out)  # third fc layer
        out = self.relu(out)
        out = self.op_layer(out)  # O/P layer
        out = self.sigmoid(out)

        return out


class MyGCNLayer(Layer):
    def __init__(self, adjacency, hidden_units, activation, **kwargs):
        super(MyGCNLayer, self).__init__(**kwargs)
        self.adjacency = adjacency
        self.hidden_units = hidden_units
        self.activation = tf.keras.activations.get(activation)  # activation function
        self.transform = Dense(hidden_units)  # passing (input . adjacency) through a fully connected layer
        # number of hidden units = 120 (Must be equal to number of features (lines))

    def call(self, inputs):
        input_layer = tf.cast(inputs, tf.float32)  # input layer of shape (192, 120)
        adjacency = tf.cast(self.adjacency, tf.float32)  # normalized adjacency matrix
        dot_product = K.dot(input_layer, adjacency)  # Dot product of (input . adjacency)

        seq_fts = self.transform(dot_product)  # passing it through a fully connected layer
        ret_fts = self.activation(seq_fts)  # activation

        return ret_fts

# Instantiate the model
torchmodel = TorchModel(num_lines=num_lines, seq_len=seq_len, num_layers=1, adj=normalized_adj)
torchmodel.to(device)
print(torchmodel)

# Define the loss function and optimizer
criterion = torch.nn.MSELoss()  # mean squared error
optimizer = torch.optim.Adam(torchmodel.parameters(),
                             lr=learning_rate,
                             betas=(0.9, 0.99),
                             eps=1e-07)


# %%%%%%%%%%%%%%%%%% TRAINING!! %%%%%%%%%%%%%%%%%%%%%%%
cur_epoch = 0
train_loss_hist = []
test_loss_hist = []
epoch_times = []
for epoch in range(num_epochs):
    t0 = time.perf_counter() # start timer

    # get next data batch
    batch_data, batch_labels = next(iter(train_dataloader))

    # Train on the Training Set:
    torchmodel.train()
    outputs = torchmodel.forward(batch_data)  # forward pass
    optimizer.zero_grad()  # calculate the gradient

    # Loss Function
    loss = criterion(outputs, batch_labels)
    loss.backward()
    train_loss_hist.append(loss.item())  # track loss history
    optimizer.step()  # backpropagation

    # Evaluate on the Test Set:
    torchmodel.eval()
    with torch.no_grad():
        pred = torchmodel(Time_series_X_test)
        test_loss = criterion(pred, Time_series_Y_test).item()
        test_loss_hist.append(test_loss)

    # Calculate epoch time
    t1 = time.perf_counter()
    elapsed_time = t1 - t0
    elapsed_min = elapsed_time / 60
    epoch_times.append(elapsed_time)

    # Print epoch results
    print("Epoch: %d" % epoch)
    print("\t Train loss: %1.5f" % loss.item())
    print("\t Test loss: %1.5f" % test_loss)
    print(f"\t Elapsed time: {elapsed_time:0.4f} seconds")
    print(f"\t \t ~= {elapsed_min} min")

    # Early stopping
    if epoch > 10:
        if test_loss - test_loss_hist[epoch-1] == 0:
            print("Early Stopping!")
            break


# %%%%%%%%%%%%%%%%%%%% RESULTS REPORTING %%%%%%%%%%%%%%%%%%%%%%%%%%

# Saving model
final_epoch = int(num_epochs) + int(cur_epoch)
torch.save(torchmodel.state_dict(), f'Torch_LSTM_%d_{num_epochs}_epochs.pt' % final_epoch)


# Report average time
avg_time = sum(epoch_times) / len(epoch_times)
print(f"Average time per epoch: {avg_time:0.4f} seconds")


# Plot the Loss History
plt.plot(train_loss_hist)
plt.plot(test_loss_hist)
plt.title(f'RTS GCN model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(f'GCN_torch_{num_epochs}_loss_history.png')


