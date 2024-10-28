# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 21:06:30 2022
Modified on Mon Sept 9 by awb9691
@author: ssb60
@author: Abigail Broscius awb9691@rit.edu
"""
import os
import torch
import sys
import time
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from trainingFuncs import *
from modelArchitecture import TorchModel
print("successful import!")


# Check for GPU architecture/compatibility
print("CUDA version:", torch.version.cuda)
print("GPU available? (pytorch):", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

"""
Code has been modularized to increase ease of architecture experimentation.

Command line inputs:
1) datapath (str) - path to directory containing all .npy data files for model
2) modelname (str) - name of directory to store outputs in ./outputs_from_models/
3) num_epochs (int) - the number of epochs to train for
4) resume_name (str) - the filename to resume from in /outputs_from_models/<modelname>/model/
5) resume_num (int) - epoch number that (4) is picking up from 
"""

# %%%%%%%%%%%%%%%%%%%%%%%%%% DATA PREPARATION %%%%%%%%%%%%%%%%%%%%%%%%
# Load in data
datapath = sys.argv[1]

Time_series_X_train = torch.tensor(np.load(os.path.join(datapath, r'T_s_X_train.npy')), dtype=torch.float32).to(device)
Time_series_Y_train = torch.tensor(np.load(os.path.join(datapath, r'T_s_Y_train_flattened.npy')),
                                   dtype=torch.float32).to(device)
Time_series_X_test = torch.tensor(np.load(os.path.join(datapath, r'T_s_X_test.npy')), dtype=torch.float32).to(device)
Time_series_Y_test = torch.tensor(np.load(os.path.join(datapath, r'T_s_Y_test_flattened.npy')), dtype=torch.float32).to(
    device)

# Check data shape
print("x train shape:", Time_series_X_train.shape)
print("y train shape:", Time_series_Y_train.shape)

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


# Create File System for saving output
modelname = sys.argv[2]  # name of trial run
path, pathT, pathM, pathP = createFiletree(modelname)

# %%%%%%%%%%%%%%%%%%%%%%%% DEFINE THE MODEL %%%%%%%%%%%%%%%%%%%%%%%
# 1 sample = num timestamps x num lines
seq_len = Time_series_X_train.shape[1]  # num timestamps in 1 sample
num_lines = Time_series_X_train.shape[2]  # num lines in 1 sample
num_epochs = int(sys.argv[3])
learning_rate = 0.0001  # changed from 2 to 3 zeroes


# Instantiate the model
torchmodel = TorchModel(num_lines=num_lines, seq_len=seq_len, num_layers=1)
torchmodel.to(device)
print(torchmodel)

# Resume from checkpoint, if given
if len(sys.argv) > 4:
    print("Resuming training from: ", sys.argv[4])

    # load weights
    cur_epoch = int(sys.argv[5])
    print("The current epoch is: ", cur_epoch)
    num_epochs = num_epochs - cur_epoch  # adjust epoch number to avoid file overwrites
    resume(torchmodel, pathM, sys.argv[4])

    # load loss history
    train_loss_hist = np.load(os.path.join(pathM, f'loss_hist_train_epoch_{int(cur_epoch)}.npy')).tolist()
    test_loss_hist = np.load(os.path.join(pathM, f'loss_hist_test_epoch_{int(cur_epoch)}.npy')).tolist()
else:  # start fresh training
    cur_epoch = 0
    train_loss_hist = []
    test_loss_hist = []


# Define the loss function and optimizer
criterion = torch.nn.MSELoss()  # mean squared error
optimizer = torch.optim.Adam(torchmodel.parameters(),
                             lr=learning_rate,
                             betas=(0.9, 0.99),
                             eps=1e-07)

# %%%%%%%%%%%%%%%%%% TRAINING!! %%%%%%%%%%%%%%%%%%%%%%%
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

    # Save Checkpoint
    if (epoch % 2 == 0) or (epoch == num_epochs):  # every other epoch, and final
        # save weights
        checkpoint(torchmodel, pathM, f"epoch-{int(epoch + cur_epoch)}.pth")

        # save loss history
        np.save(os.path.join(pathM, f'loss_hist_train_epoch_{int(epoch + cur_epoch)}.npy'),
                train_loss_hist, allow_pickle=True)
        np.save(os.path.join(pathM, f'loss_hist_test_epoch_{int(epoch + cur_epoch)}.npy'),
                test_loss_hist, allow_pickle=True)


# %%%%%%%%%%%%%%%%%%%% RESULTS REPORTING %%%%%%%%%%%%%%%%%%%%%%%%%%

# Saving model
final_epoch = int(num_epochs) + int(cur_epoch)
torch.save(torchmodel.state_dict(), os.path.join(pathM, ('Torch_LSTM_%d_epochs.pt' % final_epoch)))
np.save(os.path.join(pathM, 'train_history_LSTM_model.npy'), train_loss_hist, allow_pickle=True)


# Report average time
avg_time = sum(epoch_times) / len(epoch_times)
print(f"Average time per epoch: {avg_time:0.4f} seconds")


# Plot the Loss History
plt.plot(train_loss_hist)
plt.plot(test_loss_hist)
plt.title(f'{modelname} model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(os.path.join(pathP, 'loss_history.png'))


# Apply filtering to predictions
torchmodel.eval()
predictions = torchmodel(Time_series_X_test)
predictions[predictions <= 0.3333] = 0
predictions[predictions > 0.3333] = 1
pred = np.copy(predictions.detach().numpy())
np.save(os.path.join(path, 'output.npy'), pred)

idx_y = np.load(os.path.join(datapath, 'y_test_idx.npy'))  # day and hour indices
df = pd.DataFrame(pred)
df.insert(loc=0, column='day', value=idx_y[:, 0])

mask1 = df.duplicated(subset=['day'], keep="first")  # this line is to get the first occ.
df2 = df[~mask1]

pred_0_hr = df2.to_numpy()

pred_0_hr_list = []
temp = []
counter_op = 0
for i in range(1, len(pred_0_hr)):
    if (pred_0_hr[i, 0] - pred_0_hr[i - 1, 0]) < 2:
        temp.append(pred_0_hr[i, :])
        if i == len(pred_0_hr) - 1:
            temp_np = np.array(temp)
            index_for_temp = 2
            new_temp_1 = []
            for count_reshape_r in range(0, len(temp_np)):
                new_temp = []
                new_temp.append(temp_np[count_reshape_r, 0])

                for count_reshape_c in range(1, temp_np.shape[1]):
                    if ((count_reshape_c) % num_lines) == 0:
                        new_temp.append(temp_np[count_reshape_r, count_reshape_c])
                        new_temp_1.append(new_temp)
                        new_temp = []
                        new_temp.append(temp_np[count_reshape_r, 0])

                    else:
                        new_temp.append(temp_np[count_reshape_r, count_reshape_c])

            new_temp_1 = np.array(new_temp_1)
            df_3 = pd.DataFrame(new_temp_1)
            file_name = os.path.join(pathT, 'excel' + str(counter_op) + '.csv')
            df_3.to_csv(file_name)
    else:
        temp_np = np.array(temp)
        index_for_temp = 2
        new_temp_1 = []
        for count_reshape_r in range(0, len(temp_np)):
            new_temp = []
            new_temp.append(temp_np[count_reshape_r, 0])

            for count_reshape_c in range(1, temp_np.shape[1]):
                if ((count_reshape_c) % num_lines) == 0:
                    new_temp.append(temp_np[count_reshape_r, count_reshape_c])
                    new_temp_1.append(new_temp)
                    new_temp = []
                    new_temp.append(temp_np[count_reshape_r, 0])

                else:
                    new_temp.append(temp_np[count_reshape_r, count_reshape_c])

        new_temp_1 = np.array(new_temp_1)
        df_3 = pd.DataFrame(new_temp_1)
        file_name = os.path.join(pathT, 'excel' + str(counter_op) + '.csv')
        df_3.to_csv(file_name)
        temp = []
        counter_op = counter_op + 1

pred_t = pred

Time_series_Y_test[Time_series_Y_test <= 0.33] = 0
Time_series_Y_test[Time_series_Y_test > 0.33] = 1

tp = np.count_nonzero(Time_series_Y_test == 1)
tn = np.count_nonzero(Time_series_Y_test == 0)
pp = np.count_nonzero(pred == 1)
pn = np.count_nonzero(pred == 0)

print('trueval_pos: ', tp)
print('trueval_neg: ', tn)
print('positive_pred: ', pp)
print('negative_pred: ', pn)

total_positive = np.count_nonzero(Time_series_Y_test == 1)
total_negative = np.count_nonzero(Time_series_Y_test == 0)
pred_pos = np.count_nonzero(pred_t == 1)
pred_neg = np.count_nonzero(pred_t == 0)

mat_1 = Time_series_Y_test + pred_t
tp = np.count_nonzero(mat_1 == 2)
tp_arr = np.count_nonzero(mat_1 == 2, 0)
tn = np.count_nonzero(mat_1 == 0)
tn_arr = np.count_nonzero(mat_1 == 0, 0)

mat_2 = Time_series_Y_test - pred_t
fn = np.count_nonzero(mat_2 == 1)
fn_arr = np.count_nonzero(mat_2 == 1, 0)
fp = np.count_nonzero(mat_2 == -1)
fp_arr = np.count_nonzero(mat_2 == -1, 0)

df_cm = pd.DataFrame(tp_arr.T)
# df. rename(columns = {'old_col1':'new_col1', 'old_col2':'new_col2'}, inplace = True)
df_cm.rename(columns={0: 'True_Positive'}, inplace='True')
df_cm.insert(loc=1, column='True_Negative', value=tn_arr)
df_cm.insert(loc=2, column='False_Positive', value=fp_arr)
df_cm.insert(loc=3, column='False_Negative', value=fn_arr)
lines = np.linspace(1, num_lines, num=num_lines)
lines_arr = np.tile(lines, 24)
df_cm.insert(loc=0, column='Line_No', value=lines_arr)
hour = np.linspace(1, 24, num=24)
hour_arr = np.repeat(hour, num_lines)
df_cm.insert(loc=1, column='Hour', value=hour_arr)
file_name = os.path.join(path, 'confusion_matrix.csv')
df_cm.to_csv(file_name)
df_false_neg = df_cm.groupby(["Line_No"]).False_Negative.sum().reset_index()

print('true pos: ', tp)
print('true neg: ', tn)
print('false pos: ', fp)
print('false neg: ', fn)

f1_score = 2 * tp / (2 * tp + fp + fn)
print('F1-score = ', f1_score)

print('Most false neg line')
print(df_false_neg.loc[df_false_neg.False_Negative.idxmax(), 'False_Negative'])
print("No of times")
print(df_false_neg.loc[df_false_neg.False_Negative.idxmax(), 'Line_No'])

Y_percentage_test = np.load(os.path.join(datapath, 'T_s_Y_test_flattened.npy'))

indices = np.where(mat_2 == 1)

print(Y_percentage_test[indices])
fn_per = Y_percentage_test[indices]

np.save(os.path.join(path, 'False_Negative.npy'), fn_per)
plt.hist(fn_per)
plt.savefig(os.path.join(pathP, 'Histogram_False_Negative.png'))

indices = np.where(mat_2 == -1)
fp_per = Y_percentage_test[indices]
np.save(os.path.join(path, 'False_Positive.npy'), fp_per)
plt.hist(fp_per)
plt.savefig(os.path.join(pathP, 'Histogram_False_Positive.png'))

Errors_true_false = np.append(fn_per, fp_per)
plt.hist(Errors_true_false)
plt.savefig(os.path.join(pathP, 'Histogram_Errors.png'))

print('Everything worked!!!')
