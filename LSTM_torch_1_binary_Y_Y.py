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

print("successful import!")

# Check for GPU architecture/compatibility
print("CUDA version:", torch.version.cuda)
print("GPU available? (pytorch):", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)


# Load in data
Time_series_X_train = torch.tensor(np.load(r'T_s_X_train.npy'), dtype=torch.float32).to(device)
Time_series_Y_train = torch.tensor(np.load(r'T_s_Y_train_flattened.npy'), dtype=torch.float32).to(device)
Time_series_X_test = torch.tensor(np.load(r'T_s_X_test.npy'), dtype=torch.float32).to(device)
Time_series_Y_test = torch.tensor(np.load(r'T_s_Y_test_flattened.npy'), dtype=torch.float32).to(device)


# Check data shape
print("x train shape:")
print(Time_series_X_train.shape)
print("y train shape:")
print(Time_series_Y_train.shape)


# Create File System for saving output
# Parent Directories
root = "."
parent_dir = "outputs_from_models"
path = os.path.join(root, parent_dir)
directory = sys.argv[1]   # name of trial run
path = os.path.join(path, directory)
Path(path).mkdir(parents=True, exist_ok=True)
print("Directory '% s' created" % directory)

# Leaf directories
pathT = os.path.join(path, 'testing_data')
Path(pathT).mkdir(parents=True, exist_ok=True)
pathM = os.path.join(path, 'model')
Path(pathM).mkdir(parents=True, exist_ok=True)
pathP = os.path.join(path, 'plots')
Path(pathP).mkdir(parents=True, exist_ok=True)


# Create a class for model structure in torch
class TorchModel(nn.Module):
    def __init__(self, seq_len, num_lines, num_layers):
        super(TorchModel, self).__init__()
        # dataset dependencies:
        self.num_lines = num_lines
        self.seq_length = seq_len
        self.num_layers = num_layers

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
        output, (hn, cn) = self.lstm_1(x)  # first lstm layer
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


# Define model hyperparameters
seq_len = Time_series_X_train.shape[1]  # number of timestamps in 1 sample
num_lines = Time_series_X_train.shape[2]  # number of features in 1 sample
# 1 sample = num timestamps x num features(power lines)
num_epochs = int(sys.argv[2])
learning_rate = 0.001

# Instantiate the model
torchmodel = TorchModel(input_size=num_lines, seq_len=seq_len, num_layers=1)
torchmodel.to(device)
print(torchmodel)

# Define the loss function and optimizer
criterion = torch.nn.MSELoss()  # mean squared error
optimizer = torch.optim.Adam(torchmodel.parameters(),
                             lr=learning_rate,
                             betas=(0.9, 0.99),
                             eps=1e-07,
                             )

# Training!!
train_loss_hist = []
test_loss_hist = []
epoch_times = []
predictions = 0
for epoch in range(num_epochs):
    t0 = time.perf_counter()
    # Train on the Training Set:
    torchmodel.train()
    outputs = torchmodel.forward(Time_series_X_train)  # forward pass
    optimizer.zero_grad()  # calculate the gradient

    # Loss Function
    loss = criterion(outputs, Time_series_Y_train)
    loss.backward()
    train_loss_hist.append(loss.item())  # track loss history
    optimizer.step()  # backpropagation

    # Evaluate on the Test Set:
    torchmodel.eval()
    with torch.no_grad():
        pred = torchmodel(Time_series_X_test)
        test_loss = criterion(pred, Time_series_Y_test).item()
        test_loss_hist.append(test_loss)
        if (epoch + 1) == num_epochs:  # on last training epoch
            # save testing predictions
            predictions = pred

    t1 = time.perf_counter()
    elapsed_time = t1-t0
    epoch_times.append(elapsed_time)

    print("Epoch: %d, Train loss: %1.5f" % (epoch, loss.item()))
    print("\t Test loss: %1.5f" % test_loss)
    print(f"\t Elapsed time: {elapsed_time:0.4f} seconds")

# Saving model
torch.save(torchmodel.state_dict(), os.path.join(pathM, ('Torch_LSTM_%d_epochs.pt' % num_epochs)))
np.save(os.path.join(pathM, 'train_history_LSTM_model.npy'), train_loss_hist, allow_pickle=True)

# Report time
avg_time = sum(epoch_times) / len(epoch_times)
print(f"Average time per epoch: {avg_time:0.4f} seconds")

# Plot the Loss History
plt.plot(train_loss_hist)
plt.plot(test_loss_hist)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(os.path.join(pathP, 'loss_history.png'))

# Apply filtering to predictions
predictions[predictions <= 0.3333] = 0
predictions[predictions > 0.3333] = 1
pred = predictions
idx_y = np.load('y_test_idx.npy')  # day and hour indices
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
np.save(os.path.join(path, 'output.npy'), pred_t)

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

Y_percentage_test = np.load('T_s_Y_test_flattened.npy')

indices = np.where(mat_2 == 1)

print(Y_percentage_test[indices])
fn_per = Y_percentage_test[indices]

np.save(os.path.join(path,'False_Negative.npy'), fn_per)
plt.hist(fn_per)
plt.savefig(os.path.join(pathP,'Histogram_False_Negative.png'))

indices = np.where(mat_2 == -1)
fp_per = Y_percentage_test[indices]
np.save(os.path.join(path, 'False_Positive.npy'), fp_per)
plt.hist(fp_per)
plt.savefig(os.path.join(pathP, 'Histogram_False_Positive.png'))

Errors_true_false = np.append(fn_per, fp_per)
plt.hist(Errors_true_false)
plt.savefig(os.path.join(pathP,'Histogram_Errors.png'))

print('Everything worked!!!')
