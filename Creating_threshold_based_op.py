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
Time_series_Y_train = torch.tensor(np.load(os.path.join(datapath, r'T_s_Y_train_flattened.npy')), dtype=torch.float32).to(device)
Time_series_X_test = torch.tensor(np.load(os.path.join(datapath, r'T_s_X_test.npy')), dtype=torch.float32).to(device)
Time_series_Y_test = torch.tensor(np.load(os.path.join(datapath, r'T_s_Y_test_flattened.npy')), dtype=torch.float32).to(device)
idx_y = np.load(os.path.join(datapath, 'y_test_idx.npy'))

seq_len = Time_series_X_train.shape[1]  # number of timestamps in 1 sample
num_lines = Time_series_X_train.shape[2]  # number of lines in 1 sample
learning_rate = 0.001


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

print("Nonzero predictions: ", np.count_nonzero(predictions.detach().numpy()))


threshold = [0.10, 0.20, 0.30, 0.33, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
for threshold_inst in threshold:
    print("THRESHOLD = ", threshold_inst)

    # create storage directories for each thresholding value
    threshDir = str(threshold_inst) + "Testing_data"
    pathX = os.path.join(location, threshDir)
    os.makedirs(pathX, exist_ok=True)
    pathP = os.path.join(pathX, 'plots')
    os.makedirs(pathP, exist_ok=True)

    pred = np.copy(predictions.detach().numpy())
    print(f"Before threshold {threshold_inst}: \n:")
    print("\t Nonzero predictions: ", np.count_nonzero(pred))
    pred[pred <= threshold_inst] = 0
    pred[pred > threshold_inst] = 1
    print("After threshold: \n", pred)
    print("\t Nonzero predictions: ", np.count_nonzero(pred))

    # pred=y_pred_r.numpy()
    # pred = pred.detach().numpy()
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
                file_name = 'excel' + str(counter_op) + '.csv'
                threshPath = os.path.join(pathX, file_name)
                df_3.to_csv(threshPath)
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
            file_name = 'excel' + str(counter_op) + '.csv'
            threshPath = os.path.join(pathX, file_name)
            df_3.to_csv(threshPath)
            temp = []
            counter_op = counter_op + 1

    # df.to_excel('out_33_10_xy_y_56.xlsx')

    np.save(os.path.join(pathX, 'output.npy'), pred)
    pred_t = pred
    # np.save(os.path.join(path, 'output.npy'), pred_t)

    # Calculating Test Accuracy

    Time_series_Y_test[Time_series_Y_test <= threshold_inst] = 0
    Time_series_Y_test[Time_series_Y_test > threshold_inst] = 1

    total_positive = np.count_nonzero(Time_series_Y_test == 1)
    total_negative = np.count_nonzero(Time_series_Y_test == 0)
    pred_pos = np.count_nonzero(pred_t == 1)
    pred_neg = np.count_nonzero(pred_t == 0)

    mat_1 = Time_series_Y_test + pred_t
    tp = np.count_nonzero(mat_1 == 2)
    tp_arr = np.count_nonzero(mat_1 == 2, axis=0)
    tn = np.count_nonzero(mat_1 == 0)
    tn_arr = np.count_nonzero(mat_1 == 0, axis=0)

    mat_2 = Time_series_Y_test - pred_t
    fn = np.count_nonzero(mat_2 == 1)
    fn_arr = np.count_nonzero(mat_2 == 1, 0)
    fp = np.count_nonzero(mat_2 == -1)
    fp_arr = np.count_nonzero(mat_2 == -1, 0)

    # Writing Confusion Matrixs Test
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

    conf_matrix = os.path.join(pathX, 'confusion_matrix.csv')
    df_cm.to_csv(conf_matrix)
    df_false_neg = df_cm.groupby(["Line_No"]).False_Negative.sum().reset_index()

    # Testing
    print('testing')
    print(f'tp: {tp}')
    print(f'tn: {tn}')
    print(f'fp: {fp}')
    print(f'fn: {fn}')
    print('Most false neg line:',
          df_false_neg.loc[df_false_neg.False_Negative.idxmax(), 'False_Negative'])
    print("No of times: ",
          df_false_neg.loc[df_false_neg.False_Negative.idxmax(), 'Line_No'])

    acc_test = (tp + tn) / (fp + fn + tp + tn)
    print('Test Accuracy:', acc_test)
    print()

    predictions_train = torchmodel(Time_series_X_train)

    predictions_train[predictions_train <= threshold_inst] = 0
    predictions_train[predictions_train > threshold_inst] = 1
    Time_series_Y_train[Time_series_Y_train <= threshold_inst] = 0
    Time_series_Y_train[Time_series_Y_train > threshold_inst] = 1

    pred_train = predictions_train
    total_positive_train = np.count_nonzero(Time_series_Y_train == 1)
    total_negative_train = np.count_nonzero(Time_series_Y_train == 0)
    # pred_pos_train=np.count_nonzero(pred_train == 1)
    # pred_neg_train =np.sum(pred_train == 0)

    mat_1 = Time_series_Y_train + pred_train
    tp_train = np.count_nonzero(mat_1 == 2)
    tn_train = np.count_nonzero(mat_1 == 0)

    mat_2 = Time_series_Y_test - pred_t
    fn_train = np.count_nonzero(mat_2 == 1)
    fp_train = np.count_nonzero(mat_2 == -1)

    acc_train = (tp_train + tn_train) / (tp_train + tn_train + fp_train + fn_train)

    df_performance = pd.DataFrame()
    df_performance['Testing Accuracy'] = [acc_test]
    df_performance['TP_GT_testing'] = [total_positive]
    df_performance['TN_GT_testing'] = [total_negative]
    df_performance['tp_testing'] = [tp]
    df_performance['tn_testing'] = [tn]
    df_performance['fp_testing'] = [fp]
    df_performance['fn_testing'] = [fn]
    df_performance['Training_Accuracy'] = [acc_train]
    df_performance['TP_GT_training'] = [total_positive_train]
    df_performance['TN_GT_training'] = [total_negative_train]
    df_performance['tp_training'] = [tp_train]
    df_performance['tn_training'] = [tn_train]
    df_performance['fp_training'] = [fp_train]
    df_performance['fn_training'] = [fn_train]
    performance_metric_path = os.path.join(pathX,'Training_testing_acc.csv')
    df_performance.to_csv(performance_metric_path)

    '''text_kwargs = dict(ha='center', va='center', fontsize=18, color='C1')
    s_test='Test Accuracy' +' ' + str(acc_test)
    plt.figure(figsize=(10, 2))
    plt.text(0.5, 0.5, s_test, **text_kwargs)
    plt.savefig(path+'\\Plots'+'\\Acc.png')
    plt.show()'''

    Y_percentage_test = np.load(os.path.join(datapath, 'T_s_Y_test_flattened.npy'))

    indices = np.where(mat_2 == 1)

    print("Y_percentage_test:", Y_percentage_test[indices])
    fn_per = Y_percentage_test[indices]

    np.save(os.path.join(pathX, 'False_Negative.npy'), fn_per)
    plt.hist(fn_per)
    histPath = os.path.join(pathX, 'plots_from_theshold')
    os.makedirs(histPath, exist_ok=True)
    histFNPath = os.path.join(histPath, 'Histogram_False_Negative.png')
    plt.savefig(histFNPath)
    # plt.show()

    indices = np.where(mat_2 == -1)
    fp_per = Y_percentage_test[indices]
    np.save(os.path.join(pathX, 'False_Positive.npy'), fp_per)
    plt.hist(fp_per)
    histFPPath = os.path.join(histPath, 'Histogram_False_Positive.png')
    plt.savefig(histFPPath)
    # plt.show()

    Errors_true_false = np.append(fn_per, fp_per)
    plt.hist(Errors_true_false)
    plt.savefig(os.path.join(histPath, 'Histogram_Errors.png'))
    # plt.show()

print('DONE!')
