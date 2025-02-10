#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Lambda, Layer
import keras
from tensorflow.keras import backend as K
from tensorflow import keras
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.layers import BatchNormalization
import pandas as pd
import tensorflow as tf
import sys
import os
from pathlib import Path


print(tf.__version__)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# In[2]:


# Loading data
datapath = sys.argv[1]

Time_series_X_train = np.load(os.path.join(datapath, r'T_s_X_train.npy'))
Time_series_Y_train = np.load(os.path.join(datapath, r'T_s_Y_train_flattened.npy'))
Time_series_X_test = np.load(os.path.join(datapath, r'T_s_X_test.npy'))
Time_series_Y_test = np.load(os.path.join(datapath, r'T_s_Y_test_flattened.npy'))


print('Train X shape =', Time_series_X_train.shape)
print('Train Y shape =', Time_series_Y_train.shape)
print('Test X shape =', Time_series_X_test.shape)
print('Test Y shape =', Time_series_Y_test.shape)
# y_norm = np.load('adj_matrix_with_identity.npy')
# adj_matrix = np.load('adj_matrix.npy')
normalized_adj = np.load(os.path.join(datapath,'normalized_adj_matrix.npy'))
print('Adjacency matrix shape =', normalized_adj.shape)


# Create File System for saving output
# Parent Directories
root = "."
parent_dir = "outputs_from_models"
path = os.path.join(root, parent_dir)
directory = sys.argv[1]  # name of trial run
print("MODEL NAME: ", sys.argv[1])
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


# In[12]:


# normalized_adj[0]


# # Model

# ![GCN.webp](attachment:GCN.webp)

# In[8]:


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

    # In[9]:


LSTM_model = Sequential()

LSTM_model.add(MyGCNLayer(input_shape=(192, 120), adjacency=normalized_adj, hidden_units=120, activation='relu'))

# LSTM_model.add(LSTM(1000, activation='sigmoid',return_sequences=True, input_shape=(192, 120)))
# LSTM_model.add(LSTM(500, activation='tanh',return_sequences=False, input_shape=(192, 120)))

LSTM_model.add(LSTM(1000, activation='sigmoid', return_sequences=True))
LSTM_model.add(LSTM(500, activation='tanh', return_sequences=False))

LSTM_model.add(Dense(3000, activation='relu'))
LSTM_model.add(Dense(1000, activation='relu'))
LSTM_model.add(Dense(3000, activation='relu'))
LSTM_model.add(Dense(2880, activation='sigmoid'))

# In[10]:

print("Learning rate: ", sys.argv[2])
opt = keras.optimizers.Adam(learning_rate=float(sys.argv[2]),
                            beta_1=0.9,
                            beta_2=0.99,
                            epsilon=1e-07,
                            amsgrad=False)

# In[11]:


LSTM_model.compile(loss='mse', optimizer=opt, metrics=['accuracy', 'mae'], run_eagerly=True)
LSTM_model.summary()

# In[13]:


# callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,restore_best_weights=True)

history = LSTM_model.fit(Time_series_X_train, Time_series_Y_train, epochs=200, batch_size=192,
                         validation_data=(Time_series_X_test, Time_series_Y_test))  #
# remember to change Time_series_X_train[0:200]


# In[15]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

plt.savefig(os.path.join(pathP, 'loss.png'))
# plt.show()

# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:
predictions = LSTM_model.predict(Time_series_X_test)
# y_pred_r=tf.round(predictions )
# y_pred_r=tf.math.minimum(y_pred_r,1)
predictions[predictions <= 0.3333] = 0
predictions[predictions > 0.3333] = 1
pred = predictions
# pred=y_pred_r.numpy()
idx_y = np.load('y_test_idx.npy')
df = pd.DataFrame(pred)
df.insert(loc=0, column='day', value=idx_y[:, 0])
# df.insert(loc=1,column='hour',value=idx_y[:,1])
# df['hour']=idx_y[:,1]

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
                    if ((count_reshape_c) % 120) == 0:
                        new_temp.append(temp_np[count_reshape_r, count_reshape_c])
                        new_temp_1.append(new_temp)
                        new_temp = []
                        new_temp.append(temp_np[count_reshape_r, 0])

                    else:
                        new_temp.append(temp_np[count_reshape_r, count_reshape_c])

            new_temp_1 = np.array(new_temp_1)
            df_3 = pd.DataFrame(new_temp_1)
            file_name = os.path.join(pathP, f"excel_{counter_op}.csv")
            # file_name=location +r'\Testing_data\excel'+str(counter_op)+'.xlsx'
            df_3.to_csv(file_name)
    else:
        temp_np = np.array(temp)
        index_for_temp = 2
        new_temp_1 = []
        for count_reshape_r in range(0, len(temp_np)):
            new_temp = []
            new_temp.append(temp_np[count_reshape_r, 0])

            for count_reshape_c in range(1, temp_np.shape[1]):
                if ((count_reshape_c) % 120) == 0:
                    new_temp.append(temp_np[count_reshape_r, count_reshape_c])
                    new_temp_1.append(new_temp)
                    new_temp = []
                    new_temp.append(temp_np[count_reshape_r, 0])

                else:
                    new_temp.append(temp_np[count_reshape_r, count_reshape_c])

        new_temp_1 = np.array(new_temp_1)
        df_3 = pd.DataFrame(new_temp_1)
        file_name = os.path.join(pathT, "excel_{counter_op}.csv")
        # location+r'\Testing_data\excel'+str(counter_op)+'.xlsx'
        df_3.to_csv(file_name)
        temp = []
        counter_op = counter_op + 1

# df.to_csv('out_33_10_xy_y_56.xlsx')
np.save(os.path.join(pathT, 'output.npy'), pred)

pred_t = pred
# np.save(location +'\output.npy',pred_t)


Time_series_Y_test[Time_series_Y_test <= 0.33] = 0
Time_series_Y_test[Time_series_Y_test > 0.33] = 1

tp = np.count_nonzero(Time_series_Y_test == 1)
tn = np.sum(Time_series_Y_test == 0)
# pp=np.count_nonzero(pred == 1)
# pn =np.sum(pred == 0)

print('trueval_pos')
print(tp)
print('trueval_neg')
print(tn)
print('positive_pred')
'''print(pp)
print('negative_pred')
print(pn)'''

total_positive = np.count_nonzero(Time_series_Y_test == 1)
total_negative = np.sum(Time_series_Y_test == 0)
pred_pos = np.count_nonzero(pred_t == 1)
pred_neg = np.sum(pred_t == 0)

mat_1 = Time_series_Y_test + pred_t
tp = np.sum(mat_1 == 2)
tp_arr = np.sum(mat_1 == 2, 0)
tn = np.sum(mat_1 == 0)
tn_arr = np.sum(mat_1 == 0, 0)

mat_2 = Time_series_Y_test - pred_t
fn = np.sum(mat_2 == 1)
fn_arr = np.sum(mat_2 == 1, 0)
fp = np.sum(mat_2 == -1)
fp_arr = np.sum(mat_2 == -1, 0)

df_cm = pd.DataFrame(tp_arr.T)
# df. rename(columns = {'old_col1':'new_col1', 'old_col2':'new_col2'}, inplace = True)
df_cm.rename(columns={0: 'True_Positive'}, inplace='True')
df_cm.insert(loc=1, column='True_Negative', value=tn_arr)
df_cm.insert(loc=2, column='False_Positive', value=fp_arr)
df_cm.insert(loc=3, column='False_Negative', value=fn_arr)
lines = np.linspace(1, 120, num=120)
lines_arr = np.tile(lines, 24)
df_cm.insert(loc=0, column='Line_No', value=lines_arr)
hour = np.linspace(1, 24, num=24)
hour_arr = np.repeat(hour, 120)
df_cm.insert(loc=1, column='Hour', value=hour_arr)
file_name = os.path.join(path, "confusion_matrix.csv")
# file_name=location+'\confusion_matrix.xlsx'
df_cm.to_csv(file_name)
df_false_neg = df_cm.groupby(["Line_No"]).False_Negative.sum().reset_index()

print('tp')
print(tp)
print('tn')
print(tn)
print('fp')
print(fp)
print('fn')
print(fn)

f1_score = 2 * tp / (2 * tp + fp + fn)
print('F1-score = ', f1_score)

print('Most false neg line')
print(df_false_neg.loc[df_false_neg.False_Negative.idxmax(), 'False_Negative'])
print("No of times")
print(df_false_neg.loc[df_false_neg.False_Negative.idxmax(), 'Line_No'])

LSTM_model.evaluate(Time_series_X_test, Time_series_Y_test)
LSTM_model.evaluate(Time_series_X_train, Time_series_Y_train)

# saving model
# LSTM_model.save(os.path.join(pathM, 'LSTM_model.h5'))
np.save(os.path.join(pathM, 'my_history_LSTM_model.npy'), history.history, allow_pickle=True)

Y_percentage_test = np.load('T_s_Y_test_flattened.npy')

indices = np.where(mat_2 == 1)

print(Y_percentage_test[indices])
fn_per = Y_percentage_test[indices]

file_name = os.path.join(pathP, "False_Negative.npy")
np.save(file_name, fn_per)
plt.hist(fn_per)
plt.savefig(os.path.join(pathP, 'Histogram_False_Negative.png'))
plt.show()

indices = np.where(mat_2 == -1)
fp_per = Y_percentage_test[indices]
file_name = os.path.join(pathP, "False_Positive.npy")
np.save(file_name, fp_per)
plt.figure(2)
plt.hist(fp_per)
plt.savefig(os.path.join(pathP, 'Histogram_False_Positive.png'))
plt.show()

Errors_true_false = np.append(fn_per, fp_per)
plt.figure(3)
plt.hist(Errors_true_false)
plt.savefig(os.path.join(pathP, 'Histogram_Errors.png'))
plt.show()

print('Everything worked!!!')