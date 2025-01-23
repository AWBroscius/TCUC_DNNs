#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from spektral.layers import GCNConv
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd 
import tensorflow as tf
print(tf.__version__)


# In[2]:


#Loading data
Time_series_X_train = np.load(r'T_s_X_train.npy')
print('Train X shape =', Time_series_X_train.shape)
Time_series_Y_train = np.load(r'T_s_Y_train_flattened.npy')
print('Train Y shape =', Time_series_Y_train.shape)
Time_series_X_test = np.load(r'T_s_X_test.npy')
print('Test X shape =', Time_series_X_test.shape)
Time_series_Y_test = np.load(r'T_s_Y_test_flattened.npy')
print('Test Y shape =', Time_series_Y_test.shape)
#y_norm = np.load('adj_matrix_with_identity.npy')
#adj_matrix = np.load('adj_matrix.npy')
normalized_adj = np.load('normalized_adj_matrix.npy')
print('Adjacency matrix shape =', normalized_adj.shape)


# In[12]:


#normalized_adj[0]


# # Model

# ![GCN.webp](attachment:GCN.webp)

# In[8]:


class MyGCNLayer(Layer):
    def __init__(self, adjacency, hidden_units, activation, **kwargs):
        super(MyGCNLayer, self).__init__(**kwargs)
        self.adjacency = adjacency
        self.hidden_units = hidden_units
        self.activation = tf.keras.activations.get(activation) # activation function 
        self.transform = Dense(hidden_units) # passing (input . adjacency) through a fully connected layer 
        # number of hidden units = 120 (Must be equal to number of features (lines))

    def call(self, inputs):
        input_layer = tf.cast(inputs, tf.float32) # input layer of shape (192, 120)
        adjacency = tf.cast(self.adjacency, tf.float32) # normalized adjacency matrix 
        dot_product = K.dot(input_layer, adjacency) # Dot product of (input . adjacency)

        seq_fts = self.transform(dot_product) # passing it through a fully connected layer
        ret_fts = self.activation(seq_fts) # activation 
        
        return ret_fts   


# In[9]:


LSTM_model = Sequential()

LSTM_model.add(MyGCNLayer(input_shape=(192, 120), adjacency=normalized_adj, hidden_units=120, activation='relu'))

#LSTM_model.add(LSTM(1000, activation='sigmoid',return_sequences=True, input_shape=(192, 120)))
#LSTM_model.add(LSTM(500, activation='tanh',return_sequences=False, input_shape=(192, 120)))

LSTM_model.add(LSTM(1000, activation='sigmoid', return_sequences=True)) 
LSTM_model.add(LSTM(500, activation='tanh', return_sequences=False)) 

LSTM_model.add(Dense(3000, activation='relu'))
LSTM_model.add(Dense(1000, activation='relu'))
LSTM_model.add(Dense(3000, activation='relu'))
LSTM_model.add(Dense(2880, activation='sigmoid'))


# In[10]:


opt = keras.optimizers.Adam(learning_rate=0.001,
                            beta_1=0.9,
                            beta_2=0.99,
                            epsilon=1e-07,
                            amsgrad=False)


# In[11]:


LSTM_model.compile(loss='mse', optimizer=opt, metrics=['accuracy', 'mae'],run_eagerly=True)
LSTM_model.summary()


# In[13]:


callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,restore_best_weights=True)

history=LSTM_model.fit(Time_series_X_train[0:200], Time_series_Y_train[0:200], epochs=3, batch_size=192,
                       validation_data=(Time_series_X_test,Time_series_Y_test),callbacks=[callback])#
# remember to change Time_series_X_train[0:200]


# In[15]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

#plt.savefig(location+r'\Plots'+r'\loss.png')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




