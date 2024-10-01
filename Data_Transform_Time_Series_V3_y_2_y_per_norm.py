# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 21:14:06 2022

@author: ssb60
"""

import pandas as pd
import numpy as np
import random

y_data =np.load('y_train_with_index.npy')
y_data_test=np.load('y_test_with_index.npy')

# Only for x2y and xy2y
#x_data =np.load('x_train_with_index_xy.npy')
#x_data_test=np.load('x_test_with_index_xy.npy')

#x_data=np.concatenate((x_data,y_data[:,2:]),axis=1)
#x_data_test=np.concatenate((x_data_test,y_data_test[:,2:]),axis=1)
#x_data[:,2:] = (x_data[:,2:]-np.min(x_data[:,2:],0))/(np.max(x_data[:,2:],0)-np.min(x_data[:,2:],0))
#x_data_test[:,2:]= (x_data_test[:,2:]-np.min(x_data_test[:,2:],0))/(np.max(x_data_test[:,2:],0)-np.min(x_data_test[:,2:],0))

def lstm_data_transform(x_data, y_data, num_steps_x,num_steps_y,next_data_point):
    """ Changes data to the format for LSTM training 
for sliding window approach """
    # Prepare the list for the transformed data
    X, y = list(), list()
    i=0
    # Loop of the entire data set
    while(i<=x_data.shape[0]):
        #for i in range(x_data.shape[0]):
        # compute a new (sliding window) index
        end_ix = i + num_steps_x
        end_iy= i+num_steps_x+num_steps_y
        # if index is larger than the size of the dataset, we stop
        if end_ix >= x_data.shape[0]:
            break
        if end_iy >= x_data.shape[0]:
            break
        # Get a sequence of data for x
        seq_X = x_data[i:end_ix]
        
        # Get only the last element of the sequency for y
        
        seq_y = y_data[end_ix:end_iy]

                
        
        # Append the list with sequencies
        X.append(seq_X)
        y.append(seq_y)
        i=i+next_data_point # chunks sliding by 1 hour 
             
    # Make final arrays
    x_array = np.array(X)
    y_array = np.array(y)
    return x_array, y_array

# train with 8 days to predict for 9th day 
# 8 days = 192 hours 
# 9th day = 24 hours 
Time_series_X_train, Time_series_Y_train =  lstm_data_transform(y_data, y_data,192,24,1)
Time_series_X_test, Time_series_Y_test =  lstm_data_transform(y_data_test, y_data_test,192,24,1)

print(np.shape(Time_series_X_train)) # (6504, 192, 122) 8 days, it is 6504 chunks of 8 day data 
print(np.shape(Time_series_Y_train)) # (6504, 24, 122) 9th day
print(np.shape(Time_series_X_test)) # (1992, 192, 122) 8 days 
print(np.shape(Time_series_Y_test)) # (1992, 24, 122) 9th day

# arranging in LSTM format 
Time_series_X_train_1=[]
Time_series_Y_train_1=[]

for i in range (len(Time_series_X_train)): # lenght of train data = 6504
    temp=[]
    flag_discontinous=0
    for j in (Time_series_X_train[i]): # each hour (index + 120 lines) - each row of 2D matrix 
        temp.append(j) # len = 192 
    for j in (Time_series_Y_train[i]): 
        temp.append(j) # len = 192 + 24 
    for j in range(0,len(temp)-1):
        if (temp[j+1][1] -temp[j][1])>1: # checking if indices are consequetive numbers 
            flag_discontinous=1
            #print(flag_discontinous)
            break 
        else:
            continue    
    if flag_discontinous==0:
            Time_series_X_train_1.append(Time_series_X_train[i]) # append if continous 
            Time_series_Y_train_1.append(Time_series_Y_train[i])
            
Time_series_X_test_1=[]
Time_series_Y_test_1=[]
            
for i in range (len(Time_series_X_test)):
    temp=[]
    flag_discontinous=0
    for j in (Time_series_X_test[i]):
        temp.append(j)
    for j in (Time_series_Y_test[i]):
        temp.append(j)
    for j in range(0,len(temp)-1):
        if (temp[j+1][1] -temp[j][1])>1:
            flag_discontinous=1
            #print(flag_discontinous)
            break 
        else:
            continue    
    if flag_discontinous==0:
            Time_series_X_test_1.append(Time_series_X_test[i])
            Time_series_Y_test_1.append(Time_series_Y_test[i])

          
Time_series_X_train_1=np.array(Time_series_X_train_1)
Time_series_Y_train_1=np.array(Time_series_Y_train_1) 
Time_series_X_test_1=np.array(Time_series_X_test_1)
Time_series_Y_test_1=np.array(Time_series_Y_test_1)   

print(y_data.shape)
print ('Train x and Y')
print ('Shapes')
print(Time_series_X_train.shape)
#print(Time_series_X_train[0].shape)
print(Time_series_Y_train.shape)
#print(Time_series_Y_train[0].shape)
print(Time_series_X_train_1.shape)
#print(Time_series_X_train_1[0].shape)
print(Time_series_Y_train_1.shape)
#print(Time_series_Y_train_1[0].shape)

print ('Test x and Y')
print ('Shapes')
print(y_data_test.shape)
print(Time_series_X_test.shape)
print(Time_series_X_test[0].shape)
print(Time_series_Y_test.shape)
print(Time_series_Y_test[0].shape)

print(Time_series_X_test_1.shape)
print(Time_series_X_test_1[0].shape)
print(Time_series_Y_test_1.shape)
print(Time_series_Y_test_1[0].shape)

idx_test_y= Time_series_Y_test_1[:,:,0:2] # matrix of indexes 
#print(np.shape(idx_test_y))
idx_test_y_flat=[]

for i in range(len(idx_test_y)):
    temp=[]
    for j in idx_test_y[i]: 
        for k in j:
            temp.append(k)
    
    idx_test_y_flat.append(temp)   
    
idx_test_y_flat=np.array(idx_test_y_flat)
np.save('y_test_idx.npy',idx_test_y_flat[:,0:2])
print(idx_test_y_flat.shape)
import pandas 
idx_test_y_df=pd.DataFrame(idx_test_y_flat)
idx_test_y_df.to_excel('y_test_idx_1.xlsx')


# data without index 
Time_series_X_train_1=Time_series_X_train_1[:,:,2:Time_series_X_train_1.shape[2]] 
Time_series_Y_train_1=Time_series_Y_train_1[:,:,2:Time_series_Y_train_1.shape[2]]

np.save('T_s_X_train.npy',Time_series_X_train_1)
np.save('T_s_Y_train_unflattened.npy',Time_series_Y_train_1)

Time_series_X_test_1=Time_series_X_test_1[:,:,2:Time_series_X_test_1.shape[2]]
Time_series_Y_test_1=Time_series_Y_test_1[:,:,2:Time_series_Y_test_1.shape[2]]

np.save('T_s_X_test.npy',Time_series_X_test_1)
np.save('T_s_Y_test_unflattened.npy',Time_series_Y_test_1)

# Flattening 
print('Flattening:')
Time_series_Y_train_1_flat=[]
Time_series_Y_test_1_flat=[]
for i in range(len(Time_series_Y_train_1)):
    temp=[]
    for j in Time_series_Y_train_1[i]: 
        for k in j:
            temp.append(k)
    
    Time_series_Y_train_1_flat.append(temp)   
    
Time_series_Y_train_1_flat=np.array(Time_series_Y_train_1_flat)
print(Time_series_Y_train_1_flat.shape)

for i in range(len(Time_series_Y_test_1)):
    temp=[]
    for j in Time_series_Y_test_1[i]: 
        for k in j:
            temp.append(k)
    
    Time_series_Y_test_1_flat.append(temp)   
    
Time_series_Y_test_1_flat=np.array(Time_series_Y_test_1_flat)
print(Time_series_Y_test_1_flat.shape)
#Flattening 

import pandas as pd 
df_test=pd.DataFrame(Time_series_Y_test_1_flat)
df_test.to_excel('Time_series_Y_xy_y_8.xlsx')


np.save('T_s_Y_train_flattened.npy',Time_series_Y_train_1_flat)    
np.save('T_s_Y_test_flattened.npy',Time_series_Y_test_1_flat)

print('Data Transform done running!!!')