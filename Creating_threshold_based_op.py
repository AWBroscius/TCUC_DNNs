# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 22:39:21 2022

@author: farha
"""
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import keras
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
import os


def my_metric_fn(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    #print(y_pred.dtype)
    y_true = tf.cast(y_true > 0.33, y_true.dtype) * 1
    y_pred_r = tf.cast(y_pred > 0.33, y_pred.dtype) * 1
    
    '''y_true=tf.round(y_true)
    y_pred_r=tf.round(y_pred)'''
    
    
    
    #return (y_pred.get_shape().as_list()[1]*y_pred.get_shape().as_list()[0] - tf.math.count_nonzero((y_true - y_pred_r)*100))/(y_pred.get_shape().as_list()[1]*y_pred.get_shape().as_list()[0])
    return 1-(tf.math.count_nonzero( y_true- y_pred_r)/(y_pred.get_shape().as_list()[1]*y_pred.get_shape().as_list()[0]))


directory = "X_to_Y_8_days__reg_test_adam"
parent_dir = r"C:\Users\farha\OneDrive\Desktop\TLS LSTM code\Outputs_from_models"
#"D:\TL_11_18\Transmission_Line_Screen\Outputs_from_models"
location=parent_dir + "\\" + directory

LSTM_model = Sequential()
LSTM_model.add(LSTM(1000, activation='sigmoid',return_sequences=True, input_shape=(192, 120)))
LSTM_model.add(LSTM(500, activation='tanh',return_sequences=False, input_shape=(192, 120)))
LSTM_model.add(Dense(3000, activation='relu'))
#LSTM_model.add(BatchNormalization())
#LSTM_model.add(Dropout(0.1))
LSTM_model.add(Dense(1000, activation='relu'))
#LSTM_model.add(BatchNormalization())
#LSTM_model.add(Dropout(0.1))
LSTM_model.add(Dense(3000, activation='relu'))
#LSTM_model.add(BatchNormalization())
#LSTM_model.add(Dropout(0.1))
LSTM_model.add(Dense(2880, activation='sigmoid'))
#opt = keras.optimizers.Adam(learning_rate=0.1e-8,decay=0.8)
opt = keras.optimizers.Adam(learning_rate=0.001,
                            beta_1=0.9,
                            beta_2=0.99,
                            epsilon=1e-07,
                            amsgrad=False) #1e-6   0.0001 0.9
#opt=keras.optimizers.SGD(learning_rate=0.00001,decay=0.8)

LSTM_model.compile(loss='mse', optimizer=opt, metrics=['accuracy', 'mae'],run_eagerly=True)
LSTM_model.summary()
LSTM_model.load_weights(location+'\Model'+'\LSTM_model.h5')

#LSTM_model=tf.keras.models.load_model(location+'\Model'+'\LSTM_model') #,custom_objects={ 'my_metric_fn': my_metric_fn})

Time_series_X_test = np.load(r'C:\Users\farha\OneDrive\Desktop\TLS LSTM code\T_s_X_test.npy')
Time_series_Y_test = np.load(r'C:\Users\farha\OneDrive\Desktop\TLS LSTM code\T_s_Y_test_flattened.npy')
#predictions = LSTM_model.predict(Time_series_X_test)




location=parent_dir + "\\" + directory

threshold=[0.10, 0.20,0.30,0.33,0.40,0.50,0.60,0.70,0.80,0.90]
for threshold_inst in threshold: 
    path=location+ '\\'+ str(threshold_inst)+"Testing_data"
    os.makedirs(path)
    path_plot=path+'\\'+'plots'
    os.makedirs(path_plot)
    Time_series_X_test = np.load(r'C:\Users\farha\OneDrive\Desktop\TLS LSTM code\T_s_X_test.npy')
    Time_series_Y_test = np.load(r'C:\Users\farha\OneDrive\Desktop\TLS LSTM code\T_s_Y_test_flattened.npy')
    Time_series_X_train =np.load(r'C:\Users\farha\OneDrive\Desktop\TLS LSTM code\T_s_X_train.npy')
    Time_series_Y_train =np.load(r'C:\Users\farha\OneDrive\Desktop\TLS LSTM code\T_s_Y_train_flattened.npy')
    predictions = LSTM_model.predict(Time_series_X_test)

    predictions[predictions<=threshold_inst]=0
    predictions[predictions > threshold_inst]=1
    pred=predictions
    #pred=y_pred_r.numpy()
    idx_y=np.load(r'C:\Users\farha\OneDrive\Desktop\TLS LSTM code\y_test_idx.npy')
    df=pd.DataFrame(pred)
    df.insert(loc=0,column='day',value=idx_y[:,0])


    mask1 = df.duplicated(subset=['day'],keep = "first") # this line is to get the first occ.
    df2=df[~mask1]

    pred_0_hr =df2.to_numpy()

    pred_0_hr_list =[]
    temp=[]
    counter_op=0
    for i in range (1,len(pred_0_hr)):
        if ( pred_0_hr[i,0]-pred_0_hr[i-1,0])<2:
            temp.append(pred_0_hr[i,:])
            if i== len(pred_0_hr)-1:
                temp_np=np.array(temp)
                index_for_temp=2
                new_temp_1=[]
                for count_reshape_r in range (0,len(temp_np)):
                    new_temp=[]
                    new_temp.append(temp_np[count_reshape_r,0])
                        
                    for count_reshape_c in range (1,temp_np.shape[1]):
                        if ((count_reshape_c) % 120)==0 : 
                            new_temp.append(temp_np[count_reshape_r,count_reshape_c])
                            new_temp_1.append(new_temp)
                            new_temp=[]
                            new_temp.append(temp_np[count_reshape_r,0])
                            
                        else: 
                            new_temp.append(temp_np[count_reshape_r,count_reshape_c])

                               
                new_temp_1=np.array(new_temp_1)
                df_3=pd.DataFrame(new_temp_1)
                file_name=path +'\\excel'+str(counter_op)+'.xlsx'
                df_3.to_excel(file_name)
        else:
            temp_np=np.array(temp)
            index_for_temp=2
            new_temp_1=[]
            for count_reshape_r in range (0,len(temp_np)):
                new_temp=[]
                new_temp.append(temp_np[count_reshape_r,0])
                    
                for count_reshape_c in range (1,temp_np.shape[1]):
                    if ((count_reshape_c ) % 120)==0 : 
                        new_temp.append(temp_np[count_reshape_r,count_reshape_c])
                        new_temp_1.append(new_temp)
                        new_temp=[]
                        new_temp.append(temp_np[count_reshape_r,0])
                        
                    else: 
                        new_temp.append(temp_np[count_reshape_r,count_reshape_c])
                        
           
            new_temp_1=np.array(new_temp_1)
            df_3=pd.DataFrame(new_temp_1)
            file_name=path+'\\excel'+str(counter_op)+'.xlsx'
            df_3.to_excel(file_name)
            temp=[]
            counter_op=counter_op+1
            

    #df.to_excel('out_33_10_xy_y_56.xlsx')
    np.save(path +'\output.npy',pred)


    pred_t=pred
    np.save(path +'\output.npy',pred_t)


    #Calculating Test Accuracy

    Time_series_Y_test[Time_series_Y_test<=threshold_inst]=0
    Time_series_Y_test[Time_series_Y_test > threshold_inst]=1

    total_positive=np.count_nonzero(Time_series_Y_test == 1)
    total_negative =np.sum(Time_series_Y_test == 0)
    pred_pos=np.count_nonzero(pred_t == 1)
    pred_neg =np.sum(pred_t == 0)

    mat_1 = Time_series_Y_test + pred_t
    tp =np.sum(mat_1 == 2)
    tp_arr=np.sum(mat_1 == 2,0)
    tn=np.sum(mat_1 == 0)
    tn_arr=np.sum(mat_1 == 0,0)

    mat_2 = Time_series_Y_test - pred_t
    fn = np.sum(mat_2 == 1)
    fn_arr=np.sum(mat_2 == 1,0)
    fp =np.sum(mat_2 ==-1)
    fp_arr=np.sum(mat_2 == -1,0)

    #Writing Confusion Matrixs Test
    df_cm=pd.DataFrame(tp_arr.T)
    #df. rename(columns = {'old_col1':'new_col1', 'old_col2':'new_col2'}, inplace = True)
    df_cm.rename(columns={0:'True_Positive'},inplace='True')
    df_cm.insert(loc=1,column='True_Negative',value=tn_arr)
    df_cm.insert(loc=2,column='False_Positive',value=fp_arr)
    df_cm.insert(loc=3,column='False_Negative',value=fn_arr)
    lines =np.linspace(1, 120, num=120)
    lines_arr = np.tile(lines,24)
    df_cm.insert(loc=0,column='Line_No',value=lines_arr)
    hour=np.linspace(1, 24, num=24)
    hour_arr = np.repeat(hour,120)
    df_cm.insert(loc=1,column='Hour',value=hour_arr)
    file_name=path+'\confusion_matrix.xlsx'
    df_cm.to_excel(file_name)
    df_false_neg = df_cm.groupby(["Line_No"]).False_Negative.sum().reset_index()

    #Testing 
    print('testing')
    print('tp')
    print(tp)
    print('tn')
    print(tn)
    print('fp')
    print(fp)
    print('fn')
    print(fn)
    print('Most false neg line')
    print(df_false_neg.loc[df_false_neg.False_Negative.idxmax(),'False_Negative'])
    print("No of times")
    print(df_false_neg.loc[df_false_neg.False_Negative.idxmax(),'Line_No'])
    
    acc_test =(tp+tn)/(fp+fn+tp+tn)
    print('Test Accuracy')
    print(acc_test)
    
    predictions_train = LSTM_model.predict(Time_series_X_train)
    predictions_train[predictions_train<=threshold_inst]=0
    predictions_train[predictions_train > threshold_inst]=1
    Time_series_Y_train[Time_series_Y_train<=threshold_inst]=0
    Time_series_Y_train[Time_series_Y_train > threshold_inst]=1

    pred_train=predictions_train
    total_positive_train=np.count_nonzero(Time_series_Y_train == 1)
    total_negative_train =np.sum(Time_series_Y_train == 0)
    #pred_pos_train=np.count_nonzero(pred_train == 1)
    #pred_neg_train =np.sum(pred_train == 0)

    mat_1 = Time_series_Y_train + pred_train
    tp_train =np.sum(mat_1 == 2)
    tn_train=np.sum(mat_1 == 0)
   

    mat_2 = Time_series_Y_test - pred_t
    fn_train = np.sum(mat_2 == 1)
    fp_train =np.sum(mat_2 ==-1)
    
    acc_train=(tp_train+tn_train)/(tp_train+tn_train+fp_train+fn_train)
    
    df_performance =pd.DataFrame()
    df_performance['Testing Accuracy']=[acc_test]
    df_performance['TP_GT_testing']=[total_positive]
    df_performance['TN_GT_testing']=[total_negative]
    df_performance['tp_testing']= [tp]
    df_performance['tn_testing'] =[tn]
    df_performance['fp_testing']=[fp]
    df_performance['fn_testing'] =[fn]
    df_performance['Training_Accuracy']=[acc_train]
    df_performance['TP_GT_training']=[total_positive_train]
    df_performance['TN_GT_training']=[total_negative_train]
    df_performance['tp_training']=[tp_train]
    df_performance['tn_training']=[tn_train]
    df_performance['fp_training']=[fp_train]
    df_performance['fn_training'] =[fn_train]
    performance_metric_path=path+'\\Training_testing_acc.xlsx'  
    df_performance.to_excel(performance_metric_path)
    
    '''text_kwargs = dict(ha='center', va='center', fontsize=18, color='C1')
    s_test='Test Accuracy' +' ' + str(acc_test)
    plt.figure(figsize=(10, 2))
    plt.text(0.5, 0.5, s_test, **text_kwargs)
    plt.savefig(path+'\\Plots'+'\\Acc.png')
    plt.show()'''

    

    Y_percentage_test = np.load(r'C:\Users\farha\OneDrive\Desktop\TLS LSTM code\T_s_Y_test_flattened.npy')
        
    indices = np.where(mat_2 == 1)

    print(Y_percentage_test[indices])
    fn_per=Y_percentage_test[indices]

    np.save(path+'\False_Negative.npy',fn_per)
    plt.hist(fn_per)
    plt.savefig(path+'\\Plots'+'\\Histogram_False_Negative.png')
    plt.show() 
      
    indices = np.where(mat_2 == -1)
    fp_per=Y_percentage_test[indices]
    np.save(path+'\False_Positive.npy',fp_per)
    plt.hist(fp_per)
    plt.savefig(path+'\\Plots'+'\\Histogram_False_Positive.png')
    plt.show() 

    Errors_true_false =np.append(fn_per,fp_per)
    plt.hist(Errors_true_false)
    plt.savefig(path+'\\Plots'+'\\Histogram_Errors.png')
    plt.show() 
    
print('Done...!')

    