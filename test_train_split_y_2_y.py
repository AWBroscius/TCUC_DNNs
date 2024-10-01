# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 23:16:51 2022

@author: farhan
"""

import pandas as pd
import numpy as np
import random


#df_train_X = pd.read_excel('Xdata_norm.xlsx') # not req for y2y training 
df_train_y = pd.read_excel('Y_norm.xlsx') # Normalized line flows 
y_data = np.array(df_train_y)
#x_data = np.array(df_train_X)

chunk_size =20*24 # testing size
counter_for_block= 90*24 # quarter size 

c_s =20
c_f_b=90
random.seed(0)

t_1=random.randint(1, c_f_b- c_s) # training Q1
t_2=random.randint(c_f_b+1, c_f_b*2-c_s) # training Q2
t_3=random.randint(c_f_b*2+1, c_f_b*3 -c_s) # training Q3
t_4=random.randint(c_f_b*3+1, c_f_b*4 -c_s) # training Q4

test_1=t_1*24
test_2=t_2*24
test_3=t_3*24
test_4=t_4*24

print(test_1,'\n',test_2,'\n',test_3,'\n',test_4,'\n')
print(t_1,'\n',t_2,'\n',t_3,'\n',t_4,'\n')



y_test=[]
for i in range(test_1-3*24,test_1+chunk_size):
    y_test.append(y_data[i,:])
    
for i in range(test_2-3*24,test_2+chunk_size):
    y_test.append(y_data[i,:])
    
for i in range(test_3-3*24,test_3+chunk_size):
    y_test.append(y_data[i,:])
    
for i in range(test_4-3*24,test_4+chunk_size):
    y_test.append(y_data[i,:])

y_train =[]
for i in range (0, len(y_data)):
    if i in range(test_1,test_1+chunk_size):
        continue 
    elif i in range(test_2,test_2+chunk_size):
        continue
    elif i in range(test_3,test_3+chunk_size):
        continue
    elif i in range(test_4,test_4+chunk_size):
        continue
    else:
        y_train.append (y_data[i,:])
        
y_train =np.array(y_train)
y_test=np.array(y_test)

print(y_test.shape)
print(y_train.shape)

np.save('y_test_with_index.npy',y_test)
np.save('y_train_with_index.npy',y_train)

y_train_noindex=y_train[:,2:]
y_test_noindex=y_test[:,2:]

print(y_test_noindex.shape)
print(y_train_noindex.shape)

np.save('y_test_with_no_index.npy',y_test_noindex)
np.save('y_train_with_no_index.npy',y_train_noindex)


'''
O/P :
    1200 
     3456 
     4464 
     7296 

    50 
     144 
     186 
     304 

    (2208, 122)
    (6720, 122)
    (2208, 120)
    (6720, 120)'''