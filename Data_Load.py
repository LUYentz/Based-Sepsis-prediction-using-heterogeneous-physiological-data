import numpy as np
import pandas as pd
import PreProcess as pp
import h5py
import os
import glob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

RANDOM_SEED = 42 # 原本設定

# 資料集中化
def feature_normalization(x):
    scaler = StandardScaler().fit()
    data = scaler.transform(x)
    return data

# 
def load_data(data_name = 'WISDMar', subsep =224):
    if data_name == 'SepsisDataset':
        X_Train, X_Test, Y_Train, Y_Test, N_FEATURES, Class = load_SepSet(subsep)
    elif data_name == 'LSNet_SepsisDataset':
        X_Train, X_Test, Y_Train, Y_Test, N_FEATURES, Class = load_LSTNet_SepSet(subsep)
    elif data_name == 'SepsisDataset_random':
        X_Train, X_Test, Y_Train, Y_Test, N_FEATURES, Class = load_SepSet_random(subsep)
    else:
        X_Train, X_Test, Y_Train, Y_Test, N_FEATURES, Class = load_SepSet(subsep)
    return X_Train, X_Test, Y_Train, Y_Test, N_FEATURES, Class
   

def load_SepSet(subsep):
    
    N_TIME_STEPS = subsep
    N_FEATURES = 1
    step = subsep
    Sep_classes = 2

    #pp.Save_Array('M:/BC case/Label/Sepsis', subsep)
    #pp.Save_Array('M:/BC case/Label/NonSepsis', subsep)
    reshaped_segments = np.load('Data/SepA1B1-NorCho/2048_Normal_Sepsis_data_3W.npy')
    reshaped_labels = np.load('Data/SepA1B1-NorCho/2048_Normal_Sepsis_label_3W.npy')
    print('SepsisA2B2_NorAllday')

    print(reshaped_segments.shape)
    
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(reshaped_segments, reshaped_labels, test_size = 0.1, random_state = RANDOM_SEED, shuffle = True)
    
    return X_Train, X_Test, Y_Train, Y_Test, N_FEATURES, Sep_classes

def load_LSTNet_SepSet(subsep):

    N_TIME_STEPS = subsep
    N_FEATURES = 1
    step = subsep
    Sep_classes = 2
    
    #pp.Save_Array('M:/BC case/Label/Sepsis')
    #pp.Save_Array('M:/BC case/Label/NonSepsis')

    S_reshaped_segments = np.load('data.npy')
    S_reshaped_labels = np.load('label.npy')
    NS_reshaped_segments = np.load('data.npy')
    NS_reshaped_labels = np.load('label.npy')
    
    S_X_Train, S_X_Test, S_Y_Train, S_Y_Test = train_test_split(S_reshaped_segments, S_reshaped_labels, test_size = 0.3, random_state = RANDOM_SEED)
    NS_X_Train, NS_X_Test, NS_Y_Train, NS_Y_Test = train_test_split(NS_reshaped_segments, NS_reshaped_labels, test_size = 0.3, random_state = RANDOM_SEED)
    
    X_Train = np.concatenate(( S_X_Train, NS_X_Train),axis=0)
    X_Test = np.concatenate(( S_X_Test, NS_X_Test), axis =0)
    Y_Train = np.concatenate((S_Y_Train, NS_Y_Train),axis = 0)
    Y_Test = np.concatenate((S_Y_Test, NS_Y_Test), axis = 0)
    np_sengments = np.concatenate((S_reshaped_segments, NS_reshaped_segments), axis = 0)
    np_labels = np.concatenate((S_reshaped_labels, NS_reshaped_labels), axis = 0)

    return X_Train, X_Test, Y_Train, Y_Test, N_FEATURES, Sep_classes

def load_SepSet_random(subsep):
    print('radom')
    
    N_TIME_STEPS = subsep
    N_FEATURES = 1
    step = subsep
    Sep_classes = 2

    X_Train = np.load('Data/V1_SepA1B1ran_NorALL/Train_4096_Normal_Sepsis_data_7W.npy')
    Y_Train = np.load('Data/V1_SepA1B1ran_NorALL/Train_4096_Normal_Sepsis_label_7W.npy')
    X_Train, Y_Train = shuffle(X_Train, Y_Train)
    print(X_Train.shape)
    X_Test = np.load('Data/V1_SepA1B1ran_NorALL/Test_4096_Normal_Sepsis_data_2W.npy')
    Y_Test = np.load('Data/V1_SepA1B1ran_NorALL/Test_4096_Normal_Sepsis_label_2W.npy')
    X_Test, Y_Test = shuffle(X_Test, Y_Test)
    print(X_Test.shape)
    print('SepsisA2B2_NorAllday')
    
    return X_Train, X_Test, Y_Train, Y_Test, N_FEATURES, Sep_classes