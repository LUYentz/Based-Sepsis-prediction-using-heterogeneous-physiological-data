import glob

import scipy.signal as sig
import numpy as np
import pandas as pd
import os
import random
import time
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical

def feature_normalization(X):
    #X = X.astype(float)
    scaler = StandardScaler().fit(X)
    data = scaler.transform(X)
    return data

def DC_cancellation(Signal):
    X = sig.detrend(Signal)
    DC_Signal = X - np.mean(X)
    DC_Signal = DC_Signal / max( abs(DC_Signal))
    return DC_Signal


def Filter(Signal,fs):
    L_cutoff = 5
    H_cutoff = 15
    N = 3
    Wn = [L_cutoff, H_cutoff] *2 /fs
    [b, a] = sig.butter(N, Wn, 'bandpass')
    F_signal = sig.filtfilt( b, a, Signal)
    F_signal = F_signal / max( abs(F_signal))
    return F_signal


def PrePocess(Signal,fs):
    DC_sig = DC_cancellation(Signal)
    F_sig = Filter(Signal,fs)
    return F_sig

def Data_pro1(data, Sepstr, Norsrt):
    w = len(data.II)
    data = data[data.II != 0]
    s = len(data.II)
    print(str(w) + '/' + str(s)) if w != s else []
    if Sepstr == -1 and Norsrt == -1:
        df = data.iloc[:,np.r_[3,5]]
        df = df[df.II != 0]
        df = df[df.Label != 0] #刪除不為敗血症

        label = df.Label
        df = df.iloc[:,np.r_[0]]
    else:
        df = data.iloc[:, np.r_[3]]
        label = data.iloc[:, np.r_[5]]

    df = df.dropna()
    normdf = feature_normalization(df)
        
    categorical_labels = to_categorical(label, num_classes=2)

    del label, data, df, w,s
    return normdf, categorical_labels

def Data_pro2(data, onehot_encoded, N_TIME_STEPS, step, N_FEATURES,classes):
    segments = []
    labels = []

    for i in range(0, len(data) - N_TIME_STEPS, step):
        II = data[ i : i + N_TIME_STEPS, 0]
        li = onehot_encoded[ i : i + N_TIME_STEPS]
        segments.append([II])
        labels.append(li)
    del II,li,i
    return segments, labels

def Save_Array(group ,path , size = 2048):
    dirs = glob.glob(path + "/*.csv")
    N_FEATURES = 1
    classes = 2
    onehot_encoded =[]
    norm_df = []
    print('output folder created') if os.path.isdir(group) else os.mkdir(group)
    
    for dir in dirs:
     # #for i in range(167,len(dirs)):
        ##dir = dirs[i]
        print('\n'+str(dirs.index(dir))+'/'+str(len(dirs))+" // "+dir)
        datas = pd.read_csv( dir, header= 0, sep = ',' )
        norm_df, onehot_encoded = Data_pro1(datas , dir.find('NonSepsis'),dir.find('Normal'))
        del datas
        
        segments, labels = Data_pro2(norm_df , onehot_encoded, size, size, N_FEATURES,classes)
        reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, 1, size, N_FEATURES)
        print('data shape')
        print(reshaped_segments.shape)
        reshaped_labels = np.asarray(labels, dtype= np.float32).reshape(-1, 1, size, classes)
        print('label shape')
        print(reshaped_labels.shape)
        
        y1 = np.load(group +'/'+str(size)+'_data.npy') if os.path.isfile(group +'/'+str(size)+'_data.npy') else []
        np.save(group +'/'+str(size)+'_data.npy', reshaped_segments) if y1 == [] else np.save(group +'/'+str(size)+'_data.npy', np.concatenate((y1,reshaped_segments),axis=0))
        y2 = np.load(group +'/'+str(size)+'_label.npy') if os.path.isfile(group +'/'+str(size)+'_label.npy') else []
        np.save(group +'/'+str(size)+'_label.npy', reshaped_labels) if y2 == [] else np.save(group +'/'+str(size)+'_label.npy', np.concatenate((y2,reshaped_labels),axis=0))
        print('accumulation data shape')
        print(y1.shape) if dir != dirs[0]  else print(str(len(y1)))

        del reshaped_labels,labels, y2, reshaped_segments, segments, y1, norm_df, onehot_encoded
    
    if dirs!=[]: 
        y1 = np.load(group +'/'+str(size)+'_data.npy') 
        print('\n\n total data shape')
        print(str(y1.shape))
    else:
        print('miss file')
    print('finish')


def random_path(group_path ,k):
    Train_fpathlist =[]
    Test_fpathlist = []
    folderlist =os.listdir(group_path)
    fp = open('Sepsis/dir_'+time.strftime("%m%d_%H%M",time.localtime())+'.txt','a')
    for folders in folderlist:
        n = folderlist.index(folders)                                   
        print(str(n+1)+'/'+str(len(folderlist)))
        fpaths = glob.glob(group_path+'/'+folders+'/*.csv')
        if n >= math.floor(len(folderlist)*0.75):
            Test_fpathlist.extend(random.sample(fpaths,k)) if len(fpaths) > k else print(folders+' lack')
            print('Test\t\t'+str(n),file=fp)
            print(Test_fpathlist,file=fp)
        else:
            Train_fpathlist.extend(random.sample(fpaths,k)) if len(fpaths) > k else print(folders+' lack')
            print('Train\t\t'+str(n),file=fp)
            print(Train_fpathlist,file=fp)
    fp.close()
    return Train_fpathlist,Test_fpathlist


def preprocess_data(group,dg,dirs,N_FEATURES,classes,size=2048):
    for dir in dirs:
        print('\n'+str(dirs.index(dir))+'/'+str(len(dirs))+" // "+dir)
        datas = pd.read_csv( dir, header= 0, sep = ',' )
        norm_df, onehot_encoded = Data_pro1(datas , dir.find('NonSepsis'),dir.find('Normal'))
        del datas
        
        segments, labels = Data_pro2(norm_df , onehot_encoded, size, size, N_FEATURES,classes)
        reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, 1, size, N_FEATURES)
        print('data shape')
        print(reshaped_segments.shape)
        reshaped_labels = np.asarray(labels, dtype= np.float32).reshape(-1, 1, size, classes)
        print('label shape')
        print(reshaped_labels.shape)
        
        data_path = group +'/'+dg+'_'+str(size)+'_data.npy'
        y1 = np.load(data_path) if os.path.isfile(data_path) else []
        np.save(data_path, reshaped_segments) if y1 == [] else np.save(data_path, 
            np.concatenate((y1,reshaped_segments),axis=0))
        
        label_path = group +'/'+dg+'_'+str(size)+'_label.npy'
        y2 = np.load(label_path) if os.path.isfile(label_path) else []
        np.save(label_path, reshaped_labels) if y2 == [] else np.save(label_path, 
            np.concatenate((y2,reshaped_labels),axis=0))
        
        print('accumulation data shape')
        print(y1.shape) if dir != dirs[0]  else print(str(len(y1)))

        del reshaped_labels,labels, y2, reshaped_segments, segments, y1, norm_df, onehot_encoded,data_path,label_path
    
    if dirs!=[]: 
        y1 = np.load(group +'/'+dg+'_'+str(size)+'_data.npy') 
        print('\n\n total data shape')
        print(str(y1.shape))
    else:
        print('miss file')    


def Save_Array_bypatient(group ,group_dir,nf , size = 2048):

    Train_dirs,Test_dirs = random_path(group_dir,nf)
    N_FEATURES = 1
    classes = 2
    onehot_encoded =[]
    norm_df = []
    print('output folder created') if os.path.isdir(group) else os.mkdir(group)
    preprocess_data(group,'Train',Train_dirs,N_FEATURES,classes,size)
    preprocess_data(group,'Test',Test_dirs,N_FEATURES,classes,size)
        
    print('finish')
