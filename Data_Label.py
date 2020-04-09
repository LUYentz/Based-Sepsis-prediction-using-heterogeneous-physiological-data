import pandas as pd
import numpy as np
import os
import glob
import datetime as dt



def getBCTime(filename):
    dateStr = '20'+ filename[-28:-26]+ '_'+filename[-32:-30]+ '_'+ filename[-30:-28]+ '_' \
        +filename[-25:-23]+ '_'+filename[-23:-21]
    BCTime = dt.datetime.strptime(dateStr,"%Y_%m_%d_%H_%M")
    del dateStr, filename
    return BCTime

def Label(df, BCtime, fname):
    df['Label'] = np.zeros(len(df.II), dtype=int)
    print(df.head(2))
    
    if fname.find("PBC") == 0 :
        S_time = BCtime - dt.timedelta(days =3.5)
        E_time = BCtime + dt.timedelta(days =2.5)
        df.loc[(df.Time > S_time) & (df.Time < E_time), 'Label'] = 1
        Group = 'Sepsis' if 1 in df.Label.values else 'Sepsis_non'
        
    elif fname.find('NBC') != -1 :
        Group = 'NonSepsis_non' if fname.find('PBC') != -1 else 'NonSepsis'

    if fname.find('Normal') == -1: 
        df = df.iloc[:,np.r_[0:5,6]]
    else:
        Group ='Normal'

    print(df.tail(2))
    print(Group)
    
    return df, Group

def BC_process(df, BCtime):
    e_chotime = BCtime + dt.timedelta(days =2.5)
    s_chotime = BCtime - dt.timedelta(days =3.5)
    df['Time'] = pd.to_datetime((df['Date']+'-'+df['Time(min)']),format="%Y-%m-%d-%H:%M:%S")
    df = df[(df.Time > s_chotime) & (df.Time < e_chotime)]

    del BCtime, e_chotime, s_chotime
    return df

def Label_run(fpath):
    folder = fpath
    f_list = glob.glob(folder)
    f_list.sort()

    for filename in f_list:
        print(str(f_list.index(filename)) + '//'+filename + " is processing...")
        if os.path.isfile('M:/BC case/Label/Sepsis/'+filename[-20:-4]+".csv") or os.path.isfile('M:/BC case/Label/NonSepsis/'+filename[-20:-4]+'.csv') \
            or os.path.isfile('M:/BC case/Label/NonSepsis_non/'+filename[-41:-4]+".csv") or os.path.isfile('M:/BC case/Label/Sepsis_non/'+filename[-20:-4]+".csv")\
            or os.path.isfile('M:/BC case/Label/Normal/'+filename[-20:-4]+".csv")    :
            print(filename[-20:-4]+ "已存在")
        else:
            data = pd.read_csv(filename, header= 0, sep=',')
            if filename.find("Normal") == -1:
                BC_time = getBCTime(filename) 
                New_data = BC_process(data, BC_time) if filename.find('Normal') == -1 else  data
                New_data, Class = Label(New_data, BC_time, filename[19:29])
            else:
                New_data = data
                New_data, Class = Label(New_data, '', filename[19:29])
            
            path = 'M:/BC case/Label/'+ Class
            print('Patient folder created') if os.path.isdir(path) else os.mkdir(path)
            New_data.to_csv(path+'/'+filename[-20:],sep=',',header=True, index=False)
            del data, New_data
    print("Finish")

