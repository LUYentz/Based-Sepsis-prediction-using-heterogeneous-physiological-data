import numpy as np
import os
import Data_Label as dlabel
import Data_Load as dload
import PreProcess as PreP
import Move


def conNorSep(subsep,N ,outfolder):
    print('concatenate data:Normal and Sepsis')
    print('output folder created') if os.path.isdir(outfolder) else os.mkdir(outfolder)
    if os.path.isfile('Normal/Train_'+str(subsep)+'_data.npy') and os.path.isfile('Sepsis/Train_'+str(subsep)+'_data.npy'):
        
        data1 = np.load('Normal/Test_'+str(subsep)+'_data.npy')
        data2 = np.load('Sepsis/Test_'+str(subsep)+'_data.npy')
        for i in range(0,N,100):
            con_data = np.concatenate((data1[i:(i+100),:,:,:],data2[i:(i+100),:,:,:]),axis=0)
            total_data = np.concatenate((total_data, con_data), axis=0) if i != 0 else con_data
            print(total_data.shape)

        del data1,data2,con_data,    
        np.save(outfolder+'/Test_'+str(subsep)+'_Normal_Sepsis_data_'+str(int(N/10000))+'W.npy', total_data)
        del total_data
        label1 = np.load('Normal/Test_'+str(subsep)+'_label.npy')
        label2 = np.load('Sepsis/Test_'+str(subsep)+'_label.npy')
        for i in range(0,N,100):
            con_label = np.concatenate((label1[i:(i+100),:,:,:],label2[i:(i+100),:,:,:]),axis=0)
            total_label = np.concatenate((total_label, con_label), axis=0) if i != 0 else con_label
            print(total_label.shape)

        del label1,label2,con_label,i
        np.save(outfolder+'/Test_'+str(subsep)+'_Normal_Sepsis_label_'+str(int(N/10000))+'W.npy', total_label)
        print('\n total label shape')
        print(total_label.shape)
        del total_label
        
    else:
        print('files does not exist')
    print('finish')

#Move.Noramldata('Data pre/Normal_Move.xlsx')
#PreP.Save_Array('Normal','M:/BC case/Label/Normal_choose/*')
#dlabel.Label_run("M:/BC case/NoLabel/PBC/*/*.csv")
#dlabel.Label_run("M:/BC case/NoLabel/NBC/*/*.csv")
#dlabel.Label_run("M:/BC case/NoLabel/NBC_hisPBC/*/*.csv")
#PreP.Save_Array('Normal','M:/BC case/Label/Normal/*',size = 4096)   # 4096/125=32.8
#PreP.Save_Array('Sepsis','M:/BC case/Label/Sepsis',size = 4096)   # 4096/125=32.8
#dlabel.Label_run("M:/BC case/NoLabel/PBC/*/*.csv")
#PreP.Save_Array('Normal','M:/BC case/Label/Normal_random/*',size=4096)
#conNorSep(4096,100000,'Data/SepA1B1_NorALL')
#Move.Noramldata('Data pre/Normal_Move_randomAll.xlsx')
#PreP.Save_Array('Sepsis','M:/BC case/Label/Sepsis-A2B2',size = 4096)   # 4096/125=32.8
#conNorSep(4096,100000,'Data/SepA2B2_NorALL')

#4096*2/125=65.6s
#PreP.Save_Array('Sepsis','M:/BC case/Label/Sepsis-A2B2/',size = 4096*2)
#PreP.Save_Array('Normal','M:/BC case/Label/Normal_randomall/*',size=4096*2)
#conNorSep(4096*2,50000,'Data/SepA2B2_NorALL')


#37632/(125*60)=5.01min

# random_patient
#PreP.Save_Array_bypatient('Sepsis','M:/BC case/Label/Sepsis-A2B2',7,size = 4096)  #Sepsis 21 person/training:16 
#PreP.Save_Array_bypatient('Normal','M:/BC case/Label/Normal_randomall',3 ,size = 4096)  #normal 40person/train:30/test:10
conNorSep(4096,25200,'Data/V1_SepA1B1ran_NorALL')
#X_train, X_test, Y_train, Y_test, N_FEATURES, classes = dload.load_data('SepsisDataset_random',2048)