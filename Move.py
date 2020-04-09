import os
import shutil
import glob
import pandas as pd
import Data_Label as dl
import PreProcess as pp


columns = ["Date","Time(ms)","II","RESP"]

def mkdir_folder(foldername, group, disk ='C://',GA = 23):
    fname = disk +'BC case/NoLabel/'+group+'/'+str(GA)+'wk_'+foldername
    os.mkdir(fname) if not os.path.isdir(fname) else print('folder created')
    
    return fname


def fname_time(BCtime):
    BC_time = str(BCtime)
    if BC_time.find('/10/') !=-1 or  BC_time.find('/11/') !=-1 or BC_time.find('/12/') != -1:
        fname_by_time = '0'+ BC_time[5] + BC_time[7:9] + BC_time[2:4] +'_'+ BC_time[11:13]+BC_time[15:17]
        
    else:
        fname_by_time = BC_time[5:7] + BC_time[8:10] + BC_time[2:4] +'_'+ BC_time[12:14]+BC_time[16:18]
    print(fname_by_time)
    
    return fname_by_time

def copy_file(file_name ,ftype ,folder_name):
    f_name = file_name + '.' + ftype
    if not os.path.isfile(folder_name +'/' + f_name[-20:]):
        shutil.copy(f_name, folder_name)
    else:
        print( f_name + ' existed')

"""" .txt to .csv and save to new folder  """
def txt_procrss(file_path, file_name, save_folder):
    opfile = save_folder+'/'+file_name+'.csv'
    txt_path = file_path+'/'+file_name
    l_folder = save_folder.replace('NoLabel','Label')

    oplfile = l_folder +'/'+file_name+'.csv' 
    
    if  (not os.path.isfile(opfile)) or (not os.path.isfile(oplfile)):
        raw = pd.read_csv(txt_path+ '.txt', header = 0,index_col=False, names = columns, low_memory=True )
        if raw.empty == 0:
            raw.dropna(inplace=True)
            new = raw["Date"].str.split(" ", n = 1, expand = True)
            raw["Date"] = new[0]
            raw["Time(min)"] = new[1]
            raw = raw[["Date","Time(min)","Time(ms)","II","RESP"]]
            raw.to_csv(opfile, index = None, header=True) if not os.path.isfile(opfile) else print("Nolabel data existed")
        
            """ Label data """
            os.mkdir(l_folder) if not os.path.isdir(l_folder) else print('folder created')
            if not os.path.isfile(oplfile): 
                New_data, Class = dl.Label(raw, '', opfile)
                print('Label folder created') if os.path.isdir(l_folder[:-14]) else os.mkdir(l_folder[:-14])
                New_data.to_csv(oplfile, sep=',', header=True, index=False )
            else:
                print("Nolabe data existed") 
    else:
        print(".txt and .PDI existed")


def BCdata(excel_path,group):
    df = pd.read_excelread_excel(io=excel_path,header=0)
    L = len(df)

    for n in range(0,L):
        patientNo = str(df['chart no'][n])
        if df.GA < 27:
            file_path = 'F:/'+str(df['GA'][n])+'wk/'+ patientNo
        else:
            file_path = 'L:/'+str(df['GA'][n])+'wk/'+ patientNo 

        f_list = glob.glob(file_path + '.txt')
        f_list.sort()
        Lf = len(f_list)

        if len(f_list)!=0:
            print(str(n)+ '//' + patientNo+':processing...')
            for nf in range(0, Lf):
                file_s = str(df['S file name'][n])
                file_e = str(df['E file name'][n])
                filename_s = f_list[nf]
                if  filename_s[-20:-4] == file_s:
                    print('copy and move')
                    folder = mkdir_folder(patientNo+'_'+fname_time(df['BC time'][n]), group,'M://', df['GA'][n])

                    for i in range(0,104):
                        filename  = f_list[nf +i]
                        copy_file(filename[-20:-4], 'PDI',folder)     #copy PDI
                        copy_file(filename[-20:-4], 'RFA',folder)     #copy RFA
                        copy_file(filename[-20:-4], 'RFP',folder)     #copy RFP
                        copy_file(filename[-20:-4], 'RFV',folder)     #copy RFV
                        txt_procrss(folder, filename[-20:-4], folder)
                        if filename[-20:]==(file_e + '.txt'):
                            print(patientNo+str(n)+'finish')
                            break
        else:
            print(patientNo+'_'+fname_time(df['BC time'][n])+'is missing')

    print("finish")

def Noramldata(excel_path):
    df = pd.read_excel(io=excel_path,header=0)
    L = len(df)
    os.mkdir('M://BC case/NoLabel/Normal') if not os.path.isdir('M://BC case/NoLabel/Normal') else print('unlabel Group folder created')
    os.mkdir('M://BC case/Label/Normal') if not os.path.isdir('M://BC case/Label/Normal') else print('label Group folder created')
    for n in range(0,L):
        patientNo = str(df['chart no'][n])

        file_path = 'L:/Special/No_use_rowdata/Just_BD_BC/'+str(df['GA'][n])+'wk/'+ patientNo +'/'
        
        Savefolder = mkdir_folder(patientNo,"Normal","M://",df['GA'][n])

        filename = file_path + df['file name'][n]
        print(str(n) +'//'+filename+':processing...')
        copy_file(filename, 'PDI',Savefolder)     #copy PDI
        copy_file(filename, 'RFA',Savefolder)     #copy RFA
        copy_file(filename, 'RFP',Savefolder)     #copy RFP
        copy_file(filename, 'RFV',Savefolder)     #copy RFV

        """" .txt to .csv and save to new folder  """

        txt_procrss(file_path, df['file name'][n], Savefolder )

       
        print(filename+ ':finsh')

    print("finish")
