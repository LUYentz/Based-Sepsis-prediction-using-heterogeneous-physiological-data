# encoding= utf8
import numpy as np
import tensorflow as tf
import PreProcess as PrePro
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
# from kera.optimizers import CSVLoggerpip
from keras.optimizers import Adam, SGD,RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
#
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.decomposition import PCA
from keras.utils import np_utils
from sklearn.manifold import TSNE
# other moudle
import common
import pandas as pd
import logging
import argparse
import time
import segnet
import os
import sklearn
import math
import openpyxl
# imprort unet_224_moel
import unet_info
import maskrcnn
import unet_model
import Data_Load


def plot_TSNE(features,label_data,z,data_type,path):
    pca = PCA(n_components=2)# 总的类别
    nsamples, nx, ny, nz= features.shape 
    x = features.reshape([nsamples*nx*ny,nz])
    pca_result = pca.fit_transform(x)
    print('Variance PCA: {}'.format(np.sum(pca.explained_variance_ratio_)))

    #Run T-SNE on the PCA features.
    tsne = TSNE(n_components=2, verbose = 1)
    tsne_results = tsne.fit_transform(pca_result[:1000000])
    #-------------------------------可视化--------------------------------
    
    #--- test data
    y = label_data.reshape([nsamples,nx*ny,nz])
    y_test_cat = np_utils.to_categorical(y[:1000000], num_classes = 2)# 总的类别
    color_map = np.argmax(y_test_cat, axis=1)
    plt.figure(figsize=(10,10))
    for cl in range(2):# 总的类别
        indices = np.where(color_map==cl)
        indices = indices[0]
        plt.scatter(tsne_results[indices,0], tsne_results[indices, 1], label=cl)
    plt.legend()
    plt.title(str(z)+'_'+data_type+'_T-sne')
    plt.show()
    plt.savefig(path+"/"+str(z)+'_'+data_type+'_T-sne.png')

# GPU
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
unet_info.begin()

# data load
subseq = 2048
dbName  = "SepsisDataset"
print("Load data")
X_train, X_test, Y_train, Y_test, N_FEATURES, classes = Data_Load.load_data(dbName,2048)
#get dense prediction results with overlap(transformed win data label's ground truth)
# y_test_resh = y_test.reshape(y_test.shape[0], y_test.shape[2], -1)
# y_test_resh_argmax = np.argmax(y_test_resh, axis=2)
# labels_test_unary = y_test_resh_argmax.reshape(y_test_resh_argmax.size)
# file_labels_test_unary = 'labels_gd_'+args.dataset+'_'+str(subseq)+'0317.npy'
# np.save(file_labels_test_unary,labels_test_unary)


# Setting
## 10-flod cross validation###
epochs = 1 
batch_size = 64
optim_type = 'adam'
learning_rate = 0.02
sum_time = 0
net = 'unet'
block = '5'
ksize = 10
pice = int(X_train.shape[0]/ ksize)

## output folder and parameter
nowtime = time.strftime("%m%d_%H%M",time.localtime())
s_path = 'Result/'+str(subseq)
print('Patient subsep Floder created') if os.path.isdir(s_path) else os.mkdir(s_path)
rf_path = s_path + '/'+nowtime + '_'+ str(subseq)
print('Patient Floder created') if os.path.isdir(rf_path) else os.mkdir(rf_path)
valcvscores = []
testcvscores = []
print('k-fold Floder created') if os.path.isdir(rf_path +'/k_flod') else os.mkdir(rf_path+'/k_flod')

# 10-floder
for k in range(ksize):
    fp = open(rf_path+'/Result_'+nowtime+'.txt','a')
    trainX = X_train[np.r_[0:k*pice,(k+1)*pice:X_train.shape[0]]]
    trainY = Y_train[np.r_[0:k*pice,(k+1)*pice:X_train.shape[0]]]
    valX = X_train[np.r_[k*pice:(k+1)*pice]]
    valY = Y_train[np.r_[k*pice:(k+1)*pice]]
    print("\n",file=fp)
    print(str(k)+" Running...")
    print("\n",file=fp)
    print(str(k)+" Running...",file=fp)
    # create model
    Kmodel = unet_model.ZF_UNET_224(subseq=subseq, filters=32, INPUT_CHANNELS=N_FEATURES, OUTPUT_MASK_CHANNELS=classes)
    
    # Compile model
    optim = Adam(lr=learning_rate)
    Kmodel.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    
    # checkpoint
    lr_reducer = ReduceLROnPlateau(monitor='val_loss',factor=np.sqrt(0.1),cooldown=0,patience=10, min_lr=1e-12)
    filepath=rf_path+'/k_flod/'+str(k)+'_'+nowtime+'_'+"weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss',mode='min',verbose=1, save_best_only=True)   
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='auto',baseline=None)
    # save best weight model
    callbacks = [lr_reducer, early_stopper, checkpoint]
    
    # Fit the model
    khistor =Kmodel.fit(trainX, trainY,validation_data=(valX, valY),batch_size=batch_size, epochs=epochs, callbacks=callbacks,shuffle=True)

    #-------------------------------获取模型最后一层的数据--------------------------------
    trucated_model=unet_model.create_truncate_model(Kmodel,subseq=subseq,filters=32,INPUT_CHANNELS=N_FEATURES,OUTPUT_MASK_CHANNELS=classes)
    val_hidden_features = trucated_model.predict(valX)
    plot_TSNE(val_hidden_features,valY,k,'validation_set',rf_path)
    del val_hidden_features
    #-------------------------------PCA,tSNE降维分析-------------------------------------
    plot_TSNE(test_hidden_features,Y_test,k,'test_set',rf_path)
    test_hidden_features = trucated_model.predict(X_test)
    del test_hidden_features

    # evaluate the model
    valKscores = Kmodel.evaluate(valX, valY, verbose=0)
    print("%s: %.2f%%" % (Kmodel.metrics_names[1], valKscores[1]*100), file=fp)
    print("%s: %.2f%%" % (Kmodel.metrics_names[0], valKscores[0]), file=fp)
    valcvscores.append(valKscores)

    testKscores = Kmodel.evaluate(X_test, Y_test, batch_size=batch_size)
    print("test lost:" + str(testKscores[0]) + "   test accuracy:" + str(testKscores[1]) + '\n',file=fp)
    valcvscores.append(testKscores)

    # flod save
    file_model = rf_path+'/k_flod/'+str(k)+'_Model_'+nowtime + '_'+ str(subseq) +'_' + str(batch_size) + '_' + optim_type + '.h5py'
    Kmodel.save(file_model)
    file_weight = rf_path+'/k_flod/'+str(k)+'_Weight_'+nowtime + '_'+ str(subseq) +'_' + str(batch_size) + '_' + optim_type + '.h5py'
    Kmodel.save_weights(file_weight)
    from openpyxl import load_workbook
    # convert the history.history dict to a pandas DataFrame:     
    khist_df = pd.DataFrame(khistor.history) 
    # save to csv: 
    if os.path.isfile(rf_path+'/'+nowtime+'_history.xlsx'):
        book = load_workbook(rf_path+'/'+nowtime+'_history.xlsx')     
        writer = pd.ExcelWriter(rf_path+'/'+nowtime+'_history.xlsx', engine='openpyxl')
        writer.book = book
        writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    else:
        writer = pd.ExcelWriter(rf_path+'/'+nowtime+'_history.xlsx')

    khist_df.to_excel(writer,str(k)+'_flod')
    writer.save()
    fp.close()
    
    del trainX,trainY, valX, valY, Kmodel,khist_df


fp = open(rf_path+'/Result_'+nowtime+'.txt','a')
print("\n",file=fp)
print("acc",file=fp)
print("mean:"+str(np.mean(valcvscores[1,:])*100),file=fp)
print("std:"+str(np.std(valcvscores[1,:])),file=fp)
print("loss",file=fp)
print("mean:"+str(np.mean(valcvscores[0,:])),file=fp)
print("std:"+str(np.std(valcvscores[0,:])),file=fp)

print("\n",file=fp)
print("test",file=fp)
print("mean:"+str(np.mean(testcvscores[1,:])*100),file=fp)
print("std:"+str(np.std(testcvscores[1,:])),file=fp)
print("loss",file=fp)
print("mean:"+str(np.mean(testcvscores[0,:])),file=fp)
print("std:"+str(np.std(testcvscores[0,:])),file=fp)


# convert the history.history dict to a pandas DataFrame:     
cvsval_df = pd.DataFrame(valcvscores,columns=['acc','loss']) 
cvstest_df = pd.DataFrame(testcvscores,columns=['acc','loss']) 
# or save to csv:
writercvscore  = pd.ExcelWriter(rf_path+'/'+nowtime+'_cvscore.xlsx')
cvsval_df.to_excel(writercvscore,'val')
cvstest_df.to_excel(writercvscore,'test')
fp.close()
writer.close()                              
writercvscore.close()
print('Train model finish')



'''
# create model
if (net == 'unet')and(block == '5'):
    model = unet_model.ZF_UNET_224(subseq=subseq, filters=32, INPUT_CHANNELS=N_FEATURES, OUTPUT_MASK_CHANNELS=classes)
elif (net == 'unet')and(block == '4'):
    model = unet_model.ZF_UNET_224_4(subseq=subseq, filters=32, INPUT_CHANNELS=N_FEATURES, OUTPUT_MASK_CHANNELS=classes)
elif (net == 'unet')and(block == '3'):
    model = unet_model.ZF_UNET_224_3(subseq=subseq, filters=32, INPUT_CHANNELS=N_FEATURES,
                                     OUTPUT_MASK_CHANNELS=classes)
elif (net == 'unet')and(block == '2'):
    model = unet_model.ZF_UNET_224_2(subseq=subseq, filters=32, INPUT_CHANNELS=N_FEATURES,
                                     OUTPUT_MASK_CHANNELS=classes)
elif (net == 'segnet')and(block == '5'):
    model = segnet.segnet(subseq=subseq, INPUT_CHANNELS=N_FEATURES, filters=64, n_labels=classes, kernel=3,
                          pool_size=(1, 2))
elif net == 'fcn':
    model = unet_model.FCN(inputsize=subseq, deconv_output_size=subseq, INPUT_CHANNELS=N_FEATURES,
                           num_classes=classes)
elif net == 'maskrcnn':
    model = maskrcnn.Mask(subseq=28, INPUT_CHANNELS=N_FEATURES, filters=32, n_labels=classes, kernel=3)

# model = segnet.segnet(subseq=subseq, INPUT_CHANNELS=N_FEATURES,filters=64, n_labels = act_classes, kernel=3, pool_size=(1, 2))
# model = segnet.segnet4(subseq=subseq, INPUT_CHANNELS=N_FEATURES,filters=64, n_labels = act_classes, kernel=3, pool_size=(1, 2))
# model = segnet.segnet3(subseq=subseq, INPUT_CHANNELS=N_FEATURES,filters=64, n_labels = act_classes, kernel=3, pool_size=(1, 2))
# model = segnet.segnet2(subseq=subseq, INPUT_CHANNELS=N_FEATURES,filters=64, n_labels = act_classes, kernel=3, pool_size=(1, 2))

# model = maskrcnn.Mask(subseq=28, INPUT_CHANNELS=N_FEATURES, filters=32, n_labels=act_classes,kernel=3)

# model = unet_model.FCN(inputsize=subseq,deconv_output_size=subseq,INPUT_CHANNELS=N_FEATURES,num_classes=act_classes)
# model = unet_model.ZF_UNET_224_3(subseq=subseq,filters=32, INPUT_CHANNELS=N_FEATURES, OUTPUT_MASK_CHANNELS=act_classes)


# optimer setting
if optim_type == 'SGD':
    optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
else:
    optim = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])



# setting train parameter
lr_reducer = ReduceLROnPlateau(monitor='val_loss',factor=np.sqrt(0.1),cooldown=0,patience=10, min_lr=1e-12) #learing rate


callbacks = [lr_reducer]

# early_stopper = EarlyStopping(monitor='val_loss',
#                             patience=30)
#
# callbacks = [lr_reducer, early_stopper]

history=model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.3, callbacks=callbacks)
acc=np.array(history.history['acc'])
loss=np.array(history.history['loss'])
val_acc = np.array(history.history['val_acc'])
val_loss = np.array(history.history['val_loss'])
for i in range(acc.shape[0]):
    logging.info('Epoch: {} loss: {:.4f} accuracy{:.4f} val_loss: {:.4f} val_accuracy{:.4f}%\n'.format(i+1, loss[i], acc[i],val_loss[i],val_acc[i]))

# Polt
plt.figure(0)
plt.plot(val_acc)
plt.title('U net Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['acc_train', 'acc_test'], loc='upper left')
plt.savefig('2019'+dbName+'_U Net_model_acc.png')
plt.show()


plt.figure(30)
plt.plot(loss)
plt.plot(val_loss)
plt.title('U net Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['loss_train','loss_test'], loc='upper left')
plt.savefig('2019'+dbName+'_U Net_model_loss.png')
plt.show()
''''''

start = time.clock()
testloss, testaccuracy = model.evaluate(X_test, Y_test, batch_size=batch_size)
end = time.clock()
sum_time += (end - start)
print(str(end - start))
print("test lost:" + str(testloss) + "   test accuracy:" + str(testaccuracy))
logging.info('mean_time={}'.format(str(end - start)))

y_test_resh = Y_test.reshape(Y_test.shape[0], Y_test.shape[2], -1)
y_test_resh_argmax = np.argmax(y_test_resh, axis=2)
labels_test_unary = y_test_resh_argmax.reshape(y_test_resh_argmax.size)
file_labels_test_unary = 'labels_gd_'+dbName+'_'+str(subseq)+'_'+ net+ str(block)+'_0503.npy'
np.save(file_labels_test_unary,labels_test_unary)

y_pred_raw = model.predict(X_test, batch_size=batch_size)
y_pred_resh = y_pred_raw.reshape(y_pred_raw.shape[0], y_pred_raw.shape[2], -1)
y_pred_resh_argmax = np.argmax(y_pred_resh, axis=2)
y_pred = y_pred_resh_argmax.reshape(y_pred_resh_argmax.size)
y_pred_prob = y_pred_resh.reshape(y_pred_resh_argmax.size,y_pred_resh.shape[2])
print("---------------------------")
print(y_pred_prob.shape)
file_y_pred = 'y_pred_'+dbName+'_'+str(subseq)+'_'+net+'_'+block+'_'+optim_type+'.npy'
np.save(file_y_pred,y_pred)
file_y_pred_prob = 'y_pred_prob_'+dbName+'_'+str(subseq)+'_'+net+'_'+block+'_'+optim_type+'.npy'
np.save(file_y_pred_prob,y_pred_prob)


print("Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(labels_test_unary,y_pred))
print("Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(labels_test_unary,y_pred))
print("Root mean squared error (RMSEㄣ): %f" % math.sqrt(sklearn.metrics.mean_squared_error(labels_test_unary,y_pred)))
print("R square (R^2):                 %f" % sklearn.metrics.r2_score(labels_test_unary,y_pred))
'''