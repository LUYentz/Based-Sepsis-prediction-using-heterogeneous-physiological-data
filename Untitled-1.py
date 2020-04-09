# -*- coding: utf-8 -*-
"""
Visualizing the hidden Layer in Neural Networks @ MNIST dataset
Created on Sat Dec 28 19:49:26 2019
@author: liujie
# Credit:
# Visualizing Layer Representations in Neural Networks
# https://becominghuman.ai/visualizing-representations-bd9b62447e38
"""

from __future__ import print_function
import numpy as np

from keras.utils import np_utils
import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from keras.models import Model


#-----------------------------模型超参数------------------------------
num_epochs = 5
batch_size = 500
learning_rate = 0.01

#-----------------------------数据获取及预处理------------------------------
#tf.keras.datasets
class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        # MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)      # [60000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)        # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)    # [60000]
        self.test_label = self.test_label.astype(np.int32)      # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_label[index]

data_loader = MNISTLoader()
print(data_loader.train_data.shape)
print(data_loader.train_label.shape)


#-----------------------------模型的构建------------------------------
#Keras Functional API 模式建立模型
inputs = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dense(units=10)(x)
outputs = tf.keras.layers.Softmax()(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_top_k_categorical_accuracy]
)
'''
#-----------------------------模型构建的第2种写法------------------------------
#Keras Sequential API 模式建立模型
from tensorflow.keras import layers, models
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_accuracy]
)
'''
#-----------------------------数据读取和Fit模型------------------------------
data_loader = MNISTLoader()
model.fit(data_loader.train_data, data_loader.train_label, epochs=num_epochs, batch_size=batch_size)

#-------------------------------获取模型最后一层的数据--------------------------------
# 获取x = tf.keras.layers.Flatten()(x)数据
# Reference:https://becominghuman.ai/visualizing-representations-bd9b62447e38
def create_truncated_model(trained_model):
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Flatten()(x)
    #model = tf.keras.Model(inputs=inputs, outputs=x)
    print(model.summary())

    for i, layer in enumerate(trained_model.layers):
        layer.set_weights(trained_model.layers[i].get_weights())
    trained_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return trained_model

truncated_model = create_truncated_model(model)
hidden_features = truncated_model.predict(data_loader.train_data)

#-------------------------------PCA,tSNE降维分析--------------------------------
pca = PCA(n_components=10)# 总的类别
pca_result = pca.fit_transform(hidden_features)
print('Variance PCA: {}'.format(np.sum(pca.explained_variance_ratio_)))

#Run T-SNE on the PCA features.
tsne = TSNE(n_components=2, perplexity=40,verbose = 1).fit_transform(pca_result[:500])
print(pca_result[:500].shape)

#-------------------------------可视化--------------------------------

y_test_cat = np_utils.to_categorical(data_loader.train_label[:500], num_classes = 10)# 总的类别
color_map = np.argmax(y_test_cat, axis=1)
plt.figure(figsize=(10,10))
for cl in range(10):# 总的类别
    indices = np.where(color_map==cl)
    indices = indices[0]
    plt.scatter(tsne[indices,0], tsne[indices, 1], label=cl)
plt.legend()
plt.show()