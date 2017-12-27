# -*- coding: utf-8 -*-

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam, SGD
from keras.layers.normalization import BatchNormalization

batch_size = 128
num_classes = 10
epochs = 4
#epochs = 20
# 实验的时候4 epochs基本上都够用了

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#this is for batch normalization
x_train /= 255
x_test /= 255

print x_train.shape[0], 'train samples'
print x_test.shape[0], 'test samples'


# 将类别的数值转换为one-hot的数组型数据结构
# to_categorical(y, nb_classes=None)
# 将类别向量(从0到nb_classes的整数向量)映射为二值类别矩阵, 用于应用到以categorical_crossentropy为目标函数的模型中.
# y: 类别向量; nb_classes:总共类别数
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
#for i in range(10):
#	model.add(Dense(700 - i * 50, activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
#model.add(BatchNormalization())


#model.add(Dropout(0.2))
#model.add(Dense(32, activation='relu'))
#model.add(Dropout(0.2))



#model.add(Dense(32, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', 
			optimizer=Adam(), 
			metrics=['accuracy'])
# 训练模型
# Keras以Numpy数组作为输入数据和标签的数据类型
# fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None)
# nb_epoch：整数，训练的轮数，训练数据将会被遍历nb_epoch次。Keras中nb开头的变量均为"number of"的意思
# verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
# shuffle：布尔值，表示是否在训练过程中每个epoch前随机打乱输入样本的顺序。
# fit函数返回一个History的对象，其History.
# history属性记录了损失函数和其他指标的数值随epoch变化的情况，如果有验证集的话，也包含了验证集的这些指标变化情况

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
					verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print 'Test loss:', score[0]
print 'Test accuracy', score[1]

