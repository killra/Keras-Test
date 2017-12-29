import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise

import numpy as np
import sklearn.preprocessing as prep
import matplotlib.pyplot as plt

batch_size = 128
epochs = 5


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_origin = x_train.reshape(60000, 784)
x_test_origin = x_test.reshape(10000, 784)

x_train = x_train_origin.astype('float32')
x_test = x_test_origin.astype('float32')

#this is for batch normalization


def standard_scale(X_train, X_test):
	preprocessor = prep.StandardScaler().fit(X_train)
	X_train = preprocessor.transform(X_train)
	X_test = preprocessor.transform(X_test)
	return X_train, X_test

x_train, x_test = standard_scale(x_train, x_test)

model = Sequential()
model.add(GaussianNoise(input_shape=(784,), stddev=1.0))
model.add(Dense(600, activation='softplus', kernel_initializer='glorot_uniform'))
model.add(Dense(200, activation='softplus', kernel_initializer='glorot_uniform'))
model.add(Dense(100, activation='softplus', kernel_initializer='glorot_uniform'))
model.add(Dense(200, activation='softplus'))
model.add(Dense(600, activation='softplus'))
model.add(Dense(784, activation=None))

model.summary()
model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['accuracy'])

history = model.fit(x_train, x_train, batch_size=batch_size, epochs=epochs,
					verbose=1, validation_data=(x_test, x_test))
score = model.evaluate(x_test, x_test, verbose=0)
print 'Test loss:', score[0]
print 'Test accuracy', score[1]

# print "Visualization:"

# examples_to_show = 10
# #result = autoencoder.reconstruct(x_test[:examples_to_show])

# f, a = plt.subplots(2, 10, figsize=(10, 2))
# for i in range(examples_to_show):
# 	a[0][i].imshow(np.reshape(x_test_origin[i], (28, 28)))
# 	#a[1][i].imshow(np.reshape(result[i], (28, 28)))
# f.show()
# plt.show()
# plt.waitforbuttonpress()











