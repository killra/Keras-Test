import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras.models import Model
from keras import regularizers

import numpy as np
import matplotlib.pyplot as plt

batch_size = 128
epochs = 5


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_origin = x_train.reshape(60000, 784)
x_test_origin = x_test.reshape(10000, 784)

x_train = x_train_origin.astype('float32')
x_test = x_test_origin.astype('float32')

# this is for batch normalization
x_train /= 255
x_test /= 255

# the beginning of building model by Funcition
input_img = Input(shape=(784,))
encoder = Dense(200, activation='softplus', 
				kernel_initializer='glorot_uniform', 
				activity_regularizer=regularizers.l1(10e-7))(input_img)

decoder = Dense(784, activation=None)(encoder)

autoencoder = Model(inputs=input_img, outputs=decoder)
encoderModel = Model(inputs=input_img, outputs=encoder)

autoencoder.summary()
autoencoder.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['accuracy'])

history = autoencoder.fit(x_train, x_train, batch_size=batch_size, epochs=epochs,
					verbose=1, validation_data=(x_test, x_test))
score = autoencoder.evaluate(x_test, x_test, verbose=0)
print 'Test loss:', score[0]
print 'Test accuracy', score[1]

encoded_result = encoderModel.predict(x_test)
print encoded_result[0]

print "Visualization:"

examples_to_show = 10
encoded_imgs = autoencoder.predict(x_test)

f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(examples_to_show):
	a[0][i].imshow(np.reshape(x_test_origin[i], (28, 28)))
	a[1][i].imshow(np.reshape(encoded_imgs[i], (28, 28)))
f.show()
plt.show()
plt.waitforbuttonpress()











