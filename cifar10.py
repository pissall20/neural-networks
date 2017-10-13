#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 23:55:51 2017

@author: hduser
"""

import numpy as np
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.datasets import cifar10
from keras.constraints import max_norm
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras import backend as K
from time import time
K.set_image_dim_ordering("th")

np.random.seed(7)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train, X_test = X_train.astype('float32'), X_test.astype('float32')
X_train, X_test = X_train/255.0, X_test/255.0

y_train, y_test = np_utils.to_categorical(y_train), np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=max_norm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3), activation='relu', padding='same', kernel_constraint=max_norm(3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=max_norm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=max_norm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

epochs = 60
lrate = 0.0001
decay = 0.01/epochs
adam = Adam(lr=lrate, decay=decay, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())


model_start_time = time()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
model_finish = time()


# 9.92 hrs
print("Total time taken :{0} hours".format((model_finish - model_start_time)/60/60))

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
# 75.70
print("Accuracy: %.2f%%" % (scores[1]*100))

model.save("cnn_cifar10.pkl")