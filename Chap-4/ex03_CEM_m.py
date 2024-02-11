#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tuesday Nov 14:39:32 2023

@author:ozgur and murat
"""
#==============================================================================
### Part 1
#Import libraries
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import alibi
from alibi.explainers import CEM

# Load the MNIST dataset
# x_train contains the training data
# y_train contains the training data's labels
# x_test contains the test data
# y_test contains the test data's labels

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print('x_train shape:', X_train.shape, 'y_train shape:',
      y_train.shape)
plt.gray()
plt.imshow(X_test[5]);


# Preprocess the data, inclusing scaling and shaping the data
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32') / 255.0
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32') / 255.0

# Define the cnn model architecture
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=5000, validation_split=0.1)

#Save the model
model.save('mnist_cnn.h5')

#Load and test accuracy on test dataset
cnn = load_model('mnist_cnn.h5')
cnn.summary()
score = cnn.evaluate(X_test, y_test, verbose=0)
print("="*30)
print('Test accuracy: ', score[1])
print("="*30)

# Define the encoder layers
encoder = keras.Sequential([
    keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2), padding='same'),
    keras.layers.Conv2D(1, (3, 3), activation=None, padding='same')
])

# Define the decoder layers
decoder = keras.Sequential([
    keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(14, 14, 1)),
    keras.layers.UpSampling2D((2, 2)),
    keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    keras.layers.Conv2D(1, (3, 3), activation=None, padding='same')
])

# Combine encoder and decoder
autoencoder = keras.Sequential([encoder, decoder])

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(X_train, X_train, batch_size=128, epochs=4, validation_data=(X_test, X_test), verbose=0)

#Save the autoencoder
autoencoder.save('mnist_autoencoder.h5', save_format='h5')

# Summary  the autoencoder
autoencoder.summary()

# Compare original with decoded images
ae = load_model('mnist_autoencoder.h5')
#ae.summary()
decoded_imgs = ae.predict(X_test)
n = 5
plt.figure(figsize=(20, 4))
for i in range(1, n+1):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(X_test[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

#Generate contrastive explanation with pertinent negative
#Explained instance
idx = 5
X = X_test[idx].reshape((1,) + X_test[idx].shape)

# Model prediction
cnn.predict(X).argmax(), cnn.predict(X).max()

#CEM parameters
mode = 'PN'  # 'PN' (pertinent negative) or 'PP' (pertinent positive)
shape = (1,) + X_train.shape[1:]  # instance shape
kappa = 0.  # minimum difference needed between the prediction probability for the perturbed instance on the
            # class predicted by the original instance and the max probability on the other classes 
            # in order for the first loss term to be minimized
beta = .1  # weight of the L1 loss term
gamma = 100  # weight of the optional auto-encoder loss term
c_init = 1.  # initial weight c of the loss term encouraging to predict a different class (PN) or 
              # the same class (PP) for the perturbed instance compared to the original instance to be explained
c_steps = 10  # nb of updates for c
max_iterations = 1000  # nb of iterations per value of c
feature_range = (X_train.min(),X_train.max())  # feature range for the perturbed instance
clip = (-1000.,1000.)  # gradient clipping
lr = 1e-2  # initial learning rate
no_info_val = -1. # a value, float or feature-wise, which can be seen as containing no info to make a prediction
                  # perturbations towards this value means removing features, and away means adding features
                  # for our MNIST images, the background (-0.5) is the least informative, 
                  # so positive/negative perturbations imply adding/removing features
 

print("="*30)
#tf.compat.v1.disable_v2_behavior()
#tf.compat.v1.disable_eager_execution()


#Initialize CEM explainer and explain instance
cem = CEM(cnn, mode, shape, kappa=kappa, beta=beta, feature_range=feature_range, 
          gamma=gamma, ae_model=ae, max_iterations=max_iterations, 
          c_init=c_init, c_steps=c_steps, learning_rate_init=lr, clip=clip, no_info_val=no_info_val)

explanation = cem.explain(X)

#Pertinent negative
print('Pertinent negative prediction: {}'.format(explanation.PN_pred))
plt.imshow(explanation.PN.reshape(28, 28));
plt.show()


#Generate pertinent positive
mode = 'PP'


#Initialize CEM explainer and explain instance
cem = CEM(cnn, mode, shape, kappa=kappa, beta=beta, feature_range=feature_range, 
          gamma=gamma, ae_model=ae, max_iterations=max_iterations, 
          c_init=c_init, c_steps=c_steps, learning_rate_init=lr, clip=clip, no_info_val=no_info_val)

print("="*30)
explanation = cem.explain(X)

# @title Pertinent positive
print('Pertinent positive prediction: {}'.format(
      explanation.PP_pred))
plt.imshow(explanation.PP.reshape(28, 28));
plt.show()
