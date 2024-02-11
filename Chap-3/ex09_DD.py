#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 14:29:31 2023

@author: ozgur
"""
#==============================================================================
### Part 1
#Import Libraries
import tensorflow as tf
from tqdm.keras import TqdmCallback
from tensorflow.keras.datasets import mnist
import keras
from keras import layers
import numpy as np

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to the range [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Define empty lists to store the metrics
test_loss = []
test_accuracy = []
test_loss_fgsm = []
test_accuracy_fgsm = []
test_loss_bim = []
test_accuracy_bim = []

# Defining the original model
def get_model():
    model = keras.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Reshape(target_shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax")
    ])
    
    # Train the original model on the training data
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    
    return model

# Training the defense distilled model
def train_defense_model(x_train, y_train, T=100, alpha=0.1):
    """
    Trains a defensively distilled model given training data and hyperparameters.
    Args:
    x_train (numpy.ndarray): Training data input.
    y_train (numpy.ndarray): Training data labels.
    T (int): Temperature parameter for softened labels.
    alpha (float): Strength of the KL regularization term.
    
    Returns:
    The defensively distilled model.
    """

    # Define the distilled model
    distilled_model = keras.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Reshape(target_shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Lambda(lambda x: x / T),  # Softmax temperature scaling
        layers.Dense(10, activation="softmax")
    ])

    # Train the original model
    model = get_model()
    model.fit(x_train, y_train, epochs=10, batch_size=2000, verbose=0, callbacks=[TqdmCallback(verbose=1)])

    # Generate softened labels for the training data using the original model
    y_train_soft = model.predict(x_train)
    y_train_soft = np.exp(np.log(y_train_soft) / T)
    y_train_soft = y_train_soft / np.sum(y_train_soft, axis=1, keepdims=True)

    # Train the distilled model on the softened labels
    def kl_divergence(y_true, y_pred):
        return keras.losses.kullback_leibler_divergence(y_true, y_pred) * alpha

    # Compile the distilled model
    distilled_model.compile(loss="categorical_crossentropy",
                            optimizer="adam",
                            metrics=["accuracy"])
    
    # Train the distilled model
    distilled_model.fit(x_train, y_train_soft, epochs=10, batch_size=2000,
                         callbacks=[TqdmCallback(verbose=1)], verbose=0,
                         validation_data=(x_train, y_train))  # Use training data as validation data for distillation

    return distilled_model


#==============================================================================
### PART 2
# Import libraries
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.basic_iterative_method import basic_iterative_method

# Define the attack parameter
epsilon = 0.1

# Define empty lists to store the metrics
test_loss = []
test_accuracy = []

# Initialize the odel
org_model = get_model()

# Train the model
org_model.fit(x_train, y_train, epochs=10, batch_size=2000, verbose=0,
          callbacks=[TqdmCallback(verbose=1)])

# Evaluate the model on the test set before defensive distillation
score = org_model.evaluate(x_test, y_test, verbose=0)
test_loss.append(score[0])
test_accuracy.append(score[1])

# Evaluate the model on the test set before defensive distillation with FGSM attack
x_test_fgsm = fast_gradient_method(org_model, x_test, eps=epsilon, norm=np.inf, targeted=False)
score = org_model.evaluate(x_test_fgsm, y_test, verbose=0)
test_loss.append(score[0])
test_accuracy.append(score[1])

# Evaluate the model on the test set before defensive distillation with BIM attack
x_test_bim = basic_iterative_method(org_model, x_test, eps=epsilon, eps_iter=0.01, nb_iter=10,
                                      norm=np.inf, targeted=False,
                                      sanity_checks=False)
score = org_model.evaluate(x_test_bim, y_test, verbose=0)
test_loss.append(score[0])
test_accuracy.append(score[1])

# Train distilled model
distilled_model = train_defense_model(x_train, y_train, T=100)

# Evaluate the model on the test set before defensive distillation
score = distilled_model.evaluate(x_test, y_test, verbose=0)
test_loss.append(score[0])
test_accuracy.append(score[1])

# Evaluate the model on the test set before defensive distillation with FGSM attack
x_test_fgsm = fast_gradient_method(distilled_model, x_test, eps=epsilon, norm=np.inf, targeted=False)
score = distilled_model.evaluate(x_test_fgsm, y_test, verbose=0)
test_loss.append(score[0])
test_accuracy.append(score[1])

# Evaluate the model on the test set before defensive distillation with BIM attack
x_test_bim = basic_iterative_method(distilled_model, x_test, eps=epsilon, eps_iter=0.01, nb_iter=10,
                                      norm=np.inf, targeted=False,
                                      sanity_checks=False)
score = distilled_model.evaluate(x_test_bim, y_test, verbose=0)
test_loss.append(score[0])
test_accuracy.append(score[1])

#==============================================================================
### PART 3
# Import libraries
import matplotlib.pyplot as plt

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Plot test accuracy
axes[0].bar(['No Defense', 'FGSM', 'BIM', 'Distilled', 'Distilled+FGSM', 'Distilled+BIM'], test_accuracy)
axes[0].set_title('Test Accuracy')
axes[0].tick_params(axis='x', rotation=90)  # rotate xtick labels by 90 degrees

# Plot test loss
axes[1].bar(['No Defense', 'FGSM', 'BIM', 'Distilled', 'Distilled+FGSM', 'Distilled+BIM'], test_loss)
axes[1].set_title('Test Loss')
axes[1].tick_params(axis='x', rotation=90)  # rotate xtick labels by 90 degrees

# Adjust the padding between the subplots
plt.tight_layout()

#plt.savefig("../../../Module-3-AML_files/Module-3-AML_25_0.pdf",bbox_inches='tight')

# Show the plot
plt.show()