#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 14:17:46 2023

@author: ozgur
"""
#==============================================================================
### Part 1
#Import Libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.basic_iterative_method import basic_iterative_method
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# Load the pre-trained model
model = load_model('model.h5')

# Epsilon value
epsilon = 0.1

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

# Evaluate the model on the test set before adversarial training
score = model.evaluate(x_test, y_test, verbose=0)
test_loss.append(score[0])
test_accuracy.append(score[1])

# Evaluate the model on the test set before adversarial training with FGSM attack
x_test_fgsm = fast_gradient_method(model, x_test, eps=epsilon, norm=np.inf, targeted=False)
score_fgsm = model.evaluate(x_test_fgsm, y_test, verbose=0)
test_loss_fgsm.append(score_fgsm[0])
test_accuracy_fgsm.append(score_fgsm[1])

# Evaluate the model on the test set before adversarial training with BIM attack
x_test_bim = basic_iterative_method(model, x_test, eps=epsilon, eps_iter=0.01, nb_iter=10,
                                      norm=np.inf, targeted=False,
                                      sanity_checks=False)
score_bim = model.evaluate(x_test_bim, y_test, verbose=0)
test_loss_bim.append(score_bim[0])
test_accuracy_bim.append(score_bim[1])

# Define the adversarial training method
def adversarial_training(x, y, model, epochs, epsilon, batch_size):
    for epoch in tqdm(range(epochs)):
        for batch in tqdm(range(0, len(x), batch_size), leave=False):
            x_batch = x[batch:batch+batch_size]
            y_batch = y[batch:batch+batch_size]
            # Generate adversarial examples using FGSM and BIM attacks
            perturbation_fgsm = fast_gradient_method(model, x_batch, eps=epsilon, norm=np.inf, targeted=False)
            perturbation_bim = basic_iterative_method(model, x_batch, eps=epsilon, eps_iter=0.01, nb_iter=10,
                                                      norm=np.inf, targeted=False,
                                                      sanity_checks=False)

            # Combine the original image with the adversarial perturbation
            x_batch_fgsm = x_batch + perturbation_fgsm
            x_batch_bim = x_batch + perturbation_bim

        # Train the model on the original image and the adversarial example
        loss_fgsm = model.train_on_batch(x_batch_fgsm, y_batch)
        loss_bim = model.train_on_batch(x_batch_bim, y_batch)

        # Evaluate the model on the test set with FGSM attack
        perturbation_fgsm = fast_gradient_method(model, x_test, eps=epsilon, norm=np.inf, targeted=False)
        x_test_fgsm = x_test + perturbation_fgsm
        score_fgsm = model.evaluate(x_test_fgsm, y_test, verbose=0)
        test_loss_fgsm.append(score_fgsm[0])
        test_accuracy_fgsm.append(score_fgsm[1])

        # Evaluate the model on the test set with BIM attack
        perturbation_bim = basic_iterative_method(model, x_test, eps=epsilon, eps_iter=0.01, nb_iter=10,
                                                   norm=np.inf, targeted=False,
                                                   sanity_checks=False)
        x_test_bim = x_test + perturbation_bim
        score_bim = model.evaluate(x_test_bim, y_test, verbose=0)
        test_loss_bim.append(score_bim[0])
        test_accuracy_bim.append(score_bim[1])

        # Evaluate the model on the test set
        score = model.evaluate(x_test, y_test, verbose=0)
        
        test_loss.append(score[0])
        test_accuracy.append(score[1])
        
        print('Epoch:', epoch, '- Test loss:', score[0], '- Test accuracy:', score[1])

# Set the hyperparameters for adversarial training
epsilon = 0.1
epochs = 10
batch_size = 4096

# Run adversarial training
adversarial_training(x_train, y_train, model, epochs, epsilon, batch_size)


### PART 2

# Plot the prediction performance metrics
epochs_range = range(epochs + 1)  # Add 1 to include the evaluation before adversarial training

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, test_loss, label='Test Loss')
plt.plot(epochs_range, test_loss_fgsm, label='Test Loss with FGSM attack')
plt.plot(epochs_range, test_loss_bim, label='Test Loss with BIM attack')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, test_accuracy, label='Test Accuracy')
plt.plot(epochs_range, test_accuracy_fgsm, label='Test Accuracy with FGSM attack')
plt.plot(epochs_range, test_accuracy_bim, label='Test Accuracy with BIM attack')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Adjust the padding between the subplots
plt.tight_layout()

# plt.savefig("../../../Module-3-AML_files/Module-3-AML_21_0.pdf",bbox_inches='tight')

# Show the plot
plt.show()