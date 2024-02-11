#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 12:42:37 2023

@author: ozgur
"""
# Import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to the range [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Load the pre-trained model
model = load_model('model.h5')

# Generate adversarial examples using BIM
epsilon = 0.1
target_adv = np.zeros_like(y_test)
TARGET_CLASS = 5
target_adv[:, TARGET_CLASS] = 1
target_adv = target_adv.argmax(axis=1)

y_attack_target = (np.ones((y_test.shape[0],)) * 1).astype(int)

adv_x_test = projected_gradient_descent(model, x_test, eps=epsilon, eps_iter=0.01, nb_iter=100,
                                    clip_min=0.0, clip_max=1.0, norm=np.inf, y=target_adv,
                                    targeted=True, sanity_checks=False)

# Get predicted labels for the test set
y_pred = model.predict(x_test, verbose=0).argmax(axis=1)
adv_y_pred = model.predict(adv_x_test, verbose=0).argmax(axis=1)

# Find indices where original input prediction is not 0 but adversarial version's prediction is 0
indices = np.where((y_pred != TARGET_CLASS) & (adv_y_pred == TARGET_CLASS))[0][:10]

# Plot the examples
fig, axs = plt.subplots(2, 10, figsize=(10, 3))
axs = axs.flatten()
for i in range(10):
    axs[i].imshow(x_test[indices[i]], cmap="gray")
    axs[i].set_title(f"Label: {y_pred[indices[i]]}")
    axs[i].axis("off")
    axs[i + 10].imshow(adv_x_test[indices[i]], cmap="gray")
    axs[i + 10].set_title(f"Label: {adv_y_pred[indices[i]]}")
    axs[i + 10].axis("off")

# Adjust the padding between the subplots
plt.tight_layout()

# plt.savefig("../../../Module-3-AML_files/Module-3-AML_11_0.pdf",bbox_inches='tight')

# Show the plot
plt.show()

# Get predicted labels for the test set before the attack
confusion_mtx = confusion_matrix(y_test.argmax(axis=1), y_pred)

# Get predicted labels for the test set after the attack
confusion_mtx_adv = confusion_matrix(y_test.argmax(axis=1), adv_y_pred)

# Plot the confusion matrices in the same subplot
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)
axs[0].set_xticks(range(10))
axs[0].set_yticks(range(10))
axs[0].set_xlabel('Predicted label')
axs[0].set_ylabel('True label')
axs[0].set_title('Confusion Matrix (Before Attack)')

axs[1].imshow(confusion_mtx_adv, interpolation='nearest', cmap=plt.cm.Blues)
axs[1].set_xticks(range(10))
axs[1].set_yticks(range(10))
axs[1].set_xlabel('Predicted label')
axs[1].set_ylabel('True label')
axs[1].set_title('Confusion Matrix (After Attack)')

plt.tight_layout()

# plt.savefig("../../../Module-3-AML_files/Module-3-AML_11_1.pdf",bbox_inches='tight')

plt.show()