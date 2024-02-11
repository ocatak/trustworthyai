#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 12:28:43 2023

@author: ozgur
"""
#==============================================================================
### Part 1
# Import libraries
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm.keras import TqdmCallback

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to the range [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Define the pre-trained MNIST model
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

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# Train the model on the training set
model.fit(x_train, y_train, batch_size=2000, epochs=10, validation_split=0.1,
          verbose=0, callbacks=[TqdmCallback(verbose=1)])

# Save the model
model.save('model.h5')

#==============================================================================
### Part 2
# Generate adversarial examples using FGSM with untargeted values
epsilon = 0.2
adv_x_test = fast_gradient_method(model, x_test, epsilon, np.inf)

# Get predicted labels for the test set
y_pred = model.predict(x_test, verbose=0).argmax(axis=1)
adv_y_pred = model.predict(adv_x_test, verbose=0).argmax(axis=1)

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

# Adjust the padding between the subplots
plt.tight_layout()

#plt.savefig("../../../Module-3-AML_files/Module-3-AML_7_3.pdf",bbox_inches='tight')

# Show the plot
plt.show()

# Plot some examples with predicted labels as titles
fig, axs = plt.subplots(2, 10, figsize=(10, 3))
axs = axs.flatten()
sample_indices = random.sample(range(len(x_test)), 10)
for i, sample_index in enumerate(sample_indices):
    axs[i].imshow(x_test[sample_index], cmap="gray")
    axs[i].set_title(f"Label: {y_pred[sample_index]}")
    axs[i].axis("off")
    axs[i + 10].imshow(adv_x_test[sample_index], cmap="gray")
    axs[i + 10].set_title(f"Label: {adv_y_pred[sample_index]}")
    axs[i + 10].axis("off")
    
 # Adjust the padding between the subplots
plt.tight_layout()

#plt.savefig("../../../Module-3-AML_files/Module-3-AML_7_4.pdf",bbox_inches='tight')

# Show the plot
plt.show()

#==============================================================================
## PART 3
# Generate adversarial examples using FGSM with targeted values
# Define variables and set up a target class
epsilon = 0.5
target_adv = np.zeros_like(y_test)
TARGET_CLASS = 5
target_adv[:, TARGET_CLASS] = 1
target_adv = target_adv.argmax(axis=1)

# Generate adversarial examples
adv_x_test = fast_gradient_method(model, x_test, epsilon, np.inf, y = target_adv)

# Get predicted labels for the test set
y_pred = model.predict(x_test, verbose=1).argmax(axis=1)
adv_y_pred = model.predict(adv_x_test, verbose=1).argmax(axis=1)

# Get predicted labels for the test set before the attack
confusion_mtx = confusion_matrix(y_test.argmax(axis=1), y_pred)

# Get predicted labels for the test set after the attack
confusion_mtx_adv = confusion_matrix(y_test.argmax(axis=1), adv_y_pred)

# Find indices where original input prediction is not 0 but adversarial version's prediction is 0
indices = np.where((y_pred == TARGET_CLASS) & (adv_y_pred != TARGET_CLASS))[0][:10]

# Plot the selected examples
fig, axs = plt.subplots(2, 5, figsize=(10, 4))
axs = axs.flatten()
for i, index in enumerate(indices):
    axs[i].imshow(x_test[index], cmap="gray")
    axs[i].set_title(f"Label: {y_pred[index]}\nAdv Label: {adv_y_pred[index]}")
    axs[i].axis("off")
    
   # Adjust the padding between the subplots  
plt.tight_layout()

# plt.savefig("../../../Module-3-AML_files/Module-3-AML_9_1.pdf",bbox_inches='tight')

# Show the plot
plt.show()

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

 # Adjust the padding between the subplots
plt.tight_layout()

# plt.savefig("../../../Module-3-AML_files/Module-3-AML_9_2.pdf",bbox_inches='tight')

# Show the plot
plt.show()