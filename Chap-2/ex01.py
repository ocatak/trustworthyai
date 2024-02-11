#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 10:19:54 2023

@author: ozgur and murat
"""
#==============================================================================
### Part 1
#Import Libraries
import tensorflow as tf
import uncertainty_wizard as uwiz
from tqdm.keras import TqdmCallback

# Load the MNIST digits dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to the range [0, 1]
x_train = (x_train.astype('float32') / 255).reshape(x_train.shape[0], 28, 28, 1)
x_test = (x_test.astype('float32') / 255).reshape(x_test.shape[0], 28, 28, 1)

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

# Create the model
model = uwiz.models.StochasticSequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Dropouts !!!
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compiling and fitting is the same as in regular keras models as well:
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, validation_split=0.1, batch_size=10000, epochs=20,
          verbose=0, callbacks=[TqdmCallback(verbose=1)])

# quantifiers = ['var_ratio', 'pred_entropy', 'mutu_info', 'mean_softmax']
quantifiers = ['var_ratio', 'pred_entropy', 'mean_softmax']
results = model.predict_quantified(x_test,
                                   quantifier=quantifiers,
                                   batch_size=64,
                                   sample_size=32,
                                   verbose=0)

#==============================================================================
### Part 2

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Initialize the random number generator
np.random.seed(10)

# Calculate the predictions and prediction uncertainties using the pcs and mean_softmax quantifiers
pcs_predictions, pcs_uncertainties = results[0]
pred_entropy_predictions, pred_entropy_uncertainties = results[1]
mean_softmax_predictions, mean_softmax_uncertainties = results[2]

# Compute the confusion matrix
confusion_mtx = confusion_matrix(y_test, pcs_predictions)

# Create a subplot with 2 rows and 2 columns
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Increase the font size for all subplots
fontsize = 20

# Plot a histogram of the prediction uncertainties from pcs in the first subplot
axs[0, 0].hist(pcs_uncertainties, bins=50)
axs[0, 0].set_xlabel('Variation Ratio', fontsize=fontsize)
axs[0, 0].set_ylabel('Count', fontsize=fontsize)
axs[0, 0].set_title('Uncertainty - VariationRatio', fontsize=fontsize)

# Plot a histogram of the prediction uncertainties from pred_entropy in the second subplot
axs[0, 1].hist(pred_entropy_uncertainties, bins=50)
axs[0, 1].set_xlabel('Prediction uncertainty (PredictiveEntropy)', fontsize=fontsize)
axs[0, 1].set_ylabel('Count', fontsize=fontsize)
axs[0, 1].set_title('Uncertainty - PredictiveEntropy', fontsize=fontsize)

# # Plot a histogram of the prediction uncertainties from mutu_info in the third subplot
# axs[0, 2].hist(mutu_info_uncertainties, bins=50)
# axs[0, 2].set_xlabel('Mutual Information', fontsize=fontsize)
# axs[0, 2].set_ylabel('Count', fontsize=fontsize)
# axs[0, 2].set_title('Uncertainty - MutualInformation', fontsize=fontsize)

# Plot a histogram of the prediction uncertainties from mean_softmax in the third subplot
axs[1, 0].hist(mean_softmax_uncertainties, bins=50)
axs[1, 0].set_xlabel('Mean Softmax', fontsize=fontsize)
axs[1, 0].set_ylabel('Count', fontsize=fontsize)
axs[1, 0].set_title('Uncertainty - MeanSoftmax', fontsize=fontsize)

# Plot the confusion matrix in the fifth subplot
axs[1, 1].imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)
axs[1, 1].set_xticks(np.arange(10))
axs[1, 1].set_yticks(np.arange(10))
axs[1, 1].set_xlabel('Predicted label', fontsize=fontsize)
axs[1, 1].set_ylabel('True label', fontsize=fontsize)
axs[1, 1].set_title('Confusion Matrix', fontsize=fontsize)

# Remove the unused subplot
#fig.delaxes(axs[1, 2])

# Adjust the padding between the subplots
plt.tight_layout()

# plt.savefig("Module-2-Uncertainty_6_0.pdf",bbox_inches='tight')

# Show the plot
plt.show()

#==============================================================================
# Part 3

# Set a threshold to determine highly uncertain instances
threshold = 2.1 # !!! You can adjust this parameter !!!

# Get the predictions, prediction uncertainties, and predicted labels
pred_labels, pred_entropy_uncertainties = results[1][0], results[1][1]

# Find the indices of highly uncertain instances
highly_uncertain_indices = np.where(pred_entropy_uncertainties > threshold)[0]

# Display the highly uncertain images
num_images = len(highly_uncertain_indices)
rows = int(np.ceil(num_images / 5))
fig, axs = plt.subplots(rows, 5, figsize=(20, rows*4))

# Increase the font size for titles
title_fontsize = 20

# Initiate a Loop to Iterate Highly Uncertain Images
for i, idx in enumerate(highly_uncertain_indices):
    row, col = divmod(i, 5)
    axs[row, col].imshow(x_test[idx, :, :, 0], cmap='gray')
    axs[row, col].set_title(f'Pred: {pred_labels[idx]}, Uncertainty: {pred_entropy_uncertainties[idx]:.2f}',
                           fontsize=title_fontsize)  # Set the font size for titles
    axs[row, col].axis('off')

# Adjust the padding between the subplots
plt.tight_layout()

# plt.savefig("Module-2-Uncertainty_8_0.pdf",bbox_inches='tight')

# Show the plot
plt.show()

#==============================================================================
# Part 4

# Display the highly uncertain images
num_images = len(highly_uncertain_indices)
rows = int(np.ceil(num_images / 5))
fig, axs = plt.subplots(rows, 5, figsize=(20, rows*4))

# Increase the font size for axes labels
axes_label_fontsize = 25

# Initiate a Loop to Iterate Highly Uncertain Images
for i, idx in enumerate(highly_uncertain_indices):
    row, col = divmod(i, 5)
    predictions = model.inner.predict(np.expand_dims(x_test[idx], axis=0), verbose=0)[0]
    
    axs[row, col].bar(range(10), predictions, align='center')
    axs[row, col].set_xticks(range(10))
    axs[row, col].set_ylabel('Softmax value', fontsize=axes_label_fontsize)  # Increase font size
    axs[row, col].set_xlabel('Classes', fontsize=axes_label_fontsize)  # Increase font size
    
    axs[row, col].set_ylim(0.00, 0.5)
    axs[row, col].grid()

# Adjust the padding between the subplots
plt.tight_layout()

# Save results
# plt.savefig("Module-2-Uncertainty_10_0.pdf",bbox_inches='tight')

# Show the plot
plt.show()

