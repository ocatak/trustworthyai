#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:22:06 2023

@author: ozgur and murat
"""
#==============================================================================
### Part 1
#Import Libraries
import tensorflow as tf
import uncertainty_wizard as uwiz
from tqdm.keras import TqdmCallback
import numpy as np


# Load the MNIST digits dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to the range [0, 1]
x_train = (x_train.astype('float32') / 255).reshape(x_train.shape[0], 28, 28, 1)
x_test = (x_test.astype('float32') / 255).reshape(x_test.shape[0], 28, 28, 1)

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

#Initialize the random number generator
np.random.seed(10)

#Set the total number of ensembles
TOTAL_ENSEMBLES = 5

# Define the Model Architecture and the Training Process
def model_creation_and_training(model_id: int):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    # Calculate the training subset size for each ensemble model
    sample_size = int(x_train.shape[0] / TOTAL_ENSEMBLES)
    
    # Randomly select the indices of the training instance
    train_indices = np.random.choice(len(x_train), size=sample_size, replace=False)
    train_images_subset = x_train[train_indices]
    train_labels_subset = y_train[train_indices]
    
    # Train the model
    # Note that we set the number of epochs to just 1, to be able to run this notebook quickly
    # Set the number of epochs higher if you want to optimally train the network
    fit_history = model.fit(train_images_subset, train_labels_subset,
                            validation_split=0.1, batch_size=10000, epochs=50,
                            verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(patience=2),
                                                  TqdmCallback(verbose=1)])
    
    # Return the model
    return model, fit_history.history

# Initialize the Ensemble Object
ensemble = uwiz.models.LazyEnsemble(num_models=TOTAL_ENSEMBLES,
                                    model_save_path="tmp/ensemble",
                                    default_num_processes=0)
# Creates, trains and persists atomic models using our function defined above
training_histories = ensemble.create(create_function=model_creation_and_training)

#==============================================================================
# Part 2

# Get two one-dimensional np arrays: One containing the predictions and one containing the confidences
# Define a List of Quantifiers
quantifiers = ['var_ratio', 'pred_entropy', 'mean_softmax']

# Predict the  Quantifiers
ensemble_results = ensemble.predict_quantified(x_test,
                                               quantifier=quantifiers,
                                               batch_size=64,
                                               verbose=0)

#==============================================================================
# Part 3

#Import Libraries
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Calculate the predictions and prediction uncertainties using the pcs and mean_softmax quantifiers
pcs_predictions, pcs_uncertainties = ensemble_results[0]
pred_entropy_predictions, pred_entropy_uncertainties = ensemble_results[1]
mean_softmax_predictions, mean_softmax_uncertainties = ensemble_results[2]

# Compute the confusion matrix
confusion_mtx = confusion_matrix(y_test, pcs_predictions)

# Create a subplot with 2 rows and 2 columns
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Increase the font size for specified text
fontsize = 20

# Plot a histogram of the prediction uncertainties from pcs in the second subplot
axs[0, 0].hist(pcs_uncertainties, bins=50)
axs[0, 0].set_xlabel('Variation Ratio', fontsize=fontsize)
axs[0, 0].set_ylabel('Count', fontsize=fontsize)
axs[0, 0].set_title('Uncertainty - VariationRatio', fontsize=fontsize)

# Plot a histogram of the prediction uncertainties from mean_softmax in the third subplot
axs[0, 1].hist(pred_entropy_uncertainties, bins=50)
axs[0, 1].set_xlabel('Prediction uncertainty (PredictiveEntropy)', fontsize=fontsize)
axs[0, 1].set_ylabel('Count', fontsize=fontsize)
axs[0, 1].set_title('Uncertainty - PredictiveEntropy', fontsize=fontsize)

# Plot a histogram of the prediction uncertainties from mean_softmax in the third subplot
axs[1, 0].hist(mean_softmax_predictions, bins=50)
axs[1, 0].set_xlabel('Mean Softmax', fontsize=fontsize)
axs[1, 0].set_ylabel('Count', fontsize=fontsize)
axs[1, 0].set_title('Uncertainty - MeanSoftmax', fontsize=fontsize)

# Plot the confusion matrix in the first subplot (last row)
axs[1, 1].imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)
axs[1, 1].set_xticks(np.arange(10))
axs[1, 1].set_yticks(np.arange(10))
axs[1, 1].set_xlabel('Predicted label', fontsize=fontsize)
axs[1, 1].set_ylabel('True label', fontsize=fontsize)
axs[1, 1].set_title('Confusion Matrix', fontsize=fontsize)

# # Remove the unused subplot
# fig.delaxes(axs[1, 2])

# Adjust the padding between the subplots
plt.tight_layout()

plt.savefig("Module-2-Uncertainty_18_0.pdf",bbox_inches='tight')

# Show the plot
plt.show()

#==============================================================================
# Part 4

# Set a threshold to determine highly uncertain instances
threshold = 2.3 # !!! You can adjust this parameter !!!

# Get the predictions, prediction uncertainties, and predicted labels
pred_labels, pred_entropy_uncertainties = ensemble_results[1][0], ensemble_results[1][1]

# Find the indices of highly uncertain instances
highly_uncertain_indices = np.where(pred_entropy_uncertainties > threshold)[0]

# Display the highly uncertain images
num_images = len(highly_uncertain_indices)

# Calculate the number of rows
rows = int(np.ceil(num_images / 5))
fig, axs = plt.subplots(rows, 5, figsize=(20, rows*4))

# Increase the font size for specified text
fontsize = 20

# Initiate a Loop to Iterate Highly Uncertain Images
for i, idx in enumerate(highly_uncertain_indices):
    row, col = divmod(i, 5)
    axs[row, col].imshow(x_test[idx, :, :, 0], cmap='gray')
    axs[row, col].set_title(f'Pred: {pred_labels[idx]}, Uncertainty: {pred_entropy_uncertainties[idx]:.2f}',
                           fontsize=fontsize)
    axs[row, col].axis('off')

# Adjust the padding between the subplots
plt.tight_layout()

#Save resutls
plt.savefig("Module-2-Uncertainty_20_0.pdf",bbox_inches='tight')

# Show the plot
plt.show()

#==============================================================================
# Part 5

#Import Libraries
from tensorflow import keras

# Display the highly uncertain images
num_images = len(highly_uncertain_indices)

# Calculate the number of rows
rows = int(np.ceil(num_images / 5))

# Create a Figure with Subplots
fig, axs = plt.subplots(rows, 5, figsize=(20, rows*4))

# Increase the font size for specified text
fontsize = 20

# Initiate a Loop to Iterate Highly Uncertain Images
for i, idx in enumerate(highly_uncertain_indices):
    row, col = divmod(i, 5)
    softmax_outputs = []
    for model_id in range(TOTAL_ENSEMBLES):
        ensemble_local_model = keras.models.load_model('tmp/ensemble/' + str(model_id))
        predictions = ensemble_local_model.predict(np.expand_dims(x_test[idx], axis=0), verbose=0)[0]
        softmax_outputs.append(predictions)
    
    # Store softmax outputs
    softmax_outputs = np.array(softmax_outputs).mean(axis=0)
    
    # Plot softmax outputs
    axs[row, col].bar(range(10), softmax_outputs, align='center')
    axs[row, col].set_xticks(range(10))
    axs[row, col].set_ylabel('Softmax value', fontsize=fontsize)
    axs[row, col].set_xlabel('Classes', fontsize=fontsize)
    
    axs[row, col].set_ylim(0.00, 0.5)
    axs[row, col].grid()

# Adjust the padding between the subplots
plt.tight_layout()

# Save results
plt.savefig("Module-2-Uncertainty_22_4.pdf",bbox_inches='tight')

# Show the plot
plt.show()

