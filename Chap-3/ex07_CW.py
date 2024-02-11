#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 13:57:56 2023

@author: ozgur
"""
#==============================================================================
### Part 1
#Import Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2
from tqdm.notebook import tqdm
from tensorflow.keras.models import load_model
from tqdm.keras import TqdmCallback

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

# Train the model on the training set
model.fit(x_train, y_train, batch_size=5000, epochs=10, validation_split=0.1,
          verbose=0, callbacks=[TqdmCallback(verbose=1)])


# Create variables and set up key parameters
adv_x_test = []
adv_y_org = []
rand_idx_list = []

TARGET_CLASS = 5
y_tmp_target = np.zeros((1, 10))
y_tmp_target[:, TARGET_CLASS] = 1

 # Generate adversarial examples using Carlini-Wagner L2 attack
for _ in tqdm(range(150)):  
    rand_idx = np.random.randint(0, x_test.shape[0])
    rand_idx_list.append(rand_idx)

    tmp_input = x_test[rand_idx:rand_idx + 1, :]
    adv_y_org.append(y_test[rand_idx])

    logits_model = tf.keras.Model(model.input, model.layers[-1].output)
    
    adv_x_test_tmp = carlini_wagner_l2(model, tmp_input.reshape((1, 28, 28, 1)),
                                       targeted=True, y=[TARGET_CLASS],
                                       batch_size=128, confidence=100.0,
                                       abort_early=True, max_iterations=500,
                                       clip_min=0.0, clip_max=1.0)
    
    adv_x_test.append(adv_x_test_tmp.reshape(28, 28))
    
    
#==============================================================================   
### PART 2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Get predicted labels for the test set
y_pred = model.predict(x_test[rand_idx_list,:], verbose=0).argmax(axis=1)
y_org = y_test[rand_idx_list].argmax(axis=1)

adv_y_pred = model.predict(np.array(adv_x_test), verbose=0).argmax(axis=1)
y_org_pred = model.predict(x_test, verbose=0).argmax(axis=1)

# Find indices where original input prediction is not 0 but adversarial version's prediction is 0
indices = np.where((adv_y_org != TARGET_CLASS) & (adv_y_pred == TARGET_CLASS))[0][:10]

# Create lists for analysis
y_list = []
y_pred_cw_list = []
y_adv_cw_list = []

# Plot the examples
fig, axs = plt.subplots(2, 10, figsize=(10, 3))
axs = axs.flatten()
for i in range(10):
    idx_val = indices[i]
    idx_val = rand_idx_list[idx_val]
    axs[i].imshow(x_test[idx_val], cmap="gray")
    axs[i].set_title(f"Label: {y_pred[indices[i]]}")
    axs[i].axis("off")
    axs[i + 10].imshow(adv_x_test[indices[i]], cmap="gray")
    axs[i + 10].set_title(f"Label: {adv_y_pred[indices[i]]}")
    axs[i + 10].axis("off")
    
    y_pred_cw_list.append(y_pred[indices[i]])
    y_adv_cw_list.append(adv_y_pred[indices[i]])
    y_list.append(y_test[indices[i]].argmax(axis=0))

# Adjust the padding between the subplots
plt.tight_layout()

# plt.savefig("../../../Module-3-AML_files/Module-3-AML_18_0.pdf",bbox_inches='tight')

# Show the plot
plt.show()

# Get predicted labels for the test set before the attack
confusion_mtx = confusion_matrix(y_test.argmax(axis=1), y_org_pred)

# Get predicted labels for the test set after the attack
confusion_mtx_adv = confusion_matrix(y_list, y_adv_cw_list)

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

# plt.savefig("../../../Module-3-AML_files/Module-3-AML_18_1.pdf",bbox_inches='tight')

# Show the plot
plt.show()