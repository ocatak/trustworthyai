#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 12:18:18 2023

@author: ozgur and murat
"""
#==============================================================================
### Part 1
#Import libraries
import tensorflow as tf
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
import numpy as np
import matplotlib.pyplot as plt
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

model.fit(x_train, y_train, validation_split=0.1, batch_size=10000, epochs=50,
          verbose=0, callbacks=[TqdmCallback(verbose=1)])

#==============================================================================
### Part 2

# Generate adversarial examples using FGSM
epsilon = 0.1
adv_x_test = fast_gradient_method(model.inner, x_test, epsilon, np.inf)

# Class probabilities for original and adversarial examples
quantifiers = ['var_ratio', 'pred_entropy', 'mean_softmax']

# Predict class probabilities for original examples
results = model.predict_quantified(x_test,
                                   quantifier=quantifiers,
                                   batch_size=1000,
                                   sample_size=32,
                                   verbose=1)

# Calculate the predictions and prediction uncertainties using the pcs and mean_softmax quantifiers
predictions_orig, entropy_orig = results[1]

# Predict class probabilities for adversarial examples
results_adv = model.predict_quantified(adv_x_test.numpy(),
                                   quantifier=quantifiers,
                                   batch_size=1000,
                                   sample_size=32,
                                   verbose=1)
# Calculate the predictions and prediction uncertainties using the pcs and mean_softmax quantifiers
predictions_adv, entropy_adv = results_adv[1]

#==============================================================================
### Part 3
# Set a threshold to determine highly uncertain instances
threshold = 1.95

# Find the indices of highly uncertain adversarial examples
highly_uncertain_indices = np.where(entropy_adv > threshold)[0]

# Sort the highly uncertain indices based on uncertainty values
sorted_indices = highly_uncertain_indices[np.argsort(entropy_adv[highly_uncertain_indices])]

# Plot the top 3 rows of highly uncertain adversarial examples
for i in range(0, min(len(sorted_indices), 2), 2):
    plt.figure(figsize=(15, 7))
    
    for j in range(2):
        if i + j < len(sorted_indices):
            idx = sorted_indices[i + j]
            uncertainty_orig = entropy_orig[idx]
            uncertainty_adv = entropy_adv[idx]
            label_orig = np.argmax(predictions_orig[idx])
            label_adv = np.argmax(predictions_adv[idx])
    
            plt.subplot(2, 4, j * 2 + 1)
            plt.imshow(x_test[idx].squeeze(), cmap='gray')
            plt.title(f"Image {i+j+1} - Original\nPrediction: {label_orig} - Uncertainty: {uncertainty_orig:.2f}")

            plt.subplot(2, 4, j * 2 + 2)
            plt.imshow(adv_x_test[idx], cmap='gray')
            plt.title(f"Image {i+j+1} - Adversarial\nPrediction: {label_adv} - Uncertainty: {uncertainty_adv:.2f}")
  
# Adjust the padding between the subplots
plt.tight_layout()

# Save results
plt.savefig("Module-2-Uncertainty_24_5.pdf",bbox_inches='tight')

# Show the plot
plt.show()


#==============================================================================
# Part4

# Create a subplot with 1 row and 2 columns
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Plot the distribution of uncertainty values for original examples in the first subplot
axs[0].hist(entropy_orig, bins=50, color='blue', alpha=0.7)
axs[0].set_xlabel('Uncertainty')
axs[0].set_ylabel('Count')
axs[0].set_title('Distribution of Uncertainty for Normal Inputs')

# Plot the distribution of uncertainty values for adversarial examples in the second subplot
axs[1].hist(entropy_adv, bins=50, color='red', alpha=0.7)
axs[1].set_xlabel('Uncertainty')
axs[1].set_ylabel('Count')
axs[1].set_title('Distribution of Uncertainty for Adversarial Inputs')

# Adjust the padding between the subplots
plt.tight_layout()

plt.savefig("Module-2-Uncertainty_26_0.pdf",bbox_inches='tight')


# Show the plot
plt.show()

#==============================================================================
# Part5
# Detect adversarial inputs using uncertainty
threshold = entropy_orig.mean()
adversarial_indices = np.where(entropy_adv > threshold )[0]
normal_indices = np.where(entropy_orig <= threshold)[0]

# Calculate success rate of detection
success_rate = len(adversarial_indices) / (len(adversarial_indices) + len(normal_indices))

print(f"Success rate of detecting adversarial inputs: {success_rate:.2%}")



