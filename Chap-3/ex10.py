#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 14:47:34 2023

@author: ozgur
"""
#==============================================================================
### Part 1
#Import Libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Define the model
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Define the loss function
def loss_fn(y_true, y_pred):
    # Calculate the cross-entropy loss
    ce_loss = keras.losses.categorical_crossentropy(y_true, y_pred)
    # Calculate the KL divergence between the original and perturbed predictions
    kl_div = keras.losses.kl_divergence(y_true, y_pred)
    # Apply gradient masking
    masked_grads = tape.gradient(kl_div, model.trainable_variables)
    # Return the combined loss and masked gradients
    return tf.reduce_mean(ce_loss + kl_div), masked_grads

# Define the optimizer
optimizer = keras.optimizers.Adam()

# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Training parameters
batch_size = 2000
epochs = 300
steps_per_epoch = x_train.shape[0] // batch_size

# Training loop
for epoch in tqdm(range(epochs)):
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    
    for step in range(steps_per_epoch):
        # Get a batch of training data
        batch_x = x_train[step * batch_size : (step + 1) * batch_size]
        batch_y = y_train[step * batch_size : (step + 1) * batch_size]
        
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = model(batch_x)
            # Compute the loss and masked gradients on the batch
            loss, masked_grads = loss_fn(batch_y, y_pred)
        
        # Apply the masked gradients to the model variables
        optimizer.apply_gradients(zip(masked_grads, model.trainable_variables))
        
        # Update the epoch loss and accuracy
        epoch_loss += loss.numpy()
        epoch_accuracy += np.mean(np.argmax(batch_y, axis=-1) == np.argmax(y_pred, axis=-1))
    
    epoch_loss /= steps_per_epoch
    epoch_accuracy /= steps_per_epoch
    if epoch % 25 == 0:
        print(f'Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Accuracy={epoch_accuracy:.4f}')

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'Test Loss={test_loss:.4f}, Test Accuracy={test_accuracy:.4f}')

#==============================================================================
### PART 2
# Compute gradients for a sample batch
sample_batch_x = x_train[:batch_size]
sample_batch_y = y_train[:batch_size]

sample_batch_x_tensor = tf.convert_to_tensor(sample_batch_x, dtype=tf.float32)

with tf.GradientTape() as tape:
    tape.watch(sample_batch_x_tensor)
    sample_batch_y_pred = model(sample_batch_x_tensor)
    sample_loss = keras.losses.categorical_crossentropy(sample_batch_y, sample_batch_y_pred)

sample_gradients = tape.gradient(sample_loss, sample_batch_x_tensor)

# Apply masking to gradients
sample_batch_x_perturbed = sample_batch_x_tensor + tf.random.normal(shape=sample_batch_x_tensor.shape, mean=0.0, stddev=0.01)
with tf.GradientTape() as tape:
    tape.watch(sample_batch_x_perturbed)
    sample_batch_y_pred_perturbed = model(sample_batch_x_perturbed)
    sample_loss_perturbed = keras.losses.categorical_crossentropy(sample_batch_y, sample_batch_y_pred_perturbed)

sample_gradients_perturbed = tape.gradient(sample_loss_perturbed, sample_batch_x_perturbed)

#==============================================================================
### PART 3

# Compute the difference between gradients before and after masking
sample_gradients_diff = sample_gradients_perturbed - sample_gradients

# img_ind = np.random.randint(sample_batch_x.shape[0])
img_ind = 882

# Determine the maximum and minimum values of gradients for consistent axis limits
z_min = min(np.min(sample_gradients[img_ind].numpy()), np.min(sample_gradients_perturbed[img_ind].numpy()), np.min(sample_gradients_diff[img_ind].numpy()))
z_max = max(np.max(sample_gradients[img_ind].numpy()), np.max(sample_gradients_perturbed[img_ind].numpy()), np.max(sample_gradients_diff[img_ind].numpy()))

# Create 3G Plot
fig = plt.figure(figsize=(15, 5))

ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')


# Plot Gradients Before Masking
Z = sample_gradients[img_ind].numpy()
ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_title('Gradients Before Masking')
ax1.set_xlim([0, 28])
ax1.set_ylim([0, 28])
ax1.set_zlim([z_min, z_max])

# Plot Gradients After Masking
Z_perturbed = sample_gradients_perturbed[img_ind].numpy()
ax2.plot_surface(X, Y, Z_perturbed, cmap='viridis')
ax2.set_title('Gradients After Masking')
ax2.set_xlim([0, 28])
ax2.set_ylim([0, 28])
ax2.set_zlim([z_min, z_max])

# Plot Gradients Difference'
Z_diff = sample_gradients_diff[img_ind].numpy()
ax3.plot_surface(X, Y, Z_diff, cmap='viridis')
ax3.set_title('Gradients Difference')
ax3.set_xlim([0, 28])
ax3.set_ylim([0, 28])
ax3.set_zlim([z_min, z_max])

# Adjust the padding between the subplots
plt.tight_layout()

# plt.savefig("../../../Module-3-AML_files/Module-3-AML_33_0.pdf",bbox_inches='tight')
# Show the plot
plt.show()