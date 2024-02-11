#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 14:42:33 2023

@author: ozgur
"""

import tensorflow as tf
from tqdm.notebook import tqdm

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Define the loss function with randomization-based mitigation
def loss_fn(y_true, y_pred):
    # Randomization parameters
    epsilon = 0.1
    sigma = 0.01
    # Perturb the input data
    n = tf.shape(y_true)[0]
    noise = tf.random.normal(shape=(n, 28, 28), mean=0.0, stddev=sigma)
    perturbed_x = tf.slice(x_train, [0, 0, 0], [n, 28, 28]) + epsilon * noise
    # Evaluate the model with the perturbed input data
    y_pred_perturbed = model(perturbed_x)
    # Calculate the cross-entropy loss
    ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
    # Calculate the KL divergence between the original and perturbed predictions
    kl_div = tf.nn.softmax_cross_entropy_with_logits(y_pred, y_pred_perturbed)
    # Return the combined loss
    return tf.reduce_mean(ce_loss + kl_div)

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# Train the model
batch_size = 200
epochs = 10
steps_per_epoch = x_train.shape[0] // batch_size
for epoch in tqdm(range(epochs)):
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    for step in tqdm(range(steps_per_epoch), desc=f'Epoch {epoch+1}/{epochs}', leave=False):
        # Get a batch of training data
        batch_x = x_train[step*batch_size : (step+1)*batch_size]
        batch_y = y_train[step*batch_size : (step+1)*batch_size]
        # Train the model on the batch
        loss, accuracy = model.train_on_batch(batch_x, batch_y)
        epoch_loss += loss
        epoch_accuracy += accuracy
    epoch_loss /= steps_per_epoch
    epoch_accuracy /= steps_per_epoch
    print(f'Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Accuracy={epoch_accuracy:.4f}')

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'Test Loss={test_loss:.4f}, Test Accuracy={test_accuracy:.4f}')