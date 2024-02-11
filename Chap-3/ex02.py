#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 12:24:57 2023

@author: ozgur
"""
#====================================
# Import libraries
import tensorflow as tf
import numpy as np

# Create a small dataset with 3 columns
data = np.array([[1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0],
                 [7.0, 8.0, 9.0]])

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

# Define the optimizer and loss function
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
loss_fn = tf.keras.losses.MeanSquaredError()

# Create TensorFlow variables for the inputs and targets
x = tf.Variable(data, dtype=tf.float32)
y = tf.Variable(np.ones((3, 1)), dtype=tf.float32)

# Perform the forward pass
with tf.GradientTape() as tape:
    # Forward pass
    output = model(x)
    # Calculate the loss
    loss = loss_fn(y, output)

# Calculate the gradients using TensorFlow's automatic differentiation
gradients = tape.gradient(loss, x)

# Print the gradients
print("Gradients:")
for i, gradient in enumerate(gradients.numpy()):
    print(f"Column {i+1}: {gradient}")