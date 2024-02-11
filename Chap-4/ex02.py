#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 19:39:32 2023

@author: ozgur and murat 
"""
#==============================================================================
### Part 1
#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import shap
from tensorflow import keras
from cleverhans.tf2.attacks.momentum_iterative_method import momentum_iterative_method

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32') / 255.0
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32') / 255.0

# Define the model architecture
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=5000, validation_split=0.1)

#==============================================================================
### Part 2
# Create a SHAP explainer
explainer = shap.DeepExplainer(model, (X_test[:1000]))

#Initialize a loop for 5 iterations
for i in range(5):
    # Randomly select the image
    img_idx = np.random.randint(X_test.shape[0])
    sample = X_test[[img_idx]]
    
    # Get SHAP values for the sample
    shap_values = explainer.shap_values(sample)
    
    # Predict the sample and get True Label
    pred_proba = model.predict(sample)
    org_y = y_test[img_idx]
    
    # Format and concatenate information about prediction probabilities associated with different classes
    index_names = np.array([str(x) + "\n" + '{:7.3%}'.format(pred_proba[0][x]) for x in range(10)]).reshape(1, 10)

    # Print the predicted label and the true label associated with an image 
    print("Predicted label :{}\nTrue label :{}".format(pred_proba.argmax(), org_y))
    
    # Plot the SHAP values
    shap.image_plot(shap_values, -sample.reshape(1, 28, 28, 1), index_names, show=False)
    
    # Save the results
    # plt.savefig("Module-4-XAI_7_1.pdf",bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    
 #==============================================================================
 ### Part 3
 # Import libraries  
import numpy as np
import matplotlib.pyplot as plt
import shap
from tensorflow import keras


# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32') / 255.0
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32') / 255.0

# Define the model architecture
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=5000, validation_split=0.1)

# Define variables
epsilon = 0.1
TARGET_CLASS = 5

# Create a SHAP explainer
explainer = shap.DeepExplainer(model, (X_test[:1000]))

#Initialize a loop for 5 iterations
for i in range(5):
    # Randomly select the image
    img_idx = np.random.randint(X_test.shape[0])
    sample = X_test[[img_idx]]
 
    # Apply Adversarial attack
    sample = momentum_iterative_method(model, sample, eps=epsilon, eps_iter=0.01, nb_iter=1000,
                                       decay_factor=1.0, clip_min=0.0, clip_max=1.0, y=[TARGET_CLASS],
                                       targeted=True, sanity_checks=False).numpy()
    
    # Get SHAP values for the sample
    shap_values = explainer.shap_values(sample)
    
    # Predict the sample and get True Label
    pred_proba = model.predict(sample)
    org_y = y_test[img_idx]
    
    # Format and concatenate information about prediction probabilities associated with different classes
    index_names = np.array([str(x) + "\n" + '{:7.3%}'.format(pred_proba[0][x]) for x in range(10)]).reshape(1, 10)
    
    # Print the predicted label and the true label associated with an image 
    print("Predicted label :{}\nTrue label :{}".format(pred_proba.argmax(), org_y))
    
    # Plot the SHAP values
    shap.image_plot(shap_values, -sample.reshape(1, 28, 28, 1), index_names, show=False)
    
    # Save the results
    plt.savefig("Module-4-XAI_7_2.pdf",bbox_inches='tight')
    
    # Show the plot
    plt.show()