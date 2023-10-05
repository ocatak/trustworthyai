#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 19:13:34 2023

@author: ozgur
"""

import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

def to_rgb(x):
    x_rgb = np.zeros((x.shape[0], 28, 28, 3))
    for i in range(3):
        x_rgb[..., i] = x[..., 0]
    return x_rgb

x_train = to_rgb(x_train)
x_test = to_rgb(x_test)


model = keras.Sequential(
    [
     Conv2D(16, 3, activation='relu', input_shape=(28, 28, 3)),
     MaxPooling2D(),
     Flatten(),
     Dense(10)
    ]
)

model.compile(
  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  optimizer=keras.optimizers.Adam(),
  metrics=['accuracy']
)

model.fit(
        x_train, 
        y_train, 
        epochs=20, 
        batch_size=10000, 
        validation_data=(x_test, y_test))

### PART 2

from tensorflow import keras
from lime.wrappers.scikit_image import SegmentationAlgorithm
from tqdm.notebook import tqdm
from skimage.color import label2rgb


# Create the explainer object
explainer = lime_image.LimeImageExplainer(random_state=42, verbose=False)
segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)

# Create a figure with subplots
fig, axes = plt.subplots(2, 10, figsize=(12, 4))

input_img_idx_list = []
for i, ax in tqdm(enumerate(axes.flat)):
    # Randomly select an image
    input_img_idx = np.random.randint(x_test.shape[0])
    input_img = x_test[input_img_idx:input_img_idx + 1]
    
    # Get the ground truth and predicted labels
    org_label = y_test[input_img_idx]
    pred_label = model.predict(input_img, verbose=0)[0].argmax()
    
    # Explain the instance
    explanation = explainer.explain_instance(x_test[input_img_idx], model.predict,
                                             top_labels=10, hide_color=0, num_samples=100,
                                             segmentation_fn=segmenter)
    
    image, mask = explanation.get_image_and_mask(
        model.predict(input_img).argmax(axis=1)[0],
        positive_only=True, hide_rest=False,
        num_features=10, min_weight=0.01
    )
    
    # Plot the original image with the SHAP explanation
    # ax.imshow(mark_boundaries(image, mask))
    ax.imshow(label2rgb(mask, image, bg_label=0), interpolation='nearest')
    
    ax.set_title('GT:' + str(org_label) + ' Pred:' + str(pred_label))
    ax.axis('off')
    
    input_img_idx_list.append(input_img_idx)

fig.tight_layout()


plt.savefig("../../../Module-4-XAI_files/Module-4-XAI_3_22.pdf",bbox_inches='tight')

plt.show()

### PART 3
from cleverhans.tf2.attacks.momentum_iterative_method import momentum_iterative_method

epsilon = 0.3
TARGET_CLASS = 5

# Create a figure with subplots
fig, axes = plt.subplots(2, 10, figsize=(12, 4))

for i, ax in tqdm(enumerate(axes.flat)):
    # Randomly select an image
    input_img_idx = input_img_idx_list[i]
    input_img = x_test[input_img_idx:input_img_idx + 1]
    
    input_img = momentum_iterative_method(model, input_img, eps=epsilon, eps_iter=0.01, nb_iter=1000,
                                          decay_factor=1.0, clip_min=0.0, clip_max=1.0, y=[TARGET_CLASS],
                                          targeted=True, sanity_checks=False)
    
    # Get the ground truth and predicted labels
    org_label = y_test[input_img_idx]
    pred_label = model.predict(input_img, verbose=0)[0].argmax()
    
    # Explain the instance
    explanation = explainer.explain_instance(x_test[input_img_idx], model.predict,
                                             top_labels=10, hide_color=0, num_samples=100,
                                             segmentation_fn=segmenter)
    
    image, mask = explanation.get_image_and_mask(
        model.predict(input_img).argmax(axis=1)[0],
        positive_only=True, hide_rest=False,
        num_features=10, min_weight=0.01
    )
    
    # Plot the original image with the SHAP explanation
    # ax.imshow(mark_boundaries(image, mask))
    ax.imshow(label2rgb(mask,image, bg_label = 1), interpolation = 'nearest')
    ax.set_title('GT:' + str(org_label) + ' Pred:' + str(pred_label))
    ax.axis('off')
    
fig.tight_layout()

plt.savefig("../../../Module-4-XAI_files/Module-4-XAI_4_21.pdf",bbox_inches='tight')

plt.show()