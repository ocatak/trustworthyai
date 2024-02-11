#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:41:55 2024

@author: ozgur
"""

from phe import paillier
import numpy as np

# Paillier Encryption Setup
# Here, we initialize the Paillier cryptosystem. This step involves generating
# a pair of keys - a public key and a private key. The public key will be used
# by Bob to encrypt his data, and the private key will be used by Bob to decrypt
# the results.
public_key, private_key = paillier.generate_paillier_keypair()

# Alice's Part (Model Owner)
# Alice has a trained logistic regression model. For this demonstration, we're
# using predefined weights and an intercept. In a real-world scenario, these
# would be the learned parameters from Alice's model training process.
weights = np.array([0.5, -0.3])  # Example weights
intercept = 0.1  # Example intercept

# Logistic function definition
# This is the logistic function used in logistic regression for binary
# classification. It maps any input value to a value between 0 and 1, which
# is interpreted as the probability of belonging to a particular class.
def logistic_function(z):
    return 1 / (1 + np.exp(-z))

# Encrypted prediction function
# This function takes the model's weights, intercept, and Bob's encrypted data
# to compute the encrypted prediction. The prediction is done using a linear
# combination (dot product) of the weights and the data, to which the intercept
# is then added. The entire computation is performed on encrypted data, thus
# preserving privacy.
def encrypted_prediction(weights, intercept, encrypted_data):
    # Encrypted dot product of weights and data
    encrypted_dot_product = sum([w * x for w, x in zip(weights, encrypted_data)])
    # Adding the intercept to the dot product
    encrypted_linear_combination = encrypted_dot_product + intercept
    return encrypted_linear_combination

# Bob's Part (Data Owner)
# Bob has some personal data that he wants to classify using Alice's model.
# However, he does not want to reveal his data to Alice. For this demonstration,
# we're using a small array to represent Bob's data.
bob_data = np.array([1.2, 0.4])  # Example data

# Bob encrypts his data using Alice's public key. This ensures that his data
# stays confidential, and only encrypted data is shared with Alice.
encrypted_data = [public_key.encrypt(x) for x in bob_data]

# Classification
# Alice receives the encrypted data from Bob. She then computes the encrypted
# prediction using her model's weights and intercept. This computation is
# entirely performed on encrypted data, ensuring that Alice does not gain
# access to Bob's actual data.
encrypted_result = encrypted_prediction(weights, intercept, encrypted_data)

# Decryption
# After receiving the encrypted result from Alice, Bob decrypts it using his
# private key. This step reveals the classification result (in an encrypted form)
# of his data, without Alice ever having access to the actual data.
decrypted_result = private_key.decrypt(encrypted_result)

# Final Prediction
# Bob applies the logistic function locally to convert the decrypted result
# into a probability. He then classifies the data based on this probability.
# For instance, if the probability is 0.5 or higher, the data is classified
# into one class (e.g., 'spam'), otherwise, it's classified into the other
# (e.g., 'not spam').
prediction = logistic_function(decrypted_result)
predicted_class = 1 if prediction >= 0.5 else 0

# Output the classification result
print("Predicted Class:", predicted_class)
