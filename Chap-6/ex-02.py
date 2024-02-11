#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 14:50:44 2023

@author: ozgur and murat
"""

# Import libraries
from phe import paillier
import random
import numpy as np

# Generate Paillier Key Pair for Alice
alice_pubkey, alice_privkey = paillier.generate_paillier_keypair()

# Define Alice's and Bob's Friend Lists
alice_friends = [1, 2, 3, 4, 5]
bob_friends = [3, 4, 5, 6, 7, 8, 2]

# Encrypt Alice's Friend List
alice_enc_friends = [alice_pubkey.encrypt(friend) for friend in alice_friends]

# Compute the result of subtracting bob_friend from alice_enc_friend
r = random.randint(1, alice_pubkey.n - 1)
bob_results = [(alice_enc_friend - bob_friend) for alice_enc_friend in alice_enc_friends for bob_friend in bob_friends]

# Check Intersection
idx = 0
for result in bob_results:
    dec_result = alice_privkey.decrypt(result)
    if dec_result == 0:
        found_friend_idx = np.floor(idx / len(bob_friends)).astype(int)
        print("Intersection found:", alice_friends[found_friend_idx])
    idx = idx + 1