#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 14:50:44 2023

@author: ozgur
"""

from phe import paillier
import random
import numpy as np

alice_pubkey, alice_privkey = paillier.generate_paillier_keypair()

alice_friends = [1, 2, 3, 4, 5]
bob_friends = [3, 4, 5, 6, 7, 8, 2]

alice_enc_friends = [alice_pubkey.encrypt(friend) for friend in alice_friends]

r = random.randint(1, alice_pubkey.n - 1)
bob_results = [(alice_enc_friend - bob_friend) for alice_enc_friend in alice_enc_friends for bob_friend in bob_friends]

idx = 0
for result in bob_results:
    dec_result = alice_privkey.decrypt(result)
    if dec_result == 0:
        found_friend_idx = np.floor(idx / len(bob_friends)).astype(int)
        print("Intersection found:", alice_friends[found_friend_idx])
    idx = idx + 1