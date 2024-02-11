#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 10:03:30 2023

@author: ozgur and murat
"""
# Import Libraries
import numpy as np
from Pyfhel import Pyfhel

# Generate CKKS Key Pair
HE = Pyfhel()           # Creating empty Pyfhel object
ckks_params = {
    'scheme': 'CKKS',   # can also be 'ckks'
    'n': 2**14,         # Polynomial modulus degree. For CKKS, n/2 values can be
                        #  encoded in a single ciphertext.
                        #  Typ. 2^D for D in [10, 15]
    'scale': 2**30,     # All the encodings will use it for float->fixed point
                        #  conversion: x_fix = round(x_float * scale)
                        #  You can use this as default scale or use a different
                        #  scale on each operation (set in HE.encryptFrac)
    'qi_sizes': [60, 30, 30, 30, 60] # Number of bits of each prime in the chain.
                        # Intermediate values should be  close to log2(scale)
                        # for each operation, to have small rounding errors.
}
HE.contextGen(**ckks_params)  # Generate context for ckks scheme
HE.keyGen()             # Key Generation: generates a pair of public/secret keys
#HE.rotateKeyGen()

# Generate Random Values
RANDOM_VAL_LENGTH = 5
ROUNDING_LENGTH = 5
val1 = np.random.rand(RANDOM_VAL_LENGTH).round(ROUNDING_LENGTH)
val2 = np.random.rand(RANDOM_VAL_LENGTH).round(ROUNDING_LENGTH)

# Display Plain Text Values
print('*' * 50)
print('val1 \t:', val1, '\nval2 \t:', val2)

# Display Plain Domain Operations (Addition \& Multiplication) and Results
print('*' * 50)
print('Plain domain addition operation\nval1 + val2 :', (val1 + val2))
print('Plain domain multiplication operation\nval1 * val2 :', (val1 * val2))


# Encrypt and Display Encrypted Values
val1_enc = HE.encryptFrac(val1) # Encryption makes use of the public key
val2_enc = HE.encryptFrac(val2) # For integers, encryptInt function is used.

print('*' * 50)
print('Encrypted val1 \t: len(', len(str(val1_enc.to_bytes())), ')', str(val1_enc.to_bytes())[:40], '...', str(val1_enc.to_bytes())[-40:])
print('Encrypted val2 \t: len(', len(str(val2_enc.to_bytes())), ')', str(val2_enc.to_bytes())[:40], '...', str(val2_enc.to_bytes())[-40:])


# Perform Homomorphic Addition Operations and Display Encrypted Results
sum1_enc = val1_enc + val2_enc
sum2_enc = val1_enc + val2

print('Encrypted sum1_enc \t: len(', len(str(sum1_enc.to_bytes())), ')', str(sum1_enc.to_bytes())[:40], '...',str(sum1_enc.to_bytes())[-40:])
print('Encrypted sum2_enc \t: len(', len(str(sum2_enc.to_bytes())), ')', str(sum2_enc.to_bytes())[:50], '...',str(sum2_enc.to_bytes())[-40:])


# Decrypt and Display Results
print('*' * 50)
sum1_dec = HE.decryptFrac(sum1_enc).round(ROUNDING_LENGTH)
sum2_dec = HE.decryptFrac(sum2_enc).round(ROUNDING_LENGTH)

print('Decrypted sum1 \t:', sum1_dec[:RANDOM_VAL_LENGTH])
print('Decrypted sum2 \t:', sum2_dec[:RANDOM_VAL_LENGTH])

# Perform Homomorphic Multiplication Operations
print('*' * 50)
mult1_enc = val1_enc * val2_enc
mult2_enc = val1_enc * val2

print('Encrypted mult1_enc \t: len(', len(str(mult1_enc.to_bytes())), ')', str(mult1_enc.to_bytes())[:40], '...',str(mult1_enc.to_bytes())[-40:])
print('Encrypted mult2_enc \t: len(', len(str(mult2_enc.to_bytes())), ')', str(mult2_enc.to_bytes())[:50], '...',str(mult2_enc.to_bytes())[-40:])

# Perform homomorphic multiplication operations
print('*' * 50)
mult1_dec = HE.decryptFrac(mult1_enc).round(ROUNDING_LENGTH)
mult2_dec = HE.decryptFrac(mult2_enc).round(ROUNDING_LENGTH)

print('Decrypted sum1 \t:', mult1_dec[:RANDOM_VAL_LENGTH])
print('Decrypted sum2 \t:', mult2_dec[:RANDOM_VAL_LENGTH])