#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 12:22:28 2023

@author: ozgur 
"""
#====================================
# Import libraries
import sympy as sp
from IPython.display import display, Markdown

# Define the variables w1 and w2 as symbols
w1, w2 = sp.symbols('w1 w2')

# Define the function f(w1, w2)
f = 3 * w1**2 + 2 * w1 * w2

# Calculate the gradients symbolically
gradients = [sp.diff(f, w) for w in [w1, w2]]

# Print the original function f
display(Markdown("**Original Function:**"))
display(f)

# Print the values of w1 and w2
display(Markdown("**Values:**"))
display(Markdown("w1 = 5, w2 = 3"))

# Print the symbolic gradients with explanations
for i, gradient in enumerate(gradients):
    display(Markdown(f"**Gradient {i+1}:**"))
    display(gradient)
    display(gradient.subs([(w1, 5), (w2, 3)]))