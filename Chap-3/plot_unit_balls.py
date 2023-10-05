#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 12:14:57 2023

@author: ozgur
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_unit_ball(norm, subplot_index):
    ax = fig.add_subplot(2, 3, subplot_index, projection='3d')  # Adjust the subplot dimensions as needed

    # Generate points on the unit sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Compute distances from the origin using the specified norm
    distances = np.power(np.power(np.abs(x), norm) + np.power(np.abs(y), norm) + np.power(np.abs(z), norm), 1/norm)

    # Plot the unit ball
    ax.plot_surface(x / distances, y / distances, z / distances, color='b', alpha=0.05, antialiased=False)
    
    # Add a central point 'x'
    ax.scatter([0], [0], [0], color='red', s=30, label='x')

    # Set plot limits and labels based on the figure size
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    max_range = max(max(xlim), max(ylim), max(zlim))
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the title based on the norm used
    if norm == 1:
        norm_name = '1-norm (Manhattan norm)'
    elif norm == 2:
        norm_name = '2-norm (Euclidean norm)'
    elif norm == np.inf:
        norm_name = 'Infinity norm'
    else:
        norm_name = f'{norm}-norm'

    ax.set_title(f'Unit Ball ({norm_name})')

# Create a figure with subplots
fig = plt.figure(figsize=(12, 8))  # Adjust the figure size as needed

# Plot the unit balls as subplots
plot_unit_ball(0.4, 1)  # 0.5-norm
plot_unit_ball(0.8, 2)  # 0.5-norm
plot_unit_ball(1, 3)    # 1-norm (Manhattan norm)
plot_unit_ball(2, 4)    # 2-norm (Euclidean norm)
plot_unit_ball(3, 5)    # 3-norm
plot_unit_ball(100, 6)  # Infinity norm

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Adjust the horizontal and vertical spacing as needed

#plt.savefig("../../../Module-3-AML_files/Module-3-AML_1_0.pdf",bbox_inches='tight')

# Show the plot
plt.show()