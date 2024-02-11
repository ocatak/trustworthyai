#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:07:44 2023

@author: ozgur
"""
# Import libraries
import flwr as fl

# Start Flower server
fl.server.start_server(
  server_address="0.0.0.0:8080",
  config=fl.server.ServerConfig(num_rounds=10),
)