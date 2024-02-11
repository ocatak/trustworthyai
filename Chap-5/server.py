import flwr as fl

print(fl.__version__)

# Start Flower server
fl.server.start_server(
  server_address="0.0.0.0:8080",
  config=fl.server.ServerConfig(num_rounds=10),
)