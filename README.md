# FederatedCryptix

## Overview

This repository contains a federated learning framework that enables collaborative training of machine learning models across multiple devices while preserving data privacy.

## Folder Structure

- **config/**: Configuration files for models and training parameters.
- **encryption/**: Contains encryption and decryption logic using TenSEAL.
- **models/**: Model implementations for TensorFlow and PyTorch.
- **communication/**: Manages WebSocket communication.
- **server/**: Central server implementation.
- **clients/**: Client device implementation.
- **utils/**: Utility functions and decorators.
- **logs/**: Log files for server and clients.
- **main_server.py**: Entry point to start the central server.
- **main_client.py**: Entry point to start a client device.

## Getting Started

### Central Server

1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
