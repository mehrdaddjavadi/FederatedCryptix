import asyncio
import logging
import pickle
import websockets
import numpy as np
from encryption.encryption import create_context, encrypt_weights, decrypt_weights

class ClientDevice:
    def __init__(self, client_id, model):
        self.client_id = client_id
        self.model = model
        self.context = create_context()
        self.connection = None
        self.logger = logging.getLogger(__name__)

    async def connect_to_server(self, uri):
        self.connection = await websockets.connect(uri)
        self.logger.info(f"Client {self.client_id} connected to server")

    async def receive_weights(self):
        async for message in self.connection:
            data = pickle.loads(message)
            self.logger.info(f"Client {self.client_id} received data: {data}")
            if 'weights' in data:
                self.model.set_weights(decrypt_weights(self.context, data['weights']))
                self.logger.info(f"Client {self.client_id} set weights")

    async def send_weights(self):
        encrypted_weights = encrypt_weights(self.context, self.model.get_weights())
        await self.connection.send(pickle.dumps({'weights': encrypted_weights}))

    async def train_model(self, x_train, y_train, training_config):
        self.model.train(x_train, y_train, training_config)
        self.logger.info(f"Client {self.client_id} trained model")

    async def send_data_request(self):
        await self.connection.send(pickle.dumps({'data_request': True}))
        self.logger.info(f"Client {self.client_id} sent data request")
