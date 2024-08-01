import asyncio
import logging
import numpy as np
from communication.connection import ConnectionServer
from websockets.exceptions import ConnectionClosedError
from encryption.encryption import create_context, encrypt_weights, decrypt_weights

class CentralServer:
    def __init__(self, connection_type='websocket', host='0.0.0.0', port=8089):
        self.model_weights = None
        self.lock = asyncio.Lock()
        self.clients = set()
        self.logger = logging.getLogger(__name__)
        self.connection = ConnectionServer(connection_type, host, port, self.handle_client)
        self.context = create_context()

    async def run_server(self):
        self.logger.info("Central Server is starting...")
        await self.connection.start()

    async def handle_client(self, websocket, path):
        client_id = len(self.clients) + 1
        self.clients.add(client_id)
        self.logger.info(f"Central Server: Client {client_id} connected")
        try:
            while True:
                message = await websocket.recv()
                data = pickle.loads(message)
                self.logger.info(f"Received from client {client_id}: {data}")

                if 'weights' in data:
                    await self.transmit_weights(data['weights'])
                elif 'data_request' in data:
                    data = await self.get_data_from_client(client_id)
                    await self.send_data_to_client(client_id, {'data': data})
        except ConnectionClosedError:
            self.logger.info(f"Central Server: Client {client_id} disconnected")
            self.clients.remove(client_id)

    async def transmit_weights(self, weights):
        async with self.lock:
            self.model_weights = weights
            encrypted_weights = encrypt_weights(self.context, self.model_weights)
            await asyncio.gather(*[self.send_weights(client_id) for client_id in self.clients])
            self.logger.info("Transmitted encrypted weights to clients")

    async def send_weights(self, client_id):
        if self.model_weights is not None:
            await self.connection.send(client_id, {'weights': self.model_weights})

    async def send_data_to_client(self, client_id, data):
        self.logger.info(f"Central Server: Sending data to client {client_id}")
        await self.connection.send(client_id, data)

    async def get_data_from_client(self, client_id):
        self.logger.info(f"Central Server: Requesting data from client {client_id}")
        await asyncio.sleep(1)  # Simulating data retrieval
        return np.random.rand(10, 3072)  # Simulated data

    def query_active_learning(self, unlabeled_data, model):
        predictions = model.predict(unlabeled_data)
        uncertainty = predictions.max(axis=1)
        selected_indices = np.argsort(uncertainty)[:5]
        return selected_indices
