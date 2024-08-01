import asyncio
import websockets
import pickle
import logging

class ConnectionServer:
    def __init__(self, connection_type, host, port, handler):
        self.host = host
        self.port = port
        self.handler = handler
        self.server = None

    async def start(self):
        self.server = await websockets.serve(self.handler, self.host, self.port)
        await self.server.wait_closed()

    async def send(self, client_id, message):
        await self.handler.send(client_id, message)

    async def receive(self, client_id):
        return await self.handler.receive(client_id)

class ConnectionClient:
    def __init__(self, uri):
        self.uri = uri
        self.connection = None

    async def connect(self):
        self.connection = await websockets.connect(self.uri)

    async def send(self, message):
        await self.connection.send(message)

    async def receive(self):
        return await self.connection.recv()
