import asyncio
from server.central_server import CentralServer

server = CentralServer()
asyncio.run(server.run_server())
