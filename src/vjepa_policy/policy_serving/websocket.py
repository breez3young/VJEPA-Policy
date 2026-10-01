"""Small policy transport compatible with the public openpi-client package.

The wire protocol sends a metadata map on connection, then accepts msgpack
observations and returns msgpack actions. NumPy arrays use openpi-client's codec.
"""

import asyncio
import logging
from http import HTTPStatus

from openpi_client import msgpack_numpy
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed


class WebsocketPolicyServer:
    def __init__(self, policy, host="127.0.0.1", port=10000, metadata=None):
        self.policy = policy
        self.host = host
        self.port = port
        self.metadata = dict(metadata or {})

    @staticmethod
    def health(connection, request):
        if request.path == "/healthz":
            return connection.respond(HTTPStatus.OK, "ready\n")
        return None

    async def handle(self, connection):
        codec = msgpack_numpy.Packer()
        try:
            await connection.send(codec.pack(self.metadata))
            async for message in connection:
                try:
                    if not isinstance(message, bytes):
                        raise ValueError("Expected a binary msgpack observation")
                    observation = msgpack_numpy.unpackb(message)
                    if not isinstance(observation, dict):
                        raise ValueError("Observation must be a mapping")
                    # Keep inference serialized: a single policy can own mutable
                    # history/RNG state and is not assumed to be thread-safe.
                    prediction = self.policy.infer(observation)
                    await connection.send(codec.pack(prediction))
                except ConnectionClosed:
                    return
                except Exception as error:
                    logging.exception("Policy request failed")
                    # OpenPI clients interpret a text response as an error.
                    await connection.send(f"{type(error).__name__}: {error}")
                    await connection.close(code=1011, reason="Policy request failed")
                    return
        except ConnectionClosed:
            return

    async def run(self):
        async with serve(
            self.handle, self.host, self.port,
            process_request=self.health, compression=None, max_size=None,
        ) as server:
            await server.serve_forever()

    def serve_forever(self):
        asyncio.run(self.run())
