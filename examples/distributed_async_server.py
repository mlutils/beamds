from threading import Thread

from src.beam.misc import BeamFakeAlg
from src.beam.distributed import AsyncServer
from src.beam import logger

# async def websocket_handler(ws):
#     # Wait for the client to send its client_id
#
#     client_id = await ws.recv()
#     logger.info(f"New WebSocket client connected: {client_id}")
#     self.ws_clients[client_id] = ws
#     await ws.wait_closed()


if __name__ == '__main__':

    # import asyncio
    # import websockets
    #
    # async def echo(websocket, path):
    #     async for message in websocket:
    #         # You can add additional processing to the message here
    #         echo_message = f"Echo: {message}"
    #         await websocket.send(echo_message)

    # def run_ws_server(ws_server):
    #
    #     logger.info("Starting...")
    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)
    #     loop.run_until_complete(ws_server)
    #     loop.run_forever()
    #
    #     # Close the loop when done
    #     loop.close()
    #
    # ws = websockets.serve(echo, "127.0.0.1", 8765)
    #
    # # wst = Thread(target=run_ws_server, args=(ws,))
    # # wst.start()
    # # wst.join()
    #
    # asyncio.get_event_loop().run_until_complete(ws)
    # asyncio.get_event_loop().run_forever()

    # def run_ws_server(host, port):
    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)
    #
    #     start_server = websockets.serve(echo, host, port)
    #     loop.run_until_complete(start_server)
    #     loop.run_forever()
    #
    #
    # wst = Thread(target=run_ws_server, args=("127.0.0.1", 8765))
    # wst.start()
    # wst.join()
    #
    # print('done!')

    # Create a fake algorithm
    fake_alg = BeamFakeAlg(sleep_time=10., variance=0.5, error_rate=0.1)

    def postrun(**kwargs):
        logger.info(f'Server side callback: Task has completed for {kwargs}')

    server = AsyncServer(fake_alg, postrun=postrun, port=28450, ws_port=28402,
                         )
    server.run()
    print('Done!')
