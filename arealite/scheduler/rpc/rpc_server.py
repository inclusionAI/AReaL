import argparse
import pickle
import threading
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from arealite.scheduler.utils import deserialize_with_metadata
import cloudpickle


class EngineRPCServer(BaseHTTPRequestHandler):
    engine = None

    def do_POST(self):
        try:
            length = int(self.headers["Content-Length"])
            data = self.rfile.read(length)
            if self.path == "/create_engine":
                engine_obj, init_args = cloudpickle.loads(data)
                EngineRPCServer.engine = engine_obj
                EngineRPCServer.engine.initialize(init_args)
                print("Engine created and initialized on RPC server.")
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"OK")
            elif self.path == "/call":
                if EngineRPCServer.engine is None:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b"Engine not initialized")
                    logging.error("Call received but engine not initialized.")
                    return
                action, args, kwargs = cloudpickle.loads(data)
                method = getattr(EngineRPCServer.engine, action)
                logging.info(
                    f"RPC server calling engine method: {action} - {args} - {kwargs}"
                )
                result = method(*args, **kwargs)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(cloudpickle.dumps(result))
            else:
                self.send_response(404)
                self.end_headers()
        except Exception as e:
            import traceback

            logging.error(f"Exception in do_POST: {e}\n{traceback.format_exc()}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(
                f"Exception: {e}\n{traceback.format_exc()}".encode("utf-8")
            )


def start_rpc_server(port):
    server = HTTPServer(("0.0.0.0", port), EngineRPCServer)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    file_handler = logging.FileHandler("/tmp/output.log", mode="a")  # mode="w" 会覆盖文件
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    print(f"About to start RPC server on {args.port}")
    start_rpc_server(args.port)
    print(f"RPC server running on port {args.port}")
