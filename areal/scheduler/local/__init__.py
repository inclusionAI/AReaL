import concurrent.futures
import logging
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from typing import List

from areal.scheduler.base import Scheduler, Worker
from areal.scheduler.rpc.rpc_client import RPCClient
from areal.scheduler.utils import find_free_port, wait_for_port


class LocalScheduler(Scheduler):
    def __init__(self, config):
        super().__init__(config)
        self.procs = []  # Store subprocess objects
        self.engine_workers = defaultdict(list)

    def _build_rpc_client(self, config):
        return RPCClient()

    def create_workers(self, worker_key, scheduler_config, *args, **kwargs):
        num_workers = scheduler_config.get("num_workers", 1)

        # Use a thread pool to launch and register workers in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Prepare future tasks for each worker
            futures = [
                executor.submit(self._start_single_worker, worker_key, i)
                for i in range(num_workers)
            ]

            for future in concurrent.futures.as_completed(futures):
                self.engine_workers[worker_key].append(future.result())

    def _start_single_worker(self, worker_key, worker_index):
        """Helper function to start and register one worker."""
        rpc_port, engine_port = find_free_port(), find_free_port()

        # Start the subprocess
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "areal.scheduler.rpc.rpc_server",
                "--port",
                str(rpc_port),
            ]
        )
        self.procs.append(proc)  # Store the process object to manage it later

        # Register the worker with the RPC client
        current_time = datetime.now().strftime("%H%M%S%f")  # 格式化为 时分秒毫秒
        worker_id = f"{worker_key}-{worker_index}-{current_time}"
        self.worker_map[worker_id] = ("127.0.0.1", rpc_port)
        self.rpc_client.register(worker_id, "127.0.0.1", rpc_port)

        logging.info(
            f"Launched worker {worker_id} on rpc_port {rpc_port}, engine_port {engine_port}"
        )
        return Worker(id=worker_id, ip="127.0.0.1", ports=[engine_port])

    def get_workers(self, worker_key, timeout: float = 60.0) -> List[Worker]:
        """Waits for all workers to be ready in parallel."""
        engine_list = self.engine_workers[worker_key]
        if not engine_list:
            logging.info("No workers to wait for.")
            return []

        logging.info(
            f"Waiting for all {len(engine_list)}  {worker_key} workers to be ready (total timeout: {timeout}s)..."
        )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(engine_list)
        ) as executor:
            future_to_worker = {
                executor.submit(
                    wait_for_port,
                    engine.ip,
                    self.worker_map[engine.id][1],  # rpc port
                    timeout=timeout,
                ): engine.id
                for engine in engine_list
            }

            done, not_done = concurrent.futures.wait(
                future_to_worker.keys(),
                timeout=timeout,
                return_when=concurrent.futures.ALL_COMPLETED,
            )

            all_ready = True

            # 处理已完成的任务
            for future in done:
                worker_id = future_to_worker[future]
                ip, port = self.worker_map[worker_id]
                try:
                    if future.result():
                        logging.info(
                            f"✅ Worker {worker_id} is ready on rpc port {port}."
                        )
                    else:
                        logging.warning(
                            f"⚠️ Worker {worker_id} on port {port} failed to become ready within its individual timeout."
                        )
                        all_ready = False
                except Exception as e:
                    logging.error(
                        f"❌ An exception occurred while waiting for worker {worker_id}: {e}"
                    )
                    all_ready = False

            if not_done:
                logging.error(
                    f"❌ Global timeout of {timeout}s reached. The following workers did not complete:"
                )
                for future in not_done:
                    worker_id = future_to_worker[future]
                    logging.error(
                        f"  - Worker {worker_id} at {self.worker_map[worker_id]}"
                    )
                all_ready = False

            if all_ready:
                logging.info(f"All {worker_key} workers are ready.")
                return self.engine_workers[worker_key]
            else:
                logging.error("Not all {worker_key} workers became ready in time.")
                return []

    def delete_workers(self):
        """Properly terminate all worker subprocesses."""
        logging.info(
            "Shutting down local scheduler and terminating worker processes..."
        )
        for proc in self.procs:
            proc.terminate()  # or proc.kill()

        for proc in self.procs:
            proc.wait(timeout=5)  # Wait for processes to terminate
        logging.info("All worker processes terminated.")

    # Other methods remain the same
    def create_engine(self, worker_id, engine_class, init_args):
        # print(f"Creating engine on worker {worker_id}")
        return self.rpc_client.create_engine(worker_id, engine_class, init_args)

    def call_engine(self, worker_id, method, *args, **kwargs):
        print(f"Calling '{method}' on worker {worker_id} with arg: {args} {kwargs}")
        return self.rpc_client.call_engine(worker_id, method, *args, **kwargs)
