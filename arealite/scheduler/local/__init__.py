import subprocess
import sys
import logging
import os
import inspect
import concurrent.futures
from arealite.scheduler.base import Scheduler, Worker
from arealite.scheduler.rpc.rpc_client import RPCClient
from arealite.scheduler.utils import find_free_port, wait_for_port


import abc
import logging
from typing import Any, List


class LocalScheduler(Scheduler):
    def __init__(self, config):
        super().__init__(config)
        self.procs = []  # Store subprocess objects
        self.worker_infos = []

    def _build_rpc_client(self, config):
        return RPCClient()

    def create_workers(self, scheduler_config, *args, **kwargs):
        num_workers = scheduler_config.get("num_workers", 1)

        # Use a thread pool to launch and register workers in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Prepare future tasks for each worker
            futures = [
                executor.submit(self._start_single_worker, i)
                for i in range(num_workers)
            ]

            for future in concurrent.futures.as_completed(futures):
                self.worker_infos.append(future.result())

    def _start_single_worker(self, worker_index):
        """Helper function to start and register one worker."""
        rpc_port, engine_port = find_free_port(), find_free_port()

        # Start the subprocess
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "arealite.scheduler.rpc.rpc_server",
                "--port",
                str(rpc_port),
            ]
        )
        self.procs.append(proc)  # Store the process object to manage it later

        # Register the worker with the RPC client
        worker_id = f"local-{worker_index}"
        self.worker_map[worker_id] = ("127.0.0.1", rpc_port)
        self.rpc_client.register(worker_id, "127.0.0.1", rpc_port)

        logging.info(
            f"Launched worker {worker_id} on rpc_port {rpc_port}, engine_port {engine_port}"
        )
        return (worker_id, "127.0.0.1", engine_port)

    def get_workers(self, timeout: float = 60.0) -> List[Worker]:
        """Waits for all workers to be ready in parallel."""
        if not self.worker_map:
            logging.info("No workers to wait for.")
            return True

        logging.info(
            f"Waiting for all {len(self.worker_map)} workers to be ready (total timeout: {timeout}s)..."
        )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.worker_map)
        ) as executor:
            future_to_worker = {
                executor.submit(wait_for_port, ip, port, timeout=timeout): worker_id
                for worker_id, (ip, port) in self.worker_map.items()
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
                        logging.info(f"✅ Worker {worker_id} is ready on port {port}.")
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
                logging.info("All workers are ready.")
                return self.worker_infos
            else:
                logging.error("Not all workers became ready in time.")
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
        logging.info(f"Creating engine on worker {worker_id}")
        return self.rpc_client.create_engine(worker_id, engine_class, init_args)

    def call_engine(self, worker_id, method, *args, **kwargs):
        logging.info(
            f"Calling '{method}' on worker {worker_id} with arg: {args} {kwargs}"
        )
        return self.rpc_client.call_engine(worker_id, method, *args, **kwargs)
