import asyncio
import heapq
import threading
from abc import ABC, abstractmethod

from areal.utils import logging

logger = logging.getLogger("SchedulePolicy")


class SchedulePolicy(ABC):
    DEFAULT_MAX_CONCURRENT_PER_WORKER = 256

    def __init__(self, workers: list, max_num_seqs: int):
        if not workers:
            raise RuntimeError("No workers available to choose from.")
        self._chosen_worker_dict = {}
        self.max_concurrent_per_worker = (
            max_num_seqs if max_num_seqs else self.DEFAULT_MAX_CONCURRENT_PER_WORKER
        )
        self._lock = asyncio.Lock()
        self._idle_event = asyncio.Event()

    def get_schedule_name(self):
        return self.__class__.__name__

    @staticmethod
    def get_current_running_name():
        return f"{threading.current_thread().name}-{asyncio.current_task().get_name()}"

    async def __aenter__(self):
        chosen_worker = await self.choose_worker()
        self._chosen_worker_dict[self.get_current_running_name()] = chosen_worker
        return chosen_worker

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        chosen_worker = self._chosen_worker_dict.get(
            self.get_current_running_name(), None
        )
        if chosen_worker:
            await self.release_worker(chosen_worker)
            self._chosen_worker_dict[self.get_current_running_name()] = None
        return exc_type is None

    @abstractmethod
    async def choose_worker(self):
        pass

    @abstractmethod
    async def release_worker(self, _chosen_worker):
        pass


class LeastRequestPrioritySchedulePolicy(SchedulePolicy):
    def __init__(self, workers: list, max_num_seqs: int):
        super().__init__(workers, max_num_seqs)
        self.current_process_requests_state = [
            [0, (worker.id, worker)] for worker in workers
        ]
        heapq.heapify(self.current_process_requests_state)
        logger.info(
            f"{self.get_schedule_name()} init done ,{self.max_concurrent_per_worker=} {self.current_process_requests_state=}"
        )

    async def choose_worker(self):
        chosen_worker = None
        while chosen_worker is None:
            async with self._lock:
                highest_priority_worker = self.current_process_requests_state[0][1][1]
                if (
                    self.current_process_requests_state[0][0]
                    < self.max_concurrent_per_worker
                ):
                    logger.debug(
                        f"{self.get_current_running_name()} chooses worker: {highest_priority_worker.id}"
                    )
                    self.current_process_requests_state[0][0] += 1
                    heapq.heapify(self.current_process_requests_state)
                    chosen_worker = highest_priority_worker
            if chosen_worker is None:
                logger.debug(
                    f"{self.get_current_running_name()} is waiting to choose worker..."
                )
                await self._idle_event.wait()
                self._idle_event.clear()
                logger.debug(
                    f"{self.get_current_running_name()} is notified to choose worker..."
                )
        return chosen_worker

    async def release_worker(self, worker):
        async with self._lock:
            for i, (process_requests_count, (worker_id, _)) in enumerate(
                self.current_process_requests_state
            ):
                if worker.id == worker_id:
                    self.current_process_requests_state[i][0] = max(
                        0, process_requests_count - 1
                    )
                    heapq.heapify(self.current_process_requests_state)
                    self._idle_event.set()
                    logger.debug(
                        f"{self.get_current_running_name()} has released worker {worker.id=} ..."
                    )
                    return
            raise RuntimeError(f"No workers available to release.{worker.id=}")


class RoundRobinSchedulePolicy(SchedulePolicy):
    def __init__(self, workers: list, max_num_seqs: int, no_block: bool = True):
        super().__init__(workers, max_num_seqs)
        self.current_process_requests_state = [
            [0, (worker.id, worker)] for worker in workers
        ]
        self.index = 0
        self.no_block = no_block
        logger.info(
            f"{self.get_schedule_name()} init done ,{self.no_block=} {self.max_concurrent_per_worker=} {self.current_process_requests_state=}"
        )

    async def choose_worker(self):
        if self.no_block:
            return await self._choose_worker_no_block()

        chosen_worker = None
        while chosen_worker is None:
            async with self._lock:
                highest_priority_worker = self.current_process_requests_state[
                    self.index
                ][1][1]
                if (
                    self.current_process_requests_state[self.index][0]
                    < self.max_concurrent_per_worker
                ):
                    logger.debug(
                        f"{self.get_current_running_name()} chooses worker: {highest_priority_worker.id}"
                    )
                    self.current_process_requests_state[self.index][0] += 1
                    chosen_worker = highest_priority_worker
                    self.index = (self.index + 1) % len(
                        self.current_process_requests_state
                    )
            if chosen_worker is None:
                logger.debug(
                    f"{self.get_current_running_name()} is waiting to choose worker..."
                )
                await self._idle_event.wait()
                self._idle_event.clear()
                logger.debug(
                    f"{self.get_current_running_name()} is notified to choose worker..."
                )
        return chosen_worker

    async def release_worker(self, worker):
        if self.no_block:
            return
        async with self._lock:
            for i, (process_requests_count, (worker_id, _)) in enumerate(
                self.current_process_requests_state
            ):
                if worker.id == worker_id:
                    self.current_process_requests_state[i][0] = max(
                        0, process_requests_count - 1
                    )
                    self._idle_event.set()
                    logger.debug(
                        f"{self.get_current_running_name()} has released worker {worker.id=} ..."
                    )
                    return
            raise RuntimeError(f"No workers available to release.{worker.id=}")

    async def _choose_worker_no_block(self):
        highest_priority_worker = self.current_process_requests_state[self.index][1][1]
        self.index = (self.index + 1) % len(self.current_process_requests_state)
        return highest_priority_worker
