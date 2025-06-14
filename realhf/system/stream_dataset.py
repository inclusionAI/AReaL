import queue
import sys
import threading
import time
from typing import Any, List, Optional

from torch.utils.data import ConcatDataset, Dataset

from realhf.api.core.config import DatasetAbstraction
from realhf.api.core.data_api import (
    DatasetUtility,
    SequenceSample,
    make_dataset,
    register_dataset,
)
from realhf.base import constants, logging
from realhf.system.push_pull_stream import NameResolvingZmqPuller

logger = logging.getLogger("StreamDataset")


class PullerStreamDataset(Dataset):
    def __init__(
        self,
        util: DatasetUtility,
        dataset_cfgs: List[DatasetAbstraction],
        pull_timeout_ms=100,
    ):
        # This dataset is just used for computing the dataset size,
        # and the number of steps per epoch.
        datasets = [
            make_dataset(
                dataset_cfg,
                seed=util.seed,
                dp_rank=util.dp_rank,
                world_size=util.world_size,
                tokenizer_or_tokenizer_name=util.tokenizer,
            )
            for dataset_cfg in dataset_cfgs
        ]
        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            dataset = ConcatDataset(datasets)
        self.dataset_size = len(dataset)
        del dataset, datasets

        self.pull_timeout_ms = pull_timeout_ms
        self.data_queue = queue.Queue(maxsize=self.dataset_size * util.world_size)
        self._stop_event = threading.Event()

        # Pass ZMQ context (thread-safe) and let worker create the socket
        self.util = util
        self.worker_thread = threading.Thread(target=self._pull_data_worker)
        self.worker_thread.start()

    def _pull_data_worker(self):
        """Worker thread that creates its own ZMQ puller and streams data."""
        # Initialize the puller inside the worker thread
        stream = NameResolvingZmqPuller(
            constants.experiment_name(),
            constants.trial_name(),
            puller_index=self.util.dp_rank,
        )
        try:
            while not self._stop_event.is_set():
                try:
                    data = stream.pull(timeout_ms=self.pull_timeout_ms)
                    processed_data = [
                        SequenceSample.from_json_compatible(x) for x in data
                    ]
                    logger.debug(
                        f"Get data {[x.ids[0] for x in processed_data]} from puller stream."
                    )
                    self.data_queue.put(processed_data)
                except queue.Empty:
                    logger.debug(f"No data from puller stream.")
                    time.sleep(0.1)
                    continue
        finally:
            # Ensure socket is closed in the same thread
            del stream
            # Exit if this thread has an error
            sys.exit(1)

    def __getitem__(self, idx: int) -> Optional[Any]:
        samples = []
        while True:
            try:
                samples += self.data_queue.get_nowait()
            except queue.Empty:
                break
        return samples

    def __len__(self) -> int:
        return self.dataset_size

    def __del__(self):
        self._stop_event.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)


register_dataset("puller_stream", PullerStreamDataset)
