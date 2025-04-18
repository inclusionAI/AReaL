import queue
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
from realhf.base import constants
from realhf.system.push_pull_stream import NameResolvingZmqPuller


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
                experiment_name=constants.experiment_name(),
                trial_name=constants.trial_name(),
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
        self.data_queue = queue.Queue(maxsize=self.dataset_size)
        self._stop_event = threading.Event()

        # Pass ZMQ context (thread-safe) and let worker create the socket
        self.util = util
        self.worker_thread = threading.Thread(
            target=self._pull_data_worker,
            daemon=True,
        )
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
                    self.data_queue.put_nowait(processed_data)
                except queue.Empty:
                    time.sleep(0.1)
                    continue
        finally:
            # Ensure socket is closed in the same thread
            del stream

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
