import logging
from arealite.scheduler.local import LocalScheduler


class MyEngine:
    def __init__(self, config) -> None:
        self.config = config

    def initialize(self, config):
        logging.info(f"MyEngine initialized with {config}")

    def infer(self, x, y):
        logging.info(f"MyEngine.infer called with x={x}")
        return x * y + self.config["value"]
