import abc
import logging


class Scheduler(abc.ABC):
    def __init__(self, config: dict):
        self.config = config
        self.rpc_client = self._build_rpc_client(config)
        self.worker_map = {}
        logging.info(f"Scheduler initialized with config: {config}")

    @abc.abstractmethod
    def _build_rpc_client(self, config):
        pass

    @abc.abstractmethod
    def create_workers(self, scheduler_config: dict) -> list:
        """
        启动worker，返回 [(id, ip, port), ...]
        """
        pass

    @abc.abstractmethod
    def create_engine(self, worker_id, engine_obj):
        """
        远程创建engine实例
        """
        pass

    @abc.abstractmethod
    def call(self, worker_id, method, arg):
        """
        数据面调用
        """
        pass
