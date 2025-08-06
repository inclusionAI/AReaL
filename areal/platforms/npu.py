from .platform import Platform
from realhf.base import logging

logger = logging.getLogger("NPU Platform")


class NpuPlatform(Platform):
    device_name: str = "ASCEND"
    device_type: str = "npu"
    dispatch_key: str = "PrivateUse1"
    ray_device_key: str = "NPU"
    device_control_env_var: str = "ASCEND_RT_VISIBLE_DEVICES"
    ray_experimental_noset: str = "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"
    communication_backend: str = "hccl"

    @classmethod
    def clear_cublas_workspaces(cls) -> None:
        pass

    @classmethod
    def get_vllm_worker_class(clas):
        try:
            from vllm import envs

            if envs.VLLM_USE_V1:
                from vllm_ascend.worker.worker_v1 import NPUWorker as Worker

                logger.info("Successfully imported vLLM V1 Worker.")
                return Worker
            else:
                from vllm_ascend.worker.worker import NPUWorker as Worker

                logger.info("Successfully imported vLLM V0 Worker.")
                return Worker
        except ImportError as e:
            logger.error("Failed to import vLLM Worker. " "Make sure vLLM is installed correctly: %s", e)
            raise RuntimeError("vLLM is not installed or not properly configured.") from e

    @classmethod
    def set_allocator_settings(cls) -> None:
        pass

    @classmethod
    def get_custom_env_vars(cls) -> dict:
        env_vars = {
            "TORCHINDUCTOR_COMPILE_THREADS": "2",
        }
        return env_vars