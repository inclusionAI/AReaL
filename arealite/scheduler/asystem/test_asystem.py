#!/usr/bin/env python3
"""
AsystemScheduler 基本功能测试

类似于 arealite/scheduler/test/test_local.py 的测试方式
"""

import logging
from arealite.scheduler.asystem import AsystemScheduler
from arealite.scheduler.base import SchedulingConfig, ContainerSpec


class MyEngine:
    """简单的测试引擎类"""
    def __init__(self, config):
        self.config = config

    def initialize(self, init_config):
        logging.info(f"MyEngine initialized with {init_config}")

    def infer(self, x, y):
        logging.info(f"MyEngine.infer called with x={x}, y={y}")
        return x * y + self.config["value"]


def main():
    """基本功能测试"""
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s %(levelname)s %(message)s"
    )
    
    # 创建调度器
    config = {
        "type": "asystem",
        "endpoint": "http://33.215.20.149:8081",
        "expr_name": "test_expr", 
        "trial_name": "test_trial"
    }
    sched = AsystemScheduler(config)
    
    # 创建容器规格
    spec = ContainerSpec(
        cpu=1000,
        gpu=1,
        mem=2048,
        container_image="/storage/openpsi/images/areal-latest.sif",
        cmd="python -m test_worker",
        env_vars={"TEST_VAR": "test_value"},
        port=8080
    )
    
    # 创建调度配置
    scheduling_config = SchedulingConfig(
        replicas=2,
        specs=[spec]
    )
    
    try:
        # 创建工作节点
        sched.create_workers(scheduling_config)
        
        # 等待工作节点准备就绪
        workers = sched.get_workers(timeout=60)
        
        if not workers:
            logging.error("No workers available. Test failed.")
            return
        
        # 获取第一个工作节点
        worker = workers[0]
        worker_id = worker.id
        
        logging.info(f"Using worker with id={worker_id}, ip={worker.ip}, ports={worker.ports}")
        
        # 创建引擎
        engine_obj = MyEngine({"value": 24})
        assert sched.create_engine(worker_id, engine_obj, {"init": 1})
        
        # 调用引擎方法
        result = sched.call_engine(worker_id, "infer", 100, 10)
        print("Result:", result)
        
        # 验证结果格式
        assert isinstance(result, dict)
        assert result.get("status") == "success"
        
        print("AsystemScheduler basic test passed!")
        
    except Exception as e:
        logging.error(f"Test failed with error: {e}")
        raise
    finally:
        # 清理资源
        sched.delete_workers()


if __name__ == "__main__":
    main() 