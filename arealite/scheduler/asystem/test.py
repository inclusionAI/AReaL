#!/usr/bin/env python3
"""
AsystemScheduler 使用示例

此示例展示了如何使用重构后的AsystemScheduler，与LocalScheduler API保持一致。
"""

import logging
from arealite.scheduler.asystem import AsystemScheduler
from arealite.scheduler.base import SchedulingConfig, ContainerSpec


class MockEngine:
    """模拟引擎类，用于测试"""
    def __init__(self, config):
        self.config = config
        logging.info(f"MockEngine initialized with config: {config}")

    def get_value(self):
        return self.config.get("value", 42)


def main():
    """完整的使用示例"""
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s %(levelname)s %(message)s"
    )
    
    # 1. 创建调度器 - 类似LocalScheduler的用法
    config = {
        "type": "asystem",
        "endpoint": "http://33.215.20.149:8081",  # 可选，有默认值
        "expr_name": "areal-example",
        "trial_name": "test-trial-01"
    }
    scheduler = AsystemScheduler(config)
    
    # 2. 构建容器配置 - 使用base模块的标准数据类
    ctl_spec = ContainerSpec(
        cpu=1000,
        gpu=0,
        mem=1024,
        container_image="/storage/openpsi/images/areal-latest.sif",
        cmd="bash /storage/openpsi/codes/v0.1.2/AReaL/realhf/scheduler/asystem/scripts/init.sh",
        env_vars={
            "LANG": "C",
            "LC_ALL": "C",
            "OMP_NUM_THREADS": "32",
            "NCCL_DEBUG": "INFO",
        },
        port=8080
    )
    
    model_worker_spec = ContainerSpec(
        cpu=4000,
        gpu=1, 
        mem=20000,
        container_image="/storage/openpsi/images/areal-latest.sif",
        cmd="python -m realhf.system.worker",
        env_vars={
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_DEBUG": "INFO",
        },
        port=8081
    )
    
    # 3. 构建调度配置 - 使用base模块的标准数据类
    scheduling_config = SchedulingConfig(
        replicas=3,  # 工作节点副本数
        specs=[ctl_spec, model_worker_spec]  # 容器规格列表
    )
    
    try:
        # 4. 创建工作节点 - 类似LocalScheduler.create_workers()
        print("正在创建工作节点...")
        scheduler.create_workers(scheduling_config)
        print("工作节点创建请求已提交")
        
        # 5. 等待工作节点启动 - 类似LocalScheduler.get_workers()
        print("等待工作节点启动...")
        workers = scheduler.get_workers(timeout=300)
        
        if not workers:
            print("没有工作节点可用，测试失败")
            return
            
        print(f"工作节点启动成功！共{len(workers)}个工作节点：")
        for worker in workers:
            print(f"  {worker.id}: {worker.ip}:{worker.ports} ")
            
        # 6. 创建引擎 - 类似LocalScheduler.create_engine()
        print("创建引擎...")
        first_worker = workers[0]
        engine = MockEngine({"value": 100})
        
        success = scheduler.create_engine(first_worker.id, engine, {"init_param": "test"})
        if success:
            print(f"引擎在工作节点 {first_worker.id} 上创建成功")
        else:
            print(f"引擎创建失败")
            return
        
        # 7. 调用引擎方法 - 类似LocalScheduler.call_engine()
        print("调用引擎方法...")
        
        # 调用推理方法
        result = scheduler.call_engine(first_worker.id, "agenerate", 
                                     req={"text": "What is machine learning?"})
        print(f"推理结果: {result}")
        
        # 调用训练方法
        result = scheduler.call_engine(first_worker.id, "train_batch",
                                     input_={"data": "training_batch"},
                                     loss_fn="mse_loss")
        print(f"训练结果: {result}")
        
        # 获取调度配置
        result = scheduler.call_engine(first_worker.id, "get_scheduling_config")
        print(f"调度配置: {result}")
        
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 8. 清理资源 - 类似LocalScheduler.delete_workers()
        print("清理资源...")
        scheduler.delete_workers()
        print("资源清理完成")


def test_compatibility_with_local_scheduler():
    """
    演示AsystemScheduler与LocalScheduler的API兼容性
    """
    print("\n=== API兼容性测试 ===")
    
    # AsystemScheduler和LocalScheduler具有相同的接口
    asystem_config = {
        "type": "asystem",
        "expr_name": "test_expr",
        "trial_name": "test_trial"
    }
    
    local_config = {
        "type": "local"
    }
    
    # 可以用同样的方式创建调度器
    asystem_scheduler = AsystemScheduler(asystem_config)
    # local_scheduler = LocalScheduler(local_config)  # 如果需要对比
    
    # 两者具有相同的方法签名：
    # - create_workers(scheduler_config, *args, **kwargs)
    # - get_workers(timeout=None) -> List[Worker]
    # - delete_workers(name)
    # - create_engine(worker_id, engine_obj, *args, **kwargs)
    # - call_engine(worker_id, method, *args, **kwargs)
    
    print("AsystemScheduler与LocalScheduler具有完全相同的API接口")


if __name__ == "__main__":
    main()
    test_compatibility_with_local_scheduler()