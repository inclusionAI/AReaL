#!/usr/bin/env python3
"""
使用真实API请求测试AsystemScheduler的完整流程
模拟Asystem API服务器和worker节点，验证端到端的功能
"""

import json
import logging
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

try:
    import cloudpickle
except ImportError:
    import pickle as cloudpickle

from arealite.scheduler.asystem import AsystemScheduler
from arealite.scheduler.base import ContainerSpec, SchedulingConfig


class MockAsystemAPIServer(BaseHTTPRequestHandler):
    """模拟Asystem API服务器"""

    # 类变量存储作业信息
    jobs_db = {}  # job_uid -> job_info
    job_counter = 0

    def do_POST(self):
        """处理POST请求"""
        try:
            if self.path == "/api/v1/AsystemJobs":
                self._handle_submit_job()
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not Found")
        except Exception as e:
            logging.error(f"MockAsystemAPIServer error: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode())

    def do_GET(self):
        """处理GET请求"""
        try:
            if self.path.startswith("/api/v1/jobs/"):
                self._handle_get_job()
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not Found")
        except Exception as e:
            logging.error(f"MockAsystemAPIServer error: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode())

    def _handle_submit_job(self):
        """处理作业提交"""
        length = int(self.headers["Content-Length"])
        data = self.rfile.read(length)
        job_request = json.loads(data.decode())

        # 生成作业ID
        MockAsystemAPIServer.job_counter += 1
        job_uid = f"job_mock_{MockAsystemAPIServer.job_counter}_{uuid.uuid4().hex[:8]}"

        # 生成作业名称
        job_name = f"mock_job_{MockAsystemAPIServer.job_counter}_{int(time.time())}"

        # 创建作业信息
        job_info = {
            "uid": job_uid,
            "jobName": job_name,
            "status": "RUNNING",
            "asystemWorkerGroups": job_request.get("asystemWorkerGroups", []),
            "labels": job_request.get("labels", {}),
            "timeout": job_request.get("timeout", 7200),
            # 模拟资源分配结果
            "resourceDistributions": {
                "k8s": [
                    {"ip": "10.0.1.101", "instance": "k8s-worker-1"},
                    {"ip": "10.0.1.102", "instance": "k8s-worker-2"},
                ]
            },
            # 模拟WorkerGroupStatuses（包含端口信息）
            "workerGroupStatuses": self._generate_worker_statuses(job_request),
        }

        # 存储作业信息
        MockAsystemAPIServer.jobs_db[job_uid] = job_info

        # 返回作业创建响应
        response = {"uid": job_uid, "jobName": job_name, "status": "SUBMITTED"}

        logging.info(f"MockAsystemAPI: Created job {job_uid}")

        self.send_response(201)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def _generate_worker_statuses(
        self, job_request: Dict[str, Any]
    ) -> List[List[Dict[str, Any]]]:
        """根据作业请求生成WorkerGroupStatuses"""
        worker_group_statuses = []

        for group_idx, worker_group in enumerate(
            job_request.get("asystemWorkerGroups", [])
        ):
            replicas = worker_group.get("replicas", 1)
            container_specs = worker_group.get("containerSpecList", [])

            # 为每个worker组生成worker状态列表
            worker_statuses = []
            for replica_idx in range(replicas):
                # 生成worker状态
                worker_status = {
                    "ip": f"10.0.{group_idx + 1}.{replica_idx + 101}",
                    "instance": f"worker-{group_idx}-{replica_idx}",
                    "containerStatuses": [],
                }

                # 为每个容器生成端口
                for container_idx, container_spec in enumerate(container_specs):
                    port_count = container_spec.get("portCount", 1)

                    # 生成端口列表
                    base_port = (
                        8080 + group_idx * 100 + replica_idx * 10 + container_idx
                    )
                    ports = [base_port + i for i in range(port_count)]

                    container_status = {
                        "name": container_spec.get(
                            "name", f"container_{container_idx}"
                        ),
                        "ports": ports,
                    }
                    worker_status["containerStatuses"].append(container_status)

                # 处理worker组级别的portCount
                if (
                    not worker_status["containerStatuses"]
                    and "portCount" in worker_group
                ):
                    port_count = worker_group["portCount"]
                    base_port = 8080 + group_idx * 100 + replica_idx * 10
                    ports = [base_port + i for i in range(port_count)]

                    container_status = {"name": "default", "ports": ports}
                    worker_status["containerStatuses"].append(container_status)

                worker_statuses.append(worker_status)

            worker_group_statuses.append(worker_statuses)

        return worker_group_statuses

    def _handle_get_job(self):
        """处理获取作业状态"""
        # 从路径中提取job_uid
        job_uid = self.path.split("/")[-1]

        if job_uid in MockAsystemAPIServer.jobs_db:
            job_info = MockAsystemAPIServer.jobs_db[job_uid]

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(job_info).encode())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Job not found")

    def log_message(self, format, *args):
        """静默HTTP日志"""
        pass


class MockWorkerRPCServer(BaseHTTPRequestHandler):
    """模拟worker节点的RPC服务器"""

    # 类变量存储引擎信息
    engines = {}  # worker_id -> engine

    def do_POST(self):
        """处理RPC请求"""
        try:
            length = int(self.headers["Content-Length"])
            data = self.rfile.read(length)

            if self.path == "/create_engine":
                self._handle_create_engine(data)
            elif self.path == "/call":
                self._handle_call(data)
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not Found")

        except Exception as e:
            logging.error(f"MockWorkerRPCServer error: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode())

    def _handle_create_engine(self, data):
        """处理创建引擎请求"""
        try:
            engine_obj, init_config = cloudpickle.loads(data)

            # 获取worker标识
            worker_id = (
                f"{self.server.server_address[0]}:{self.server.server_address[1]}"
            )

            # 存储引擎
            MockWorkerRPCServer.engines[worker_id] = engine_obj

            # 初始化引擎
            if hasattr(engine_obj, "initialize"):
                engine_obj.initialize(init_config)

            logging.info(f"MockWorker: Engine created on {worker_id}")

            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")

        except Exception as e:
            logging.error(f"MockWorker: Error creating engine: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode())

    def _handle_call(self, data):
        """处理方法调用请求"""
        try:
            worker_id = (
                f"{self.server.server_address[0]}:{self.server.server_address[1]}"
            )

            if worker_id not in MockWorkerRPCServer.engines:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"Engine not initialized")
                return

            method, args, kwargs = cloudpickle.loads(data)
            engine = MockWorkerRPCServer.engines[worker_id]

            # 调用方法
            method_func = getattr(engine, method)
            result = method_func(*args, **kwargs)

            logging.info(
                f"MockWorker: Called {method} on {worker_id}, result: {result}"
            )

            self.send_response(200)
            self.end_headers()
            self.wfile.write(cloudpickle.dumps(result))

        except Exception as e:
            logging.error(f"MockWorker: Error calling method: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode())

    def log_message(self, format, *args):
        """静默HTTP日志"""
        pass


class MockEngine:
    """测试用的引擎类"""

    def __init__(self, config):
        self.config = config
        self.initialized = False
        self.call_count = 0

    def initialize(self, init_config):
        self.init_config = init_config
        self.initialized = True
        logging.info(f"MockEngine initialized with {init_config}")

    def infer(self, x, y):
        """推理方法"""
        self.call_count += 1
        result = x * y + self.config.get("value", 0)
        return result

    def get_status(self):
        """获取状态"""
        return {
            "initialized": self.initialized,
            "call_count": self.call_count,
            "config": self.config,
        }


class TestRealAsystemAPI:
    """测试真实的Asystem API请求处理"""

    def __init__(self):
        self.asystem_server = None
        self.worker_servers = []
        self.asystem_port = 18081
        self.worker_ports = [18082, 18083, 18084]  # 为多个worker预留端口

    def setup_mock_servers(self):
        """启动mock服务器"""
        # 启动Asystem API服务器
        self.asystem_server = HTTPServer(
            ("127.0.0.1", self.asystem_port), MockAsystemAPIServer
        )
        asystem_thread = threading.Thread(
            target=self.asystem_server.serve_forever, daemon=True
        )
        asystem_thread.start()

        # 启动worker RPC服务器
        for port in self.worker_ports:
            worker_server = HTTPServer(("127.0.0.1", port), MockWorkerRPCServer)
            worker_thread = threading.Thread(
                target=worker_server.serve_forever, daemon=True
            )
            worker_thread.start()
            self.worker_servers.append(worker_server)

        # 等待服务器启动
        time.sleep(0.2)

        logging.info(f"Mock Asystem API server started on port {self.asystem_port}")
        logging.info(f"Mock worker RPC servers started on ports {self.worker_ports}")

    def cleanup_mock_servers(self):
        """清理mock服务器"""
        if self.asystem_server:
            self.asystem_server.shutdown()
            self.asystem_server.server_close()

        for server in self.worker_servers:
            server.shutdown()
            server.server_close()

        logging.info("Mock servers stopped")

    def test_full_workflow(self):
        """测试完整的工作流程"""
        print("🧪 Testing full AsystemScheduler workflow with real API request...")
        print("=" * 70)

        # 1. 创建调度器
        config = {
            "type": "asystem",
            "endpoint": f"http://127.0.0.1:{self.asystem_port}",
            "expr_name": "test_real_api",
            "trial_name": "test_trial_001",
        }
        scheduler = AsystemScheduler(config)

        # 2. 创建调度配置（模拟真实的作业请求）
        cpu_spec = ContainerSpec(
            cpu=1000,
            gpu=0,
            mem=2048,
            container_image="mock_image",
            cmd="/bin/sh -c 'sleep 1000;'",
            env_vars={
                "WORKER_IMAGE": "123",
                "WORKER_COMMAND": "456",
                "WORKER_COUNT": "2",
                "WORKER_TYPE": "model_worker",
                "RUN_NAME": "1_test",
            },
            port=8080,
        )

        gpu_spec = ContainerSpec(
            cpu=1000,
            gpu=0,  # 实际没有GPU
            mem=2048,
            container_image="mock_image",
            cmd="/bin/sh -c 'sleep 1000;'",
            env_vars={
                "WORKER_IMAGE": "123",
                "WORKER_COMMAND": "456",
                "WORKER_COUNT": "2",
                "WORKER_TYPE": "model_worker",
                "RUN_NAME": "1_test",
            },
            port=8081,
        )

        scheduling_config = SchedulingConfig(replicas=2, specs=[cpu_spec, gpu_spec])

        try:
            # 3. 提交作业
            print("📤 Step 1: Submitting job to Asystem...")
            scheduler.create_workers(scheduling_config)

            # 4. 等待worker就绪
            print("⏳ Step 2: Waiting for workers to be ready...")
            workers = scheduler.get_workers(timeout=10)

            if not workers:
                print("❌ No workers available")
                return False

            print(f"✅ Got {len(workers)} workers ready:")
            for worker in workers:
                print(f"  - Worker {worker.id}: {worker.ip}:{worker.ports}")

            # 5. 在worker上创建引擎
            print("🏗️  Step 3: Creating engines on workers...")
            engine_results = []
            for i, worker in enumerate(workers):
                engine_obj = MockEngine({"value": 100 + i * 10})
                init_config = {"worker_id": worker.id, "index": i}

                # 手动设置worker IP到127.0.0.1以连接mock服务器
                worker.ip = "127.0.0.1"
                worker.ports = [self.worker_ports[i % len(self.worker_ports)]]

                # 手动注册worker信息
                worker_info = {
                    "ip": worker.ip,
                    "ports": worker.ports,
                    "instance": worker.id,
                }
                scheduler.rpc_client._register_worker_info(worker.id, worker_info)

                result = scheduler.create_engine(worker.id, engine_obj, init_config)
                engine_results.append(result)

                print(
                    f"  - Engine creation on {worker.id}: {'✅ SUCCESS' if result else '❌ FAILED'}"
                )

            # 6. 调用引擎方法
            print("🔧 Step 4: Calling engine methods...")
            call_results = []
            for i, worker in enumerate(workers):
                try:
                    # 测试推理方法
                    result = scheduler.call_engine(worker.id, "infer", 5, 10)
                    call_results.append(result)
                    print(f"  - infer(5, 10) on {worker.id}: {result}")

                    # 测试获取状态
                    status = scheduler.call_engine(worker.id, "get_status")
                    print(f"  - get_status() on {worker.id}: {status}")

                except Exception as e:
                    print(f"  - Error calling {worker.id}: {e}")
                    call_results.append(None)

            # 7. 验证结果
            print("\n📊 Step 5: Verification results...")

            success_count = sum(1 for r in engine_results if r)
            print(
                f"Engine creation success rate: {success_count}/{len(engine_results)}"
            )

            valid_calls = sum(1 for r in call_results if r is not None)
            print(f"Method call success rate: {valid_calls}/{len(call_results)}")

            if success_count > 0 and valid_calls > 0:
                print("🎉 Full workflow test PASSED!")
                return True
            else:
                print("❌ Full workflow test FAILED!")
                return False

        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback

            traceback.print_exc()
            return False
        finally:
            # 8. 清理
            print("🧹 Step 6: Cleanup...")
            scheduler.delete_workers()


def main():
    """主测试函数"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    # 清理之前的测试数据
    MockAsystemAPIServer.jobs_db.clear()
    MockAsystemAPIServer.job_counter = 0
    MockWorkerRPCServer.engines.clear()

    # 创建测试实例
    test = TestRealAsystemAPI()

    try:
        # 启动mock服务器
        test.setup_mock_servers()

        # 运行测试
        success = test.test_full_workflow()

        if success:
            print(
                "\n🎊 All tests passed! AsystemScheduler can handle real API requests."
            )
        else:
            print("\n💥 Some tests failed. Please check the implementation.")

    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # 清理
        test.cleanup_mock_servers()


if __name__ == "__main__":
    main()
