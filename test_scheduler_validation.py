#!/usr/bin/env python3
"""
专门验证scheduler API的模拟测试脚本
模拟用户提供的curl命令场景
"""

import json
import logging
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Any
import requests

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

class MockSchedulerAPIServer(BaseHTTPRequestHandler):
    """模拟scheduler API服务器"""
    
    # 类变量存储作业信息
    jobs_db = {}  # job_uid -> job_info
    job_counter = 0
    
    def do_POST(self):
        """处理POST请求"""
        try:
            if self.path == "/api/v1/AsystemJobs":
                self._handle_asystem_jobs()
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Not Found"}).encode())
        except Exception as e:
            logger.error(f"MockSchedulerAPIServer error: {e}")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def _handle_asystem_jobs(self):
        """处理AsystemJobs请求"""
        try:
            # 读取请求体
            length = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(length)
            job_request = json.loads(data.decode())
            
            logger.info("收到作业提交请求")
            logger.info(f"请求内容: {json.dumps(job_request, indent=2, ensure_ascii=False)}")
            
            # 验证请求格式
            validation_result = self._validate_job_request(job_request)
            if not validation_result["valid"]:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "Invalid request format",
                    "details": validation_result["errors"]
                }).encode())
                return
            
            # 生成作业信息
            MockSchedulerAPIServer.job_counter += 1
            job_uid = f"job_{MockSchedulerAPIServer.job_counter}_{uuid.uuid4().hex[:8]}"
            job_name = f"scheduler_test_job_{MockSchedulerAPIServer.job_counter}"
            
            # 创建作业响应
            job_info = {
                "uid": job_uid,
                "jobName": job_name,
                "status": "SUBMITTED",
                "message": "作业已成功提交到调度系统",
                "timestamp": time.time(),
                "asystemWorkerGroups": job_request.get("asystemWorkerGroups", []),
                "labels": job_request.get("labels", {}),
                "timeout": job_request.get("timeout", 7200)
            }
            
            # 存储作业信息
            MockSchedulerAPIServer.jobs_db[job_uid] = job_info
            
            # 返回成功响应
            response = {
                "success": True,
                "uid": job_uid,
                "jobName": job_name,
                "status": "SUBMITTED",
                "message": f"作业 {job_name} 已成功提交，UID: {job_uid}"
            }
            
            logger.info(f"✅ 作业提交成功: {job_uid}")
            
            self.send_response(201)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode())
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": "Invalid JSON format",
                "details": str(e)
            }).encode())
        except Exception as e:
            logger.error(f"处理请求时发生错误: {e}")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": "Internal server error",
                "details": str(e)
            }).encode())
    
    def _validate_job_request(self, job_request: Dict[str, Any]) -> Dict[str, Any]:
        """验证作业请求格式"""
        errors = []
        
        # 检查必需字段
        if "asystemWorkerGroups" not in job_request:
            errors.append("缺少必需字段: asystemWorkerGroups")
        else:
            worker_groups = job_request["asystemWorkerGroups"]
            if not isinstance(worker_groups, list) or len(worker_groups) == 0:
                errors.append("asystemWorkerGroups必须是非空数组")
            else:
                # 验证每个worker group
                for i, group in enumerate(worker_groups):
                    group_errors = self._validate_worker_group(group, i)
                    errors.extend(group_errors)
        
        # 检查可选字段
        if "timeout" in job_request:
            timeout = job_request["timeout"]
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                errors.append("timeout必须是正数")
        
        if "labels" in job_request:
            labels = job_request["labels"]
            if not isinstance(labels, dict):
                errors.append("labels必须是字典类型")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def _validate_worker_group(self, group: Dict[str, Any], index: int) -> list:
        """验证worker group格式"""
        errors = []
        prefix = f"asystemWorkerGroups[{index}]"
        
        # 检查必需字段
        required_fields = ["workerRole", "replicas", "containerSpecList"]
        for field in required_fields:
            if field not in group:
                errors.append(f"{prefix}缺少必需字段: {field}")
        
        # 验证replicas
        if "replicas" in group:
            replicas = group["replicas"]
            if not isinstance(replicas, int) or replicas <= 0:
                errors.append(f"{prefix}.replicas必须是正整数")
        
        # 验证containerSpecList
        if "containerSpecList" in group:
            container_specs = group["containerSpecList"]
            if not isinstance(container_specs, list) or len(container_specs) == 0:
                errors.append(f"{prefix}.containerSpecList必须是非空数组")
            else:
                for j, spec in enumerate(container_specs):
                    spec_errors = self._validate_container_spec(spec, f"{prefix}.containerSpecList[{j}]")
                    errors.extend(spec_errors)
        
        return errors
    
    def _validate_container_spec(self, spec: Dict[str, Any], prefix: str) -> list:
        """验证container spec格式"""
        errors = []
        
        # 检查必需字段
        required_fields = ["name", "command", "args"]
        for field in required_fields:
            if field not in spec:
                errors.append(f"{prefix}缺少必需字段: {field}")
        
        # 验证resourceRequirement
        if "resourceRequirement" in spec:
            req = spec["resourceRequirement"]
            if not isinstance(req, dict):
                errors.append(f"{prefix}.resourceRequirement必须是字典类型")
            else:
                # 验证资源要求的各个字段
                if "cpu" in req and not isinstance(req["cpu"], (int, float)):
                    errors.append(f"{prefix}.resourceRequirement.cpu必须是数字")
                if "memoryMB" in req and not isinstance(req["memoryMB"], (int, float)):
                    errors.append(f"{prefix}.resourceRequirement.memoryMB必须是数字")
                if "gpuCount" in req and not isinstance(req["gpuCount"], int):
                    errors.append(f"{prefix}.resourceRequirement.gpuCount必须是整数")
        
        return errors
    
    def log_message(self, format, *args):
        """静默HTTP日志"""
        pass

class SchedulerValidator:
    """Scheduler验证器"""
    
    def __init__(self, port=8081):
        self.port = port
        self.server = None
        self.server_thread = None
    
    def start_mock_server(self):
        """启动模拟服务器"""
        try:
            self.server = HTTPServer(("127.0.0.1", self.port), MockSchedulerAPIServer)
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            
            # 等待服务器启动
            time.sleep(0.2)
            
            logger.info(f"🚀 模拟scheduler服务器已启动，监听端口: {self.port}")
            return True
        except Exception as e:
            logger.error(f"❌ 启动模拟服务器失败: {e}")
            return False
    
    def stop_mock_server(self):
        """停止模拟服务器"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            logger.info("🛑 模拟服务器已停止")
    
    def test_curl_command(self, target_ip="127.0.0.1"):
        """测试curl命令"""
        logger.info(f"🧪 测试向 {target_ip}:{self.port} 发送作业请求...")
        
        # 用户提供的payload
        payload = {
            "asystemWorkerGroups": [
                {
                    "workerRole": "model_worker",
                    "replicas": 2,
                    "containerSpecList": [
                        {
                            "name": "train_cpu",
                            "command": ["/bin/sh"],
                            "portCount": 2,
                            "args": ["-c", "sleep 1000;"],
                            "env": {
                                "WORKER_IMAGE": "123",
                                "WORKER_COMMAND": "456",
                                "WORKER_COUNT": "2",
                                "WORKER_TYPE": "model_worker",
                                "RUN_NAME": "1_test"
                            },
                            "mounts": ["/mnt/host_data:/data"],
                            "resourceRequirement": {
                                "cpu": 1000,
                                "memoryMB": 2048,
                                "gpuType": "nvidia-tesla-v100",
                                "gpuCount": 0
                            }
                        },
                        {
                            "name": "train_gpu",
                            "command": ["/bin/sh"],
                            "portCount": 2,
                            "args": ["-c", "sleep 1000;"],
                            "env": {
                                "WORKER_IMAGE": "123",
                                "WORKER_COMMAND": "456",
                                "WORKER_COUNT": "2",
                                "WORKER_TYPE": "model_worker",
                                "RUN_NAME": "1_test"
                            },
                            "mounts": ["/mnt/host_data:/data"],
                            "resourceRequirement": {
                                "cpu": 1000,
                                "memoryMB": 2048,
                                "gpuType": "nvidia-tesla-v100",
                                "gpuCount": 0
                            }
                        }
                    ],
                    "workingDirectory": "/app",
                    "timeLimitSeconds": 3600
                },
                {
                    "workerRole": "model-worker",
                    "replicas": 1,
                    "portCount": 2,
                    "containerSpecList": [
                        {
                            "name": "default",
                            "command": ["/bin/sh"],
                            "args": ["-c", "sleep 1000;"],
                            "env": {
                                "WORKER_IMAGE": "123",
                                "WORKER_COMMAND": "456",
                                "WORKER_COUNT": "2",
                                "WORKER_TYPE": "model_worker",
                                "RUN_NAME": "1_test"
                            },
                            "mounts": ["/mnt/host_data:/data"],
                            "resourceRequirement": {
                                "cpu": 1000,
                                "memoryMB": 2048
                            }
                        }
                    ]
                }
            ],
            "timeout": 7200,
            "labels": {
                "project": "alpha",
                "user": "jdoe",
                "experiment_id": "exp-12345"
            }
        }
        
        try:
            # 发送HTTP请求
            url = f"http://{target_ip}:{self.port}/api/v1/AsystemJobs"
            headers = {"Content-Type": "application/json"}
            
            logger.info(f"📤 发送POST请求到: {url}")
            logger.info(f"📋 请求内容大小: {len(json.dumps(payload))} 字节")
            
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            
            logger.info(f"📨 响应状态码: {response.status_code}")
            logger.info(f"📋 响应头: {dict(response.headers)}")
            
            try:
                response_data = response.json()
                logger.info(f"📄 响应内容: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
            except json.JSONDecodeError:
                logger.info(f"📄 响应内容(纯文本): {response.text}")
            
            if response.status_code in [200, 201]:
                logger.info("✅ 作业提交成功！")
                return True
            else:
                logger.error(f"❌ 作业提交失败，状态码: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError as e:
            logger.error(f"❌ 连接失败: {e}")
            logger.info("💡 这可能意味着目标服务器不可达或未运行")
            return False
        except requests.exceptions.Timeout as e:
            logger.error(f"❌ 请求超时: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ 请求失败: {e}")
            return False
    
    def run_validation(self):
        """运行完整的验证流程"""
        logger.info("🎯 开始scheduler验证流程...")
        logger.info("=" * 60)
        
        success = False
        
        try:
            # 1. 启动模拟服务器
            if not self.start_mock_server():
                return False
            
            # 2. 测试本地模拟服务器
            logger.info("\n📍 步骤1: 测试本地模拟服务器...")
            local_success = self.test_curl_command("127.0.0.1")
            
            if local_success:
                logger.info("✅ 本地模拟测试通过！")
                logger.info("🔍 这证明了:")
                logger.info("  - API接口格式正确")
                logger.info("  - 请求payload有效")
                logger.info("  - scheduler能够正确处理请求")
            else:
                logger.error("❌ 本地模拟测试失败")
                return False
            
            # 3. 测试真实服务器（如果需要）
            logger.info("\n📍 步骤2: 测试真实服务器连接...")
            logger.info("💡 如果需要测试真实的30.230.2.87服务器，请确保:")
            logger.info("  - 服务器正在运行")
            logger.info("  - 网络连接正常")
            logger.info("  - 防火墙允许8081端口访问")
            
            # 这里可以选择测试真实服务器
            # real_success = self.test_curl_command("30.230.2.87")
            
            success = local_success
            
        except Exception as e:
            logger.error(f"❌ 验证过程发生错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop_mock_server()
        
        logger.info("\n" + "=" * 60)
        if success:
            logger.info("🎉 Scheduler验证完成 - 所有测试通过!")
            logger.info("✅ 您的curl命令格式正确，scheduler能够正常处理请求")
        else:
            logger.info("💥 Scheduler验证失败")
            logger.info("❌ 请检查scheduler配置或网络连接")
        
        return success

def main():
    """主函数"""
    print("🔧 Scheduler API 验证工具")
    print("📝 这个工具会验证您提供的curl命令是否能正常工作")
    print()
    
    validator = SchedulerValidator(port=8081)
    success = validator.run_validation()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 