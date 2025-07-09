import socket
import logging
import importlib
import importlib.util
import json
import sys
import os
import re
import base64
from pathlib import Path
import pickle


def find_free_port():
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    logging.info(f"Found free port: {port}")
    return port


def wait_for_port(ip, port, timeout=5.0):
    import time

    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((ip, port), timeout=0.5):
                logging.info(f"Port {port} on {ip} is now open.")
                return True
        except Exception:
            time.sleep(0.1)
    logging.error(f"Timeout waiting for port {port} on {ip}")
    return False


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def import_module(path: str, pattern: re.Pattern):
    """导入指定路径下匹配模式的所有模块"""
    dirname = Path(path)
    for x in os.listdir(dirname.absolute()):
        if not pattern.match(x):
            continue

        # 构建模块路径
        full_path = os.path.join(dirname, x)
        module_path = os.path.splitext(full_path)[0]

        # 从路径中提取相对模块名
        assert "realhf" in module_path
        start_idx = module_path.rindex("realhf")
        relative_module_path = module_path[start_idx:]
        module_name = relative_module_path.replace(os.sep, ".").replace("realhf.", "")

        # 确保模块名有效
        if not module_name:
            continue

        full_module_name = f"realhf.{module_name}"

        # 检查是否已导入
        if full_module_name in sys.modules:
            continue

        logger.info(f"Automatically importing module: {full_module_name}")
        try:
            importlib.import_module(full_module_name)
        except ImportError as e:
            logger.warning(f"Failed to import {full_module_name}: {e}")


def import_usercode(module_path: str, module_name: str):
    """从指定文件路径导入用户代码模块"""
    # 检查模块是否已存在
    if module_name in sys.modules:
        return sys.modules[module_name]

    # 创建模块规范
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(
            f"Could not find spec for module {module_name} at {module_path}"
        )

    # 创建并执行模块
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    logger.info(f"Successfully imported user module: {module_name} from {module_path}")
    return module


def serialize_with_metadata(obj, args=None):
    cls = obj.__class__
    module_name = cls.__module__
    class_name = cls.__name__

    return pickle.dumps(
        {
            "module_name": module_name,
            "class_name": class_name,
            "obj": pickle.dumps(obj),
            "arg": pickle.dumps(args),
        }
    )


def deserialize_with_metadata(data):
    payload = pickle.loads(data)
    module_name = payload["module_name"]
    class_name = payload["class_name"]

    try:
        logging.info(
            f"importing module: '{module_name}' to load class definition {class_name}."
        )
        importlib.import_module(module_name)
    except ImportError as e:
        raise e

    return pickle.loads(payload["obj"]), pickle.loads(payload["arg"])
