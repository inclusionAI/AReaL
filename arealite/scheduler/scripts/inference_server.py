import argparse
import subprocess
import os
import sys

import uvicorn
import random
import time
from fastapi import FastAPI, HTTPException, Request


def get_serve_port(args):
    port = args.port
    port_str = os.environ.get('PORT_LIST', '').strip()

    # 检查是否设置
    if not port_str:
        return port
        # 按逗号分割并去除每个元素的空格
    ports = [p.strip() for p in port_str.split(',')]
    # 检查数组是否为空
    if not ports:
        return port
    # 获取第 0 个元素
    first_port = ports[0]
    return int(first_port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="InferenceServer")
    parser.add_argument("--port", type=int, default=None, required=False)
    args = parser.parse_args()
    port = get_serve_port(args)

    app = FastAPI()


    @app.post("/initialize")
    async def initialize(request: Request):
        """
        接收command参数并启动子进程
        """
        body = await request.json()
        if "command" not in body:
            raise HTTPException(status_code=400, detail="Missing command parameter")

        command = body["command"]

        try:
            # 启动子进程执行命令
            process = subprocess.Popen(
                command.split(),
                stdout=sys.stdout,
                stderr=sys.stderr
            )
            return {"pid": process.pid, "status": "started"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to start process: {str(e)}")


    # 随机延迟启动（保持原有逻辑）
    sleep_time = random.randint(1, 7)
    time.sleep(sleep_time)

    available_port = port
    print(f"[inference_server] Starting server on port: {available_port}...")

    uvicorn.run(app, host="0.0.0.0", port=available_port)
