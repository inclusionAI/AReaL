# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
This script references code from Adaptive Parallel Reasoning (APR)/TinyRL's utils.py https://github.com/Parallel-Reasoning/APR/blob/main/tinyrl/utils.py and SGLang's test_utils.py https://github.com/sgl-project/sglang/blob/5d087891c93a6b66f0fd48b82fcf0a479d3e6ca5/python/sglang/test/test_utils.py#L545

The original script as well as the part from the original script used in this script are under Apache License 2.0 https://github.com/Parallel-Reasoning/APR/blob/main/LICENSE and https://github.com/sgl-project/sglang/blob/main/LICENSE
"""

import subprocess
import time
import requests
import os
from typing import Optional
from sglang.srt.utils import kill_process_tree

def popen_launch_server(
    model: str,
    base_url: str,
    timeout: float,
    model_name: str = "model",
    api_key: Optional[str] = None,
    other_args: list[str] = (),
    env: Optional[dict] = None,
    return_stdout_stderr: Optional[tuple] = None,
    skip_actual_launch: bool = False,
    use_os_system: bool = False,
    wait_before_check: int = 0,
):
    _, host, port = base_url.split(":")
    host = host[2:]

    command = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--host",
        host,
        "--port",
        port,
        "--served-model-name",
        model_name,
        *other_args,
    ]

    if api_key:
        command += ["--api-key", api_key]

    print(f"Launching server with command: {' '.join(command)}")

    if skip_actual_launch:
        process = None
    else:
        if use_os_system:
            command_str = " ".join(command) + " &"
            print(f"Executing command: {command_str}")
            os.system(command_str)
            # Servers launched with os.system do not return a process object and are not terminated automatically.
            process = None
        else:
            if return_stdout_stderr:
                process = subprocess.Popen(
                    command,
                    stdout=return_stdout_stderr[0],
                    stderr=return_stdout_stderr[1],
                    env=env,
                    text=True,
                )
            else:
                process = subprocess.Popen(
                    command, 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL, 
                    env=env
                )

    if wait_before_check > 0:
        print(f"Waiting for {wait_before_check} seconds before checking server status...")
        time.sleep(wait_before_check)

    start_time = time.time()
    with requests.Session() as session:
        while time.time() - start_time < timeout:
            try:
                headers = {
                    "Content-Type": "application/json; charset=utf-8",
                    "Authorization": f"Bearer {api_key}",
                }
                response = session.get(
                    f"{base_url}/health_generate",
                    headers=headers,
                )
                if response.status_code == 200:
                    return process
            except requests.RequestException:
                pass
            time.sleep(10)
    raise TimeoutError("Server failed to start within the timeout period.")

def terminate_process(process):
    kill_process_tree(process.pid)
