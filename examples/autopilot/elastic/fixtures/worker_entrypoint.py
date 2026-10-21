# SPDX-License-Identifier: Apache-2.0

"""Set a worker's single GPU before importing AReaL and initializing CUDA."""

import argparse
import os
import runpy


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker-index", required=True, type=int)
    args, _ = parser.parse_known_args()
    devices = [int(value) for value in os.environ["AREAL_ELASTIC_GPU_POOL"].split(":")]
    if not 0 <= args.worker_index < len(devices):
        raise ValueError("Worker index is outside the training GPU pool")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[args.worker_index])
    runpy.run_module("areal.infra.rpc.rpc_server", run_name="__main__")


if __name__ == "__main__":
    main()
