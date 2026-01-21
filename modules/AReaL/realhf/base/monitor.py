# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import asyncio
import contextlib
import dataclasses
import enum
import json
import os
import pickle
import re
import time
from collections import defaultdict
from statistics import mean
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

import numpy as np
import psutil
import pynvml
import torch
import tqdm
import tqdm.asyncio

import realhf.base.constants as constants
import realhf.base.logging as logging

if TYPE_CHECKING:
    from realhf.api.core.config import ModelName

logger = logging.getLogger("benchmark")

IF_MARK = False


@dataclasses.dataclass
class RolloutStat:
    submitted: int = 0
    accepted: int = 0
    running: int = 0


def mock_time_mark(name, identifier, t, step):
    if IF_MARK:
        logger.debug(f"*{name}* #{identifier}#  ${t}$ ns step &{step}&")


def time_mark(name, identifier, step=0):
    if IF_MARK:
        logger.debug(
            f"*{name}* #{identifier}#  ${int(time.time_ns())}$ ns step &{step}&"
        )


def parse_time_mark_in_line(line, name, step_range=None):
    if f"*{name}*" in line:
        identifer, t, step = (
            line.split("#")[1],
            int(line.split("$")[1]),
            int(line.split("&")[1]),
        )
        if step_range:
            if step >= step_range[1] or step < step_range[0]:
                return None
        return identifer, t

    else:
        return None


def parse_time_mark_in_file(file, name, step_range=None):
    time_points = defaultdict(list)
    with open(file, "r") as f:
        count = 0
        res_count = 0
        for line in f.readlines():
            count += 1
            res = parse_time_mark_in_line(line, name, step_range=step_range)
            if res is not None:
                res_count += 1
                identifier, time_point = res
                time_points[identifier].append(time_point)
    return time_points


def parse_time_mark_in_dir(dir, name, step_range=None):
    time_points = {}
    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        tpsf = parse_time_mark_in_file(file_path, name, step_range=step_range)
        for k, v in tpsf.items():
            if k not in time_points:
                time_points[k] = v
            else:
                time_points[k].extend(v)
    return time_points


MATPLOTLIB_COLORS = [
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "black",
    "brown",
    "gray",
    "cyan",
    "magenta",
    "lime",
    "olive",
    "navy",
]


def summary_time_points(
    start_keys,
    end_keys,
    identifiers,
    dir_name=None,
    file_name=None,
    start_time=None,
    figsize=(12, 4),
    end_time=None,
    step_range=None,
    save_fig_path="time_points.png",
    draw_boundary=False,
):
    """Plot and summary time marks in logs."""
    import matplotlib.pyplot as plt

    assert file_name or dir_name, "dir or file name must be specified"
    all_time_points = {}
    if file_name is None:
        for k in start_keys:
            all_time_points[k] = parse_time_mark_in_dir(
                dir_name, k, step_range=step_range
            )
        for k in end_keys:
            all_time_points[k] = parse_time_mark_in_dir(
                dir_name, k, step_range=step_range
            )
    else:
        for k in start_keys:
            all_time_points[k] = parse_time_mark_in_file(
                file_name, k, step_range=step_range
            )
        for k in end_keys:
            all_time_points[k] = parse_time_mark_in_file(
                file_name, k, step_range=step_range
            )

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_ylim(-1, len(identifiers))
    ax.set_yticks(list(range(len(identifiers))))
    ax.set_yticklabels(identifiers)

    label_set = {sk: False for sk in start_keys}
    infos = {}
    min_time = None
    max_time = None
    for id_index, identifier in enumerate(identifiers):
        time_sum = {}
        time_list = {}
        for start_key_idx, (start_key, end_key) in enumerate(zip(start_keys, end_keys)):
            time_sum[start_key] = 0
            time_list[start_key] = []
            try:
                start_time_points = np.array(all_time_points[start_key][identifier])
                end_time_points = np.array(all_time_points[end_key][identifier])
            except KeyError:
                continue
            assert len(start_time_points) == len(end_time_points)

            if start_time is not None:
                valid_indices_st = np.where(start_time_points > start_time)
                valid_indices_et = np.where(start_time_points < end_time)
                valid_indices = np.intersect1d(valid_indices_st, valid_indices_et)
                start_time_points = start_time_points[valid_indices]
                end_time_points = end_time_points[valid_indices]

            # plot time point pairs
            for stp, etp in zip(list(start_time_points), list(end_time_points)):
                min_time = stp if min_time is None else min(min_time, stp)
                max_time = etp if max_time is None else max(max_time, etp)
                time_sum[start_key] += etp - stp
                time_list[start_key].append(etp - stp)

                if label_set[start_key] is False:
                    label = start_key
                    label_set[start_key] = True
                else:
                    label = None

                ax.barh(
                    y=id_index,
                    width=etp - stp,
                    left=stp,
                    height=0.8,
                    color=MATPLOTLIB_COLORS[start_key_idx],
                    label=label,
                )

                if draw_boundary:
                    ax.plot(
                        [stp, stp],
                        [id_index - 0.4, id_index + 0.4],
                        color="black",
                        linestyle="-",
                        linewidth=0.5,
                    )
                    ax.plot(
                        [etp, etp],
                        [id_index - 0.4, id_index + 0.4],
                        color="black",
                        linestyle="-",
                        linewidth=0.5,
                    )

        infos[identifier] = (time_sum, time_list)

    ax.set_xlim(min_time, max_time)
    total_width = max_time - min_time
    xticks = np.arange(
        min_time - total_width // 12, max_time - total_width // 12, 10 * 1e9
    )
    xtick_labels = [f"{int((i//1e9)%1000)}" for i in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)

    # summary time cost percent
    for id_index, identifier in enumerate(identifiers):
        print("=" * 30)
        print(f"Identifier {identifier} time cost percent:")
        bubble_time = 100
        time_sum, time_list = infos[identifier]
        for k in time_sum:
            time_perc = round(time_sum[k] / (max_time - min_time) * 100, 2)
            # print time cost percent
            avg_val = (
                round(mean(time_list[k]) / 10e6, 2) if len(time_list[k]) > 0 else "-"
            )
            max_val = (
                round(max(time_list[k]) / 10e6, 2) if len(time_list[k]) > 0 else "-"
            )
            min_val = (
                round(min(time_list[k]) / 10e6, 2) if len(time_list[k]) > 0 else "-"
            )

            bubble_time -= time_perc
            print(
                f"{k} -- {time_perc} %, "
                f"avg, min, max = {avg_val}, {min_val}, {max_val} ms, "
                f"sum, n = {round(time_sum[k]/10e6, 2)} ms, {len(time_list[k])}"
            )
        print(f"bubble time -- {round(bubble_time, 2)}%")

    plt.legend(loc=(1.01, 0.0))
    plt.tight_layout()

    plt.savefig(save_fig_path)


def gpu_utilization_monitor(worker_idx: int, interval: float, ttl: float):
    pynvml.nvmlInit()
    gpu_idx = worker_idx % 8
    tik = time.time()
    while time.time() - tik < ttl:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_memory = memory_info.total / (1024**2)  # Convert bytes to megabytes
        used_memory = memory_info.used / (1024**2)
        memory_usage_percentage = (used_memory / total_memory) * 100
        logger.debug(
            f"Worker Index {worker_idx}, GPU {gpu_idx}: "
            f"Compute Utilization - {utilization.gpu}%, "
            f"Total Memory - {total_memory:.2f}MB, Used Memory - {used_memory:.2f}MB, "
            f"Memory Usage - {memory_usage_percentage:.2f}%"
        )
        time.sleep(interval)
    pynvml.nvmlShutdown()


# Helper function to calculate FLOPs using the Megatron-LM paper's formula
def calculate_llama_train_flops(
    checkpoint_activations_factor: int,
    batch_size: int,
    seqlens: List[int],
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    vocab_size: int,
):
    return checkpoint_activations_factor * caculuate_llama_forward_flops(
        batch_size,
        seqlens,
        num_layers,
        hidden_size,
        intermediate_size,
        vocab_size,
    )


def caculuate_llama_forward_flops(
    batch_size: int,
    seqlens: List[int],
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    vocab_size: int,
):
    assert len(seqlens) == batch_size
    attn_flops = sum(x**2 for x in seqlens) * hidden_size
    return (
        2
        * num_layers
        * (
            4 * sum(seqlens) * hidden_size**2
            + 2 * attn_flops
            + 3 * sum(seqlens) * hidden_size * intermediate_size
        )
        + 4 * sum(seqlens) * vocab_size * hidden_size
    )


def calculate_llama_gen_flops(
    batch_size,
    prompt_lens,
    gen_len,
    num_layers,
    hidden_size,
    intermediate_size,
    vocab_size,
):
    flops = caculuate_llama_forward_flops(
        batch_size,
        prompt_lens,
        num_layers,
        hidden_size,
        intermediate_size,
        vocab_size,
    )
    for i in range(gen_len):
        prefix_lens = [x + i for x in prompt_lens]
        flops += (
            2
            * num_layers
            * (
                4 * batch_size * hidden_size**2
                + 2 * (sum(prefix_lens) + batch_size) * hidden_size
                + 3 * batch_size * hidden_size * intermediate_size
            )
            + 4 * batch_size * vocab_size * hidden_size
        )
    return flops


#################### CUDA Kernel Time Statistics Start ####################
# Categorizing CUDA kernels into computation, communication, memory IO, and MISC/IDLE,
# used to plot the percentage of time spent on each category and show how much we can
# improve over vanilla parallel strategies.

COMPUTE_KERNEL_KEYS = [
    "elementwise_kernel",
    "gemm",
    "aten::",
    "at::native::",
    "flash",
    "backward_kernel",
    "reduce_kernel",
    "multi_tensor_apply",
    "gae_kernel",
    "gemvx::kernel",
    "cublas",
    "cudnn",
    "cutlass",
]

P2P_COMM_KERNEL_KEYS = [
    "ncclDevKernel_SendRecv",
]

COLL_COMM_KERNEL_KEYS = [
    "ncclDevKernel_AllReduce",
    "ncclDevKernel_ReduceScatter",
    "ncclDevKernel_AllGather",
]

MEM_KERNEL_KEYS = [
    "Memcpy",
    "cleanup",
    "Memset",
]

MISC_KERNEL_KEYS = [
    "at_cuda_detail",
    "CudaCodeGen",
]


class CUDAKernelTimeCategory(enum.Enum):
    COMPUTE = "compute"
    P2P_COMM = "p2p_comm"
    COLL_COMM = "coll_comm"
    MEM = "memoryIO"
    IDLE = "idle"
    MISC = "misc"

    @classmethod
    def from_name(cls, name):
        # Order may matter. MEM & COMM keys are easier to find out.
        if any(k in name for k in MEM_KERNEL_KEYS):
            return cls.MEM
        if any(k in name for k in P2P_COMM_KERNEL_KEYS):
            return cls.P2P_COMM
        if any(k in name for k in COLL_COMM_KERNEL_KEYS):
            return cls.COLL_COMM
        if any(k in name for k in MISC_KERNEL_KEYS):
            return cls.MISC
        if any(k in name for k in COMPUTE_KERNEL_KEYS):
            return cls.COMPUTE
        raise NotImplementedError(f"Unknown kernel type. Name is `{name}`")


class CUDAKernelTimeStat:  # in us

    def __init__(self, world_size, **kwargs):
        self.world_size = world_size
        for k in CUDAKernelTimeCategory:
            setattr(self, k.value, kwargs.get(k.value, 0))

    @property
    def total(self):
        return sum([getattr(self, k.value) for k in CUDAKernelTimeCategory])

    def percentage(self) -> Dict:
        return {
            k.value: getattr(self, k.value) / self.total for k in CUDAKernelTimeCategory
        }

    def __add__(self, other):
        return CUDAKernelTimeStat(
            world_size=self.world_size + other.world_size,
            **{
                k.value: getattr(self, k.value) + getattr(other, k.value)
                for k in CUDAKernelTimeCategory
            },
        )

    def __truediv__(self, x):
        assert self.world_size % x == 0
        return CUDAKernelTimeStat(
            world_size=self.world_size // x,
            **{k.value: getattr(self, k.value) / x for k in CUDAKernelTimeCategory},
        )

    def gpu_average(self):
        return self / self.world_size

    def __repr__(self):
        import tabulate

        headers = [
            "",
            "Total",
            "Computation",
            "P2P Comm",
            "Collective Comm",
            "Memory IO",
            "Idle",
            "Misc",
        ]
        line1 = [
            "Time (s)",
            self.total / 1e6,
            *[getattr(self, k.value) / 1e6 for k in CUDAKernelTimeCategory],
        ]
        line1 = [f"{x:.2f}" if isinstance(x, float) else x for x in line1]
        line2 = [
            "Percentage (%)",
            "-",
            *[f"{self.percentage()[k.value]:.2%}" for k in CUDAKernelTimeCategory],
        ]
        tab_str = tabulate.tabulate(
            [headers, line1, line2],
            headers="firstrow",
            tablefmt="fancy_grid",
            stralign="center",
        )
        return (
            f" Number of GPUs: {self.world_size} ".center(
                len(tab_str.split("\n")[0]), "="
            )
            + "\n"
            + tab_str
        )


@dataclasses.dataclass
class KernelEventEntry:
    ts: int
    tid: int
    dur: int
    category: CUDAKernelTimeCategory


@dataclasses.dataclass
class KernelEventBoundary:
    ts: int
    is_start: bool
    category: CUDAKernelTimeCategory


def kernelStatFromEvents(
    entries: List[KernelEventEntry],
    global_start_ts,
    global_end_ts,
):
    events: List[KernelEventBoundary] = []
    for entry in entries:
        events.append(KernelEventBoundary(entry.ts, True, entry.category))
        events.append(KernelEventBoundary(entry.ts + entry.dur, False, entry.category))
    # A trick to count for idle time waiting other processes
    events.append(
        KernelEventBoundary(global_start_ts, True, CUDAKernelTimeCategory.IDLE)
    )
    events.append(
        KernelEventBoundary(global_end_ts, False, CUDAKernelTimeCategory.IDLE)
    )

    events.sort(key=lambda x: x.ts)

    times = {k: 0 for k in CUDAKernelTimeCategory}
    active = {k: 0 for k in CUDAKernelTimeCategory}

    current_time = events[0].ts

    for i in range(len(events)):
        next_time = events[i].ts

        # Priority: compute > communication > memory > misc > idle
        if i > 0 and next_time != current_time:
            duration = next_time - current_time
            if active[CUDAKernelTimeCategory.COMPUTE] > 0:
                times[CUDAKernelTimeCategory.COMPUTE] += duration
            elif active[CUDAKernelTimeCategory.COLL_COMM] > 0:
                times[CUDAKernelTimeCategory.COLL_COMM] += duration
            elif active[CUDAKernelTimeCategory.P2P_COMM] > 0:
                times[CUDAKernelTimeCategory.P2P_COMM] += duration
            elif active[CUDAKernelTimeCategory.MEM] > 0:
                times[CUDAKernelTimeCategory.MEM] += duration
            elif active[CUDAKernelTimeCategory.MISC] > 0:
                times[CUDAKernelTimeCategory.MISC] += duration
            else:
                times[CUDAKernelTimeCategory.IDLE] += duration
        active[events[i].category] += 1 if events[i].is_start else -1
        current_time = next_time

    assert all(v == 0 for v in active.values()), active
    return CUDAKernelTimeStat(world_size=1, **{k.value: v for k, v in times.items()})


async def _load_events_async(file_path, semaphore) -> List[Dict]:
    import aiofiles

    async with semaphore:
        pid = int(file_path.rstrip(".json").split("_r")[-1])
        async with aiofiles.open(file_path, "r") as f:
            content = await f.read()
        events = json.loads(content)["traceEvents"]
        events = list(
            filter(
                lambda x: "cat" in x and x["cat"] in ["gpu_user_annotation", "kernel"],
                events,
            )
        )
        # Replace with the actual process id, starting from 0 to #gpus-1
        for ev in events:
            ev["pid"] = pid
    return events, pid


async def _load_all_events(root_dir, mfc_name) -> List[Dict]:
    trace_file_paths = []
    for fn in os.listdir(root_dir):
        if not fn.startswith(mfc_name):
            continue
        trace_file_paths.append(os.path.join(root_dir, fn))

    # The JSON file can be large, up to 2GB. Load them concurrently.
    semaphore = asyncio.Semaphore(8)
    tasks = [_load_events_async(file_path, semaphore) for file_path in trace_file_paths]

    all_events = {}
    for coro in tqdm.asyncio.tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Loading JSON files"
    ):
        try:
            events, pid = await coro
            all_events[pid] = events
        except Exception as e:
            print(f"Error loading JSON file: {e}")

    return all_events


def kernelStatFromTrace(root_dir: str, mfc_name: str):
    cache_file = os.path.join(root_dir, f"_cached_{mfc_name}.json")
    if os.path.exists(cache_file):
        logger.info(f'Loading trace JSON files of MFC "{mfc_name}" from cache...')
        with open(cache_file, "r") as f:
            all_events = json.load(f)
        all_events = {int(pid): v for pid, v in all_events.items()}
    else:
        if not any(fn.startswith(mfc_name) for fn in os.listdir(root_dir)):
            raise RuntimeError(
                f"No trace file found for the given MFC name: {mfc_name}."
            )

        load_json_tik = time.perf_counter()
        logger.info(
            f'Loading trace JSON files of MFC "{mfc_name}" concurrently from {root_dir}...'
        )
        all_events: Dict[int, List[Dict]] = asyncio.run(
            _load_all_events(root_dir, mfc_name)
        )
        logger.info(
            f"{len(all_events)} JSON file loaded. "
            f"Time consumption: {time.perf_counter() - load_json_tik:.2f} secs. "
            f"Processing..."
        )

        with open(cache_file, "w") as f:
            json.dump(all_events, f)

    # To remove the wait time from nccl send/recv, collect send/recv kernels annotations.
    # These annotations look like "nccl:send 0->1". For each annotation, find the execution time
    # of its pair. The actual execution time should the minimum of the two.
    send_recv_annotations = {
        pid: [
            ev
            for ev in events
            if ev["cat"] == "gpu_user_annotation"
            and ev["name"].startswith("nccl:recv")
            or ev["name"].startswith("nccl:send")
        ]
        for pid, events in all_events.items()
    }
    for events in send_recv_annotations.values():
        events.sort(key=lambda x: x["ts"])

    send_recv_time = {pid: [] for pid, events in send_recv_annotations.items()}

    def _matches_next_sr(type_, src, dst):
        if type_ == "send":
            annot = send_recv_annotations[dst][0]
            m = re.match(r"nccl:recv (\d+)<-(\d+)", annot["name"])
            if not m:
                return False
            peer_dst, peer_src = map(int, m.groups())
            if peer_src != src or peer_dst != dst:
                return False
            return True
        else:
            assert type_ == "recv"
            annot = send_recv_annotations[src][0]
            m = re.match(r"nccl:send (\d+)->(\d+)", annot["name"])
            if not m:
                return False
            peer_src, peer_dst = map(int, m.groups())
            if peer_src != src or peer_dst != dst:
                return False
            return True

    def resolve_next_sr_time(pid):
        # Resolve send/recv time recursively, just like a Tetris game
        annot = send_recv_annotations[pid][0]
        if annot["name"].startswith("nccl:send"):
            src, dst = map(
                int, re.match(r"nccl:send (\d+)->(\d+)", annot["name"]).groups()
            )
            assert src == pid, (src, pid)
            while not _matches_next_sr("send", src, dst):
                resolve_next_sr_time(dst)
        else:
            assert annot["name"].startswith("nccl:recv")
            dst, src = map(
                int, re.match(r"nccl:recv (\d+)<-(\d+)", annot["name"]).groups()
            )
            assert dst == pid, (dst, pid)
            while not _matches_next_sr("recv", src, dst):
                resolve_next_sr_time(src)
        ev1, ev2 = send_recv_annotations[src].pop(0), send_recv_annotations[dst].pop(0)
        dur = min(ev1["dur"], ev2["dur"])
        send_recv_time[src].append(dur)
        send_recv_time[dst].append(dur)

    for pid in tqdm.tqdm(all_events, desc="Resolving send/recv times"):
        while len(send_recv_annotations[pid]) > 0:
            resolve_next_sr_time(pid)

    kernel_events: Dict[int, List[KernelEventEntry]] = defaultdict(list)
    global_start = min(ev["ts"] for events in all_events.values() for ev in events)
    global_end = max(ev["ts"] for events in all_events.values() for ev in events)
    for pid in tqdm.tqdm(all_events, desc="Processing events"):
        for ev in all_events[pid]:
            if ev["cat"] != "kernel":
                continue
            assert ev["dur"] > 0, ev
            cat = CUDAKernelTimeCategory.from_name(ev["name"])
            if cat == CUDAKernelTimeCategory.P2P_COMM:
                assert len(send_recv_time[pid]) > 0
                ev["dur"] = send_recv_time[pid].pop(0)
            kernel_events[pid].append(
                KernelEventEntry(
                    ts=ev["ts"], tid=ev["tid"], dur=ev["dur"], category=cat
                )
            )
    assert all(len(times) == 0 for times in send_recv_time.values()), [
        len(times) == 0 for times in send_recv_time.values()
    ]
    for events in kernel_events.values():
        events.sort(key=lambda x: x.ts)

    x = None
    for events in tqdm.tqdm(
        kernel_events.values(),
        total=len(kernel_events),
        desc="Gathering kernel time stats for all processes...",
    ):
        stats = kernelStatFromEvents(events, global_start, global_end)
        if x is None:
            x = stats
        else:
            x = x + stats
    return x
