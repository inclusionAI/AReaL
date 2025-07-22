# Modified from DeepSpeed.
# Copyright [2025] Microsoft Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.distributed as dist
from packaging.version import Version

import realhf.base.constants as constants


def can_send_recv() -> bool:
    # torch_version = Version(torch_info["version"])
    torch_version = Version(torch.__version__)
    sendrecv_min = Version("1.8")
    return torch_version >= sendrecv_min


assert can_send_recv()


def _is_valid_send_recv(src_stage, dest_stage):
    first_stage = 0
    last_stage = constants.grid().pipe_parallel_size - 1
    assert (
        abs(src_stage - dest_stage) == 1
        or (src_stage == first_stage and dest_stage == last_stage)
        or (src_stage == last_stage and dest_stage == first_stage)
    ), f"Functionality currently limited to send and receive between adjacent ranks only (src={src_stage}, dst={dest_stage})"


def send(tensor, dest_stage, async_op=False):
    # NOTE: The input is the stage id rather than the global rank
    src_stage = constants.grid().get_stage_id()
    _is_valid_send_recv(src_stage, dest_stage)

    dest_rank = constants.grid().stage_to_global(stage_id=dest_stage)
    send_method = dist.isend if async_op else dist.send
    return send_method(tensor, constants.to_global_pg_rank(dest_rank))


def recv(tensor, src_stage, async_op=False):
    # NOTE: The input is the stage id rather than the global rank
    dest_stage = constants.grid().get_stage_id()
    _is_valid_send_recv(src_stage, dest_stage)

    src_rank = constants.grid().stage_to_global(stage_id=src_stage)
    recv_method = dist.irecv if async_op else dist.recv
    return recv_method(tensor, constants.to_global_pg_rank(src_rank))
