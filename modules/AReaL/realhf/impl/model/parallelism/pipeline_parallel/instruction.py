# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import sys
from typing import *


class PipeInstruction:
    """Base class for all instructions to be executed by the pipeline engine.

    All keyword arguments are stored as members similar to a ``namedtuple``. These are
    then accessible to the :class:`PipeEngine` during execution.

    Each instruction has following attributes:
    1. stage_id: pipeline stage that the instruction should be executed by.
    2. micro_batch_id: the ID of data micro batch that this instruction should be executed on.
    3. step_id: usually used by generation schedules, identical to generation token id.
    4. priority: priority of the instruction, higher priority instructions should be scheduled first in
                 the dynamic schedule.
    5. deps: list of instructions that this instruction depends on.
    6. bind: Instruction that this instruction is binded with. Binded instructions are send/recv instructions
             that should appear in pair, belong to different (adjancent) stages and have the same micro_batch_id.
             Two binded instructions should have the same dependency list. This feature is used to avoid NCCL
             communication deadlock.
    """

    def __init__(
        self,
        stage_id: int,
        micro_batch_id: int,
        deps: List["PipeInstruction"] = [],
        bind: List["PipeInstruction"] = [],
        step_id: int = 0,
        schedule_id: int = 0,
    ):
        self.stage_id = stage_id
        self.micro_batch_id = micro_batch_id
        self.deps = deps
        self.bind = bind
        self.name = self.__class__.__name__
        self.step_id = step_id
        self.args = (stage_id, micro_batch_id, step_id)
        self.__str_encode = (
            f"{self.name};{self.stage_id};{self.micro_batch_id};{self.step_id}"
        )
        self.schedule_id = schedule_id

    def __repr__(self):
        return f"{self.name}{self.args}"
        # return call_to_str(self.name, self.args, self.kwargs)

    def __eq__(self, other: "PipeInstruction"):
        return (
            self.stage_id == other.stage_id
            and self.micro_batch_id == other.micro_batch_id
            and self.step_id == other.step_id
            and self.name == other.name
        )

    def __lt__(self, other: "PipeInstruction"):
        # order by stage_id, micro_batch_id, step_id
        # used to sort finded results in InstructionSet
        return (
            self.stage_id < other.stage_id
            or (
                self.stage_id == other.stage_id
                and self.micro_batch_id < other.micro_batch_id
            )
            or (
                self.stage_id == other.stage_id
                and self.micro_batch_id == other.micro_batch_id
                and self.step_id < other.step_id
            )
        )

    def encode_str(self) -> str:
        return self.__str_encode

    def encode(self):
        return self.encode_str().encode(encoding="utf-8")

    @classmethod
    def decode(cls, encoded: Union[bytes, str]) -> "PipeInstruction":
        if isinstance(encoded, bytes):
            s = encoded.decode(encoding="utf-8")
        else:
            s = encoded
        cls_name, stage_id, micro_batch_id, step_id = s.split(";")
        cls_ = getattr(sys.modules[__name__], cls_name)
        return cls_(
            stage_id=int(stage_id),
            micro_batch_id=int(micro_batch_id),
            step_id=int(step_id),
        )


def decode_stage_by_encoded(s: str):
    return int(s.split(";")[1])


class OptimizerStep(PipeInstruction):
    """Performs one step with the optimizer and zeros gradients.

    .. note:: Should be issued after :class:`ReduceGrads` and :class:`ReduceTiedGrads`.

    .. note:: Can be a synchronization point among data-parallel ranks.
    """

    pass


class ReduceGrads(PipeInstruction):
    """Reduce the computed gradients among data-parallel processes within the
    stage."""

    pass


# Compute
class ForwardPass(PipeInstruction):
    """Compute a forward pass."""

    pass


class BackwardPass(PipeInstruction):
    """Compute a backward pass and accumulate gradients."""

    pass


# Communication
class SendActivation(PipeInstruction):
    """Send activations to the next stage in the pipeline."""

    pass


class RecvActivation(PipeInstruction):
    """Receive activations from the previous stage in the pipeline."""

    pass


class SendGrad(PipeInstruction):
    """Send computed gradients to the previous pipeline stage.

    with respect to the received activations
    """

    pass


class RecvGrad(PipeInstruction):
    """Receive computed gradients the next pipeline stage."""

    pass


# generation
class SendNextTokens(PipeInstruction):
    """In GenerateSchedule, send next tokens to the first stage.

    Only available in the last stage.
    """

    pass


class RecvNextTokens(PipeInstruction):
    """In GenerateSchedule, recv next tokens from the last stage.

    Only available in the first stage.
    """

    pass
