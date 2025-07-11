# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

from abc import ABC, abstractmethod

from realhf.impl.model.parallelism.pipeline_parallel.instruction import (
    BackwardPass,
    ForwardPass,
    OptimizerStep,
    RecvActivation,
    RecvGrad,
    RecvNextTokens,
    ReduceGrads,
    SendActivation,
    SendGrad,
    SendNextTokens,
)


class PipeSchedule(ABC):
    """Directs the execution of a pipeline engine by generating sequences of
    :class:`PipeInstruction`.

    Schedules are generators that yield sequences of
    :class:`PipeInstruction` to process the micro-batches in one batch.
    Each yielded step is atomic in the sense that a barrier
    synchronization can be placed between successive steps without
    deadlock.

    Below is an example schedule that implements data parallelism with gradient accumulation:

    .. code-block:: python

        class DataParallelSchedule(PipeSchedule):
            def steps(self):
                for step_id in range(self.micro_batches):
                    cmds = [
                        LoadMicroBatch(buffer_id=0),
                        ForwardPass(buffer_id=0),
                        BackwardPass(buffer_id=0),
                    ]
                    if step_id == self.micro_batches - 1:
                        cmds.extend([
                            ReduceGrads(),
                            OptimizerStep(),
                        ])
                    yield cmds

            def num_pipe_buffers(self):
                return 1

    Args:
        micro_batches (int): The number of micro-batches that comprise a batch.
        stages (int): The number of pipeline stages.
        stage_id (int): The pipe stage that will execute the generated schedule.
    """

    def __init__(self, micro_batches, stages, stage_id):
        super().__init__()
        self.micro_batches = micro_batches
        self.stages = stages
        self.stage_id = stage_id
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1

        self.terminate_hooks = []
        self.current_micro_batch_id = 0

    def terminate(self):
        return self.terminate_hooks

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(micro_batches={self.micro_batches}, stages={self.stages}, stage_id={self.stage_id})"

    @abstractmethod
    def steps(self):
        """Yield a list of :class:`PipeInstruction` for each step in the
        schedule.

        .. note::
            Schedules must implement ``steps()`` to define the schedule.

        Returns:
            Instructions to be executed as one step of the pipeline
        """
        pass

    def num_pipe_buffers(self):
        """The number of pipeline buffers that will be used by this stage.

        .. note::
            Schedules should specialize ``num_pipe_buffers()`` for memory savings at scale.

        Returns:
            The number of buffers for the engine to allocate.
        """
        return self.micro_batches

    def _valid_micro_batch(self, micro_batch_id):
        return 0 <= micro_batch_id < self.micro_batches

    def _valid_stage(self, stage_id):
        return 0 <= stage_id < self.stages

    @property
    def stage(self):
        """Stage index used to configure this schedule."""
        return self.stage_id

    @property
    def num_stages(self):
        """The number of total pipeline stages used to configure this
        schedule."""
        return self.stages

    @property
    def n_pp_mbs(self):
        """The number of total micro_batches used to configure this
        schedule."""
        return self.micro_batches

    @property
    def is_first_stage(self):
        """True if the configured ``stage_id`` is the first stage in the
        pipeline."""
        return self.stage_id == 0

    @property
    def is_last_stage(self):
        """True if the configured ``stage_id`` is the last stage in the
        pipeline."""
        return self.stage_id == self.stages - 1

    def _buffer_idx(self, micro_batch_id):
        """Map a micro-batch index to a pipeline buffer index.

        This method uses a cyclic allocation strategy.

        Args:
            micro_batch_id (int): The micro-batch index relative to the beginning of the schedule.

        Returns:
            int: The index of the buffer that should store data.
        """
        assert self._valid_micro_batch(micro_batch_id)
        return micro_batch_id % self.num_pipe_buffers()

    def __iter__(self):
        self.it = None
        return self

    def __next__(self):
        if self.it is None:
            self.it = self.steps()
        return next(self.it)


class InferenceSchedule(PipeSchedule):
    """A schedule for inferencing batches using pipeline parallelism."""

    def steps(self):
        """"""
        total_steps = self.micro_batches + self.stages - 1
        for step_id in range(total_steps):
            cmds = []
            micro_batch_id = step_id - self.stage_id

            if _is_even(self.stage_id):
                if self._valid_stage(self.next_stage):
                    if self._valid_micro_batch(micro_batch_id - 1):
                        cmds.append(SendActivation(self.stage_id, micro_batch_id - 1))
                if self._valid_stage(self.prev_stage):
                    if self._valid_micro_batch(micro_batch_id):
                        cmds.append(RecvActivation(self.stage_id, micro_batch_id))
            else:
                if self._valid_stage(self.prev_stage):
                    if self._valid_micro_batch(micro_batch_id):
                        cmds.append(RecvActivation(self.stage_id, micro_batch_id))

                if self._valid_stage(self.next_stage):
                    if self._valid_micro_batch(micro_batch_id - 1):
                        cmds.append(SendActivation(self.stage_id, micro_batch_id - 1))

            if self._valid_micro_batch(micro_batch_id):
                cmds.append(ForwardPass(self.stage_id, micro_batch_id))

            yield step_id, micro_batch_id, cmds

    def num_pipe_buffers(self):
        """Only two pipeline buffers are required for inferencing.

        Returns:
            ``2``
        """
        return 2


class GenerateSchedule(PipeSchedule):
    """A schedule for generate.

    Difference between this schedule and InferenceSchedule is that last
    stage will not load data, and the last stage will send the result to
    the first stage for the next generation round.
    """

    def __init__(self, micro_batches, stages, stage_id, max_new_tokens):
        super().__init__(micro_batches, stages, stage_id)
        self.prev_stage = self.prev_stage % self.stages
        self.next_stage = self.next_stage % self.stages
        self.max_new_tokens = max_new_tokens
        self.max_steps = (
            max_new_tokens * max(self.n_pp_mbs, self.stages) + self.n_pp_mbs - 1
        )  # a configurable upper bound

    def _valid_token_id(self, token_id):
        return token_id < self.max_new_tokens

    def steps(self):
        last_micro_batch_id = -1
        last_token_id = -1
        for step_id in range(self.max_steps):
            cmds = []
            micro_batch_id = (
                (step_id - self.stage_id) % max(self.n_pp_mbs, self.stages)
                if step_id - self.stage_id >= 0
                else -1
            )  # micro batch id for current stage
            first_round = (
                step_id < self.n_pp_mbs
            )  # whether it is the first round of generate
            last_stage_last_mbid = (
                (step_id - self.stages) % max(self.n_pp_mbs, self.stages)
                if step_id >= self.stages
                else -1
            )
            # the micro_batch_id of the last stage on last step
            token_id = (step_id - self.stage_id) // max(self.n_pp_mbs, self.stages)
            # token id in current round

            # TODO: from last stage to first stage, need one buffer for each microbatch?
            if _is_even(self.stage_id):
                if (
                    self._valid_micro_batch(last_micro_batch_id)
                    and self._valid_token_id(last_token_id)
                    and not self.is_last_stage
                ):
                    cmds.append(
                        SendActivation(
                            self.stage_id, last_micro_batch_id, step_id=token_id
                        )
                    )
                # intermediate stage recv
                if (
                    self._valid_micro_batch(micro_batch_id)
                    and self._valid_token_id(token_id)
                    and not self.is_first_stage
                ):
                    cmds.append(
                        RecvActivation(self.stage_id, micro_batch_id, step_id=token_id)
                    )
            else:
                # odd stage could not be first stage
                if self._valid_micro_batch(micro_batch_id) and self._valid_token_id(
                    token_id
                ):
                    cmds.append(
                        RecvActivation(self.stage_id, micro_batch_id, step_id=token_id)
                    )
                # last stage should not send activation except first stage requires
                if (
                    self._valid_micro_batch(last_micro_batch_id)
                    and self._valid_token_id(last_token_id)
                    and not self.is_last_stage
                ):
                    cmds.append(
                        SendActivation(
                            self.stage_id, last_micro_batch_id, step_id=token_id
                        )
                    )

            # last stage send next tokens when first stage requires.
            if (
                self.is_last_stage
                and self._valid_micro_batch(last_micro_batch_id)
                and self._valid_token_id(last_token_id)
            ):
                cmds.append(
                    SendNextTokens(
                        self.stage_id,
                        last_micro_batch_id,
                        step_id=last_token_id,
                    )
                )
            if self.is_first_stage and self._valid_micro_batch(last_stage_last_mbid):
                cmds.append(
                    RecvNextTokens(
                        self.stage_id, last_stage_last_mbid, step_id=token_id
                    )
                )

            if self._valid_micro_batch(micro_batch_id) and self._valid_token_id(
                token_id
            ):
                cmds.append(
                    ForwardPass(self.stage_id, micro_batch_id, step_id=token_id)
                )

            last_micro_batch_id = micro_batch_id
            last_token_id = token_id
            yield step_id, micro_batch_id, cmds

    def num_pipe_buffers(self):
        """2 buffers for inter stage transfer (except last stage to first
        stage) self.n_pp_mbs buffers for last stage to first stage transfer.

        Returns:
            ``2 + self.n_pp_mbs``
        """
        return 2  # + self.n_pp_mbs


class TrainSchedule(PipeSchedule):
    """A schedule for training a batch using hybrid parallelism.

    Pipeline parallelism is extracted through gradient accumulation and
    thus convergence follows that of a data parallel approach with the
    same batch size.
    """

    def steps(self):
        """"""
        prev_micro_batch_id = -1
        total_steps = 2 * (self.micro_batches + self.stages - 1)
        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also whether it is a
            # forward or backward pass step.
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)
            cmds = []

            # Exchange activations
            if is_forward:
                if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(
                    self.prev_stage
                ):
                    cmds.append(SendGrad(self.stage_id, prev_micro_batch_id, step_id=0))
                if self._valid_micro_batch(micro_batch_id) and self._valid_stage(
                    self.prev_stage
                ):
                    cmds.append(
                        RecvActivation(self.stage_id, micro_batch_id, step_id=0)
                    )
            else:
                if self._valid_micro_batch(micro_batch_id) and self._valid_stage(
                    self.next_stage
                ):
                    cmds.append(RecvGrad(self.stage_id, micro_batch_id, step_id=0))
                if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(
                    self.next_stage
                ):
                    cmds.append(
                        SendActivation(self.stage_id, prev_micro_batch_id, step_id=0)
                    )

            # Computation
            if self._valid_micro_batch(micro_batch_id):
                if is_forward:
                    cmds.append(ForwardPass(self.stage_id, micro_batch_id, step_id=0))
                else:
                    cmds.append(BackwardPass(self.stage_id, micro_batch_id, step_id=0))

            # Model step at the end of the batch
            if step_id == total_steps - 1:
                cmds.append(ReduceGrads(self.stage_id, micro_batch_id, step_id=0))
                cmds.append(OptimizerStep(self.stage_id, micro_batch_id, step_id=0))

            # Prepare state for next time
            prev_micro_batch_id = micro_batch_id
            yield step_id, micro_batch_id, cmds

    def num_pipe_buffers(self):
        """Return the number of pipeline buffers required for this stage.

        This is equivalent to the maximum number of in-flight forward
        passes, since we need to remember the activations of forward
        passes in order to run backpropagation. For synchronous 1F1B,
        this is equivalent to the index difference between this stage
        and the last stage.
        """
        buffers = min(self.stages - self.stage_id, self.micro_batches)
        return max(2, buffers)

    def _step_to_micro_batch(self, step_id):
        if _is_even(step_id) and _is_even(self.stage_id):
            micro_batch_id = self._even_step_forward_id(step_id)
            is_forward = True

        elif _is_odd(step_id) and _is_odd(self.stage_id):
            micro_batch_id = self._odd_step_forward_id(step_id)
            is_forward = True

        elif _is_even(step_id) and _is_odd(self.stage_id):
            micro_batch_id = self._even_step_backward_id(step_id)
            is_forward = False

        elif _is_odd(step_id) and _is_even(self.stage_id):
            micro_batch_id = self._odd_step_backward_id(step_id)
            is_forward = False

        else:
            assert False

        return micro_batch_id, is_forward

    def _even_step_forward_id(self, step_id):
        base = step_id // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _odd_step_forward_id(self, step_id):
        base = (step_id - 1) // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _even_step_backward_id(self, step_id):
        base = step_id // 2
        micro_batch_id = int(base - self.stages + (self.stage_id + 1) // 2)
        return micro_batch_id

    def _odd_step_backward_id(self, step_id):
        base = ((step_id - 1) // 2) - self.stages + 1
        micro_batch_id = int(base + self.stage_id // 2)
        return micro_batch_id


class DataParallelSchedule(PipeSchedule):
    """An example schedule that trains using traditional data parallelism with
    gradient accumulation."""

    def steps(self):
        """"""
        for step_id in range(self.micro_batches):
            cmds = [
                ForwardPass(self.stage_id, step_id, step_id=0),
                BackwardPass(self.stage_id, step_id, step_id=0),
            ]
            if step_id == self.micro_batches - 1:
                cmds.extend(
                    [
                        ReduceGrads(self.stage_id, step_id, step_id=0),
                        OptimizerStep(self.stage_id, step_id, step_id=0),
                    ]
                )
            yield cmds

    def num_pipe_buffers(self):
        """Only one pipeline buffer needed."""
        return 1


def _is_even(x):
    return x % 2 == 0


def _is_odd(x):
    return x % 2 != 0
