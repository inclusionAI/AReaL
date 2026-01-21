# Copyright 2025 Ant Group Inc.

from typing import List, Union

from vllm.distributed import parallel_state
from vllm.engine.llm_engine import LLMEngine, SchedulerOutputState
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.sequence import ExecuteModelRequest

from realhf.base import logging

logger = logging.getLogger("vllm LLMEngine")


class LLMEngine_(LLMEngine):
    """Overwriting the `step` method to synchronize the output across the
    tensor parallel group. See the annotation comment for detailed changes.

    vLLM implements tensor/pipeline parallelism using a single
    `LLMEngine` instance with N-1 remote workers, and only the
    driver worker is supposed to be responsible for output processing.
    The remote workers are only responsible for executing the model.

    Since we follow the convention of replicating data across
    the TP/PP group, it's necessary to synchronize the output
    from the model executor and replicate it for TP/PP.
    """

    def step(self) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
        if self.parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "Pipeline parallelism is only supported through AsyncLLMEngine "
                "as performance will be severely degraded otherwise."
            )

        # For llm_engine, there is no pipeline parallel support, so the engine
        # used is always 0.
        virtual_engine = 0

        # These are cached outputs from previous iterations. None if on first
        # iteration
        cached_outputs = self.cached_scheduler_outputs[virtual_engine]
        seq_group_metadata_list = cached_outputs.seq_group_metadata_list
        scheduler_outputs = cached_outputs.scheduler_outputs
        allow_async_output_proc = cached_outputs.allow_async_output_proc

        ctx = self.scheduler_contexts[virtual_engine]

        # Clear outputs for each new scheduler iteration
        ctx.request_outputs.clear()

        # Skip the scheduler if there are any remaining steps in the seq groups.
        # This ensures that the scheduler is only called again when the current
        # batch has completed.
        if not self._has_remaining_steps(seq_group_metadata_list):
            # Schedule iteration
            (seq_group_metadata_list, scheduler_outputs, allow_async_output_proc) = (
                self.scheduler[virtual_engine].schedule()
            )

            ctx.seq_group_metadata_list = seq_group_metadata_list
            ctx.scheduler_outputs = scheduler_outputs

            # Maybe switch from async mode to sync mode
            if not allow_async_output_proc and len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)

            if (
                self.scheduler_config.is_multi_step
                and scheduler_outputs.num_lookahead_slots > 0
            ):
                # cache the scheduler outputs for the next iteration if we have
                # lookahead slots
                self._cache_scheduler_outputs_for_multi_step(
                    virtual_engine,
                    seq_group_metadata_list,
                    scheduler_outputs,
                    allow_async_output_proc,
                )

        assert seq_group_metadata_list is not None
        assert scheduler_outputs is not None

        if not scheduler_outputs.is_empty():
            finished_requests_ids = self.scheduler[
                virtual_engine
            ].get_and_reset_finished_requests_ids()

            # Check if we have a cached last_output from the previous iteration.
            # For supporting PP this is probably the best way to pass the
            # sampled_token_ids, as a separate broadcast over all the PP stages
            # will cause one virtual engine's microbatch to block the pipeline.
            last_sampled_token_ids = self._get_last_sampled_token_ids(virtual_engine)

            execute_model_req = ExecuteModelRequest(
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
                num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
                running_queue_size=scheduler_outputs.running_queue_size,
                finished_requests_ids=finished_requests_ids,
                # We use ExecuteModelRequest to pass the last sampled_token_ids
                # to each of the non-last PP stages for in-place prepare_input.
                last_sampled_token_ids=last_sampled_token_ids,
            )

            if allow_async_output_proc:
                execute_model_req.async_callback = self.async_callbacks[virtual_engine]

            outputs = self.model_executor.execute_model(
                execute_model_req=execute_model_req
            )

            ######################## Changed here ########################
            if (
                not self.model_executor.driver_worker.is_driver_worker
                and allow_async_output_proc
            ):
                # The following line processes the output asynchronously with GPU computation.
                # Since non-driver workers exit `execute_model` as long as the logits has been
                # computed, we should manually invoke the callback here, otherwise the outputs
                # will not be processes and the non-driver worker will run forever.
                execute_model_req.async_callback()
            # Synchronizing outputs.
            outputs = parallel_state.get_tp_group().broadcast_object(outputs, src=0)
            ######################## Changed here ########################

            # We need to do this here so that last step's sampled_token_ids can
            # be passed to the next iteration for PP.
            if self.scheduler_config.is_multi_step:
                self._update_cached_scheduler_output(virtual_engine, outputs)
        else:
            # Nothing scheduled => If there is pending async postprocessor,
            # then finish it here.
            if len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)
            # No outputs in this case
            outputs = []

        # Finish the current step for all the sequence groups.
        if self.scheduler_config.is_multi_step:
            for seq_group in seq_group_metadata_list:
                seq_group.finish_step()

        if not self._has_remaining_steps(seq_group_metadata_list):
            # clear the cache if we have finished all the steps.
            if self.scheduler_config.is_multi_step:
                self.cached_scheduler_outputs[0] = SchedulerOutputState()

            # is_first_step_output is True only when the num_steps of all
            # the sequences are 1. When the num_steps > 1,
            # multi_step_model_runner does the first-step output append.
            is_first_step_output: bool = (
                False
                if not seq_group_metadata_list
                else seq_group_metadata_list[0].state.num_steps == 1
            )

            # Add results to the output_queue
            ctx.append_output(
                outputs=outputs,
                seq_group_metadata_list=seq_group_metadata_list,
                scheduler_outputs=scheduler_outputs,
                is_async=allow_async_output_proc,
                is_last_step=True,
                is_first_step_output=is_first_step_output,
            )

            if outputs and allow_async_output_proc:
                assert (
                    len(outputs) == 1
                ), "Async postprocessor expects only a single output set"

                self._advance_to_next_step(
                    outputs[0],
                    seq_group_metadata_list,
                    scheduler_outputs.scheduled_seq_groups,
                )

            # Check if need to run the usual non-async path
            if not allow_async_output_proc:
                self._process_model_outputs(ctx=ctx)

                # Log stats.
                self.do_log_stats(scheduler_outputs, outputs)

                # Tracing
                self.do_tracing(scheduler_outputs)
        else:
            # Multi-step case
            return ctx.request_outputs

        if not self.has_unfinished_requests():
            # Drain async postprocessor (if exists)
            if len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)
            assert len(ctx.output_queue) == 0

            # Stop the execute model loop in parallel workers until there are
            # more requests to process. This avoids waiting indefinitely in
            # torch.distributed ops which may otherwise timeout, and unblocks
            # the RPC thread in the workers so that they can process any other
            # queued control plane messages, such as add/remove lora adapters.
            logger.debug("Stopping remote worker execution loop.")
            self.model_executor.stop_remote_worker_execution_loop()

        return ctx.request_outputs
