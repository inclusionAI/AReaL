import ray
import numpy as np
from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    vLLMHttpServerBase,
    SamplingParams, 
    TokensPrompt, 
    RolloutConfig,
    RewardModelConfig,
    HFModelConfig,
    RolloutMode,
    ActorHandle,
)
from typing import Any, Optional

from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
)
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from ..replica import VerlToolTokenOutput


@ray.remote(num_cpus=1)
class VerlToolvLLMHttpServer(vLLMHttpServerBase):
    """vLLM http server in single node, this is equivalent to launch server with command line:
    ```
    vllm serve --tensor-parallel-size=8 ...
    ```
    """

    def __init__(
        self,
        config: RolloutConfig | RewardModelConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
    ):
        original_max_model_len = config.max_model_len
        super().__init__(config, model_config, rollout_mode, workers, replica_rank, node_rank, gpus_per_node, nnodes)
        self.config.max_model_len = max(original_max_model_len, self.config.max_model_len) if original_max_model_len is not None else self.config.max_model_len
        
    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
    ) -> VerlToolTokenOutput:
        """Generate sequence with token-in-token-out."""
        # TODO(@wuxibin): switch to `/generate` http endpoint once multi-modal support ready.
        max_tokens = min(self.config.max_model_len - len(prompt_ids), sampling_params.get("max_tokens", self.config.response_length))
        sampling_params["max_tokens"] = max_tokens
        sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None
        sampling_params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))
        sampling_params = SamplingParams(**sampling_params)
        prompt_ids = _qwen2_5_dedup_multimodal_tokens(prompt_ids, self.model_config.processor)
        
        multi_modal_data: dict[str, Any] = {}
        if image_data:
            multi_modal_data["image"] = image_data
        if audio_data:
            multi_modal_data["audio"] = audio_data
        prompt = TokensPrompt(
            prompt_token_ids=prompt_ids, multi_modal_data=multi_modal_data or None
        )

        # Add lora request
        lora_request = None
        if self.model_config.lora_rank > 0:
            # Make sure we also check that the lora is already loaded in the engine
            lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
            if lora_loaded:
                lora_request = LoRARequest(
                    lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH
                )

        generator = self.engine.generate(
            prompt=prompt, sampling_params=sampling_params, request_id=request_id, lora_request=lora_request
        )

        # Get final response
        final_res: Optional[RequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None

        token_ids = final_res.outputs[0].token_ids
        log_probs = None
        if sampling_params.logprobs is not None:
            log_probs = [logprobs[token_ids[i]].logprob for i, logprobs in enumerate(final_res.outputs[0].logprobs)]
        finish_reason = final_res.outputs[0].finish_reason
        stop_reason = final_res.outputs[0].stop_reason
        text = final_res.outputs[0].text
        finished = final_res.finished

        return VerlToolTokenOutput(token_ids=token_ids, log_probs=log_probs, finish_reason=finish_reason, stop_reason=stop_reason, text=text, finished=finished)


def _qwen2_5_dedup_multimodal_tokens(prompt_ids: list[int], processor):
    """Deduplicate consecutive image and *audio* tokens in prompt_ids for Qwen2.5-VL/Omni, since vLLM will replicate the
    <|image_pad|> token by image_data and <|AUDIO|> token by audio_data.

    For example,
    ```
    <|vision_start|><|image_pad|><|image_pad|>...<|image_pad|><|vision_end|>
    =>
    <|vision_start|><|image_pad|><|vision_end|>
    
    <|audio_bos|><|AUDIO|><|AUDIO|>...<|AUDIO|><|audio_eos|>
    =>
    <|audio_bos|><|AUDIO|><|audio_eos|>
    ```
    """
    if processor is None:
        return prompt_ids
    
    # Check for Qwen2_5OmniProcessor (needs to get image_token_id from tokenizer)
    is_qwen2_5_omni = "Qwen2_5OmniProcessor" in processor.__class__.__name__
    # Check for Qwen2VLImageProcessor (has image_token_id attribute)
    is_qwen2vl = not is_qwen2_5_omni and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__
    
    if is_qwen2vl or is_qwen2_5_omni:
        prompt_ids = np.array(prompt_ids)

        # Create a mask where True indicates elements to keep
        mask = np.ones(len(prompt_ids), dtype=bool)

        # Get image_token_id based on processor type
        if is_qwen2vl:
            # Qwen2VLImageProcessor has image_token_id attribute
            image_token_id = processor.image_token_id
        else:
            # Qwen2_5OmniProcessor: get image_token_id from tokenizer
            image_token_id = processor.tokenizer.image_token_id

        # Deduplicate consecutive image tokens
        is_image_token = prompt_ids == image_token_id
        mask[1:] &= ~(is_image_token[1:] & is_image_token[:-1])

        if is_qwen2_5_omni:
            # For Qwen2_5OmniProcessor, also deduplicate consecutive audio tokens
            audio_token_id = processor.tokenizer.audio_token_id
            is_audio_token = prompt_ids == audio_token_id
            mask[1:] &= ~(is_audio_token[1:] & is_audio_token[:-1])

        return prompt_ids[mask].tolist()
    else:
        return prompt_ids
