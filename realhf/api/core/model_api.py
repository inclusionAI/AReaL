# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import abc
import asyncio
import dataclasses
import keyword
from typing import Any, Callable, Dict, Hashable, List, Literal, Optional, Tuple, Union

import aiohttp
import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
import transformers

import realhf.base.logging as logging
from realhf.api.cli_args import GenerationHyperparameters
from realhf.api.core.config import (
    ModelAbstraction,
    ModelBackendAbstraction,
    ModelInterfaceAbstraction,
    ModelName,
    ModelWrapperAbstraction,
)
from realhf.api.core.data_api import MicroBatchSpec, SequenceSample, load_hf_tokenizer
from realhf.base.datapack import flat2d
from realhf.base.recover import StepInfo

logger = logging.getLogger("model_api")


class ZeroTotalLossWeightException(Exception):
    pass


@dataclasses.dataclass
class GenRespMeta:
    qid: str
    accepted: bool
    n_tokens: int


@dataclasses.dataclass
class GenReqMeta:
    ## Meta info used to schedule the request. ##
    qid: Hashable
    prompt_len: int
    group_size: int
    new_token_budget: int
    predicted_new_tokens: int | None
    previous_server_url: str = ""
    previous_version: int = -1


@dataclasses.dataclass
class ModelVersionReq:
    server_url: str


@dataclasses.dataclass
class APIGenerateInput:
    # The unique query id of this prompt
    qid: Hashable
    # prompt token ids
    prompt_ids: List[int]
    # prompt token ids + generated prefix, the input to server
    input_ids: List[int]
    # the sampling params to server, may limit n=1 and max_new_tokens
    # for partial rollout
    gconfig: GenerationHyperparameters
    # stop tokens, usually EOS and PAD
    stop_token_ids: List[int] = dataclasses.field(default_factory=list)
    # whether to return logprobs
    return_logprob: bool = True
    # logprobs of preivous generation
    # length len(input_ids) - len(prompt_ids)
    prev_logprobs: List[float] = dataclasses.field(default_factory=list)
    # the weight version when submitting this request
    version_start: int = -1

    # other metadata
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class APIGenerateOutput:
    ## input re-export ##
    qid: Hashable
    prompt_ids: List[int]
    input_ids: List[int]
    gconfig: GenerationHyperparameters
    prev_logprobs: List[float] = dataclasses.field(default_factory=list)
    version_start: int = -1
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    ## outputs. To be amended by the reply. ##
    # output token ids
    output_ids: List[List[int]] = dataclasses.field(default_factory=list)
    # output logprobs with the same length as output_ids
    output_logprobs: List[List[float]] = dataclasses.field(default_factory=list)
    # the weight version when finishing this request
    version_end: List[int] = dataclasses.field(default_factory=list)
    # whether truncated
    no_eos: List[bool] = dataclasses.field(default_factory=list)

    # statistics
    latency: float = float("inf")
    ttft: float = float("inf")  # Time to first token
    itl: List[float] = dataclasses.field(
        default_factory=list
    )  # List of inter-token latencies

    @classmethod
    def from_input(cls, inp: APIGenerateInput):
        return cls(
            qid=inp.qid,
            prompt_ids=inp.prompt_ids,
            input_ids=inp.input_ids,
            gconfig=inp.gconfig,
            prev_logprobs=inp.prev_logprobs,
            version_start=inp.version_start,
            metadata=inp.metadata,
        )

    @staticmethod
    def concat(outputs: List["APIGenerateOutput"]):
        assert len(set([o.qid for o in outputs])) == 1
        return APIGenerateOutput(
            qid=outputs[0].qid,
            prompt_ids=outputs[0].prompt_ids,
            input_ids=outputs[0].input_ids,
            gconfig=outputs[0].gconfig,
            prev_logprobs=outputs[0].prev_logprobs,
            version_start=outputs[0].version_start,
            metadata=outputs[0].metadata,
            output_ids=sum([o.output_ids for o in outputs], []),
            output_logprobs=sum([o.output_logprobs for o in outputs], []),
            version_end=sum([o.version_end for o in outputs], []),
            no_eos=sum([o.no_eos for o in outputs], []),
            latency=max([o.latency for o in outputs]),
            ttft=max([o.ttft for o in outputs]),
            itl=sum([o.itl for o in outputs], []),
        )

    @property
    def group_size(self):
        return len(self.output_ids)

    @property
    def output_lens(self):
        return [len(x) for x in self.output_ids]

    @property
    def input_len(self):
        return len(self.input_ids)

    @property
    def prompt_len(self):
        return len(self.prompt_ids)

    @property
    def gen_lens(self):
        return [len(x) + self.input_len - self.prompt_len for x in self.output_ids]

    def get_logprobs(self) -> List[List[float]]:
        logprobs = []
        for logp in self.output_logprobs:
            assert len(self.prev_logprobs) == self.input_len - self.prompt_len, (
                len(self.prev_logprobs),
                self.input_len,
                self.prompt_len,
            )
            logprobs.append([0.0] * (self.prompt_len - 1) + self.prev_logprobs + logp)
        return logprobs


@dataclasses.dataclass
class BundledGenerationOutputs:
    ## Used for collecting generation outputs for env interaction or training. ##

    # unique query id in the dataset
    qid: Hashable
    # prompt token ids
    prompt_ids: List[int]
    # output token ids excluding the prompt
    output_ids: List[List[int]]
    # whole sequences including the prompt
    seqs: List[List[int]]
    # whole logprobs, one token shorter than seq
    # logps at prompt tokens are zero
    logprobs: List[List[float]]
    # whether truncated
    no_eos: List[bool]
    # server weight version when starting generation
    version_start: List[int]
    # server weight version when generation ends
    version_end: List[int]

    @classmethod
    def from_api_outputs(cls, outputs: List[APIGenerateOutput]):
        assert len(set(o.qid for o in outputs)) == 1
        prompt_len = len(outputs[0].prompt_ids)
        seqs = []
        logprobs = []
        version_starts = []
        for o in outputs:
            for out in o.output_ids:
                seqs.append(o.input_ids + out)
            for logp in o.get_logprobs():
                logprobs.append(logp)
            version_starts += [o.version_start] * o.group_size
        return cls(
            qid=outputs[0].qid,
            prompt_ids=outputs[0].prompt_ids,
            seqs=seqs,
            output_ids=[seq[prompt_len:] for seq in seqs],
            logprobs=logprobs,
            no_eos=sum([o.no_eos for o in outputs], []),
            version_start=version_starts,
            version_end=sum([o.version_end for o in outputs], []),
        )

    @property
    def output_logprobs(self):
        return [lp[self.prompt_len - 1 :] for lp in self.logprobs]

    @property
    def output_lens(self):
        return [len(out) for out in self.output_ids]

    @property
    def seqlens(self):
        return [len(seq) for seq in self.seqs]

    @property
    def prompt_len(self):
        return len(self.prompt_ids)


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(
    total=6 * 60 * 60,
    connect=300,
)


class LLMAPIClient:
    def __init__(
        self, generate_url: str, update_weights_url: str, concurrency_limit: int = -1
    ):
        self.update_weights_url = update_weights_url
        self.generate_url = generate_url
        self.concurrency_limit = concurrency_limit

        self.session: aiohttp.ClientSession
        self.semaphore: asyncio.Semaphore

    async def __aenter__(self):
        conn = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300, force_close=True)
        self.session = aiohttp.ClientSession(
            timeout=AIOHTTP_TIMEOUT,
            connector=conn,
            read_bufsize=1024 * 1024 * 10,
        )
        if self.concurrency_limit > 0:
            self.semaphore = asyncio.Semaphore(self.concurrency_limit)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def async_add_generate_request(
        self, req: APIGenerateInput, stream: bool = True
    ) -> APIGenerateOutput:

        if self.concurrency_limit > 0:
            async with self.semaphore:
                return await self._do_generate(req, stream=stream)
        else:
            return await self._do_generate(req, stream=stream)

    async def _do_generate(
        self, req: APIGenerateInput, stream: bool = True
    ) -> APIGenerateOutput:
        raise NotImplementedError()

    async def async_update_weights_from_disk(self, path):
        raise NotImplementedError()


@dataclasses.dataclass
class ReaLMoEConfig:
    """Configuration for MoE models.

    :param num_experts: The number of experts in the mixture of experts.
    :type num_experts: int
    :param top_k: The number of experts to route per token, also
        interpreted as the `top-k` routing parameter.
    :type top_k: int
    :param routing_type: The load balancing type for the MoE router. Can
        be "aux_loss", "sinkhorn", or "none".
    :type routing_type: str
    :param aux_loss_coeff: The coefficient for the auxiliary loss.
        Effective only when routing_type="aux_loss".
    :type aux_loss_coeff: float
    :param capacity_factor: The capacity factor of each expert. An
        expert will drop tokens if the number of tokens exceeds
        capacity_factor * (num_tokens / num_experts). No tokens will be
        dropped if capacity_factor is None.
    :type capacity_factor: float or None
    :param pad_to_capacity: Whether to pad the input to the capacity of
        the expert.
    :type pad_to_capacity: bool
    :param token_drop_policy: The token drop policy for the MoE. Can be
        either "prob" or "position". If "prob", the tokens with the
        lowest probabilities will be dropped. If "position", tokens at
        the end of each batch will be dropped.
    :type token_drop_policy: str
    :param z_loss_coeff: The coefficient for the z-loss.
    :type z_loss_coeff: float
    :param input_jitter_eps: The input jitter noise for the router.
    :type input_jitter_eps: float
    """

    num_experts: int = 8
    top_k: int = 2
    routing_type: str = "aux_loss"
    aux_loss_coeff: float = 1e-3
    capacity_factor: float = None
    pad_to_capacity: bool = False
    token_drop_policy: str = "probs"
    z_loss_coeff: float = 0.0
    input_jitter_eps: Optional[float] = None
    use_grouped_gemm: bool = False


@dataclasses.dataclass
class ReaLModelConfig:
    """Configuration for the ReaLModel.

    :param n_layers: The number of transformer blocks.
    :type n_layers: int
    :param n_kv_heads: The number of key-value attention heads.
    :type n_kv_heads: int
    :param n_q_heads: The number of query attention heads.
    :type n_q_heads: int
    :param head_dim: The dimension of each attention head.
        If None, it defaults to hidden_dim // n_q_heads.
        If specified, the query layer will have the shape
        (hidden_dim, head_dim * n_q_heads).
    :type head_dim: int or None
    :param hidden_dim: The hidden dimension of the transformer block.
    :type hidden_dim: int
    :param intermediate_dim: The dimension of the intermediate layer in the MLP.
    :type intermediate_dim: int
    :param vocab_size: The vocabulary size.
    :type vocab_size: int
    :param n_positions: The maximum context length. Can be None for
        rotary embedding, where the context length is determined during runtime.
    :type n_positions: Optional[int]
    :param embd_pdrop: The dropout probability for the embedding layer.
    :type embd_pdrop: float
    :param resid_pdrop: The dropout probability for the residual connections.
    :type resid_pdrop: float
    :param attn_pdrop: The dropout probability for the attention weights.
    :type attn_pdrop: float
    :param layer_norm_epsilon: The epsilon value for layer normalization.
    :type layer_norm_epsilon: float
    :param activation_function: The activation function for the MLP.
    :type activation_function: str
    :param scale_attn_by_inverse_layer_idx: Whether to scale the attention weights
        by the inverse of the layer index.
    :type scale_attn_by_inverse_layer_idx: bool
    :param use_attention_bias: Whether to use bias for QKV layers.
    :type use_attention_bias: bool
    :param use_attn_proj_bias: Whether to use bias for the attention projection layer.
    :type use_attn_proj_bias: bool
    :param layer_norm_type: The type of layer normalization. Can be None, "rms", or "gemma".
    :type layer_norm_type: Optional[str]
    :param mlp_type: The type of the MLP. Can be None, "llama", or "moe".
    :type mlp_type: Optional[str]
    :param apply_rotary: Whether to apply rotary embedding.
    :type apply_rotary: bool
    :param rotary_base: The exponential base for the rotary embedding.
    :type rotary_base: float
    :param rotary_interleaved: Whether to use interleaved rotary embedding.
    :type rotary_interleaved: bool
    :param rotary_scaling: The scaling factor for the rotary embedding.
    :type rotary_scaling: Optional[float]
    :param rotary_scaling_type: The type of scaling for the rotary embedding.
    :type rotary_scaling_type: Optional[str]
    :param normalize_embed: Whether to normalize the embeddings
        before passing them through the transformer blocks. Used by Gemma.
    :type normalize_embed: bool
    :param abs_position_embedding_offset: The offset for the absolute position embedding.
        Used by OPT, but OPT is currently not supported.
    :type abs_position_embedding_offset: int
    :param do_layernorm_before: Whether to apply layer normalization before the attention
        rather than after. Used by OPT, but OPT is currently not supported.
    :type do_layernorm_before: bool
    :param tied_embedding: Whether to share the embeddings and output weights.
        Used by models like GPT-2 and Gemma.
    :type tied_embedding: bool
    :param sliding_window: The sliding window size for the attention.
        Currently a placeholder and not supported.
    :type sliding_window: Optional[int]
    :param moe: Configuration for MoE models, only effective when mlp_type="moe".
    :type moe: Optional[ReaLMoEConfig]
    :param is_critic: Whether the model is a critic model.
    :type is_critic: bool
    """

    ### Architectural configurations. ###
    n_layers: int
    n_kv_heads: int
    n_q_heads: int
    hidden_dim: int
    intermediate_dim: int  # for mlp, usually 4*h
    vocab_size: int
    n_positions: int
    head_dim: Optional[int] = None
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    activation_function: str = "gelu"
    scale_attn_by_inverse_layer_idx: bool = True
    scale_attn_weights: bool = True
    # llama does not use attention bias and uses special MLP/LayerNorm layers
    use_attention_bias: bool = True
    use_attn_proj_bias: bool = True
    use_mlp_bias: bool = False
    layer_norm_type: Optional[str] = None
    mlp_type: Optional[str] = None
    # rotary embedding
    apply_rotary: bool = False
    rotary_base: float = 10000.0
    rotary_interleaved: bool = False
    rotary_scaling: Optional[float] = None
    rotary_scaling_type: Optional[str] = None
    rotary_special_impl: Optional[str] = None
    # for gemma
    normalize_embed: bool = False
    # for qwen3
    qk_layernorm: bool = False
    # for opt, it's 2
    abs_position_embedding_offset: int = 0
    do_layernorm_before: bool = True
    # for bailing
    norm_head: bool = False
    norm_softmax: bool = False
    # Tied embedding
    tied_embedding: bool = False
    sliding_window: Optional[int] = None
    # MoE Config
    moe: Optional[ReaLMoEConfig] = None

    # Whether it is a critic/reward model that outputs scores.
    is_critic: bool = False

    # The HuggingFace checkpoint
    base_model_path: Optional[str] = None

    def __post_init__(self):
        if self.is_critic and self.tied_embedding:
            raise ValueError("Critic model cannot share embeddings and output weights.")
        if self.head_dim is None:
            self.head_dim = self.hidden_dim // self.n_q_heads


@dataclasses.dataclass
class FinetuneSpec:
    """The specification for the fine-tuning task.

    :param total_train_epochs: The total number of epochs for training.
    :type total_train_epochs: int
    :param dataset_size: The total number of data.
    :type dataset_size: int
    :param train_batch_size: The batch size for training.
    :type train_batch_size: int
    """

    total_train_epochs: int
    dataset_size: int
    train_batch_size: int

    @property
    def total_train_steps(self):
        dsize = self.dataset_size * self.total_train_epochs
        return (dsize + self.train_batch_size - 1) // self.train_batch_size

    def is_new_epoch(self, version: StepInfo) -> bool:
        return (
            version.global_step * self.train_batch_size
        ) // self.dataset_size > version.epoch

    def is_epoch_last_step(self, version: StepInfo) -> bool:
        return (
            self.dataset_size
            - version.global_step * self.train_batch_size % self.dataset_size
        ) <= self.train_batch_size

    def inc_version(self, version: StepInfo) -> StepInfo:
        if self.is_new_epoch(version):
            version.epoch += 1
            version.epoch_step = 0
        version.epoch_step += 1
        version.global_step += 1
        return version


class PipelinableEngine(abc.ABC):
    """Defines the signature for modules after backend initialization.

    Modules with this signature will be passed to :class:`ModelInterface`
    for model function call execution.
    """

    def train_batch(
        self,
        input_: SequenceSample,
        mb_spec: MicroBatchSpec,
        loss_fn: Callable[[torch.Tensor, SequenceSample], torch.Tensor],
        loss_weight_fn: Callable[[torch.Tensor, SequenceSample], float],
        version_steps: int,
        token_normalize_scope: Literal["global", "dp"] = "global",
    ) -> Dict:
        """Update the model with a batch of data and a loss function.

        :param input_: The input data. It should contain at least the key ``packed_input_ids``,
            which includes the concatenated token sequences. It should also include any other
            entries required to compute the loss.
        :type input_: SequenceSample
        :param loss_fn: The loss function. It takes the output of the forward pass and the
            input data, returning the loss.
        :type loss_fn: Callable[[torch.Tensor, SequenceSample], torch.Tensor]
        :param loss_weight_fn: This function is used to calculate the number of valid tokens
            when normalizing loss across micro batches and DP ranks. Can be `lambda: 1`
            if just taking the average over batches.
        :type loss_weight_fn: Callable[[torch.Tensor, SequenceSample], float]
        :param version_steps: The global step counter for this experiment,
            used by the backend to determine the learning rate schedule.
        :type version_steps: int
        :param global_normalize_scope: The scope of token-wise loss normalization. Choices:
            global: average across all micro batches across DP ranks.
            dp: average across micro batches in current DP rank.
            Default to "global".
        :type global_normalize_scope: Literal["global", "dp"]
        """
        raise NotImplementedError()

    @torch.no_grad()
    def eval_batch(
        self,
        input_: SequenceSample,
        mb_spec: MicroBatchSpec,
        loss_fn: Callable[[torch.Tensor, SequenceSample], torch.Tensor],
    ) -> torch.Tensor | None:
        """Evaluate the model using the forward pass and loss function.

        This method wraps :meth:`forward` with a customized ``post_hook`` and ``aggregate_fn``.

        :param input_: The input data. It should contain at least the key ``packed_input_ids``,
            which includes the concatenated token sequences. It should also include any other
            entries required to compute the loss.
        :type input_: SequenceSample
        :param loss_fn: The loss function. It takes the output of the forward pass and the
            input data, returning the loss.
        :type loss_fn: Callable[[torch.Tensor, SequenceSample], torch.Tensor]
        :return: The aggregated scalar loss if on the last pipe stage.
        :rtype: torch.Tensor | None
        """

        def _loss_fn(out, inp_):
            # To prevent calling data reordering.
            return float(loss_fn(out, inp_))

        return self.forward(
            input_=input_,
            mb_spec=mb_spec,
            post_hook=_loss_fn,
            aggregate_fn=sum,
        )

    def forward(
        self,
        input_: SequenceSample,
        mb_spec: MicroBatchSpec,
        output_seqlens: List[List[int]] | None = None,
        post_hook: Callable[[torch.Tensor, SequenceSample], Any] | None = None,
        aggregate_fn: Callable[[List[Any]], Any] = torch.cat,
    ) -> Any | None:
        """Run the forward pass or inference on the model. Note that it is
        gradient-free.

        To train the model, use :meth:`train_batch` instead.

        :param input_: The input data. It should contain at least the key ``packed_input_ids``,
            which includes the concatenated token sequences.
        :type input_: SequenceSample
        :param post_hook: A function to apply to the output after the forward pass.
            It takes the output tensor and the input data, returning an arbitrary result.
            With a post_hook, we can process the output in mini-batches,
            reducing memory usage for operations such as gathering log-probabilities.
            If None, this function just returns the output tensor.
        :type post_hook: Callable[[torch.Tensor, SequenceSample], Any] | None
        :param aggregate_fn: A function to aggregate the results of the post_hook.
        :type aggregate_fn: Callable[[List[Any]], Any]
        :return: The aggregated result of the post_hook from the last pipeline stage. Returns None otherwise.
            The output before post_hook is a concatenated tensor along the batch-sequence dimension, similar to
            ``packed_input_ids``. For example, if we have 3 sequences with lengths [2, 3, 4],
            and the vocabulary size is 1000, ``packed_input_ids`` should have shape [9],
            and the logits should have shape [9, 1000].
        :rtype: Any | None
        """
        raise NotImplementedError()

    def generate(
        self,
        input_: SequenceSample,
        mb_spec: MicroBatchSpec,
        tokenizer: transformers.PreTrainedTokenizerFast,
        gconfig: GenerationHyperparameters = dataclasses.field(
            default_factory=GenerationHyperparameters
        ),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
        """Generate outputs from the model.

        :param input_: The input data. It should contain at least the key ``packed_input_ids``,
            which includes the concatenated prompts.
        :type input_: SequenceSample
        :param tokenizer: The tokenizer for the model.
        :type tokenizer: transformers.PreTrainedTokenizerFast
        :param gconfig: The generation hyperparameters.
        :type gconfig: GenerationHyperparameters
        :return: For the last pipeline stage, returns the generated tokens, log probabilities, and optionally the logits mask.
            See :class:`GenerationHyperparameters` for more details about the logits mask.
            Returns None for other stages.
            The outputs are stacked tensors along the batch dimension. For example,
            if we have 3 prompts with lengths [2, 3, 4], a maximum generated length of 5,
            and a vocabulary size of 1000, ``packed_input_ids`` should have shape [9],
            generated tokens and log probabilities should have shape [3, 5],
            and the logits should have shape [3, 5, 1000].
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None
        """
        raise NotImplementedError()


@dataclasses.dataclass
class Model:
    """A collection consisting of a neural network, a tokenizer, and metadata
    with a unique name.

    :param name: The unique name of the model.
    :type name: ModelName
    :param module: The neural network module. Its parameters may be
        sharded by tensor or pipeline parallelism.
    :type module: PipelinableEngine | torch.nn.Module
    :param tokenizer: The tokenizer associated with the model.
    :type tokenizer: transformers.PreTrainedTokenizerFast
    :param device: The device on which to run the model.
    :type device: Union[str, torch.device]
    :param dtype: The data type of the model. Defaults to torch.float16
        if None.
    :type dtype: Optional[torch.dtype]
    :param version: The version of the model.
    :type version: StepInfo
    :param ft_spec: The fine-tuning specification for the model.
        Generally not used.
    :type ft_spec: FinetuneSpec
    """

    name: ModelName
    module: PipelinableEngine | torch.nn.Module
    tokenizer: transformers.PreTrainedTokenizerFast
    device: Union[str, torch.device]
    dtype: Optional[torch.dtype] = None
    version: StepInfo = dataclasses.field(default_factory=StepInfo)
    ft_spec: FinetuneSpec = None  # will be initialized by the backend
    backend_name: Optional[str] = None  # will be initialized by the backend

    def __post_init__(self):
        if self.module is None:
            return
        try:
            self.module = self.module.to(self.device)
        except ValueError as e:
            # 4-bit and 8-bit model may fail here
            logger.warning(
                f"Failed to move model to device {self.device} because {e}. Abort to device."
            )

    def inc_version(self):
        self.ft_spec.inc_version(self.version)


class ModelBackend(abc.ABC):
    """A backend that wraps :class:`Model` to provide additional
    functionalities such as pipelined model function calls and ZeRO
    optimization.

    Current backend implementations include inference, DeepSpeed, and Megatron.
    The inference backend provides only inference and generation APIs,
    while the DeepSpeed and Megatron backends also support training.

    The backend offers two main functionalities:

    1. Pipelined generation, inference, and training, implemented in ReaL.

    2. ZeRO optimization, implemented in DeepSpeed and Megatron.

    After initialization, the ``module`` attribute in :class:`Model`
    will have the same signature as :class:`PipelinableEngine`.
    See ``realhf/impl/model/backend`` for concrete implementations.
    """

    @abc.abstractmethod
    def _initialize(self, model: Model, spec: FinetuneSpec) -> Model:
        raise NotImplementedError()

    def initialize(self, model: Model, spec: FinetuneSpec) -> Model:
        """Initialize the model with the backend to support pipelining and
        distributed optimization."""
        model.ft_spec = spec
        return self._initialize(model, spec)

    def destroy(self, model: Model):
        """Destroy the backend and release GPU memory."""
        pass

    def save(self, model: Model, save_dir: str):
        """Save backend states, e.g., optimizer states in the Adam
        optimizer."""
        pass

    def load(self, model: Model, load_dir: str):
        """Load backend states during recover."""
        pass


class NullBackend(ModelBackend):

    def _initialize(self, model: Model, spec: FinetuneSpec) -> Model:
        return model


def null_model(name: ModelName, device: Union[str, torch.device]) -> Model:
    return Model(name, torch.nn.Identity(), None, device)


def tokenizer_only_model(
    name: ModelName, device: Union[str, torch.device], tokenizer_path: str
) -> Model:
    return Model(name, torch.nn.Identity(), load_hf_tokenizer(tokenizer_path), device)


class ModelInterface(abc.ABC):
    """An interface for model training, evaluation, inference, and generation.

    This interface is designed to follow the dependency injection pattern.
    We pass the model to the interface and call its methods, ensuring that model APIs
    and algorithms are fully decoupled. For example, REINFORCE and PPO can exhibit
    different behaviors during training. Separate interfaces can be written for these
    algorithms while using the same model that provides basic forward-backward-update
    functionality (i.e., :class:`PipelinableEngine`).

    During runtime, the master worker requests model workers to execute a specific
    interface type (e.g., generate) on a specific model. The model worker locates
    the corresponding model, passes it into the requested interface, performs the
    computation, and returns the result.

    Users can easily create new interfaces to support customized usage.
    See :doc:`customization` for more details.
    """

    def save(self, model: Model, save_dir: str):
        pass

    def evaluate(
        self,
        model: Model,
        eval_dataloader: torch.utils.data.DataLoader,
    ) -> Dict:
        # NOTE: No n_mbs here because the batch size can be configured in the dataloader.
        return {}

    def inference(
        self,
        model: Model,
        data: SequenceSample,
        mb_spec: MicroBatchSpec,
    ) -> SequenceSample | None:
        raise NotImplementedError()

    def generate(
        self,
        model: Model,
        data: SequenceSample,
        mb_spec: MicroBatchSpec,
    ) -> SequenceSample | None:
        raise NotImplementedError()

    def train_step(
        self,
        model: Model,
        data: SequenceSample,
        mb_spec: MicroBatchSpec,
    ) -> Dict | List[Dict]:
        raise NotImplementedError()

    # Mock methods for creating data and profiling an individual MFC.
    def _mock_generate(self, model: Model, data: SequenceSample):
        return data

    def _mock_inference(self, model: Model, data: SequenceSample):
        return data

    def _mock_train_step(self, model: Model, data: SequenceSample):
        return data

    def mock(
        self,
        type_: str,
        model: Model,
        data: SequenceSample,
    ) -> SequenceSample:
        if type_ == "generate":
            return self._mock_generate(model, data)
        elif type_ == "inference":
            return self._mock_inference(model, data)
        elif type_ == "train_step":
            return self._mock_train_step(model, data)
        else:
            raise ValueError(f"Unsupported interface type {type_}")


class NullInterface(ModelInterface):

    def inference(
        self, model: Model, data: SequenceSample, mb_spec: MicroBatchSpec
    ) -> SequenceSample:
        scores = np.random.randn(sum(len(x) for x in data.seqlens["packed_prompts"]))
        rewards = torch.from_numpy(scores).to(device=model.device, dtype=torch.float32)
        res = SequenceSample(
            keys=["rewards"],
            trailing_shapes=dict(rewards=()),
            dtypes=dict(rewards=torch.float32),
            ids=data.ids,
            seqlens=dict(
                rewards=[
                    torch.tensor([1 for _ in range(len(x))], dtype=torch.int32)
                    for x in data.seqlens["packed_prompts"]
                ],
            ),
            data=dict(rewards=rewards),
        )
        # record rewards for each piece of data
        avg_scores = []
        offset = 0
        for i in range(data.bs):
            score_lis = scores[offset : offset + len(data.seqlens["packed_prompts"][i])]
            avg_scores.append(score_lis.mean().item())
            offset += len(data.seqlens["packed_prompts"][i])
        assert offset == sum(len(x) for x in data.seqlens["packed_prompts"])
        res.metadata["scores"] = avg_scores
        return res

    def train_step(
        self, model: Model, data: SequenceSample, mb_spec: MicroBatchSpec
    ) -> Dict | List[Dict]:
        from realhf.base import constants

        n_tokens = sum(flat2d(data.seqlens[data._get_split_key()]))
        n_tokens = torch.tensor(
            n_tokens, dtype=torch.long, device=constants.current_device()
        )
        dist.all_reduce(n_tokens, group=constants.data_parallel_group())
        if constants.parallelism_rank() == 0:
            logger.info(f"Number of tokens in NullInterface training: {int(n_tokens)}")
        model.inc_version()
        return {}

    def save(self, model: Model, save_dir: str):
        module = model.module.module
        module.save_to_hf(
            tokenizer=model.tokenizer,
            save_dir=save_dir,
        )


ALL_MODEL_CLASSES = {}
ALL_INTERFACE_CLASSES = {}
ALL_BACKEND_CLASSES = {}
ALL_WRAPPER_CLASSES = {}


def register_model(name, model_cls):
    assert name not in ALL_MODEL_CLASSES
    ALL_MODEL_CLASSES[name] = model_cls


def register_interface(name, cls_):
    assert name not in ALL_INTERFACE_CLASSES
    assert issubclass(cls_, ModelInterface)
    ALL_INTERFACE_CLASSES[name] = cls_


def register_backend(name, cls_):
    assert name not in ALL_BACKEND_CLASSES
    assert issubclass(cls_, ModelBackend)
    ALL_BACKEND_CLASSES[name] = cls_


def register_wrapper(name, cls_):
    assert name not in ALL_WRAPPER_CLASSES
    ALL_WRAPPER_CLASSES[name] = cls_


def make_model_wrapper(
    cfg: ModelWrapperAbstraction,
) -> Callable[[Model], Model]:
    cls_ = ALL_WRAPPER_CLASSES[cfg.type_]
    return cls_(**cfg.args)


def make_model(
    cfg: ModelAbstraction, name: ModelName, device: Union[str, torch.device]
) -> Model:
    model_cls = ALL_MODEL_CLASSES[cfg.type_]
    model = model_cls(**cfg.args, name=name, device=device)
    assert isinstance(model, Model)
    for w in cfg.wrappers:
        model = make_model_wrapper(w)(model)
        assert isinstance(model, Model)
    return model


def make_interface(cfg: ModelInterfaceAbstraction) -> ModelInterface:
    cls_ = ALL_INTERFACE_CLASSES[cfg.type_]
    return cls_(**cfg.args)


def make_backend(cfg: ModelBackendAbstraction) -> ModelBackend:
    cls_ = ALL_BACKEND_CLASSES[cfg.type_]
    return cls_(**cfg.args)


register_interface("null", NullInterface)
register_backend("null", NullBackend)
register_model("null", null_model)
register_model("tokenizer", tokenizer_only_model)

SUPPORTED_MODELS = []
HF_MODEL_FAMILY_REGISTRY = {}


def is_valid_function_name(name):
    if not name.isidentifier():
        return False
    if keyword.iskeyword(name):
        return False
    return True


def register_hf_family(
    name: str,
    hf_cls_name: str,
    config_from_hf_converter: Callable[
        [transformers.PretrainedConfig], ReaLModelConfig
    ],
    config_to_hf_converter: Callable[[ReaLModelConfig], transformers.PretrainedConfig],
    sd_from_hf_converter: Callable[[Dict, ReaLModelConfig], Dict],
    sd_to_hf_converter: Callable[[Dict, ReaLModelConfig], Dict],
    embedding_param_names: Callable[[ReaLModelConfig], List[str]],
    tblock_param_names: Callable[[ReaLModelConfig, int], List[str]],
    head_param_names: Callable[[ReaLModelConfig], List[str]],
    real_config_maker: Optional[Callable] = None,
):
    if name in SUPPORTED_MODELS:
        raise ValueError(f"Model {name} is already registered.")
    if not is_valid_function_name(name):
        raise ValueError(f"Model name {name} is not a valid function name in Python.")
    SUPPORTED_MODELS.append(name)
    HF_MODEL_FAMILY_REGISTRY[name] = dict(
        name=name,
        hf_cls_name=hf_cls_name,
        config_from_hf_converter=config_from_hf_converter,
        config_to_hf_converter=config_to_hf_converter,
        sd_from_hf_converter=sd_from_hf_converter,
        sd_to_hf_converter=sd_to_hf_converter,
        embedding_param_names=embedding_param_names,
        tblock_param_names=tblock_param_names,
        head_param_names=head_param_names,
        real_config_maker=real_config_maker,
    )
