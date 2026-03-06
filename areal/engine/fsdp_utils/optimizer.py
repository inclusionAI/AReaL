import os
from collections.abc import Iterator

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.nn import Parameter


def to_precision_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert string to corresponding torch dtype, only supports bfloat16 and float32.

    Args:
        dtype_str: Data type string, supports "bfloat16" or "float32"

    Returns:
        Corresponding torch dtype

    Raises:
        ValueError: If the input dtype is not supported
    """
    dtype_str = dtype_str.lower()
    if dtype_str in ["bfloat16", "bf16"]:
        return torch.bfloat16
    elif dtype_str in ["float32", "fp32"]:
        return torch.float32
    else:
        raise ValueError(
            f"Unsupported dtype: {dtype_str}. Only 'bfloat16' and 'float32' are supported."
        )


# https://github.com/meta-llama/llama-cookbook/blob/v0.0.5/src/llama_cookbook/policies/anyprecision_optimizer.py
class AnyPrecisionAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterator[Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        use_kahan_summation: bool = True,
        momentum_dtype: str = "bfloat16",
        variance_dtype: str = "bfloat16",
        compensation_buffer_dtype: str = "bfloat16",
    ):
        """
        AnyPrecisionAdamW: a flexible precision AdamW optimizer
        with optional Kahan summation for high precision weight updates.
        Allows direct control over momentum, variance and auxiliary compensation buffer dtypes.
        Optional Kahan summation is used to offset precision reduction for the weight updates.
        This allows full training in BFloat16 (equal or better than FP32 results in many cases)
        due to high precision weight updates.

        Args:
            params (iterable): iterable of parameters to optimize or dicts defining parameter groups
            lr (float, optional): learning rate (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square (default: (0.9, 0.999))
            eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
            weight_decay (float, optional): weight decay coefficient (default: 1e-2)

            # Any Precision specific
            use_kahan_summation = creates auxiliary buffer to ensure high precision
            model param updates (default: True)
            momentum_dtype = dtype for momentum  (default: bfloat16)
            variance_dtype = dtype for uncentered variance (default: bfloat16)
            compensation_buffer_dtype = dtype for Kahan summation buffer (default: bfloat16)

            # Usage
            This optimizer implements optimizer states, and Kahan summation
            for high precision updates, all in user controlled dtypes.
            Defaults are variance in BF16, Momentum in BF16.
            This can be run in FSDP mixed precision, amp, or full precision,
            depending on what training pipeline you wish to work with.

            Setting to use_kahan_summation = False, and changing momentum and
            variance dtypes to FP32, reverts this to a standard AdamW optimizer.

        """
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "use_kahan_summation": use_kahan_summation,
            "momentum_dtype": momentum_dtype,
            "variance_dtype": variance_dtype,
            "compensation_buffer_dtype": compensation_buffer_dtype,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """

        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            use_kahan_summation = group["use_kahan_summation"]

            momentum_dtype = to_precision_dtype(group["momentum_dtype"])
            variance_dtype = to_precision_dtype(group["variance_dtype"])
            compensation_buffer_dtype = to_precision_dtype(
                group["compensation_buffer_dtype"]
            )
            for p in group["params"]:
                assert isinstance(p, torch.Tensor)  # lint
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError(
                        "AnyPrecisionAdamW does not support sparse gradients."
                    )

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)

                    # momentum - EMA of gradient values
                    state["exp_avg"] = torch.zeros_like(p, dtype=momentum_dtype)

                    # variance uncentered - EMA of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=variance_dtype)

                    # optional Kahan summation - accumulated error tracker
                    if use_kahan_summation:
                        state["compensation"] = torch.zeros_like(
                            p, dtype=compensation_buffer_dtype
                        )

                # Main processing
                # update the steps for each param group update
                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                grad = p.grad

                if weight_decay:  # weight decay, AdamW style
                    p.data.mul_(1 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # update momentum
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad, grad, value=1 - beta2
                )  # update uncentered variance

                bias_correction1 = 1 - beta1**step  # adjust using bias1
                step_size = lr / bias_correction1

                denom_correction = (
                    1 - beta2**step
                ) ** 0.5  # adjust using bias2 and avoids math import
                centered_variance = (exp_avg_sq.sqrt() / denom_correction).add_(
                    eps, alpha=1
                )

                if use_kahan_summation:  # lr update to compensation
                    compensation = state["compensation"]
                    compensation.addcdiv_(exp_avg, centered_variance, value=-step_size)

                    # update weights with compensation (Kahan summation)
                    # save error back to compensation for next iteration
                    temp_buffer = p.detach().clone()
                    p.data.add_(compensation)
                    compensation.add_(temp_buffer.sub_(p.data))
                else:  # usual AdamW updates
                    p.data.addcdiv_(exp_avg, centered_variance, value=-step_size)


class PerLayerGPUOptimizerStep:
    """Per-layer GPU optimizer step with async H2D/D2H prefetching.

    Instead of running Adam on CPU for all ~67GB of optimizer states,
    streams 1-2 layers at a time (~1.5GB each) to GPU, achieving ~50-80x speedup.

    Requires:
    - optimizer_offload=True (optimizer states on CPU between steps)
    - offload_policy=False (params and grads MUST remain on GPU)

    If CPUOffloadPolicy is used (params/grads on CPU), raises ValueError at config level.
    """

    def __init__(self, model, optimizer, device_id, prefetch_layers=1):
        self.optimizer = optimizer
        self.device = (
            torch.device(f"cuda:{device_id}")
            if isinstance(device_id, int)
            else torch.device(device_id)
        )
        self.prefetch_layers = prefetch_layers
        self._layer_param_groups = self._build_layer_groups(model)
        self._init_states_and_pin()
        self._init_events()

    def _build_layer_groups(self, model) -> list:
        """Group params by FSDP2-wrapped sub-modules (excluding root).

        After apply_fsdp2, each transformer layer and embedding becomes an
        FSDPModule. We find all non-root FSDPModule instances and group their
        params. Any remaining params (final norm, lm_head) form a residual group.
        """
        fsdp_children = []
        for name, module in model.named_modules():
            if name and isinstance(module, FSDPModule):
                fsdp_children.append(module)

        assigned = set()
        groups = []
        for module in fsdp_children:
            params = [
                p
                for p in module.parameters()
                if p.requires_grad and id(p) not in assigned
            ]
            if params:
                groups.append(params)
                for p in params:
                    assigned.add(id(p))

        # Residual: final norm, lm_head, etc.
        residual = [
            p for p in model.parameters() if p.requires_grad and id(p) not in assigned
        ]
        if residual:
            groups.append(residual)
        return groups

    def _get_local_tensor(self, t):
        """Extract regular tensor from DTensor, or return as-is."""
        return t._local_tensor if hasattr(t, "_local_tensor") else t

    def _init_states_and_pin(self):
        """Pre-initialize optimizer states on CPU and pin them for async H2D/D2H.

        Without pinning, .to(non_blocking=True) on CPU tensors is synchronous,
        killing all pipeline overlap. Pinning enables true async DMA transfers.

        IMPORTANT: States must be created on CPU regardless of current param device.
        When param_offload=True, params are loaded to GPU before this is called,
        so zeros_like(param) would create states on GPU (~31GB for 32B model),
        causing OOM during forward/backward. We explicitly create on CPU and
        stream per-layer during optimizer step.

        Called from __init__ (before backward), so we init states for ALL
        trainable params regardless of whether they have gradients yet.
        """
        for group in self._layer_param_groups:
            for param in group:
                if not param.requires_grad:
                    continue
                state = self.optimizer.state[param]
                if len(state) == 0:
                    local_p = self._get_local_tensor(param.data)
                    state["step"] = torch.tensor(0.0, dtype=torch.float32)
                    state["exp_avg"] = torch.zeros(
                        local_p.shape, dtype=local_p.dtype, device="cpu"
                    )
                    state["exp_avg_sq"] = torch.zeros(
                        local_p.shape, dtype=local_p.dtype, device="cpu"
                    )
                else:
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in state:
                            local = self._get_local_tensor(state[key])
                            if local.device.type != "cpu":
                                state[key] = local.to("cpu")
                # Pin optimizer state tensors for async transfers
                for key in ("exp_avg", "exp_avg_sq", "step"):
                    tensor = self._get_local_tensor(state[key])
                    if tensor.device.type == "cpu" and not tensor.is_pinned():
                        state[key] = tensor.pin_memory()

    def _init_events(self):
        """Pre-allocate CUDA events for pipeline synchronization and timing.

        Events are reused across step() calls to avoid per-step allocation overhead.
        """
        num_groups = len(self._layer_param_groups)
        self._h2d_stream = torch.cuda.Stream(device=self.device)
        self._d2h_stream = torch.cuda.Stream(device=self.device)
        self._pre_wait_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(num_groups)
        ]
        self._post_wait_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(num_groups)
        ]
        self._compute_end_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(num_groups)
        ]
        self._h2d_start_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(num_groups)
        ]
        self._h2d_end_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(num_groups)
        ]
        self._d2h_start_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(num_groups)
        ]
        self._d2h_end_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(num_groups)
        ]

    def _prefetch_layer(self, layer_idx):
        """H2D: copy layer's optimizer states (and params/grads if on CPU) to GPU.

        Auto-detects tensor device: skips H2D for tensors already on GPU.
        States are already initialized and pinned by _init_states_and_pin().
        """
        result = {}
        for param in self._layer_param_groups[layer_idx]:
            if param.grad is None:
                continue
            state = self.optimizer.state[param]
            local_p = self._get_local_tensor(param.data)
            local_g = self._get_local_tensor(param.grad.data)
            local_m = self._get_local_tensor(state["exp_avg"])
            local_v = self._get_local_tensor(state["exp_avg_sq"])

            def _to_gpu(t, device=self.device):
                return t if t.device == device else t.to(device, non_blocking=True)

            p_on_gpu = local_p.device == self.device

            result[id(param)] = {
                "param": param,
                "state": state,
                "p_was_cpu": not p_on_gpu,
                "cpu_p": local_p,
                "cpu_m": local_m,
                "cpu_v": local_v,
                "gpu_p": _to_gpu(local_p),
                "gpu_g": _to_gpu(local_g),
                "gpu_m": _to_gpu(local_m),
                "gpu_v": _to_gpu(local_v),
                "gpu_step": state["step"].to(self.device, non_blocking=True)
                if not state["step"].is_cuda
                else state["step"],
            }
        return result

    def _run_adam_for_layer(self, gpu_states):
        """Call torch.optim.adam.adam() functional API on GPU tensors."""
        from torch.optim.adam import adam

        group = self.optimizer.param_groups[0]
        beta1, beta2 = group["betas"]
        params, grads, exp_avgs, exp_avg_sqs, steps = [], [], [], [], []
        for buf in gpu_states.values():
            params.append(buf["gpu_p"])
            grads.append(buf["gpu_g"])
            exp_avgs.append(buf["gpu_m"])
            exp_avg_sqs.append(buf["gpu_v"])
            steps.append(buf["gpu_step"])
        if not params:
            return
        adam(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            [],  # max_exp_avg_sqs (empty for non-amsgrad)
            steps,
            amsgrad=group.get("amsgrad", False),
            has_complex=False,
            beta1=beta1,
            beta2=beta2,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            eps=group.get("eps", 1e-8),
            maximize=group.get("maximize", False),
            foreach=True,
            capturable=False,
            differentiable=False,
            fused=None,
            grad_scale=None,
            found_inf=None,
            decoupled_weight_decay=group.get("decoupled_weight_decay", False),
        )

    def _offload_layer(self, gpu_states):
        """D2H: copy updated param data + optimizer states back to CPU.

        Only copies back tensors that were originally on CPU.
        CPU destination tensors are pinned, enabling async D2H via non_blocking.
        """
        for buf in gpu_states.values():
            if buf["p_was_cpu"]:
                buf["cpu_p"].copy_(buf["gpu_p"], non_blocking=True)
            buf["cpu_m"].copy_(buf["gpu_m"], non_blocking=True)
            buf["cpu_v"].copy_(buf["gpu_v"], non_blocking=True)
            state = buf["state"]
            if isinstance(state["step"], torch.Tensor):
                state["step"].copy_(buf["gpu_step"], non_blocking=True)

    @torch.no_grad()
    def step(self):
        """Per-layer GPU optimizer step with async prefetch pipeline.

        Uses CUDA events (not wait_stream) for fine-grained synchronization,
        allowing H2D prefetch of layer i+2 to overlap with Adam on layer i
        and D2H offload of layer i-1.

        After completion, populates self.last_step_metrics dict with:
        - step_time_s: wall-clock time of the entire step
        - peak_memory_gb: peak GPU memory during the step
        - num_layer_groups: number of layer groups processed
        - compute_stall_count: layers where compute waited for H2D
        - compute_stall_time_s: total time compute was blocked by H2D
        - avg_h2d_ms, avg_compute_ms, avg_d2h_ms: per-phase avg timing
        """
        import time

        verbose = os.getenv("VERL_MEMORY_PROFILE", "0") == "1"

        # Track peak memory: reset before step, read after
        torch.cuda.reset_peak_memory_stats(self.device)
        mem_before = torch.cuda.memory_allocated(self.device)
        t_start = time.perf_counter()

        h2d_stream = self._h2d_stream
        d2h_stream = self._d2h_stream
        compute_stream = torch.cuda.current_stream(self.device)
        num_groups = len(self._layer_param_groups)
        gpu_states = [None] * num_groups

        # Reuse pre-allocated events
        pre_wait_events = self._pre_wait_events
        post_wait_events = self._post_wait_events
        compute_end_events = self._compute_end_events
        h2d_start_events = self._h2d_start_events
        h2d_end_events = self._h2d_end_events
        d2h_start_events = self._d2h_start_events
        d2h_end_events = self._d2h_end_events

        # Prefetch initial layers with per-layer events
        for i in range(min(self.prefetch_layers + 1, num_groups)):
            with torch.cuda.stream(h2d_stream):
                h2d_stream.record_event(h2d_start_events[i])
                gpu_states[i] = self._prefetch_layer(i)
                h2d_stream.record_event(h2d_end_events[i])

        # Process each layer
        for i in range(num_groups):
            # Record BEFORE wait — measures actual GPU-side stall time
            compute_stream.record_event(pre_wait_events[i])
            compute_stream.wait_event(h2d_end_events[i])
            compute_stream.record_event(post_wait_events[i])

            self._run_adam_for_layer(gpu_states[i])
            compute_stream.record_event(compute_end_events[i])

            # Prefetch next layer (overlaps with D2H below)
            next_idx = i + self.prefetch_layers + 1
            if next_idx < num_groups:
                with torch.cuda.stream(h2d_stream):
                    h2d_stream.record_event(h2d_start_events[next_idx])
                    gpu_states[next_idx] = self._prefetch_layer(next_idx)
                    h2d_stream.record_event(h2d_end_events[next_idx])

            # Offload current layer (waits only for this layer's compute)
            d2h_stream.wait_event(compute_end_events[i])
            with torch.cuda.stream(d2h_stream):
                d2h_stream.record_event(d2h_start_events[i])
                self._offload_layer(gpu_states[i])
                d2h_stream.record_event(d2h_end_events[i])
            gpu_states[i] = None  # free GPU memory

        d2h_stream.synchronize()

        # Prevent cross-phase cache pollution
        torch.cuda.empty_cache()

        if verbose and (not dist.is_initialized() or dist.get_rank() == 0):
            _diag_alloc = torch.cuda.memory_allocated(self.device)
            _diag_reserved = torch.cuda.memory_reserved(self.device)
            _diag_free, _diag_total = torch.cuda.mem_get_info(self.device)
            _diag_device = _diag_total - _diag_free
            _diag_cached = _diag_reserved - _diag_alloc
            print(
                f"[PerLayerOptStep:diag] after_step: "
                f"alloc={_diag_alloc / (1024**3):.2f}GB, "
                f"reserved={_diag_reserved / (1024**3):.2f}GB, "
                f"cached_free={_diag_cached / (1024**3):.2f}GB, "
                f"device={_diag_device / (1024**3):.2f}GB"
            )

        # Collect metrics (all streams synchronized)
        step_time = time.perf_counter() - t_start
        peak_memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)
        mem_after = torch.cuda.memory_allocated(self.device)

        # Compute per-phase timings from CUDA events
        h2d_times, compute_times, d2h_times, wait_times = [], [], [], []
        for i in range(num_groups):
            wait_times.append(pre_wait_events[i].elapsed_time(post_wait_events[i]))
            compute_times.append(
                post_wait_events[i].elapsed_time(compute_end_events[i])
            )
            h2d_times.append(h2d_start_events[i].elapsed_time(h2d_end_events[i]))
            d2h_times.append(d2h_start_events[i].elapsed_time(d2h_end_events[i]))

        stall_threshold_ms = 0.1
        stall_count = sum(1 for w in wait_times[1:] if w > stall_threshold_ms)
        stall_time_ms = sum(w for w in wait_times[1:] if w > stall_threshold_ms)

        avg_h2d = sum(h2d_times) / len(h2d_times) if h2d_times else 0.0
        avg_compute = sum(compute_times) / len(compute_times) if compute_times else 0.0
        avg_d2h = sum(d2h_times) / len(d2h_times) if d2h_times else 0.0

        self.last_step_metrics = {
            "step_time_s": step_time,
            "peak_memory_gb": peak_memory_gb,
            "num_layer_groups": num_groups,
            "avg_h2d_ms": avg_h2d,
            "avg_compute_ms": avg_compute,
            "avg_d2h_ms": avg_d2h,
            "compute_stall_count": stall_count,
            "compute_stall_time_ms": stall_time_ms,
        }

        if verbose and (not dist.is_initialized() or dist.get_rank() == 0):
            mem_delta_gb = (mem_after - mem_before) / (1024**3)
            print(
                f"[PerLayerOptStep] {num_groups} groups, prefetch={self.prefetch_layers}, "
                f"time={step_time:.2f}s, peak={peak_memory_gb:.2f}GB, "
                f"mem_delta={mem_delta_gb:+.3f}GB"
            )
            print(
                f"[PerLayerOptStep] avg H2D={avg_h2d:.2f}ms, "
                f"compute={avg_compute:.2f}ms, D2H={avg_d2h:.2f}ms | "
                f"stalls={stall_count}/{num_groups}"
            )
            if stall_count > 0:
                print(
                    f"[PerLayerOptStep] WARNING: {stall_count} compute stalls detected. "
                    f"Consider increasing optimizer_step_prefetch_layers "
                    f"(current={self.prefetch_layers})"
                )
