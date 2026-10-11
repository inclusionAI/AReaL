# Required recovery for the Flash Next recipe

The launcher can require recovery with `QWEN_REQUIRE_RECOVER=1`. A preflight checkpoint
check cannot guarantee that a checkpoint is still available when the trainer loads it.
The recovery handler returns `None` if no checkpoint is found, and the training loop
then starts at step zero.

Check the initialized trainer's `recover_info` inside the recipe's trainer context and
before calling `train`. If recovery is required and the loaded state is absent, raise a
clear error. The trainer context closes resources, and the existing `finally` restores
the SGLang controller factory. Ordinary fresh training remains available when required
recovery is unset or disabled.

A generic recovery configuration field would affect more engines and callers. Repeating
the preflight check would retain the gap between checking and loading. The recipe check
directly observes the loaded state using the launcher's existing environment contract.

The check applies to training. The `swe-eval` profile deliberately disables recovery and
runs with zero training steps, so an inherited recovery requirement does not block that
evaluation path.

CPU tests exercise the real recipe entry with stubbed service boundaries, covering fresh
training, missing required recovery, loaded recovery, cleanup and factory restoration.
They do not prove distributed checkpoint payload loading. A live recovery still requires
evidence of model, optimizer, LR and RNG restoration on all ranks, followed by a
completed optimizer update and a new full checkpoint.
