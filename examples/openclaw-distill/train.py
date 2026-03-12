"""Self-distillation training: external users interact via proxy gateway URL.

Unlike the standard OpenClaw online RL example, this uses the self-distilled
policy optimization framework.  User code calls
``/chat/completions`` with the admin API key, and completions are
buffered for self-distillation training — no session lifecycle
(start/end session, set reward) is required.
"""

import sys

from areal import SelfDistillationTrainer
from areal.api.cli_args import SelfDistillConfig, load_expr_config


def main(args):
    config, _ = load_expr_config(args, SelfDistillConfig)

    with SelfDistillationTrainer(config) as trainer:
        trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
