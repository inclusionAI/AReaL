# SPDX-License-Identifier: Apache-2.0

"""GSM8K sandbox RL training using AgentWorkflow with E2B sandbox.

This is a convenience alias for ``train_sandbox_agent.py``.

Usage::

    python examples/sandbox/gsm8k_sandbox_rl.py \\
        --config examples/sandbox/gsm8k_agent_sandbox.yaml
"""

from train_sandbox_agent import main  # isort: skip
import sys

if __name__ == "__main__":
    main(sys.argv[1:])
