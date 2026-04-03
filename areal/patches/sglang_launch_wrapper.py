"""Wrapper entry point for SGLang launch_server with PP bug fix.

Drop-in replacement for `python3 -m sglang.launch_server`.
Applies monkey-patch first, then delegates to SGLang's original entry point.

Usage:
    python3 -m areal.patches.sglang_launch_wrapper --tp-size 2 --pp-size 2 ...
"""

import sys
import os


def main():
    # Step 1: Apply the PP monkey-patch BEFORE any SGLang model init
    from areal.patches.sglang_pp_fix import apply_sglang_pp_fix
    apply_sglang_pp_fix()

    # Step 2: Delegate to SGLang's original launch_server
    from sglang.launch_server import main as sglang_main
    sglang_main()


if __name__ == "__main__":
    main()
