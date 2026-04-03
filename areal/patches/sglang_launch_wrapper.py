"""Wrapper entry point for SGLang launch_server with PP bug fix.

Drop-in replacement for `python3 -m sglang.launch_server`.
Applies monkey-patch first, then delegates to SGLang's original entry point.

Usage:
    python3 -m areal.patches.sglang_launch_wrapper --tp-size 2 --pp-size 2 ...
"""

import runpy


def main():
    # Step 1: Apply the PP monkey-patch BEFORE any SGLang model init
    from areal.patches.sglang_pp_fix import apply_sglang_pp_fix
    apply_sglang_pp_fix()

    # Step 2: Delegate to SGLang's original launch_server via runpy
    # This executes the module's if __name__ == "__main__" block,
    # preserving sys.argv for argparse, exactly as if the user ran:
    #   python3 -m sglang.launch_server ...
    runpy.run_module("sglang.launch_server", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
