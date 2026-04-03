"""Wrapper entry point for SGLang launch_server with PP bug fix.

Sets up environment for cross-process patch injection, then delegates
to SGLang's original entry point.
"""

import os
import sys
import runpy


def main():
    # Step 1: Set environment variable to activate the fix in all processes
    os.environ["AREAL_SGLANG_PP_FIX"] = "1"

    # Step 2: Prepend _site_hook/ directory to PYTHONPATH
    # This ensures our sitecustomize.py runs in every child process
    # spawned by multiprocessing (even with start_method="spawn")
    hook_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_site_hook"
    )
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    if existing_pythonpath:
        os.environ["PYTHONPATH"] = hook_dir + os.pathsep + existing_pythonpath
    else:
        os.environ["PYTHONPATH"] = hook_dir

    # Step 3: Apply the patch in the CURRENT process immediately
    from areal.patches.sglang_pp_fix import apply_sglang_pp_fix
    apply_sglang_pp_fix()

    # Step 4: Delegate to SGLang's original launch_server
    runpy.run_module("sglang.launch_server", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
