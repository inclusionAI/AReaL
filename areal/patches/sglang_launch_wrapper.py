import os
import sys
import runpy

def main():
    os.environ["AREAL_SGLANG_PP_FIX"] = "1"
    hook_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_site_hook")
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    if existing_pythonpath:
        os.environ["PYTHONPATH"] = hook_dir + os.pathsep + existing_pythonpath
    else:
        os.environ["PYTHONPATH"] = hook_dir

    from areal.patches.sglang_pp_fix import apply_sglang_pp_fix
    apply_sglang_pp_fix()

    from areal.patches.sglang_pp_vocab_fix import apply_sglang_pp_vocab_fix
    apply_sglang_pp_vocab_fix()

    runpy.run_module("sglang.launch_server", run_name="__main__", alter_sys=True)

if __name__ == "__main__":
    main()