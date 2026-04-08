import os
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

    # Fix NCCL rank collision in weight-update group when PP > 1.
    # Without this patch, pp_rank is ignored in the rank computation,
    # causing multiple workers to join the NCCL group with the same rank
    # and hang indefinitely (manifests as 504 Gateway Timeout).
    from areal.patches.patch_sglang_pp_rank_fix import apply_sglang_pp_rank_fix

    apply_sglang_pp_rank_fix()

    runpy.run_module("sglang.launch_server", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
