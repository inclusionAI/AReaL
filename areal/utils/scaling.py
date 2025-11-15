import ast
import time

from areal.utils import logging, name_resolve
from areal.utils.name_resolve import NameEntryNotFoundError

logger = logging.getLogger("Scaler")


def handle_scale_up(name_resolve: name_resolve, actor, rollout, weight_update_meta):
    """
    Handle scale-up logic when scale_up_request is detected.
    Requires: name_resolve, actor, rollout.
    """
    new_scale = 0
    req_raw = None
    try:
        req_raw = name_resolve.get("scale_up_request")
        new_scale = ast.literal_eval(req_raw)["scaled_k"]
    except NameEntryNotFoundError:
        logger.info("scale_up_request not found")
        pass  # no request → don't wait

    logger.info(f"scale_up_request {req_raw}")

    if req_raw:
        # Now wait until scale_up_done is posted from scaler process
        start = time.time()
        try:
            name_resolve.delete("scale_up_request")
        except NameEntryNotFoundError:
            pass

        while True:
            try:
                done_raw = name_resolve.get("scale_up_done")
            except NameEntryNotFoundError:
                done_raw = None

            if done_raw:
                logger.info(f"[areal] Scale-up finished: {done_raw}")
                name_resolve.add(
                    "scale_up_time",
                    {"time": time.time() - start},
                    replace=True,
                )

                try:
                    name_resolve.delete("scale_up_request")
                except NameEntryNotFoundError:
                    pass
                try:
                    name_resolve.delete("scale_up_done")
                except NameEntryNotFoundError:
                    pass
                # Increase teh number of scale in rollout engine and actor. To get correct world size
                actor.scaling_count = actor.scaling_count + new_scale
                rollout._engine.backend.scaling_count = (
                    rollout._engine.backend.scaling_count + new_scale
                )
                rollout._engine.distributed_weight_update_initialized = False
                actor._re_init_weight_update_from_distributed(weight_update_meta)

                break
            time.sleep(0.5)
