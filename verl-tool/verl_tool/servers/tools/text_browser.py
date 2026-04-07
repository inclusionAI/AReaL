import ray
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from .base import BaseTool, register_tool, registered_tools
from mini_webarena.env_worker import WikiQAEnv

@ray.remote
class WikiEnvActor:
    def __init__(self, question: str, gt: str, url: str = None):
        self.env = WikiQAEnv(question, gt, url=url, prompt_format="last")

    def start_env(self) -> str:
        obs = self.env.render()
        return obs

    def step_env(self, query: str) -> (str, int):
        obs, done, valid = self.env.step(query)
        if done:
            self.env.close()
        return obs, done, valid


@register_tool
class TextBrowserTool(BaseTool):
    """
    TextBrowserTool uses Ray actors to manage WikiQAEnv sessions.
    Each trajectory_id has a dedicated actor. It supports initial
    render (action=None) and step operations.
    """
    tool_type = "text_browser"

    def __init__(self, num_workers=32):
        super().__init__(num_workers)
        # Maps trajectory_id to Ray Actor
        self.env_actors = {}
        # Track creation order for cleanup
        self.actor_creation_order = []

    # -------------------------------------------------------------------------
    # BaseTool interface methods (some are no-ops here, but we must implement them)
    # -------------------------------------------------------------------------
    def get_usage_inst(self) -> str:
        """Return usage instructions."""
        return "TextBrowserTool uses Ray actors to manage WikiQAEnv sessions."

    def has_env(self, trajectory_id):
        return trajectory_id in self.env_actors

    def load_env(self, trajectory_id: str):
        """Return a live actor or `None` if the trajectory is unknown."""
        return self.env_actors.get(trajectory_id)

    def save_env(self, trajectory_id: str, actor):
        """Register / refresh an actor and update LRU ordering."""
        # Should not exist if exist;
        if self.env_actors.get(trajectory_id) is None:
            self.env_actors[trajectory_id] = actor
            self.actor_creation_order.append(trajectory_id)
            self._cleanup_actors_if_needed()
        else:
            # If it exists, check if it's the same actor, otherwise raise an error
            if self.env_actors[trajectory_id] != actor:
                raise RuntimeError(f"Actor with trajectory_id {trajectory_id} already exists.")
            if trajectory_id in self.actor_creation_order:
                self.actor_creation_order.remove(trajectory_id)
            self.actor_creation_order.append(trajectory_id)

    def delete_env(self, trajectory_id):
        """Kill and remove the actor."""
        if trajectory_id in self.env_actors:
            ray.kill(self.env_actors[trajectory_id], no_restart=True)
            del self.env_actors[trajectory_id]
        if trajectory_id in self.actor_creation_order:
            self.actor_creation_order.remove(trajectory_id)

    def parse_action(self, action):
        # """Parse action (here we return it as-is)."""
        # return action, True
        # if action is contain a substring like <think>balabala</think>```balabala```, return true, otherwise false
        """
        Check if the action contains a pattern like <think>...</think>```...```.
        Return (action, True) if found, otherwise (action, False).
        """
        if action  == "" or action is None: # Tentitively allow empty action, since first obs is needed
            return action, True
        pattern = r"<think>.*?</think>\s```.*?```"
        matched = re.search(pattern, action, re.DOTALL)
        # print("[INFO] action:", action)
        # print("[INFO] matched:", matched)
        return action, bool(matched)


    def conduct_action(self, trajectory_id: str, action: str, extra_field: dict):
        """
        Execute a *single* action on the environment for `trajectory_id`.

        Returns
        -------
        obs : str
            Environment observation (empty string if episode finished).
        done : bool
            Whether the episode ended with this step.
        valid : bool
            Whether the action itself was valid.
        """
        # 1) Ensure an actor exists (lazy creation).
        actor = self.load_env(trajectory_id)
        if actor is None:
            # Create a brand-new WikiEnvActor for this trajectory.
            question = extra_field.get("question", "placeholder")
            gt       = extra_field.get("gt",        "placeholder")
            url      = extra_field.get("url",       None)
            actor = WikiEnvActor.remote(question, gt, url)
            self.save_env(trajectory_id, actor)

        # 2) Decide whether we are rendering the first page or taking a step.
        fut = (
            actor.start_env.remote()
            if action is None or action == ""
            else actor.step_env.remote(action)
        )

        # 3) Wait for the Ray RPC to finish (blocks the calling thread only).
        result = ray.get(fut)
        if isinstance(result, tuple):           # step_env
            obs, done, valid = result
        else:                                   # start_env
            obs, done, valid = result, False, False

        # 4) Refresh LRU order *after* the step.
        if trajectory_id in self.actor_creation_order:
            self.actor_creation_order.remove(trajectory_id)
        self.actor_creation_order.append(trajectory_id)

        # 5) Clean-up if the episode finished.
        if done:
            obs = ""            # Clear the final observation
            self.delete_env(trajectory_id)

        return obs, done, valid

    def get_observations(self, trajectory_ids, actions, extra_fields):
        """
        Batched version of `conduct_action` with thread-pool parallelism.
        (A process-pool is **not** required; Ray already runs the envs
        out-of-process.)

        Parameters
        ----------
        trajectory_ids : list[str]
        actions        : list[str | None]
        extra_fields   : list[dict]

        Returns
        -------
        observations : list[str]
        dones        : list[bool]
        valid_flags  : list[bool]
        """
        # print("[INFO] Using thread pool for parallel processing...")
        # print("[INFO] trajectory_ids:", trajectory_ids)
        # print("[INFO] actions:", actions)
        # print("[INFO] extra_fields:", extra_fields)

        import json
        from pathlib import Path
        from concurrent.futures import ThreadPoolExecutor

        n = len(trajectory_ids)
        observations = [""]   * n
        dones        = [False] * n
        valid_flags  = [True]  * n

        # ----------------------------------------------------------------- #
        # Parallel fan-out using a thread pool                              #
        # ----------------------------------------------------------------- #
        def _worker(idx: int):
            tid   = trajectory_ids[idx]
            act   = actions[idx]
            extra = extra_fields[idx].get("extra_fields", extra_fields[idx])
            try:
                return (*self.conduct_action(tid, act, extra), None)
            except Exception as e:
                return ("", False, False, e)   # bubble error to main thread

        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = [pool.submit(_worker, i) for i in range(n)]
            for i, fut in enumerate(futures):
                obs, done, valid, err = fut.result()
                observations[i] = obs
                dones[i]        = done
                valid_flags[i]  = valid
                if err:
                    print(f"[ERROR] trajectory_id={trajectory_ids[i]}: {err}")

        # ----------------------------------------------------------------- #
        # Fire-and-forget JSONL logging                                     #
        # ----------------------------------------------------------------- #
        try:
            log_path = Path("browser_server_logs.jsonl")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "input": {
                                "trajectory_ids": trajectory_ids,
                                "actions": actions,
                                "extra_fields": extra_fields,
                            },
                            "output": {
                                "observations": observations,
                                "dones": dones,
                                "valid_flags": valid_flags,
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception as e:
            # Logging failures must *never* break main logic
            print(f"[WARN] Failed to write browser_server_logs.jsonl: {e}")

        return observations, dones, valid_flags

    # # -------------------------------------------------------------------------
    # # Core logic that uses Ray actors
    # # -------------------------------------------------------------------------
    # def get_observations(self, trajectory_ids, actions, extra_fields):
    #     import json
    #     from pathlib import Path
    #
    #     futures = []
    #
    #     # ---------------------------------------------------------------------
    #     # 1. Dispatch Ray RPCs -------------------------------------------------
    #     # ---------------------------------------------------------------------
    #     for i, trajectory_id in enumerate(trajectory_ids):
    #         action = actions[i]
    #         extra = extra_fields[i].get("extra_fields", extra_fields[i])
    #
    #         # Lazily create an actor for every new trajectory
    #         if trajectory_id not in self.env_actors:
    #             question = extra.get("question", "placeholder")
    #             gt = extra.get("gt", "placeholder")
    #             url = extra.get("url", None)
    #             actor = WikiEnvActor.remote(question, gt, url)
    #             self.env_actors[trajectory_id] = actor
    #             self.actor_creation_order.append(trajectory_id)
    #
    #         actor = self.env_actors[trajectory_id]
    #
    #         # Decide whether to render or step
    #         fut = actor.start_env.remote() if action is None or action == "" else actor.step_env.remote(action)
    #         futures.append((i, trajectory_id, fut))
    #
    #         self._cleanup_actors_if_needed()
    #
    #     # ---------------------------------------------------------------------
    #     # 2. Gather results ----------------------------------------------------
    #     # ---------------------------------------------------------------------
    #     observations = [""] * len(trajectory_ids)
    #     dones = [False] * len(trajectory_ids)
    #     valid_flags = [True] * len(trajectory_ids)
    #
    #     for i, trajectory_id, fut in futures:
    #         try:
    #             result = ray.get(fut)
    #             if isinstance(result, tuple):           # step_env
    #                 obs, done, valid = result
    #             else:                                   # start_env
    #                 obs, done, valid = result, False, False
    #
    #             observations[i] = obs
    #             dones[i] = done
    #             valid_flags[i] = valid
    #
    #             # refresh LRU list
    #             if trajectory_id in self.actor_creation_order:
    #                 self.actor_creation_order.remove(trajectory_id)
    #             self.actor_creation_order.append(trajectory_id)
    #
    #             if done:
    #                 observations[i] = ""               # clear final obs
    #                 self.delete_env(trajectory_id)
    #
    #         except Exception as e:
    #             print(f"Error while processing trajectory_id={trajectory_id}: {e}")
    #             observations[i] = ""
    #             dones[i] = False
    #             valid_flags[i] = False
    #
    #     # ---------------------------------------------------------------------
    #     # 3. Persist one‑line JSON log -----------------------------------------
    #     # ---------------------------------------------------------------------
    #     try:
    #         log_path = Path("browser_server_logs.jsonl")
    #         log_path.parent.mkdir(parents=True, exist_ok=True)
    #         with log_path.open("a", encoding="utf-8") as f:
    #             f.write(
    #                 json.dumps(
    #                     {
    #                         "input": {
    #                             "trajectory_ids": trajectory_ids,
    #                             "actions": actions,
    #                             "extra_fields": extra_fields,
    #                         },
    #                         "output": {
    #                             "observations": observations,
    #                             "dones": dones,
    #                             "valid_flags": valid_flags,
    #                         },
    #                     },
    #                     ensure_ascii=False,
    #                 )
    #                 + "\n"
    #             )
    #     except Exception as e:
    #         # logging failure should never break main logic
    #         print(f"[WARN] Failed to write browser_server_logs.jsonl: {e}")
    #
    #     # ---------------------------------------------------------------------
    #     # 4. Return to caller --------------------------------------------------
    #     # ---------------------------------------------------------------------
    #     return observations, dones, valid_flags

    # -------------------------------------------------------------------------
    # Helper method
    # -------------------------------------------------------------------------
    def _cleanup_actors_if_needed(self):
        """Remove oldest actors if count exceeds limit."""
        while len(self.env_actors) > 128:
            raise RuntimeError("Too many actors, please reduce the number of concurrent requests.")
            oldest = self.actor_creation_order.pop(0)
            self.delete_env(oldest)