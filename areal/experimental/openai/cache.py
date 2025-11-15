from collections import OrderedDict, defaultdict

from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.utils import logging

logger = logging.getLogger("AReaLOpenAI Client")


class CompletionCache(OrderedDict[str, InteractionWithTokenLogpReward]):
    def set_reward(self, id: str, reward: float) -> None:
        """Set reward for a specific completion/response by its ID."""
        self[id].reward = reward

    def set_final_reward(self, reward: float) -> None:
        """Set reward for the most recent completion/response."""
        last_interaction_id = next(reversed(self))
        self[last_interaction_id].reward = reward

    def apply_reward_discount(self, turn_discount: float = 1.0) -> None:
        """Apply backward discounted rewards across cached completions/responses.

        This method iterates over the cached completions/responses in reverse creation
        (insertion) order and applies a geometric discount to propagate reward
        signal backward in time. The most recent completion/response is treated as the
        starting point. If it does not have an explicit reward, a warning is
        logged and a default reward of ``0.0`` is used. For each earlier
        completion/response, its reward is initialized to ``0.0`` if unset, then the
        discounted reward from the next later completion/response is added:

        ``reward[i] += reward[i+1] * turn_discount``.

        Typically called before exporting completions/responses in 'individual' style
        to each completion/response is assigned with a valid reward value.

        Parameters
        ----------
        turn_discount : float, optional
            The per-turn discount factor applied when propagating reward
            backward from a later completion/response to an earlier one, by default 1.0.

        Returns
        -------
        Dict[str, InteractionWithTokenLogpReward]
            A shallow copy of the completion/response cache after rewards have been
            updated in-place.
        """
        # Assign rewards to interactions in cache based on their creation order
        interaction_time_sequence = list(
            reversed([interaction for _, interaction in self.items()])
        )

        # Check if the last-created interaction has a reward set
        if interaction_time_sequence:
            if interaction_time_sequence[0].reward is None:
                logger.warning(
                    "The most recent interaction does not have a reward set. "
                    "All interactions will have None reward."
                )
                interaction_time_sequence[0].reward = 0.0
            # Propagate rewards backwards with discounting if reward is not set
            for i in range(1, len(interaction_time_sequence)):
                if interaction_time_sequence[i].reward is None:
                    interaction_time_sequence[i].reward = 0.0
                interaction_time_sequence[i].reward += (
                    interaction_time_sequence[i - 1].reward * turn_discount
                )
        return dict(**self)

    def export_interactions(
        self, style: str = "concat"
    ) -> dict[str, InteractionWithTokenLogpReward]:
        """Export cached completions/responses in different formats.

        When ``style='concat'``, this method constructs a conversation tree by
        linking completions/responses whose input message lists form a strict-prefix
        relationship. The longest-prefix rule is used to determine each node's
        parent. It then returns only leaf-node completions/responses (those without
        children). No reward propagation is performed here.

        When ``style='individual'``, all cached completions/responses are returned as-is
        without constructing the tree.

        Parameters
        ----------
        style : str, optional
            The export style, either ``'concat'`` (build tree and return leaves)
            or ``'individual'`` (return all), by default 'concat'.

        Returns
        -------
        Dict[str, InteractionWithTokenLogpReward]
            A mapping from completion/response ID to completion/response objects. For
            ``'concat'``, this contains only leaf nodes. For ``'individual'``,
            this contains all cached completions/responses.

        Raises
        ------
        ValueError
            If an unsupported ``style`` is provided.
        """
        cache = self
        if len(cache) == 0:
            return {}

        if style == "concat":
            for interaction in cache.values():
                if interaction.chat_template_type != "concat":
                    raise ValueError(
                        "Cannot export interactions in 'concat' style when "
                        "interaction.chat_template_type != 'concat' for any interaction. "
                        "This is because when applying chat template using some "
                        "tokenizers, there might be some tokens added or removed "
                        "(e.g. think tokens), making it impossible to construct the conversation tree. "
                        "Please use 'individual' style instead."
                    )

            def _is_prefix(a: list[dict], b: list[dict]) -> bool:
                # True if a is a strict prefix of b
                if len(a) >= len(b):
                    return False
                for i in range(len(a)):
                    if a[i] != b[i]:
                        return False
                return True

            # Precompute normalized data
            meta = {}
            for interaction_id, interaction in cache.items():
                if interaction.is_completion:
                    norm_data = interaction.messages or []
                else:  # response
                    norm_data = interaction.input_data
                meta[interaction_id] = {
                    "norm_data": norm_data,
                    "obj": interaction,
                }

            # 1) Construct parent-child relationships using longest prefix rule
            # Sort potential children by (data length asc, created asc)
            # so parents are available
            ordered = sorted(
                meta.items(),
                key=lambda kv: (
                    len(kv[1]["norm_data"]),
                    (
                        kv[1]["obj"].completion.created
                        if kv[1]["obj"].is_completion
                        else kv[1]["obj"].response.created_at
                    ),
                ),
            )

            # Reset parents before rebuilding
            for _, info in ordered:
                info["obj"].parent = None

            for child_id, child_info in ordered:
                child_data = child_info["norm_data"]
                best_parent = None
                best_len = -1
                for parent_id, parent_info in ordered:
                    if parent_id == child_id:
                        continue
                    parent_data = parent_info["norm_data"]
                    if _is_prefix(parent_data, child_data):
                        plen = len(str(parent_data))
                        # choose the longest prefix
                        if plen > best_len:
                            best_parent = parent_info["obj"]
                            best_len = plen
                child_info["obj"].parent = best_parent

            # Build children mapping to find leaf nodes.
            children_map: dict[
                str,
                list[InteractionWithTokenLogpReward],
            ] = defaultdict(list)
            for _, info in meta.items():
                obj = info["obj"]
                if obj.parent is not None:
                    if obj.is_completion:
                        children_map[obj.parent.completion.id].append(obj)
                    else:  # response
                        children_map[obj.parent.response.id].append(obj)

            # Return only leaf nodes (nodes without children)
            parents_with_children = set(children_map.keys())
            leaf_only: dict[str, InteractionWithTokenLogpReward] = {}
            for interaction_id, info in meta.items():
                obj = info["obj"]
                obj_id = obj.completion.id if obj.is_completion else obj.response.id
                if obj_id not in parents_with_children:
                    leaf_only[interaction_id] = obj
            return leaf_only
        elif style == "individual":
            return dict(**cache)
        else:
            raise ValueError(f"Invalid export interactions style {style}")
