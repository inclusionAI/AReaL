from collections import OrderedDict

from openai.types.responses.response_input_param import ResponseInputParam

from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.utils import logging

logger = logging.getLogger("AReaLOpenAI Interaction Cache")


class InteractionCache(OrderedDict[str, InteractionWithTokenLogpReward]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._apply_reward_discount_called = False
        self._parent_relationship_built = False

    def set_reward(self, id: str, reward: float) -> None:
        """Set reward for a specific completion/response by its ID."""
        self[id].reward = reward

    def set_final_reward(self, reward: float) -> None:
        """Set reward for the most recent completion/response."""
        last_interaction_id = next(reversed(self))
        self[last_interaction_id].reward = reward

    def apply_reward_discount(
        self, turn_discount: float = 1.0
    ) -> dict[str, InteractionWithTokenLogpReward]:
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
        assert not self._apply_reward_discount_called, (
            "apply_reward_discount should only be called once."
        )
        self._apply_reward_discount_called = True
        reversed_interactions = list(reversed(self.values()))

        if reversed_interactions:
            current_reward = 0.0
            for i, interaction in enumerate(reversed_interactions):
                if interaction.reward is None:
                    # If the last-created interaction has no reward set, log a warning
                    if i == 0:
                        logger.warning(
                            "The most recent interaction does not have a reward set. "
                            "All interactions will have None reward."
                        )
                    interaction.reward = 0.0

                current_reward = current_reward * turn_discount + interaction.reward
                interaction.reward = current_reward
        return dict(**self)

    def _build_parent_child_relationships(self) -> None:
        self._parent_relationship_built = True
        if len(self) == 0:
            return

        def _is_prefix(
            a: list[dict] | str | ResponseInputParam,
            b: list[dict] | str | ResponseInputParam,
        ) -> bool:
            # True if a is a strict prefix of b
            if len(a) >= len(b):
                return False
            return b[: len(a)] == a

        # 1) Construct parent-child relationships using longest prefix rule
        # Sort potential children by (data length asc, created asc)
        # so parents are available
        interactions = sorted(
            self.values(), key=lambda x: (len(x.current_data), x.created_at)
        )

        # Reset parents before rebuilding
        for interaction in interactions:
            interaction.parent = None

        for i, child in enumerate(interactions):
            # find the longest prefix in reverse order
            for j in range(i):
                parent = interactions[i - 1 - j]
                if _is_prefix(parent.current_data, child.current_data):
                    child.parent = parent
                    break

    def export_interactions(
        self, style: str, reward_discount: float | None = None
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
        if reward_discount is not None and not self._apply_reward_discount_called:
            self.apply_reward_discount(turn_discount=reward_discount)

        cache = self
        if len(cache) == 0:
            return {}

        for id, interaction in self.items():
            if interaction.interaction_id != id:
                raise ValueError(
                    f"Interaction ID mismatch: {interaction.interaction_id} != {id}"
                )

        if style == "concat":
            for interaction in self.values():
                if interaction.chat_template_type != "concat":
                    raise ValueError(
                        "Cannot export interactions in 'concat' style when "
                        "interaction.chat_template_type != 'concat' for any interaction. "
                        "This is because when applying chat template using some "
                        "tokenizers, there might be some tokens added or removed "
                        "(e.g. think tokens), making it impossible to construct the conversation tree. "
                        "Please use 'individual' style instead."
                    )

            if not self._parent_relationship_built:
                self._build_parent_child_relationships()

            # Build children mapping to find leaf nodes.
            has_children = set()
            for obj in self.values():
                if obj.parent is not None:
                    has_children.add(obj.parent.interaction_id)

            # Return only leaf nodes (nodes without children)
            return {
                id: interaction
                for id, interaction in self.items()
                if id not in has_children
            }
        elif style == "individual":
            return dict(**cache)
        else:
            raise ValueError(f"Invalid export interactions style {style}")
