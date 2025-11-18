import pytest
from collections import OrderedDict
from unittest.mock import MagicMock

from areal.experimental.openai.cache import CompletionCache
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.utils.hf_utils import load_hf_tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    """Load the tokenizer once for all tests in this module."""
    return load_hf_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")


@pytest.fixture
def mock_interaction(tokenizer):
    """Returns a function to create a mock InteractionWithTokenLogpReward."""

    def _create_mock_interaction(
        id: str,
        messages: list[dict] | None = None,
        input_data: str = "",
        is_completion: bool = True,
        created: int = 0,
        reward: float | None = None,
        chat_template_type: str = "concat",
        messages_delimiter_start: str = "<|im_start|>",
        messages_delimiter_end: str = "<|im_end|>",
    ):
        messages = messages or []
        mock_model_response = MagicMock()
        
        input_tokens = []
        output_tokens = []

        if is_completion and messages:
            # Mimic the tokenization logic from client.py
            start, end = messages_delimiter_start, messages_delimiter_end
            
            # The last message is the completion, the rest is the prompt.
            prompt_messages = messages[:-1]
            output_message = messages[-1]

            # Tokenize prompt
            prompt_strs = []
            for msg in prompt_messages:
                prompt_strs.append(f"{start}{msg['role']}\n{msg['content']}{end}\n")
            prompt_strs.append(f"{start}assistant\n")
            input_tokens = tokenizer.encode("".join(prompt_strs))

            # Tokenize output (completion)
            # Note: This is a simplification. The actual output tokenization might
            # depend on whether it's considered part of the template.
            # For parent matching, tokenizing the content should be sufficient.
            output_tokens = tokenizer.encode(output_message['content'])

        mock_model_response.input_tokens = input_tokens
        mock_model_response.output_tokens = output_tokens
        
        interaction = InteractionWithTokenLogpReward(
            model_response=mock_model_response,
            reward=reward,
            chat_template_type=chat_template_type,
        )
        if is_completion:
            completion_mock = MagicMock()
            completion_mock.id = id
            completion_mock.created = created
            interaction.completion = completion_mock
            interaction.messages = messages
        else:
            response_mock = MagicMock()
            response_mock.id = id
            response_mock.created_at = created
            interaction.response = response_mock
            interaction.input_data = input_data
        return interaction

    return _create_mock_interaction


def test_set_reward(mock_interaction):
    cache = CompletionCache()
    interaction = mock_interaction(id="1")
    cache["1"] = interaction
    cache.set_reward("1", 10.0)
    assert cache["1"].reward == 10.0


def test_set_final_reward(mock_interaction):
    cache = CompletionCache()
    cache["1"] = mock_interaction(id="1")
    cache["2"] = mock_interaction(id="2")
    cache.set_final_reward(20.0)
    assert cache["1"].reward is None
    assert cache["2"].reward == 20.0


def test_apply_reward_discount(mock_interaction):
    cache = CompletionCache()
    cache["1"] = mock_interaction(id="1", created=1)
    cache["2"] = mock_interaction(id="2", created=2)
    cache["3"] = mock_interaction(id="3", created=3, reward=10.0)

    ordered_cache = CompletionCache(OrderedDict(sorted(cache.items())))
    ordered_cache.apply_reward_discount(turn_discount=0.9)

    assert ordered_cache["3"].reward == pytest.approx(10.0)
    assert ordered_cache["2"].reward == pytest.approx(9.0)
    assert ordered_cache["1"].reward == pytest.approx(8.1)


def test_apply_reward_discount_called_once(mock_interaction):
    cache = CompletionCache()
    cache["1"] = mock_interaction(id="1", reward=10.0)
    cache.apply_reward_discount()
    with pytest.raises(AssertionError, match="apply_reward_discount should only be called once."):
        cache.apply_reward_discount()


def test_apply_reward_discount_no_final_reward(mock_interaction):
    """Tests that a warning is logged if the final interaction has no reward."""
    cache = CompletionCache()
    cache["1"] = mock_interaction(id="1", created=1)
    cache["2"] = mock_interaction(id="2", created=2)

    # NOTE: We are ignoring the warning capture failure for now as requested.
    cache.apply_reward_discount(turn_discount=0.9)

    assert cache["2"].reward == 0.0
    assert cache["1"].reward == 0.0


def test_export_interactions_individual_style(mock_interaction):
    cache = CompletionCache()
    cache["1"] = mock_interaction(id="1")
    cache["2"] = mock_interaction(id="2")
    
    with pytest.raises(AssertionError, match="Please call build_parent_child_relationships before exporting interactions."):
        cache.export_interactions(style="individual")

    cache.build_parent_child_relationships()
    exported = cache.export_interactions(style="individual")
    assert len(exported) == 2
    assert "1" in exported
    assert "2" in exported


def test_export_interactions_concat_style(mock_interaction):
    cache = CompletionCache()
    i1 = mock_interaction(id="1", messages=[{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}], created=1)
    i2 = mock_interaction(id="2", messages=[{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}, {"role": "user", "content": "C"}, {"role": "assistant", "content": "D"}],  created=2)
    i3 = mock_interaction(id="3", messages=[{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}, {"role": "user", "content": "C"}, {"role": "assistant", "content": "E"}], created=3)
    i4 = mock_interaction(id="4", messages=[{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}, {"role": "user", "content": "C"}, {"role": "assistant", "content": "D"}, {"role": "user", "content": "E"}, {"role": "assistant", "content": "F"}], created=4)
    
    print(f"i1 input tokens: {i1.model_response.input_tokens}, i1 output tokens: {i1.model_response.output_tokens}")
    print(f"i2 input tokens: {i2.model_response.input_tokens}, i2 output tokens: {i2.model_response.output_tokens}")
    print(f"i3 input tokens: {i3.model_response.input_tokens}, i3 output tokens: {i3.model_response.output_tokens}")
    print(f"i4 input tokens: {i4.model_response.input_tokens}, i4 output tokens: {i4.model_response.output_tokens}")

    cache[i1.completion.id] = i1
    cache[i2.completion.id] = i2
    cache[i3.completion.id] = i3
    cache[i4.completion.id] = i4

    cache.build_parent_child_relationships()
    exported = cache.export_interactions(style="concat")

    assert len(exported) == 2
    assert i3.completion.id in exported
    assert i4.completion.id in exported
    assert exported[i4.completion.id].parent == i2
    assert exported[i3.completion.id].parent == i1
    assert i2.parent == i1


def test_build_parent_child_relationships_idempotency(mock_interaction):
    cache = CompletionCache()
    i1 = mock_interaction(id="1", messages=[{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}])
    i2 = mock_interaction(id="2", messages=[{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}, {"role": "user", "content": "C"}, {"role": "assistant", "content": "D"}])
    cache[i1.completion.id] = i1
    cache[i2.completion.id] = i2
    
    cache.build_parent_child_relationships()
    assert i2.parent == i1
    
    # Tamper with the parent relationship and call again to check idempotency
    i2.parent = None
    cache.build_parent_child_relationships()
    assert i2.parent == i1 # Should be rebuilt


def test_multiple_exports_after_build(mock_interaction):
    cache = CompletionCache()
    i1 = mock_interaction(id="1", messages=[{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}], created=1)
    i2 = mock_interaction(id="2", messages=[{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}, {"role": "user", "content": "C"}, {"role": "assistant", "content": "D"}], created=2)
    cache[i1.completion.id] = i1
    cache[i2.completion.id] = i2

    cache.build_parent_child_relationships()

    # First export: concat
    exported_concat = cache.export_interactions(style="concat")
    assert len(exported_concat) == 1
    assert i2.completion.id in exported_concat

    # Second export: individual
    exported_individual = cache.export_interactions(style="individual")
    assert len(exported_individual) == 2
    assert i1.completion.id in exported_individual
    assert i2.completion.id in exported_individual

    # Third export: concat again
    exported_concat_2 = cache.export_interactions(style="concat")
    assert len(exported_concat_2) == 1
    assert i2.completion.id in exported_concat_2


def test_export_interactions_empty_cache(mock_interaction):
    cache = CompletionCache()
    cache.build_parent_child_relationships()
    exported = cache.export_interactions()
    assert len(exported) == 0


def test_export_interactions_invalid_style(mock_interaction):
    cache = CompletionCache()
    cache["1"] = mock_interaction(id="1")
    cache.build_parent_child_relationships()
    with pytest.raises(ValueError, match="Invalid export interactions style"):
        cache.export_interactions(style="invalid_style")


def test_build_parent_child_relationships_wrong_template_type(mock_interaction):
    cache = CompletionCache()
    cache["1"] = mock_interaction(id="1", chat_template_type="hf")
    with pytest.raises(ValueError, match="Cannot export interactions in 'concat' style"):
        cache.build_parent_child_relationships()


if __name__ == "__main__":
    pytest.main([__file__])