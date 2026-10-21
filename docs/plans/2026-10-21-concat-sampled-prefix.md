# Preserve sampled prefixes in concat conversations

Concat children inherit their parent's action logprobs, masks, weight versions, turn
IDs, rewards and routing data. Their prompt must therefore preserve every sampled parent
token at the same position. Stripping trailing PAD or repeated EOS tokens and inserting
one EOS violates this requirement.

Preserve the full sampled prefix and add a masked EOS only when needed to close an
assistant turn. Locate newly added messages by rendering the canonical parent template
with the same materialized tools as the child, rather than counting sampled EOS tokens.
Reject changed or shortened prefixes before inheriting training metadata. Allow
equal-length continuations to retain the parent's metadata and token rewards.

Keep this correction independent of image preprocessing and SGLang changes. Regression
tests cover normal EOS, PAD, repeated stop tokens, length and abort termination,
successive turns, tool iterators, invalid prefixes and equal-length continuations. Run
existing multimodal and routing tests to verify compatibility with upstream main.

The fix addresses reproducible boundary cases. It does not establish the cause of the
observed reward collapse; the inspected trajectory boundaries did not demonstrate these
triggers.
