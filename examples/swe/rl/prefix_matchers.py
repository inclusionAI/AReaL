"""Custom prefix matchers for InteractionCache parent-child matching.

A prefix matcher determines whether message list *a* is a "prefix" of
message list *b*.  The default implementation uses exact element-wise
equality.  The matchers here implement relaxed comparison strategies
tailored to specific agent behaviours.

Usage::

    # In config YAML or observe_cc.py
    prefix_matcher: "examples.swe.rl.prefix_matchers.swe_prefix_matcher"
"""

from __future__ import annotations


def _tool_call_ids(tool_calls: list[dict]) -> list[str]:
    """Extract ordered tool_call IDs from a tool_calls list."""
    return [tc.get("id", "") for tc in tool_calls if isinstance(tc, dict)]


def _messages_match(a: dict, b: dict) -> bool:
    """Check whether two message dicts are semantically equivalent.

    Matching rules by message role:

    * **system / user / assistant without tool_calls**: ``content`` and
      ``thinking`` (if present) must be equal.
    * **assistant with tool_calls**: ``content``, ``thinking`` (if present),
      and the ordered list of ``tool_calls[].id`` must be equal.
      ``tool_calls[].function.arguments`` is **ignored** because the agent
      CLI may rewrite them between turns (``cd`` prefix stripping, trailing
      whitespace removal, etc.).
    * **tool** (tool response): ``role`` and ``tool_call_id`` must be equal.
      ``content`` (the tool execution output) is **ignored** — the same
      ``tool_call_id`` guarantees it is the same invocation.
    """
    if a.get("role") != b.get("role"):
        return False

    role = a.get("role")

    if role == "tool":
        # Tool response: match on tool_call_id only.
        return a.get("tool_call_id") == b.get("tool_call_id")

    # For system / user / assistant: content must match.
    # Default to "" so that absent ``content`` key equals explicit ``""``.
    if a.get("content", "") != b.get("content", ""):
        return False

    # Thinking blocks must match if present on either side.
    if a.get("thinking") != b.get("thinking"):
        return False

    # For assistant with tool_calls: compare tool_call IDs only.
    a_tc = a.get("tool_calls")
    b_tc = b.get("tool_calls")
    if a_tc is not None or b_tc is not None:
        if a_tc is None or b_tc is None:
            # One has tool_calls, the other doesn't.
            return False
        if not isinstance(a_tc, list) or not isinstance(b_tc, list):
            return a_tc == b_tc
        if len(a_tc) != len(b_tc):
            return False
        if _tool_call_ids(a_tc) != _tool_call_ids(b_tc):
            return False

    return True


def swe_prefix_matcher(a: list[dict], b: list[dict]) -> bool:
    """Relaxed prefix matcher for SWE-bench agent trajectories.

    Returns ``True`` if message list *a* is a semantic prefix of *b*,
    tolerating known agent-side rewrites in tool_call arguments and tool
    response content.

    See :func:`_messages_match` for per-message comparison rules.
    """
    if len(a) > len(b):
        return False
    for am, bm in zip(a, b):
        if am == bm:
            # Fast path: exact match.
            continue
        if not _messages_match(am, bm):
            return False
    return True
