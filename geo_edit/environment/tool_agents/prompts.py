"""Default prompt templates for Tool Agents."""

# Default system prompt for tool agents
DEFAULT_SYSTEM_PROMPT = (
    "You are a tool agent. Analyze the image and answer the question. "
    "Return JSON with at least one field in {analysis, text, result, error}."
)