# Notes

## Ollama

Very easy to use.
`ollama serve` start server
`ollama pull <model>` download model
`ollama list` check what models are available
`ollama ps` check what models are running

Default port is 11434.
Base URL: `http://localhost:11434/v1`


No clear option to kill the process once it's running.
So just do ...
```bash
ps aux | grep ollama
kill -9 <pid>
```

## LiteLLM

Support for tool calls is somewhat confusing.
Lots of things can be hidden behind the scenes.
Always good to monitor what is actually sent to the model (see below for an example of an issue)

## LiteLLM & Ollama

Needs to use `ollama_chat` prefix to use tool calls.
If using `ollama` prefix, it will inject the tools as a system message.
- The prompt litellm use will force tool call, even if in "auto" mode.

```python
# litellm/litellm_core_utils/prompt_templates/factory.py
# Function call template
def function_call_prompt(messages: list, functions: list):
    function_prompt = """Produce JSON OUTPUT ONLY! Adhere to this format {"name": "function_name", "arguments":{"argument_name": "argument_value"}} The following functions are available to you:"""
    for function in functions:
        function_prompt += f"""\n{function}\n"""

    function_added_to_prompt = False
    for message in messages:
        if "system" in message["role"]:
            message["content"] += f""" {function_prompt}"""
            function_added_to_prompt = True

    if function_added_to_prompt is False:
        messages.append({"role": "system", "content": f"""{function_prompt}"""})

    return messages
```

## OpenAI client
Client doesn't seem to have prompt wrappers to custom handle tool calls.
It just passes the tools def as they are provided.

## Langfuse
When settin up langfuse hooks for litellm, and choosing "ollama" prefix. The system message that is added by litellm is not recorded in langfuse.
We can see it by setting `litellm._turn_on_debug()` and looking at debut messages.
Why is it not recorded?


## vLLM
Some difficulies setting it up.
Needs rust compiler installed.
LLMs docs usually recommand using vLLM for serving.

Serving model
```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --host localhost --port 8000 --trust-remote-code
``` 

Test it with curl:
```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

## vLLM & LiteLLM

LiteLLM mentions that prefix is either:
"hosted_vllm" or "vllm"
Not completely clear what the difference is.
- "vllm" is the one that works but I feel this is spinning up a vllm server at each call.
- "hosted_vllm" doesn't work.

Issues running vLLM with litellm:
- Using litellm:
    - Reloading the weights at each call.
    - Even curl call doesn't work at time.
    - Requires debugging....