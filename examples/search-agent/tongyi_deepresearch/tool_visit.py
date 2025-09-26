import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Union

import aiohttp
import tiktoken
from openai import OpenAI
from qwen_agent.tools.base import BaseTool, register_tool

from areal.utils.http import get_default_connector

try:
    from .prompt import *
except ImportError:  # Fallback when executed directly (no package parent known)
    module_dir = Path(__file__).parent
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    from prompt import *


VISIT_SERVER_TIMEOUT = int(os.getenv("VISIT_SERVER_TIMEOUT", 200))
WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 150000))

JINA_API_KEYS = os.getenv("JINA_API_KEYS", "")


@staticmethod
def truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text

    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


OSS_JSON_FORMAT = """# Response Formats
## visit_content
{"properties":{"rational":{"type":"string","description":"Locate the **specific sections/data** directly related to the user's goal within the webpage content"},"evidence":{"type":"string","description":"Identify and extract the **most relevant information** from the content, never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.","summary":{"type":"string","description":"Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal."}}}}"""


@register_tool("visit", allow_overwrite=True)
class Visit(BaseTool):
    # The `description` tells the agent the functionality of this tool.
    name = "visit"
    description = "Visit webpage(s) and return the summary of the content."
    # The `parameters` tell the agent what input parameters the tool has.
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "minItems": 1,
                "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs.",
            },
            "goal": {
                "type": "string",
                "description": "The goal of the visit for webpage(s).",
            },
        },
        "required": ["url", "goal"],
    }

    # The `call` method is the main function of the tool.
    async def call(self, params: Union[str, dict], **kwargs) -> str:  # type: ignore[override]
        try:
            url = params["url"]
            goal = params["goal"]
        except Exception:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"

        start_time = time.time()
        log_folder = "log"
        os.makedirs(log_folder, exist_ok=True)

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=30,
                sock_connect=10,
                connect=10,
            ),
            read_bufsize=1024 * 1024 * 10,
            connector=get_default_connector(),
        ) as session:
            if isinstance(url, str):
                response = await self.readpage_jina(session, url, goal)
            else:
                assert isinstance(url, List)
                responses: List[str] = []
                for u in url:
                    if time.time() - start_time > 900:
                        cur_response = "The useful information in {url} for user goal {goal} as follows: \n\n".format(
                            url=url, goal=goal
                        )
                        cur_response += (
                            "Evidence in page: \n"
                            + "The provided webpage content could not be accessed. Please check the URL or file format."
                            + "\n\n"
                        )
                        cur_response += (
                            "Summary: \n"
                            + "The webpage content could not be processed, and therefore, no information is available."
                            + "\n\n"
                        )
                    else:
                        try:
                            cur_response = await self.readpage_jina(session, u, goal)
                        except Exception as e:  # pragma: no cover
                            cur_response = f"Error fetching {u}: {str(e)}"
                    responses.append(cur_response)
                response = "\n=======\n".join(responses)

        print(f"Summary Length {len(response)}; Summary Content {response}")
        return response.strip()

    def call_server(self, msgs, max_retries=2):
        api_key = os.environ.get("API_KEY")
        url_llm = os.environ.get("API_BASE")
        model_name = os.environ.get("SUMMARY_MODEL_NAME", "")
        client = OpenAI(
            api_key=api_key,
            base_url=url_llm,
        )
        for attempt in range(max_retries):
            try:
                chat_response = client.chat.completions.create(
                    model=model_name, messages=msgs, temperature=0.7
                )
                content = chat_response.choices[0].message.content
                if content:
                    try:
                        json.loads(content)
                    except:
                        # extract json from string
                        left = content.find("{")
                        right = content.rfind("}")
                        if left != -1 and right != -1 and left <= right:
                            content = content[left : right + 1]
                    return content
            except Exception as e:
                # print(e)
                if attempt == (max_retries - 1):
                    return ""
                continue

    async def jina_readpage(self, session: aiohttp.ClientSession, url: str) -> str:
        """
        Read webpage content using Jina service.

        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page

        Returns:
            str: The webpage content or error message
        """
        max_retries = 3
        timeout = 50

        headers = {"Authorization": f"Bearer {JINA_API_KEYS}"}
        for attempt in range(max_retries):
            try:
                async with session.get(
                    f"https://r.jina.ai/{url}", headers=headers, timeout=timeout
                ) as resp:
                    if resp.status == 200:
                        return await resp.text()
                    text = await resp.text()
                    print(text)
                    raise ValueError("jina readpage error")
            except Exception:
                await asyncio.sleep(0.5)
                if attempt == max_retries - 1:
                    return "[visit] Failed to read page."

        return "[visit] Failed to read page."

    async def html_readpage_jina(self, session: aiohttp.ClientSession, url: str) -> str:
        max_attempts = 8
        for attempt in range(max_attempts):
            content = await self.jina_readpage(session, url)
            service = "jina"
            print(service)
            if (
                content
                and not content.startswith("[visit] Failed to read page.")
                and content != "[visit] Empty content."
                and not content.startswith("[document_parser]")
            ):
                return content
        return "[visit] Failed to read page."

    async def readpage_jina(
        self, session: aiohttp.ClientSession, url: str, goal: str
    ) -> str:
        """
        Attempt to read webpage content by alternating between jina and aidata services.

        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page

        Returns:
            str: The webpage content or error message
        """

        summary_page_func = self.call_server
        max_retries = int(os.getenv("VISIT_SERVER_MAX_RETRIES", 1))

        content = await self.html_readpage_jina(session, url)

        if (
            content
            and not content.startswith("[visit] Failed to read page.")
            and content != "[visit] Empty content."
            and not content.startswith("[document_parser]")
        ):
            content = truncate_to_tokens(content, max_tokens=95000)
            messages = [
                {
                    "role": "user",
                    "content": EXTRACTOR_PROMPT.format(
                        webpage_content=content, goal=goal
                    ),
                }
            ]
            parse_retry_times = 0
            raw = summary_page_func(messages, max_retries=max_retries)
            summary_retries = 3
            while len(raw) < 10 and summary_retries >= 0:
                truncate_length = (
                    int(0.7 * len(content)) if summary_retries > 0 else 25000
                )
                status_msg = (
                    (
                        f"[visit] Summary url[{url}] "
                        f"attempt {3 - summary_retries + 1}/3, "
                        f"content length: {len(content)}, "
                        f"truncating to {truncate_length} chars"
                    )
                    if summary_retries > 0
                    else (
                        f"[visit] Summary url[{url}] failed after 3 attempts, "
                        f"final truncation to 25000 chars"
                    )
                )
                print(status_msg)
                content = content[:truncate_length]
                extraction_prompt = EXTRACTOR_PROMPT.format(
                    webpage_content=content, goal=goal
                )
                messages = [{"role": "user", "content": extraction_prompt}]
                raw = summary_page_func(messages, max_retries=max_retries)
                summary_retries -= 1

            parse_retry_times = 2
            if isinstance(raw, str):
                raw = raw.replace("```json", "").replace("```", "").strip()
            while parse_retry_times < 3:
                try:
                    raw = json.loads(raw)
                    break
                except:
                    raw = summary_page_func(messages, max_retries=max_retries)
                    parse_retry_times += 1

            if parse_retry_times >= 3:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(
                    url=url, goal=goal
                )
                useful_information += (
                    "Evidence in page: \n"
                    + "The provided webpage content could not be accessed. Please check the URL or file format."
                    + "\n\n"
                )
                useful_information += (
                    "Summary: \n"
                    + "The webpage content could not be processed, and therefore, no information is available."
                    + "\n\n"
                )
            else:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(
                    url=url, goal=goal
                )
                useful_information += (
                    "Evidence in page: \n" + str(raw["evidence"]) + "\n\n"
                )
                useful_information += "Summary: \n" + str(raw["summary"]) + "\n\n"

            if len(useful_information) < 10 and summary_retries < 0:
                print("[visit] Could not generate valid summary after maximum retries")
                useful_information = "[visit] Failed to read page"

            return useful_information

        # If no valid content was obtained after all retries
        useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(
            url=url, goal=goal
        )
        useful_information += (
            "Evidence in page: \n"
            + "The provided webpage content could not be accessed. Please check the URL or file format."
            + "\n\n"
        )
        useful_information += (
            "Summary: \n"
            + "The webpage content could not be processed, and therefore, no information is available."
            + "\n\n"
        )
        return useful_information
