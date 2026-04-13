#!/usr/bin/env python3
"""
Convert MCP-Universe benchmark configs (YAML + JSON tasks) into verl-tool RL datasets.

For each domain config (e.g., web_search.yaml), we produce a parquet under
  data/mcp_universe/<domain>/test.parquet

Row schema:
- data_source: domain
- prompt: [ {role: system, content: <React system>}, {role: user, content: <question>} ]
- reward_model: {style: rule, ground_truth?: str, evaluators?: list}
- extra_info: {index, question, use_specified_server, mcp_servers}

Notes:
- We embed the agent instruction from the YAML agent spec into the system prompt,
  and append a standard ReAct guidance on <tool_call> and <answer>.
- For web_search tasks using LLM-as-a-judge with a "correct_answer", we store it to ground_truth.
  Other tasks keep original evaluators so the reward manager can evaluate structure-based checks.
"""
from __future__ import annotations

import os
import argparse
import json
from typing import Any, Dict, List, Optional

import yaml
from datasets import Dataset
from pprint import pprint


# Historical prompt snippet kept during development. Not used.
# Intentionally removed to avoid escape-related syntax issues on newer Python.


def load_yaml_multi(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))
    return docs


def build_tools_prompt(mcp_servers: List[Dict[str, Any]], tools_description: Optional[str] = None) -> str:
    if not tools_description:
        return ""
    lines = [
        "You may call one or more functions to assist with the user query.",
        "",
        "You are provided with function signatures within <tools></tools> XML tags:",
        "<tools>",
        tools_description.strip(),
        "</tools>",
    ]
    return "\n".join(lines).strip()


def _json_template_from_output_format(output_format: Dict[str, Any]) -> str:
    """Render a JSON template block from a task's output_format.

    Keeps exact keys and structure; values are placeholders as-is so the
    model sees the required shape clearly.
    """
    import json
    try:
        return json.dumps(output_format, indent=2, ensure_ascii=False)
    except Exception:
        # Fallback to str if something odd appears
        return str(output_format)


def _example_from_output_format(output_format: Dict[str, Any]) -> str:
    """Produce a minimal example JSON object matching the shape of output_format.

    This replaces placeholder strings (e.g., "[NUMBER]", "NAME") with simple
    dummy values to demonstrate types while keeping the exact key names.
    The example is for format reference only; it will not necessarily pass
    semantic evaluators.
    """
    import re

    def _fill(val):
        if isinstance(val, dict):
            return {k: _fill(v) for k, v in val.items()}
        if isinstance(val, list):
            return [_fill(val[0])] if val else []
        if isinstance(val, (int, float)):
            return val
        if isinstance(val, str):
            s = val.strip()
            s_lower = s.lower()
            # common numeric placeholders
            if any(tok in s_lower for tok in ["[number]", "xx.x", "x.x", "0.x", "percent", "percentage"]):
                return 0.0
            # id-like
            if "id" in s_lower:
                return "id_example"
            # address-like
            if "address" in s_lower:
                return "123 Example St, City"
            # name-like
            if any(tok in s_lower for tok in ["name", "title", "route_name"]):
                return "Example"
            # generic fallback: strip bracket hints like [VALUE]
            if re.match(r"^\[.*\]$", s):
                return "VALUE"
            return s
        return val

    import json
    try:
        return json.dumps(_fill(output_format), indent=2, ensure_ascii=False)
    except Exception:
        return str(output_format)


def build_system_prompt(agent_instruction: str,
                        question: str,
                        tools_prompt: str,
                        output_format: Optional[Dict[str, Any]] = None,
                        domain_hint: Optional[str] = None) -> str:
    instruction = agent_instruction.strip() if agent_instruction else "You are an assistant."
    # ReAct template aligned with MCP-Universe guidance, augmented with tag usage
    # Build the header and tool-call example
    text = (
    "You are a ReAct (Reasoning and Acting) agent.\n"
    f"{instruction}\n\n"
    + (tools_prompt + "\n\n" if tools_prompt else "")
    + "You need to answer the following question:\n\n"
    f"Question: {question}\n\n"
    "Your goal is to reason about the question and decide on the best course of action to answer it accurately.\n"
    "You need to choose the appropriate tool based on the question. If no tool is needed, reply directly.\n"
    "Please use only the tools that are explicitly defined above.\n\n"
    "Instructions:\n"
    "1. Analyze the query, previous reasoning steps, and results.\n"
    "2. Decide on the next action: use a tool or provide a final answer.\n"
    "3. You MUST output the final answer within 20 steps.\n"
    "4. Respond in the following format:\n\n"
    "If you need to use a tool:\n"
    '<think>Your detailed reasoning about what to do next</think>'
    '<tool_call>{"name":"tool-name","arguments":{"argument-name":"argument-value"}}</tool_call>\n\n'
    "For example:\n"
    "<think>I need to find the weather in Tokyo</think>"
    '<tool_call>{"name":"get_weather","arguments":{"city":"Tokyo"}}</tool_call>\n\n'
    "If you have enough information to answer the query:\n"
    "<think>Your final reasoning process to derive the answer</think>"
    "__FINAL_ANSWER_EXAMPLE__"
    "Remember:\n"
    "- Be thorough in your reasoning.\n"
    "- Use tools when you need more information.\n"
    "- Always base your reasoning on the actual results from tool use.\n"
    "- If a tool returns no results or fails, acknowledge this and consider using a different tool or approach.\n"
    "- Provide a final answer when you're confident you have sufficient information.\n"
    "- The response must be in a valid XML format.\n"
    )
    # Inject the final-answer example according to task output format (user-preferred placement)
    if isinstance(output_format, dict) and output_format:
        tmpl = _json_template_from_output_format(output_format)
        example = _example_from_output_format(output_format)
        final_block = (
            f"<answer>\n{tmpl}\n</answer>\n\n"
            "For example (format only):\n"
            f"<think>Derive the final structured result</think><answer>\n{example}\n</answer>\n\n"
        )
        text = text.replace("__FINAL_ANSWER_EXAMPLE__", final_block)
        # Add one explicit reminder for JSON-within-XML
        text += (
            "\nNote:\n"
            "- Place exactly one valid JSON object inside <answer>...</answer>.\n"
            "- Match keys and nesting exactly as shown. Do not add extra keys.\n"
        )
    else:
        final_block = (
            "<answer>Final answer to the query</answer>\n\n"
            "For example:\n"
            "<think>The weather in Tokyo is sunny</think><answer>sunny</answer>\n\n"
        )
        text = text.replace("__FINAL_ANSWER_EXAMPLE__", final_block)

    return text



def _format_tools_description(tool_map: Dict[str, List[Dict[str, Any]]]) -> str:
    import copy
    import json

    entries: List[str] = []
    for tools in tool_map.values():
        for tool in tools or []:
            name = tool.get("name", "")
            if not name:
                continue
            description = tool.get("description") or ""
            schema = tool.get("inputSchema") or tool.get("parameters") or {}
            parameters = copy.deepcopy(schema) if isinstance(schema, dict) else {}
            if not isinstance(parameters, dict):
                parameters = {}
            parameters.setdefault("type", "object")
            parameters.setdefault("properties", {})
            if not isinstance(parameters.get("properties"), dict):
                parameters["properties"] = {}
            normalized_props: Dict[str, Any] = {}
            for arg_name, meta in parameters["properties"].items():
                if not isinstance(meta, dict):
                    normalized_props[arg_name] = meta
                    continue
                desc = meta.get("description")
                title = meta.get("title")
                if not desc:
                    desc = title or arg_name
                ordered_meta: Dict[str, Any] = {}
                if desc:
                    ordered_meta["description"] = desc
                if title:
                    ordered_meta["title"] = title
                for k, v in meta.items():
                    if k in ("description", "title"):
                        continue
                    ordered_meta[k] = v
                normalized_props[arg_name] = ordered_meta
            parameters["properties"] = normalized_props
            parameters.setdefault("required", [])
            if not isinstance(parameters.get("required"), list):
                parameters["required"] = []
            entry = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
            entries.append(json.dumps(entry, ensure_ascii=False))
    return "\n".join(entries)


def _query_tools_for_servers(server_names: List[str]) -> Optional[str]:
    """Query gateway/single endpoint to build a Tools Description.

    Returns text if at least one server responds; otherwise None.
    """
    try:
        import asyncio
        from verl_tool.servers.tools.utils.mcp_client import list_tools_and_resources

        async def _collect():
            tool_map: Dict[str, List[Dict[str, Any]]] = {}
            for s in server_names:
                try:
                    tools, _ = await list_tools_and_resources(s)
                    tool_map[s] = [
                        {
                            "name": getattr(t, "name", ""),
                            "description": getattr(t, "description", None),
                            "inputSchema": getattr(t, "inputSchema", None),
                        }
                        for t in getattr(tools, "tools", []) or []
                    ]
                except Exception:
                    continue
            return tool_map

        try:
            loop = asyncio.get_event_loop()
            tool_map = loop.run_until_complete(_collect()) if not loop.is_running() else asyncio.run(_collect())
        except RuntimeError:
            tool_map = asyncio.run(_collect())
        if any(tool_map.get(k) for k in tool_map.keys()):
            return _format_tools_description(tool_map)
        return None
    except Exception:
        return None


def extract_ground_truth_from_evaluators(evaluators: List[Dict[str, Any]]) -> str | None:
    # Handle the common web_search pattern: raw -> google_search.llm_as_a_judge with correct_answer
    for ev in evaluators or []:
        op = ev.get("op", "")
        if isinstance(op, str) and op.endswith("google_search.llm_as_a_judge"):
            op_args = ev.get("op_args", {}) or {}
            gt = op_args.get("correct_answer")
            if isinstance(gt, str) and gt.strip():
                return gt.strip()
    return None


def _trim(s: str, n: int = 160) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "..."


def process_domain(
    config_root: str,
    yaml_file: str,
    out_root: str,
    *,
    preview: int = 0,
    max_prompt_chars: int = 200,
    human_dump: Optional[str] = "readable.md",
    human_max_rows: int = 20,
    human_max_chars: int = 2000,
):
    # YAMLs live under `<configs_root>/test/`, while task JSONs referenced inside YAML
    # are relative to `<configs_root>/` (e.g., `test/web_search/...json`).
    cfg_path = os.path.join(config_root, "test", yaml_file)
    docs = load_yaml_multi(cfg_path)
    # find agent doc and benchmark doc
    agent_doc = next((d for d in docs if str(d.get("kind", "")).lower() == "agent"), None)
    bench_doc = next((d for d in docs if str(d.get("kind", "")).lower() == "benchmark"), None)
    assert agent_doc and bench_doc, f"Invalid config: {cfg_path}"

    agent_spec = agent_doc.get("spec", {})
    agent_cfg = agent_spec.get("config", {})
    agent_servers = agent_cfg.get("servers", []) or []
    instruction = agent_cfg.get("instruction", "You are an assistant.")

    tasks = bench_doc.get("spec", {}).get("tasks", [])
    rows: List[Dict[str, Any]] = []
    for idx, rel in enumerate(tasks):
        # task path is relative to mcpuniverse/benchmark/configs/
        task_path = os.path.join(config_root, rel)
        with open(task_path, "r", encoding="utf-8") as f:
            task = json.load(f)
        question = task.get("question", "")
        output_format = task.get("output_format", {}) or None
        evaluators = task.get("evaluators", [])
        use_specified_server = bool(task.get("use_specified_server", False))
        task_mcp_servers = task.get("mcp_servers", [])
        # Prefer per-task servers; fallback to agent-level servers when absent
        mcp_servers = task_mcp_servers if task_mcp_servers else agent_servers

        ground_truth = extract_ground_truth_from_evaluators(evaluators)
        reward_model: Dict[str, Any] = {"style": "rule"}
        if ground_truth is not None:
            reward_model["ground_truth"] = ground_truth
        # We currently evaluate only with EM; omit complex evaluators to keep schema simple

        # Build TOOLS_PROMPT from declared MCP servers in task config
        server_names = []
        for s in mcp_servers or []:
            if isinstance(s, dict) and s.get("name"):
                server_names.append(s["name"])
            elif isinstance(s, str):
                server_names.append(s)
        tools_desc = _query_tools_for_servers(server_names)
        tools_prompt = build_tools_prompt(mcp_servers, tools_desc)
        # Build final system prompt embedding the current question and tools prompt
        domain_hint = os.path.splitext(os.path.basename(yaml_file))[0]
        sys_prompt = build_system_prompt(instruction, question, tools_prompt, output_format=output_format, domain_hint=domain_hint)

        row = {
            "data_source": os.path.splitext(os.path.basename(yaml_file))[0],
            "prompt": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": question},
            ],
            "ability": "mcp",
            "reward_model": reward_model,
            "extra_info": {
                "index": idx,
                "yaml": yaml_file,
                "task": rel,
                "question": question,
                "use_specified_server": use_specified_server,
            "mcp_servers": mcp_servers,
            },
        }
        rows.append(row)

    # Preview summary and samples
    if preview and rows:
        print("\n=== Domain Summary ===")
        print(f"yaml: {yaml_file}")
        print(f"num_items: {len(rows)}")
        with_gt = sum(1 for r in rows if isinstance(r.get("reward_model"), dict) and r["reward_model"].get("ground_truth") is not None)
        print(f"with_ground_truth: {with_gt}")
        # collect server names
        server_names = []
        for r in rows:
            mcp_servers = (r.get("extra_info", {}) or {}).get("mcp_servers", [])
            for s in mcp_servers or []:
                name = s.get("name") if isinstance(s, dict) else None
                if name:
                    server_names.append(name)
        uniq_servers = sorted(set(server_names))
        print(f"servers_declared: {uniq_servers}")

        print("\n--- Sample rows ---")
        for i, r in enumerate(rows[:preview]):
            sys_txt = r["prompt"][0]["content"] if r.get("prompt") else ""
            user_txt = r["prompt"][1]["content"] if r.get("prompt") and len(r["prompt"]) > 1 else ""
            gt = r["reward_model"].get("ground_truth") if isinstance(r.get("reward_model"), dict) else None
            print(f"[#{i}] index={r['extra_info']['index']} task={r['extra_info']['task']}")
            print("question:", _trim(str(r['extra_info'].get('question','')), 200))
            print("ground_truth:", _trim(str(gt), 200) if gt is not None else None)
            print("system_prompt(~chars):", len(sys_txt))
            print(_trim(sys_txt, max_prompt_chars))
            print("user(~chars):", len(user_txt))
            print(_trim(user_txt, 120))
            print()

    out_dir = os.path.join(out_root, os.path.splitext(os.path.basename(yaml_file))[0])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test.parquet")
    ds = Dataset.from_list(rows)
    ds.to_parquet(out_path)
    print(f"Wrote {len(ds)} rows to {out_path}")

    # Optional human-readable dump
    if human_dump:
        hr_path = os.path.join(out_dir, human_dump)
        with open(hr_path, "w", encoding="utf-8") as f:
            domain = os.path.splitext(os.path.basename(yaml_file))[0]
            # Header summary
            f.write(f"Domain: {domain}\n")
            f.write(f"Items: {len(rows)}\n")
            # collect union servers
            server_names = []
            for r in rows:
                for s in (r.get("extra_info", {}) or {}).get("mcp_servers", []) or []:
                    if isinstance(s, dict) and s.get("name"):
                        server_names.append(s["name"])
                    elif isinstance(s, str):
                        server_names.append(s)
            uniq_servers = ", ".join(sorted(set(server_names)))
            f.write(f"Servers (union): {uniq_servers}\n\n")
            # Rows (truncated)
            upto = min(human_max_rows, len(rows))
            for i in range(upto):
                r = rows[i]
                sys_txt = r["prompt"][0]["content"] if r.get("prompt") else ""
                user_txt = r["prompt"][1]["content"] if r.get("prompt") and len(r["prompt"]) > 1 else ""
                has_tools_desc = "### Tools Description ###" in sys_txt
                mcp_srvs = r.get("extra_info", {}).get("mcp_servers", [])
                mcp_srvs_str = ", ".join(
                    [s.get("name") for s in mcp_srvs if isinstance(s, dict) and s.get("name")] +
                    [s for s in mcp_srvs if isinstance(s, str)]
                )
                f.write(f"## Item {i} | idx={r['extra_info']['index']}\n")
                f.write(f"Task: {r['extra_info'].get('task','')}\n")
                f.write(f"Servers (per item): {mcp_srvs_str}\n")
                f.write(f"ToolsDescription: {has_tools_desc}\n")
                f.write("Question:\n")
                f.write((user_txt[:human_max_chars] + ("...\n" if len(user_txt) > human_max_chars else "\n")))
                f.write("SystemPrompt (truncated):\n")
                f.write((sys_txt[:human_max_chars] + ("\n...\n" if len(sys_txt) > human_max_chars else "\n")))
                f.write("\n")
        print(f"Wrote human-readable preview to {hr_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert MCP-Universe configs to verl-tool datasets")
    parser.add_argument("--configs_root", type=str,
                        default="verl-tool/benchmarks/MCP-Universe/mcpuniverse/benchmark/configs",
                        help="Root dir of MCP-Universe benchmark configs (the directory that contains domain YAMLs and the test/ folder)")
    parser.add_argument("--out_root", type=str, default="data/mcp_universe", help="Output root directory")
    parser.add_argument("--domains", type=str,
                        default="web_search,location_navigation,browser_automation,financial_analysis,repository_management,3d_design",
                        help="Comma-separated domain yaml names (without .yaml) to convert")
    parser.add_argument("--preview", type=int, default=2, help="Print N sample rows and summary per domain (0 to disable)")
    parser.add_argument("--max_prompt_chars", type=int, default=200, help="Max chars to show for system prompt preview")
    parser.add_argument("--human_dump", type=str, default="readable.md", help="Human-readable dump filename in each domain folder (empty to disable)")
    parser.add_argument("--human_max_rows", type=int, default=20, help="Max rows to include in human-readable dump")
    parser.add_argument("--human_max_chars", type=int, default=2000, help="Max chars per prompt in human-readable dump")
    args = parser.parse_args()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    for d in domains:
        yaml_file = f"{d}.yaml"
        process_domain(
            args.configs_root,
            yaml_file,
            args.out_root,
            preview=args.preview,
            max_prompt_chars=args.max_prompt_chars,
            human_dump=(args.human_dump if args.human_dump else None),
            human_max_rows=args.human_max_rows,
            human_max_chars=args.human_max_chars,
        )


if __name__ == "__main__":
    main()
