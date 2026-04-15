#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Serve a local web UI for visualizing trajectory files with inline stats.

Usage:
  python refactored_scripts/serve_trajectories.py --root data/<your_step6_dir> --port 8899

Security:
- Only serves files under the provided --root directory.
- Binds to 127.0.0.1 by default.
- IDs in API responses are indexes, so there is no path traversal surface.
"""

import argparse
import json
import re
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List, Optional
from urllib.parse import parse_qs, urlparse


MAX_CONTENT_CHARS = 200_000  # Prevent runaway memory usage in UI/API.

# Initialize tokenizer (with fallback to word count)
_tokenizer = None
_tokenizer_name = None
try:
    from transformers import AutoTokenizer
    # Try user's requested tokenizer first
    for model_name in ["Qwen/Qwen3-8B"]:
        try:
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            _tokenizer_name = model_name
            print(f"Tokenizer loaded successfully: {model_name}")
            break
        except Exception as e:
            continue
    if _tokenizer is None:
        print(f"Warning: Could not load any Qwen tokenizer, falling back to word count")
except Exception as e:
    print(f"Warning: Could not load tokenizer library, falling back to word count: {e}")
    _tokenizer = None


def count_tokens(text: str) -> int:
    """Count tokens using tokenizer or fallback to words."""
    if _tokenizer is not None:
        try:
            return len(_tokenizer.encode(text, add_special_tokens=False))
        except:
            pass
    return len(text.split())


def compute_trajectory_stats(text: str) -> dict:
    """Compute detailed stats from trajectory content."""
    stats = {
        "tokens": count_tokens(text),
        "parallel_ratio": None,
        "acceleration_ratio": None,
        "num_tokens_in_the_longest_thread": None,
        "total_num_tokens": None,
        "avg_thread_length": None,
        "avg_tokens_per_parallel_block": None,
        "avg_threads_per_parallel_block": None,
        "avg_outlines_length": None,
        "avg_conclusion_length": None,
    }

    # Find parallel blocks
    parallel_matches = list(re.finditer(r'<Parallel>(.*?)</Parallel>', text, re.DOTALL))
    if not parallel_matches:
        stats["parallel_ratio"] = 0.0
        stats["total_num_tokens"] = stats["tokens"]
        stats["num_tokens_in_the_longest_thread"] = stats["tokens"]
        if stats["total_num_tokens"] > 0:
            stats["acceleration_ratio"] = 0.0
        return stats

    stats["total_num_tokens"] = stats["tokens"]

    # Calculate metrics
    inside_tokens = 0
    thread_lengths = []
    parallel_block_tokens = []
    parallel_block_thread_counts = []
    outlines_lengths = []
    conclusion_lengths = []
    longest_thread_tokens = 0
    last_end_idx = 0

    for match in parallel_matches:
        # Add tokens from the non-parallel text segment before this block
        non_parallel_segment = text[last_end_idx:match.start()]
        longest_thread_tokens += count_tokens(non_parallel_segment)
        block_content = match.group(1)

        # Count tokens in block (excluding placeholders)
        block_without_placeholders = re.sub(
            r'<Thread>\s*\d+:\s*\[placeholder\]\s*</Thread>',
            '',
            block_content,
            flags=re.IGNORECASE
        )
        block_tokens = count_tokens(block_without_placeholders)
        inside_tokens += block_tokens
        parallel_block_tokens.append(count_tokens('<Parallel>' + block_without_placeholders + '</Parallel>'))

        # Outlines length (include leading/trailing whitespace for accurate token counting)
        outlines_match = re.search(r'\s*<Outlines>.*?</Outlines>\s*', block_content, re.DOTALL)
        if outlines_match:
            outlines_tokens = count_tokens(outlines_match.group(0))
            outlines_lengths.append(outlines_tokens)

        # Conclusion length (include leading/trailing whitespace for accurate token counting)
        conclusion_match = re.search(r'\s*<Conclusion>.*?</Conclusion>\s*', block_content, re.DOTALL)
        if conclusion_match:
            conclusion_tokens = count_tokens(conclusion_match.group(0))
            conclusion_lengths.append(conclusion_tokens)

        # Thread lengths
        thread_matches = list(re.finditer(r'<Thread>.*?</Thread>', block_content, re.DOTALL))
        block_thread_tokens = []
        for t_match in thread_matches:
            thread_content = t_match.group(0)
            if not re.search(r'<Thread>\s*\d+:\s*\[placeholder\]\s*</Thread>', thread_content, re.IGNORECASE):
                t_tokens = count_tokens(thread_content)
                thread_lengths.append(t_tokens)
                block_thread_tokens.append(t_tokens)

        parallel_block_thread_counts.append(len(block_thread_tokens))

        # Add to longest thread calculation
        block_contribution = count_tokens('<Parallel></Parallel>')
        if outlines_match:
            block_contribution += outlines_tokens
        if conclusion_match:
            block_contribution += conclusion_tokens
        if block_thread_tokens:
            block_contribution += max(block_thread_tokens)
        longest_thread_tokens += block_contribution

        # Update last_end_idx to track where this block ended
        last_end_idx = match.end()

    # Add tokens from the final text segment after the last parallel block
    final_segment = text[last_end_idx:]
    longest_thread_tokens += count_tokens(final_segment)

    # Calculate final stats
    if stats["total_num_tokens"] > 0:
        # Mirror computation in filter-format-correct-and-obtain-stats: empty blocks yield None.
        stats["parallel_ratio"] = inside_tokens / stats["total_num_tokens"] if inside_tokens > 0 else None
    stats["num_tokens_in_the_longest_thread"] = longest_thread_tokens

    # Calculate acceleration ratio: measures potential speedup from parallelization
    # acceleration_ratio = 1 - (longest_thread / total_tokens)
    # 0 = no speedup (fully sequential), higher values = more parallelization
    if longest_thread_tokens is not None and stats["total_num_tokens"] > 0:
        stats["acceleration_ratio"] = 1 - (longest_thread_tokens / stats["total_num_tokens"])

    stats["avg_thread_length"] = sum(thread_lengths) / len(thread_lengths) if thread_lengths else None
    stats["avg_tokens_per_parallel_block"] = sum(parallel_block_tokens) / len(parallel_block_tokens) if parallel_block_tokens else None
    stats["avg_threads_per_parallel_block"] = sum(parallel_block_thread_counts) / len(parallel_block_thread_counts) if parallel_block_thread_counts else None
    stats["avg_outlines_length"] = sum(outlines_lengths) / len(outlines_lengths) if outlines_lengths else None
    stats["avg_conclusion_length"] = sum(conclusion_lengths) / len(conclusion_lengths) if conclusion_lengths else None

    return stats


@dataclass
class Trajectory:
    idx: int
    path: Path
    name: str
    size_bytes: int
    lines: int
    words: int
    chars: int
    tokens: int
    detailed_stats: dict


def natural_sort_key(path: Path) -> list:
    """Natural sort key for file paths (e.g., 1.txt, 2.txt, 10.txt instead of 1.txt, 10.txt, 2.txt)"""
    import re
    parts = []
    for part in re.split(r'(\d+)', path.name):
        if part.isdigit():
            parts.append(int(part))
        else:
            parts.append(part.lower())
    return parts


def scan(root: Path) -> List[Trajectory]:
    files = sorted((p for p in root.glob("*.txt") if p.is_file()), key=natural_sort_key)
    trajectories: List[Trajectory] = []
    for i, p in enumerate(files):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            # If unreadable, skip but keep the server alive.
            continue
        size = p.stat().st_size
        lines = text.count("\n") + 1 if text else 0
        words = len(text.split())
        chars = len(text)
        detailed_stats = compute_trajectory_stats(text)
        tokens = detailed_stats["tokens"]
        trajectories.append(
            Trajectory(
                idx=i,
                path=p,
                name=p.name,
                size_bytes=size,
                lines=lines,
                words=words,
                chars=chars,
                tokens=tokens,
                detailed_stats=detailed_stats,
            )
        )
    return trajectories


def format_size(bytes_count: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_count < 1024 or unit == "GB":
            return f"{bytes_count:.1f} {unit}"
        bytes_count /= 1024
    return f"{bytes_count:.1f} GB"


class TrajectoryHandler(BaseHTTPRequestHandler):
    def _json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, body: str, status=200):
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._html(self.render_index())
        if parsed.path == "/api/trajectories":
            return self._json(self.list_trajectories())
        if parsed.path == "/api/trajectory":
            return self._json(self.get_trajectory(parsed), status=200)
        self.send_error(404, "Not Found")

    def list_trajectories(self):
        return {
            "root": str(self.server.root),
            "count": len(self.server.trajectories),
            "stats": [
                {
                    "id": t.idx,
                    "name": t.name,
                    "size_bytes": t.size_bytes,
                    "size_pretty": format_size(t.size_bytes),
                    "lines": t.lines,
                    "words": t.words,
                    "chars": t.chars,
                    "tokens": t.tokens,
                    **t.detailed_stats,
                }
                for t in self.server.trajectories
            ],
        }

    def get_trajectory(self, parsed):
        qs = parse_qs(parsed.query or "")
        try:
            idx = int(qs.get("id", [""])[0])
        except Exception:
            return {"error": "invalid id"}
        if idx < 0 or idx >= len(self.server.trajectories):
            return {"error": "id out of range"}
        traj = self.server.trajectories[idx]
        try:
            text = traj.path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return {"error": f"could not read file: {e}"}
        truncated = len(text) > MAX_CONTENT_CHARS
        if truncated:
            text = text[:MAX_CONTENT_CHARS]
        return {
            "id": traj.idx,
            "name": traj.name,
            "size_bytes": traj.size_bytes,
            "size_pretty": format_size(traj.size_bytes),
            "lines": traj.lines,
            "words": traj.words,
            "chars": traj.chars,
            "tokens": traj.tokens,
            "content": text,
            "truncated": truncated,
            **traj.detailed_stats,
        }

    def log_message(self, fmt, *args):
        # Keep output minimal.
        return

    def render_index(self) -> str:
        # All assets are inline for offline safety.
        return """<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ThreadWeaver Trajectory Explorer</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
      --bg-primary: #ffffff;
      --bg-secondary: #f8f9fa;
      --bg-tertiary: #e9ecef;
      --card-bg: rgba(255, 255, 255, 0.9);
      --card-hover: rgba(248, 249, 250, 1);
      --accent-primary: #059669;
      --accent-secondary: #2563eb;
      --accent-tertiary: #7c3aed;
      --accent-warning: #d97706;
      --text-primary: #1f2937;
      --text-secondary: #4b5563;
      --text-muted: #9ca3af;
      --border: rgba(209, 213, 219, 0.5);
      --border-hover: rgba(156, 163, 175, 0.5);
      --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
      --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
      --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
      --shadow-glow: 0 0 20px rgba(5, 150, 105, 0.15);
      --radius-sm: 6px;
      --radius-md: 10px;
      --radius-lg: 16px;
      --transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
      --xml-tag: #0369a1;
      --xml-attr: #059669;
      --xml-text: #1f2937;
    }

    body.theme-dark {
      --bg-primary: #0f172a;
      --bg-secondary: #111827;
      --bg-tertiary: #1f2937;
      --card-bg: rgba(15, 23, 42, 0.9);
      --card-hover: rgba(30, 41, 59, 0.95);
      --accent-primary: #22c55e;
      --accent-secondary: #60a5fa;
      --accent-tertiary: #c084fc;
      --accent-warning: #f59e0b;
      --text-primary: #e5e7eb;
      --text-secondary: #cbd5e1;
      --text-muted: #94a3b8;
      --border: rgba(148, 163, 184, 0.3);
      --border-hover: rgba(148, 163, 184, 0.5);
      --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.4);
      --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
      --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.6);
      --shadow-glow: 0 0 20px rgba(34, 197, 94, 0.25);
      --xml-tag: #7dd3fc;
      --xml-attr: #34d399;
      --xml-text: #e5e7eb;
      background: radial-gradient(circle at 20% 20%, rgba(34, 197, 94, 0.1), transparent 30%), radial-gradient(circle at 80% 0%, rgba(96, 165, 250, 0.12), transparent 28%), #0b1224;
      color: var(--text-primary);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
      color: var(--text-primary);
      min-height: 100vh;
      padding: 24px;
      line-height: 1.6;
    }

    .header {
      max-width: 1600px;
      margin: 0 auto 32px;
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      flex-wrap: wrap;
      gap: 20px;
    }

    .header-left h1 {
      font-size: 2.5rem;
      font-weight: 700;
      background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      margin-bottom: 8px;
      letter-spacing: -0.02em;
    }

    .subtitle {
      color: var(--text-secondary);
      font-size: 1rem;
    }

    .header-right {
      display: flex;
      gap: 12px;
      align-items: center;
    }

    .stat-pill {
      background: var(--card-bg);
      backdrop-filter: blur(10px);
      padding: 12px 20px;
      border-radius: var(--radius-lg);
      border: 1px solid var(--border);
      box-shadow: var(--shadow-md);
      display: flex;
      flex-direction: column;
      gap: 4px;
      min-width: 140px;
    }

    .stat-pill .label {
      font-size: 0.75rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-weight: 600;
    }

    .stat-pill .value {
      font-size: 1.5rem;
      font-weight: 700;
      background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }

    .container {
      max-width: 1600px;
      margin: 0 auto;
    }

    .stats-overview {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }

    .stat-card {
      background: var(--card-bg);
      backdrop-filter: blur(10px);
      padding: 20px;
      border-radius: var(--radius-md);
      border: 1px solid var(--border);
      box-shadow: var(--shadow-md);
      transition: var(--transition);
    }

    .stat-card:hover {
      border-color: var(--border-hover);
      transform: translateY(-2px);
      box-shadow: var(--shadow-lg);
    }

    .stat-card .stat-label {
      font-size: 0.875rem;
      color: var(--text-muted);
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .stat-card .stat-value {
      font-size: 1.75rem;
      font-weight: 700;
      color: var(--text-primary);
    }

    .stat-card .stat-subtext {
      font-size: 0.75rem;
      color: var(--text-secondary);
      margin-top: 4px;
    }

    .main-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 24px;
    }

    @media (min-width: 1200px) {
      .main-grid {
        grid-template-columns: 400px 1fr;
      }
    }

    .panel {
      background: var(--card-bg);
      backdrop-filter: blur(10px);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 24px;
      box-shadow: var(--shadow-lg);
      transition: var(--transition);
    }

    .panel-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border);
    }

    .panel-title {
      font-size: 1.25rem;
      font-weight: 600;
      color: var(--text-primary);
    }

    .search-box {
      position: relative;
      margin-bottom: 16px;
    }

    .search-box input {
      width: 100%;
      padding: 12px 16px 12px 40px;
      background: var(--bg-secondary);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      color: var(--text-primary);
      font-size: 0.875rem;
      transition: var(--transition);
    }

    .search-box input:focus {
      outline: none;
      border-color: var(--accent-primary);
      box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
    }

    .search-box::before {
      content: "🔍";
      position: absolute;
      left: 14px;
      top: 50%;
      transform: translateY(-50%);
      opacity: 0.5;
    }

    .traj-list {
      max-height: 600px;
      overflow-y: auto;
      margin: -8px;
      padding: 8px;
    }

    .traj-item {
      padding: 12px 16px;
      margin-bottom: 8px;
      background: var(--bg-secondary);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      cursor: pointer;
      transition: var(--transition);
    }

    .traj-item:hover {
      background: var(--bg-tertiary);
      border-color: var(--accent-primary);
      transform: translateX(4px);
    }

    .traj-item.active {
      background: var(--bg-tertiary);
      border-color: var(--accent-primary);
      box-shadow: var(--shadow-glow);
    }

    .traj-item .traj-name {
      font-weight: 600;
      color: var(--text-primary);
      margin-bottom: 6px;
      font-size: 0.875rem;
    }

    .traj-item .traj-meta {
      display: flex;
      gap: 12px;
      font-size: 0.75rem;
      color: var(--text-muted);
    }

    .content-panel {
      display: flex;
      flex-direction: column;
      min-height: 600px;
    }

    .content-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
      flex-wrap: wrap;
      gap: 16px;
    }

    .content-title {
      font-size: 1.125rem;
      font-weight: 600;
      color: var(--text-primary);
    }

    .content-actions {
      display: flex;
      gap: 8px;
    }

    .btn {
      padding: 8px 16px;
      background: var(--bg-secondary);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      color: var(--text-primary);
      font-size: 0.875rem;
      cursor: pointer;
      transition: var(--transition);
      font-weight: 500;
    }

    .btn:hover {
      background: var(--bg-tertiary);
      border-color: var(--accent-primary);
    }

    .tags {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 16px;
    }

    .tag {
      padding: 6px 12px;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    .tag.parallel {
      background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(139, 92, 246, 0.1));
      color: var(--accent-tertiary);
      border: 1px solid rgba(139, 92, 246, 0.3);
    }

    .tag.thinking {
      background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(59, 130, 246, 0.1));
      color: var(--accent-secondary);
      border: 1px solid rgba(59, 130, 246, 0.3);
    }

    .tag.boxed {
      background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(16, 185, 129, 0.1));
      color: var(--accent-primary);
      border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .tag.outline {
      background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(245, 158, 11, 0.1));
      color: var(--accent-warning);
      border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(100px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }

    @media (max-width: 1200px) {
      .metrics-grid {
        grid-template-columns: repeat(3, minmax(100px, 1fr));
      }
    }

    @media (max-width: 768px) {
      .metrics-grid {
        grid-template-columns: repeat(2, minmax(100px, 1fr));
      }
    }

    .metric {
      background: var(--bg-secondary);
      padding: 12px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border);
      text-align: center;
    }

    .metric .metric-value {
      font-size: 1.25rem;
      font-weight: 700;
      color: var(--accent-primary);
    }

    .metric .metric-label {
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-top: 4px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .content-viewer {
      flex: 1;
      background: var(--bg-primary);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      overflow: hidden;
      max-height: 70vh;
    }

    .editor-container {
      height: 70vh;
      min-height: 420px;
    }

    .monaco-fallback {
      padding: 20px;
      margin: 0;
      white-space: pre-wrap;
      word-wrap: break-word;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.875rem;
      line-height: 1.7;
    }

    .tooltip {
      position: relative;
      display: inline-block;
      cursor: help;
      border-bottom: 1px dotted var(--text-muted);
    }

    .tooltip .tooltiptext {
      visibility: hidden;
      width: 280px;
      background-color: var(--text-primary);
      color: var(--bg-primary);
      text-align: left;
      border-radius: var(--radius-sm);
      padding: 12px;
      position: absolute;
      z-index: 1000;
      bottom: 125%;
      left: 50%;
      margin-left: -140px;
      opacity: 0;
      transition: opacity 0.3s;
      font-size: 0.875rem;
      line-height: 1.4;
      box-shadow: var(--shadow-lg);
    }

    .tooltip .tooltiptext::after {
      content: "";
      position: absolute;
      top: 100%;
      left: 50%;
      margin-left: -5px;
      border-width: 5px;
      border-style: solid;
      border-color: var(--text-primary) transparent transparent transparent;
    }

    .tooltip:hover .tooltiptext {
      visibility: visible;
      opacity: 1;
    }

    .alert {
      padding: 12px 16px;
      border-radius: var(--radius-sm);
      margin-top: 12px;
      font-size: 0.875rem;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .alert.warning {
      background: rgba(245, 158, 11, 0.1);
      border: 1px solid rgba(245, 158, 11, 0.3);
      color: var(--accent-warning);
    }

    .empty-state {
      text-align: center;
      padding: 60px 20px;
      color: var(--text-muted);
    }

    .empty-state svg {
      width: 64px;
      height: 64px;
      margin-bottom: 16px;
      opacity: 0.3;
    }

    ::-webkit-scrollbar {
      width: 8px;
      height: 8px;
    }

    ::-webkit-scrollbar-track {
      background: var(--bg-secondary);
      border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
      background: var(--border);
      border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
      background: var(--border-hover);
    }

    .loading {
      display: inline-block;
      width: 16px;
      height: 16px;
      border: 2px solid var(--border);
      border-top-color: var(--accent-primary);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    .fade-in {
      animation: fadeIn 0.3s ease-in;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>
  <div class="header">
    <div class="header-left">
      <h1>ThreadWeaver Explorer</h1>
      <div class="subtitle">Visualize and analyze trajectory generation outputs</div>
    </div>
    <div class="header-right">
      <div class="stat-pill">
        <div class="label">Total Files</div>
        <div class="value" id="total-count">-</div>
      </div>
    </div>
  </div>

  <div class="container">
    <div class="stats-overview" id="stats-overview"></div>

    <div class="main-grid">
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title">Trajectories</div>
        </div>

        <div class="search-box">
          <input type="text" id="search" placeholder="Search trajectories..." />
        </div>

        <div class="traj-list" id="traj-list"></div>
      </div>

      <div class="panel content-panel">
        <div class="content-header">
          <div class="content-title" id="content-title">Select a trajectory</div>
          <div class="content-actions">
            <button class="btn" id="copy-btn" style="display:none;">📋 Copy</button>
            <button class="btn" id="highlight-btn" style="display:none;">🌙 Dark</button>
          </div>
        </div>

        <div id="content-tags" class="tags"></div>
        <div id="content-metrics" class="metrics-grid"></div>
        <div class="content-viewer" id="content-viewer">
          <div id="editor-container" class="editor-container">
            <div class="empty-state">
              <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <div>Select a trajectory from the list to view its content</div>
            </div>
          </div>
        </div>
        <div id="truncated-alert" style="display:none;" class="alert warning">
          ⚠️ Content truncated to improve performance
        </div>
      </div>
    </div>
  </div>

<script src="https://cdn.jsdelivr.net/npm/monaco-editor@0.45.0/min/vs/loader.js"></script>
<script>
const fmt = new Intl.NumberFormat('en-US');
let trajectories = [];
let currentId = null;
let editorInstance = null;
let monacoApi = null;
let monacoTheme = 'vs';
let monacoLoaderPromise = null;
let currentContent = '';
let placeholderCleared = false;

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function loadMonaco() {
  if (!monacoLoaderPromise) {
    monacoLoaderPromise = new Promise((resolve, reject) => {
      if (typeof require === 'undefined') {
        reject(new Error('Monaco loader unavailable (CDN blocked?)'));
        return;
      }
      require.config({ paths: { 'vs': 'https://cdn.jsdelivr.net/npm/monaco-editor@0.45.0/min/vs' } });
      require(['vs/editor/editor.main'], monaco => {
        monacoApi = monaco;
        resolve(monaco);
      }, reject);
    });
  }
  return monacoLoaderPromise;
}

async function getEditor() {
  if (editorInstance) return editorInstance;
  const monaco = await loadMonaco();
  const container = document.getElementById('editor-container');
  if (!placeholderCleared) {
    container.innerHTML = '';
    placeholderCleared = true;
  }
  editorInstance = monaco.editor.create(container, {
    value: '',
    language: 'xml',
    readOnly: true,
    wordWrap: 'on',
    minimap: { enabled: false },
    scrollBeyondLastLine: false,
    automaticLayout: true,
    fontSize: 13,
    folding: true,
  });
  monaco.editor.setTheme(monacoTheme);
  return editorInstance;
}

async function renderContent(content) {
  currentContent = (content || '').replace(/^\\n+/, '');
  try {
    const editor = await getEditor();
    editor.setValue(currentContent);
  } catch (err) {
    const viewer = document.getElementById('content-viewer');
    viewer.innerHTML = '';
    const pre = document.createElement('pre');
    pre.className = 'monaco-fallback';
    pre.textContent = currentContent;
    viewer.appendChild(pre);
  }
}

function toggleTheme() {
  monacoTheme = monacoTheme === 'vs' ? 'vs-dark' : 'vs';
  if (monacoApi) {
    monacoApi.editor.setTheme(monacoTheme);
  }
  document.body.classList.toggle('theme-dark', monacoTheme === 'vs-dark');
}

function calculateStats(data) {
  const totalLines = data.reduce((sum, t) => sum + t.lines, 0);
  const totalWords = data.reduce((sum, t) => sum + t.words, 0);
  const totalChars = data.reduce((sum, t) => sum + t.chars, 0);
  const totalSize = data.reduce((sum, t) => sum + t.size_bytes, 0);
  const totalTokens = data.reduce((sum, t) => sum + (t.tokens || t.words), 0);

  const avgLines = Math.round(totalLines / data.length);
  const avgWords = Math.round(totalWords / data.length);
  const avgChars = Math.round(totalChars / data.length);
  const avgTokens = Math.round(totalTokens / data.length);

  // Calculate parallel ratio stats
  const parallelRatios = data.filter(t => t.parallel_ratio !== null && t.parallel_ratio !== undefined).map(t => t.parallel_ratio);
  const avgParallelRatio = parallelRatios.length > 0 ? parallelRatios.reduce((sum, r) => sum + r, 0) / parallelRatios.length : null;

  // Calculate acceleration ratio stats
  const accelerationRatios = data.filter(t => t.acceleration_ratio !== null && t.acceleration_ratio !== undefined).map(t => t.acceleration_ratio);
  const avgAccelerationRatio = accelerationRatios.length > 0 ? accelerationRatios.reduce((sum, r) => sum + r, 0) / accelerationRatios.length : null;

  return {
    total: data.length,
    totalLines, totalWords, totalChars, totalSize, totalTokens,
    avgLines, avgWords, avgChars, avgTokens,
    maxLines: Math.max(...data.map(t => t.lines)),
    minLines: Math.min(...data.map(t => t.lines)),
    avgParallelRatio,
    avgAccelerationRatio
  };
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  return (bytes / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
}

function renderStatsOverview(stats) {
  const overview = document.getElementById('stats-overview');
  const cards = [
    `<div class="stat-card fade-in">
      <div class="stat-label">📊 Average Lines</div>
      <div class="stat-value">${fmt.format(stats.avgLines)}</div>
      <div class="stat-subtext">Min: ${fmt.format(stats.minLines)} • Max: ${fmt.format(stats.maxLines)}</div>
    </div>`,
    `<div class="stat-card fade-in">
      <div class="stat-label">🎯 Average Tokens</div>
      <div class="stat-value">${fmt.format(stats.avgTokens)}</div>
      <div class="stat-subtext">Total: ${fmt.format(stats.totalTokens)}</div>
    </div>`,
    `<div class="stat-card fade-in">
      <div class="stat-label">💾 Total Size</div>
      <div class="stat-value">${formatBytes(stats.totalSize)}</div>
      <div class="stat-subtext">Across ${stats.total} files</div>
    </div>`
  ];

  if (stats.avgParallelRatio !== null) {
    cards.push(`<div class="stat-card fade-in">
      <div class="stat-label">⚡ Avg Parallel Ratio</div>
      <div class="stat-value">${(stats.avgParallelRatio * 100).toFixed(1)}%</div>
      <div class="stat-subtext">Parallelization efficiency</div>
    </div>`);
  }

  if (stats.avgAccelerationRatio !== null) {
    cards.push(`<div class="stat-card fade-in">
      <div class="stat-label">🚀 Avg Acceleration Ratio</div>
      <div class="stat-value">${(stats.avgAccelerationRatio * 100).toFixed(1)}%</div>
      <div class="stat-subtext">Potential speedup from parallelization</div>
    </div>`);
  }

  overview.innerHTML = cards.join('');
}

function renderTrajectoryList(data, filter = '') {
  const list = document.getElementById('traj-list');
  const filtered = filter
    ? data.filter(t => t.name.toLowerCase().includes(filter.toLowerCase()))
    : data;

  list.innerHTML = filtered.map(t => `
    <div class="traj-item ${currentId === t.id ? 'active' : ''}" onclick="selectTrajectory(${t.id})" data-id="${t.id}">
      <div class="traj-name">#${t.id} — ${t.name}</div>
      <div class="traj-meta">
        <span>${fmt.format(t.lines)} lines</span>
        <span>${fmt.format(t.tokens || t.words)} tokens</span>
        <span>${t.size_pretty}</span>
      </div>
    </div>
  `).join('');
}

function analyzeContent(content) {
  return {
    hasParallel: content.includes('<Parallel>'),
    hasThink: content.includes('<think>') || content.includes('<Think>'),
    hasBoxed: content.includes('\\\\boxed'),
    hasOutlines: content.includes('<Outlines>'),
    hasThreads: (content.match(/<Thread>/g) || []).length,
    hasConclusion: content.includes('<Conclusion>')
  };
}

async function selectTrajectory(id) {
  currentId = id;
  const data = await fetchJSON(`/api/trajectory?id=${id}`);

  if (data.error) {
    renderContent(data.error);
    document.getElementById('content-title').textContent = 'Error';
    document.getElementById('content-tags').innerHTML = '';
    document.getElementById('content-metrics').innerHTML = '';
    document.getElementById('truncated-alert').style.display = 'none';
    document.getElementById('copy-btn').style.display = 'none';
    document.getElementById('highlight-btn').style.display = 'none';
    return;
  }

  // Update active state
  document.querySelectorAll('.traj-item').forEach(item => {
    item.classList.toggle('active', parseInt(item.dataset.id) === id);
  });

  // Analysis
  const analysis = analyzeContent(data.content);

  // Render tags
  const tags = [];
  if (analysis.hasParallel) tags.push('<div class="tag parallel">🔀 Parallel</div>');
  if (analysis.hasThink) tags.push('<div class="tag thinking">💭 Thinking</div>');
  if (analysis.hasBoxed) tags.push('<div class="tag boxed">📦 Boxed Answer</div>');
  if (analysis.hasOutlines) tags.push('<div class="tag outline">📋 Outlines</div>');
  if (analysis.hasThreads) tags.push(`<div class="tag outline">🧵 ${analysis.hasThreads} Thread${analysis.hasThreads > 1 ? 's' : ''}</div>`);

  document.getElementById('content-tags').innerHTML = tags.join('');

  // Helper function for metric with tooltip
  function metricWithTooltip(value, label, tooltip) {
    if (value === null || value === undefined) return '';
    const displayValue = typeof value === 'number' ? (Number.isInteger(value) ? fmt.format(value) : value.toFixed(2)) : value;
    return `
      <div class="metric">
        <div class="metric-value">${displayValue}</div>
        <div class="metric-label tooltip">${label}
          ${tooltip ? `<span class="tooltiptext">${tooltip}</span>` : ''}
        </div>
      </div>
    `;
  }

  // Render metrics with tooltips
  const metricsHtml = [
    metricWithTooltip(data.lines, 'Lines', 'Total number of lines in the file'),
    metricWithTooltip(data.tokens, 'Tokens', 'Total number of tokens (using Qwen3 tokenizer or word count as fallback)'),
    metricWithTooltip(data.chars, 'Characters', 'Total number of characters in the file'),
    metricWithTooltip(data.size_pretty, 'Size', 'File size on disk'),
  ];

  // Add detailed stats if available
  if (data.parallel_ratio !== null && data.parallel_ratio !== undefined) {
    metricsHtml.push(metricWithTooltip(
      data.parallel_ratio,
      'Parallel Ratio',
      'Ratio of tokens inside <Parallel> blocks to total tokens (higher = more parallelization)'
    ));
  }

  if (data.acceleration_ratio !== null && data.acceleration_ratio !== undefined) {
    metricsHtml.push(metricWithTooltip(
      data.acceleration_ratio,
      'Acceleration Ratio',
      'Potential speedup from parallelization: 1 - (longest_thread_tokens / total_tokens). 0 = no speedup (fully sequential), higher values = more potential parallelization speedup'
    ));
  }

  if (data.num_tokens_in_the_longest_thread !== null) {
    metricsHtml.push(metricWithTooltip(
      data.num_tokens_in_the_longest_thread,
      'Longest Thread',
      'Number of tokens in the longest sequential execution path through the parallel structure'
    ));
  }

  if (data.avg_thread_length !== null) {
    metricsHtml.push(metricWithTooltip(
      data.avg_thread_length,
      'Avg Thread Length',
      'Average number of tokens per thread across all parallel blocks'
    ));
  }

  if (data.avg_tokens_per_parallel_block !== null) {
    metricsHtml.push(metricWithTooltip(
      data.avg_tokens_per_parallel_block,
      'Avg Block Size',
      'Average number of tokens per <Parallel> block'
    ));
  }

  if (data.avg_threads_per_parallel_block !== null) {
    metricsHtml.push(metricWithTooltip(
      data.avg_threads_per_parallel_block,
      'Avg Threads/Block',
      'Average number of threads per <Parallel> block'
    ));
  }

  if (data.avg_outlines_length !== null) {
    metricsHtml.push(metricWithTooltip(
      data.avg_outlines_length,
      'Avg Outlines Length',
      'Average number of tokens in <Outlines> sections (the planning phase before threads)'
    ));
  }

  if (data.avg_conclusion_length !== null) {
    metricsHtml.push(metricWithTooltip(
      data.avg_conclusion_length,
      'Avg Conclusion Length',
      'Average number of tokens in <Conclusion> sections (synthesis after parallel threads)'
    ));
  }

  document.getElementById('content-metrics').innerHTML = metricsHtml.join('');

  // Update content
  document.getElementById('content-title').textContent = data.name;
  renderContent(data.content);

  document.getElementById('truncated-alert').style.display = data.truncated ? 'block' : 'none';
  document.getElementById('copy-btn').style.display = 'block';
  const themeBtn = document.getElementById('highlight-btn');
  themeBtn.style.display = 'block';
  themeBtn.textContent = monacoTheme === 'vs' ? '🌙 Dark' : '☀️ Light';
  if (monacoTheme === 'vs-dark') {
    document.body.classList.add('theme-dark');
  } else {
    document.body.classList.remove('theme-dark');
  }
}

// Event listeners
document.getElementById('search').addEventListener('input', (e) => {
  renderTrajectoryList(trajectories, e.target.value);
});

document.getElementById('copy-btn').addEventListener('click', () => {
  navigator.clipboard.writeText(currentContent).then(() => {
    const btn = document.getElementById('copy-btn');
    btn.textContent = '✓ Copied!';
    setTimeout(() => btn.textContent = '📋 Copy', 2000);
  });
});

document.getElementById('highlight-btn').addEventListener('click', () => {
  const btn = document.getElementById('highlight-btn');
  toggleTheme();
  btn.textContent = monacoTheme === 'vs' ? '🌙 Dark' : '☀️ Light';
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowDown' && currentId !== null && currentId < trajectories.length - 1) {
    e.preventDefault();
    selectTrajectory(currentId + 1);
    document.querySelector(`[data-id="${currentId + 1}"]`)?.scrollIntoView({ block: 'nearest' });
  }
  if (e.key === 'ArrowUp' && currentId !== null && currentId > 0) {
    e.preventDefault();
    selectTrajectory(currentId - 1);
    document.querySelector(`[data-id="${currentId - 1}"]`)?.scrollIntoView({ block: 'nearest' });
  }
  if (e.key === '/' && document.activeElement.id !== 'search') {
    e.preventDefault();
    document.getElementById('search').focus();
  }
});

async function init() {
  const meta = await fetchJSON('/api/trajectories');
  trajectories = meta.stats;

  document.getElementById('total-count').textContent = meta.count;

  const stats = calculateStats(trajectories);
  renderStatsOverview(stats);
  renderTrajectoryList(trajectories);

  if (trajectories.length) selectTrajectory(trajectories[0].id);
}

init().catch(err => {
  renderContent(`Failed to load: ${err.message}`);
});
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Serve a trajectory browser UI.")
    parser.add_argument("--root", type=str, required=True, help="Directory containing trajectory .txt files (e.g., step6.1 output).")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind. Defaults to 127.0.0.1 for safety.")
    parser.add_argument("--port", type=int, default=8899, help="Port to serve on.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Root does not exist or is not a directory: {root}")

    trajectories = scan(root)
    if not trajectories:
        raise SystemExit(f"No .txt trajectories found under {root}")

    class _Server(ThreadingHTTPServer):
        daemon_threads = True

    server = _Server((args.host, args.port), TrajectoryHandler)
    server.root = root
    server.trajectories = trajectories

    print(f"Serving {len(trajectories)} trajectories from {root}")
    print(f"Open http://{args.host}:{args.port} in your browser.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
