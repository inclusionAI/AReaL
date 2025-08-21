#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Literal, Optional

TrainingType = Literal["SFT", "RL"]

# --- Language filters ---
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")  # Chinese chars

def contains_chinese(s: str) -> bool:
    return bool(s) and bool(CHINESE_RE.search(s))

# --- Topic filters (simple heuristics) ---
# Strong off-topic keywords (sports & generic newsy terms that often leak in)
OFFTOPIC_RE = re.compile(
    r"\b("
    r"super\s*bowl|nfl|nba|mlb|nhl|ncaa|premier\s*league|fifa|uefa|world\s*cup|olympics|cricket|baseball|basketball|soccer|hockey|tennis|golf|"
    r"jaguars|raiders|patriots|cowboys|lakers|yankees|warriors|man\s*city|man\s*united|real\s*madrid|barcelona"
    r")\b",
    re.IGNORECASE,
)

# Werewolf/mafia-style gameplay vocab (loose and inclusive)
WEREWOLF_RE = re.compile(
    r"\b("
    r"werewolf|wolf|wolves|villager|town|seer|witch|hunter|guardian|bodyguard|doctor|priest|"
    r"vote|voting|lynch|eliminate|elim|hammer|wagon|accuse|alibi|reads?|sus(picious)?|scum|"
    r"flip|reve(al|al)|investigate|peek|night\s*kill|nk|"
    r"night|day|d\d+|n\d+|day\s*\d+|night\s*\d+|"
    r"role|role\s*claim|rc|counter\s*claim|cc|claim|"
    r"alive\s*players|host|gm|game\s*state|eod|sod"
    r")\b",
    re.IGNORECASE,
)

def is_on_topic(question: Optional[str], answer: Optional[str], role: Optional[str]) -> bool:
    """Return True if the QA looks like a werewolf/mafia gameplay exchange."""
    q = question or ""
    a = answer or ""
    r = role or ""
    text = f"{q}\n{a}\n{r}"

    # If it matches strong off-topic cues → reject (e.g., your Jaguars/Raiders example)
    if OFFTOPIC_RE.search(text):
        return False

    # Must show at least some werewolf/mafia vocabulary across question/answer/role
    if not WEREWOLF_RE.search(text):
        return False

    return True

# --- File helpers ---
def _is_student_file(p: Path) -> bool:
    return p.name.endswith("_qalogs.json")

def _is_teacher_file(p: Path) -> bool:
    return p.name.endswith("_tlogs.json")

def _episode_id_from_dir(p: Path) -> Optional[int]:
    try:
        return int(p.parent.name)
    except (ValueError, TypeError):
        return None

def _safe_list(v):
    return v if isinstance(v, list) else []

# --- Emission ---
def _emit_qa_items(
    entry: Dict,
    source: str,
    episode_id: Optional[int],
    file_stem: str,
    training_type: TrainingType,
    include_summaries: bool,
    enable_topic_filter: bool,
) -> List[Dict]:
    out = []
    agent = entry.get("agent")
    role = entry.get("role")
    qas = _safe_list(entry.get("QAs"))

    # Core QAs
    for i, qa in enumerate(qas):
        q, a = qa.get("question"), qa.get("answer")
        if not isinstance(q, str) or not isinstance(a, str):
            continue
        # Chinese filter
        if contains_chinese(q) or contains_chinese(a):
            continue
        # Topic filter
        if enable_topic_filter and not is_on_topic(q, a, role):
            continue
        out.append({
            "training_type": training_type,
            "source": source,                  # "teacher" or "student"
            "episode": episode_id,             # numeric folder if parseable
            "file": file_stem,                 # filename stem
            "agent": agent,
            "role": role,
            "turn_index": i,
            "question": q,
            "answer": a,
        })

    # Optional summarization as synthetic QA
    if include_summaries:
        sp, sm = entry.get("summary_prompt"), entry.get("summary")
        if isinstance(sp, str) and isinstance(sm, str):
            if not (contains_chinese(sp) or contains_chinese(sm)):
                if not enable_topic_filter or is_on_topic(sp, sm, role):
                    out.append({
                        "training_type": training_type,
                        "source": source,
                        "episode": episode_id,
                        "file": file_stem,
                        "agent": agent,
                        "role": role,
                        "turn_index": None,   # summaries aren't in main QAs index
                        "question": sp,
                        "answer": sm,
                        "is_summary": True,
                    })
    return out

# --- Loader ---
def load_qa_dataset(
    logs_root: str | Path,
    training_type: TrainingType = "SFT",
    include_summaries: bool = False,
    enable_topic_filter: bool = True,
) -> List[Dict]:
    """
    Build a flat Q&A dataset with keys:
      training_type, source, episode, file, agent, role, turn_index, question, answer, [is_summary]
    """
    root = Path(logs_root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Logs root not found or not a directory: {logs_root}")

    dataset: List[Dict] = []

    numbered_dirs = [p for p in root.iterdir() if p.is_dir()]
    def _episode_sort_key(p: Path):
        try:
            return (0, int(p.name))
        except ValueError:
            return (1, p.name)
    numbered_dirs.sort(key=_episode_sort_key)

    for d in numbered_dirs:
        for p in sorted(d.glob("*.json")):
            # SFT → teacher only; RL → teacher + student
            if training_type == "SFT":
                if not _is_teacher_file(p):
                    continue
            else:  # RL
                if not (_is_student_file(p) or _is_teacher_file(p)):
                    continue

            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[WARN] Skipping bad JSON {p}: {e}")
                continue

            if not isinstance(data, list):
                continue

            source = "teacher" if _is_teacher_file(p) else "student"
            episode_id = _episode_id_from_dir(p)
            file_stem = p.stem

            for entry in data:
                if isinstance(entry, dict):
                    dataset.extend(_emit_qa_items(
                        entry=entry,
                        source=source,
                        episode_id=episode_id,
                        file_stem=file_stem,
                        training_type=training_type,
                        include_summaries=include_summaries,
                        enable_topic_filter=enable_topic_filter,
                    ))

    # Stable ordering
    dataset.sort(key=lambda item: (
        item.get("episode"),
        item.get("file"),
        0 if item.get("source") == "teacher" else 1,
        item.get("turn_index") if isinstance(item.get("turn_index"), int) else 999999
    ))
    return dataset

# --- Writer ---
def _write_jsonl(path: Path, rows: List[Dict], limit: Optional[int] = None):
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            if limit is not None and count >= limit:
                break
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            count += 1

# --- CLI ---
def main():
    ap = argparse.ArgumentParser(description="Build a Q&A dataset from werewolf logs")
    ap.add_argument("logs_root", type=str, help="Root path to logs directory")
    ap.add_argument("--training-type", choices=["SFT", "RL"], default="SFT",
                    help="SFT: teacher-only; RL: teacher+student")
    ap.add_argument("--include-summaries", action="store_true",
                    help="Include (summary_prompt, summary) as synthetic QA items")
    ap.add_argument("--no-topic-filter", action="store_true",
                    help="Disable werewolf-topic filtering (keeps Chinese filter)")
    ap.add_argument("--out", type=str, default="",
                    help="Write JSONL to this path; if omitted, prints a preview")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of lines written to JSONL")
    args = ap.parse_args()

    ds = load_qa_dataset(
        logs_root=args.logs_root,
        training_type=args.training_type,
        include_summaries=args.include_summaries,
        enable_topic_filter=not args.no_topic_filter,
    )
    print(f"Collected {len(ds)} examples after filters.")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(out_path, ds, limit=args.limit)
        print(f"Wrote JSONL to: {out_path.resolve()} "
              f"({args.limit if args.limit is not None else len(ds)} lines max)")
    else:
        preview = ds[: min(5, len(ds))]
        print(json.dumps(preview, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
