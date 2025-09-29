#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Werewolf Agent Trajectory Analyzer (dedupe-enabled)
---------------------------------------------------
Enhancements vs previous version:
- Avoid repeated/similar examples within the same markdown report:
  * Global "seen" tracking so the same snippet won't appear in multiple buckets
  * Similarity-based de-duplication with TF-IDF + cosine threshold per bucket/cluster
- Command-line toggles: --sim-threshold, --global-no-repeat

Usage:
python analyser.py --glob "*.txt" --sim-threshold 0.8 \
    --global-no-repeat --report_root report14-wt7b-vabl \
    --root /storage/openpsi/experiments/logs/admin/xmy-werewolf-comp-14/werewolf-t7b-vs4-villager-abl/generated \
    --abandon-werewolf-thought

Options:
--global-no-repeat
--abandon-werewolf-thought

Outputs:
- data/thoughts.csv, data/interesting_thoughts.csv, data/keyword_counts.csv
- plots/*.png
- report/InterestingThoughts.md
"""
from __future__ import annotations

import argparse
import re
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Tuple
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

THOUGHT_HINTS = [
    # r"\b(?:thought|thinking|internal|private|c.?o.?t|chain[- ]?of[- ]?thought|monologue)\b",
    # r"\b(?:reasoning|rationale|deliberation|inner voice|plan)\b",
    r"->",
    r"<(?:think|think?|answer|answer?)>",
    # r"\((?:thinking|internal|private)\)",
]

STRATEGY_KEYWORDS = sorted(set([
    "accuse","defend","alibi","vote","voting","wagon","hammer","lynch","eliminate","execute","flip",
    "town","villager","wolf","werewolf","seer","doctor","witch","hunter","mafia","cop","medic",
    "claim","counterclaim","fakeclaim","softclaim","hardclaim","clear","soft-clear","townread",
    "scumread","nullread","tmi","wifom","omgus","bus","busser","bussing","pocket","sheep","sheeping",
    "tunnel","iso",
    "day","night","d1","d2","n1","n2","lylo","mylo","poe","pool",
    "frame","frames","framing","blame","pressure","bait","gambit","double bluff","double-bluff","bluff",
    "distancing","angle shoot","angleshoot","angleshooting","spew","spewing","spewed",
    "solve","solver","solve path","solve-path","solveplan","solve plan",
    "mechanics","mech","vt","pr","pr-claim",
    "plan","strategy","tactic","route","line","lines","approach","case","cases","agenda","wincon","win condition",
    "protect","guard","kill","nk","nightkill","bus drive","redirect","poison","track","watch","peek","check",
    "probability","odds","likelihood","confidence","uncertain","certain","maybe","perhaps","likely","unlikely",
    "because","therefore","hence","so that","if","then","else",
]))

HEDGE_TERMS = [
    "maybe","perhaps","probably","possibly","unsure","uncertain","not sure","i think","i guess","i feel",
    "lean","leans","seems","appears","could be","might be"
]

CONFIDENCE_TERMS = [
    "certain","sure","confident","confirmed","definitely","absolutely","guaranteed","locked","snap","bank it"
]

DECEPTION_TERMS = [
    "fake","lying","lie","lied","bluff","double bluff","pretend","deceive","mislead","distort","frame","bus","distancing"
]

ALLIANCE_TERMS = [
    "ally","alliance","teammate","partner","team","pair","duo","trio","pocket","buddy","sheep","locktown"
]

NUMERIC_PATTERN = re.compile(r"\b(?:\d{1,3}(?:\.\d+)?\s*%|\b(?:0?\.\d+|[01])\b)\b")

SPEAKER_RE = re.compile(r"^player\d+\b")
ROLE_HINT_RE = re.compile(r"\b(villager|werewolf|wolf|foreseer|doctor|witch|hunter|mafia|cop|medic|pr|vt)\b", re.I)
TURN_HINT_RE = re.compile(r"\b(?:day|night)\s*(\d+)|\bD(\d+)\b|\bN(\d+)\b", re.I)

@dataclass
class Thought:
    file: str
    speaker: Optional[str]
    role_hint: Optional[str]
    turn: Optional[str]
    text: str
    line_no: int

def find_turn(text: str) -> Optional[str]:
    m = TURN_HINT_RE.search(text)
    if not m:
        return None
    g = m.groups()
    if g[0]:
        return m.group(0).strip()
    if g[1]:
        return f"D{g[1]}"
    if g[2]:
        return f"N{g[2]}"
    return m.group(0).strip()

def is_thought_line(line: str) -> bool:
    l = line.lower()
    return any(re.search(pat, l) for pat in THOUGHT_HINTS)

def extract_speaker_role(line: str) -> Tuple[Optional[str], Optional[str]]:
    m = SPEAKER_RE.match(line.strip())
    speaker = None
    role_hint = None
    if m:
        speaker = m.group(0) or m.group(2)
        if len(m.groups()) > 1:
            rh = ROLE_HINT_RE.search(m.group(1))
            role_hint = rh.group(0).lower() if rh else None
    if not role_hint:
        rh2 = ROLE_HINT_RE.search(line)
        if rh2:
            role_hint = rh2.group(0).lower()
    return speaker, role_hint

def clean_text(s: str) -> str:
    s = re.sub(r"^\s*(\[/?(THINK|INTERNAL|PRIVATE|THOUGHTS?)\]|\(.*?(thinking|internal|private).*?\))[:\- ]*\s*", "", s, flags=re.I)
    s = re.sub(r"^\s*(Thoughts?|Thinking|Internal|Private)[:\- ]+\s*", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def slice_block(lines: List[str], start_idx: int, max_len: int = 6) -> str:
    out = [clean_text(lines[start_idx])]
    for i in range(start_idx + 1, min(len(lines), start_idx + 1 + max_len)):
        if SPEAKER_RE.match(lines[i].strip()):
            break
        if lines[i].strip() == "" and len(out) > 0:
            break
        out.append(clean_text(lines[i]))
    return " ".join(out).strip()

def harvest_thoughts(path: Path, encoding: str, min_len: int, max_snippets: int, abandon_werewolf_thought: bool=False) -> List[Thought]:
    raw = path.read_text(encoding=encoding, errors="ignore")
    lines = raw.splitlines()

    traj_idx = None
    for i in range(len(lines)-1, -1, -1):
        if "Trajectory:" in lines[i]:
            traj_idx = i
            break

    if traj_idx is not None:
        lines = lines[traj_idx + 1:]
    else:
        return []

    thoughts: List[Thought] = []
    for idx, line in enumerate(lines):
        if is_thought_line(line):
            speaker, role_hint = extract_speaker_role(line)
            turn = find_turn(line)
            text = slice_block(lines, idx)
            if len(text) >= min_len:
                if abandon_werewolf_thought and role_hint == "werewolf":
                    continue
                thoughts.append(Thought(path.name, speaker, role_hint, turn, text, idx + 1))
                if len(thoughts) >= max_snippets:
                    break
    if not thoughts:
        for idx, line in enumerate(lines):
            if SPEAKER_RE.match(line) and any(kw in line.lower() for kw in ["plan","strategy","probability","vote","accuse","defend","claim","bluff"]):
                speaker, role_hint = extract_speaker_role(line)
                turn = find_turn(line)
                text = slice_block(lines, idx)
                if len(text) >= min_len:
                    if abandon_werewolf_thought and role_hint == "werewolf":
                        continue
                    thoughts.append(Thought(path.name, speaker, role_hint, turn, text, idx + 1))
                    if len(thoughts) >= max_snippets:
                        break
    return thoughts

def score_interest(row: pd.Series) -> Dict[str, float]:
    t = row["text"].lower()
    scores = {
        "has_strategy_kw": float(any(kw in t for kw in STRATEGY_KEYWORDS)),
        "hedge_density": sum(t.count(h) for h in HEDGE_TERMS) / max(1, len(t.split())),
        "confidence_hits": sum(t.count(c) for c in CONFIDENCE_TERMS),
        "deception_hits": sum(t.count(d) for d in DECEPTION_TERMS),
        "alliance_hits": sum(t.count(a) for a in ALLIANCE_TERMS),
        "numeric_probs": float(bool(NUMERIC_PATTERN.search(t))),
        "length": len(t),
        "if_then": float((" if " in t and " then " in t) or ("if" in t.split() and "then" in t.split())),
        "because": float("because" in t),
        "therefore": float("therefore" in t or "hence" in t),
    }
    score = (
        1.2 * scores["has_strategy_kw"]
        + 0.8 * scores["numeric_probs"]
        + 1.0 * scores["deception_hits"]
        + 0.6 * scores["alliance_hits"]
        + 0.5 * scores["confidence_hits"]
        + 0.7 * scores["if_then"]
        + 0.5 * scores["because"]
        + 0.3 * scores["therefore"]
        + 0.0005 * scores["length"]
        - 0.7 * scores["hedge_density"]
    )
    scores["interest_score"] = score
    return scores

def cluster_topics(texts: List[str], k: int, random_state: int = 42) -> Tuple[np.ndarray, List[List[Tuple[str,float]]]]:
    if len(texts) < k:
        k = max(2, min(len(texts), k))
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=5000)
    X = vect.fit_transform(texts)
    model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = model.fit_predict(X)
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = np.array(vect.get_feature_names_out())
    top_terms = []
    for i in range(k):
        inds = order_centroids[i, :15]
        top_terms.append([(terms[j], float(model.cluster_centers_[i, j])) for j in inds])
    return labels, top_terms

# ------------------------------
# De-duplication utilities
# ------------------------------

def normalize_sig(text: str) -> str:
    t = re.sub(r"\s+", " ", text.strip().lower())
    t = re.sub(r"[^\w\s%]", "", t)  # remove punctuation but keep %
    return t

def select_diverse(records: List[dict], text_key: str, topn: int, sim_threshold: float = 0.85) -> List[dict]:
    """Greedy selection of top-N diverse items by 'interest_score' with cosine similarity de-duplication."""
    if not records:
        return []
    pool = sorted(records, key=lambda r: (-float(r.get("interest_score", 0.0)), len(r.get(text_key,""))))
    selected: List[dict] = []
    texts = [r[text_key] for r in pool]
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=5000)
    X = vect.fit_transform(texts)
    chosen_idx: List[int] = []
    for idx, r in enumerate(pool):
        if len(selected) >= topn:
            break
        if not chosen_idx:
            selected.append(r); chosen_idx.append(idx); continue
        sims = cosine_similarity(X[idx], X[chosen_idx]).ravel()
        if sims.max() < sim_threshold:
            selected.append(r); chosen_idx.append(idx)
    # if we still have fewer than topn, fill with least-similar leftovers
    if len(selected) < topn:
        for idx, r in enumerate(pool):
            if len(selected) >= topn or idx in chosen_idx: 
                continue
            sims = cosine_similarity(X[idx], X[chosen_idx]).ravel() if chosen_idx else np.array([0.0])
            if sims.mean() < sim_threshold:  # a bit looser
                selected.append(r); chosen_idx.append(idx)
    return selected[:topn]

def write_report(df_int: pd.DataFrame, cluster_info: Dict[int, Dict], topn_per_bucket: int, out_md: Path,
                 sim_threshold: float = 0.85, global_no_repeat: bool = True) -> None:
    lines = []
    lines.append("# Interesting Werewolf Thoughts — Auto Report\n")
    lines.append(f"Total interesting snippets: **{len(df_int)}**\n")

    # global "seen" tracking to avoid repeating the same example across buckets
    seen: set[str] = set()

    def not_seen_filter(df: pd.DataFrame) -> pd.DataFrame:
        if not global_no_repeat:
            return df
        mask = ~df["text"].apply(lambda t: normalize_sig(t) in seen)
        return df[mask]

    buckets = {
        "Deception / Bluffing": df_int[df_int["deception_hits"] > 0],
        "Alliance / Teaming": df_int[df_int["alliance_hits"] > 0],
        "Probability / Mathy": df_int[df_int["numeric_probs"] > 0],
        "If-Then / Causal Reasoning": df_int[(df_int["if_then"] > 0) | (df_int["because"] > 0) | (df_int["therefore"] > 0)],
        "High-Confidence Assertions": df_int[df_int["confidence_hits"] > 0],
        "Strategy Keywords (general)": df_int[df_int["has_strategy_kw"] > 0],
    }

    for title, sub in buckets.items():
        sub = not_seen_filter(sub)
        if sub.empty:
            continue
        lines.append(f"\n## {title}\n")
        # choose diverse examples
        candidates = sub.sort_values("interest_score", ascending=False).to_dict(orient="records")
        diverse = select_diverse(candidates, "text", min(topn_per_bucket, len(candidates)), sim_threshold=sim_threshold)
        for r in diverse:
            sig = normalize_sig(r["text"])
            seen.add(sig)
            meta = f"*{r['file']}* | speaker={r.get('speaker') or '?'} | role={r.get('role_hint') or '?'} | turn={r.get('turn') or '?'} | score={r['interest_score']:.2f}"
            lines.append(f"- {meta}\n  \n  > {r['text']}\n")

    lines.append("\n## Discovered Topic Clusters\n")
    for cid, info in cluster_info.items():
        terms = ", ".join([t for t,_ in info.get("top_terms", [])[:10]])
        lines.append(f"### Cluster {cid}: {terms}\n")
        examples = info.get("examples", [])
        # filter out globally seen examples, then diversify again
        ex_df = pd.DataFrame(examples)
        if not ex_df.empty:
            ex_df = not_seen_filter(ex_df)
            examples = ex_df.sort_values("interest_score", ascending=False).to_dict(orient="records")
            examples = select_diverse(examples, "text", min(6, len(examples)), sim_threshold=sim_threshold)
        for r in examples:
            sig = normalize_sig(r["text"])
            seen.add(sig)
            meta = f"*{r['file']}* | speaker={r.get('speaker') or '?'} | role≈{r.get('role_hint') or '?'} | turn={r.get('turn') or '?'} | score={r['interest_score']:.2f}"
            lines.append(f"- {meta}\n  \n  > {r['text']}\n")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")

def make_charts(df: pd.DataFrame, outdir: Path) -> None:
    import matplotlib.pyplot as plt
    outdir.mkdir(parents=True, exist_ok=True)

    counts = {}
    for kw in STRATEGY_KEYWORDS:
        counts[kw] = df["text"].str.lower().str.contains(rf"\b{re.escape(kw)}\b", regex=True).sum()
    kc = pd.Series(counts).sort_values(ascending=False).head(20)
    kc.to_csv(outdir / "top_strategy_keywords.csv")

    plt.figure()
    kc.plot(kind="bar")
    plt.title("Top Strategy Keywords (top 20)")
    plt.tight_layout()
    plt.savefig(outdir / "top_strategy_keywords.png", dpi=160)
    plt.close()

    plt.figure()
    df["interest_score"].plot(kind="hist", bins=30)
    plt.title("Interest Score Distribution")
    plt.tight_layout()
    plt.savefig(outdir / "interest_score_hist.png", dpi=160)
    plt.close()

    plt.figure()
    plt.scatter(df["hedge_density"], df["confidence_hits"] + df["deception_hits"] + df["alliance_hits"], alpha=0.5)
    plt.xlabel("Hedge density")
    plt.ylabel("Confidence/Deception/Alliance hits")
    plt.title("Hedging vs. Strong Claims")
    plt.tight_layout()
    plt.savefig(outdir / "hedge_vs_claims_scatter.png", dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Folder containing *.txt trajectories")
    ap.add_argument("--report_root", type=str, default="report", help="Folder to place analyzations.")
    ap.add_argument("--glob", type=str, default="*.txt", help="Filename glob")
    ap.add_argument("--encoding", type=str, default="utf-8")
    ap.add_argument("--max-snippets", type=int, default=500)
    ap.add_argument("--global-max-snippets", type=int, default=50000)
    ap.add_argument("--min-thought-len", type=int, default=10)
    ap.add_argument("--clusters", type=int, default=8)
    ap.add_argument("--sample-per-interest", type=int, default=8)
    ap.add_argument("--sim-threshold", type=float, default=0.85, help="Cosine similarity threshold for dedup (lower = more diverse)")
    ap.add_argument("--global-no-repeat", action="store_true", help="If set, never repeat the same snippet across buckets/clusters")
    ap.add_argument("--abandon-werewolf-thought", action="store_true", help="If set, never analyse werewolf thoughts")
    args = ap.parse_args()

    root = Path(args.root)
    files = list(root.glob(f"**/{args.glob}"))
    random.shuffle(files)
    if not files:
        raise SystemExit(f"No files matching {args.glob!r} found in {root.resolve()}")

    all_thoughts: list[dict] = []
    for p in files:
        thoughts = harvest_thoughts(p, args.encoding, args.min_thought_len, args.max_snippets, args.abandon_werewolf_thought)
        all_thoughts.extend(asdict(t) for t in thoughts)
        print(f"Harvested {len(thoughts)} thoughts from {p}!", flush=True)
        if len(all_thoughts) >= args.global_max_snippets:
            break

    if not all_thoughts:
        raise SystemExit("No thought snippets were extracted. Consider tuning THOUGHT_HINTS or lowering --min-thought-len.")

    print("Analysing...")

    df = pd.DataFrame(all_thoughts)
    df.drop_duplicates(subset=["file","line_no","text"], inplace=True)

    scores = df.apply(score_interest, axis=1, result_type="expand")
    df = pd.concat([df, scores], axis=1)

    report_root = args.report_root

    out_report = Path(f"{report_root}")
    out_data = Path(f"{report_root}/data")
    out_plots = Path(f"{report_root}/plots")
    out_report.mkdir(exist_ok=True)
    out_data.mkdir(exist_ok=True)
    out_plots.mkdir(exist_ok=True)
    df.to_csv(out_data / "thoughts.csv", index=False)

    thr = max(df["interest_score"].quantile(0.75), 1.5)
    df_int = df[df["interest_score"] >= thr].copy()
    df_int.to_csv(out_data / "interesting_thoughts.csv", index=False)

    labels = None
    cluster_info: Dict[int, Dict] = {}
    try:
        labels, top_terms = cluster_topics(df_int["text"].tolist(), args.clusters)
        df_int["cluster"] = labels
        for cid in sorted(set(labels)):
            sub = df_int[df_int["cluster"] == cid].copy()
            examples = sub.sort_values("interest_score", ascending=False).head(12).to_dict(orient="records")
            examples = select_diverse(examples, "text", min(6, len(examples)), sim_threshold=args.sim_threshold)
            cluster_info[cid] = {"top_terms": top_terms[cid], "examples": examples}
    except Exception:
        cluster_info = {}

    # keyword counts
    def contains_kw(text: str, kw: str) -> bool:
        return re.search(rf"\b{re.escape(kw)}\b", text.lower()) is not None

    kw_counts = []
    for kw in STRATEGY_KEYWORDS:
        count = int(df["text"].apply(lambda s: contains_kw(s, kw)).sum())
        kw_counts.append({"keyword": kw, "count": count})
    kw_df = pd.DataFrame(kw_counts).sort_values("count", ascending=False)
    kw_df.to_csv(out_data / "keyword_counts.csv", index=False)

    make_charts(df, out_plots)

    write_report(
        df_int, cluster_info, args.sample_per_interest,
        out_report / "InterestingThoughts.md",
        sim_threshold=args.sim_threshold,
        global_no_repeat=args.global_no_repeat
    )

    print("\n✔ Done.\n")
    print(f"- Saved all thoughts to: {out_data / 'thoughts.csv'}")
    print(f"- Saved interesting thoughts to: {out_data / 'interesting_thoughts.csv'} (threshold >= {thr:.2f})")
    if cluster_info:
        print(f"- Topic clusters discovered: {len(cluster_info)}")
    else:
        print("- Topic clustering skipped or failed (insufficient data).")
    print(f"- Charts in: {out_plots.resolve()}")
    print(f"- Report: {out_report / 'InterestingThoughts.md'}")

if __name__ == "__main__":
    main()