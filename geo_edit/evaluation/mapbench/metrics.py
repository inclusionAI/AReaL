from __future__ import annotations

import json
import math
import re
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_PATTERN_WITH_DIR = re.compile(r"^(.*?) -> (.*?) \((.*?)\)$")
_PATTERN_NO_DIR = re.compile(r"(?:\d+\.\s)?(.*?) -> ([^(]*)")

FAILURE_EMPTY = -1
FAILURE_INVALID_LANDMARK = -2
FAILURE_DISCONTINUITY = -3
FAILURE_PARSE = -4


def load_graph_from_json(graph_json: str) -> nx.Graph:
    data = json.loads(graph_json)
    if "edges" in data and "links" not in data:
        data["links"] = data.pop("edges")
    G = nx.node_link_graph(data)
    converted = nx.Graph()
    for node, attrs in G.nodes(data=True):
        if isinstance(node, list):
            node = tuple(node)
        converted.add_node(node, **attrs)
    for u, v, attrs in G.edges(data=True):
        if isinstance(u, list):
            u = tuple(u)
        if isinstance(v, list):
            v = tuple(v)
        converted.add_edge(u, v, **attrs)
    return converted


def get_landmarks(G: nx.Graph) -> List[str]:
    skip = {"Intersections", "Adjacent"}
    return [
        data["label"]
        for _, data in G.nodes(data=True)
        if "label" in data and data["label"] not in skip
    ]


def match_landmark(
    G: nx.Graph,
    name: str,
    landmark_list: List[str],
    embeddings_cache: Dict[str, np.ndarray],
    model: SentenceTransformer,
) -> str:
    if name in embeddings_cache:
        name_emb = embeddings_cache[name]
    else:
        name_emb = model.encode([name])[0]
        embeddings_cache[name] = name_emb

    best_score = -1.0
    best_match = landmark_list[0] if landmark_list else name
    for lm in landmark_list:
        if lm in embeddings_cache:
            lm_emb = embeddings_cache[lm]
        else:
            lm_emb = model.encode([lm])[0]
            embeddings_cache[lm] = lm_emb
        score = cosine_similarity([name_emb], [lm_emb])[0][0]
        if score > best_score:
            best_score = score
            best_match = lm
    return best_match


def find_node_by_label(G: nx.Graph, label: str):
    for node, data in G.nodes(data=True):
        if data.get("label") == label:
            return node
    return None


def _euclidean_path_distance(path: list) -> float:
    dist = 0.0
    for i in range(len(path) - 1):
        dx = path[i][0] - path[i + 1][0]
        dy = path[i][1] - path[i + 1][1]
        dist += math.sqrt(dx * dx + dy * dy)
    return dist


def path_eval(
    G: nx.Graph,
    nav_steps: List[str],
    start: str,
    end: str,
    model: SentenceTransformer,
) -> Tuple[int, float]:
    landmark_list = get_landmarks(G)
    embeddings_cache: Dict[str, np.ndarray] = {}

    start = match_landmark(G, start, landmark_list, embeddings_cache, model)
    end = match_landmark(G, end, landmark_list, embeddings_cache, model)

    start_node = find_node_by_label(G, start)
    end_node = find_node_by_label(G, end)
    if start_node is None or end_node is None:
        return FAILURE_INVALID_LANDMARK, 0.0

    try:
        gt_path = nx.shortest_path(G, source=start_node, target=end_node)
    except nx.NetworkXNoPath:
        return FAILURE_INVALID_LANDMARK, 0.0

    gt_dis = _euclidean_path_distance(gt_path)
    if gt_dis == 0:
        return FAILURE_INVALID_LANDMARK, 0.0

    if not nav_steps:
        return FAILURE_EMPTY, 0.0

    path_dis = 0.0
    prev = start

    for cnt, step in enumerate(nav_steps):
        if "(" in step and ")" not in step:
            step = step + ")"

        m = _PATTERN_WITH_DIR.match(step)
        if m is None:
            m = _PATTERN_NO_DIR.match(step)
        if m is None:
            return FAILURE_PARSE, 0.0

        lm1_raw, lm2_raw = m.group(1).strip(), m.group(2).strip()
        lm1 = match_landmark(G, lm1_raw, landmark_list, embeddings_cache, model)
        lm2 = match_landmark(G, lm2_raw, landmark_list, embeddings_cache, model)

        if lm1 not in landmark_list or lm2 not in landmark_list:
            return FAILURE_INVALID_LANDMARK, 0.0

        if prev != lm1:
            return FAILURE_DISCONTINUITY, 0.0
        prev = lm2

        if cnt == 0 and lm1 != start:
            return FAILURE_DISCONTINUITY, 0.0
        if cnt == len(nav_steps) - 1 and lm2 != end:
            return FAILURE_DISCONTINUITY, 0.0

        node1 = find_node_by_label(G, lm1)
        node2 = find_node_by_label(G, lm2)
        if node1 is None or node2 is None:
            return FAILURE_INVALID_LANDMARK, 0.0

        try:
            seg_path = nx.shortest_path(G, source=node1, target=node2)
        except nx.NetworkXNoPath:
            return FAILURE_DISCONTINUITY, 0.0

        path_dis += _euclidean_path_distance(seg_path)

    path_score = path_dis / gt_dis
    return 1, path_score
