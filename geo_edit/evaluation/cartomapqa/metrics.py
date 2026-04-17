"""Paper-exact metric functions for CartoMapQA evaluation.

Ported from CartoMapQA/eval.py to match the exact metrics reported in the paper.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


# ---------------------------------------------------------------------------
# Regression metrics (STMF-counting, RLE, MTMF-counting)
# ---------------------------------------------------------------------------


def regression_metrics(
    true: Sequence[float],
    pred: Sequence[float],
) -> Dict[str, float]:
    if not true:
        return {"rmse": 0.0, "mae": 0.0, "mape": 0.0, "r2": 0.0}
    return {
        "rmse": math.sqrt(mean_squared_error(true, pred)),
        "mae": mean_absolute_error(true, pred),
        "mape": mean_absolute_percentage_error(true, pred),
        "r2": r2_score(true, pred),
    }


# ---------------------------------------------------------------------------
# Binary classification metrics (STMF-presence)
# ---------------------------------------------------------------------------


def binary_prf1(
    true: Sequence[str],
    pred: Sequence[str],
) -> Dict[str, float]:
    label_map = {"yes": 1, "no": 0}
    true_num = [label_map[t] for t in true if t in label_map]
    pred_num = [label_map[p] for p in pred if p in label_map]
    n = min(len(true_num), len(pred_num))
    if n == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    true_num, pred_num = true_num[:n], pred_num[:n]
    correct = sum(1 for t, p in zip(true_num, pred_num) if t == p)
    return {
        "accuracy": correct / n,
        "precision": precision_score(
            true_num, pred_num, average="macro", zero_division=0
        ),
        "recall": recall_score(true_num, pred_num, average="macro", zero_division=0),
        "f1": f1_score(true_num, pred_num, average="macro", zero_division=0),
    }


# ---------------------------------------------------------------------------
# Set-based name listing metrics (STMF-name_listing, MTMF)
# ---------------------------------------------------------------------------


def name_listing_prf1(
    true: Sequence[str],
    pred: Sequence[str],
) -> Dict[str, float]:
    true_set = {t.strip() for t in true if t.strip()}
    pred_set = {p.strip() for p in pred if p.strip()}
    tp = len(true_set & pred_set)
    precision = tp / len(pred_set) if pred_set else (1.0 if not true_set else 0.0)
    recall = tp / len(true_set) if true_set else (1.0 if not true_set else 0.0)
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# MFS / MML accuracy
# ---------------------------------------------------------------------------


def exact_match_accuracy(
    true: Sequence[str],
    pred: Sequence[str],
) -> Dict[str, float]:
    if not true:
        return {"accuracy": 0.0, "correct": 0, "total": 0}
    correct = sum(
        1 for t, p in zip(true, pred) if t.strip().lower() == p.strip().lower()
    )
    return {"accuracy": correct / len(true), "correct": correct, "total": len(true)}


def _levenshtein_ratio(a: str, b: str) -> float:
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return 1.0 - prev[lb] / max(la, lb)


def _road_similar(a: str, b: str, threshold: float = 0.85) -> bool:
    return a == b or _levenshtein_ratio(a, b) >= threshold


def mml_match(gt_road1: str, gt_road2: str, pred_road1: str, pred_road2: str) -> bool:
    g1, g2 = gt_road1.strip().lower(), gt_road2.strip().lower()
    p1, p2 = pred_road1.strip().lower(), pred_road2.strip().lower()
    return (_road_similar(g1, p1) and _road_similar(g2, p2)) or (
        _road_similar(g1, p2) and _road_similar(g2, p1)
    )


# ---------------------------------------------------------------------------
# SRN route evaluation
# ---------------------------------------------------------------------------


def normalize_route(route: List[str]) -> List[str]:
    route = list(route)
    idx_to_remove: List[int] = []
    for idx, action in enumerate(route):
        if (
            action == "continue straight"
            and idx + 1 < len(route)
            and idx - 1 >= 0
            and route[idx + 1] == route[idx - 1]
        ):
            idx_to_remove.extend([idx, idx + 1])
    for idx in sorted(idx_to_remove, reverse=True):
        route.pop(idx)
    return route


def route_eval(true: List[str], answer: List[str]) -> Tuple[bool, int]:
    correct_step_count = 0
    for t_step, a_step in zip(true, answer):
        if t_step != a_step:
            return False, correct_step_count
        correct_step_count += 1
    return len(true) == len(answer), correct_step_count


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1 = math.radians(lat1), math.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)
    d_lon = lon2 - lon1
    x = math.sin(d_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(
        d_lon
    )
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def get_turn_direction(bearing1: float, bearing2: float) -> str:
    diff = (bearing2 - bearing1 + 360) % 360
    if diff < 40 or diff > 320:
        return "continue straight"
    if diff < 140:
        return "turn right"
    if diff > 220:
        return "turn left"
    return "make a U-turn and continue straight"


def generate_route_directions(
    route: List,
    road_names: List[str],
) -> List[str]:
    output = ["blue"]
    current_bearing = 0.0
    for i in range(len(route) - 1):
        lat1, lon1 = float(route[i][1]), float(route[i][2])
        lat2, lon2 = float(route[i + 1][1]), float(route[i + 1][2])
        next_bearing = calculate_bearing(lat1, lon1, lat2, lon2)
        output.append(get_turn_direction(current_bearing, next_bearing))
        output.append(road_names[i])
        current_bearing = next_bearing
    output.append("red")
    return output


def check_valid_route(
    origin: list,
    dest: list,
    path: List[str],
    conj_dict: dict,
    road_dict: dict,
) -> bool:
    if not path or path[0] != "blue" or path[-1] != "red":
        return False
    if len(path) < 3:
        return False

    road_origin = path[2]
    road_destination = path[-2]
    if road_origin not in road_dict or road_destination not in road_dict:
        return False

    origin_ok = any(
        str(origin[0]) == str(node[0])
        for link in road_dict[road_origin]
        for node in road_dict[road_origin][link]
    )
    dest_ok = any(
        str(dest[0]) == str(node[0])
        for link in road_dict[road_destination]
        for node in road_dict[road_destination][link]
    )
    if not origin_ok or not dest_ok:
        return False

    action_keywords = ["left", "right", "straight", "u-turn"]
    street_names = [
        item
        for item in path
        if item not in ("blue", "red") and not any(kw in item for kw in action_keywords)
    ]

    if len(street_names) <= 1:
        return True

    for name in street_names:
        if name not in road_dict:
            return False

    deduped = [street_names[0]]
    for name in street_names[1:]:
        if name != deduped[-1]:
            deduped.append(name)

    junction_nodes: list = []
    for i in range(len(deduped) - 1):
        if deduped[i] not in conj_dict.get(deduped[i + 1], {}):
            return False
        junction_nodes.append(conj_dict[deduped[i + 1]][deduped[i]][0])

    junction_nodes.insert(0, origin)
    junction_nodes.append(dest)
    correct_path = generate_route_directions(junction_nodes, deduped)

    return all(a == b for a, b in zip(path, correct_path))


def srn_metrics(
    results: List[Dict],
) -> Dict[str, object]:
    if not results:
        return {}
    total = len(results)
    best_path_count = sum(1 for r in results if r["is_success"])
    step_accs = [r["step_accuracy"] for r in results]
    conn_rates = [r["is_connected"] for r in results]

    overall = {
        "shortest_path_success_rate": best_path_count / total,
        "avg_step_accuracy": sum(step_accs) / total,
        "avg_connectivity_rate": sum(1 for c in conn_rates if c) / total,
    }

    zoom_groups: Dict[int, List[Dict]] = {}
    for r in results:
        zl = r.get("zoom_level")
        if zl is not None:
            zoom_groups.setdefault(zl, []).append(r)

    per_zoom: Dict[int, Dict[str, float]] = {}
    for zl, group in sorted(zoom_groups.items()):
        n = len(group)
        per_zoom[zl] = {
            "shortest_path_success_rate": sum(1 for r in group if r["is_success"]) / n,
            "avg_step_accuracy": sum(r["step_accuracy"] for r in group) / n,
            "connectivity_rate": sum(1 for r in group if r["is_connected"]) / n,
        }
    return {"overall": overall, "per_zoom": per_zoom}
