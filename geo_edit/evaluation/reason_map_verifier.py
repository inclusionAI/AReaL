"""Rule-based route verification for ReasonMap (base) dataset.

Parses model output into route sections, then validates each section
against the metro network topology (station existence, route continuity,
transfer point matching).

Scoring:
1. Extract route sections from model response (split by ``--``)
2. Parse ``Route Name`` / ``Departure Stop`` / ``Arrival Stop`` from each section
3. Validate topology: departure == station_1, arrival == station_2,
   routes exist, stations exist on routes, transfers match
4. Binary score: 1.0 if fully valid, 0.0 otherwise
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple


_STATION_ANNOTATIONS = re.compile(
    r"\s*[（(]"
    r"(?:换乘站|支线起始站|Transfer Station|Branch-starting Station"
    r"|2线起始站|\d+线起始站)"
    r"[）)]\s*"
)
_SUFFIX_STATION = re.compile(r"站$")


def clean_station_name(name: str) -> str:
    """Normalise a station name for fuzzy comparison.

    Strips parenthetical annotations (e.g. ``(Transfer Station)``),
    trailing ``站`` suffix, surrounding whitespace, and applies Unicode
    NFKC normalisation so that full-width / half-width characters match.
    """
    name = _STATION_ANNOTATIONS.sub("", name)
    name = _SUFFIX_STATION.sub("", name)
    name = unicodedata.normalize("NFKC", name)
    return name.strip()


def _build_clean_metro(
    metro_data: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """Return a mapping ``{line_name: [clean_station, ...]}``."""
    return {
        line: [clean_station_name(s) for s in stations]
        for line, stations in metro_data.items()
    }


def extract_route_sections(response: str) -> List[Dict[str, str]]:
    """Parse model response into structured route sections.

    Expected format (sections separated by ``--``):

    .. code-block:: text

        Route Name: Line 3
        Departure Stop: 石河北海站
        Arrival Stop: 九里站
        Number of Via Stops: 5
    """
    sections = re.split(r"\n\s*--\s*\n|^--\s*$", response, flags=re.MULTILINE)
    route_data: List[Dict[str, str]] = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        route_name_m = re.search(r"Route Name:\s*(.+?)(?:\n|$)", section)
        departure_m = re.search(r"Departure Stop:\s*(.+?)(?:\n|$)", section)
        arrival_m = re.search(r"Arrival Stop:\s*(.+?)(?:\n|$)", section)

        if not (route_name_m and departure_m and arrival_m):
            continue

        route_data.append({
            "route_name": route_name_m.group(1).strip(),
            "departure": departure_m.group(1).strip(),
            "arrival": arrival_m.group(1).strip(),
        })

    return route_data


def verify_route(
    route_sections: List[Dict[str, str]],
    station_1: str,
    station_2: str,
    metro_data: Dict[str, List[str]],
) -> Tuple[bool, str]:
    """Validate parsed route sections against the metro topology.

    Returns ``(is_valid, reason)``.
    """
    if not route_sections:
        return False, "no_route_sections"

    clean_metro = _build_clean_metro(metro_data)
    cs1 = clean_station_name(station_1)
    cs2 = clean_station_name(station_2)

    first = route_sections[0]
    last = route_sections[-1]

    if clean_station_name(first["departure"]) != cs1:
        return False, (
            f"wrong_departure: expected={cs1} got={clean_station_name(first['departure'])}"
        )
    if clean_station_name(last["arrival"]) != cs2:
        return False, (
            f"wrong_arrival: expected={cs2} got={clean_station_name(last['arrival'])}"
        )

    for i, sec in enumerate(route_sections):
        route_name = sec["route_name"]
        dep = clean_station_name(sec["departure"])
        arr = clean_station_name(sec["arrival"])

        matched_line: Optional[List[str]] = None
        for line_name, stations in clean_metro.items():
            if line_name == route_name or clean_station_name(route_name) == clean_station_name(line_name):
                matched_line = stations
                break

        if matched_line is None:
            return False, f"unknown_route: '{route_name}' not in metro_data"

        if dep not in matched_line:
            return False, f"section_{i}_departure_not_on_route: '{dep}' not on '{route_name}'"
        if arr not in matched_line:
            return False, f"section_{i}_arrival_not_on_route: '{arr}' not on '{route_name}'"

        if i < len(route_sections) - 1:
            next_dep = clean_station_name(route_sections[i + 1]["departure"])
            if arr != next_dep:
                return False, (
                    f"transfer_mismatch: section_{i} arrival='{arr}' "
                    f"!= section_{i+1} departure='{next_dep}'"
                )

    return True, "valid"


def reason_map_score(
    response: str,
    station_1: str,
    station_2: str,
    metro_data: Dict[str, List[str]],
) -> Tuple[float, str]:
    """Score a ReasonMap route prediction.

    Returns ``(score, reason)`` where score is 1.0 (correct) or 0.0.
    """
    answer_m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    text = answer_m.group(1).strip() if answer_m else response

    sections = extract_route_sections(text)
    if not sections:
        sections = extract_route_sections(response)

    is_valid, reason = verify_route(sections, station_1, station_2, metro_data)
    return (1.0 if is_valid else 0.0), reason


def reason_map_judge(
    question: str,
    ground_truth: str,
    prediction: str,
    image_path: str = "",
    meta_info_extra: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """TrajectoryJudge-compatible interface for ReasonMap route verification.

    ``meta_info_extra`` must contain ``station_1``, ``station_2``, and
    ``metro_data`` (either a dict or a JSON string).
    """
    if not meta_info_extra:
        return False, "reason_map_judge_error: missing meta_info_extra"

    station_1 = meta_info_extra.get("station_1", "")
    station_2 = meta_info_extra.get("station_2", "")
    metro_raw = meta_info_extra.get("metro_data", {})

    if isinstance(metro_raw, str):
        try:
            metro_data = json.loads(metro_raw)
        except (json.JSONDecodeError, TypeError):
            return False, f"reason_map_judge_error: invalid metro_data JSON"
    else:
        metro_data = metro_raw

    if not station_1 or not station_2 or not metro_data:
        return False, "reason_map_judge_error: missing station_1/station_2/metro_data"

    score, reason = reason_map_score(prediction, station_1, station_2, metro_data)
    if score >= 1.0:
        return True, "valid"
    return False, f"reason_map_score=0.0 ({reason})"
