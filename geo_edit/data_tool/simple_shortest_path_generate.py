import os
import json
import random
import math
import heapq
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFont

# ---------------------------
# Config
# ---------------------------
OUT_DIR = "vlm_sp_unique_dataset"
IMG_ORIG_DIR = os.path.join(OUT_DIR, "images_original")      # 200 张原始图（无权重）
IMG_ANN_DIR = os.path.join(OUT_DIR, "images_annotated")      # 200 张标注图（有权重）
JSONL_PATH = os.path.join(OUT_DIR, "dataset.jsonl")          # 400 行 case

SEED = 7

LEVELS = [4, 8, 12, 16]
LEVEL_COUNTS = {4: 25, 8: 50, 12: 75, 16: 50}  # 总计 200 张原始图

WEIGHT_MIN = 1
WEIGHT_MAX = 50                   # 拉大权重范围，有助于减少最短路并列

# 按节点数动态设置画布尺寸
CANVAS_MAP = {
    4: 768,
    8: 1024,
    12: 1280,
    16: 1536
}
MARGIN_MAP = {
    4: 90,
    8: 120,
    12: 140,
    16: 160
}
NODE_RADIUS_MAP = {
    4: 36,
    8: 36,
    12: 34,
    16: 34
}

EDGE_WIDTH = 4

BG_COLOR = (255, 255, 255)
EDGE_COLOR = (0, 0, 0)
NODE_FILL = (0, 165, 225)
NODE_OUTLINE = (0, 145, 200)
NODE_TEXT_COLOR = (255, 255, 255)
WEIGHT_TEXT_COLOR = (0, 0, 0)

FONT_PATH = None                  # 可改成你的 ttf 路径
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\seguisb.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]
FONT_NODE_SIZE = 48
FONT_W_SIZE = 40

# 额外连边数量（在生成树基础上再加少量额外边，控制密度）
EXTRA_EDGE_COUNT = {4: 2, 8: 3, 12: 4, 16: 5}
# 最小夹角（度）：避免某个节点处两条线过于贴近
MIN_ANGLE_DEG = {4: 18.0, 8: 22.0, 12: 22.0, 16: 22.0}

MAX_GRAPH_TRIES = 5000
MAX_QUERY_TRIES = 2000


# ---------------------------
# Fonts
# ---------------------------
def load_font(size: int) -> ImageFont.FreeTypeFont:
    if FONT_PATH and os.path.exists(FONT_PATH):
        return ImageFont.truetype(FONT_PATH, size)
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


# ---------------------------
# Text measure (Pillow-compatible)
# ---------------------------
def text_wh(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    # textbbox returns (left, top, right, bottom)
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    return r - l, b - t


# ---------------------------
# Labels: A..Z, AA..AZ, BA.. etc
# ---------------------------
def idx_to_label(i: int) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    s = ""
    x = i
    while True:
        s = letters[x % 26] + s
        x = x // 26 - 1
        if x < 0:
            break
    return s


def make_node_labels(n: int) -> List[str]:
    return [idx_to_label(i) for i in range(n)]


def euclid(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ---------------------------
# Positions: mixed layout (outer ring + some inner points) with shape diversity
# ---------------------------
def ring_positions(labels: List[str], rng: random.Random, canvas: int, margin: int, node_radius: int) -> Dict[str, Tuple[int, int]]:
    return layout_ordered_ring_positions(labels, rng, canvas, margin, node_radius)


# ---------------------------
# Graph utilities
# ---------------------------
def build_adj(edges: List[Tuple[str, str, int]]) -> Dict[str, List[Tuple[str, int]]]:
    adj: Dict[str, List[Tuple[str, int]]] = {}
    for u, v, w in edges:
        adj.setdefault(u, []).append((v, w))
        adj.setdefault(v, []).append((u, w))
    return adj


def is_connected(labels: List[str], edges: List[Tuple[str, str, int]]) -> bool:
    if not labels:
        return True
    adj = build_adj(edges)
    start = labels[0]
    stack = [start]
    seen = {start}
    while stack:
        u = stack.pop()
        for v, _w in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == len(labels)


# ---------------------------
# Dijkstra with counting shortest paths
# ---------------------------
def dijkstra_unique_path(edges: List[Tuple[str, str, int]], source: str, target: str) -> Tuple[List[str], int, int]:
    """
    返回 (path, dist, count_paths)
    count_paths 是最短路条数（只要 >1 就说明不唯一）
    """
    adj = build_adj(edges)
    INF = 10**18
    dist = {source: 0}
    count = {source: 1}
    parent: Dict[str, str] = {}

    pq = [(0, source)]
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == target:
            break
        for v, w in adj.get(u, []):
            nd = d + w
            old = dist.get(v, INF)
            if nd < old:
                dist[v] = nd
                count[v] = count[u]
                parent[v] = u
                heapq.heappush(pq, (nd, v))
            elif nd == old:
                count[v] = count.get(v, 0) + count[u]

    if target not in dist:
        return [], INF, 0

    path = [target]
    cur = target
    while cur != source:
        if cur not in parent:
            return [], dist[target], count.get(target, 0)
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path, dist[target], count.get(target, 0)


# ---------------------------
# Graph generation: non-crossing spanning tree + a few extra edges
# ---------------------------
def assign_weights_by_length(edges: List[Tuple[str, str]], pos: Dict[str, Tuple[int, int]]) -> List[Tuple[str, str, int]]:
    lengths = [euclid(pos[u], pos[v]) for u, v in edges]
    if not lengths:
        return []
    max_len = max(lengths)
    if max_len < 1e-6:
        return [(u, v, (WEIGHT_MIN + WEIGHT_MAX) // 2) for (u, v) in edges]

    scale = WEIGHT_MAX / max_len
    weighted: List[Tuple[str, str, int]] = []
    for (u, v), dist in zip(edges, lengths):
        w = int(round(dist * scale))
        w = max(WEIGHT_MIN, min(WEIGHT_MAX, w))
        weighted.append((u, v, w))
    return weighted


def generate_graph(labels: List[str],
                   pos: Dict[str, Tuple[int, int]],
                   extra_edges: int,
                   rng: random.Random,
                   min_angle_deg: float) -> List[Tuple[str, str, int]]:
    def seg_intersect(a1: Tuple[float, float], a2: Tuple[float, float],
                      b1: Tuple[float, float], b2: Tuple[float, float]) -> bool:
        def orient(p, q, r) -> float:
            return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

        def on_segment(p, q, r) -> bool:
            return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

        o1 = orient(a1, a2, b1)
        o2 = orient(a1, a2, b2)
        o3 = orient(b1, b2, a1)
        o4 = orient(b1, b2, a2)

        if o1 == 0 and on_segment(a1, b1, a2):
            return True
        if o2 == 0 and on_segment(a1, b2, a2):
            return True
        if o3 == 0 and on_segment(b1, a1, b2):
            return True
        if o4 == 0 and on_segment(b1, a2, b2):
            return True

        return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

    def crosses_existing(u: str, v: str, segs: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> bool:
        p1 = (pos[u][0], pos[u][1])
        p2 = (pos[v][0], pos[v][1])
        for (a, b) in segs:
            if (a == p1 or a == p2 or b == p1 or b == p2):
                continue
            if seg_intersect(p1, p2, a, b):
                return True
        return False

    def crosses_existing_labels(u: str, v: str, segs: List[Tuple[str, str]]) -> bool:
        p1 = (pos[u][0], pos[u][1])
        p2 = (pos[v][0], pos[v][1])
        for a, b in segs:
            if u == a or u == b or v == a or v == b:
                continue
            if seg_intersect(p1, p2, (pos[a][0], pos[a][1]), (pos[b][0], pos[b][1])):
                return True
        return False

    nodes = labels[:]
    rng.shuffle(nodes)

    degrees = {lab: 0 for lab in nodes}
    adj: Dict[str, List[str]] = {lab: [] for lab in nodes}
    edges: List[Tuple[str, str]] = []
    existing = set()
    segments: List[Tuple[str, str]] = []

    min_angle_rad = math.radians(min_angle_deg)

    def angle_ok(u: str, v: str) -> bool:
        ux, uy = pos[u]
        vx, vy = pos[v]
        vvec = (vx - ux, vy - uy)
        vlen = math.hypot(vvec[0], vvec[1])
        if vlen < 1e-6:
            return False
        for nbr in adj[u]:
            if nbr == v:
                continue
            nx, ny = pos[nbr]
            nvec = (nx - ux, ny - uy)
            nlen = math.hypot(nvec[0], nvec[1])
            if nlen < 1e-6:
                continue
            dot = vvec[0] * nvec[0] + vvec[1] * nvec[1]
            cosv = max(-1.0, min(1.0, dot / (vlen * nlen)))
            ang = math.acos(cosv)
            if ang < min_angle_rad:
                return False
        return True

    def add_edge(u: str, v: str, allow_cross: bool = False) -> bool:
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in existing:
            return False
        if not allow_cross and crosses_existing_labels(u, v, segments):
            return False
        if not angle_ok(u, v) or not angle_ok(v, u):
            return False
        edges.append((a, b))
        existing.add((a, b))
        segments.append((a, b))
        degrees[u] += 1
        degrees[v] += 1
        adj[u].append(v)
        adj[v].append(u)
        return True

    # Build a spanning tree (prefer short and non-crossing edges)
    tree = {nodes[0]}
    while len(tree) < len(nodes):
        candidates: List[Tuple[float, str, str]] = []
        for u in tree:
            for v in nodes:
                if v in tree:
                    continue
                candidates.append((euclid(pos[u], pos[v]), u, v))
        candidates.sort(key=lambda x: x[0])

        chosen = None
        for _d, u, v in candidates:
            if add_edge(u, v, allow_cross=False):
                chosen = v
                break
        if chosen is None:
            for _d, u, v in candidates:
                if add_edge(u, v, allow_cross=True):
                    chosen = v
                    break
        if chosen is None:
            return []
        tree.add(chosen)

    # Add a few extra edges, still trying to keep it readable
    n = len(nodes)
    max_degree = 3 if n <= 4 else 4
    all_pairs: List[Tuple[float, str, str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            u = nodes[i]
            v = nodes[j]
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in existing:
                continue
            all_pairs.append((euclid(pos[u], pos[v]), u, v))
    all_pairs.sort(key=lambda x: x[0])

    added = 0
    for _d, u, v in all_pairs:
        if added >= extra_edges:
            break
        if degrees[u] >= max_degree or degrees[v] >= max_degree:
            continue
        if add_edge(u, v, allow_cross=False):
            added += 1

    return assign_weights_by_length(edges, pos)


def layout_ordered_ring_positions(labels: List[str], rng: random.Random, canvas: int, margin: int, node_radius: int) -> Dict[str, Tuple[int, int]]:
    n = len(labels)
    cx, cy = canvas / 2, canvas / 2
    base_r = canvas / 2 - margin - node_radius

    def min_pair_dist(points: List[Tuple[float, float]]) -> float:
        best = float("inf")
        for i in range(len(points)):
            x1, y1 = points[i]
            for j in range(i + 1, len(points)):
                x2, y2 = points[j]
                d = math.hypot(x1 - x2, y1 - y2)
                if d < best:
                    best = d
        return best

    def scale_to_target(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        if w == 0 or h == 0:
            return points

        # initial scale from unit ring to pixels
        base_scale = base_r
        w_px = w * base_scale
        h_px = h * base_scale
        area_ratio = (w_px * h_px) / (canvas * canvas)

        target_min = 0.48
        target_max = 0.72
        target = rng.uniform(target_min, target_max)

        if area_ratio < target:
            scale_adj = math.sqrt(target / max(area_ratio, 1e-6))
        elif area_ratio > target_max:
            scale_adj = math.sqrt(target_max / area_ratio)
        else:
            scale_adj = 1.0

        max_scale = min((canvas - 2 * margin) / w_px, (canvas - 2 * margin) / h_px)
        scale = base_scale * min(scale_adj, max_scale)

        return [(x * scale, y * scale) for (x, y) in points]

    def choose_inner_count() -> int:
        if n <= 4:
            return 0
        base = int(round(n * rng.uniform(0.22, 0.32)))
        base = max(1, base)
        return min(base, n - 3)

    min_dist_norm = (node_radius * 2.25) / max(base_r, 1.0)
    min_dist_norm_sq = min_dist_norm * min_dist_norm

    def far_enough(x: float, y: float, pts: List[Tuple[float, float]]) -> bool:
        for px, py in pts:
            if (x - px) ** 2 + (y - py) ** 2 < min_dist_norm_sq:
                return False
        return True

    last_pos: Dict[str, Tuple[int, int]] = {}
    for _ in range(200):
        inner_count = choose_inner_count()
        outer_count = n - inner_count

        outer_angles = [2 * math.pi * i / outer_count for i in range(outer_count)]
        rot = rng.uniform(0, 2 * math.pi)
        jitter_a = rng.uniform(0.06, 0.18)
        outer_angles = sorted([a + rot + rng.uniform(-jitter_a, jitter_a) for a in outer_angles])

        mode = rng.choice(["ring", "ellipse", "bulge2", "bulge3", "teardrop", "skew"])
        amp = rng.uniform(0.08, 0.18)
        phase = rng.uniform(0, 2 * math.pi)
        k = 2 if mode == "bulge2" else 3
        sx = rng.uniform(0.85, 1.15)
        sy = rng.uniform(0.85, 1.15)
        shear = rng.uniform(-0.18, 0.18)

        def apply_affine(x: float, y: float) -> Tuple[float, float]:
            if mode == "ellipse":
                return x * sx, y * sy
            if mode == "skew":
                return x + shear * y, y
            return x, y

        pts: List[Tuple[float, float]] = []

        # outer ring points
        for ang in outer_angles:
            r = 1.0 + rng.uniform(-0.05, 0.05)
            if mode in ("bulge2", "bulge3"):
                r *= 1.0 + amp * math.cos(k * ang + phase)
            elif mode == "teardrop":
                r *= 1.0 + amp * math.cos(ang - phase)
            x = r * math.cos(ang)
            y = r * math.sin(ang)
            x, y = apply_affine(x, y)
            pts.append((x, y))

        # inner points
        inner_pts: List[Tuple[float, float]] = []
        inner_r = rng.uniform(0.32, 0.58)
        tries = 0
        while len(inner_pts) < inner_count and tries < 8000:
            tries += 1
            rr = inner_r * math.sqrt(rng.random())
            aa = rng.uniform(0, 2 * math.pi)
            x = rr * math.cos(aa) + rng.uniform(-0.02, 0.02)
            y = rr * math.sin(aa) + rng.uniform(-0.02, 0.02)
            x, y = apply_affine(x, y)
            if not far_enough(x, y, pts) or not far_enough(x, y, inner_pts):
                continue
            inner_pts.append((x, y))

        if len(inner_pts) < inner_count:
            continue

        rng.shuffle(inner_pts)
        all_pts = pts + inner_pts
        all_pts = scale_to_target(all_pts)

        pts_px: List[Tuple[float, float]] = []
        for x, y in all_pts:
            px = cx + x
            py = cy + y
            px = min(max(margin, px), canvas - margin)
            py = min(max(margin, py), canvas - margin)
            pts_px.append((px, py))

        if min_pair_dist(pts_px) < node_radius * 2.2:
            last_pos = {lab: (int(px), int(py)) for lab, (px, py) in zip(labels, pts_px)}
            continue

        # Map labels: outer labels around the boundary, inner labels for the remaining points
        pos: Dict[str, Tuple[int, int]] = {}
        outer_labels = labels[:outer_count]
        inner_labels = labels[outer_count:]

        for lab, (px, py) in zip(outer_labels, pts_px[:outer_count]):
            pos[lab] = (int(px), int(py))
        for lab, (px, py) in zip(inner_labels, pts_px[outer_count:]):
            pos[lab] = (int(px), int(py))

        last_pos = pos
        return last_pos

    return last_pos


def edge_list_text(edges: List[Tuple[str, str, int]]) -> str:
    parts = []
    for u, v, w in sorted(edges):
        parts.append(f"{u}{v}={w}")
    return ", ".join(parts)


# ---------------------------
# Rendering
# ---------------------------
def draw_graph_base(draw: ImageDraw.ImageDraw,
                    labels: List[str],
                    pos: Dict[str, Tuple[int, int]],
                    edges: List[Tuple[str, str, int]],
                    font_node: ImageFont.FreeTypeFont,
                    node_radius: int):
    for u, v, _w in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        draw.line((x1, y1, x2, y2), fill=EDGE_COLOR, width=EDGE_WIDTH)

    for lab in labels:
        x, y = pos[lab]
        bbox = (x - node_radius, y - node_radius, x + node_radius, y + node_radius)
        draw.ellipse(bbox, fill=NODE_FILL, outline=NODE_OUTLINE, width=2)

        tw, th = text_wh(draw, lab, font_node)
        draw.text((x - tw / 2, y - th / 2), lab, fill=NODE_TEXT_COLOR, font=font_node)


def draw_weights(draw: ImageDraw.ImageDraw,
                 pos: Dict[str, Tuple[int, int]],
                 edges: List[Tuple[str, str, int]],
                 font_w: ImageFont.FreeTypeFont,
                 rng: random.Random,
                 node_radius: int,
                 canvas: int):
    def point_in_rect(px: float, py: float, rect: Tuple[float, float, float, float]) -> bool:
        return rect[0] <= px <= rect[2] and rect[1] <= py <= rect[3]

    def seg_intersect(a1: Tuple[float, float], a2: Tuple[float, float],
                      b1: Tuple[float, float], b2: Tuple[float, float]) -> bool:
        def orient(p, q, r) -> float:
            return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

        def on_segment(p, q, r) -> bool:
            return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

        o1 = orient(a1, a2, b1)
        o2 = orient(a1, a2, b2)
        o3 = orient(b1, b2, a1)
        o4 = orient(b1, b2, a2)

        if o1 == 0 and on_segment(a1, b1, a2):
            return True
        if o2 == 0 and on_segment(a1, b2, a2):
            return True
        if o3 == 0 and on_segment(b1, a1, b2):
            return True
        if o4 == 0 and on_segment(b1, a2, b2):
            return True

        return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

    def rect_intersects_segment(rect: Tuple[float, float, float, float],
                                p1: Tuple[float, float],
                                p2: Tuple[float, float]) -> bool:
        if point_in_rect(p1[0], p1[1], rect) or point_in_rect(p2[0], p2[1], rect):
            return True
        l, t, r, b = rect
        corners = [(l, t), (r, t), (r, b), (l, b)]
        edges_rect = [
            (corners[0], corners[1]),
            (corners[1], corners[2]),
            (corners[2], corners[3]),
            (corners[3], corners[0]),
        ]
        for e1, e2 in edges_rect:
            if seg_intersect(p1, p2, e1, e2):
                return True
        return False

    def rect_circle_intersect(rect: Tuple[float, float, float, float],
                              cx: float, cy: float, r: float) -> bool:
        l, t, rr, bb = rect
        closest_x = min(max(cx, l), rr)
        closest_y = min(max(cy, t), bb)
        dx = cx - closest_x
        dy = cy - closest_y
        return dx * dx + dy * dy <= r * r

    cx = sum(p[0] for p in pos.values()) / len(pos)
    cy = sum(p[1] for p in pos.values()) / len(pos)
    segments = [((pos[u][0], pos[u][1]), (pos[v][0], pos[v][1])) for (u, v, _w) in edges]
    label_rects: List[Tuple[float, float, float, float]] = []

    for idx, (u, v, w) in enumerate(edges):
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            continue
        nx, ny = -dy / length, dx / length

        text = str(w)
        tw, th = text_wh(draw, text, font_w)

        # Prefer placing the label slightly outside of the graph center along the edge normal
        t0 = 0.5 + rng.uniform(-0.03, 0.03)
        t_candidates = [t0, 0.5, 0.4, 0.6, 0.33, 0.67]

        rect = None
        for t in t_candidates:
            mx = x1 + dx * t
            my = y1 + dy * t
            vx = mx - cx
            vy = my - cy
            sign = 1 if (vx * nx + vy * ny) >= 0 else -1

            base_offset = max(node_radius * 0.6, th * 0.65) + 8
            for _side in (sign, -sign):
                for step in range(14):
                    offset = base_offset + step * 6
                    x = mx + nx * _side * offset
                    y = my + ny * _side * offset
                    cand = (x - tw / 2, y - th / 2, x + tw / 2, y + th / 2)

                    # keep fully inside canvas (no clamping that can cause overlaps)
                    if cand[0] < 2 or cand[1] < 2 or cand[2] > canvas - 2 or cand[3] > canvas - 2:
                        continue
                    if any(rect_intersects_segment(cand, s1, s2) for s1, s2 in segments):
                        continue
                    if any(rect_circle_intersect(cand, pos[nlab][0], pos[nlab][1], node_radius + 2) for nlab in pos):
                        continue
                    if any(
                        not (cand[2] < r[0] or cand[0] > r[2] or cand[3] < r[1] or cand[1] > r[3])
                        for r in label_rects
                    ):
                        continue
                    rect = cand
                    break
                if rect is not None:
                    break
            if rect is not None:
                break

        if rect is None:
            raise RuntimeError("Failed to place weight labels without overlap")

        draw.text((rect[0], rect[1]), text, fill=WEIGHT_TEXT_COLOR, font=font_w)
        label_rects.append(rect)


def render_image(labels: List[str],
                 pos: Dict[str, Tuple[int, int]],
                 edges: List[Tuple[str, str, int]],
                 annotate_weights: bool,
                 rng: random.Random,
                 canvas: int,
                 node_radius: int) -> Image.Image:
    img = Image.new("RGB", (canvas, canvas), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font_node = load_font(FONT_NODE_SIZE)
    font_w = load_font(FONT_W_SIZE)

    draw_graph_base(draw, labels, pos, edges, font_node, node_radius)
    if annotate_weights:
        draw_weights(draw, pos, edges, font_w, rng, node_radius=node_radius, canvas=canvas)
    return img


# ---------------------------
# Choose source-target with unique shortest path
# ---------------------------
def pick_unique_query(labels: List[str], edges: List[Tuple[str, str, int]], rng: random.Random) -> Tuple[str, str, List[str], int]:
    for _ in range(MAX_QUERY_TRIES):
        s, t = rng.sample(labels, 2)
        path, d, cnt = dijkstra_unique_path(edges, s, t)
        if len(path) >= 2 and cnt == 1:
            return s, t, path, d
    return "", "", [], -1


# ---------------------------
# Main
# ---------------------------
def main():
    rng = random.Random(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(IMG_ORIG_DIR, exist_ok=True)
    os.makedirs(IMG_ANN_DIR, exist_ok=True)

    records = []
    base_id = 0
    case_id = 0

    for n in LEVELS:
        canvas = CANVAS_MAP[n]
        margin = MARGIN_MAP[n]
        node_radius = NODE_RADIUS_MAP[n]
        extra_edges = EXTRA_EDGE_COUNT[n]
        min_angle = MIN_ANGLE_DEG[n]
        target_count = LEVEL_COUNTS[n]

        for _ in range(target_count):
            labels = make_node_labels(n)
            edges = []
            pos = {}
            img_ann = None
            s = t = ""
            gt_path: List[str] = []
            gt_dist = -1

            for _try in range(MAX_GRAPH_TRIES):
                pos = ring_positions(labels, rng, canvas=canvas, margin=margin, node_radius=node_radius)
                edges = generate_graph(labels, pos, extra_edges=extra_edges, rng=rng, min_angle_deg=min_angle)
                if not edges:
                    continue

                if n == 4 and len(edges) < 5:
                    continue

                if not is_connected(labels, edges):
                    continue

                s, t, gt_path, gt_dist = pick_unique_query(labels, edges, rng)
                if not gt_path:
                    continue

                # Ensure weight labels can be placed without overlap; otherwise retry a new graph/layout
                try:
                    img_ann = render_image(labels, pos, edges, annotate_weights=True, rng=rng, canvas=canvas, node_radius=node_radius)
                except RuntimeError:
                    img_ann = None
                    continue

                break
            else:
                raise RuntimeError(f"Failed to generate unique-shortest graph for n={n}")

            # 原始图（无权重）
            orig_name = f"g{base_id:03d}_n{n}_orig.png"
            orig_path = os.path.join(IMG_ORIG_DIR, orig_name)
            img_orig = render_image(labels, pos, edges, annotate_weights=False, rng=rng, canvas=canvas, node_radius=node_radius)
            img_orig.save(orig_path)

            # 标注图（有权重）
            ann_name = f"g{base_id:03d}_n{n}_ann.png"
            ann_path = os.path.join(IMG_ANN_DIR, ann_name)
            if img_ann is None:
                # should never happen due to the check above
                img_ann = render_image(labels, pos, edges, annotate_weights=True, rng=rng, canvas=canvas, node_radius=node_radius)
            img_ann.save(ann_path)

            # text case：原始图 + 文本权重
            rec_text = {
                "case_id": case_id,
                "base_graph_id": base_id,
                "level_nodes": n,
                "condition": "text",
                "image_path": os.path.join("images_original", orig_name),
                "query": {"source": s, "target": t},
                "ground_truth": {"path": gt_path},
                "edge_list_text": edge_list_text(edges),
                "edges": [{"u": u, "v": v, "w": w} for (u, v, w) in edges]
            }
            records.append(rec_text)
            case_id += 1

            # image case：标注图（图上有权重）
            rec_image = {
                "case_id": case_id,
                "base_graph_id": base_id,
                "level_nodes": n,
                "condition": "image",
                "image_path": os.path.join("images_annotated", ann_name),
                "query": {"source": s, "target": t},
                "ground_truth": {"path": gt_path},
                "edge_list_text": "",
                "edges": [{"u": u, "v": v, "w": w} for (u, v, w) in edges]
            }
            records.append(rec_image)
            case_id += 1

            base_id += 1

    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Done.")
    print(f"Original images:   {IMG_ORIG_DIR} (should be 200)")
    print(f"Annotated images:  {IMG_ANN_DIR} (should be 200)")
    print(f"JSONL cases:       {JSONL_PATH} (should be 400 lines)")
    print("Levels:", {n: LEVEL_COUNTS[n] for n in LEVELS})


if __name__ == "__main__":
    main()
