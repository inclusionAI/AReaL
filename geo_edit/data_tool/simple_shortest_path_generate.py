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
EDGE_INFO_DIR = os.path.join(OUT_DIR, "edge_info")

SEED = 7

LEVELS = [32, 48]
LEVEL_COUNTS = {32: 20, 48: 20}  # 总计 40 张原始图

# 是否生成最短路问题（每图 5 个）；当前先生成清晰图片并保存边权信息
GENERATE_QUERIES = False
NUM_QUERIES_PER_IMAGE = 5

# 仅追加多中心额外样本，不重生成已存在样本
APPEND_EXTRA_MULTICENTER_ONLY = True
EXTRA_MULTICENTER_COUNT = 10
EXTRA_MULTICENTER_LEVEL = 48
EXTRA_MULTICENTER_QUERY_COUNT = 10

WEIGHT_MIN = 1
WEIGHT_MAX = 50                   # 拉大权重范围，有助于减少最短路并列

# 按节点数动态设置画布尺寸
CANVAS_MAP = {
    4: 768,
    8: 1024,
    12: 1280,
    16: 1536,
    32: 3600,
    48: 4800,
}
MARGIN_MAP = {
    4: 90,
    8: 120,
    12: 140,
    16: 160,
    32: 320,
    48: 400,
}
NODE_RADIUS_MAP = {
    4: 36,
    8: 36,
    12: 34,
    16: 34,
    32: 46,
    48: 42,
}

EDGE_WIDTH = 4

BG_COLOR = (255, 255, 255)
EDGE_COLOR = (0, 0, 0)
NODE_FILL = (0, 165, 225)
NODE_OUTLINE = (0, 145, 200)
NODE_TEXT_COLOR = (0, 0, 0)
WEIGHT_TEXT_COLOR = (220, 0, 0)

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
FONT_NODE_SIZE = 40
FONT_W_SIZE = 40

# 额外连边数量（在生成树基础上再加少量额外边，控制密度）
# 额外连边数量（在生成树基础上再加少量额外边，控制密度）
# 额外连边数量（在生成树基础上再加少量额外边，控制密度）
EXTRA_EDGE_COUNT = {4: 2, 8: 3, 12: 4, 16: 5, 32: 6, 48: 8}
# 额外长边数量（优先从较长边中挑选，提升长距离连边）
LONG_EDGE_COUNT = {32: 2, 48: 3}
# 两条线段中点的最小距离（像素）：避免同一节点处线段过于贴近
MIN_MIDPOINT_DIST = {4: 18.0, 8: 22.0, 12: 22.0, 16: 22.0, 32: 18.0, 48: 16.0}
# 当边足够长时，允许更小的中点距离（豁免阈值）
MIDPOINT_EXEMPT_LEN = {32: 180.0, 48: 220.0}
# 最小三角形面积（像素^2），仅对高点数生效，避免极小三角形
MIN_TRIANGLE_AREA = {32: 6000.0, 48: 9000.0}

MAX_GRAPH_TRIES = 12000
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


def get_next_base_id() -> int:
    best = -1
    for folder in (IMG_ORIG_DIR, IMG_ANN_DIR):
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if not name.startswith("g") or "_n" not in name:
                continue
            part = name[1:name.find("_n")]
            if part.isdigit():
                best = max(best, int(part))
    return best + 1


def get_next_case_id() -> int:
    if not os.path.exists(JSONL_PATH):
        return 0
    best = -1
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = obj.get("case_id")
            if isinstance(cid, int):
                best = max(best, cid)
    return best + 1


# ---------------------------
# Geometry helpers
# ---------------------------
def dist_point_to_seg(px: float, py: float,
                      x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-6:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


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


def segment_distance(a1: Tuple[float, float], a2: Tuple[float, float],
                     b1: Tuple[float, float], b2: Tuple[float, float]) -> float:
    if seg_intersect(a1, a2, b1, b2):
        return 0.0
    return min(
        dist_point_to_seg(a1[0], a1[1], b1[0], b1[1], b2[0], b2[1]),
        dist_point_to_seg(a2[0], a2[1], b1[0], b1[1], b2[0], b2[1]),
        dist_point_to_seg(b1[0], b1[1], a1[0], a1[1], a2[0], a2[1]),
        dist_point_to_seg(b2[0], b2[1], a1[0], a1[1], a2[0], a2[1]),
    )


def rects_overlap(a: Tuple[float, float, float, float],
                  b: Tuple[float, float, float, float],
                  pad: float = 0.0) -> bool:
    return not (a[2] + pad < b[0] - pad or b[2] + pad < a[0] - pad or a[3] + pad < b[1] - pad or b[3] + pad < a[1] - pad)


def rect_circle_overlap(rect: Tuple[float, float, float, float],
                        cx: float, cy: float, r: float) -> bool:
    x1, y1, x2, y2 = rect
    px = min(max(cx, x1), x2)
    py = min(max(cy, y1), y2)
    return (px - cx) ** 2 + (py - cy) ** 2 <= r * r


def seg_intersects_rect(a: Tuple[float, float], b: Tuple[float, float],
                        rect: Tuple[float, float, float, float],
                        pad: float = 0.0) -> bool:
    x1, y1, x2, y2 = rect
    x1 -= pad
    y1 -= pad
    x2 += pad
    y2 += pad
    if x1 <= a[0] <= x2 and y1 <= a[1] <= y2:
        return True
    if x1 <= b[0] <= x2 and y1 <= b[1] <= y2:
        return True
    r1 = (x1, y1)
    r2 = (x2, y1)
    r3 = (x2, y2)
    r4 = (x1, y2)
    if seg_intersect(a, b, r1, r2):
        return True
    if seg_intersect(a, b, r2, r3):
        return True
    if seg_intersect(a, b, r3, r4):
        return True
    if seg_intersect(a, b, r4, r1):
        return True
    return False


# ---------------------------
# Positions: mixed layout (outer ring + some inner points) with shape diversity
# ---------------------------
def ring_positions(labels: List[str], rng: random.Random, canvas: int, margin: int, node_radius: int) -> Dict[str, Tuple[int, int]]:
    return layout_ordered_ring_positions(labels, rng, canvas, margin, node_radius)


def layout_multicenter_positions(labels: List[str],
                                 rng: random.Random,
                                 canvas: int,
                                 margin: int,
                                 node_radius: int,
                                 min_centers: int = 2,
                                 max_centers: int = 5) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, int], List[int]]:
    n = len(labels)
    span = canvas - 2 * margin
    min_node_dist = node_radius * 2.15
    min_node_dist_sq = min_node_dist * min_node_dist

    def split_sizes(total: int, k: int) -> List[int]:
        min_per = max(5, total // (k * 2))
        sizes = [min_per] * k
        remain = total - min_per * k
        for _ in range(remain):
            sizes[rng.randrange(k)] += 1
        rng.shuffle(sizes)
        return sizes

    def far_from_existing(x: float, y: float, pts: List[Tuple[float, float]]) -> bool:
        for px, py in pts:
            if (x - px) ** 2 + (y - py) ** 2 < min_node_dist_sq:
                return False
        return True

    labels_shuffled = labels[:]
    rng.shuffle(labels_shuffled)

    for _ in range(500):
        k = rng.randint(min_centers, max_centers)
        sizes = split_sizes(n, k)

        centers: List[Tuple[float, float]] = []
        min_center_dist = span * (0.20 if k <= 3 else 0.16)
        for _ci in range(k):
            placed = False
            for _try in range(400):
                cx = rng.uniform(margin + span * 0.15, canvas - margin - span * 0.15)
                cy = rng.uniform(margin + span * 0.15, canvas - margin - span * 0.15)
                if all(math.hypot(cx - ox, cy - oy) >= min_center_dist for ox, oy in centers):
                    centers.append((cx, cy))
                    placed = True
                    break
            if not placed:
                break
        if len(centers) != k:
            continue

        pts: List[Tuple[float, float]] = []
        cluster_of_idx: List[int] = []
        ok = True
        for cid, cnt in enumerate(sizes):
            cx, cy = centers[cid]
            spread = span * rng.uniform(0.055, 0.085)
            local = 0
            tries = 0
            while local < cnt and tries < 12000:
                tries += 1
                x = cx + rng.gauss(0, spread)
                y = cy + rng.gauss(0, spread)
                if x < margin + node_radius or x > canvas - margin - node_radius:
                    continue
                if y < margin + node_radius or y > canvas - margin - node_radius:
                    continue
                if not far_from_existing(x, y, pts):
                    continue
                pts.append((x, y))
                cluster_of_idx.append(cid)
                local += 1
            if local < cnt:
                ok = False
                break
        if not ok:
            continue

        pos: Dict[str, Tuple[int, int]] = {}
        cluster_of: Dict[str, int] = {}
        for lab, (x, y), cid in zip(labels_shuffled, pts, cluster_of_idx):
            pos[lab] = (int(x), int(y))
            cluster_of[lab] = cid
        return pos, cluster_of, list(range(k))

    return {}, {}, []


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
                   min_triangle_area: float = 0.0,
                   long_edge_count: int = 0,
                   min_midpoint_dist: float = 0.0,
                   node_radius: float = 0.0,
                   midpoint_exempt_len: float = 0.0) -> List[Tuple[str, str, int]]:
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

    degrees = {lab: 0 for lab in nodes}
    adj: Dict[str, List[str]] = {lab: [] for lab in nodes}
    edges: List[Tuple[str, str]] = []
    existing = set()
    segments: List[Tuple[str, str]] = []

    min_midpoint_dist_sq = min_midpoint_dist * min_midpoint_dist

    def edge_len(u: str, v: str) -> float:
        return euclid(pos[u], pos[v])

    def midpoint_dist_ok(u: str, v: str) -> bool:
        if min_midpoint_dist <= 0:
            return True
        ux, uy = pos[u]
        vx, vy = pos[v]
        mx = (ux + vx) / 2.0
        my = (uy + vy) / 2.0
        len_uv = edge_len(u, v)
        for nbr in adj[u]:
            len_un = edge_len(u, nbr)
            if midpoint_exempt_len > 0 and len_uv >= midpoint_exempt_len and len_un >= midpoint_exempt_len:
                continue
            nx, ny = pos[nbr]
            omx = (ux + nx) / 2.0
            omy = (uy + ny) / 2.0
            if (mx - omx) ** 2 + (my - omy) ** 2 < min_midpoint_dist_sq:
                return False
        for nbr in adj[v]:
            len_vn = edge_len(v, nbr)
            if midpoint_exempt_len > 0 and len_uv >= midpoint_exempt_len and len_vn >= midpoint_exempt_len:
                continue
            nx, ny = pos[nbr]
            omx = (vx + nx) / 2.0
            omy = (vy + ny) / 2.0
            if (mx - omx) ** 2 + (my - omy) ** 2 < min_midpoint_dist_sq:
                return False
        return True

    def triangle_area(u: str, v: str, w: str) -> float:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        x3, y3 = pos[w]
        return abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2.0

    def triangle_area_ok(u: str, v: str) -> bool:
        if min_triangle_area <= 0:
            return True
        # Only check triangles formed by this edge
        for w in adj[u]:
            if w in adj[v]:
                if triangle_area(u, v, w) < min_triangle_area:
                    return False
        return True

    def seg_hits_other_nodes(u: str, v: str) -> bool:
        if node_radius <= 0:
            return False
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dx = x2 - x1
        dy = y2 - y1
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-6:
            return True
        for w in nodes:
            if w == u or w == v:
                continue
            cx, cy = pos[w]
            t = ((cx - x1) * dx + (cy - y1) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            px = x1 + t * dx
            py = y1 + t * dy
            dist_sq = (cx - px) ** 2 + (cy - py) ** 2
            if dist_sq <= (node_radius + 2) ** 2:
                return True
        return False

    def add_edge(u: str, v: str, allow_cross: bool = False) -> bool:
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in existing:
            return False
        if not allow_cross and crosses_existing_labels(u, v, segments):
            return False
        if not midpoint_dist_ok(u, v):
            return False
        if not triangle_area_ok(u, v):
            return False
        if seg_hits_other_nodes(u, v):
            return False
        edges.append((a, b))
        existing.add((a, b))
        segments.append((a, b))
        degrees[u] += 1
        degrees[v] += 1
        adj[u].append(v)
        adj[v].append(u)
        return True

    # Split into outer ring and inner core (metro-like structure)
    n = len(nodes)
    cx = sum(pos[lab][0] for lab in nodes) / max(n, 1)
    cy = sum(pos[lab][1] for lab in nodes) / max(n, 1)

    dist_info: List[Tuple[float, float, str]] = []
    for lab in nodes:
        dx = pos[lab][0] - cx
        dy = pos[lab][1] - cy
        dist_info.append((math.hypot(dx, dy), math.atan2(dy, dx), lab))

    if n <= 8:
        inner_count = max(0, n - 6)
    else:
        inner_ratio = rng.uniform(0.36, 0.46)
        min_outer = 8
        min_inner = 4
        max_inner = max(min_inner, n - min_outer)
        inner_count = int(round(n * inner_ratio))
        inner_count = min(max_inner, max(min_inner, inner_count))

    dist_info.sort(key=lambda x: x[0])
    inner = [lab for _d, _a, lab in dist_info[:inner_count]]
    outer = [lab for _d, _a, lab in dist_info[inner_count:]]
    if not outer:
        outer = nodes[:]
        inner = []

    angle_map = {lab: ang for _d, ang, lab in dist_info}
    outer = sorted(outer, key=lambda lab: angle_map[lab])
    inner_set = set(inner)
    outer_set = set(outer)

    def cluster_inner_nodes(inner_list: List[str]) -> Tuple[List[List[str]], List[str], Dict[str, int]]:
        if not inner_list:
            return [], [], {}
        if len(inner_list) >= 12:
            k = rng.choice([2, 3])
        elif len(inner_list) >= 6:
            k = 2
        else:
            k = 1
        k = min(k, len(inner_list))

        centers = [inner_list[rng.randrange(len(inner_list))]]
        while len(centers) < k:
            best_lab = None
            best_dist = -1.0
            for lab in inner_list:
                d = min(edge_len(lab, c) for c in centers)
                if d > best_dist:
                    best_dist = d
                    best_lab = lab
            centers.append(best_lab if best_lab is not None else inner_list[0])

        centroids = [(pos[lab][0], pos[lab][1]) for lab in centers]
        for _ in range(6):
            clusters: List[List[str]] = [[] for _ in range(k)]
            for lab in inner_list:
                x, y = pos[lab]
                idx = min(range(k), key=lambda i: (x - centroids[i][0]) ** 2 + (y - centroids[i][1]) ** 2)
                clusters[idx].append(lab)
            for i in range(k):
                if not clusters[i]:
                    clusters[i].append(inner_list[rng.randrange(len(inner_list))])
                xs = [pos[lab][0] for lab in clusters[i]]
                ys = [pos[lab][1] for lab in clusters[i]]
                centroids[i] = (sum(xs) / len(xs), sum(ys) / len(ys))

        hubs: List[str] = []
        cluster_of: Dict[str, int] = {}
        for i, cluster in enumerate(clusters):
            cx0, cy0 = centroids[i]
            hub = min(cluster, key=lambda lab: (pos[lab][0] - cx0) ** 2 + (pos[lab][1] - cy0) ** 2)
            hubs.append(hub)
            for lab in cluster:
                cluster_of[lab] = i
        return clusters, hubs, cluster_of

    # Outer ring
    if len(outer) >= 2:
        for i in range(len(outer)):
            u = outer[i]
            v = outer[(i + 1) % len(outer)]
            if not add_edge(u, v, allow_cross=False):
                return []

    clusters, hubs, cluster_of = cluster_inner_nodes(inner)

    # Inner spanning tree (per cluster)
    if inner:
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            tree = {cluster[0]}
            while len(tree) < len(cluster):
                candidates: List[Tuple[float, str, str]] = []
                for u in tree:
                    for v in cluster:
                        if v in tree:
                            continue
                        candidates.append((edge_len(u, v), u, v))
                candidates.sort(key=lambda x: x[0])

                chosen = None
                for _d, u, v in candidates:
                    if add_edge(u, v, allow_cross=False):
                        chosen = v
                        break
                if chosen is None:
                    return []
                tree.add(chosen)

        # Connect clusters via hubs (chain by angle)
        if len(hubs) > 1:
            hubs_order = sorted(hubs, key=lambda lab: math.atan2(pos[lab][1] - cy, pos[lab][0] - cx))
            for i in range(1, len(hubs_order)):
                u = hubs_order[i - 1]
                v = hubs_order[i]
                if add_edge(u, v, allow_cross=False):
                    continue
                cand: List[Tuple[float, str, str]] = []
                for a in clusters[cluster_of[u]]:
                    for b in clusters[cluster_of[v]]:
                        cand.append((edge_len(a, b), a, b))
                cand.sort(key=lambda x: x[0])
                added = False
                for _d, a, b in cand:
                    if add_edge(a, b, allow_cross=False):
                        added = True
                        break
                if not added:
                    return []

    # Spokes: connect outer nodes to the inner core
    if inner:
        inner_by_center = sorted(inner, key=lambda lab: euclid(pos[lab], (cx, cy)))
        hub_set = set(hubs)
        for o in outer:
            candidates = sorted(hubs, key=lambda lab: edge_len(o, lab))
            if len(candidates) < len(inner_by_center):
                candidates += [lab for lab in inner_by_center if lab not in hub_set]
            added = False
            for v in candidates:
                if add_edge(o, v, allow_cross=False):
                    added = True
                    break
            if not added:
                return []

    # Add a few extra edges (only inner-inner or outer-inner)
    def max_degree(lab: str) -> int:
        return 4 if lab in outer_set else 6

    all_pairs: List[Tuple[float, str, str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            u = nodes[i]
            v = nodes[j]
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in existing:
                continue
            if u in outer_set and v in outer_set:
                continue
            all_pairs.append((edge_len(u, v), u, v))
    all_pairs.sort(key=lambda x: x[0])

    added = 0
    if long_edge_count > 0:
        for _d, u, v in reversed(all_pairs):
            if added >= min(long_edge_count, extra_edges):
                break
            if degrees[u] >= max_degree(u) or degrees[v] >= max_degree(v):
                continue
            if add_edge(u, v, allow_cross=False):
                added += 1

    for _d, u, v in all_pairs:
        if added >= extra_edges:
            break
        if degrees[u] >= max_degree(u) or degrees[v] >= max_degree(v):
            continue
        if add_edge(u, v, allow_cross=False):
            added += 1

    return assign_weights_by_length(edges, pos)


def generate_multicenter_graph(labels: List[str],
                               pos: Dict[str, Tuple[int, int]],
                               cluster_of: Dict[str, int],
                               rng: random.Random,
                               node_radius: float = 0.0,
                               min_midpoint_dist: float = 0.0,
                               midpoint_exempt_len: float = 0.0) -> List[Tuple[str, str, int]]:
    nodes = labels[:]
    clusters: Dict[int, List[str]] = {}
    for lab in nodes:
        cid = cluster_of.get(lab, 0)
        clusters.setdefault(cid, []).append(lab)
    if not clusters:
        return []

    degrees = {lab: 0 for lab in nodes}
    adj: Dict[str, List[str]] = {lab: [] for lab in nodes}
    edges: List[Tuple[str, str]] = []
    existing = set()
    segments: List[Tuple[str, str]] = []

    min_midpoint_dist_sq = min_midpoint_dist * min_midpoint_dist

    def edge_len(u: str, v: str) -> float:
        return euclid(pos[u], pos[v])

    def crosses_existing_labels(u: str, v: str, segs: List[Tuple[str, str]]) -> bool:
        p1 = (pos[u][0], pos[u][1])
        p2 = (pos[v][0], pos[v][1])
        for a, b in segs:
            if u == a or u == b or v == a or v == b:
                continue
            if seg_intersect(p1, p2, (pos[a][0], pos[a][1]), (pos[b][0], pos[b][1])):
                return True
        return False

    def midpoint_dist_ok(u: str, v: str) -> bool:
        if min_midpoint_dist <= 0:
            return True
        ux, uy = pos[u]
        vx, vy = pos[v]
        mx = (ux + vx) / 2.0
        my = (uy + vy) / 2.0
        len_uv = edge_len(u, v)
        for nbr in adj[u]:
            len_un = edge_len(u, nbr)
            if midpoint_exempt_len > 0 and len_uv >= midpoint_exempt_len and len_un >= midpoint_exempt_len:
                continue
            nx, ny = pos[nbr]
            omx = (ux + nx) / 2.0
            omy = (uy + ny) / 2.0
            if (mx - omx) ** 2 + (my - omy) ** 2 < min_midpoint_dist_sq:
                return False
        for nbr in adj[v]:
            len_vn = edge_len(v, nbr)
            if midpoint_exempt_len > 0 and len_uv >= midpoint_exempt_len and len_vn >= midpoint_exempt_len:
                continue
            nx, ny = pos[nbr]
            omx = (vx + nx) / 2.0
            omy = (vy + ny) / 2.0
            if (mx - omx) ** 2 + (my - omy) ** 2 < min_midpoint_dist_sq:
                return False
        return True

    def seg_hits_other_nodes(u: str, v: str) -> bool:
        if node_radius <= 0:
            return False
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dx = x2 - x1
        dy = y2 - y1
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-6:
            return True
        for w in nodes:
            if w == u or w == v:
                continue
            cx, cy = pos[w]
            t = ((cx - x1) * dx + (cy - y1) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            px = x1 + t * dx
            py = y1 + t * dy
            if (cx - px) ** 2 + (cy - py) ** 2 <= (node_radius + 2) ** 2:
                return True
        return False

    def add_edge(u: str, v: str) -> bool:
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in existing:
            return False
        if crosses_existing_labels(u, v, segments):
            return False
        if not midpoint_dist_ok(u, v):
            return False
        if seg_hits_other_nodes(u, v):
            return False
        edges.append((a, b))
        existing.add((a, b))
        segments.append((a, b))
        degrees[u] += 1
        degrees[v] += 1
        adj[u].append(v)
        adj[v].append(u)
        return True

    def max_degree(lab: str) -> int:
        return 6

    # Intra-cluster tree + extra local edges
    for _cid, cnodes in clusters.items():
        if len(cnodes) <= 1:
            continue
        tree = {cnodes[0]}
        while len(tree) < len(cnodes):
            cand: List[Tuple[float, str, str]] = []
            for u in tree:
                for v in cnodes:
                    if v in tree:
                        continue
                    cand.append((edge_len(u, v), u, v))
            cand.sort(key=lambda x: x[0])
            added = False
            for _d, u, v in cand:
                if degrees[u] >= max_degree(u) or degrees[v] >= max_degree(v):
                    continue
                if add_edge(u, v):
                    tree.add(v)
                    added = True
                    break
            if not added:
                return []

        intra_target = max(2, len(cnodes) // 3)
        intra_added = 0
        cand2: List[Tuple[float, str, str]] = []
        for i in range(len(cnodes)):
            for j in range(i + 1, len(cnodes)):
                u = cnodes[i]
                v = cnodes[j]
                a, b = (u, v) if u < v else (v, u)
                if (a, b) in existing:
                    continue
                cand2.append((edge_len(u, v), u, v))
        cand2.sort(key=lambda x: x[0])
        for _d, u, v in cand2:
            if intra_added >= intra_target:
                break
            if degrees[u] >= max_degree(u) or degrees[v] >= max_degree(v):
                continue
            if add_edge(u, v):
                intra_added += 1

    # Sparse inter-cluster links: only 2-3 lines total
    cids = sorted(clusters.keys())
    centers: Dict[int, Tuple[float, float]] = {}
    for cid, cnodes in clusters.items():
        xs = [pos[lab][0] for lab in cnodes]
        ys = [pos[lab][1] for lab in cnodes]
        centers[cid] = (sum(xs) / len(xs), sum(ys) / len(ys))

    cluster_pairs: List[Tuple[float, int, int]] = []
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            a = cids[i]
            b = cids[j]
            cluster_pairs.append((euclid((int(centers[a][0]), int(centers[a][1])), (int(centers[b][0]), int(centers[b][1]))), a, b))
    cluster_pairs.sort(key=lambda x: x[0])

    inter_target = rng.randint(2, 3)
    inter_target = min(inter_target, len(cluster_pairs))
    inter_added = 0
    used_pairs = set()
    for _dist, ca, cb in cluster_pairs:
        if inter_added >= inter_target:
            break
        if (ca, cb) in used_pairs:
            continue
        cand: List[Tuple[float, str, str]] = []
        for u in clusters[ca]:
            for v in clusters[cb]:
                cand.append((edge_len(u, v), u, v))
        cand.sort(key=lambda x: x[0])
        linked = False
        for _d, u, v in cand:
            if degrees[u] >= max_degree(u) or degrees[v] >= max_degree(v):
                continue
            if add_edge(u, v):
                inter_added += 1
                used_pairs.add((ca, cb))
                linked = True
                break
        if not linked:
            continue

    if inter_added < 2:
        return []

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
        base = int(round(n * rng.uniform(0.30, 0.40)))
        base = max(2, base)
        return min(base, n - 4)

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

        # inner points (2-3 centers)
        inner_pts: List[Tuple[float, float]] = []
        if inner_count > 0:
            inner_r = rng.uniform(0.32, 0.60)
            if inner_count >= 10:
                k_centers = rng.choice([2, 3])
            elif inner_count >= 6:
                k_centers = 2
            else:
                k_centers = 1
            k_centers = min(k_centers, inner_count)

            centers: List[Tuple[float, float]] = []
            center_r = rng.uniform(inner_r * 0.4, inner_r * 0.85)
            center_min_dist = inner_r * 0.45
            tries = 0
            while len(centers) < k_centers and tries < 2000:
                tries += 1
                rr = center_r * math.sqrt(rng.random())
                aa = rng.uniform(0, 2 * math.pi)
                cx0 = rr * math.cos(aa)
                cy0 = rr * math.sin(aa)
                if all((cx0 - x) ** 2 + (cy0 - y) ** 2 >= center_min_dist ** 2 for x, y in centers):
                    centers.append((cx0, cy0))
            if len(centers) < k_centers:
                centers = [(0.0, 0.0)]
                k_centers = 1

            counts = [inner_count // k_centers] * k_centers
            for i in range(inner_count % k_centers):
                counts[i] += 1

            for (cx0, cy0), cnt in zip(centers, counts):
                spread = rng.uniform(0.08, 0.12)
                tries = 0
                while cnt > 0 and tries < 8000:
                    tries += 1
                    x = cx0 + rng.gauss(0, spread)
                    y = cy0 + rng.gauss(0, spread)
                    if x * x + y * y > inner_r * inner_r:
                        continue
                    x, y = apply_affine(x, y)
                    if not far_enough(x, y, pts) or not far_enough(x, y, inner_pts):
                        continue
                    inner_pts.append((x, y))
                    cnt -= 1

            if len(inner_pts) < inner_count:
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
        parts.append(f"d({u},{v})={w}")
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


def weight_label_boxes(draw: ImageDraw.ImageDraw,
                       pos: Dict[str, Tuple[int, int]],
                       edges: List[Tuple[str, str, int]],
                       font_w: ImageFont.FreeTypeFont,
                       node_radius: int,
                       canvas: int) -> List[Tuple[str, str, int, Tuple[float, float, float, float]]]:
    boxes = []
    for u, v, w in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        text = str(w)
        tw, th = text_wh(draw, text, font_w)
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        dx = x2 - x1
        dy = y2 - y1
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-6:
            continue
        nx = -dy / seg_len
        ny = dx / seg_len
        offset = max(node_radius * 0.55, math.hypot(tw, th) / 2 + 6)

        candidates = []
        for sign in (-1, 1):
            tx = mx + nx * offset * sign - tw / 2
            ty = my + ny * offset * sign - th / 2
            tx = min(max(2, tx), canvas - tw - 2)
            ty = min(max(2, ty), canvas - th - 2)
            cx = tx + tw / 2
            cy = ty + th / 2
            min_dist = float("inf")
            for lab, (px, py) in pos.items():
                if lab == u or lab == v:
                    continue
                min_dist = min(min_dist, math.hypot(cx - px, cy - py))
            candidates.append((min_dist, (tx, ty, tx + tw, ty + th)))

        best = max(candidates, key=lambda t: t[0])[1]
        boxes.append((u, v, w, best))
    return boxes


def is_graph_clear(labels: List[str],
                   pos: Dict[str, Tuple[int, int]],
                   edges: List[Tuple[str, str, int]],
                   node_radius: int,
                   canvas: int,
                   draw: ImageDraw.ImageDraw,
                   font_w: ImageFont.FreeTypeFont) -> bool:
    # node overlap
    min_node_dist = node_radius * 2.0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a = labels[i]
            b = labels[j]
            if euclid(pos[a], pos[b]) < min_node_dist:
                return False

    # edge overlap / too close (non-adjacent)
    min_edge_sep = 1.5
    edge_pairs = [(u, v) for (u, v, _w) in edges]
    for i in range(len(edge_pairs)):
        u1, v1 = edge_pairs[i]
        p1 = pos[u1]
        p2 = pos[v1]
        for j in range(i + 1, len(edge_pairs)):
            u2, v2 = edge_pairs[j]
            if u1 in (u2, v2) or v1 in (u2, v2):
                continue
            q1 = pos[u2]
            q2 = pos[v2]
            if segment_distance(p1, p2, q1, q2) < min_edge_sep:
                return False

    # label clarity
    boxes = weight_label_boxes(draw, pos, edges, font_w, node_radius, canvas)

    # label-label overlap
    for i in range(len(boxes)):
        _u1, _v1, _w1, b1 = boxes[i]
        for j in range(i + 1, len(boxes)):
            _u2, _v2, _w2, b2 = boxes[j]
            if rects_overlap(b1, b2, pad=2):
                return False

    # label vs nodes / edges
    for u, v, _w, box in boxes:
        for lab, (cx, cy) in pos.items():
            if rect_circle_overlap(box, cx, cy, node_radius + 2):
                return False
        for a, b, _w2 in edges:
            if (a == u and b == v) or (a == v and b == u):
                p1 = pos[a]
                p2 = pos[b]
                if seg_intersects_rect(p1, p2, box, pad=0):
                    return False
                break

    return True


def draw_weights(draw: ImageDraw.ImageDraw,
                 pos: Dict[str, Tuple[int, int]],
                 edges: List[Tuple[str, str, int]],
                 font_w: ImageFont.FreeTypeFont,
                 rng: random.Random,
                 node_radius: int,
                 canvas: int):
    boxes = weight_label_boxes(draw, pos, edges, font_w, node_radius, canvas)
    for _u, _v, w, box in boxes:
        x1, y1, _x2, _y2 = box
        draw.text((x1, y1), str(w), fill=WEIGHT_TEXT_COLOR, font=font_w)


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
def pick_unique_queries(labels: List[str],
                        edges: List[Tuple[str, str, int]],
                        rng: random.Random,
                        count: int) -> List[Tuple[str, str, List[str], int]]:
    results: List[Tuple[str, str, List[str], int]] = []
    seen = set()
    max_tries = MAX_QUERY_TRIES * max(1, count)
    for _ in range(max_tries):
        if len(results) >= count:
            break
        s, t = rng.sample(labels, 2)
        key = tuple(sorted((s, t)))
        if key in seen:
            continue
        path, d, cnt = dijkstra_unique_path(edges, s, t)
        if len(path) >= 2 and cnt == 1:
            results.append((s, t, path, d))
            seen.add(key)
    return results


def remove_existing_multicenter_samples():
    for folder in (IMG_ORIG_DIR, IMG_ANN_DIR):
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if "_mc_" in name and name.endswith(".png"):
                try:
                    os.remove(os.path.join(folder, name))
                except OSError:
                    pass

    if os.path.isdir(EDGE_INFO_DIR):
        for name in os.listdir(EDGE_INFO_DIR):
            if "_mc_" in name and name.endswith(".json"):
                try:
                    os.remove(os.path.join(EDGE_INFO_DIR, name))
                except OSError:
                    pass

    if os.path.exists(JSONL_PATH):
        kept = []
        with open(JSONL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    item = json.loads(s)
                except json.JSONDecodeError:
                    continue
                style = item.get("graph_style")
                ipath = item.get("image_path", "")
                if style == "multicenter_no_ring" or "_mc_" in ipath:
                    continue
                kept.append(item)
        if kept:
            with open(JSONL_PATH, "w", encoding="utf-8") as f:
                for item in kept:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            try:
                os.remove(JSONL_PATH)
            except OSError:
                pass


def refresh_edge_info_texts():
    if not os.path.isdir(EDGE_INFO_DIR):
        return
    for name in os.listdir(EDGE_INFO_DIR):
        if not name.endswith(".json"):
            continue
        path = os.path.join(EDGE_INFO_DIR, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                item = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        edges_obj = item.get("edges", [])
        edges = []
        for e in edges_obj:
            if isinstance(e, dict) and "u" in e and "v" in e and "w" in e:
                edges.append((e["u"], e["v"], int(e["w"])))
        item["edge_list_text"] = edge_list_text(edges)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(item, f, ensure_ascii=False)


def append_extra_multicenter_samples():
    rng = random.Random(SEED + 999)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(IMG_ORIG_DIR, exist_ok=True)
    os.makedirs(IMG_ANN_DIR, exist_ok=True)
    os.makedirs(EDGE_INFO_DIR, exist_ok=True)
    remove_existing_multicenter_samples()
    refresh_edge_info_texts()

    n = EXTRA_MULTICENTER_LEVEL
    canvas = CANVAS_MAP[n]
    margin = MARGIN_MAP[n]
    node_radius = NODE_RADIUS_MAP[n]
    min_mid_dist = MIN_MIDPOINT_DIST.get(n, 0.0)
    midpoint_exempt = MIDPOINT_EXEMPT_LEN.get(n, 0.0)
    font_w = load_font(FONT_W_SIZE)
    clear_draw = ImageDraw.Draw(Image.new("RGB", (canvas, canvas), BG_COLOR))

    base_id = get_next_base_id()
    case_id = get_next_case_id()
    records = []

    for _ in range(EXTRA_MULTICENTER_COUNT):
        labels = make_node_labels(n)
        pos = {}
        cluster_of: Dict[str, int] = {}
        cluster_ids: List[int] = []
        edges: List[Tuple[str, str, int]] = []
        queries: List[Tuple[str, str, List[str], int]] = []
        img_ann = None

        for _try in range(MAX_GRAPH_TRIES):
            pos, cluster_of, cluster_ids = layout_multicenter_positions(
                labels, rng, canvas=canvas, margin=margin, node_radius=node_radius, min_centers=2, max_centers=4
            )
            if not pos:
                continue
            edges = generate_multicenter_graph(
                labels,
                pos,
                cluster_of,
                rng=rng,
                node_radius=node_radius,
                min_midpoint_dist=min_mid_dist,
                midpoint_exempt_len=midpoint_exempt,
            )
            if not edges:
                continue
            if not is_connected(labels, edges):
                continue
            if not is_graph_clear(labels, pos, edges, node_radius=node_radius, canvas=canvas, draw=clear_draw, font_w=font_w):
                continue
            queries = pick_unique_queries(labels, edges, rng, count=EXTRA_MULTICENTER_QUERY_COUNT)
            if len(queries) < EXTRA_MULTICENTER_QUERY_COUNT:
                continue
            img_ann = render_image(labels, pos, edges, annotate_weights=True, rng=rng, canvas=canvas, node_radius=node_radius)
            break
        else:
            raise RuntimeError("Failed to generate extra multicenter sample.")

        orig_name = f"g{base_id:03d}_n{n}_mc_orig.png"
        ann_name = f"g{base_id:03d}_n{n}_mc_ann.png"
        orig_path = os.path.join(IMG_ORIG_DIR, orig_name)
        ann_path = os.path.join(IMG_ANN_DIR, ann_name)

        img_orig = render_image(labels, pos, edges, annotate_weights=False, rng=rng, canvas=canvas, node_radius=node_radius)
        img_orig.save(orig_path)
        if img_ann is None:
            img_ann = render_image(labels, pos, edges, annotate_weights=True, rng=rng, canvas=canvas, node_radius=node_radius)
        img_ann.save(ann_path)

        edge_info = {
            "base_graph_id": base_id,
            "level_nodes": n,
            "graph_style": "multicenter_no_ring",
            "cluster_count": len(set(cluster_ids)),
            "cluster_of": cluster_of,
            "nodes": labels,
            "edges": [{"u": u, "v": v, "w": w} for (u, v, w) in edges],
            "edge_list_text": edge_list_text(edges),
            "image_original": os.path.join("images_original", orig_name),
            "image_annotated": os.path.join("images_annotated", ann_name),
        }
        with open(os.path.join(EDGE_INFO_DIR, f"g{base_id:03d}_n{n}_mc_edges.json"), "w", encoding="utf-8") as ef:
            json.dump(edge_info, ef, ensure_ascii=False)

        for s, t, gt_path, _gt_dist in queries:
            rec_text = {
                "case_id": case_id,
                "base_graph_id": base_id,
                "level_nodes": n,
                "graph_style": "multicenter_no_ring",
                "condition": "text",
                "image_path": os.path.join("images_original", orig_name),
                "query": {"source": s, "target": t},
                "ground_truth": {"path": gt_path},
                "edge_list_text": edge_list_text(edges),
                "edges": [{"u": u, "v": v, "w": w} for (u, v, w) in edges]
            }
            records.append(rec_text)
            case_id += 1

            rec_image = {
                "case_id": case_id,
                "base_graph_id": base_id,
                "level_nodes": n,
                "graph_style": "multicenter_no_ring",
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

    mode = "a" if os.path.exists(JSONL_PATH) else "w"
    with open(JSONL_PATH, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Done append extra multicenter samples.")
    print(f"Added images: {EXTRA_MULTICENTER_COUNT}")
    print(f"Added JSONL lines: {len(records)}")


# ---------------------------
# Main
# ---------------------------
def main():
    if APPEND_EXTRA_MULTICENTER_ONLY:
        append_extra_multicenter_samples()
        return

    rng = random.Random(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(IMG_ORIG_DIR, exist_ok=True)
    os.makedirs(IMG_ANN_DIR, exist_ok=True)
    os.makedirs(EDGE_INFO_DIR, exist_ok=True)

    records = []
    base_id = 0
    case_id = 0

    for n in LEVELS:
        canvas = CANVAS_MAP[n]
        margin = MARGIN_MAP[n]
        node_radius = NODE_RADIUS_MAP[n]
        extra_edges = EXTRA_EDGE_COUNT[n]
        min_mid_dist = MIN_MIDPOINT_DIST.get(n, 0.0)
        min_triangle_area = MIN_TRIANGLE_AREA.get(n, 0.0)
        long_edges = LONG_EDGE_COUNT.get(n, 0)
        midpoint_exempt = MIDPOINT_EXEMPT_LEN.get(n, 0.0)
        target_count = LEVEL_COUNTS[n]
        font_w = load_font(FONT_W_SIZE)
        clear_draw = ImageDraw.Draw(Image.new("RGB", (canvas, canvas), BG_COLOR))

        for _ in range(target_count):
            labels = make_node_labels(n)
            edges = []
            pos = {}
            img_ann = None
            queries: List[Tuple[str, str, List[str], int]] = []

            for _try in range(MAX_GRAPH_TRIES):
                pos = ring_positions(labels, rng, canvas=canvas, margin=margin, node_radius=node_radius)
                edges = generate_graph(
                    labels,
                    pos,
                    extra_edges=extra_edges,
                    rng=rng,
                    min_triangle_area=min_triangle_area,
                    long_edge_count=long_edges,
                    min_midpoint_dist=min_mid_dist,
                    node_radius=node_radius,
                    midpoint_exempt_len=midpoint_exempt,
                )
                if not edges:
                    continue

                if n == 4 and len(edges) < 5:
                    continue

                if not is_connected(labels, edges):
                    continue

                if not is_graph_clear(labels, pos, edges, node_radius=node_radius, canvas=canvas, draw=clear_draw, font_w=font_w):
                    continue

                if GENERATE_QUERIES:
                    queries = pick_unique_queries(labels, edges, rng, count=NUM_QUERIES_PER_IMAGE)
                    if len(queries) < NUM_QUERIES_PER_IMAGE:
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

            edge_info = {
                "base_graph_id": base_id,
                "level_nodes": n,
                "nodes": labels,
                "edges": [{"u": u, "v": v, "w": w} for (u, v, w) in edges],
                "edge_list_text": edge_list_text(edges),
                "image_original": os.path.join("images_original", orig_name),
                "image_annotated": os.path.join("images_annotated", ann_name),
            }
            with open(os.path.join(EDGE_INFO_DIR, f"g{base_id:03d}_n{n}_edges.json"), "w", encoding="utf-8") as ef:
                json.dump(edge_info, ef, ensure_ascii=False)

            if GENERATE_QUERIES:
                for s, t, gt_path, _gt_dist in queries:
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

    if GENERATE_QUERIES:
        with open(JSONL_PATH, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Done.")
    total_images = sum(LEVEL_COUNTS.values())
    print(f"Original images:   {IMG_ORIG_DIR} (should be {total_images})")
    print(f"Annotated images:  {IMG_ANN_DIR} (should be {total_images})")
    if GENERATE_QUERIES:
        print(f"JSONL cases:       {JSONL_PATH} (should be {total_images * 10} lines)")
    else:
        print(f"Edge info files:   {EDGE_INFO_DIR} (should be {total_images} files)")
    print("Levels:", {n: LEVEL_COUNTS[n] for n in LEVELS})


if __name__ == "__main__":
    main()
