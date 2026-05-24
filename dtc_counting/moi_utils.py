import math
from typing import Dict, List, Tuple

import numpy as np

Point = Tuple[float, float]
Vector = Tuple[Point, Point]


def load_moi_vectors(path: str) -> Dict[int, Vector]:
    vectors: Dict[int, Vector] = {}
    if not path:
        return vectors
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) != 5:
                continue
            mid = int(float(parts[0]))
            x1, y1, x2, y2 = [float(p) for p in parts[1:]]
            vectors[mid] = ((x1, y1), (x2, y2))
    return vectors


def write_moi_vectors(path: str, vectors: Dict[int, Vector]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for mid in sorted(vectors):
            (x1, y1), (x2, y2) = vectors[mid]
            f.write(f"{mid},{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}\n")


def vector_norm(vec: Vector) -> float:
    (x1, y1), (x2, y2) = vec
    return float(math.hypot(x2 - x1, y2 - y1))


def _score(candidate: Vector, reference: Vector) -> float:
    cs, ce = candidate
    rs, re = reference
    cv = np.array([ce[0] - cs[0], ce[1] - cs[1]], dtype=np.float32)
    rv = np.array([re[0] - rs[0], re[1] - rs[1]], dtype=np.float32)
    cn = float(np.linalg.norm(cv))
    rn = float(np.linalg.norm(rv))
    if cn < 1e-6 or rn < 1e-6:
        return float("inf")
    cos = float(np.dot(cv, rv) / (cn * rn))
    cos = max(-1.0, min(1.0, cos))
    angle = math.acos(cos)
    dist = math.hypot(cs[0] - rs[0], cs[1] - rs[1]) + math.hypot(ce[0] - re[0], ce[1] - re[1])
    return angle * 100.0 + dist


def _best_orientation(candidate: Vector, reference: Vector) -> Tuple[float, Vector]:
    forward = _score(candidate, reference)
    reversed_candidate = (candidate[1], candidate[0])
    backward = _score(reversed_candidate, reference)
    if backward < forward:
        return backward, reversed_candidate
    return forward, candidate


def align_to_reference(generated: Dict[int, Vector], reference: Dict[int, Vector]) -> Dict[int, Vector]:
    """Return generated vectors keyed by the closest official/reference MOI ids.

    Generated PCA/SAM vectors often have arbitrary ids, and PCA vectors can point
    in either direction. This alignment gives evaluation the same movement-id
    semantics as the ground truth when a reference MOI file is available.
    """
    gen_items = [(mid, vec) for mid, vec in sorted(generated.items()) if vector_norm(vec) >= 1e-6]
    ref_items = [(mid, vec) for mid, vec in sorted(reference.items()) if vector_norm(vec) >= 1e-6]
    if not gen_items or not ref_items:
        return {}

    cost = np.zeros((len(gen_items), len(ref_items)), dtype=np.float32)
    oriented: List[List[Vector]] = []
    for gi, (_, gvec) in enumerate(gen_items):
        row_oriented: List[Vector] = []
        for ri, (_, rvec) in enumerate(ref_items):
            pair_cost, candidate = _best_orientation(gvec, rvec)
            cost[gi, ri] = pair_cost
            row_oriented.append(candidate)
        oriented.append(row_oriented)

    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
    except Exception:
        pairs = []
        used_rows = set()
        used_cols = set()
        flat = sorted((float(cost[r, c]), r, c) for r in range(cost.shape[0]) for c in range(cost.shape[1]))
        for _, r, c in flat:
            if r in used_rows or c in used_cols:
                continue
            used_rows.add(r)
            used_cols.add(c)
            pairs.append((r, c))
            if len(used_rows) == min(cost.shape):
                break
        rows = np.array([r for r, _ in pairs], dtype=int)
        cols = np.array([c for _, c in pairs], dtype=int)

    aligned: Dict[int, Vector] = {}
    for r, c in zip(rows.tolist(), cols.tolist()):
        ref_id = ref_items[c][0]
        aligned[ref_id] = oriented[r][c]
    return aligned
