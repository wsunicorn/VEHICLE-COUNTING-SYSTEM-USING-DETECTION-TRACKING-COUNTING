import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import SAM, YOLO


Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]

# ------------------------------------------------------------------
# Thresholds for identifying genuinely-moving (on-road) vehicles.
# Parked cars, slow parking-lot crawlers, and circular maneuvers
# must all be rejected so they cannot pollute the ROI polygon.
#
# _MIN_MOVING_DISP   : net start→end displacement in pixels
# _MIN_PATH_LENGTH   : cumulative path length in pixels
# _MIN_STRAIGHTNESS  : displacement / path_length ratio
#                      (0 = full circle, 1 = perfectly straight)
#                      Parking maneuvers ≈ 0.1-0.3; road ≈ 0.5-1.0
# _TOP_MARGIN_FRAC   : ignore trail points in the top N% of the frame
#                      (sky, distant background, far parking lots)
# ------------------------------------------------------------------
_MIN_MOVING_DISP  = 80.0
_MIN_PATH_LENGTH  = 130.0
_MIN_STRAIGHTNESS = 0.40
_TOP_MARGIN_FRAC  = 0.20
_MIN_TRAIL_LEN    = 10   # minimum frames a trail must span


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap ROI and MOI using SAM + YOLO prompts.")
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument("--weights", required=True, help="Path to detector weights (best2.pt).")
    parser.add_argument("--output-json", required=True, help="Output JSON for ROI and MOI vectors.")
    parser.add_argument("--sam-model", default="sam_b.pt", help="SAM checkpoint for ultralytics SAM.")
    parser.add_argument("--max-frames", type=int, default=120, help="Number of initial frames for bootstrapping.")
    parser.add_argument("--conf", type=float, default=0.3, help="Detector confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Detector image size.")
    parser.add_argument("--moi-count", type=int, default=8, help="Expected number of MOIs.")
    parser.add_argument("--moi-min-len", type=float, default=50.0, help="Min pixel length of a trajectory vector to become a MOI candidate.")
    parser.add_argument("--roi-expand", type=float, default=1.03, help="Scale factor to slightly expand ROI polygon.")
    parser.add_argument(
        "--roi-shape",
        default="rect",
        choices=["poly", "rect"],
        help="'rect' = axis-aligned bounding rectangle (neat 4-vertex box); 'poly' = simplified convex hull.",
    )
    parser.add_argument("--top-margin", type=float, default=_TOP_MARGIN_FRAC,
                        help="Fraction of frame height to exclude from the top when building ROI "
                             "(removes sky, far background, parking lots at top of frame).")
    parser.add_argument("--save-overlay", default="", help="Optional overlay image path.")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────
# Lightweight centroid tracker (no Kalman needed for bootstrap)
# ──────────────────────────────────────────────────────────────────

class _CentroidTracker:
    """
    Greedy nearest-centroid tracker used to identify moving vehicles.
    Tracks accumulate a trail of center points.  When a track goes
    stale it is moved to ``completed_trails`` for later analysis.
    """

    def __init__(
        self,
        match_dist: float = 90.0,
        max_gap: int = 8,
        min_moving_disp: float = _MIN_MOVING_DISP,
    ) -> None:
        self.match_dist       = match_dist
        self.max_gap          = max_gap
        self.min_moving_disp  = min_moving_disp
        self._tracks: Dict[int, dict] = {}
        self._next_id         = 0
        # All trails (moving + stationary) with length >= _MIN_TRAIL_LEN
        self.completed_trails: List[List[Point]] = []

    # ------------------------------------------------------------------
    def update(self, dets: List[Tuple[Point, BBox]]) -> None:
        """
        dets: list of ((cx, cy), (x1, y1, x2, y2))
        Updates internal track table and moves stale tracks to
        ``completed_trails``.
        """
        # Age all tracks
        for t in self._tracks.values():
            t["gap"] += 1

        # Drop stale tracks → archive their trails
        stale = [k for k, v in self._tracks.items() if v["gap"] > self.max_gap]
        for k in stale:
            trail = self._tracks[k]["trail"]
            if len(trail) >= _MIN_TRAIL_LEN:
                self.completed_trails.append(trail)
            del self._tracks[k]

        # Greedy nearest-centroid matching
        used: set = set()
        for (cx, cy), box in dets:
            best_id, best_d = None, float("inf")
            for tid, t in self._tracks.items():
                if tid in used:
                    continue
                lx, ly = t["last_center"]
                d = math.hypot(cx - lx, cy - ly)
                if d < best_d and d < self.match_dist:
                    best_d, best_id = d, tid

            if best_id is not None:
                t = self._tracks[best_id]
                t["last_center"] = (cx, cy)
                t["last_box"]    = box
                t["trail"].append((cx, cy))
                t["gap"] = 0
                t["age"] += 1
                used.add(best_id)
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {
                    "first_center": (cx, cy),
                    "last_center":  (cx, cy),
                    "last_box":     box,
                    "trail":        [(cx, cy)],
                    "gap":          0,
                    "age":          1,
                }

    # ------------------------------------------------------------------
    def moving_tracks(self) -> List[dict]:
        """
        Return active tracks whose cumulative displacement exceeds
        ``min_moving_disp`` — these are genuinely-moving vehicles.
        """
        result = []
        for t in self._tracks.values():
            if t["age"] < 3:
                continue
            fx, fy = t["first_center"]
            lx, ly = t["last_center"]
            if math.hypot(lx - fx, ly - fy) >= self.min_moving_disp:
                result.append(t)
        return result

    # ------------------------------------------------------------------
    def finalize(self) -> None:
        """Flush remaining live tracks into ``completed_trails``."""
        for t in self._tracks.values():
            if len(t["trail"]) >= _MIN_TRAIL_LEN:
                self.completed_trails.append(t["trail"])
        self._tracks.clear()

    # ------------------------------------------------------------------
    def moving_completed_trails(self) -> List[List[Point]]:
        """
        Return completed trails that clearly belong to on-road through-
        traffic: minimum net displacement, minimum total path length,
        and minimum straightness ratio.  This combination rejects:
          - parked cars (disp too small)
          - slow parking-lot crawlers (path length too small)
          - circular parking maneuvers (straightness too low)
        """
        out = []
        for trail in self.completed_trails:
            if len(trail) < _MIN_TRAIL_LEN:
                continue
            fx, fy = trail[0]
            lx, ly = trail[-1]
            disp = math.hypot(lx - fx, ly - fy)
            if disp < self.min_moving_disp:
                continue
            # Cumulative path length
            path_len = sum(
                math.hypot(trail[i + 1][0] - trail[i][0],
                           trail[i + 1][1] - trail[i][1])
                for i in range(len(trail) - 1)
            )
            if path_len < _MIN_PATH_LENGTH:
                continue
            # Straightness: filters out U-turns and parking maneuvers
            if path_len > 0 and (disp / path_len) < _MIN_STRAIGHTNESS:
                continue
            out.append(trail)
        return out


# ──────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────

def _is_vehicle(label: str) -> bool:
    kws = ("car", "truck", "bus", "van", "vehicle", "motor", "pickup")
    return any(kw in label for kw in kws)


def filter_mask_components(mask: np.ndarray, min_area: int = 800, max_components: int = 1) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8
    areas = [(i, int(stats[i, cv2.CC_STAT_AREA])) for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not areas:
        return mask_u8
    areas.sort(key=lambda x: x[1], reverse=True)
    keep = {idx for idx, _ in areas[:max_components]}
    out = np.zeros_like(mask_u8)
    for idx in keep:
        out[labels == idx] = 1
    return out


def expand_polygon(poly: np.ndarray, w: int, h: int, scale: float) -> np.ndarray:
    if poly.size == 0:
        return poly
    pts    = poly.astype(np.float32)
    center = pts.mean(axis=0, keepdims=True)
    expanded = (pts - center) * float(scale) + center
    expanded[:, 0] = np.clip(expanded[:, 0], 0, w - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, h - 1)
    return expanded.astype(np.int32)


def normalize_vectors_2d(arr: np.ndarray, dim: int = 4) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, dim), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim == 2 and arr.shape[1] == 1 and arr.shape[0] == dim:
        arr = arr.reshape(1, dim)
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    if arr.ndim != 2:
        return np.zeros((0, dim), dtype=np.float32)
    if arr.shape[1] < dim:
        pad = np.zeros((arr.shape[0], dim - arr.shape[1]), dtype=np.float32)
        arr = np.hstack([arr, pad])
    return arr[:, :dim]


def sort_vector_centers(centers: np.ndarray) -> np.ndarray:
    centers = normalize_vectors_2d(centers, dim=4)
    if centers.shape[0] == 0:
        return centers
    angles = np.arctan2(
        centers[:, 3] - centers[:, 1],
        centers[:, 2] - centers[:, 0],
    )
    return centers[np.argsort(angles)]


def kmeans_moi(vectors: List[np.ndarray], k: int) -> np.ndarray:
    if not vectors:
        return np.zeros((0, 4), dtype=np.float32)
    data = normalize_vectors_2d(np.vstack(vectors), dim=4)
    if data.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    if data.shape[0] == 1:
        return data
    k = max(1, min(k, len(data)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.05)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    return sort_vector_centers(centers)


# ──────────────────────────────────────────────────────────────────
# ROI polygon fitting — DIRECT TRAIL-POINT APPROACH
#
# Instead of going through a binary mask → contour → hull (which
# produces jagged/fragmented results when the mask has noise), we
# take the convex hull of every trail point from qualifying vehicles.
#
# Why this works:
#   • Moving vehicles travel only on the road surface.
#   • After filtering for displacement + path-length + straightness,
#     only true through-traffic trails remain.
#   • convexHull of those points = the road surface, by construction.
#   • The resulting polygon is ALWAYS a single, smooth convex shape.
# ──────────────────────────────────────────────────────────────────

def build_roi_from_trails(
    trails: List[List[Point]],
    h: int,
    w: int,
    roi_expand: float,
    top_margin_frac: float = _TOP_MARGIN_FRAC,
    roi_shape: str = "rect",
) -> Optional[np.ndarray]:
    """
    Compute a clean ROI polygon directly from the centroid trails of
    qualifying (on-road) moving vehicles.

    roi_shape='rect'  → axis-aligned bounding rectangle (neat, 4-vertex)
    roi_shape='poly'  → aggressively-simplified convex hull (4-6 vertices)
    """
    y_cutoff = h * top_margin_frac
    pts: List[List[int]] = []
    for trail in trails:
        for (x, y) in trail:
            if y >= y_cutoff:             # skip top (sky / far background)
                pts.append([int(round(x)), int(round(y))])

    if len(pts) < 4:
        return None

    pts_arr = np.array(pts, np.int32)

    if roi_shape == "rect":
        # Axis-aligned bounding rectangle → cleanest rectangular look
        x, y, rw, rh = cv2.boundingRect(pts_arr)
        # Apply expand around center of the rect
        cx, cy = x + rw / 2.0, y + rh / 2.0
        new_w, new_h = rw * roi_expand, rh * roi_expand
        x0 = int(max(0, cx - new_w / 2))
        y0 = int(max(0, cy - new_h / 2))
        x1 = int(min(w - 1, cx + new_w / 2))
        y1 = int(min(h - 1, cy + new_h / 2))
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32)

    # roi_shape == "poly"
    hull    = cv2.convexHull(pts_arr)     # always a single convex shape
    perim   = cv2.arcLength(hull, True)

    # Douglas-Peucker: aggressively simplify toward 4 vertices
    best = hull
    for factor in [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.09,
                   0.11, 0.14, 0.18, 0.23, 0.30, 0.40, 0.50]:
        simplified = cv2.approxPolyDP(hull, factor * perim, True)
        best = simplified
        if len(simplified) <= 4:
            break

    poly = best[:, 0, :]
    return expand_polygon(poly, w, h, roi_expand)


# ──────────────────────────────────────────────────────────────────
# Trajectory heatmap → binary road mask
# ──────────────────────────────────────────────────────────────────

def build_trajectory_mask(trails: List[List[Point]], h: int, w: int,
                          line_thickness: Optional[int] = None) -> np.ndarray:
    """
    Draw vehicle movement trails on a canvas and blur to obtain a
    smooth road-surface mask.  Only truly moving vehicles (non-parked)
    leave trails, so parking lots are naturally excluded.
    """
    if not trails:
        return np.zeros((h, w), np.uint8)

    thickness = line_thickness or max(18, min(w, h) // 30)
    canvas = np.zeros((h, w), np.float32)

    for trail in trails:
        if len(trail) < 2:
            continue
        for i in range(len(trail) - 1):
            p1 = (int(round(trail[i][0])),     int(round(trail[i][1])))
            p2 = (int(round(trail[i + 1][0])), int(round(trail[i + 1][1])))
            cv2.line(canvas, p1, p2, 1.0, thickness)

    # Blur to soften jagged lines into a smooth road region
    blur_k = max(31, (thickness * 3) | 1)   # must be odd
    blurred = cv2.GaussianBlur(canvas, (blur_k, blur_k), 0)
    # Threshold: pixels with at least 30% of the peak heatmap value
    peak = float(blurred.max())
    if peak < 1e-6:
        return np.zeros((h, w), np.uint8)
    binary = (blurred >= peak * 0.10).astype(np.uint8) * 255
    return binary


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    detector = YOLO(args.weights)
    sam      = SAM(args.sam_model)
    tracker  = _CentroidTracker(min_moving_disp=_MIN_MOVING_DISP)

    first_frame: Optional[np.ndarray] = None
    accum_votes: Optional[np.ndarray] = None
    frames_with_detections = 0
    h = w = 0

    frame_idx = 0
    while frame_idx < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        if first_frame is None:
            first_frame = frame.copy()
            h, w = frame.shape[:2]
            accum_votes = np.zeros((h, w), dtype=np.float32)

        det = detector.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]

        dets_for_tracker: List[Tuple[Point, BBox]] = []
        if det.boxes is not None and len(det.boxes) > 0:
            xyxy = det.boxes.xyxy.cpu().numpy()
            cls  = det.boxes.cls.cpu().numpy().astype(int)
            for b, c in zip(xyxy, cls):
                if not _is_vehicle(det.names[int(c)].lower()):
                    continue
                x1, y1, x2, y2 = b.tolist()
                cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                dets_for_tracker.append(((cx, cy), (x1, y1, x2, y2)))

        tracker.update(dets_for_tracker)

        # ── SAM on MOVING vehicles only ────────────────────────────────
        # Tight expansion: 30% each horizontal side, 50% below for road
        # surface — much smaller than the old 1.5× / 2.0× that let in
        # entire parking rows.
        sam_boxes = []
        for t in tracker.moving_tracks():
            x1, y1, x2, y2 = t["last_box"]
            bw, bh = x2 - x1, y2 - y1
            ex1 = max(0.0,       x1 - 0.3 * bw)
            ey1 = max(0.0,       y1 - 0.15 * bh)
            ex2 = min(w - 1.0,   x2 + 0.3 * bw)
            ey2 = min(h - 1.0,   y2 + 0.5 * bh)   # extra below = road under vehicle
            sam_boxes.append([ex1, ey1, ex2, ey2])

        if sam_boxes:
            frames_with_detections += 1
            sam_res = sam.predict(frame, bboxes=np.array(sam_boxes, dtype=np.float32), verbose=False)
            for r in sam_res:
                if getattr(r, "masks", None) is None:
                    continue
                for mk in r.masks.data.cpu().numpy().astype(np.uint8):
                    if mk.shape[0] != h or mk.shape[1] != w:
                        mk = cv2.resize(mk, (w, h), interpolation=cv2.INTER_NEAREST)
                    accum_votes += (mk > 0).astype(np.float32)

        frame_idx += 1

    cap.release()

    if first_frame is None:
        raise RuntimeError("Video has no readable frames.")

    tracker.finalize()

    # ── 1. Collect qualifying moving trails ───────────────────────
    moving_trails = tracker.moving_completed_trails()

    # ── 2. Build ROI directly from trail-point convex hull ────────
    #
    # This is the primary, preferred method:
    #   • Guaranteed single convex polygon  (no fragmented blobs)
    #   • Guaranteed straight edges          (no jagged zigzag)
    #   • Naturally excludes parking lots    (filtered by displacement
    #     + path-length + straightness)
    #
    roi_poly = build_roi_from_trails(
        moving_trails, h, w, args.roi_expand,
        top_margin_frac=args.top_margin,
        roi_shape=args.roi_shape,
    )

    if roi_poly is None:
        # ── Fallback: use SAM accumulation mask when trails are sparse ──
        # (e.g. very short video, rare traffic)
        min_votes  = max(2.0, frames_with_detections * 0.08)
        sam_binary = (accum_votes >= min_votes).astype(np.uint8) * 255

        K  = max(15, min(w, h) // 50)
        k1 = np.ones((K,     K),     np.uint8)
        k2 = np.ones((K * 2, K * 2), np.uint8)
        sam_binary = cv2.morphologyEx(sam_binary, cv2.MORPH_CLOSE, k1, iterations=3)
        blurred    = cv2.GaussianBlur(sam_binary.astype(np.float32), (71, 71), 0)
        sam_binary = (blurred > 20).astype(np.uint8) * 255
        sam_binary = cv2.morphologyEx(sam_binary, cv2.MORPH_CLOSE, k2, iterations=2)
        sam_binary = filter_mask_components(
            sam_binary, min_area=max(800, (h * w) // 1200), max_components=1
        ) * 255

        contours, _ = cv2.findContours(sam_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if args.roi_shape == "rect":
                x, y, rw, rh = cv2.boundingRect(c)
                cx_r, cy_r = x + rw / 2.0, y + rh / 2.0
                new_w, new_h = rw * args.roi_expand, rh * args.roi_expand
                x0 = int(max(0, cx_r - new_w / 2))
                y0 = int(max(0, cy_r - new_h / 2))
                x1 = int(min(w - 1, cx_r + new_w / 2))
                y1 = int(min(h - 1, cy_r + new_h / 2))
                roi_poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32)
            else:
                hull  = cv2.convexHull(c)
                perim = cv2.arcLength(hull, True)
                best  = hull
                for factor in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.14, 0.19, 0.25, 0.35, 0.50]:
                    simplified = cv2.approxPolyDP(hull, factor * perim, True)
                    best = simplified
                    if len(simplified) <= 4:
                        break
                roi_poly = expand_polygon(best[:, 0, :], w, h, args.roi_expand)
        else:
            roi_poly = np.array(
                [[w // 4, h // 4], [3 * w // 4, h // 4],
                 [3 * w // 4, 3 * h // 4], [w // 4, 3 * h // 4]], np.int32
            )

    # ── 6. Build MOI vectors from moving trajectory data ───────────
    traj_vectors: List[np.ndarray] = []
    for trail in moving_trails:
        if len(trail) < _MIN_TRAIL_LEN:
            continue
        # Represent each trajectory as a start→end vector using the
        # central 80% of the trail (skip noisy entry/exit detections)
        i0 = max(0,            int(0.10 * len(trail)))
        i1 = min(len(trail)-1, int(0.90 * len(trail)))
        sx, sy = trail[i0]
        ex, ey = trail[i1]
        norm = math.hypot(ex - sx, ey - sy)
        if norm < args.moi_min_len:
            continue
        traj_vectors.append(np.array([sx, sy, ex, ey], np.float32))

    centers = kmeans_moi(traj_vectors, args.moi_count)
    moi_vectors: Dict[str, List[float]] = {}
    for idx, c in enumerate(centers, start=1):
        moi_vectors[str(idx)] = [float(c[0]), float(c[1]), float(c[2]), float(c[3])]

    xs = roi_poly[:, 0].astype(float)
    ys = roi_poly[:, 1].astype(float)
    width_ratio = (xs.max() - xs.min()) / max(1.0, float(w - 1))
    height_ratio = (ys.max() - ys.min()) / max(1.0, float(h - 1))
    full_like = width_ratio > 0.98 and height_ratio > 0.98
    quality_status = "low_confidence" if full_like or len(moi_vectors) == 0 else "ok"
    quality_reason = ""
    if full_like:
        quality_reason = "Trajectory-guided SAM produced a full-frame-like ROI."
    elif len(moi_vectors) == 0:
        quality_reason = "Trajectory-guided SAM produced no valid MOI vectors."

    # ── 7. Write JSON ───────────────────────────────────────────────
    payload = {
        "roi": roi_poly.astype(float).tolist(),
        "moi_vectors": moi_vectors,
        "quality": {
            "status": quality_status,
            "reason": quality_reason,
            "is_full_frame_fallback": bool(full_like),
            "moving_trails_used": len(moving_trails),
            "trajectory_vector_count": len(traj_vectors),
            "valid_moi_count": len(moi_vectors),
            "roi_width_ratio": round(float(width_ratio), 4),
            "roi_height_ratio": round(float(height_ratio), 4),
        },
        "note": (
            "Auto-generated by sam_bootstrap.py (trajectory-guided). "
            f"ROI has {len(roi_poly)} vertices. "
            f"Moving trails used: {len(moving_trails)}. "
            "Please review before production use."
        ),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)

    # ── 8. Optional overlay for visual verification ────────────────
    if args.save_overlay:
        vis = first_frame.copy()
        # Draw trajectory heatmap in semi-transparent green
        traj_mask = build_trajectory_mask(moving_trails, h, w)
        if np.any(traj_mask > 0):
            heat_rgb        = np.zeros((h, w, 3), np.uint8)
            heat_rgb[:,:,1] = traj_mask          # green channel
            vis = cv2.addWeighted(vis, 0.8, heat_rgb, 0.3, 0)
        # Draw ROI polygon (cyan)
        cv2.polylines(vis, [roi_poly.astype(np.int32)], True, (0, 255, 255), 2)
        # Draw MOI arrows (orange)
        for mid, vec in moi_vectors.items():
            x1, y1, x2, y2 = [int(v) for v in vec]
            cv2.arrowedLine(vis, (x1, y1), (x2, y2), (0, 165, 255), 2, tipLength=0.08)
            cv2.putText(vis, mid, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.imwrite(args.save_overlay, vis)

    print(
        f"Done. Wrote: {args.output_json} | "
        f"ROI vertices: {len(roi_poly)} | "
        f"MOI vectors: {len(moi_vectors)} | "
        f"Moving trails: {len(moving_trails)} | "
        f"SAM frames: {frames_with_detections}"
    )


if __name__ == "__main__":
    main()
