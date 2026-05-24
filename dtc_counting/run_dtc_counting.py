import argparse
import csv
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from scipy.optimize import linear_sum_assignment
except Exception as exc:
    raise RuntimeError(
        "Missing scipy. Install with: pip install scipy"
    ) from exc


BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]


@dataclass
class Detection:
    bbox: BBox
    cls_id: int
    conf: float
    center: Point
    hist: np.ndarray


@dataclass
class Track:
    track_id: int
    cls_id: int
    mean: np.ndarray
    cov: np.ndarray
    bbox: BBox
    center: Point
    hist: np.ndarray
    first_frame: int
    last_frame: int
    age: int = 1
    hits: int = 1
    missed: int = 0
    counted: bool = False
    ever_inside_roi: bool = False
    is_illegal: bool = False
    history: List[Point] = field(default_factory=list)


class MultiStepTracker:
    """Kalman + Hungarian tracker with three-stage association: IoU, Mahalanobis, histogram."""

    def __init__(
        self,
        iou_threshold: float = 0.1,
        mahalanobis_threshold: float = 16.0,
        hist_dist_threshold: float = 0.45,
        center_dist_threshold: float = 50.0,
        max_missed: int = 35,
        min_hits: int = 3,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.mahalanobis_threshold = mahalanobis_threshold
        self.hist_dist_threshold = hist_dist_threshold
        self.center_dist_threshold = center_dist_threshold
        self.max_missed = max_missed
        self.min_hits = min_hits

        self.tracks: Dict[int, Track] = {}
        self.recently_removed: List[Track] = []
        self.next_track_id = 1

        dt = 1.0
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = dt
        self.H = np.zeros((4, 8), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        q_pos = 1.0
        q_vel = 4.0
        self.Q = np.diag([q_pos, q_pos, q_pos, q_pos, q_vel, q_vel, q_vel, q_vel]).astype(np.float32)
        self.R = np.diag([10.0, 10.0, 0.05, 15.0]).astype(np.float32)

    @staticmethod
    def _iou(a: BBox, b: BBox) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union

    @staticmethod
    def _center_distance(a: Point, b: Point) -> float:
        return float(math.hypot(a[0] - b[0], a[1] - b[1]))

    @staticmethod
    def _hist_distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.abs(a - b)))

    @staticmethod
    def _xyah_to_bbox(xyah: np.ndarray) -> BBox:
        x, y, a, h = float(xyah[0]), float(xyah[1]), float(xyah[2]), float(xyah[3])
        h = max(2.0, h)
        a = max(0.05, a)
        w = a * h
        return x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0

    @staticmethod
    def _bbox_to_xyah(bbox: BBox) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        w = max(2.0, x2 - x1)
        h = max(2.0, y2 - y1)
        x = x1 + w / 2.0
        y = y1 + h / 2.0
        a = w / h
        return np.array([x, y, a, h], dtype=np.float32)

    def _kf_predict(self, mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean_p = self.F @ mean
        cov_p = self.F @ cov @ self.F.T + self.Q
        return mean_p, cov_p

    def _kf_update(self, mean: np.ndarray, cov: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y = z - (self.H @ mean)
        s = self.H @ cov @ self.H.T + self.R
        k = cov @ self.H.T @ np.linalg.inv(s)
        mean_u = mean + (k @ y)
        i = np.eye(cov.shape[0], dtype=np.float32)
        cov_u = (i - k @ self.H) @ cov
        return mean_u, cov_u

    def _mahalanobis_cost(self, tr: Track, det: Detection) -> float:
        z = self._bbox_to_xyah(det.bbox)
        mu = self.H @ tr.mean
        s = self.H @ tr.cov @ self.H.T + self.R
        d = z - mu
        return float(d.T @ np.linalg.inv(s) @ d)

    @staticmethod
    def _hungarian(cost: np.ndarray, max_cost: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if cost.size == 0:
            return [], list(range(cost.shape[0])), list(range(cost.shape[1]))

        rows, cols = linear_sum_assignment(cost)
        matches: List[Tuple[int, int]] = []
        used_r = set()
        used_c = set()
        for r, c in zip(rows.tolist(), cols.tolist()):
            if cost[r, c] > max_cost:
                continue
            matches.append((r, c))
            used_r.add(r)
            used_c.add(c)

        unmatched_r = [r for r in range(cost.shape[0]) if r not in used_r]
        unmatched_c = [c for c in range(cost.shape[1]) if c not in used_c]
        return matches, unmatched_r, unmatched_c

    def _build_cost_iou(self, track_ids: List[int], detections: List[Detection], det_ids: List[int]) -> np.ndarray:
        cost = np.full((len(track_ids), len(det_ids)), 1e6, dtype=np.float32)
        for ri, tid in enumerate(track_ids):
            tr = self.tracks[tid]
            for ci, did in enumerate(det_ids):
                det = detections[did]
                if tr.cls_id != det.cls_id:
                    continue
                iou = self._iou(tr.bbox, det.bbox)
                if iou < self.iou_threshold:
                    continue
                cost[ri, ci] = 1.0 - iou
        return cost

    def _build_cost_mahalanobis(self, track_ids: List[int], detections: List[Detection], det_ids: List[int]) -> np.ndarray:
        cost = np.full((len(track_ids), len(det_ids)), 1e6, dtype=np.float32)
        for ri, tid in enumerate(track_ids):
            tr = self.tracks[tid]
            for ci, did in enumerate(det_ids):
                det = detections[did]
                if tr.cls_id != det.cls_id:
                    continue
                md2 = self._mahalanobis_cost(tr, det)
                if md2 > self.mahalanobis_threshold:
                    continue
                cost[ri, ci] = md2
        return cost

    def _build_cost_hist(self, track_ids: List[int], detections: List[Detection], det_ids: List[int]) -> np.ndarray:
        cost = np.full((len(track_ids), len(det_ids)), 1e6, dtype=np.float32)
        for ri, tid in enumerate(track_ids):
            tr = self.tracks[tid]
            for ci, did in enumerate(det_ids):
                det = detections[did]
                if tr.cls_id != det.cls_id:
                    continue
                center_d = self._center_distance(tr.center, det.center)
                if center_d > self.center_dist_threshold:
                    continue
                hd = self._hist_distance(tr.hist, det.hist)
                if hd > self.hist_dist_threshold:
                    continue
                cost[ri, ci] = hd
        return cost

    def update(self, detections: List[Detection], frame_idx: int) -> Dict[int, Track]:
        self.recently_removed = []

        # Predict all tracklets for time t.
        for tr in self.tracks.values():
            tr.mean, tr.cov = self._kf_predict(tr.mean, tr.cov)
            tr.bbox = self._xyah_to_bbox(tr.mean[:4])
            tr.center = (float(tr.mean[0]), float(tr.mean[1]))

        track_ids = list(self.tracks.keys())
        det_ids = list(range(len(detections)))
        matched_pairs: List[Tuple[int, int]] = []

        # Stage 1: IoU matching.
        c1 = self._build_cost_iou(track_ids, detections, det_ids)
        m1, ut1, ud1 = self._hungarian(c1, max_cost=1.0 - self.iou_threshold)
        matched_pairs.extend([(track_ids[r], det_ids[c]) for r, c in m1])

        t2 = [track_ids[r] for r in ut1]
        d2 = [det_ids[c] for c in ud1]

        # Stage 2: Mahalanobis matching.
        c2 = self._build_cost_mahalanobis(t2, detections, d2)
        m2, ut2, ud2 = self._hungarian(c2, max_cost=self.mahalanobis_threshold)
        matched_pairs.extend([(t2[r], d2[c]) for r, c in m2])

        t3 = [t2[r] for r in ut2]
        d3 = [d2[c] for c in ud2]

        # Stage 3: visual histogram matching.
        c3 = self._build_cost_hist(t3, detections, d3)
        m3, ut3, ud3 = self._hungarian(c3, max_cost=self.hist_dist_threshold)
        matched_pairs.extend([(t3[r], d3[c]) for r, c in m3])

        unmatched_t = [t3[r] for r in ut3]
        unmatched_d = [d3[c] for c in ud3]

        # Update matched tracklets.
        for tid, did in matched_pairs:
            tr = self.tracks[tid]
            det = detections[did]
            z = self._bbox_to_xyah(det.bbox)
            tr.mean, tr.cov = self._kf_update(tr.mean, tr.cov, z)
            tr.bbox = det.bbox
            tr.center = det.center
            tr.hist = det.hist
            tr.last_frame = frame_idx
            tr.age += 1
            tr.hits += 1
            tr.missed = 0
            tr.history.append(det.center)

        # Propagate unmatched tracklets.
        for tid in unmatched_t:
            tr = self.tracks[tid]
            tr.age += 1
            tr.missed += 1
            tr.history.append(tr.center)

        # Initiate new tracklets.
        for did in unmatched_d:
            det = detections[did]
            mean = np.zeros((8,), dtype=np.float32)
            mean[:4] = self._bbox_to_xyah(det.bbox)
            cov = np.diag([20.0, 20.0, 1.0, 30.0, 50.0, 50.0, 5.0, 50.0]).astype(np.float32)
            tid = self.next_track_id
            self.next_track_id += 1
            self.tracks[tid] = Track(
                track_id=tid,
                cls_id=det.cls_id,
                mean=mean,
                cov=cov,
                bbox=det.bbox,
                center=det.center,
                hist=det.hist,
                first_frame=frame_idx,
                last_frame=frame_idx,
                history=[det.center],
            )

        dead_ids = [tid for tid, tr in self.tracks.items() if tr.missed > self.max_missed]
        for tid in dead_ids:
            self.recently_removed.append(self.tracks[tid])
            del self.tracks[tid]

        return self.tracks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vehicle counting with DTC framework (Kalman + Hungarian).")
    parser.add_argument("--video", required=True, help="Path to video file.")
    parser.add_argument("--weights", required=True, help="Path to YOLO weights (best2.pt).")
    parser.add_argument("--output-csv", required=True, help="Output counting CSV path.")
    parser.add_argument("--roi-file", required=True, help="ROI file path (cam_x.txt) for counting boundary.")
    parser.add_argument("--eroi-file", default="", help="Optional eROI file path (if empty, roi-file is used).")
    parser.add_argument("--iroi-file", default="", help="Optional iROI file path for illegal paths filtering.")
    parser.add_argument("--moi-vectors", default="", help="Optional file with MOI vectors: id,x1,y1,x2,y2 per line.")
    parser.add_argument("--movement-description", default="", help="Optional movement description file for number of MOIs.")
    parser.add_argument("--video-clip-id", type=int, default=1, help="video_clip_id written to CSV.")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument(
        "--class-conf",
        default="",
        help="Optional per-class confidence overrides, e.g. car=0.25,truck=0.45.",
    )
    parser.add_argument(
        "--min-count-history",
        type=int,
        default=5,
        help="Minimum number of trajectory points before a track can be counted.",
    )
    parser.add_argument(
        "--min-count-displacement",
        type=float,
        default=30.0,
        help="Minimum 10%%-90%% trajectory displacement in pixels before counting.",
    )
    parser.add_argument("--moi-angle-weight", type=float, default=300.0, help="Angle weight for vector-based MOI assignment.")
    parser.add_argument("--moi-distance-weight", type=float, default=0.35, help="Endpoint-distance weight for vector-based MOI assignment.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every N-th frame for faster runtime.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many frames (0 means full video).")
    parser.add_argument("--show", action="store_true", help="Show visualized video while processing.")
    parser.add_argument("--save-video", default="", help="Optional output visualization video path.")
    parser.add_argument("--output-timing", default="", help="Optional JSON file to write timing info for S1_Efficiency.")
    return parser.parse_args()


def load_polygon(path: str) -> np.ndarray:
    points: List[Tuple[int, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            x_str, y_str = raw.split(",")
            points.append((int(float(x_str)), int(float(y_str))))
    if len(points) < 3:
        raise ValueError(f"Polygon file must contain at least 3 points: {path}")
    return np.array(points, dtype=np.int32)


def load_moi_vectors(path: str) -> Dict[int, Tuple[Point, Point]]:
    vectors: Dict[int, Tuple[Point, Point]] = {}
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
            mid, x1, y1, x2, y2 = parts
            vectors[int(mid)] = ((float(x1), float(y1)), (float(x2), float(y2)))
    return vectors


def infer_movement_count(path: str) -> int:
    if not path or not os.path.exists(path):
        return 8
    movement_ids = set()
    pat = re.compile(r"movement\s+(\d+)\s*:", re.IGNORECASE)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if m:
                movement_ids.add(int(m.group(1)))
    return max(movement_ids) if movement_ids else 8


def point_in_polygon(point: Point, polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon.astype(np.float32), point, False) >= 0


def bbox_roi_overlap_ratio(bbox: BBox, polygon: np.ndarray, grid: int = 5) -> float:
    """Approximate bbox-polygon overlap by dense point sampling inside bbox."""
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return 0.0
    xs = np.linspace(x1, x2, num=max(2, grid), dtype=np.float32)
    ys = np.linspace(y1, y2, num=max(2, grid), dtype=np.float32)
    inside = 0
    total = 0
    poly = polygon.astype(np.float32)
    for yy in ys:
        for xx in xs:
            total += 1
            if cv2.pointPolygonTest(poly, (float(xx), float(yy)), False) >= 0:
                inside += 1
    return float(inside) / float(max(1, total))


def bbox_effectively_in_roi(bbox: BBox, center: Point, polygon: np.ndarray) -> bool:
    """Use mixed tests so large vehicles crossing ROI border are still considered inside."""
    if point_in_polygon(center, polygon):
        return True

    x1, y1, x2, y2 = bbox
    corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
    for p in corners:
        if point_in_polygon(p, polygon):
            return True

    overlap = bbox_roi_overlap_ratio(bbox, polygon, grid=5)
    return overlap >= 0.15


def clip_bbox(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> BBox:
    x1 = max(0.0, min(float(w - 1), x1))
    y1 = max(0.0, min(float(h - 1), y1))
    x2 = max(0.0, min(float(w - 1), x2))
    y2 = max(0.0, min(float(h - 1), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def crop_hist(frame: np.ndarray, bbox: BBox) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in clip_bbox(*bbox, w, h)]
    crop = frame[y1 : y2 + 1, x1 : x2 + 1]
    if crop.size == 0:
        return np.zeros((48,), dtype=np.float32)
    hist = []
    for c in range(3):
        hch = cv2.calcHist([crop], [c], None, [16], [0, 256]).flatten()
        if hch.sum() > 0:
            hch = hch / hch.sum()
        hist.append(hch)
    return np.concatenate(hist).astype(np.float32)


def class_to_aicity_id(name: str) -> Optional[int]:
    n = name.lower()
    if "car" in n:
        return 1
    if "truck" in n:
        return 2
    return None


def parse_class_conf(spec: str) -> Dict[int, float]:
    overrides: Dict[int, float] = {}
    if not spec:
        return overrides
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --class-conf item: {item!r}")
        key, value = [part.strip() for part in item.split("=", 1)]
        cls_id = class_to_aicity_id(key)
        if cls_id is None:
            raise ValueError(f"Unsupported class in --class-conf: {key!r}")
        overrides[cls_id] = float(value)
    return overrides


def movement_id_from_vectors(
    track: Track,
    vectors: Dict[int, Tuple[Point, Point]],
    angle_weight: float = 300.0,
    distance_weight: float = 0.35,
) -> int:
    start = track.history[max(0, int(0.1 * len(track.history)) - 1)]
    end = track.history[min(len(track.history) - 1, int(0.9 * len(track.history)))]
    tv = np.array([end[0] - start[0], end[1] - start[1]], dtype=np.float32)
    tnorm = np.linalg.norm(tv) + 1e-6

    best_mid = sorted(vectors.keys())[0]
    best_score = float("inf")
    for mid, (s, e) in vectors.items():
        mv = np.array([e[0] - s[0], e[1] - s[1]], dtype=np.float32)
        mnorm = np.linalg.norm(mv) + 1e-6
        cos = float(np.dot(tv, mv) / (tnorm * mnorm))
        cos = max(-1.0, min(1.0, cos))
        angle = math.acos(cos)
        dist = float(math.hypot(start[0] - s[0], start[1] - s[1]) + math.hypot(end[0] - e[0], end[1] - e[1]))
        score = angle * angle_weight + dist * distance_weight
        if score < best_score:
            best_score = score
            best_mid = mid
    return best_mid


def movement_id_fallback(track: Track, movement_count: int) -> int:
    if len(track.history) < 2:
        return 1
    start = track.history[0]
    end = track.history[-1]
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    mapped = int(((angle + math.pi) / (2.0 * math.pi)) * movement_count) + 1
    return max(1, min(movement_count, mapped))


def collect_detections(
    model: YOLO,
    frame: np.ndarray,
    conf_thres: float,
    imgsz: int,
    eroi_polygon: np.ndarray,
    class_conf: Optional[Dict[int, float]] = None,
) -> List[Detection]:
    result = model.predict(frame, conf=conf_thres, imgsz=imgsz, verbose=False)[0]
    detections: List[Detection] = []
    class_conf = class_conf or {}
    names = result.names
    if result.boxes is None:
        return detections

    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls_idx in zip(xyxy, confs, clss):
        name = names[int(cls_idx)]
        aicity_cls = class_to_aicity_id(name)
        if aicity_cls is None:
            continue
        if float(conf) < class_conf.get(aicity_cls, conf_thres):
            continue

        x1, y1, x2, y2 = map(float, box.tolist())
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        if not bbox_effectively_in_roi((x1, y1, x2, y2), (cx, cy), eroi_polygon):
            continue

        hist = crop_hist(frame, (x1, y1, x2, y2))
        detections.append(
            Detection(
                bbox=(x1, y1, x2, y2),
                cls_id=aicity_cls,
                conf=float(conf),
                center=(cx, cy),
                hist=hist,
            )
        )
    return detections


# Color palette for track visualization (BGR)
_TRACK_COLORS = [
    (230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66),
    (0, 114, 178), (213, 94, 0), (204, 121, 167), (100, 200, 50),
    (50, 100, 200), (200, 50, 100), (150, 150, 0), (0, 150, 150),
]


def draw_frame(
    frame: np.ndarray,
    roi_polygon: np.ndarray,
    eroi_polygon: np.ndarray,
    iroi_polygon: Optional[np.ndarray],
    tracks: Dict[int, Track],
    count_stats: Dict[Tuple[int, int], int],
    moi_vectors: Optional[Dict[int, Tuple[Point, Point]]] = None,
    frame_idx: int = 0,
) -> np.ndarray:
    vis = frame.copy()

    # Draw ROI polygons with semi-transparent fill
    roi_overlay = vis.copy()
    cv2.fillPoly(roi_overlay, [roi_polygon], (0, 255, 255))
    cv2.addWeighted(roi_overlay, 0.08, vis, 0.92, 0, vis)
    cv2.polylines(vis, [roi_polygon], isClosed=True, color=(0, 255, 255), thickness=2)
    if eroi_polygon is not None and not np.array_equal(eroi_polygon, roi_polygon):
        cv2.polylines(vis, [eroi_polygon], isClosed=True, color=(0, 200, 80), thickness=2)
    if iroi_polygon is not None:
        iroi_overlay = vis.copy()
        cv2.fillPoly(iroi_overlay, [iroi_polygon], (0, 0, 255))
        cv2.addWeighted(iroi_overlay, 0.12, vis, 0.88, 0, vis)
        cv2.polylines(vis, [iroi_polygon], isClosed=True, color=(0, 0, 255), thickness=2)

    # Draw MOI arrows (Disabled as requested)
    # if moi_vectors:
    #     for mid, (start, end) in moi_vectors.items():
    #         sx, sy = int(start[0]), int(start[1])
    #         ex, ey = int(end[0]), int(end[1])
    #         cv2.arrowedLine(vis, (sx, sy), (ex, ey), (0, 180, 255), 2, tipLength=0.06)
    #         cv2.putText(vis, f"M{mid}", (sx - 5, sy - 8),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

    # Draw tracked vehicles: bbox, ID, trajectory trail
    for tid, tr in tracks.items():
        if tr.hits < 2:
            continue  # Skip unconfirmed tracks

        color = _TRACK_COLORS[tid % len(_TRACK_COLORS)]
        x1, y1, x2, y2 = [int(v) for v in tr.bbox]
        cls_tag = "C" if tr.cls_id == 1 else "T"

        # Bounding box & Label (Disabled to avoid obscuring the video)
        # box_thick = 2 if tr.hits >= 3 else 1
        # cv2.rectangle(vis, (x1, y1), (x2, y2), color, box_thick)
        # 
        # label = f"#{tid} {cls_tag}"
        # (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        # cv2.rectangle(vis, (x1, y1 - lh - 6), (x1 + lw + 4, y1), color, -1)
        # cv2.putText(vis, label, (x1 + 2, y1 - 4),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw a small dot at the center of the vehicle (Disabled as requested)
        # cx, cy = int(tr.center[0]), int(tr.center[1])
        # cv2.circle(vis, (cx, cy), 3, color, -1)

        # Trajectory trail (Disabled as requested)
        # trail = tr.history[-30:]
        # for i in range(1, len(trail)):
        #     p1 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
        #     p2 = (int(trail[i][0]), int(trail[i][1]))
        #     alpha = i / len(trail)
        #     thick = max(1, int(alpha * 3))
        #     cv2.line(vis, p1, p2, color, thick)

    # Count stats panel (top-right)
    frame_w = vis.shape[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    y = 25

    # Total count header
    total_counted = sum(count_stats.values())
    header = f"Total: {total_counted}"
    (hw, hh), _ = cv2.getTextSize(header, font, 0.7, 2)
    hx = frame_w - hw - 14
    cv2.rectangle(vis, (hx - 6, y - hh - 4), (hx + hw + 6, y + 4), (0, 0, 0), -1)
    cv2.putText(vis, header, (hx, y), font, 0.7, (0, 255, 200), 2, cv2.LINE_AA)
    y += hh + 14

    for (mid, cls_id), value in sorted(count_stats.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        tag = "car" if cls_id == 1 else "truck"
        text = f"m{mid}-{tag}: {value}"
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x = frame_w - tw - 10
        cv2.rectangle(vis, (x - 4, y - th - 3), (x + tw + 4, y + baseline + 1), (0, 0, 0), -1)
        cv2.putText(vis, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += th + baseline + 6

    # Frame counter (bottom-left)
    frame_text = f"Frame: {frame_idx}"
    cv2.putText(vis, frame_text, (10, vis.shape[0] - 12),
                font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return vis


def track_displacement(track: Track) -> float:
    if len(track.history) < 2:
        return 0.0
    i0 = max(0, int(0.10 * len(track.history)))
    i1 = min(len(track.history) - 1, int(0.90 * len(track.history)))
    start = track.history[i0]
    end = track.history[i1]
    return float(math.hypot(end[0] - start[0], end[1] - start[1]))


def count_track(
    tr: Track,
    frame_idx: int,
    movement_count: int,
    moi_vectors: Dict[int, Tuple[Point, Point]],
    rows: List[Tuple[int, int, int, int]],
    count_stats: Dict[Tuple[int, int], int],
    video_clip_id: int,
    min_count_history: int,
    min_count_displacement: float,
    moi_angle_weight: float,
    moi_distance_weight: float,
) -> None:
    if tr.counted or tr.is_illegal or tr.hits < 3 or not tr.ever_inside_roi:
        return
    if len(tr.history) < min_count_history:
        return
    if track_displacement(tr) < min_count_displacement:
        return
    movement_id = (
        movement_id_from_vectors(tr, moi_vectors, moi_angle_weight, moi_distance_weight)
        if moi_vectors
        else movement_id_fallback(tr, movement_count)
    )
    rows.append((video_clip_id, frame_idx, movement_id, tr.cls_id))
    count_stats[(movement_id, tr.cls_id)] = count_stats.get((movement_id, tr.cls_id), 0) + 1
    tr.counted = True


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    roi_polygon = load_polygon(args.roi_file)
    eroi_polygon = load_polygon(args.eroi_file) if args.eroi_file else roi_polygon
    iroi_polygon = load_polygon(args.iroi_file) if args.iroi_file else None
    moi_vectors = load_moi_vectors(args.moi_vectors)
    movement_count = infer_movement_count(args.movement_description)
    class_conf = parse_class_conf(args.class_conf)

    model = YOLO(args.weights)
    tracker = MultiStepTracker()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    writer = None
    writer_path = ""
    if args.save_video:
        # Keep output duration close to input duration when processing with frame stride.
        out_fps = fps / max(1, int(args.frame_stride))
        out_fps = max(1.0, float(out_fps))

        requested = args.save_video
        req_dir = os.path.dirname(requested)
        if req_dir:
            os.makedirs(req_dir, exist_ok=True)

        root, ext = os.path.splitext(requested)
        candidates = [
            (requested, "mp4v"),
            (requested, "avc1"),
            (requested, "H264"),
            (f"{root}.avi" if ext.lower() != ".avi" else requested, "MJPG"),
        ]

        for path_try, codec in candidates:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer_try = cv2.VideoWriter(path_try, fourcc, out_fps, (w, h))
            if writer_try.isOpened():
                writer = writer_try
                writer_path = path_try
                print(f"[video] using codec={codec} path={writer_path}")
                break
            writer_try.release()

        if writer is None:
            print("[video] warning: failed to open VideoWriter for all codecs.")

    rows: List[Tuple[int, int, int, int]] = []
    count_stats: Dict[Tuple[int, int], int] = {}
    frame_idx = 0
    start_ts = time.time()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    while True:
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        ok, frame = cap.read()
        if not ok:
            break

        if args.frame_stride > 1 and (frame_idx % args.frame_stride != 0):
            frame_idx += 1
            continue

        detections = collect_detections(model, frame, args.conf, args.imgsz, eroi_polygon, class_conf)
        tracks = tracker.update(detections, frame_idx)

        for tr in tracks.values():
            inside_roi = bbox_effectively_in_roi(tr.bbox, tr.center, roi_polygon)
            tr.ever_inside_roi = tr.ever_inside_roi or inside_roi
            if iroi_polygon is not None and bbox_effectively_in_roi(tr.bbox, tr.center, iroi_polygon):
                tr.is_illegal = True

            # Count by estimated exit moment from KF-predicted state.
            if tr.hits >= tracker.min_hits and tr.ever_inside_roi and (not inside_roi):
                count_track(
                    tr,
                    frame_idx,
                    movement_count,
                    moi_vectors,
                    rows,
                    count_stats,
                    args.video_clip_id,
                    args.min_count_history,
                    args.min_count_displacement,
                    args.moi_angle_weight,
                    args.moi_distance_weight,
                )

        # If a tracklet disappears after being inside ROI, estimate it as exited.
        for tr in tracker.recently_removed:
            if iroi_polygon is not None:
                for p in tr.history:
                    if point_in_polygon(p, iroi_polygon):
                        tr.is_illegal = True
                        break
            count_track(
                tr,
                frame_idx,
                movement_count,
                moi_vectors,
                rows,
                count_stats,
                args.video_clip_id,
                args.min_count_history,
                args.min_count_displacement,
                args.moi_angle_weight,
                args.moi_distance_weight,
            )

        if args.show or writer is not None:
            vis = draw_frame(frame, roi_polygon, eroi_polygon, iroi_polygon, tracks, count_stats, moi_vectors, frame_idx)
            if writer is not None:
                writer.write(vis)
            if args.show:
                cv2.imshow("DTC Counting", vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        frame_idx += 1
        if frame_idx % 100 == 0:
            elapsed = max(1e-6, time.time() - start_ts)
            fps_proc = frame_idx / elapsed
            if total_frames > 0:
                pct = 100.0 * frame_idx / total_frames
                print(
                    f"[progress] frame={frame_idx}/{total_frames} ({pct:.1f}%) "
                    f"proc_fps={fps_proc:.2f} active_tracks={len(tracks)}"
                )
            else:
                print(
                    f"[progress] frame={frame_idx} proc_fps={fps_proc:.2f} "
                    f"active_tracks={len(tracks)}"
                )

    cap.release()
    if writer is not None:
        writer.release()
        if writer_path:
            print(f"[video] wrote visualization video: {writer_path}")
    if args.show:
        cv2.destroyAllWindows()

    # Flush surviving tracklets.
    for tr in tracker.tracks.values():
        if iroi_polygon is not None:
            for p in tr.history:
                if point_in_polygon(p, iroi_polygon):
                    tr.is_illegal = True
                    break
        count_track(
            tr,
            frame_idx,
            movement_count,
            moi_vectors,
            rows,
            count_stats,
            args.video_clip_id,
            args.min_count_history,
            args.min_count_displacement,
            args.moi_angle_weight,
            args.moi_distance_weight,
        )

    rows.sort(key=lambda x: (x[1], x[2], x[3]))
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["video_clip_id", "frame_id", "movement_id", "vehicle_class_id"])
        writer_csv.writerows(rows)

    elapsed_total = max(1e-6, time.time() - start_ts)
    print(f"[timing] processed_frames={frame_idx} elapsed={elapsed_total:.2f}s avg_fps={frame_idx/elapsed_total:.2f}")
    print(f"Done. Wrote {len(rows)} counting events to: {args.output_csv}")

    # Write timing JSON for S1_Efficiency evaluation
    if args.output_timing:
        timing_dir = os.path.dirname(args.output_timing)
        if timing_dir:
            os.makedirs(timing_dir, exist_ok=True)
        import json
        timing_data = {
            "processed_frames": frame_idx,
            "elapsed_seconds": round(elapsed_total, 3),
            "avg_fps": round(frame_idx / elapsed_total, 2),
            "video_fps": fps,
            "total_video_frames": total_frames,
            "frame_stride": args.frame_stride,
        }
        with open(args.output_timing, "w", encoding="utf-8") as tf:
            json.dump(timing_data, tf, indent=2)
        print(f"[timing] Wrote timing data to: {args.output_timing}")


if __name__ == "__main__":
    main()
