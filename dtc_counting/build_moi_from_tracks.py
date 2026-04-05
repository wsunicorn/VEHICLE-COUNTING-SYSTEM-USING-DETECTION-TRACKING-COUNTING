import argparse
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from run_dtc_counting import MultiStepTracker, collect_detections, infer_movement_count, load_polygon, point_in_polygon

Point = Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MOI vectors from tracked trajectories inside ROI."
    )
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument("--weights", required=True, help="Path to YOLO weights (best2.pt).")
    parser.add_argument("--roi-file", required=True, help="Path to ROI polygon txt.")
    parser.add_argument(
        "--movement-description",
        required=True,
        help="Path to movement description txt to infer movement count.",
    )
    parser.add_argument("--output-moi", required=True, help="Output MOI txt path.")
    parser.add_argument("--max-frames", type=int, default=3600, help="Frames used for mining trajectories.")
    parser.add_argument("--min-track-len", type=int, default=12, help="Minimum trajectory length (frames).")
    parser.add_argument("--min-vector-norm", type=float, default=40.0, help="Minimum vector length in pixels.")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    return parser.parse_args()


def roi_entry_exit_vector(history: List[Point], roi_polygon: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute the MOI vector using only the portion of the track that is
    *inside* the ROI polygon.  This gives cleaner direction estimates
    because it ignores the noisy approach / departure legs outside the
    counting zone.

    Falls back to the full 10%–90% span if fewer than 2 interior
    points are found.
    """
    interior = [p for p in history if point_in_polygon(p, roi_polygon)]

    if len(interior) >= 2:
        src = interior
    else:
        # Fallback: central span of the full track
        i0 = max(0,            int(0.10 * len(history)))
        i1 = min(len(history) - 1, int(0.90 * len(history)))
        src = history[i0: i1 + 1]
        if len(src) < 2:
            src = history

    start = src[0]
    end   = src[-1]
    return np.array([start[0], start[1], end[0], end[1]], dtype=np.float32)


def cluster_vectors(vectors: List[np.ndarray], movement_count: int) -> np.ndarray:
    if not vectors:
        return np.zeros((movement_count, 4), dtype=np.float32)

    data = np.vstack(vectors).astype(np.float32)
    k    = max(1, min(movement_count, len(data)))

    # More iterations + tighter tolerance than before for better convergence
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.05)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 12, cv2.KMEANS_PP_CENTERS)

    # ── Refine each cluster center using the median (more robust to outliers) ──
    for ci in range(k):
        members = data[labels.ravel() == ci]
        if len(members) >= 3:
            centers[ci] = np.median(members, axis=0)

    # Sort by direction angle for stable movement indexing
    angles = np.arctan2(centers[:, 3] - centers[:, 1], centers[:, 2] - centers[:, 0])
    order  = np.argsort(angles)
    centers = centers[order]

    if len(centers) < movement_count:
        pad     = np.repeat(centers[-1:, :], movement_count - len(centers), axis=0)
        centers = np.vstack([centers, pad])

    return centers[:movement_count]


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    roi_polygon    = load_polygon(args.roi_file)
    movement_count = infer_movement_count(args.movement_description)

    model   = YOLO(args.weights)
    tracker = MultiStepTracker()
    finished_tracks: list = []

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    frame_idx = 0
    while frame_idx < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        detections = collect_detections(model, frame, args.conf, args.imgsz, roi_polygon)
        tracks     = tracker.update(detections, frame_idx)

        for tr in tracks.values():
            inside = point_in_polygon(tr.center, roi_polygon)
            tr.ever_inside_roi = tr.ever_inside_roi or inside

        if tracker.recently_removed:
            finished_tracks.extend(tracker.recently_removed)

        frame_idx += 1

    cap.release()

    candidate_vectors: List[np.ndarray] = []
    all_tracks = finished_tracks + list(tracker.tracks.values())
    for tr in all_tracks:
        if not tr.ever_inside_roi:
            continue
        if len(tr.history) < args.min_track_len:
            continue

        # Use the ROI-interior segment of the track for a cleaner vector
        vec = roi_entry_exit_vector(tr.history, roi_polygon)
        if vec is None:
            continue

        norm = float(np.hypot(vec[2] - vec[0], vec[3] - vec[1]))
        if norm < args.min_vector_norm:
            continue

        candidate_vectors.append(vec)

    centers = cluster_vectors(candidate_vectors, movement_count)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_moi)), exist_ok=True)
    with open(args.output_moi, "w", encoding="utf-8") as f:
        for idx, c in enumerate(centers, start=1):
            f.write(f"{idx},{c[0]:.2f},{c[1]:.2f},{c[2]:.2f},{c[3]:.2f}\n")

    print(
        f"Done. Built {len(centers)} MOI vectors from "
        f"{len(candidate_vectors)} candidate trajectories → {args.output_moi}"
    )


if __name__ == "__main__":
    main()
