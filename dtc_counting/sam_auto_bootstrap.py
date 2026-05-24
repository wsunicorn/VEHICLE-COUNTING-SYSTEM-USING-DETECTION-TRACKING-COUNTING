"""
sam_auto_bootstrap.py — ROI/MOI bootstrap using SAM Automatic mode only.

This pipeline requires NO external detector (no YOLO, no Grounding DINO).
It uses SAM's automatic mask generation to segment the first frame,
then applies heuristic filtering to identify road-surface masks.

Pipeline:
  1. Read first frame from video
  2. Run SAM in automatic (everything) mode
  3. Filter masks with road-surface heuristics:
     - Area between 5% and 80% of frame
     - Position: mask centroid in lower 80% of frame
     - Shape: prefer elongated, horizontal masks
  4. Union qualifying masks -> ROI polygon
  5. PCA on each mask component -> MOI direction vectors
  6. KMeans cluster -> final MOI vectors
  7. Export JSON + optional overlay image

Usage:
    python sam_auto_bootstrap.py \\
        --video cam_5_ThanhPhoBuon.mp4 \\
        --sam-model sam_b.pt \\
        --output-json outputs/cam5_sam_auto.json \\
        --save-overlay outputs/cam5_sam_auto_overlay.jpg \\
        --moi-count 12
"""

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap ROI and MOI using SAM automatic mode (no detector needed)."
    )
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument("--output-json", required=True, help="Output JSON for ROI and MOI vectors.")
    parser.add_argument("--sam-model", default="sam_b.pt", help="SAM checkpoint for ultralytics SAM.")
    parser.add_argument("--moi-count", type=int, default=12, help="Expected number of MOIs.")
    parser.add_argument("--roi-expand", type=float, default=1.08,
                        help="Scale factor to slightly expand ROI polygon.")
    parser.add_argument("--min-area-frac", type=float, default=0.02,
                        help="Min mask area as fraction of frame area (default 2%%).")
    parser.add_argument("--max-area-frac", type=float, default=0.80,
                        help="Max mask area as fraction of frame area (default 80%%).")
    parser.add_argument("--top-margin", type=float, default=0.15,
                        help="Fraction of frame height to exclude from top (sky/background).")
    parser.add_argument("--max-masks", type=int, default=8,
                        help="Maximum number of road masks to keep.")
    parser.add_argument("--save-overlay", default="", help="Optional overlay image path.")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────
# Mask scoring heuristic — identify road-surface masks
# ──────────────────────────────────────────────────────────────────

def score_road_mask(
    mask: np.ndarray,
    frame_h: int,
    frame_w: int,
    frame_bgr: Optional[np.ndarray] = None,
    top_margin: float = 0.15,
    min_area_frac: float = 0.02,
    max_area_frac: float = 0.80,
) -> float:
    """
    Score a binary mask for how likely it represents a road surface.

    Returns a float score >= 0.  Higher = more likely road.
    Returns -1.0 if the mask should be rejected.

    Heuristics:
      - Area: between min_area_frac and max_area_frac of frame
      - Vertical position: centroid should be in lower portion of frame
      - Elongation: road masks tend to be wider than tall (aspect ratio > 1.0)
      - Lower-frame coverage: road masks should have significant
        presence in the lower half
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return -1.0

    area = len(xs)
    frame_area = frame_h * frame_w
    area_frac = area / frame_area

    # Area filter
    if area_frac < min_area_frac or area_frac > max_area_frac:
        return -1.0

    # Centroid filter — reject if centroid is in top margin
    cy = float(ys.mean())
    if cy < frame_h * top_margin:
        return -1.0

    # Bounding box analysis
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    bbox_w = x_max - x_min + 1
    bbox_h = y_max - y_min + 1

    # Score components (all in [0, 1] range, higher = better)

    # 1. Area score: prefer larger masks (more road coverage)
    area_score = min(1.0, area_frac / 0.30)  # saturates at 30% of frame

    # 2. Vertical position score: prefer masks in lower part
    vertical_score = cy / frame_h  # 0 at top, 1 at bottom

    # 3. Lower-half coverage: what fraction of mask pixels are in bottom 60%?
    lower_threshold = int(frame_h * 0.40)
    lower_pixels = np.sum(ys >= lower_threshold)
    lower_coverage = lower_pixels / max(1, area)

    # 4. Width score: prefer wide masks (roads are horizontally spread)
    width_score = min(1.0, bbox_w / (frame_w * 0.40))

    # 5. Elongation: prefer wider-than-tall (aspect ratio > 1)
    aspect = bbox_w / max(1, bbox_h)
    elongation_score = min(1.0, aspect / 2.0)

    color_score = 0.5
    if frame_bgr is not None:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask_bool = mask > 0
        hsv_px = hsv[mask_bool]
        bgr_px = frame_bgr[mask_bool].astype(np.float32)
        if len(hsv_px) > 0:
            hue = hsv_px[:, 0]
            sat = hsv_px[:, 1]
            val = hsv_px[:, 2]
            b = bgr_px[:, 0]
            g = bgr_px[:, 1]
            r = bgr_px[:, 2]

            greenish = (
                (hue >= 35) & (hue <= 95) &
                (sat > 35) &
                (g > r * 1.08) &
                (g > b * 1.04)
            )
            green_ratio = float(np.mean(greenish))
            low_sat_ratio = float(np.mean(sat < 55))
            channel_spread = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])
            gray_ratio = float(np.mean((channel_spread < 35) | (sat < 45)))
            dark_valid_ratio = float(np.mean((val > 25) & (val < 245)))

            # A low, wide grass mask looks geometrically road-like. Reject it
            # before scoring so rainy cameras do not bootstrap the lawn.
            if green_ratio > 0.25 and gray_ratio < 0.55:
                return -1.0
            color_score = max(
                0.0,
                min(1.0, 0.45 * low_sat_ratio + 0.35 * gray_ratio + 0.20 * dark_valid_ratio - 1.10 * green_ratio),
            )

    # Weighted combination
    score = (
        0.20 * area_score +
        0.15 * vertical_score +
        0.20 * lower_coverage +
        0.12 * width_score +
        0.13 * elongation_score +
        0.20 * color_score
    )

    return score


def build_color_road_mask(
    frame_bgr: np.ndarray,
    top_margin: float = 0.15,
    min_area_frac: float = 0.02,
) -> Optional[np.ndarray]:
    """Fallback road mask from low-saturation, non-green asphalt-like pixels."""
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    b = frame_bgr[:, :, 0].astype(np.float32)
    g = frame_bgr[:, :, 1].astype(np.float32)
    r = frame_bgr[:, :, 2].astype(np.float32)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    yy = np.arange(h)[:, None]
    lower_frame = yy >= int(h * top_margin)
    greenish = (
        (hue >= 35) & (hue <= 95) &
        (sat > 35) &
        (g > r * 1.08) &
        (g > b * 1.04)
    )
    channel_spread = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])
    asphalt_like = ((sat < 65) | (channel_spread < 42)) & (val > 25) & (val < 245)
    mask = (lower_frame & asphalt_like & ~greenish).astype(np.uint8)

    kernel_close = np.ones((35, 35), np.uint8)
    kernel_open = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None
    min_area = max(800, int(h * w * min_area_frac))
    components = [
        (idx, int(stats[idx, cv2.CC_STAT_AREA]))
        for idx in range(1, num_labels)
        if int(stats[idx, cv2.CC_STAT_AREA]) >= min_area
    ]
    if not components:
        return None
    components.sort(key=lambda item: item[1], reverse=True)
    keep = {idx for idx, _ in components[:3]}
    out = np.zeros((h, w), dtype=np.uint8)
    for idx in keep:
        out[labels == idx] = 1
    return out


# ──────────────────────────────────────────────────────────────────
# ROI polygon construction
# ──────────────────────────────────────────────────────────────────

def masks_to_roi_polygon(
    masks: List[np.ndarray],
    h: int,
    w: int,
    roi_expand: float = 1.08,
) -> np.ndarray:
    """
    Union multiple binary masks, apply morphological cleanup,
    extract the largest contour, simplify, and expand.
    """
    combined = np.zeros((h, w), dtype=np.uint8)
    for mask in masks:
        combined = np.maximum(combined, (mask > 0).astype(np.uint8))

    # Morphological cleanup
    kernel_close = np.ones((25, 25), np.uint8)
    kernel_dilate = np.ones((15, 15), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    combined = cv2.dilate(combined, kernel_dilate, iterations=1)

    # Find largest contour
    contours, _ = cv2.findContours(
        combined * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        # Full-frame fallback
        return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.int32)

    c = max(contours, key=cv2.contourArea)

    # Simplify polygon
    hull = cv2.convexHull(c)
    perim = cv2.arcLength(hull, True)
    best = hull
    for factor in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]:
        simplified = cv2.approxPolyDP(hull, factor * perim, True)
        best = simplified
        if len(simplified) <= 6:
            break

    poly = best[:, 0, :]

    # Expand from centroid
    pts = poly.astype(np.float32)
    center = pts.mean(axis=0, keepdims=True)
    expanded = (pts - center) * roi_expand + center
    expanded[:, 0] = np.clip(expanded[:, 0], 0, w - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, h - 1)

    return expanded.astype(np.int32)


# ──────────────────────────────────────────────────────────────────
# PCA-based MOI vector extraction
# ──────────────────────────────────────────────────────────────────

def pca_line(points_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a PCA line through 2D points, return the two endpoints."""
    mean = points_xy.mean(axis=0)
    centered = points_xy - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    proj = centered @ direction
    p1 = mean + direction * proj.min()
    p2 = mean + direction * proj.max()
    return p1, p2


def normalize_vectors_2d(arr: np.ndarray, dim: int = 4) -> np.ndarray:
    """Return an Nx4 vector matrix even when OpenCV returns a transposed center."""
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


def extract_moi_vectors(
    masks: List[np.ndarray],
    moi_count: int,
    min_mask_pixels: int = 500,
) -> Dict[int, List[float]]:
    """
    Extract MOI direction vectors from mask components using PCA,
    then cluster with KMeans.
    """
    vectors: List[np.ndarray] = []

    for mask in masks:
        ys, xs = np.where(mask > 0)
        if len(xs) < min_mask_pixels:
            continue

        pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        p1, p2 = pca_line(pts)
        vectors.append(np.array([p1[0], p1[1], p2[0], p2[1]], dtype=np.float32))

    if not vectors:
        return {}

    data = normalize_vectors_2d(np.vstack(vectors), dim=4)
    if data.shape[0] == 0:
        return {}
    if data.shape[0] == 1:
        centers = data
    else:
        k = max(1, min(moi_count, len(data)))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        _, _, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        centers = sort_vector_centers(centers)

    moi_vectors: Dict[int, List[float]] = {}
    for idx, c in enumerate(centers, start=1):
        moi_vectors[idx] = [float(c[0]), float(c[1]), float(c[2]), float(c[3])]

    return moi_vectors


# ──────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Import SAM here to give clear error if not installed
    try:
        from ultralytics import SAM
    except ImportError as exc:
        raise RuntimeError(
            "Missing ultralytics. Install with: pip install ultralytics"
        ) from exc

    # Read first frame
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Cannot read first frame")

    h, w = frame.shape[:2]
    print(f"[sam-auto] Frame size: {w}x{h}")

    # ── Step 1: SAM Automatic mode ────────────────────────────────
    print("[sam-auto] Running SAM automatic segmentation...")
    sam = SAM(args.sam_model)

    # Use SAM's automatic mode (no prompts, segment everything)
    results = sam.predict(frame, verbose=False)

    # Collect all masks
    all_masks: List[np.ndarray] = []
    if results:
        for r in results:
            if getattr(r, "masks", None) is None:
                continue
            masks_data = r.masks.data.cpu().numpy()
            if masks_data.ndim == 2:
                masks_data = masks_data[None, :, :]
            for mk in masks_data:
                mk = np.squeeze(mk)
                if mk.ndim != 2:
                    continue
                if mk.shape[0] != h or mk.shape[1] != w:
                    mk = cv2.resize(mk.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                all_masks.append((mk > 0).astype(np.uint8))

    print(f"[sam-auto] SAM produced {len(all_masks)} masks total")

    # ── Step 2: Score and filter masks ────────────────────────────
    scored_masks: List[Tuple[float, np.ndarray]] = []
    for mask in all_masks:
        score = score_road_mask(
            mask, h, w,
            frame_bgr=frame,
            top_margin=args.top_margin,
            min_area_frac=args.min_area_frac,
            max_area_frac=args.max_area_frac,
        )
        if score > 0:
            scored_masks.append((score, mask))

    # Sort by score descending, keep top N
    scored_masks.sort(key=lambda x: x[0], reverse=True)
    road_masks = [m for _, m in scored_masks[:args.max_masks]]
    print(f"[sam-auto] Kept {len(road_masks)} road-surface masks (from {len(scored_masks)} candidates)")

    used_full_frame_fallback = False
    used_color_fallback = False
    if not road_masks:
        color_mask = build_color_road_mask(frame, top_margin=args.top_margin, min_area_frac=args.min_area_frac)
        if color_mask is not None:
            road_masks = [color_mask]
            used_color_fallback = True
            print("[sam-auto] SAM masks did not pass road-color checks. Using color road-mask fallback.")

    if not road_masks:
        print("[sam-auto] WARNING: No road masks found. Using full-frame ROI fallback.")
        roi_poly = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.int32
        )
        moi_vectors: Dict[int, List[float]] = {}
        used_full_frame_fallback = True
    else:
        # ── Step 3: Build ROI polygon ─────────────────────────────
        roi_poly = masks_to_roi_polygon(road_masks, h, w, args.roi_expand)
        print(f"[sam-auto] ROI polygon: {len(roi_poly)} vertices")

        # ── Step 4: Extract MOI vectors ───────────────────────────
        moi_vectors = extract_moi_vectors(road_masks, args.moi_count)
        print(f"[sam-auto] MOI vectors: {len(moi_vectors)}")

    xs = roi_poly[:, 0].astype(float)
    ys = roi_poly[:, 1].astype(float)
    width_ratio = (xs.max() - xs.min()) / max(1.0, float(w - 1))
    height_ratio = (ys.max() - ys.min()) / max(1.0, float(h - 1))
    full_like = used_full_frame_fallback or (width_ratio > 0.98 and height_ratio > 0.98)
    few_moi = 0 < len(moi_vectors) < min(max(args.moi_count, 1), 3)
    quality_status = "low_confidence" if full_like or used_color_fallback or len(moi_vectors) == 0 or few_moi else "ok"
    quality_reason = ""
    if full_like:
        quality_reason = "SAM Automatic produced a full-frame-like ROI."
    elif used_color_fallback:
        quality_reason = "SAM road-mask scoring rejected likely vegetation; used color fallback."
    elif len(moi_vectors) == 0:
        quality_reason = "SAM Automatic produced no valid MOI vectors."
    elif few_moi:
        quality_reason = "SAM Automatic produced too few MOI vectors from a single frame."

    # ── Step 5: Write JSON ────────────────────────────────────────
    payload = {
        "roi": roi_poly.astype(float).tolist(),
        "moi_vectors": {str(k): v for k, v in moi_vectors.items()},
        "quality": {
            "status": quality_status,
            "reason": quality_reason,
            "is_full_frame_fallback": bool(full_like),
            "used_color_fallback": bool(used_color_fallback),
            "road_masks_used": len(road_masks),
            "valid_moi_count": len(moi_vectors),
            "roi_width_ratio": round(float(width_ratio), 4),
            "roi_height_ratio": round(float(height_ratio), 4),
        },
        "note": (
            "Auto-generated by sam_auto_bootstrap.py (SAM Automatic mode, no detector). "
            f"ROI has {len(roi_poly)} vertices. "
            f"Road masks used: {len(road_masks)}. "
            "Please review before production use."
        ),
    }

    out_dir = os.path.dirname(os.path.abspath(args.output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)

    # ── Step 6: Optional overlay ──────────────────────────────────
    if args.save_overlay:
        vis = frame.copy()

        # Draw individual road masks as colored overlays
        colors = [
            (0, 200, 100), (100, 200, 0), (0, 100, 200),
            (200, 100, 0), (100, 0, 200), (0, 200, 200),
            (200, 0, 100), (200, 200, 0),
        ]
        mask_overlay = np.zeros_like(vis)
        for i, mask in enumerate(road_masks):
            color = colors[i % len(colors)]
            mask_overlay[mask > 0] = color

        vis = cv2.addWeighted(vis, 0.7, mask_overlay, 0.4, 0)

        # Draw ROI polygon (cyan)
        cv2.polylines(vis, [roi_poly.astype(np.int32)], True, (0, 255, 255), 3)

        # Draw MOI arrows (orange)
        for mid, vec in moi_vectors.items():
            x1, y1, x2, y2 = [int(v) for v in vec]
            cv2.arrowedLine(vis, (x1, y1), (x2, y2), (0, 165, 255), 2, tipLength=0.08)
            cv2.putText(
                vis, str(mid), (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
            )

        # Title text
        cv2.putText(
            vis, "SAM Automatic Bootstrap (No Detector)",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

        overlay_dir = os.path.dirname(args.save_overlay)
        if overlay_dir:
            os.makedirs(overlay_dir, exist_ok=True)
        cv2.imwrite(args.save_overlay, vis)
        print(f"[sam-auto] Overlay saved: {args.save_overlay}")

    print(
        f"Done. Wrote: {args.output_json} | "
        f"ROI vertices: {len(roi_poly)} | "
        f"MOI vectors: {len(moi_vectors)} | "
        f"Road masks: {len(road_masks)}"
    )


if __name__ == "__main__":
    main()
