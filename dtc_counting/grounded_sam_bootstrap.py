import argparse
import inspect
import json
import os
import warnings
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import SAM

try:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
except Exception as exc:
    raise RuntimeError(
        "Missing transformers dependency. Install with: pip install transformers"
    ) from exc


Point = Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grounded DINO + SAM bootstrap for ROI/MOI.")
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument("--output-json", required=True, help="Output JSON for ROI and MOI vectors.")
    parser.add_argument("--save-overlay", default="", help="Optional overlay path.")
    parser.add_argument("--sam-model", default="sam_b.pt", help="Ultralytics SAM checkpoint.")
    parser.add_argument(
        "--grounding-model",
        default="IDEA-Research/grounding-dino-base",
        help="Grounding DINO model id from HuggingFace.",
    )
    parser.add_argument(
        "--text-prompt",
        default="road surface . traffic lane . intersection",
        help="Grounding prompt.",
    )
    parser.add_argument("--box-threshold", type=float, default=0.35, help="Grounding score threshold.")
    parser.add_argument("--moi-count", type=int, default=12, help="Expected MOI count.")
    parser.add_argument("--roi-expand", type=float, default=1.12, help="Scale factor to slightly expand ROI polygon.")
    return parser.parse_args()


def largest_polygon_from_mask(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.empty((0, 2), dtype=np.int32)
    c = max(contours, key=cv2.contourArea)
    eps = 0.003 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True)
    return approx[:, 0, :]


def filter_mask_components(mask: np.ndarray, min_area: int = 800, max_components: int = 6) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8

    valid = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            valid.append((i, area))
    if not valid:
        return mask_u8

    valid.sort(key=lambda x: x[1], reverse=True)
    keep_ids = {idx for idx, _ in valid[:max_components]}
    out = np.zeros_like(mask_u8)
    for idx in keep_ids:
        out[labels == idx] = 1
    return out


def expand_polygon(poly: np.ndarray, w: int, h: int, scale: float) -> np.ndarray:
    if poly.size == 0:
        return poly
    pts = poly.astype(np.float32)
    center = pts.mean(axis=0, keepdims=True)
    expanded = (pts - center) * float(scale) + center
    expanded[:, 0] = np.clip(expanded[:, 0], 0, w - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, h - 1)
    return expanded.astype(np.int32)


def pca_line(points_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = points_xy.mean(axis=0)
    centered = points_xy - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    proj = centered @ direction
    p1 = mean + direction * proj.min()
    p2 = mean + direction * proj.max()
    return p1, p2


def normalize_vectors_2d(data: np.ndarray, dim: int = 4) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
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
    order = np.argsort(np.arctan2(centers[:, 3] - centers[:, 1], centers[:, 2] - centers[:, 0]))
    return centers[order]


def kmeans_vectors(vectors: List[np.ndarray], k: int) -> np.ndarray:
    if not vectors:
        return np.zeros((0, 4), dtype=np.float32)
    data = normalize_vectors_2d(np.vstack(vectors), dim=4)
    if data.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    k = max(1, min(k, len(data)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    _, _, centers = cv2.kmeans(data, k, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
    centers = sort_vector_centers(centers)
    if 0 < centers.shape[0] < k:
        pad = np.repeat(centers[-1:, :], k - centers.shape[0], axis=0)
        centers = np.vstack([centers, pad])
    return centers


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Cannot read first frame")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load from local cache first to avoid transient network issues on HF Hub.
    try:
        processor = AutoProcessor.from_pretrained(
            args.grounding_model,
            use_fast=False,
            local_files_only=True,
        )
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            args.grounding_model,
            local_files_only=True,
        ).to(device)
        print("Grounding model loaded from local cache.")
    except Exception:
        try:
            processor = AutoProcessor.from_pretrained(args.grounding_model, use_fast=False)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(args.grounding_model).to(device)
            print("Grounding model loaded from HF Hub.")
        except Exception as exc:
            raise RuntimeError(
                "Cannot load grounding model (local cache unavailable and HF Hub request failed)."
            ) from exc

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Try a few prompt/threshold combinations before giving up.
    candidate_prompts = [
        args.text_prompt,
        "road . traffic lane . intersection",
        "road surface . lane markings . street",
    ]
    candidate_thresholds = [args.box_threshold, 0.25, 0.15, 0.05]

    post_fn = processor.post_process_grounded_object_detection
    sig = inspect.signature(post_fn)
    target_sizes = torch.tensor([rgb.shape[:2]], device=device)

    boxes = np.zeros((0, 4), dtype=np.float32)
    for prompt in candidate_prompts:
        if boxes.shape[0] > 0:
            break
        for th in candidate_thresholds:
            inputs = processor(images=rgb, text=prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            kwargs = {}
            if "target_sizes" in sig.parameters:
                kwargs["target_sizes"] = target_sizes
            if "box_threshold" in sig.parameters:
                kwargs["box_threshold"] = th
            elif "threshold" in sig.parameters:
                kwargs["threshold"] = th
            if "text_threshold" in sig.parameters:
                kwargs["text_threshold"] = 0.20

            if "input_ids" in sig.parameters:
                results = post_fn(outputs, inputs.input_ids, **kwargs)
            else:
                results = post_fn(outputs, **kwargs)

            raw_boxes = (
                results[0]["boxes"].detach().cpu().numpy()
                if len(results)
                else np.zeros((0, 4), dtype=np.float32)
            )
            trial_boxes = normalize_vectors_2d(raw_boxes, dim=4)
            if trial_boxes.shape[0] > 0:
                print(f"Grounding DINO detected {trial_boxes.shape[0]} boxes using prompt='{prompt}' threshold={th}")
                boxes = trial_boxes
                break

    used_full_frame_fallback = False
    if boxes.shape[0] == 0:
        warnings.warn(
            "Grounding DINO found no road/lane boxes after retries. Falling back to full-frame ROI and no MOI vectors."
        )
        used_full_frame_fallback = True

    combined = np.zeros(frame.shape[:2], dtype=np.uint8)
    vectors: List[np.ndarray] = []

    if boxes.shape[0] > 0:
        sam = SAM(args.sam_model)
        sam_res = sam.predict(frame, bboxes=boxes.astype(np.float32), verbose=False)

        for r in sam_res:
            if getattr(r, "masks", None) is None:
                continue
            masks = r.masks.data.detach().cpu().numpy()
            masks = np.asarray(masks)
            if masks.ndim == 2:
                masks = masks[None, :, :]
            elif masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0, :, :]
            if masks.ndim != 3:
                continue
            masks = (masks > 0).astype(np.uint8)

            for mk in masks:
                mk2 = np.squeeze(mk)
                if mk2.ndim != 2:
                    continue
                ys, xs = np.where(mk2 > 0)
                if len(xs) < 50:
                    continue

                combined = np.maximum(combined, mk2.astype(np.uint8))
                pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
                p1, p2 = pca_line(pts)
                vectors.append(np.array([p1[0], p1[1], p2[0], p2[1]], dtype=np.float32))

    h, w = frame.shape[:2]
    combined = cv2.morphologyEx((combined > 0).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8), iterations=1)
    combined = cv2.dilate(combined, np.ones((15, 15), np.uint8), iterations=1)
    combined = filter_mask_components(combined, min_area=max(600, (h * w) // 1800), max_components=6)

    roi_poly = largest_polygon_from_mask(combined)
    if roi_poly.size == 0:
        roi_poly = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.int32)
        used_full_frame_fallback = True
    else:
        roi_poly = expand_polygon(roi_poly, w, h, args.roi_expand)

    centers = kmeans_vectors(vectors, args.moi_count)
    moi_vectors: Dict[int, List[float]] = {}
    for idx, c in enumerate(centers, start=1):
        moi_vectors[idx] = [float(c[0]), float(c[1]), float(c[2]), float(c[3])]

    xs = roi_poly[:, 0].astype(float)
    ys = roi_poly[:, 1].astype(float)
    width_ratio = (xs.max() - xs.min()) / max(1.0, float(w - 1))
    height_ratio = (ys.max() - ys.min()) / max(1.0, float(h - 1))
    full_like = used_full_frame_fallback or (width_ratio > 0.98 and height_ratio > 0.98)
    quality_status = "low_confidence" if full_like or len(moi_vectors) == 0 else "ok"
    quality_reason = ""
    if full_like:
        quality_reason = "Grounded-SAM produced a full-frame-like ROI."
    elif len(moi_vectors) == 0:
        quality_reason = "Grounded-SAM produced no valid MOI vectors."

    payload = {
        "roi": roi_poly.astype(float).tolist(),
        "moi_vectors": moi_vectors,
        "quality": {
            "status": quality_status,
            "reason": quality_reason,
            "is_full_frame_fallback": bool(full_like),
            "grounding_box_count": int(boxes.shape[0]),
            "valid_moi_count": len(moi_vectors),
            "roi_width_ratio": round(float(width_ratio), 4),
            "roi_height_ratio": round(float(height_ratio), 4),
        },
        "note": "Auto-generated by grounded_sam_bootstrap.py. Please review before production use.",
    }

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)

    if args.save_overlay:
        vis = frame.copy()
        cv2.polylines(vis, [roi_poly.astype(np.int32)], True, (0, 255, 255), 2)
        for mid, vec in moi_vectors.items():
            x1, y1, x2, y2 = [int(v) for v in vec]
            cv2.arrowedLine(vis, (x1, y1), (x2, y2), (0, 180, 255), 2, tipLength=0.07)
            cv2.putText(vis, str(mid), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imwrite(args.save_overlay, vis)

    print(f"Done. Wrote: {args.output_json}")


if __name__ == "__main__":
    main()
