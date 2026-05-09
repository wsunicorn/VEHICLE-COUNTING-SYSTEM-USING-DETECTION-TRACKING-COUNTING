"""
evaluate_counting.py — AI City Challenge 2021 Track-1 evaluation metrics.

Implements:
  • nwRMSE (normalized weighted Root Mean Square Error)
  • S1_Effectiveness  = max(0, 1 - nwRMSE)
  • S1_Efficiency     based on processing speed vs real-time
  • S1 = 0.3 × Efficiency + 0.7 × Effectiveness       … Eq. (1) in paper
  • Per-movement / per-class MAE breakdown
  • Total count accuracy

Usage:
    python evaluate_counting.py \
        --gt-csv  counting_example_cam_5_1min.csv \
        --pred-csv outputs/cam5_pred.csv \
        --total-frames 600 --video-fps 10 \
        --processing-time 45.3 \
        --output-json outputs/eval_cam5.json
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

Event = Tuple[int, int, int, int]  # (video_clip_id, frame_id, movement_id, vehicle_class_id)


# ─────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate vehicle counting results using AI City Challenge 2021 Track-1 metrics."
    )
    p.add_argument("--gt-csv", required=True, help="Ground truth counting CSV.")
    p.add_argument("--pred-csv", required=True, help="Predicted counting CSV.")
    p.add_argument("--total-frames", type=int, default=0,
                   help="Total number of frames in the video (0 = auto-detect from GT).")
    p.add_argument("--video-fps", type=float, default=10.0,
                   help="Video frames per second (default 10, matching AI City).")
    p.add_argument("--processing-time", type=float, default=0.0,
                   help="Total processing wall-clock time in seconds (for efficiency scoring).")
    p.add_argument("--segment-duration", type=float, default=5.0,
                   help="Duration (seconds) per evaluation segment (default 5s).")
    p.add_argument("--output-json", default="",
                   help="Optional path to write detailed evaluation JSON.")
    p.add_argument("--verbose", action="store_true", help="Print per-segment details.")
    return p.parse_args()


def read_csv(path: str) -> List[Event]:
    """Read a counting CSV and return list of events."""
    events: List[Event] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append((
                int(row.get("video_clip_id", 0)),
                int(row["frame_id"]),
                int(row["movement_id"]),
                int(row["vehicle_class_id"]),
            ))
    return events


# ─────────────────────────────────────────────────────────────────────
# Core nwRMSE computation (AI City Challenge 2021 formulation)
# ─────────────────────────────────────────────────────────────────────

def compute_nwrmse(
    gt_events: List[Event],
    pred_events: List[Event],
    total_frames: int,
    fps: float,
    segment_duration: float = 5.0,
    verbose: bool = False,
) -> Tuple[float, List[dict]]:
    """
    Compute normalized weighted RMSE following the AI City Challenge
    2021 Track-1 evaluation protocol.

    Steps:
      1. Divide video into segments of `segment_duration` seconds.
      2. For each (movement_id, class_id) pair, compute cumulative
         counts at the end of each segment for both GT and prediction.
      3. Weight each segment by its temporal position
         (later segments carry more weight).
      4. Compute weighted RMSE, then normalize.

    Returns (nwrmse_value, segment_details).
    """
    segment_frames = max(1, int(fps * segment_duration))
    n_segments = max(1, math.ceil(total_frames / segment_frames))

    # Collect all (movement_id, class_id) pairs from both GT and pred
    all_keys = set()
    for _, _, mid, cid in gt_events:
        all_keys.add((mid, cid))
    for _, _, mid, cid in pred_events:
        all_keys.add((mid, cid))

    if not all_keys:
        return 0.0, []

    # Build cumulative count arrays for GT and pred per key per segment
    def build_cumulative(events: List[Event]) -> Dict[Tuple[int, int], List[int]]:
        """
        Returns {(mid, cid): [cum_count_seg_0, cum_count_seg_1, ...]}
        """
        counts: Dict[Tuple[int, int], List[int]] = {}
        for key in all_keys:
            counts[key] = [0] * n_segments

        for _, frame_id, mid, cid in events:
            key = (mid, cid)
            if key not in counts:
                counts[key] = [0] * n_segments
            seg_idx = min(frame_id // segment_frames, n_segments - 1)
            counts[key][seg_idx] += 1

        # Convert to cumulative
        for key in counts:
            for s in range(1, n_segments):
                counts[key][s] += counts[key][s - 1]
        return counts

    gt_cum = build_cumulative(gt_events)
    pred_cum = build_cumulative(pred_events)

    # Compute weighted sum of squared errors
    # Weight for segment s = (s + 1) / n_segments  (linear increasing weight)
    total_weight = 0.0
    weighted_sse = 0.0
    segment_details = []

    for s in range(n_segments):
        weight = (s + 1) / n_segments
        total_weight += weight * len(all_keys)  # weight applied per key per segment

        seg_info = {
            "segment": s,
            "frame_range": f"{s * segment_frames}-{min((s + 1) * segment_frames - 1, total_frames - 1)}",
            "weight": round(weight, 4),
            "errors": {},
        }

        for key in sorted(all_keys):
            gt_val = gt_cum[key][s] if key in gt_cum else 0
            pred_val = pred_cum[key][s] if key in pred_cum else 0
            error = pred_val - gt_val
            weighted_sse += weight * (error ** 2)
            seg_info["errors"][f"m{key[0]}_c{key[1]}"] = {
                "gt_cum": gt_val,
                "pred_cum": pred_val,
                "diff": error,
            }

        segment_details.append(seg_info)

    # nwRMSE = sqrt(weighted_SSE / total_weight)
    # Normalize by the total GT count to get a ratio
    gt_total = len(gt_events)
    raw_wrmse = math.sqrt(weighted_sse / max(1.0, total_weight))

    # Normalize: divide by max(1, gt_total_per_key_average)
    avg_gt_per_key = gt_total / max(1, len(all_keys))
    nwrmse = raw_wrmse / max(1.0, avg_gt_per_key)

    if verbose:
        print(f"\n{'='*60}")
        print(f"nwRMSE Computation Details")
        print(f"{'='*60}")
        print(f"  Segments: {n_segments} (each {segment_duration}s = {segment_frames} frames)")
        print(f"  Movement-class pairs: {len(all_keys)}")
        print(f"  GT total events: {gt_total}")
        print(f"  Pred total events: {len(pred_events)}")
        print(f"  Raw wRMSE: {raw_wrmse:.4f}")
        print(f"  Avg GT per key: {avg_gt_per_key:.2f}")
        print(f"  nwRMSE: {nwrmse:.4f}")

    return nwrmse, segment_details


# ─────────────────────────────────────────────────────────────────────
# S1 score components
# ─────────────────────────────────────────────────────────────────────

def compute_effectiveness(nwrmse: float) -> float:
    """S1_Effectiveness = max(0, 1 - nwRMSE)"""
    return max(0.0, 1.0 - nwrmse)


def compute_efficiency(
    processing_time: float,
    total_frames: int,
    fps: float,
) -> float:
    """
    S1_Efficiency based on processing speed vs real-time.

    If processing_time <= video_duration → efficiency approaches 1.0
    If processing_time >> video_duration → efficiency approaches 0.0
    """
    if processing_time <= 0:
        return 0.0  # unknown

    video_duration = total_frames / max(1.0, fps)
    if video_duration <= 0:
        return 0.0

    ratio = processing_time / video_duration
    # efficiency = max(0, 1 - (ratio - 1)) when ratio > 1,
    # capped at 1.0 when ratio <= 1
    efficiency = max(0.0, min(1.0, 2.0 - ratio))
    return efficiency


def compute_s1(effectiveness: float, efficiency: float) -> float:
    """S1 = 0.3 × Efficiency + 0.7 × Effectiveness"""
    return 0.3 * efficiency + 0.7 * effectiveness


# ─────────────────────────────────────────────────────────────────────
# Per-movement / per-class detailed metrics
# ─────────────────────────────────────────────────────────────────────

def compute_per_movement_metrics(
    gt_events: List[Event],
    pred_events: List[Event],
) -> dict:
    """
    Compute per-(movement_id, class_id) metrics:
      - Total count GT vs Pred
      - Absolute error
      - MAE across all pairs
    """
    gt_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    pred_counts: Dict[Tuple[int, int], int] = defaultdict(int)

    for _, _, mid, cid in gt_events:
        gt_counts[(mid, cid)] += 1
    for _, _, mid, cid in pred_events:
        pred_counts[(mid, cid)] += 1

    all_keys = sorted(set(gt_counts.keys()) | set(pred_counts.keys()))

    breakdown = []
    total_abs_error = 0
    for key in all_keys:
        gt_val = gt_counts.get(key, 0)
        pred_val = pred_counts.get(key, 0)
        abs_err = abs(pred_val - gt_val)
        total_abs_error += abs_err
        cls_name = "car" if key[1] == 1 else "truck"
        breakdown.append({
            "movement_id": key[0],
            "class": cls_name,
            "class_id": key[1],
            "gt_count": gt_val,
            "pred_count": pred_val,
            "abs_error": abs_err,
            "relative_error": round(abs_err / max(1, gt_val), 4),
        })

    mae = total_abs_error / max(1, len(all_keys))

    # Total counts
    gt_total = sum(gt_counts.values())
    pred_total = sum(pred_counts.values())

    # Count accuracy = 1 - |pred_total - gt_total| / gt_total
    count_accuracy = 1.0 - abs(pred_total - gt_total) / max(1, gt_total)
    count_accuracy = max(0.0, count_accuracy)

    return {
        "gt_total": gt_total,
        "pred_total": pred_total,
        "total_abs_error": abs(pred_total - gt_total),
        "total_count_accuracy": round(count_accuracy, 4),
        "mae_per_movement_class": round(mae, 4),
        "num_movement_class_pairs": len(all_keys),
        "breakdown": breakdown,
    }


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if not os.path.exists(args.gt_csv):
        print(f"ERROR: GT CSV not found: {args.gt_csv}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.pred_csv):
        print(f"ERROR: Pred CSV not found: {args.pred_csv}", file=sys.stderr)
        sys.exit(1)

    gt_events = read_csv(args.gt_csv)
    pred_events = read_csv(args.pred_csv)

    # Auto-detect total frames from GT if not specified
    total_frames = args.total_frames
    if total_frames <= 0:
        max_frame = max((e[1] for e in gt_events), default=0)
        max_frame_pred = max((e[1] for e in pred_events), default=0)
        total_frames = max(max_frame, max_frame_pred) + 1
        print(f"[info] Auto-detected total_frames = {total_frames}")

    fps = args.video_fps

    # ── 1. nwRMSE ─────────────────────────────────────────────────
    nwrmse, segment_details = compute_nwrmse(
        gt_events, pred_events, total_frames, fps,
        segment_duration=args.segment_duration,
        verbose=args.verbose,
    )

    # ── 2. S1 components ─────────────────────────────────────────
    effectiveness = compute_effectiveness(nwrmse)
    efficiency = compute_efficiency(args.processing_time, total_frames, fps)
    s1 = compute_s1(effectiveness, efficiency)

    # ── 3. Per-movement breakdown ─────────────────────────────────
    movement_metrics = compute_per_movement_metrics(gt_events, pred_events)

    video_duration = total_frames / max(1.0, fps)
    proc_fps = total_frames / max(1e-6, args.processing_time) if args.processing_time > 0 else 0.0

    # ── 4. Compile results ────────────────────────────────────────
    results = {
        "metadata": {
            "gt_csv": os.path.basename(args.gt_csv),
            "pred_csv": os.path.basename(args.pred_csv),
            "total_frames": total_frames,
            "video_fps": fps,
            "video_duration_sec": round(video_duration, 2),
            "processing_time_sec": round(args.processing_time, 2),
            "processing_fps": round(proc_fps, 2),
            "segment_duration_sec": args.segment_duration,
        },
        "scores": {
            "nwRMSE": round(nwrmse, 4),
            "S1_Effectiveness": round(effectiveness, 4),
            "S1_Efficiency": round(efficiency, 4),
            "S1_Overall": round(s1, 4),
        },
        "counting": movement_metrics,
    }

    if args.verbose:
        results["segment_details"] = segment_details

    # ── 5. Print summary ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Video: {total_frames} frames @ {fps} FPS = {video_duration:.1f}s")
    if args.processing_time > 0:
        print(f"  Processing: {args.processing_time:.2f}s ({proc_fps:.1f} FPS)")
    print(f"{'~'*60}")
    print(f"  nwRMSE           : {nwrmse:.4f}")
    print(f"  S1_Effectiveness : {effectiveness:.4f}")
    print(f"  S1_Efficiency    : {efficiency:.4f}")
    print(f"  S1 (Overall)     : {s1:.4f}")
    print(f"{'~'*60}")
    print(f"  Total vehicles   : GT={movement_metrics['gt_total']}  Pred={movement_metrics['pred_total']}")
    print(f"  Count accuracy   : {movement_metrics['total_count_accuracy']:.1%}")
    print(f"  MAE per mov/cls  : {movement_metrics['mae_per_movement_class']:.2f}")
    print(f"{'~'*60}")

    # Per-movement table
    print(f"\n  {'Mov':>4} {'Class':>6} {'GT':>5} {'Pred':>5} {'Error':>6} {'Rel%':>7}")
    print(f"  {'-'*4} {'-'*6} {'-'*5} {'-'*5} {'-'*6} {'-'*7}")
    for item in movement_metrics["breakdown"]:
        rel_pct = f"{item['relative_error']*100:.1f}%"
        print(
            f"  {item['movement_id']:>4} {item['class']:>6} "
            f"{item['gt_count']:>5} {item['pred_count']:>5} "
            f"{item['abs_error']:>6} {rel_pct:>7}"
        )
    print(f"{'='*60}\n")

    # ── 6. Write JSON ─────────────────────────────────────────────
    if args.output_json:
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  Saved detailed results to: {args.output_json}")


if __name__ == "__main__":
    main()
