"""Evaluate AI City Track-1 vehicle-counting outputs.

Supported inputs:
  - Local CSV with header:
      video_clip_id,frame_id,movement_id,vehicle_class_id
  - Official-like Track-1 TXT rows:
      gen_time video_id frame_id movement_id vehicle_class_id

The effectiveness calculation follows the public AI City 2021 description:
per video/movement/class cumulative counts are compared at segment boundaries,
weighted more strongly for later segments, normalized by the true count for
that movement/class, then averaged with true-count weights.

Efficiency still remains an approximation unless the official Efficiency Base
script/result is supplied. The JSON output names this explicitly.
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class Event:
    video_id: int
    frame_id: int
    movement_id: int
    vehicle_class_id: int


Key = Tuple[int, int, int]  # video_id, movement_id, vehicle_class_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate AI City Track-1 vehicle counting outputs.")
    p.add_argument("--gt-csv", required=True, help="Ground-truth CSV/TXT path.")
    p.add_argument("--pred-csv", required=True, help="Prediction CSV/TXT path.")
    p.add_argument("--gt-format", choices=["auto", "local_csv", "aicity_txt"], default="auto")
    p.add_argument("--pred-format", choices=["auto", "local_csv", "aicity_txt"], default="auto")
    p.add_argument("--total-frames", type=int, default=0, help="Total frames; 0 means infer from data.")
    p.add_argument("--video-fps", type=float, default=10.0, help="Video FPS.")
    p.add_argument("--frame-base", choices=["auto", "0", "1"], default="auto", help="Frame-id base for segmenting.")
    p.add_argument("--processing-time", type=float, default=0.0, help="Wall-clock processing time in seconds.")
    p.add_argument("--efficiency-base", type=float, default=0.0, help="Optional hardware efficiency base factor.")
    p.add_argument("--segment-duration", type=float, default=5.0, help="Evaluation segment duration in seconds.")
    p.add_argument("--output-json", default="", help="Optional detailed JSON output path.")
    p.add_argument("--verbose", action="store_true", help="Include per-segment/per-key details in JSON.")
    return p.parse_args()


def _detect_format(path: str, requested: str) -> str:
    if requested != "auto":
        return requested
    with open(path, "r", encoding="utf-8-sig") as f:
        first = f.readline().strip()
    if "," in first or "frame_id" in first:
        return "local_csv"
    return "aicity_txt"


def read_events(path: str, requested_format: str = "auto") -> List[Event]:
    fmt = _detect_format(path, requested_format)
    events: List[Event] = []
    if fmt == "local_csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                video_raw = row.get("video_clip_id", row.get("video_id", "0"))
                events.append(
                    Event(
                        int(float(video_raw or 0)),
                        int(float(row["frame_id"])),
                        int(float(row["movement_id"])),
                        int(float(row["vehicle_class_id"])),
                    )
                )
        return events

    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.replace(",", " ").split()
            if len(parts) == 4:
                video_id, frame_id, movement_id, class_id = parts
            elif len(parts) >= 5:
                _, video_id, frame_id, movement_id, class_id = parts[:5]
            else:
                raise ValueError(f"Cannot parse {path}:{line_no}: {raw!r}")
            events.append(Event(int(float(video_id)), int(float(frame_id)), int(float(movement_id)), int(float(class_id))))
    return events


def event_key(event: Event) -> Key:
    return (event.video_id, event.movement_id, event.vehicle_class_id)


def infer_frame_base(events: Iterable[Event], requested: str) -> int:
    if requested in {"0", "1"}:
        return int(requested)
    return 0 if any(event.frame_id == 0 for event in events) else 1


def build_cumulative(
    events: List[Event],
    keys: Iterable[Key],
    n_segments: int,
    segment_frames: int,
    frame_base: int,
) -> Dict[Key, List[int]]:
    counts: Dict[Key, List[int]] = {key: [0] * n_segments for key in keys}
    for event in events:
        key = event_key(event)
        if key not in counts:
            continue
        zero_based_frame = max(0, event.frame_id - frame_base)
        seg_idx = min(zero_based_frame // segment_frames, n_segments - 1)
        counts[key][seg_idx] += 1
    for key in counts:
        for idx in range(1, n_segments):
            counts[key][idx] += counts[key][idx - 1]
    return counts


def compute_effectiveness(
    gt_events: List[Event],
    pred_events: List[Event],
    total_frames: int,
    fps: float,
    frame_base: int,
    segment_duration: float,
) -> Tuple[float, float, List[dict]]:
    segment_frames = max(1, int(round(fps * segment_duration)))
    n_segments = max(1, math.ceil(total_frames / segment_frames))
    gt_counts: Dict[Key, int] = defaultdict(int)
    for event in gt_events:
        gt_counts[event_key(event)] += 1

    gt_keys = sorted(key for key, count in gt_counts.items() if count > 0)
    if not gt_keys:
        return 0.0, 1.0, []

    gt_cum = build_cumulative(gt_events, gt_keys, n_segments, segment_frames, frame_base)
    pred_cum = build_cumulative(pred_events, gt_keys, n_segments, segment_frames, frame_base)
    weights = [idx + 1 for idx in range(n_segments)]
    weight_sum = float(sum(weights))

    weighted_score = 0.0
    total_true = 0
    details: List[dict] = []

    for key in gt_keys:
        true_count = gt_counts[key]
        total_true += true_count
        weighted_sse = 0.0
        segment_rows = []
        for idx, weight in enumerate(weights):
            diff = pred_cum[key][idx] - gt_cum[key][idx]
            weighted_sse += weight * (diff ** 2)
            segment_rows.append(
                {
                    "segment": idx,
                    "gt_cumulative": gt_cum[key][idx],
                    "pred_cumulative": pred_cum[key][idx],
                    "diff": diff,
                    "weight": weight,
                }
            )
        wrmse = math.sqrt(weighted_sse / max(1.0, weight_sum))
        key_score = max(0.0, 1.0 - wrmse / max(1, true_count))
        weighted_score += key_score * true_count
        details.append(
            {
                "video_id": key[0],
                "movement_id": key[1],
                "vehicle_class_id": key[2],
                "gt_count": true_count,
                "wRMSE": round(wrmse, 6),
                "nwRMSE_score": round(key_score, 6),
                "segments": segment_rows,
            }
        )

    effectiveness = weighted_score / max(1, total_true)
    weighted_error = 1.0 - effectiveness
    return effectiveness, weighted_error, details


def compute_efficiency(processing_time: float, total_frames: int, fps: float, efficiency_base: float) -> Tuple[float, dict]:
    video_duration = total_frames / max(1.0, fps)
    if processing_time <= 0 or video_duration <= 0:
        return 0.0, {"mode": "unknown", "reason": "processing-time not provided"}

    adjusted_time = processing_time
    if efficiency_base > 0:
        adjusted_time = processing_time / efficiency_base

    ratio = adjusted_time / video_duration
    if ratio <= 1.0:
        score = 1.0
    elif ratio >= 1.1:
        score = 0.0
    else:
        score = 1.0 - ((ratio - 1.0) / 0.1)
    return max(0.0, min(1.0, score)), {
        "mode": "aicity_approx",
        "processing_time_sec": processing_time,
        "efficiency_base": efficiency_base,
        "adjusted_time_sec": adjusted_time,
        "video_duration_sec": video_duration,
        "adjusted_time_ratio": ratio,
        "note": "Approximation; use the official AI City efficiency script for leaderboard numbers.",
    }


def compute_counting_breakdown(gt_events: List[Event], pred_events: List[Event]) -> dict:
    gt_counts: Dict[Key, int] = defaultdict(int)
    pred_counts: Dict[Key, int] = defaultdict(int)
    for event in gt_events:
        gt_counts[event_key(event)] += 1
    for event in pred_events:
        pred_counts[event_key(event)] += 1

    all_keys = sorted(set(gt_counts) | set(pred_counts))
    breakdown = []
    total_abs_by_pair = 0
    false_positive_absent_gt = 0
    for key in all_keys:
        gt_val = gt_counts.get(key, 0)
        pred_val = pred_counts.get(key, 0)
        abs_err = abs(pred_val - gt_val)
        total_abs_by_pair += abs_err
        if gt_val == 0 and pred_val > 0:
            false_positive_absent_gt += pred_val
        cls_name = "car" if key[2] == 1 else "truck"
        breakdown.append(
            {
                "video_id": key[0],
                "movement_id": key[1],
                "class": cls_name,
                "class_id": key[2],
                "gt_count": gt_val,
                "pred_count": pred_val,
                "abs_error": abs_err,
                "relative_error": round(abs_err / max(1, gt_val), 4),
            }
        )

    gt_total = len(gt_events)
    pred_total = len(pred_events)
    count_accuracy = max(0.0, 1.0 - abs(pred_total - gt_total) / max(1, gt_total))
    return {
        "gt_total": gt_total,
        "pred_total": pred_total,
        "total_abs_error": abs(pred_total - gt_total),
        "false_positive_events_with_absent_gt_key": false_positive_absent_gt,
        "total_count_accuracy": round(count_accuracy, 4),
        "mae_per_movement_class": round(total_abs_by_pair / max(1, len(all_keys)), 4),
        "num_movement_class_pairs": len(all_keys),
        "breakdown": breakdown,
    }


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.gt_csv):
        print(f"ERROR: GT file not found: {args.gt_csv}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.pred_csv):
        print(f"ERROR: prediction file not found: {args.pred_csv}", file=sys.stderr)
        sys.exit(1)

    gt_events = read_events(args.gt_csv, args.gt_format)
    pred_events = read_events(args.pred_csv, args.pred_format)
    all_events = gt_events + pred_events
    total_frames = args.total_frames
    if total_frames <= 0:
        frame_base_guess = infer_frame_base(all_events, args.frame_base)
        max_frame = max((event.frame_id for event in all_events), default=frame_base_guess)
        total_frames = max(1, max_frame - frame_base_guess + 1)
        print(f"[info] Auto-detected total_frames = {total_frames}")

    frame_base = infer_frame_base(all_events, args.frame_base)
    fps = args.video_fps
    video_duration = total_frames / max(1.0, fps)

    effectiveness, weighted_nwrmse_error, effectiveness_details = compute_effectiveness(
        gt_events,
        pred_events,
        total_frames,
        fps,
        frame_base,
        args.segment_duration,
    )
    efficiency, efficiency_details = compute_efficiency(args.processing_time, total_frames, fps, args.efficiency_base)
    s1 = 0.3 * efficiency + 0.7 * effectiveness
    counting = compute_counting_breakdown(gt_events, pred_events)
    proc_fps = total_frames / max(1e-6, args.processing_time) if args.processing_time > 0 else 0.0

    results = {
        "metadata": {
            "gt_file": os.path.basename(args.gt_csv),
            "pred_file": os.path.basename(args.pred_csv),
            "gt_format": _detect_format(args.gt_csv, args.gt_format),
            "pred_format": _detect_format(args.pred_csv, args.pred_format),
            "total_frames": total_frames,
            "frame_base": frame_base,
            "video_fps": fps,
            "video_duration_sec": round(video_duration, 2),
            "processing_time_sec": round(args.processing_time, 3),
            "processing_fps": round(proc_fps, 2),
            "segment_duration_sec": args.segment_duration,
        },
        "scores": {
            "nwRMSE": round(weighted_nwrmse_error, 4),
            "S1_Effectiveness": round(effectiveness, 4),
            "S1_Efficiency": round(efficiency, 4),
            "S1_Overall": round(s1, 4),
            "efficiency_details": efficiency_details,
        },
        "counting": counting,
    }
    if args.verbose:
        results["effectiveness_details"] = effectiveness_details

    print(f"\n{'=' * 60}")
    print("  EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"  Video: {total_frames} frames @ {fps} FPS = {video_duration:.1f}s")
    if args.processing_time > 0:
        print(f"  Processing: {args.processing_time:.2f}s ({proc_fps:.1f} FPS)")
    print(f"{'~' * 60}")
    print(f"  weighted nwRMSE error : {weighted_nwrmse_error:.4f}")
    print(f"  S1_Effectiveness      : {effectiveness:.4f}")
    print(f"  S1_Efficiency approx  : {efficiency:.4f}")
    print(f"  S1 (Overall approx)   : {s1:.4f}")
    print(f"{'~' * 60}")
    print(f"  Total vehicles        : GT={counting['gt_total']}  Pred={counting['pred_total']}")
    print(f"  Count accuracy        : {counting['total_count_accuracy']:.1%}")
    print(f"  MAE per mov/cls       : {counting['mae_per_movement_class']:.2f}")
    print(f"  FP absent-GT keys     : {counting['false_positive_events_with_absent_gt_key']}")
    print(f"{'~' * 60}")

    print(f"\n  {'Vid':>3} {'Mov':>4} {'Class':>6} {'GT':>5} {'Pred':>5} {'Error':>6} {'Rel%':>7}")
    print(f"  {'-' * 3} {'-' * 4} {'-' * 6} {'-' * 5} {'-' * 5} {'-' * 6} {'-' * 7}")
    for item in counting["breakdown"]:
        rel_pct = f"{item['relative_error'] * 100:.1f}%"
        print(
            f"  {item['video_id']:>3} {item['movement_id']:>4} {item['class']:>6} "
            f"{item['gt_count']:>5} {item['pred_count']:>5} {item['abs_error']:>6} {rel_pct:>7}"
        )
    print(f"{'=' * 60}\n")

    if args.output_json:
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  Saved detailed results to: {args.output_json}")


if __name__ == "__main__":
    main()
