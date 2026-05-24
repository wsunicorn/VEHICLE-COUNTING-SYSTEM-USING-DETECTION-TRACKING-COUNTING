"""
run_full_comparison.py — Run all 4 ROI/MOI approaches on a video and evaluate.

Baselines:
  B1: Manual ROI + official MOI vectors (or angle fallback if omitted)
  B2: Manual ROI + track-mined MOI, optionally aligned to official MOI ids
  B3: SAM Automatic bootstrap (quality checked)
  B4: Grounding DINO + SAM bootstrap (quality checked)

Outputs: per-baseline CSVs, evaluation JSONs, comparison summary.

Usage:
    python run_full_comparison.py \\
        --video cam_5_ThanhPhoBuon.mp4 \\
        --weights ../weights/best2.pt \\
        --roi-file data/AIC21_Track1_Vehicle_Counting/ROIs/cam_5.txt \\
        --movement-description data/AIC21_Track1_Vehicle_Counting/movement_description/cam_5.txt \\
        --gt-csv data/AIC21_Track1_Vehicle_Counting/counting_gt_sample/counting_example_cam_5_1min.csv \\
        --output-dir outputs/comparison
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from typing import Dict, List

from moi_utils import align_to_reference, load_moi_vectors, write_moi_vectors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full DTC comparison pipeline.")
    p.add_argument("--video", required=True, help="Video path.")
    p.add_argument("--weights", default="../weights/best2.pt", help="YOLO weights.")
    p.add_argument("--roi-file", required=True, help="Manual ROI polygon file.")
    p.add_argument("--movement-description", required=True, help="Movement description file.")
    p.add_argument("--moi-vectors", help="Manual MOI vectors file (for B1).")
    p.add_argument("--gt-csv", required=True, help="Ground truth counting CSV.")
    p.add_argument("--video-clip-id", type=int, default=10, help="video_clip_id in CSVs.")
    p.add_argument("--total-frames", type=int, default=600, help="Total video frames.")
    p.add_argument("--video-fps", type=float, default=10.0, help="Video FPS.")
    p.add_argument("--imgsz", type=int, default=1280, help="YOLO image size.")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence.")
    p.add_argument("--class-conf", default="", help="Optional per-class confidence overrides, e.g. car=0.25,truck=0.45.")
    p.add_argument("--sam-model", default="sam_b.pt", help="SAM model checkpoint.")
    p.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-base", help="Grounding DINO model id.")
    p.add_argument("--text-prompt", default="road surface . traffic lane . intersection", help="Grounding prompt.")
    p.add_argument("--mining-frames", type=int, default=1200, help="Frames for MOI mining.")
    p.add_argument("--output-dir", default="outputs/comparison", help="Output directory.")
    p.add_argument("--skip-sam-auto", action="store_true", help="Skip SAM automatic baseline.")
    p.add_argument("--skip-sam-yolo", action="store_true", help="Compatibility alias: skip Grounded-SAM baseline.")
    p.add_argument("--fallback-to-trajectory-sam", action="store_true", help="Use trajectory-guided SAM if Grounded-SAM fails.")
    p.add_argument("--allow-low-quality-bootstrap", action="store_true", help="Run DTC even when bootstrap produced full-frame ROI or no MOI.")
    p.add_argument("--min-bootstrap-moi", type=int, default=3, help="Minimum valid MOI vectors required before an automatic bootstrap is trusted.")
    return p.parse_args()


def run_cmd(cmd: List[str], label: str = "") -> bool:
    """Run a command and return True if successful."""
    print(f"\n{'='*60}")
    print(f"  Running: {label or ' '.join(cmd[:3])}")
    print(f"{'='*60}")
    print(f"  CMD: {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"  FAILED with exit code {exc.returncode}")
        traceback.print_exc()
        return False


def _valid_moi_payload(moi_vectors: dict) -> Dict[int, tuple]:
    valid: Dict[int, tuple] = {}
    for key, vec in moi_vectors.items():
        if len(vec) != 4:
            continue
        mid = int(key)
        x1, y1, x2, y2 = [float(v) for v in vec]
        if ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 < 1e-6:
            continue
        valid[mid] = ((x1, y1), (x2, y2))
    return valid


def bootstrap_quality(json_path: str, frame_w: int = 0, frame_h: int = 0, min_moi_count: int = 1) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    quality = payload.get("quality", {})
    roi = payload.get("roi", [])
    moi_vectors = _valid_moi_payload(payload.get("moi_vectors", {}))

    full_frame = bool(quality.get("is_full_frame_fallback", False))
    if frame_w > 0 and frame_h > 0 and len(roi) >= 3:
        xs = [float(p[0]) for p in roi]
        ys = [float(p[1]) for p in roi]
        width_ratio = (max(xs) - min(xs)) / max(1.0, float(frame_w - 1))
        height_ratio = (max(ys) - min(ys)) / max(1.0, float(frame_h - 1))
        full_frame = full_frame or (width_ratio > 0.98 and height_ratio > 0.98)

    status = quality.get("status", "ok")
    too_few_moi = len(moi_vectors) < max(1, min_moi_count)
    low_quality = full_frame or too_few_moi or status == "low_confidence"
    return {
        "status": "low_confidence" if low_quality else "ok",
        "is_full_frame_fallback": full_frame,
        "valid_moi_count": len(moi_vectors),
        "min_moi_count": max(1, min_moi_count),
        "too_few_moi": too_few_moi,
        "note": quality.get("reason", ""),
    }


def write_roi_and_moi_from_json(json_path: str, roi_txt: str, moi_txt: str, reference_moi_path: str = "") -> None:
    """Extract ROI/MOI from bootstrap JSON into txt files for DTC pipeline."""
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    roi = payload.get("roi", [])
    moi_vectors = _valid_moi_payload(payload.get("moi_vectors", {}))

    with open(roi_txt, "w", encoding="utf-8") as f:
        for p in roi:
            f.write(f"{p[0]},{p[1]}\n")

    if reference_moi_path:
        reference = load_moi_vectors(reference_moi_path)
        aligned = align_to_reference(moi_vectors, reference)
        if aligned:
            moi_vectors = aligned

    write_moi_vectors(moi_txt, moi_vectors)


def video_size(path: str) -> tuple:
    try:
        import cv2
    except Exception:
        return 0, 0
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0, 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return w, h


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    py = sys.executable
    frame_w, frame_h = video_size(args.video)

    results: Dict[str, dict] = {}

    # ════════════════════════════════════════════════════════════════
    # BASELINE 1: Manual ROI + Angle Fallback MOI
    # ════════════════════════════════════════════════════════════════
    b1_csv = os.path.join(args.output_dir, "b1_manual_official_moi.csv" if args.moi_vectors else "b1_manual_angle.csv")
    b1_timing = os.path.join(args.output_dir, "b1_timing.json")
    b1_video = os.path.join(args.output_dir, "b1_vis.mp4")

    cmd_b1 = [
        py, "run_dtc_counting.py",
        "--video", args.video,
        "--weights", args.weights,
        "--roi-file", args.roi_file,
        "--movement-description", args.movement_description,
        "--video-clip-id", str(args.video_clip_id),
        "--imgsz", str(args.imgsz),
        "--conf", str(args.conf),
        "--class-conf", args.class_conf,
        "--output-csv", b1_csv,
        "--output-timing", b1_timing,
        "--save-video", b1_video,
    ]
    if args.moi_vectors:
        cmd_b1.extend(["--moi-vectors", args.moi_vectors])

    ok = run_cmd(cmd_b1, "B1: Manual ROI + Given MOI Vectors")

    if ok:
        proc_time = 0.0
        if os.path.exists(b1_timing):
            with open(b1_timing) as f:
                proc_time = json.load(f).get("elapsed_seconds", 0)

        run_cmd([
            py, "evaluate_counting.py",
            "--gt-csv", args.gt_csv,
            "--pred-csv", b1_csv,
            "--total-frames", str(args.total_frames),
            "--video-fps", str(args.video_fps),
            "--processing-time", str(proc_time),
            "--output-json", os.path.join(args.output_dir, "b1_eval.json"),
        ], "Evaluate B1")

    # ════════════════════════════════════════════════════════════════
    # BASELINE 2: Manual ROI + Track-Mined MOI
    # ════════════════════════════════════════════════════════════════
    tracked_moi_for_counting = ""
    b2_moi = os.path.join(args.output_dir, "b2_moi_from_tracks.txt")

    ok = run_cmd([
        py, "build_moi_from_tracks.py",
        "--video", args.video,
        "--weights", args.weights,
        "--roi-file", args.roi_file,
        "--movement-description", args.movement_description,
        "--output-moi", b2_moi,
        "--max-frames", str(args.mining_frames),
        "--imgsz", str(args.imgsz),
        "--conf", str(args.conf),
        "--class-conf", args.class_conf,
    ], "B2: Mine MOI from tracks")

    if ok:
        b2_moi_for_counting = b2_moi
        if args.moi_vectors:
            generated = load_moi_vectors(b2_moi)
            reference = load_moi_vectors(args.moi_vectors)
            aligned = align_to_reference(generated, reference)
            if aligned:
                b2_moi_for_counting = os.path.join(args.output_dir, "b2_moi_from_tracks_aligned.txt")
                write_moi_vectors(b2_moi_for_counting, aligned)
                print(f"  Aligned B2 MOI ids to reference: {b2_moi_for_counting}")
        tracked_moi_for_counting = b2_moi_for_counting

        b2_csv = os.path.join(args.output_dir, "b2_manual_tracked_moi.csv")
        b2_timing = os.path.join(args.output_dir, "b2_timing.json")
        b2_video = os.path.join(args.output_dir, "b2_vis.mp4")

        run_cmd([
            py, "run_dtc_counting.py",
            "--video", args.video,
            "--weights", args.weights,
            "--roi-file", args.roi_file,
            "--movement-description", args.movement_description,
            "--moi-vectors", b2_moi_for_counting,
            "--video-clip-id", str(args.video_clip_id),
            "--imgsz", str(args.imgsz),
            "--conf", str(args.conf),
            "--class-conf", args.class_conf,
            "--output-csv", b2_csv,
            "--output-timing", b2_timing,
            "--save-video", b2_video,
        ], "B2: DTC with tracked MOI")

        proc_time = 0.0
        if os.path.exists(b2_timing):
            with open(b2_timing) as f:
                proc_time = json.load(f).get("elapsed_seconds", 0)

        run_cmd([
            py, "evaluate_counting.py",
            "--gt-csv", args.gt_csv,
            "--pred-csv", b2_csv,
            "--total-frames", str(args.total_frames),
            "--video-fps", str(args.video_fps),
            "--processing-time", str(proc_time),
            "--output-json", os.path.join(args.output_dir, "b2_eval.json"),
        ], "Evaluate B2")

    # ════════════════════════════════════════════════════════════════
    # BASELINE 3: SAM Automatic (No Detector)
    # ════════════════════════════════════════════════════════════════
    if not args.skip_sam_auto:
        b3_json = os.path.join(args.output_dir, "b3_sam_auto_bootstrap.json")
        b3_overlay = os.path.join(args.output_dir, "b3_sam_auto_overlay.jpg")

        ok = run_cmd([
            py, "sam_auto_bootstrap.py",
            "--video", args.video,
            "--sam-model", args.sam_model,
            "--moi-count", "12",
            "--output-json", b3_json,
            "--save-overlay", b3_overlay,
        ], "B3: SAM Automatic Bootstrap")

        if ok and os.path.exists(b3_json):
            q = bootstrap_quality(b3_json, frame_w, frame_h, args.min_bootstrap_moi)
            if q["status"] != "ok" and not args.allow_low_quality_bootstrap:
                print(
                    "  Skipping B3 DTC because SAM Automatic bootstrap is low-confidence "
                    f"(full_frame={q['is_full_frame_fallback']}, "
                    f"moi={q['valid_moi_count']}/{q['min_moi_count']})."
                )
                ok = False

        if ok and os.path.exists(b3_json):
            b3_roi = os.path.join(args.output_dir, "b3_roi.txt")
            b3_moi = os.path.join(args.output_dir, "b3_moi.txt")
            write_roi_and_moi_from_json(b3_json, b3_roi, b3_moi, args.moi_vectors or "")

            b3_csv = os.path.join(args.output_dir, "b3_sam_auto.csv")
            b3_timing = os.path.join(args.output_dir, "b3_timing.json")
            b3_video = os.path.join(args.output_dir, "b3_vis.mp4")

            run_cmd([
                py, "run_dtc_counting.py",
                "--video", args.video,
                "--weights", args.weights,
                "--roi-file", b3_roi,
                "--movement-description", args.movement_description,
                "--moi-vectors", b3_moi,
                "--video-clip-id", str(args.video_clip_id),
                "--imgsz", str(args.imgsz),
                "--conf", str(args.conf),
                "--class-conf", args.class_conf,
                "--output-csv", b3_csv,
                "--output-timing", b3_timing,
                "--save-video", b3_video,
            ], "B3: DTC with SAM Auto ROI/MOI")

            proc_time = 0.0
            if os.path.exists(b3_timing):
                with open(b3_timing) as f:
                    proc_time = json.load(f).get("elapsed_seconds", 0)

            run_cmd([
                py, "evaluate_counting.py",
                "--gt-csv", args.gt_csv,
                "--pred-csv", b3_csv,
                "--total-frames", str(args.total_frames),
                "--video-fps", str(args.video_fps),
                "--processing-time", str(proc_time),
                "--output-json", os.path.join(args.output_dir, "b3_eval.json"),
            ], "Evaluate B3")

    # ════════════════════════════════════════════════════════════════
    # BASELINE 4: Grounding DINO + SAM
    # ════════════════════════════════════════════════════════════════
    if not args.skip_sam_yolo:
        b4_moi_override = ""
        b4_summary_label = "Grounding DINO + SAM Bootstrap"
        b4_json = os.path.join(args.output_dir, "b4_grounded_sam_bootstrap.json")
        b4_overlay = os.path.join(args.output_dir, "b4_grounded_sam_overlay.jpg")

        ok = run_cmd([
            py, "grounded_sam_bootstrap.py",
            "--video", args.video,
            "--sam-model", args.sam_model,
            "--grounding-model", args.grounding_model,
            "--text-prompt", args.text_prompt,
            "--moi-count", "12",
            "--output-json", b4_json,
            "--save-overlay", b4_overlay,
        ], "B4: Grounding DINO + SAM Bootstrap")

        if not ok and args.fallback_to_trajectory_sam:
            print("  Grounded-SAM failed. Falling back to trajectory-guided SAM bootstrap.")
            b4_json = os.path.join(args.output_dir, "b4_trajectory_sam_bootstrap.json")
            b4_overlay = os.path.join(args.output_dir, "b4_trajectory_sam_overlay.jpg")
            ok = run_cmd([
                py, "sam_bootstrap.py",
                "--video", args.video,
                "--weights", args.weights,
                "--sam-model", args.sam_model,
                "--moi-count", "12",
                "--max-frames", str(min(args.mining_frames, 300)),
                "--output-json", b4_json,
                "--save-overlay", b4_overlay,
            ], "B4 fallback: Trajectory-guided SAM Bootstrap")

        if ok and os.path.exists(b4_json):
            q = bootstrap_quality(b4_json, frame_w, frame_h, args.min_bootstrap_moi)
            if q["status"] != "ok" and not args.allow_low_quality_bootstrap:
                if q["too_few_moi"] and not q["is_full_frame_fallback"] and tracked_moi_for_counting:
                    b4_moi_override = tracked_moi_for_counting
                    b4_summary_label = "Grounding DINO + SAM ROI + Track-Mined MOI"
                    print(
                        "  B4 bootstrap ROI is usable but MOI is too sparse "
                        f"({q['valid_moi_count']}/{q['min_moi_count']}). "
                        f"Using tracked-MOI fallback: {b4_moi_override}"
                    )
                    with open(os.path.join(args.output_dir, "b4_bootstrap_decision.json"), "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "decision": "use_tracked_moi_fallback",
                                "reason": "automatic bootstrap produced too few valid MOI vectors",
                                "bootstrap_quality": q,
                                "moi_fallback_path": b4_moi_override,
                            },
                            f,
                            indent=2,
                        )
                else:
                    print(
                        "  Skipping B4 DTC because bootstrap is low-confidence "
                        f"(full_frame={q['is_full_frame_fallback']}, "
                        f"moi={q['valid_moi_count']}/{q['min_moi_count']})."
                    )
                    ok = False

        if ok and os.path.exists(b4_json):
            b4_roi = os.path.join(args.output_dir, "b4_roi.txt")
            b4_moi = os.path.join(args.output_dir, "b4_moi.txt")
            write_roi_and_moi_from_json(b4_json, b4_roi, b4_moi, args.moi_vectors or "")
            if b4_moi_override:
                shutil.copyfile(b4_moi_override, b4_moi)

            b4_csv = os.path.join(args.output_dir, "b4_grounded_sam.csv")
            b4_timing = os.path.join(args.output_dir, "b4_timing.json")
            b4_video = os.path.join(args.output_dir, "b4_vis.mp4")

            run_cmd([
                py, "run_dtc_counting.py",
                "--video", args.video,
                "--weights", args.weights,
                "--roi-file", b4_roi,
                "--movement-description", args.movement_description,
                "--moi-vectors", b4_moi,
                "--video-clip-id", str(args.video_clip_id),
                "--imgsz", str(args.imgsz),
                "--conf", str(args.conf),
                "--class-conf", args.class_conf,
                "--output-csv", b4_csv,
                "--output-timing", b4_timing,
                "--save-video", b4_video,
            ], "B4: DTC with Grounded-SAM ROI/MOI")

            proc_time = 0.0
            if os.path.exists(b4_timing):
                with open(b4_timing) as f:
                    proc_time = json.load(f).get("elapsed_seconds", 0)

            run_cmd([
                py, "evaluate_counting.py",
                "--gt-csv", args.gt_csv,
                "--pred-csv", b4_csv,
                "--total-frames", str(args.total_frames),
                "--video-fps", str(args.video_fps),
                "--processing-time", str(proc_time),
                "--output-json", os.path.join(args.output_dir, "b4_eval.json"),
            ], "Evaluate B4")

    # ════════════════════════════════════════════════════════════════
    # AGGREGATE COMPARISON SUMMARY
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")

    summary = {}
    labels = {
        "b1": "Manual ROI + Official MOI" if args.moi_vectors else "Manual ROI + Angle Fallback",
        "b2": "Manual ROI + Track-Mined MOI (aligned)" if args.moi_vectors else "Manual ROI + Track-Mined MOI",
        "b3": "SAM Automatic Bootstrap",
        "b4": locals().get("b4_summary_label", "Grounding DINO + SAM Bootstrap"),
    }

    for key in ["b1", "b2", "b3", "b4"]:
        eval_file = os.path.join(args.output_dir, f"{key}_eval.json")
        if os.path.exists(eval_file):
            with open(eval_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            scores = data.get("scores", {})
            counting = data.get("counting", {})
            summary[key] = {
                "label": labels[key],
                "nwRMSE": scores.get("nwRMSE", -1),
                "S1_Effectiveness": scores.get("S1_Effectiveness", -1),
                "S1_Efficiency": scores.get("S1_Efficiency", -1),
                "S1_Overall": scores.get("S1_Overall", -1),
                "gt_total": counting.get("gt_total", 0),
                "pred_total": counting.get("pred_total", 0),
                "count_accuracy": counting.get("total_count_accuracy", 0),
                "mae": counting.get("mae_per_movement_class", -1),
            }
            print(f"\n  [{key.upper()}] {labels[key]}")
            print(f"    nwRMSE={scores.get('nwRMSE','?')}, "
                  f"S1={scores.get('S1_Overall','?')}, "
                  f"Count={counting.get('pred_total','?')}/{counting.get('gt_total','?')}, "
                  f"Accuracy={counting.get('total_count_accuracy','?')}")

    summary_path = os.path.join(args.output_dir, "comparison_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n  Saved: {summary_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
