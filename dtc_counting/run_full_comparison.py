"""
run_full_comparison.py — Run all 4 ROI/MOI approaches on a video and evaluate.

Baselines:
  B1: Manual ROI + angle fallback MOI
  B2: Manual ROI + track-mined MOI  
  B3: SAM Automatic (no detector) ROI/MOI
  B4: SAM + YOLO-prompted ROI/MOI

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
import subprocess
import sys
import traceback
from typing import Dict, List


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
    p.add_argument("--sam-model", default="sam_b.pt", help="SAM model checkpoint.")
    p.add_argument("--mining-frames", type=int, default=1200, help="Frames for MOI mining.")
    p.add_argument("--output-dir", default="outputs/comparison", help="Output directory.")
    p.add_argument("--skip-sam-auto", action="store_true", help="Skip SAM automatic baseline.")
    p.add_argument("--skip-sam-yolo", action="store_true", help="Skip SAM+YOLO baseline.")
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


def write_roi_and_moi_from_json(json_path: str, roi_txt: str, moi_txt: str) -> None:
    """Extract ROI/MOI from bootstrap JSON into txt files for DTC pipeline."""
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    roi = payload.get("roi", [])
    moi_vectors = payload.get("moi_vectors", {})

    with open(roi_txt, "w", encoding="utf-8") as f:
        for p in roi:
            f.write(f"{p[0]},{p[1]}\n")

    ordered = sorted((int(k), v) for k, v in moi_vectors.items())
    with open(moi_txt, "w", encoding="utf-8") as f:
        for mid, vec in ordered:
            f.write(f"{mid},{vec[0]},{vec[1]},{vec[2]},{vec[3]}\n")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    py = sys.executable

    results: Dict[str, dict] = {}

    # ════════════════════════════════════════════════════════════════
    # BASELINE 1: Manual ROI + Angle Fallback MOI
    # ════════════════════════════════════════════════════════════════
    b1_csv = os.path.join(args.output_dir, "b1_manual_angle.csv")
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
    ], "B2: Mine MOI from tracks")

    if ok:
        b2_csv = os.path.join(args.output_dir, "b2_manual_tracked_moi.csv")
        b2_timing = os.path.join(args.output_dir, "b2_timing.json")
        b2_video = os.path.join(args.output_dir, "b2_vis.mp4")

        run_cmd([
            py, "run_dtc_counting.py",
            "--video", args.video,
            "--weights", args.weights,
            "--roi-file", args.roi_file,
            "--movement-description", args.movement_description,
            "--moi-vectors", b2_moi,
            "--video-clip-id", str(args.video_clip_id),
            "--imgsz", str(args.imgsz),
            "--conf", str(args.conf),
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
            b3_roi = os.path.join(args.output_dir, "b3_roi.txt")
            b3_moi = os.path.join(args.output_dir, "b3_moi.txt")
            write_roi_and_moi_from_json(b3_json, b3_roi, b3_moi)

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
    # BASELINE 4: SAM + YOLO-prompted
    # ════════════════════════════════════════════════════════════════
    if not args.skip_sam_yolo:
        b4_json = os.path.join(args.output_dir, "b4_sam_yolo_bootstrap.json")
        b4_overlay = os.path.join(args.output_dir, "b4_sam_yolo_overlay.jpg")

        ok = run_cmd([
            py, "sam_bootstrap.py",
            "--video", args.video,
            "--weights", args.weights,
            "--sam-model", args.sam_model,
            "--moi-count", "12",
            "--max-frames", "120",
            "--output-json", b4_json,
            "--save-overlay", b4_overlay,
        ], "B4: SAM + YOLO-prompted Bootstrap")

        if ok and os.path.exists(b4_json):
            b4_roi = os.path.join(args.output_dir, "b4_roi.txt")
            b4_moi = os.path.join(args.output_dir, "b4_moi.txt")
            write_roi_and_moi_from_json(b4_json, b4_roi, b4_moi)

            b4_csv = os.path.join(args.output_dir, "b4_sam_yolo.csv")
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
                "--output-csv", b4_csv,
                "--output-timing", b4_timing,
                "--save-video", b4_video,
            ], "B4: DTC with SAM+YOLO ROI/MOI")

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
        "b1": "Manual ROI + Angle Fallback",
        "b2": "Manual ROI + Track-Mined MOI",
        "b3": "SAM Automatic (No Detector)",
        "b4": "SAM + YOLO-prompted",
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
