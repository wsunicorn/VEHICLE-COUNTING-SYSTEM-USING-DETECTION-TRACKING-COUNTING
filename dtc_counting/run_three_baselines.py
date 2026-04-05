import argparse
import csv
import json
import os
import subprocess
import sys
import traceback
from collections import Counter
from typing import Dict, Iterable, List, Tuple


Event = Tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 3 DTC baselines on existing AI City assets.")
    parser.add_argument(
        "--video",
        default="data/AIC21_Track1_Vehicle_Counting_Bosung/counting_gt_sample/counting_example_cam_5_1min.mp4",
        help="Video path.",
    )
    parser.add_argument("--weights", default="../best2.pt", help="YOLO weights path.")
    parser.add_argument(
        "--roi-file",
        default="data/AIC21_Track1_Vehicle_Counting_Bosung/ROIs/cam_5.txt",
        help="ROI file path.",
    )
    parser.add_argument(
        "--movement-description",
        default="data/AIC21_Track1_Vehicle_Counting_Bosung/movement_description/cam_5.txt",
        help="movement_description file path.",
    )
    parser.add_argument(
        "--gt-csv",
        default="data/AIC21_Track1_Vehicle_Counting_Bosung/counting_gt_sample/counting_example_cam_5_1min.csv",
        help="Ground truth csv for quick comparison.",
    )
    parser.add_argument("--video-clip-id", type=int, default=10, help="video_clip_id value in predictions.")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument(
        "--mining-frames",
        type=int,
        default=1200,
        help="Frame budget for MOI mining in baseline 2.",
    )
    parser.add_argument("--sam-model", default="sam_b.pt", help="SAM model name/checkpoint.")
    parser.add_argument(
        "--baseline3-mode",
        choices=["sam", "grounded-sam"],
        default="grounded-sam",
        help="Baseline 3 bootstrap mode.",
    )
    parser.add_argument(
        "--grounding-model",
        default="IDEA-Research/grounding-dino-base",
        help="Grounding DINO model id (used when baseline3-mode=grounded-sam).",
    )
    parser.add_argument(
        "--text-prompt",
        default="road surface . traffic lane . intersection",
        help="Grounding prompt (used when baseline3-mode=grounded-sam).",
    )
    parser.add_argument(
        "--fallback-to-sam",
        action="store_true",
        help="Fallback to sam_bootstrap.py if grounded_sam_bootstrap.py fails.",
    )
    parser.add_argument("--output-dir", default="outputs/baselines", help="Output directory.")
    parser.add_argument(
        "--skip-baseline3",
        action="store_true",
        help="Skip baseline 3 if SAM is not available.",
    )
    return parser.parse_args()


def run_cmd(cmd: List[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def read_events(path: str) -> List[Event]:
    events: List[Event] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(
                (
                    int(row["frame_id"]),
                    int(row["movement_id"]),
                    int(row["vehicle_class_id"]),
                )
            )
    return events


def summarize_counts(events: Iterable[Event]) -> Counter:
    counter: Counter = Counter()
    for _, movement_id, cls_id in events:
        counter[(movement_id, cls_id)] += 1
    return counter


def evaluate(gt_events: List[Event], pred_events: List[Event]) -> Dict[str, float]:
    gt_counter = Counter(gt_events)
    pred_counter = Counter(pred_events)

    tp = 0
    for key in set(gt_counter.keys()) | set(pred_counter.keys()):
        tp += min(gt_counter.get(key, 0), pred_counter.get(key, 0))

    gt_total = len(gt_events)
    pred_total = len(pred_events)
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gt_total if gt_total else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    gt_by_moi = summarize_counts(gt_events)
    pred_by_moi = summarize_counts(pred_events)
    keys = sorted(set(gt_by_moi.keys()) | set(pred_by_moi.keys()))
    abs_errors = [abs(pred_by_moi.get(k, 0) - gt_by_moi.get(k, 0)) for k in keys]
    mae_by_pair = (sum(abs_errors) / len(abs_errors)) if abs_errors else 0.0

    return {
        "gt_total": float(gt_total),
        "pred_total": float(pred_total),
        "tp_exact": float(tp),
        "precision_exact": precision,
        "recall_exact": recall,
        "f1_exact": f1,
        "mae_per_movement_class": mae_by_pair,
    }


def write_roi_and_moi_from_json(json_path: str, roi_txt: str, moi_txt: str) -> None:
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
    gt_events = read_events(args.gt_csv)

    b1_csv = os.path.join(args.output_dir, "baseline1_angle_fallback.csv")
    run_cmd(
        [
            py,
            "run_dtc_counting.py",
            "--video",
            args.video,
            "--weights",
            args.weights,
            "--roi-file",
            args.roi_file,
            "--movement-description",
            args.movement_description,
            "--video-clip-id",
            str(args.video_clip_id),
            "--imgsz",
            str(args.imgsz),
            "--conf",
            str(args.conf),
            "--output-csv",
            b1_csv,
        ]
    )

    b2_moi = os.path.join(args.output_dir, "baseline2_moi_from_tracks.txt")
    run_cmd(
        [
            py,
            "build_moi_from_tracks.py",
            "--video",
            args.video,
            "--weights",
            args.weights,
            "--roi-file",
            args.roi_file,
            "--movement-description",
            args.movement_description,
            "--output-moi",
            b2_moi,
            "--max-frames",
            str(args.mining_frames),
            "--imgsz",
            str(args.imgsz),
            "--conf",
            str(args.conf),
        ]
    )

    b2_csv = os.path.join(args.output_dir, "baseline2_track_moi.csv")
    run_cmd(
        [
            py,
            "run_dtc_counting.py",
            "--video",
            args.video,
            "--weights",
            args.weights,
            "--roi-file",
            args.roi_file,
            "--movement-description",
            args.movement_description,
            "--moi-vectors",
            b2_moi,
            "--video-clip-id",
            str(args.video_clip_id),
            "--imgsz",
            str(args.imgsz),
            "--conf",
            str(args.conf),
            "--output-csv",
            b2_csv,
        ]
    )

    results: Dict[str, Dict[str, float]] = {}
    results["baseline1_angle_fallback"] = evaluate(gt_events, read_events(b1_csv))
    results["baseline2_track_moi"] = evaluate(gt_events, read_events(b2_csv))

    if not args.skip_baseline3:
        b3_json = os.path.join(args.output_dir, "baseline3_sam_bootstrap.json")
        b3_overlay = os.path.join(args.output_dir, "baseline3_sam_overlay.jpg")
        if args.baseline3_mode == "grounded-sam":
            grounded_cmd = [
                py,
                "grounded_sam_bootstrap.py",
                "--video",
                args.video,
                "--sam-model",
                args.sam_model,
                "--grounding-model",
                args.grounding_model,
                "--text-prompt",
                args.text_prompt,
                "--moi-count",
                "12",
                "--output-json",
                b3_json,
                "--save-overlay",
                b3_overlay,
            ]
            try:
                run_cmd(grounded_cmd)
            except subprocess.CalledProcessError as exc:
                print(f"Grounded SAM failed with exit code: {exc.returncode}")
                traceback.print_exc()
                if not args.fallback_to_sam:
                    raise
                print("Grounded SAM failed, fallback to sam_bootstrap.py")
                run_cmd(
                    [
                        py,
                        "sam_bootstrap.py",
                        "--video",
                        args.video,
                        "--weights",
                        args.weights,
                        "--sam-model",
                        args.sam_model,
                        "--moi-count",
                        "12",
                        "--output-json",
                        b3_json,
                        "--save-overlay",
                        b3_overlay,
                    ]
                )
        else:
            run_cmd(
                [
                    py,
                    "sam_bootstrap.py",
                    "--video",
                    args.video,
                    "--weights",
                    args.weights,
                    "--sam-model",
                    args.sam_model,
                    "--moi-count",
                    "12",
                    "--output-json",
                    b3_json,
                    "--save-overlay",
                    b3_overlay,
                ]
            )

        b3_roi = os.path.join(args.output_dir, "baseline3_roi_from_sam.txt")
        b3_moi = os.path.join(args.output_dir, "baseline3_moi_from_sam.txt")
        write_roi_and_moi_from_json(b3_json, b3_roi, b3_moi)

        b3_csv = os.path.join(args.output_dir, "baseline3_grounded_sam_moi.csv")
        run_cmd(
            [
                py,
                "run_dtc_counting.py",
                "--video",
                args.video,
                "--weights",
                args.weights,
                "--roi-file",
                b3_roi,
                "--movement-description",
                args.movement_description,
                "--moi-vectors",
                b3_moi,
                "--video-clip-id",
                str(args.video_clip_id),
                "--imgsz",
                str(args.imgsz),
                "--conf",
                str(args.conf),
                "--output-csv",
                b3_csv,
            ]
        )

        results["baseline3_grounded_sam_moi"] = evaluate(gt_events, read_events(b3_csv))

    summary_path = os.path.join(args.output_dir, "baseline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=True, indent=2)

    print("Done. Baseline summary:")
    print(json.dumps(results, ensure_ascii=True, indent=2))
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
