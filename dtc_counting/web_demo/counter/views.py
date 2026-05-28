import csv
import json
import os
import re
import subprocess
import sys
import shlex
import threading
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render

from .forms import DemoForm, ManualForm, AutoForm

DTC_ROOT = Path(__file__).resolve().parents[2]
if str(DTC_ROOT) not in sys.path:
    sys.path.insert(0, str(DTC_ROOT))

from moi_utils import align_to_reference, load_moi_vectors, write_moi_vectors


# ── Thread-safe run registry ───────────────────────────────────────────────────
_RUN_REGISTRY: Dict[str, dict] = {}
_REGISTRY_LOCK = threading.Lock()


def _reg_init(run_id: str, steps: list, mode: str = "manual") -> None:
    with _REGISTRY_LOCK:
        _RUN_REGISTRY[run_id] = {
            "status": "running",
            "mode": mode,
            "steps": steps,
            "logs": [],
            "progress_pct": 0,
            "error": None,
            "result_ctx": None,
        }


def _reg_step(run_id: str, key: str, state: str) -> None:
    with _REGISTRY_LOCK:
        entry = _RUN_REGISTRY.get(run_id)
        if not entry:
            return
        for s in entry["steps"]:
            if s["key"] == key:
                s["state"] = state
                break


def _reg_log(run_id: str, line: str) -> None:
    with _REGISTRY_LOCK:
        entry = _RUN_REGISTRY.get(run_id)
        if not entry:
            return
        entry["logs"].append(line)
        if len(entry["logs"]) > 500:
            entry["logs"] = entry["logs"][-300:]


def _reg_pct(run_id: str, pct: float) -> None:
    with _REGISTRY_LOCK:
        entry = _RUN_REGISTRY.get(run_id)
        if entry:
            entry["progress_pct"] = pct


def _reg_done(run_id: str, ctx: dict) -> None:
    with _REGISTRY_LOCK:
        entry = _RUN_REGISTRY.get(run_id)
        if entry:
            entry["status"] = "done"
            entry["result_ctx"] = ctx
            entry["progress_pct"] = 100


def _reg_error(run_id: str, err: str) -> None:
    with _REGISTRY_LOCK:
        entry = _RUN_REGISTRY.get(run_id)
        if entry:
            entry["status"] = "error"
            entry["error"] = err


# ── Helpers ────────────────────────────────────────────────────────────────────

def _cmd_to_text(cmd: List[str]) -> str:
    return " ".join(shlex.quote(str(c)) for c in cmd)


def _save_upload(uploaded_file, target_path: Path) -> str:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "wb+") as f:
        for chunk in uploaded_file.chunks():
            f.write(chunk)
    return str(target_path)


def _load_summary(csv_path: str) -> Dict[str, int]:
    counter: Dict[str, int] = defaultdict(int)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movement_id = int(row["movement_id"])
            cls_id = int(row["vehicle_class_id"])
            key = f"m{movement_id}_c{cls_id}"
            counter[key] += 1
    return dict(sorted(counter.items(), key=lambda kv: kv[0]))


def _get_video_size(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return (1, 1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1)
    cap.release()
    return (max(1, w), max(1, h))


def _polygon_quality(roi_points, frame_w: int, frame_h: int) -> Dict[str, float]:
    if not roi_points or len(roi_points) < 3:
        return {"ok": 0.0, "area_ratio": 0.0, "width_ratio": 0.0, "height_ratio": 0.0}
    pts = []
    for p in roi_points:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            pts.append((float(p[0]), float(p[1])))
    if len(pts) < 3:
        return {"ok": 0.0, "area_ratio": 0.0, "width_ratio": 0.0, "height_ratio": 0.0}
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    area = abs(area) * 0.5
    frame_area = float(max(1, frame_w * frame_h))
    area_ratio = area / frame_area
    width_ratio = w / float(max(1, frame_w))
    height_ratio = h / float(max(1, frame_h))
    ok = 1.0 if (area_ratio >= 0.10 and width_ratio >= 0.35 and height_ratio >= 0.20) else 0.0
    return {"ok": ok, "area_ratio": area_ratio, "width_ratio": width_ratio, "height_ratio": height_ratio}


def _transcode_to_browser_mp4(src_path: Path, dst_path: Path) -> bool:
    try:
        import imageio_ffmpeg  # type: ignore
    except Exception:
        return False
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-y", "-i", str(src_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(dst_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return False
    return dst_path.exists() and dst_path.stat().st_size > 0


def _popen_log(run_id: str, cmd: list, cwd: str) -> None:
    """Run cmd, stream its stdout to registry log, parse frame progress. Raises on non-zero exit."""
    proc = subprocess.Popen(
        cmd, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        _reg_log(run_id, line)
        # Parse "[progress] frame=N/TOTAL (PCT%) …"
        m = re.search(r"\((\d+(?:\.\d+)?)%\)", line)
        if m:
            try:
                _reg_pct(run_id, float(m.group(1)))
            except ValueError:
                pass
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def _infer_movement_count_from_file(path: str) -> int:
    """Parse movement_description file and return the maximum movement ID found."""
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


def _build_track_mined_moi(
    run_id: str,
    *,
    py: str,
    dtc_dir: str,
    out_dir: Path,
    video_path: str,
    weights_path: str,
    roi_file: str,
    movement_path: str,
    movement_count_hint: int,
    reference_moi_path: str,
    data: dict,
    label: str,
) -> str:
    """Build MOI from tracked trajectories and align IDs to reference MOI when available."""
    if not roi_file or not os.path.exists(roi_file):
        _reg_log(run_id, "  Bỏ qua track-mined MOI: chưa có ROI hợp lệ.")
        return ""
    if not movement_path or not os.path.exists(movement_path):
        inferred_count = max(1, int(movement_count_hint or 8))
        auto_movement = out_dir / "movement_auto_hint.txt"
        with open(auto_movement, "w", encoding="utf-8") as f:
            for mid in range(1, inferred_count + 1):
                f.write(f"movement {mid}: auto-generated placeholder for MOI mining\n")
        movement_path = str(auto_movement)
        _reg_log(
            run_id,
            f"  Không có movement description; tạo hint tạm {inferred_count} movement để mine MOI.",
        )

    max_frames = int(data.get("max_frames") or 1200)
    mining_frames = max(120, min(max_frames, 1200))
    raw_moi = out_dir / "moi_from_tracks.txt"
    aligned_moi = out_dir / "moi_from_tracks_aligned.txt"

    _reg_log(
        run_id,
        f"→ Sinh MOI từ trajectory cho {label} ({mining_frames} frame, align về MOI chính thức nếu có)…",
    )
    cmd = [
        py,
        "build_moi_from_tracks.py",
        "--video",
        video_path,
        "--weights",
        weights_path,
        "--roi-file",
        roi_file,
        "--movement-description",
        movement_path,
        "--output-moi",
        str(raw_moi),
        "--max-frames",
        str(mining_frames),
        "--imgsz",
        str(data["imgsz"]),
        "--conf",
        str(data["conf"]),
        "--class-conf",
        data.get("class_conf", ""),
    ]

    try:
        _popen_log(run_id, cmd, dtc_dir)
    except Exception as exc:
        _reg_log(run_id, f"  Track-mined MOI lỗi: {exc}")
        return ""

    selected_moi = raw_moi
    aligned_count = 0
    if reference_moi_path and os.path.exists(reference_moi_path):
        try:
            generated = load_moi_vectors(str(raw_moi))
            reference = load_moi_vectors(reference_moi_path)
            aligned = align_to_reference(generated, reference)
            if aligned:
                write_moi_vectors(str(aligned_moi), aligned)
                selected_moi = aligned_moi
                aligned_count = len(aligned)
                _reg_log(run_id, f"  Đã align MOI theo ID chính thức: {aligned_moi.name} ({aligned_count} vectors).")
        except Exception as exc:
            _reg_log(run_id, f"  Không align được MOI theo reference, dùng MOI thô: {exc}")
    else:
        _reg_log(run_id, "  Không có reference MOI chính thức, dùng MOI từ trajectory theo thứ tự cụm.")

    decision = {
        "method": "track_mined_moi_aligned",
        "label": label,
        "mining_frames": mining_frames,
        "raw_moi_path": str(raw_moi),
        "selected_moi_path": str(selected_moi),
        "reference_moi_path": reference_moi_path or "",
        "aligned": selected_moi == aligned_moi,
        "aligned_count": aligned_count,
    }
    with open(out_dir / "moi_mining_decision.json", "w", encoding="utf-8") as f:
        json.dump(decision, f, ensure_ascii=False, indent=2)

    return str(selected_moi)


def _demo_paths(root_dir: Path) -> Dict[str, Path]:
    data_root = root_dir / "data" / "AIC21_Track1_Vehicle_Counting"
    return {
        "video": data_root / "counting_gt_sample" / "counting_example_cam_5_1min.mp4",
        "weights": root_dir.parent / "weights" / "best2.pt",
        "roi": data_root / "ROIs" / "cam_5.txt",
        "moi": data_root / "MOI_vectors" / "cam_5.txt",
        "movement": data_root / "movement_description" / "cam_5.txt",
    }


# ── Background execution thread ────────────────────────────────────────────────

def _execute_run(
    run_id: str,
    data: dict,
    out_dir: Path,
    video_path: str,
    weights_path: str,
    movement_path: str,
    roi_file: str,
    moi_vectors_path: str,
    dtc_dir: str,
    py: str,
    raw_video_url: str,
    moi_count_hint: int = 0,
    reference_moi_path: str = "",
) -> None:
    try:
        _reg_log(run_id, "✓ File đã lưu thành công.")
        _reg_step(run_id, "upload", "done")

        bootstrap_warning: Optional[str] = None
        overlay_url: Optional[str] = None

        # Determine movement count (used for --moi-count in bootstrap)
        if moi_count_hint > 0:
            moi_count = moi_count_hint
            _reg_log(run_id, f"  Số MOI vectors: {moi_count}")
        elif movement_path:
            moi_count = _infer_movement_count_from_file(movement_path)
            _reg_log(run_id, f"  Số movement từ description: {moi_count}")
        else:
            moi_count = 0  # let each bootstrap script use its own default
            _reg_log(run_id, "  Số MOI: Bootstrap sẽ tự xác định.")

        # ── Bootstrap ROI/MOI ──────────────────────────────────────────────────
        if data["auto_bootstrap"]:
            _reg_step(run_id, "bootstrap", "running")
            _reg_log(run_id, "→ Bắt đầu bootstrap ROI và sinh MOI từ trajectory…")

            b3_json = out_dir / "bootstrap_grounded_sam.json"
            b3_overlay = out_dir / "bootstrap_overlay.jpg"
            bootstrap_frames = min(int(data.get("max_frames") or 1200), 600)

            grounded_sam_ok = False
            use_grounding = data.get("use_grounding", True)
            
            if not use_grounding:
                _reg_log(run_id, "  Chế độ SAM Only (Automatic): Bỏ qua Grounding DINO...")
                cmd_boot = [
                    py, "sam_auto_bootstrap.py",
                    "--video", video_path,
                    "--output-json", str(b3_json),
                    "--save-overlay", str(b3_overlay),
                ]
                if moi_count > 0:
                    cmd_boot += ["--moi-count", str(moi_count)]
                try:
                    _popen_log(run_id, cmd_boot, dtc_dir)
                    grounded_sam_ok = True
                except Exception as e:
                    _reg_log(run_id, f"  SAM Automatic lỗi: {e}")
                    raise e
            else:
                cmd_boot = [
                    py, "grounded_sam_bootstrap.py",
                    "--video", video_path,
                    "--output-json", str(b3_json),
                    "--save-overlay", str(b3_overlay),
                    "--grounding-model", data["grounding_model"] or "IDEA-Research/grounding-dino-base",
                    "--text-prompt", data["text_prompt"] or "road surface . traffic lane . intersection",
                    "--roi-expand", "1.03",
                ]
                if moi_count > 0:
                    cmd_boot += ["--moi-count", str(moi_count)]
                try:
                    _reg_log(run_id, "  Thử Grounded-SAM…")
                    _popen_log(run_id, cmd_boot, dtc_dir)
                    grounded_sam_ok = True
                except Exception as e:
                    _reg_log(run_id, f"  Grounded-SAM lỗi ({e}) → fallback sang SAM bootstrap…")
                    cmd_sam_boot = [
                        py, "sam_bootstrap.py",
                        "--video", video_path,
                        "--weights", weights_path,
                        "--output-json", str(b3_json),
                        "--save-overlay", str(b3_overlay),
                        "--roi-expand", "1.03",
                        "--roi-shape", "rect",
                        "--imgsz", str(data["imgsz"]),
                        "--conf", str(data["conf"]),
                        "--max-frames", str(bootstrap_frames),
                    ]
                    if moi_count > 0:
                        cmd_sam_boot += ["--moi-count", str(moi_count)]
                    _popen_log(run_id, cmd_sam_boot, dtc_dir)
                    bootstrap_warning = (
                        "Grounded-SAM gặp lỗi tải model/kết nối, đã tự động fallback sang SAM bootstrap. "
                        f"Lệnh lỗi: {_cmd_to_text(cmd_boot)}"
                    )

            # Validate ROI quality
            with open(b3_json, "r", encoding="utf-8") as f:
                payload_probe = json.load(f)
            quality = payload_probe.get("quality", {})
            if quality.get("status") == "low_confidence":
                bootstrap_warning = (
                    (bootstrap_warning or "") + " "
                    + (quality.get("reason") or "Bootstrap ROI/MOI có độ tin cậy thấp.")
                ).strip()
            frame_w, frame_h = _get_video_size(video_path)
            q = _polygon_quality(payload_probe.get("roi", []), frame_w, frame_h)
            if q["ok"] < 0.5:
                roi_rerun_frames = min(bootstrap_frames, 120)
                _reg_log(
                    run_id,
                    f"  ROI quá nhỏ/hẹp (area={q['area_ratio']:.3f}, w={q['width_ratio']:.3f}, h={q['height_ratio']:.3f}) → re-run SAM ({roi_rerun_frames} frame)…"
                )
                cmd_sam_boot = [
                    py, "sam_bootstrap.py",
                    "--video", video_path,
                    "--weights", weights_path,
                    "--output-json", str(b3_json),
                    "--save-overlay", str(b3_overlay),
                    "--roi-expand", "1.05",
                    "--roi-shape", "rect",
                    "--imgsz", str(data["imgsz"]),
                    "--conf", str(data["conf"]),
                    "--max-frames", str(roi_rerun_frames),
                ]
                if moi_count > 0:
                    cmd_sam_boot += ["--moi-count", str(moi_count)]
                _popen_log(run_id, cmd_sam_boot, dtc_dir)
                bootstrap_warning = (
                    "ROI từ Grounded-SAM quá hẹp/nhỏ, đã tự động re-run SAM bootstrap. "
                    f"(area={q['area_ratio']:.3f}, width={q['width_ratio']:.3f}, height={q['height_ratio']:.3f})"
                )
                grounded_sam_ok = False  # ROI was bad, sam_bootstrap now owns the output
                # Re-read after ROI re-run
                with open(b3_json, "r", encoding="utf-8") as f:
                    payload_probe = json.load(f)
                quality = payload_probe.get("quality", {})

            # Validate MOI count. The web demo now follows the B2/B3/B4
            # comparison path: bootstrap provides ROI, while MOI is mined from
            # tracked trajectories and aligned to official MOI IDs when a
            # reference file is available.
            actual_moi = len(payload_probe.get("moi_vectors", {}))
            _reg_log(run_id, f"  Bootstrap tạo ra {actual_moi} MOI vectors.")
            if actual_moi < 3:
                warn_few = (
                    f"Bootstrap chỉ suy ra {actual_moi} MOI từ ảnh/mask. "
                    "Web sẽ dùng MOI sinh từ trajectory và align về ID MOI chính thức nếu có reference."
                )
                bootstrap_warning = ((bootstrap_warning or "") + " " + warn_few).strip()
                _reg_log(run_id, f"  {warn_few}")

            roi_txt = out_dir / "roi_from_bootstrap.txt"
            moi_txt = out_dir / "moi_from_bootstrap.txt"
            payload = payload_probe  # use final payload (already current)
            with open(roi_txt, "w", encoding="utf-8") as f:
                for p in payload.get("roi", []):
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        f.write(f"{float(p[0]):.2f},{float(p[1]):.2f}\n")
            with open(moi_txt, "w", encoding="utf-8") as f:
                items = sorted((int(k), v) for k, v in payload.get("moi_vectors", {}).items())
                for mid, vec in items:
                    if isinstance(vec, (list, tuple)) and len(vec) >= 4:
                        f.write(f"{mid},{float(vec[0]):.2f},{float(vec[1]):.2f},{float(vec[2]):.2f},{float(vec[3]):.2f}\n")

            if not roi_file:
                roi_file = str(roi_txt)
            if not moi_vectors_path:
                moi_vectors_path = str(moi_txt)

            tracked_moi = _build_track_mined_moi(
                run_id,
                py=py,
                dtc_dir=dtc_dir,
                out_dir=out_dir,
                video_path=video_path,
                weights_path=weights_path,
                roi_file=roi_file,
                movement_path=movement_path,
                movement_count_hint=moi_count,
                reference_moi_path=reference_moi_path,
                data=data,
                label="Grounding DINO + SAM ROI" if use_grounding else "SAM Automatic ROI",
            )
            if tracked_moi:
                moi_vectors_path = tracked_moi
                bootstrap_warning = (
                    (bootstrap_warning or "") + " "
                    + "MOI dùng để đếm được sinh từ trajectory và align về ID chính thức khi có reference."
                ).strip()
            elif not os.path.exists(moi_vectors_path):
                raise RuntimeError(
                    "Không tạo được MOI hợp lệ từ bootstrap hoặc trajectory. "
                    "Hãy dùng bộ demo, upload movement/MOI reference, hoặc chuyển sang chế độ thủ công."
                )
            overlay_url = f"/media/{run_id}/bootstrap_overlay.jpg"
            _reg_step(run_id, "bootstrap", "done")
            _reg_log(run_id, "✓ Bootstrap ROI + track-mined MOI hoàn thành.")

        elif not moi_vectors_path and roi_file and movement_path:
            tracked_moi = _build_track_mined_moi(
                run_id,
                py=py,
                dtc_dir=dtc_dir,
                out_dir=out_dir,
                video_path=video_path,
                weights_path=weights_path,
                roi_file=roi_file,
                movement_path=movement_path,
                movement_count_hint=moi_count,
                reference_moi_path=reference_moi_path,
                data=data,
                label="Manual ROI",
            )
            if tracked_moi:
                moi_vectors_path = tracked_moi
                bootstrap_warning = (
                    "Không có MOI upload/vẽ; web đã dùng track-mined MOI "
                    "và align về ID chính thức khi có reference."
                )

        # ── Counting ───────────────────────────────────────────────────────────
        _reg_step(run_id, "count", "running")
        _reg_log(run_id, "→ Bắt đầu đếm phương tiện…")
        _reg_pct(run_id, 0)

        out_csv = out_dir / "counting_result.csv"
        out_mp4 = out_dir / "counting_vis.mp4"

        cmd_count = [
            py, "run_dtc_counting.py",
            "--video", video_path,
            "--weights", weights_path,
            "--roi-file", roi_file,
            "--video-clip-id", str(data["video_clip_id"]),
            "--conf", str(data["conf"]),
            "--class-conf", data.get("class_conf", ""),
            "--imgsz", str(data["imgsz"]),
            "--frame-stride", str(data["frame_stride"]),
            "--output-csv", str(out_csv),
        ]
        if movement_path:
            cmd_count += ["--movement-description", movement_path]
        if moi_vectors_path:
            cmd_count += ["--moi-vectors", moi_vectors_path]

        effective_max_frames = int(data.get("max_frames") or 1200)
        if data.get("quick_preview"):
            if data.get("save_video"):
                effective_max_frames = max(effective_max_frames, 1200)
            cmd_count += ["--max-frames", str(effective_max_frames)]

        if data.get("save_video"):
            cmd_count += ["--save-video", str(out_mp4)]

        _popen_log(run_id, cmd_count, dtc_dir)
        _reg_step(run_id, "count", "done")
        _reg_log(run_id, "✓ Đếm phương tiện hoàn thành.")

        # ── Build result context ───────────────────────────────────────────────
        result_ctx: dict = {}
        result_ctx["run_id"] = run_id
        result_ctx["raw_video_url"] = raw_video_url
        result_ctx["csv_url"] = f"/media/{run_id}/counting_result.csv"
        result_ctx["run_mode"] = (
            f"Quick preview ({effective_max_frames} frames)"
            if data.get("quick_preview") else "Full run"
        )
        if bootstrap_warning:
            result_ctx["bootstrap_warning"] = bootstrap_warning
        if overlay_url:
            result_ctx["overlay_url"] = overlay_url
        result_ctx["uploaded_files"] = {
            "source": "Bộ demo cam_5" if data.get("use_demo_files") else "File upload",
            "video": Path(video_path).name,
            "weights": Path(weights_path).name,
            "roi": Path(roi_file).name if roi_file else "(từ bootstrap)",
            "movement": Path(movement_path).name if movement_path else "(không dùng)",
            "moi": Path(moi_vectors_path).name if moi_vectors_path else "(không dùng)",
            "class_conf": data.get("class_conf", ""),
        }
        if out_csv.exists():
            result_ctx["summary"] = _load_summary(str(out_csv))

        # ── Transcode video ────────────────────────────────────────────────────
        out_avi = out_dir / "counting_vis.avi"
        out_browser_mp4 = out_dir / "counting_vis_browser.mp4"
        source_video = None
        source_name = ""
        if out_mp4.exists() and out_mp4.stat().st_size > 0:
            source_video, source_name = out_mp4, "counting_vis.mp4"
        elif out_avi.exists() and out_avi.stat().st_size > 0:
            source_video, source_name = out_avi, "counting_vis.avi"

        if source_video is not None:
            _reg_step(run_id, "transcode", "running")
            _reg_log(run_id, "→ Đang chuyển mã video sang H.264…")
            ok = _transcode_to_browser_mp4(source_video, out_browser_mp4)
            _reg_step(run_id, "transcode", "done")
            if ok:
                result_ctx["video_url"] = f"/media/{run_id}/counting_vis_browser.mp4"
                result_ctx["video_download_url"] = result_ctx["video_url"]
                _reg_log(run_id, "✓ Video đã được chuyển mã thành công.")
            else:
                result_ctx["video_url"] = f"/media/{run_id}/{source_name}"
                result_ctx["video_download_url"] = result_ctx["video_url"]
                result_ctx["video_warning"] = (
                    "Không transcode được sang MP4 browser-friendly. "
                    "Nếu video hiển thị đen/0:00, hãy nhấn Tải Video để mở bằng player."
                )
        else:
            _reg_step(run_id, "transcode", "done")
            if data.get("save_video"):
                result_ctx["video_warning"] = "Không tạo được video visualize (có thể do codec OpenCV)."
            else:
                result_ctx["video_warning"] = "Đã tắt lưu video để tăng tốc độ xử lý."

        _reg_log(run_id, "✓ Tất cả hoàn thành!")
        _reg_done(run_id, result_ctx)

    except Exception as exc:
        traceback.print_exc()
        _reg_log(run_id, f"✗ LỖI: {exc}")
        _reg_error(run_id, str(exc))


# ── Views ──────────────────────────────────────────────────────────────────────

def _rebuild_result_from_fs(run_id: str) -> Optional[dict]:
    """Rebuild result context from persisted media files (used when registry cleared after server restart)."""
    out_dir = Path(settings.MEDIA_ROOT) / run_id
    csv_path = out_dir / "counting_result.csv"
    if not out_dir.exists() or not csv_path.exists():
        return None

    meta_path = out_dir / "run_meta.json"
    meta: dict = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    # Locate uploaded video in uploads/
    upload_dir = out_dir / "uploads"
    raw_video_url = ""
    if upload_dir.exists():
        for fp in upload_dir.iterdir():
            if fp.name.startswith("video_"):
                raw_video_url = f"/media/{run_id}/uploads/{fp.name}"
                break

    result_ctx: dict = {
        "run_id": run_id,
        "mode": meta.get("mode", "manual"),
        "raw_video_url": raw_video_url,
        "csv_url": f"/media/{run_id}/counting_result.csv",
        "run_mode": meta.get("run_mode", "Full run"),
        "uploaded_files": {
            "source": "Bộ demo cam_5" if meta.get("use_demo_files") else "File upload",
            "video": meta.get("video_name", ""),
            "class_conf": meta.get("class_conf", ""),
        },
        "summary": _load_summary(str(csv_path)),
    }

    browser_mp4 = out_dir / "counting_vis_browser.mp4"
    vis_mp4 = out_dir / "counting_vis.mp4"
    vis_avi = out_dir / "counting_vis.avi"
    if browser_mp4.exists() and browser_mp4.stat().st_size > 0:
        result_ctx["video_url"] = f"/media/{run_id}/counting_vis_browser.mp4"
        result_ctx["video_download_url"] = result_ctx["video_url"]
    elif vis_mp4.exists() and vis_mp4.stat().st_size > 0:
        result_ctx["video_url"] = f"/media/{run_id}/counting_vis.mp4"
        result_ctx["video_download_url"] = result_ctx["video_url"]
    elif vis_avi.exists() and vis_avi.stat().st_size > 0:
        result_ctx["video_url"] = f"/media/{run_id}/counting_vis.avi"
        result_ctx["video_download_url"] = result_ctx["video_url"]

    overlay_jpg = out_dir / "bootstrap_overlay.jpg"
    if overlay_jpg.exists():
        result_ctx["overlay_url"] = f"/media/{run_id}/bootstrap_overlay.jpg"

    return result_ctx

def _launch_run(
    request,
    form,
    mode: str,
    template: str,
    context: dict,
) -> None:
    """Shared logic to save uploads and start the background thread."""
    data = dict(form.cleaned_data)
    data.setdefault("auto_bootstrap", mode == "auto")
    data.setdefault("grounding_model", "IDEA-Research/grounding-dino-base")
    data.setdefault("text_prompt", "road surface . traffic lane . intersection")
    data["mode"] = mode

    root_dir = Path(__file__).resolve().parents[2]
    dtc_dir = str(root_dir)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(settings.MEDIA_ROOT) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    upload_dir = out_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    use_demo_files = bool(data.get("use_demo_files"))
    roi_file = ""
    movement_path = ""
    moi_vectors_path = ""
    reference_moi_path = ""
    moi_count_hint = 0
    raw_video_url = ""

    if use_demo_files:
        demo = _demo_paths(root_dir)
        required = ["video", "weights", "movement"]
        if mode == "manual":
            required += ["roi", "moi"]
        missing = [name for name in required if not demo[name].exists()]
        if missing:
            missing_text = ", ".join(f"{name}={demo[name]}" for name in missing)
            raise FileNotFoundError(f"Thiếu file demo mặc định: {missing_text}")

        video_path = str(demo["video"])
        weights_path = str(demo["weights"])
        movement_path = str(demo["movement"])
        if demo["moi"].exists():
            reference_moi_path = str(demo["moi"])
        if mode == "manual":
            roi_file = str(demo["roi"])
            moi_vectors_path = str(demo["moi"])
            try:
                moi_count_hint = sum(1 for ln in open(moi_vectors_path, encoding="utf-8") if ln.strip())
            except Exception:
                pass
        video_name = f"demo:{demo['video'].name}"
    else:
        video_path = _save_upload(
            data["video_upload"],
            upload_dir / f"video_{Path(data['video_upload'].name).name}",
        )
        weights_path = _save_upload(
            data["weights_upload"],
            upload_dir / f"weights_{Path(data['weights_upload'].name).name}",
        )
        if data.get("movement_upload"):
            movement_path = _save_upload(
                data["movement_upload"],
                upload_dir / f"movement_{Path(data['movement_upload'].name).name}",
            )

        if data.get("roi_upload"):
            roi_file = _save_upload(
                data["roi_upload"],
                upload_dir / f"roi_{Path(data['roi_upload'].name).name}",
            )
        elif (data.get("roi_json") or "").strip():
            # ROI drawn interactively on the canvas
            roi_points = json.loads(data["roi_json"])
            roi_txt = upload_dir / "roi_drawn.txt"
            with open(roi_txt, "w", encoding="utf-8") as f:
                for p in roi_points:
                    f.write(f"{int(p[0])},{int(p[1])}\n")
            roi_file = str(roi_txt)

        if data.get("moi_upload"):
            moi_vectors_path = _save_upload(
                data["moi_upload"],
                upload_dir / f"moi_{Path(data['moi_upload'].name).name}",
            )
            try:
                moi_count_hint = sum(1 for ln in open(moi_vectors_path, encoding="utf-8") if ln.strip())
            except Exception:
                pass
            reference_moi_path = moi_vectors_path
        elif (data.get("moi_json") or "").strip():
            # MOI drawn interactively on the canvas
            moi_data = json.loads(data["moi_json"])
            moi_count_hint = len(moi_data)
            moi_txt = upload_dir / "moi_drawn.txt"
            with open(moi_txt, "w", encoding="utf-8") as f:
                for v in moi_data:
                    f.write(f"{v['id']},{v['x1']},{v['y1']},{v['x2']},{v['y2']}\n")
            moi_vectors_path = str(moi_txt)
            reference_moi_path = moi_vectors_path

        raw_video_url = f"/media/{run_id}/uploads/{Path(video_path).name}"
        video_name = Path(data["video_upload"].name).name

    # Persist run metadata for history page (survives server restarts)
    run_meta = {
        "run_id": run_id,
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
        "video_name": video_name,
        "run_mode": "Quick preview" if data.get("quick_preview") else "Full run",
        "use_demo_files": use_demo_files,
        "class_conf": data.get("class_conf", ""),
    }
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as _mf:
        json.dump(run_meta, _mf, ensure_ascii=False)

    steps = [
        {"key": "upload", "label": "Lưu file tải lên", "state": "done"},
        {"key": "count", "label": "Đếm phương tiện", "state": "pending"},
        {"key": "transcode", "label": "Chuyển mã video", "state": "pending"},
    ]
    if data["auto_bootstrap"]:
        steps.insert(1, {"key": "bootstrap", "label": "Bootstrap ROI + sinh MOI từ trajectory", "state": "pending"})

    _reg_init(run_id, steps, mode=mode)

    py = sys.executable
    t = threading.Thread(
        target=_execute_run,
        args=(
            run_id, data, out_dir,
            video_path, weights_path, movement_path,
            roi_file, moi_vectors_path,
            dtc_dir, py, raw_video_url,
            moi_count_hint,
            reference_moi_path,
        ),
        daemon=True,
    )
    t.start()

    context["pending_run_id"] = run_id
    context["raw_video_url"] = raw_video_url


def dashboard(request):
    """Landing dashboard — introduces the system."""
    return render(request, "counter/index.html")


def manual_index(request):
    form = ManualForm(request.POST or None, request.FILES or None)
    context: dict = {"form": form, "page_mode": "manual"}

    if request.method == "POST" and form.is_valid():
        try:
            _launch_run(request, form, mode="manual", template="counter/manual.html", context=context)
        except Exception as exc:
            traceback.print_exc()
            context["run_error"] = f"Lỗi khi khởi tạo: {exc}"

    return render(request, "counter/manual.html", context)


def auto_index(request):
    form = AutoForm(request.POST or None, request.FILES or None)
    context: dict = {"form": form, "page_mode": "auto"}

    if request.method == "POST" and form.is_valid():
        try:
            _launch_run(request, form, mode="auto", template="counter/auto.html", context=context)
        except Exception as exc:
            traceback.print_exc()
            context["run_error"] = f"Lỗi khi khởi tạo: {exc}"

    return render(request, "counter/auto.html", context)


def run_status(request, run_id: str):
    with _REGISTRY_LOCK:
        entry = _RUN_REGISTRY.get(run_id)
    if not entry:
        return JsonResponse({"status": "not_found"}, status=404)
    return JsonResponse({
        "status": entry["status"],
        "steps": entry["steps"],
        "progress_pct": entry["progress_pct"],
        "logs": entry["logs"][-60:],
        "error": entry["error"],
    })


def run_result(request, run_id: str):
    with _REGISTRY_LOCK:
        entry = _RUN_REGISTRY.get(run_id)
    if not entry:
        # Registry cleared (server restart) — try rebuilding from filesystem
        ctx = _rebuild_result_from_fs(run_id)
        if ctx is None:
            return render(request, "counter/manual.html", {
                "form": ManualForm(),
                "page_mode": "manual",
                "run_error": f"Không tìm thấy phiên chạy {run_id}. Server có thể đã restart.",
            })
        mode = ctx.get("mode", "manual")
        ctx["page_mode"] = mode
        if mode == "auto":
            ctx["form"] = AutoForm()
            return render(request, "counter/auto.html", ctx)
        ctx["form"] = ManualForm()
        return render(request, "counter/manual.html", ctx)
    if entry["status"] != "done":
        return render(request, "counter/manual.html", {
            "form": ManualForm(),
            "page_mode": "manual",
            "run_error": f"Phiên chạy {run_id} chưa hoàn thành (status={entry['status']}).",
        })
    mode = entry.get("mode", "manual")
    ctx = dict(entry["result_ctx"])
    ctx["page_mode"] = mode
    if mode == "auto":
        ctx["form"] = AutoForm()
        return render(request, "counter/auto.html", ctx)
    ctx["form"] = ManualForm()
    return render(request, "counter/manual.html", ctx)


def history_index(request):
    """List all completed runs stored on disk."""
    media_root = Path(settings.MEDIA_ROOT)
    runs = []
    if media_root.exists():
        for folder in sorted(media_root.iterdir(), reverse=True):
            if not folder.is_dir():
                continue
            csv_path = folder / "counting_result.csv"
            if not csv_path.exists():
                continue
            run_id = folder.name
            meta_path = folder / "run_meta.json"
            meta: dict = {}
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            with _REGISTRY_LOCK:
                reg_entry = _RUN_REGISTRY.get(run_id)
            status = reg_entry["status"] if reg_entry else "done"
            runs.append({
                "run_id": run_id,
                "mode": meta.get("mode", "manual"),
                "timestamp": meta.get("timestamp", run_id),
                "video_name": meta.get("video_name", ""),
                "run_mode": meta.get("run_mode", "Full run"),
                "status": status,
            })
    return render(request, "counter/history.html", {"runs": runs, "page_mode": "history"})
