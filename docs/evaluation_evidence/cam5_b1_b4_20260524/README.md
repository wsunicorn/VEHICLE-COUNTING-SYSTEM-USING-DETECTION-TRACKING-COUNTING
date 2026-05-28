# Minh Chứng Đánh Giá Cam 5 - B1 Đến B4

Thư mục này lưu các file nhỏ được copy trực tiếp từ lần chạy thực nghiệm:

```text
dtc_counting/outputs/final_cam5_b1_b4_20260524_v4/
```

Mục tiêu là để người đọc có thể kiểm chứng bảng chỉ số trong paper/README mà không cần commit video visualize nặng. Các file `*.mp4` khoảng 30MB mỗi baseline và overlay ảnh không cần thiết cho kiểm chứng metric nên không đưa vào GitHub.

---

## 1. Kết Quả Tóm Tắt

| Baseline | ROI | MOI dùng để đếm | Pred/GT | nwRMSE | S1 Eff. | S1 Overall | Accuracy | MAE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| B1 | Official ROI | Official MOI | 95/96 | 0.2879 | 0.7121 | 0.4985 | 0.9896 | 3.3158 |
| B2 | Official ROI | Track-mined MOI aligned | 95/96 | 0.4350 | 0.5650 | 0.3955 | 0.9896 | 5.3529 |
| B3 | SAM Automatic ROI | Track-mined MOI aligned | 118/96 | 0.3906 | 0.6094 | 0.4266 | 0.7708 | 4.6667 |
| B4 | Grounding DINO + SAM ROI | Track-mined MOI aligned | 125/96 | 0.4071 | 0.5929 | 0.4150 | 0.6979 | 5.1053 |

File tóm tắt:

- `comparison_summary.json`
- `comparison_summary.csv`

---

## 2. Code Sinh Ra Các Kết Quả Này

Các file code chính:

| File code | Vai trò |
|---|---|
| `dtc_counting/run_full_comparison.py` | Script tổng điều phối B1-B4, gọi các bước bootstrap, mine MOI, chạy đếm và đánh giá |
| `dtc_counting/run_dtc_counting.py` | Pipeline chính: YOLO detection, Kalman-Hungarian tracking, ROI/MOI counting, xuất CSV |
| `dtc_counting/build_moi_from_tracks.py` | Sinh MOI từ trajectory của xe bên trong ROI |
| `dtc_counting/moi_utils.py` | Load/write MOI và align vector sinh ra về ID MOI chính thức bằng matching vector |
| `dtc_counting/sam_auto_bootstrap.py` | B3: dùng SAM Automatic để bootstrap ROI |
| `dtc_counting/grounded_sam_bootstrap.py` | B4: dùng Grounding DINO + SAM để bootstrap ROI |
| `dtc_counting/evaluate_counting.py` | Tính `nwRMSE`, `S1_Effectiveness`, `S1_Overall`, `Count Accuracy`, `MAE` |

Luồng chính:

```text
run_full_comparison.py
  -> B1: run_dtc_counting.py + official ROI/MOI
  -> B2: build_moi_from_tracks.py -> moi_utils.align_to_reference() -> run_dtc_counting.py
  -> B3: sam_auto_bootstrap.py -> ROI + track-mined aligned MOI -> run_dtc_counting.py
  -> B4: grounded_sam_bootstrap.py -> ROI + track-mined aligned MOI -> run_dtc_counting.py
  -> evaluate_counting.py cho từng baseline
  -> comparison_summary.json
```

Điểm quan trọng: B2, B3, B4 đều dùng cùng nguyên tắc MOI:

```text
trajectory vectors -> cluster/generate MOI -> align_to_reference() -> official movement IDs
```

---

## 3. Lệnh Chạy Lại B1-B4

Chạy từ thư mục `dtc_counting`:

```powershell
cd dtc_counting

python run_full_comparison.py `
  --video data/AIC21_Track1_Vehicle_Counting/counting_gt_sample/counting_example_cam_5_1min.mp4 `
  --weights ..\weights\best2.pt `
  --roi-file data/AIC21_Track1_Vehicle_Counting/ROIs/cam_5.txt `
  --movement-description data/AIC21_Track1_Vehicle_Counting/movement_description/cam_5.txt `
  --moi-vectors data/AIC21_Track1_Vehicle_Counting/MOI_vectors/cam_5.txt `
  --gt-csv data/AIC21_Track1_Vehicle_Counting/counting_gt_sample/counting_example_cam_5_1min.csv `
  --video-clip-id 10 `
  --total-frames 600 `
  --video-fps 10 `
  --imgsz 1280 `
  --conf 0.25 `
  --class-conf car=0.25,truck=0.75 `
  --mining-frames 1200 `
  --min-bootstrap-moi 3 `
  --output-dir outputs/final_cam5_b1_b4_20260524_v4
```

Yêu cầu để chạy lại:

- Có video sample `counting_example_cam_5_1min.mp4`.
- Có weights YOLO `weights/best2.pt`.
- Có checkpoint SAM nếu chạy B3/B4 đầy đủ.
- Nếu dùng Grounding DINO lần đầu, máy cần tải/cache model từ Hugging Face.

---

## 4. Lệnh Đánh Giá Riêng Một Baseline

Ví dụ đánh giá lại B2 từ CSV prediction:

```powershell
cd dtc_counting

python evaluate_counting.py `
  --gt-csv data/AIC21_Track1_Vehicle_Counting/counting_gt_sample/counting_example_cam_5_1min.csv `
  --pred-csv outputs/final_cam5_b1_b4_20260524_v4/b2_manual_tracked_moi.csv `
  --total-frames 600 `
  --video-fps 10 `
  --processing-time 154.79 `
  --output-json outputs/final_cam5_b1_b4_20260524_v4/b2_eval.json
```

Các baseline khác chỉ cần đổi `--pred-csv` và `--output-json`:

| Baseline | Prediction CSV | Eval JSON |
|---|---|---|
| B1 | `b1_manual_official_moi.csv` | `b1_eval.json` |
| B2 | `b2_manual_tracked_moi.csv` | `b2_eval.json` |
| B3 | `b3_sam_auto.csv` | `b3_eval.json` |
| B4 | `b4_grounded_sam.csv` | `b4_eval.json` |

---

## 5. Danh Sách File Minh Chứng Đã Commit

### Summary

| File | Ý nghĩa |
|---|---|
| `comparison_summary.json` | Summary tổng hợp do `run_full_comparison.py` sinh ra |
| `comparison_summary.csv` | Bản CSV để đưa vào bảng báo cáo/paper |
| `run_stdout.log` | Log stdout của lần chạy, chứa progress, số event, metric in ra console |

### B1 - Manual ROI + Official MOI

| File | Ý nghĩa |
|---|---|
| `b1_manual_official_moi.csv` | Prediction CSV của B1 |
| `b1_eval.json` | Metric chi tiết của B1 |
| `b1_timing.json` | Thời gian xử lý B1 |

### B2 - Manual ROI + Track-Mined MOI Aligned

| File | Ý nghĩa |
|---|---|
| `b2_moi_from_tracks.txt` | MOI thô sinh từ trajectory |
| `b2_moi_from_tracks_aligned.txt` | MOI sau khi align về official MOI ID |
| `b2_manual_tracked_moi.csv` | Prediction CSV của B2 |
| `b2_eval.json` | Metric chi tiết của B2 |
| `b2_timing.json` | Thời gian xử lý B2 |

### B3 - SAM Automatic ROI + Track-Mined MOI Aligned

| File | Ý nghĩa |
|---|---|
| `b3_sam_auto_bootstrap.json` | Output bootstrap ROI/MOI ban đầu từ SAM Automatic |
| `b3_bootstrap_decision.json` | Quyết định dùng track-mined MOI vì SAM chỉ sinh quá ít MOI hợp lệ |
| `b3_roi.txt` | ROI dùng cho B3 |
| `b3_moi.txt` | MOI aligned dùng cho B3 |
| `b3_sam_auto.csv` | Prediction CSV của B3 |
| `b3_eval.json` | Metric chi tiết của B3 |
| `b3_timing.json` | Thời gian xử lý B3 |

### B4 - Grounding DINO + SAM ROI + Track-Mined MOI Aligned

| File | Ý nghĩa |
|---|---|
| `b4_grounded_sam_bootstrap.json` | Output bootstrap ROI/MOI ban đầu từ Grounding DINO + SAM |
| `b4_bootstrap_decision.json` | Quyết định dùng track-mined MOI vì bootstrap sinh quá ít MOI hợp lệ |
| `b4_roi.txt` | ROI dùng cho B4 |
| `b4_moi.txt` | MOI aligned dùng cho B4 |
| `b4_grounded_sam.csv` | Prediction CSV của B4 |
| `b4_eval.json` | Metric chi tiết của B4 |
| `b4_timing.json` | Thời gian xử lý B4 |

---

## 6. File Không Commit

Các file sau có trong output gốc nhưng không đưa vào GitHub:

| File | Lý do |
|---|---|
| `b1_vis.mp4`, `b2_vis.mp4`, `b3_vis.mp4`, `b4_vis.mp4` | Video visualize nặng, không cần để kiểm chứng metric |
| `b3_sam_auto_overlay.jpg`, `b4_grounded_sam_overlay.jpg` | Ảnh minh họa ROI, có ích khi trình bày nhưng không bắt buộc cho kiểm chứng định lượng |
| `run_stderr.log` | File rỗng trong lần chạy này |

