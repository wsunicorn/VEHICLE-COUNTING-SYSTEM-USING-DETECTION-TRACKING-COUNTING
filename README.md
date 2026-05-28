# Vehicle Counting System Using Detection, Tracking, ROI and MOI

Project xây dựng hệ thống đếm phương tiện giao thông từ video camera cố định. Hệ thống không chỉ đếm tổng số xe, mà còn đếm theo **loại xe** và **hướng di chuyển**.

Đầu ra chính là CSV:

```csv
video_clip_id,frame_id,movement_id,vehicle_class_id
10,145,3,1
10,302,7,2
```

| Trường | Ý nghĩa |
|---|---|
| `video_clip_id` | ID video/camera |
| `frame_id` | frame tại thời điểm xe được đếm |
| `movement_id` | ID hướng di chuyển |
| `vehicle_class_id` | loại xe, `1 = car`, `2 = truck` |

---

## Mục Tiêu Và Ý Nghĩa

Bài toán của project là **vehicle turning-movement counting**: đếm xe theo từng luồng di chuyển trong giao lộ. Nếu chỉ dùng object detection, một xe xuất hiện trong nhiều frame sẽ bị đếm nhiều lần. Vì vậy hệ thống cần kết hợp detection, tracking, ROI/MOI và đánh giá định lượng.

Ý nghĩa của hệ thống:

- thống kê lưu lượng xe theo hướng đi,
- hỗ trợ phân tích giao thông đô thị,
- tạo web demo để trình bày pipeline computer vision,
- thử nghiệm khả năng giảm công cấu hình ROI/MOI bằng SAM và trajectory mining,
- có minh chứng đánh giá để người khác kiểm chứng kết quả.

---

## Phương Pháp Tổng Quan

```text
Video
  -> YOLO vehicle detection
  -> Kalman + Hungarian multi-object tracking
  -> ROI/MOI movement assignment
  -> Counting CSV + overlay video
  -> Evaluation metrics
```

Các thành phần chính:

| Thành phần | File chính | Vai trò |
|---|---|---|
| Detection | `dtc_counting/run_dtc_counting.py` | Dùng YOLO phát hiện `car/truck` |
| Tracking | `MultiStepTracker` trong `run_dtc_counting.py` | Nối detection thành trajectory |
| Counting | `run_dtc_counting.py` | Gán track vào ROI/MOI và ghi event |
| MOI mining | `build_moi_from_tracks.py` | Sinh MOI từ trajectory |
| MOI alignment | `moi_utils.py` | Align MOI sinh ra về ID MOI chính thức |
| SAM bootstrap | `sam_auto_bootstrap.py`, `grounded_sam_bootstrap.py` | Bootstrap ROI tự động |
| Evaluation | `evaluate_counting.py` | Tính metric đánh giá |
| Web demo | `dtc_counting/web_demo/` | Giao diện chạy thử hệ thống |

### Detection Bằng YOLO

YOLO nhận từng frame và trả về bounding box, class, confidence của xe. Project tập trung hai class:

- `car`
- `truck`

Detector được fine-tune từ dữ liệu frame giao thông đã gán nhãn. Khi chạy đếm, hệ thống có thể dùng ngưỡng riêng theo class, ví dụ:

```text
car=0.25,truck=0.75
```

### Tracking Bằng Kalman + Hungarian

Tracker dùng Kalman Filter để dự đoán vị trí tiếp theo của xe và Hungarian Matching để ghép detection mới với track cũ. Matching kết hợp:

- IoU giữa bounding box,
- Mahalanobis distance từ Kalman Filter,
- histogram màu để giảm mất ID khi xe gần nhau.

Kết quả của tracking là trajectory, tức chuỗi vị trí của cùng một xe qua nhiều frame.

### Counting Bằng ROI/MOI

ROI là vùng cần đếm. MOI là vector hướng di chuyển. Một track được đếm khi đủ điều kiện như đã vào ROI, có quỹ đạo đủ dài và chưa bị đếm trước đó.

Movement ID được gán bằng cách so sánh vector trajectory với các vector MOI. MOI có hướng và vị trí phù hợp nhất sẽ được chọn.

### Auto ROI Và Track-Mined MOI

Các nhánh tự động dùng SAM hoặc Grounding DINO + SAM để bootstrap ROI. MOI dùng trong B2/B3/B4 và web auto path được sinh từ trajectory bằng `build_moi_from_tracks.py`, sau đó align về ID chính thức bằng `moi_utils.py` nếu có file MOI reference.

---

## Dữ Liệu Và Huấn Luyện

Dữ liệu mẫu nhỏ dùng trong repo:

```text
dtc_counting/data/AIC21_Track1_Vehicle_Counting/
├── ROIs/cam_5.txt
├── MOI_vectors/cam_5.txt
├── movement_description/cam_5.txt
└── counting_gt_sample/counting_example_cam_5_1min.csv
```

Quy trình tạo detector:

```text
Video -> extract frames -> Roboflow annotation -> YOLO dataset -> fine-tune YOLO -> weights/*.pt
```

Roboflow được dùng để quản lý ảnh, vẽ bounding box cho `car/truck`, chia train/validation và export dataset theo format YOLO.

Weights YOLO trong repo:

```text
weights/best.pt
weights/best2.pt
weights/best4.pt
```

File lớn không commit:

- video dataset đầy đủ,
- output video,
- media web demo,
- SAM checkpoint `dtc_counting/sam_b.pt`.

---

## Paper Và Nghiên Cứu Tham Khảo

Paper của nhóm nằm tại:

```text
docs/Nhom19_paper_ComputerVision.docx
```

Sơ đồ hệ thống:

```text
docs/workflow_diagram.png
```

Project tham khảo hướng bài toán và metric từ **AI City Challenge Track 1 - Vehicle Counting**. Một số hướng nghiên cứu liên quan trong paper:

| Nghiên cứu/hướng | Ý nghĩa với project |
|---|---|
| AI City Challenge Track 1 | Bối cảnh bài toán đếm xe theo movement/class |
| CenterTrack-based counting | Nhấn mạnh vai trò tracking ổn định trong vehicle counting |
| Fast Vehicle Turning-Movement Counting Using Localization-Based Tracking | Gợi ý hướng dùng tracking/localization nhẹ để tăng tốc |
| Tiny-PIRATE / các hệ thống AI City Track 1 | Cho thấy ROI/MOI và tối ưu pipeline ảnh hưởng lớn đến điểm cuối |
| SAM / Grounding DINO + SAM | Hướng hỗ trợ bootstrap ROI để giảm cấu hình thủ công |

Hệ thống hiện tại không nhằm vượt leaderboard, mà tập trung xây dựng pipeline đầy đủ, có web demo, có baseline B1-B4 và có evidence để kiểm chứng kết quả.

---

## Cài Đặt

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install ultralytics opencv-python numpy scipy django transformers torch torchvision
```

Nếu dùng GPU, cài PyTorch đúng bản CUDA theo máy.

---

## Chạy Web Demo

```powershell
cd dtc_counting\web_demo
python manage.py migrate
python manage.py runserver
```

Mở:

```text
http://127.0.0.1:8000/
```

Web có hai chế độ:

| Chế độ | Mục đích |
|---|---|
| Manual | Dùng ROI/MOI có sẵn hoặc vẽ ROI/MOI trên canvas |
| Auto | Dùng SAM/Grounding-SAM bootstrap ROI, sau đó sinh MOI từ trajectory và align về ID chính thức nếu có reference |

---

## Chạy Pipeline CLI

Ví dụ chạy đếm với ROI/MOI chuẩn của cam 5:

```powershell
cd dtc_counting

python run_dtc_counting.py `
  --video data/AIC21_Track1_Vehicle_Counting/counting_gt_sample/counting_example_cam_5_1min.mp4 `
  --weights ..\weights\best2.pt `
  --roi-file data/AIC21_Track1_Vehicle_Counting/ROIs/cam_5.txt `
  --movement-description data/AIC21_Track1_Vehicle_Counting/movement_description/cam_5.txt `
  --moi-vectors data/AIC21_Track1_Vehicle_Counting/MOI_vectors/cam_5.txt `
  --video-clip-id 10 `
  --class-conf car=0.25,truck=0.75 `
  --output-csv outputs/cam5_pred.csv `
  --save-video outputs/cam5_vis.mp4
```

---

## Chạy So Sánh B1-B4

Script:

```text
dtc_counting/run_full_comparison.py
```

Các baseline:

| Baseline | Cấu hình | Mục đích |
|---|---|---|
| B1 | Manual ROI + Official MOI | Mốc tham chiếu khi ROI/MOI đúng |
| B2 | Manual ROI + Track-Mined MOI aligned | Kiểm tra MOI sinh từ trajectory |
| B3 | SAM Automatic ROI + Track-Mined MOI aligned | Kiểm tra ROI tự động từ SAM |
| B4 | Grounding DINO + SAM ROI + Track-Mined MOI aligned | Kiểm tra ROI tự động có prompt ngôn ngữ |

Lệnh chạy:

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

---

## Đánh Giá

Kết quả mẫu trên cam 5:

```text
docs/evaluation_evidence/cam5_b1_b4_20260524/comparison_summary.json
```

| Baseline | Pred/GT | nwRMSE | S1 Eff. | S1 Overall | Accuracy | MAE |
|---|---:|---:|---:|---:|---:|---:|
| B1 Manual ROI + Official MOI | 95/96 | 0.2879 | 0.7121 | 0.4985 | 98.96% | 3.32 |
| B2 Manual ROI + Track-Mined MOI | 95/96 | 0.4350 | 0.5650 | 0.3955 | 98.96% | 5.35 |
| B3 SAM Automatic ROI + Track-Mined MOI | 118/96 | 0.3906 | 0.6094 | 0.4266 | 77.08% | 4.67 |
| B4 Grounding DINO + SAM ROI + Track-Mined MOI | 125/96 | 0.4071 | 0.5929 | 0.4150 | 69.79% | 5.11 |

Metric chính:

| Metric | Ý nghĩa |
|---|---|
| `nwRMSE` | Sai số đếm tích lũy đã chuẩn hóa, càng thấp càng tốt |
| `S1_Effectiveness` | Chất lượng đếm theo movement/class, càng cao càng tốt |
| `S1_Overall` | Điểm tổng hợp theo hướng AI City |
| `Count Accuracy` | Độ đúng tổng số xe |
| `MAE` | Sai số trung bình theo movement/class |

Nhận xét ngắn:

- B1 tốt nhất vì dùng ROI/MOI chính thức.
- B2 cho thấy MOI có thể được sinh từ trajectory nhưng vẫn kém MOI chuẩn.
- B3/B4 đưa SAM/Grounding-SAM vào định lượng bằng cách dùng ROI bootstrap và MOI sinh từ trajectory.

---

## Kiểm Chứng Đánh Giá

Toàn bộ minh chứng nhỏ đã được commit tại:

```text
docs/evaluation_evidence/cam5_b1_b4_20260524/
```

Các file quan trọng:

| File | Dùng để kiểm chứng |
|---|---|
| `README.md` | Giải thích cách tái lập và ý nghĩa từng file |
| `comparison_summary.json/csv` | Bảng tổng hợp B1-B4 |
| `b1_eval.json` đến `b4_eval.json` | Metric chi tiết từng baseline |
| `b1_manual_official_moi.csv` | Prediction CSV của B1 |
| `b2_manual_tracked_moi.csv` | Prediction CSV của B2 |
| `b3_sam_auto.csv` | Prediction CSV của B3 |
| `b4_grounded_sam.csv` | Prediction CSV của B4 |
| `b2_moi_from_tracks_aligned.txt` | MOI từ trajectory sau khi align ID |
| `b3_bootstrap_decision.json` | Quyết định B3 dùng track-mined MOI |
| `b4_bootstrap_decision.json` | Quyết định B4 dùng track-mined MOI |
| `run_stdout.log` | Log console của lần chạy |

Để kiểm chứng nhanh, mở:

```text
docs/evaluation_evidence/cam5_b1_b4_20260524/README.md
```

---

## Cấu Trúc Thư Mục

```text
Project/
├── README.md
├── docs/
│   ├── Nhom19_paper_ComputerVision.docx
│   ├── workflow_diagram.png
│   └── evaluation_evidence/
├── weights/
│   ├── best.pt
│   ├── best2.pt
│   └── best4.pt
└── dtc_counting/
    ├── run_dtc_counting.py
    ├── run_full_comparison.py
    ├── evaluate_counting.py
    ├── build_moi_from_tracks.py
    ├── moi_utils.py
    ├── sam_auto_bootstrap.py
    ├── grounded_sam_bootstrap.py
    ├── data/
    ├── outputs/
    └── web_demo/
```
