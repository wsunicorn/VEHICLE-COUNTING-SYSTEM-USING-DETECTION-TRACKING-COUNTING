# Hướng Dẫn Hệ Thống Đếm Phương Tiện DTC (Theo Paper Nhóm 19)

Tài liệu này hướng dẫn đầy đủ cách chạy hệ thống Detection-Tracking-Counting với:

- YOLO detector (`best2.pt`) cho car/truck
- Tracker Kalman 8D + Hungarian matching 3 bước (IoU, Mahalanobis, Histogram)
- Counter theo ROI/MOI, có hỗ trợ eROI và iROI
- Baseline 3 với Grounded DINO + SAM để gợi ý ROI/MOI
- Web demo local bằng Django

## 1) Cấu trúc quan trọng

Trong thư mục `dtc_counting`:

- `run_dtc_counting.py`: pipeline đếm chính theo DTC
- `run_three_baselines.py`: chạy 3 baseline tự động
- `build_moi_from_tracks.py`: sinh MOI từ trajectory
- `grounded_sam_bootstrap.py`: Grounded DINO + SAM bootstrap ROI/MOI
- `sam_bootstrap.py`: bản bootstrap ROI/MOI theo SAM (fallback)
- `web_demo/`: web local Django demo

## 2) Cài đặt môi trường

Di chuyển vào thư mục `dtc_counting`, sau đó cài gói:

```powershell
python -m pip install ultralytics opencv-python numpy scipy django transformers
```

Lưu ý:

- `scipy` dùng cho Hungarian matching.
- `transformers` dùng cho Grounded DINO.
- Nếu máy có GPU CUDA, `torch` cần tương thích với CUDA đang dùng. Nếu cần, cài torch theo hướng dẫn chính thức của PyTorch.

## 3) Chạy DTC 1 video (chế độ chính)

Lệnh mẫu cho cam_5 (1 phút):

```powershell
python run_dtc_counting.py --video data/AIC21_Track1_Vehicle_Counting_Bosung/counting_gt_sample/counting_example_cam_5_1min.mp4 --weights ../best2.pt --roi-file data/AIC21_Track1_Vehicle_Counting_Bosung/ROIs/cam_5.txt --movement-description data/AIC21_Track1_Vehicle_Counting_Bosung/movement_description/cam_5.txt --video-clip-id 10 --output-csv outputs/cam5_pred.csv --save-video outputs/cam5_vis.mp4
```

File CSV đầu ra theo format AI City:

- `video_clip_id`
- `frame_id`
- `movement_id`
- `vehicle_class_id` (1=car, 2=truck)

## 4) Chạy 3 baseline để so sánh

Chạy đầy đủ 3 baseline:

```powershell
python run_three_baselines.py
```

Chạy nhanh (bỏ baseline 3):

```powershell
python run_three_baselines.py --skip-baseline3
```

Giảm thời gian cho baseline 2:

```powershell
python run_three_baselines.py --mining-frames 400 --imgsz 640
```

Chọn chế độ baseline 3:

- Grounded DINO + SAM (mặc định):

```powershell
python run_three_baselines.py --baseline3-mode grounded-sam
```

- SAM fallback:

```powershell
python run_three_baselines.py --baseline3-mode sam
```

Output trong `outputs/baselines/`:

- `baseline1_angle_fallback.csv`
- `baseline2_moi_from_tracks.txt`
- `baseline2_track_moi.csv`
- `baseline3_grounded_sam_moi.csv` (nếu có)
- `baseline_summary.json`

## 5) Grounded DINO + SAM bootstrap ROI/MOI (chạy riêng)

```powershell
python grounded_sam_bootstrap.py --video data/AIC21_Track1_Vehicle_Counting_Bosung/counting_gt_sample/counting_example_cam_5_1min.mp4 --output-json outputs/cam5_grounded_bootstrap.json --save-overlay outputs/cam5_grounded_overlay.jpg --moi-count 12
```

Sau khi có JSON, bạn có thể:

1. Kiểm tra overlay
2. Chuyển ROI/MOI sang txt
3. Truyền vào `run_dtc_counting.py` qua `--roi-file` và `--moi-vectors`

## 6) Tương ứng với paper

Phần đã bám sát paper:

- Detection bằng YOLO
- Tracking bằng Kalman 8D + Hungarian + 3 matching stage
- Counting theo ROI/MOI và movement assignment theo vector
- Baseline bootstrap ROI/MOI

Phần còn có thể nâng cấp tiếp:

- iROI/eROI đã hỗ trợ tham số, cần tạo file cho từng camera để đúng setup paper
- Đánh giá S1/nwRMSE theo đúng công thức challenge có thể bổ sung thêm script riêng

## 7) Chạy Web Demo local (Django)

### 7.1 Khởi tạo DB

```powershell
cd web_demo
python manage.py migrate
```

### 7.2 Chạy server

```powershell
python manage.py runserver
```

Mở trình duyệt:

- `http://127.0.0.1:8000/`

### 7.3 Cách demo trên web

1. Điền đường dẫn video, weights, ROI, movement_description
2. Nếu muốn tự sinh ROI/MOI, tick "Tự sinh ROI/MOI bằng Grounded SAM"
3. Bấm "Chạy Demo"
4. Xem:
   - ảnh overlay ROI/MOI (nếu bootstrap)
   - video output đã vẽ box và count
   - bảng tổng hợp đếm theo movement-class
   - link tải CSV

Kết quả web sẽ lưu trong:

- `outputs/web_demo/<timestamp>/`

## 8) Lỗi thường gặp

### 8.1 Missing scipy

- Lỗi: không tìm thấy `scipy`
- Cách sửa:

```powershell
python -m pip install scipy
```

### 8.2 Missing transformers

- Lỗi: không tìm thấy `transformers`
- Cách sửa:

```powershell
python -m pip install transformers
```

### 8.3 Baseline 3 chậm

- Giảm `imgsz`
- Dùng `--baseline3-mode sam` để test nhanh
- Hoặc skip baseline 3 để debug tracking/counting trước

## 9) Kịch bản demo đề tài

Kịch bản khuyến nghị:

1. Chạy `run_three_baselines.py --skip-baseline3` để có so sánh nhanh baseline 1-2
2. Chạy lại baseline 3 riêng bằng `grounded_sam` nếu cần
3. Mở Django demo để trình bày giao diện và video kết quả
4. Trích xuất `baseline_summary.json` và CSV làm bảng so sánh trong báo cáo
