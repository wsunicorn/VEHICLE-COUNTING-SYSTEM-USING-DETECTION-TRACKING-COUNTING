# HỆ THỐNG ĐẾM PHƯƠNG TIỆN GIAO THÔNG
# Vehicle Counting System Using Detection-Tracking-Counting Framework

> **Nhóm 19** — Nguyễn Tấn Minh · Nguyễn Ngọc Lân · Nguyễn Hữu Phúc  
> Khoa Công nghệ Thông tin, Trường Đại học Công nghiệp TP.HCM  
> GitHub: [wsunicorn/VEHICLE-COUNTING-SYSTEM-USING-DETECTION-TRACKING-COUNTING](https://github.com/wsunicorn/VEHICLE-COUNTING-SYSTEM-USING-DETECTION-TRACKING-COUNTING)

---

## Mục lục

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Kiến trúc hệ thống](#2-kiến-trúc-hệ-thống)
3. [Bộ dữ liệu](#3-bộ-dữ-liệu)
4. [Huấn luyện YOLO](#4-huấn-luyện-yolo)
5. [Module Phát hiện (Detection)](#5-module-phát-hiện-detection)
6. [Module Theo dõi (Tracking) — Kỹ thuật toán học](#6-module-theo-dõi-tracking--kỹ-thuật-toán-học)
7. [Module Đếm (Counting)](#7-module-đếm-counting)
8. [Cải tiến: Grounded SAM Bootstrap](#8-cải-tiến-grounded-sam-bootstrap)
9. [Bốn Baseline so sánh](#9-bốn-baseline-so-sánh)
10. [Độ đo đánh giá](#10-độ-đo-đánh-giá)
11. [Web Demo (Django)](#11-web-demo-django)
12. [Cài đặt môi trường](#12-cài-đặt-môi-trường)
13. [Hướng dẫn sử dụng](#13-hướng-dẫn-sử-dụng)
14. [Cấu trúc dự án](#14-cấu-trúc-dự-án)
15. [Tài liệu tham khảo](#15-tài-liệu-tham-khảo)

---

## 1. Tổng quan dự án

Hệ thống đếm phương tiện giao thông tự động theo framework **Detection-Tracking-Counting (DTC)**, nhằm phân tích lưu lượng giao thông theo thời gian thực từ camera giám sát.

**Bài toán:** Cho video giao thông V = {f₁, f₂, ..., fₙ} gồm n khung hình, cùng đa giác ROI và vector MOI định nghĩa sẵn, hệ thống xuất ra số đếm `C(loại, hướng)` cho mỗi tổ hợp loại phương tiện (car / truck) và hướng di chuyển (MOI).

**Đầu ra:** File CSV định dạng AI City Challenge:

```
video_clip_id, frame_id, movement_id, vehicle_class_id
10, 145, 3, 1       ← xe hơi đi theo hướng MOI 3 tại frame 145
10, 302, 7, 2       ← xe tải đi theo hướng MOI 7 tại frame 302
```

**Những điểm nổi bật:**

| Thành phần | Công nghệ |
|---|---|
| Phát hiện | YOLOv8 (fine-tuned) |
| Theo dõi | Kalman Filter 8D + Hungarian Matching 3 bước |
| Đếm | Vector-based MOI assignment + Kalman-exit trigger |
| Bootstrap ROI/MOI | Grounding DINO + SAM (zero-shot) |
| Web demo | Django + interactive canvas |

---

## 2. Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INPUT: Video Camera Feed                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ frame fₜ
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│   MODULE 1: DETECTION (YOLO)                                        │
│   • YOLO inference → bounding boxes {d_i = [x,y,a,h], c_i}        │
│   • Filter: only car (cls=1) and truck (cls=2)                     │
│   • eROI gate: loại detection nằm ngoài vùng quan sát              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ detections Dₜ
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│   MODULE 2: TRACKING (Kalman + Hungarian)                           │
│   ┌─ Stage 1: IoU Matching (threshold ≥ 0.1)                       │
│   ├─ Stage 2: Mahalanobis Matching (threshold d² ≤ 16.0)           │
│   └─ Stage 3: Histogram L1 Matching (48-dim, threshold ≤ 0.45)    │
│   • Kalman Predict → Update cho các track được khớp               │
│   • Tạo track mới cho detection chưa khớp                         │
│   • Xóa track sau max_missed=35 frame liên tiếp bỏ lỡ            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ tracks Tₜ (quỹ đạo liên tục)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│   MODULE 3: COUNTING (ROI/MOI)                                      │
│   • Kiểm tra ever_inside_roi và iROI illegal                       │
│   • Trigger đếm khi tracklet exit ROI (Kalman predicted position)  │
│   • Gán MOI: score = 100·θ + ‖Ŝ−Sₘ‖ + ‖Ê−Eₘ‖                    │
│   • Ghi nhận (frame_id, movement_id, cls_id) vào kết quả          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│   OUTPUT: counting_result.csv + visualization video                 │
└─────────────────────────────────────────────────────────────────────┘

         ╔══════════════════════════════════════════════════╗
         ║  OFFLINE BOOTSTRAP (chạy 1 lần trước pipeline)  ║
         ║  Grounding DINO → SAM → PCA → ROI/MOI JSON      ║
         ╚══════════════════════════════════════════════════╝
```

---

## 3. Bộ dữ liệu

### 3.1 AI City Challenge 2021 Track-1

| Thuộc tính | Thông tin |
|---|---|
| Số video | 31 clips |
| Số camera | 20 cảnh khác nhau |
| Tổng thời lượng | ~5 giờ (Dataset A: Public) |
| Độ phân giải | Cao (1920×1080 và 2704×1520) |
| Loại camera | Fix-mount, bird-eye view, low-angle |

**Đặc điểm đa dạng:**
- Góc quay: bird-eye (cam_1), low-angle (cam_14), mixed (cam_5)
- Thời tiết: nắng, bình minh, tuyết, mưa
- Mật độ: thưa (highway) đến dày đặc (ngã tư giờ cao điểm)
- Camera fisheye: cam_5, cam_14 (distortion ảnh hưởng bbox)

**Cấu hình ROI & MOI (cam_5 mẫu):**

```
Giao lộ 4 hướng có đèn tín hiệu (cam_5 fisheye)
12 MOI = 4 hướng vào × 3 hướng ra (thẳng, trái, phải)
  Movement 1: North  → East  (rẽ phải từ hướng bắc)
  Movement 2: North  → South (đi thẳng từ hướng bắc)
  Movement 3: North  → West  (rẽ trái từ hướng bắc)
  ...
  Movement 12: West → East
```

### 3.2 Bộ dữ liệu tự gán nhãn

Nhóm tự tạo dataset gán nhãn bằng cách extract frame từ video AIC21 Track-1, sử dụng nền tảng **Roboflow**.

| Cấu hình | Giá trị |
|---|---|
| Số người gán nhãn | 3 thành viên nhóm |
| Nền tảng | Roboflow (collaborative labeling) |
| Số lớp | 2 (car, truck) |
| Tỷ lệ train/val | 80% / 20% |

**Quy tắc gán nhãn (annotation guidelines):**
- Chỉ gán nhãn khi ≥ 50% thân xe nhìn thấy được
- Xe bị che khuất một phần: gán theo phần nhìn thấy
- Xe nằm ngoài ROI: loại bỏ, không gán nhãn
- Xe đang dừng tại đèn đỏ: vẫn gán nhãn nếu trong ROI
- Xe quá nhỏ (< 20×20 pixel): bỏ qua

**Phân bố lớp (ước lượng):**
```
car  : ~75–80% (dominant class)
truck: ~20–25%
```

---

## 4. Huấn luyện YOLO

### 4.1 Lựa chọn kiến trúc

Hệ thống đánh giá hai biến thể YOLO để cân bằng accuracy và speed:

| Model | Params | Speed | mAP@0.5 | Use case |
|---|---|---|---|---|
| YOLOv4-Tiny | ~6M | Rất nhanh | Thấp hơn | Edge device |
| **YOLOv8** (chọn) | ~25M (medium) | Nhanh | Cao hơn | Main pipeline |

YOLOv8 được chọn vì tích hợp sẵn Ultralytics API, hỗ trợ mixed precision training, và có PyTorch backbone dễ dàng fine-tune.

### 4.2 Chuẩn bị dữ liệu

**Bước 1 — Extract frames từ video AIC21:**

```bash
# Ví dụ: extract 1 frame mỗi 5 frames
ffmpeg -i cam_5.mp4 -vf "select=not(mod(n\,5))" -vsync vfr frames/cam5_%04d.jpg
```

**Bước 2 — Gán nhãn trên Roboflow:**
1. Upload frames lên Roboflow project
2. Annotation tool: Draw bounding boxes cho `car` và `truck`
3. Review & quality check bởi từng thành viên
4. Export định dạng **YOLOv8 (txt)** với cấu trúc:

```
dataset/
├── images/
│   ├── train/     ← 80% frames
│   └── val/       ← 20% frames
├── labels/
│   ├── train/     ← .txt files (class cx cy w h normalized)
│   └── val/
└── data.yaml
```

**File `data.yaml`:**

```yaml
path: /path/to/dataset
train: images/train
val:   images/val

nc: 2
names:
  0: car
  1: truck
```

**Lưu ý mapping class ID:**

| Roboflow label | YOLO class index | AI City class ID |
|---|---|---|
| car | 0 | 1 |
| truck | 1 | 2 |

> Code `class_to_aicity_id()` trong `run_dtc_counting.py` thực hiện mapping này.

### 4.3 Quá trình huấn luyện

**Lệnh huấn luyện chính:**

```bash
yolo train \
  model=yolov8m.pt \          # backbone pretrained trên COCO
  data=data.yaml \
  epochs=100 \
  imgsz=1280 \
  batch=16 \
  device=0 \                  # GPU 0
  project=runs/train \
  name=vehicle_counter \
  optimizer=AdamW \
  lr0=0.01 \
  lrf=0.01 \
  warmup_epochs=3 \
  mosaic=1.0 \                # Mosaic augmentation
  close_mosaic=10 \           # Tắt mosaic ở 10 epoch cuối
  hsv_h=0.015 \
  hsv_s=0.7 \
  hsv_v=0.4 \
  fliplr=0.5 \
  degrees=0.0 \               # Không rotate (camera fix-mount)
  save=True \
  plots=True
```

**Augmentation áp dụng:**

| Augmentation | Giá trị | Mục đích |
|---|---|---|
| **Mosaic** | 1.0 (bật) | Ghép 4 ảnh, tăng đa dạng scale và context |
| HSV Hue jitter | ±1.5% | Bất biến màu sắc ánh sáng |
| HSV Saturation | ±70% | Thời tiết (mưa giảm saturation) |
| HSV Value | ±40% | Điều kiện ánh sáng (bình minh, hoàng hôn) |
| Horizontal flip | 50% | Camera đối xứng |
| Rotation | 0° | Giữ nguyên (camera fix, không cần rotate) |
| Close Mosaic | 10 epoch | Ổn định training ở giai đoạn cuối |

**Transfer learning từ COCO:**

```
COCO pretrained weights
        │
        ▼  (backbone frozen ở 3 epochs đầu)
Fine-tune trên AIC21 custom dataset
        │
        ▼
best.pt / best2.pt  ← weights sử dụng trong pipeline
```

Backbone đã học được các đặc trưng cơ bản (edges, textures, shapes) từ 80 lớp COCO. Fine-tune chuyên biệt hóa cho car/truck trong cảnh giao thông.

### 4.4 Đánh giá model

**Metrics:**

```bash
yolo val model=runs/train/vehicle_counter/weights/best.pt data=data.yaml
```

| Metric | Ý nghĩa |
|---|---|
| **mAP@0.5** | mean Average Precision tại IoU=0.5 |
| **mAP@0.5:0.95** | mAP trung bình qua nhiều ngưỡng IoU |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1-score | 2 × P × R / (P + R) |

**Kết quả trên validation set (ước lượng):**

```
Class    Images  Instances  P       R       mAP50   mAP50-95
all      N       N          0.87+   0.83+   0.85+   0.58+
car      N       N          0.89+   0.85+   0.87+   0.60+
truck    N       N          0.82+   0.78+   0.80+   0.54+
```

### 4.5 Inference parameters

| Tham số | Giá trị | Mô tả |
|---|---|---|
| `--weights` | `best2.pt` | Weights file sau fine-tune |
| `--imgsz` | 1280 (CLI) / 960 (web) | Kích thước inference |
| `--conf` | 0.25–0.30 | Ngưỡng confidence tối thiểu |
| `--frame-stride` | 1 (mặc định) | Xử lý mỗi N frame |

---

## 5. Module Phát hiện (Detection)

### 5.1 Định nghĩa bài toán

Cho khung hình RGB `I_t` tại thời điểm t, bộ phát hiện tạo ra danh sách detection:

```
D_t = {(d_i, c_i)} với i = 1..K
```

Trong đó:
- `d_i = [x, y, a, h]ᵀ` — bounding box (center_x, center_y, aspect_ratio, height)
- `c_i ∈ {1, 2}` — class (1=car, 2=truck)

### 5.2 Pipeline collect_detections()

```python
def collect_detections(model, frame, conf_thres, imgsz, eroi_polygon):
    result = model.predict(frame, conf=conf_thres, imgsz=imgsz)
    # 1. Chạy YOLO inference
    # 2. Filter class: chỉ giữ car và truck
    # 3. eROI gate: loại bbox không overlap eROI (test 3 điều kiện)
    # 4. Tính histogram 48-dim cho mỗi bbox
    return detections
```

### 5.3 eROI Gate — Tiêu chí overlap hỗn hợp

Detection chỉ được giữ lại khi thỏa mãn **ít nhất 1 trong 3** điều kiện:

```
Điều kiện 1: Center point trong polygon eROI
Điều kiện 2: Ít nhất 1 trong 4 góc bbox trong polygon eROI
Điều kiện 3: Grid 5×5 sampling → overlap ratio ≥ 15%
```

Cách 3 đảm bảo xe tải có bbox lớn không bị loại bỏ oan khi vào/ra ROI.

### 5.4 Histogram Feature

Mỗi detection được gán vector đặc trưng màu 48 chiều:

```
H(d_i) ∈ ℝ⁴⁸ = [H_R(16-bin), H_G(16-bin), H_B(16-bin)]
```

Mỗi kênh được chuẩn hóa L₁ (tổng = 1). Dùng cho Stage 3 của matching.

---

## 6. Module Theo dõi (Tracking) — Kỹ thuật toán học

### 6.1 Không gian trạng thái Kalman 8D

Mỗi tracklet được biểu diễn bằng vector trạng thái 8 chiều:

```
x = [x, y, a, h, ẋ, ẏ, ȧ, ḣ]ᵀ
```

| Chiều | Biến | Mô tả |
|---|---|---|
| 0 | x | Tọa độ x tâm bounding box |
| 1 | y | Tọa độ y tâm bounding box |
| 2 | a | Aspect ratio (width/height) |
| 3 | h | Chiều cao bounding box |
| 4 | ẋ | Vận tốc x |
| 5 | ẏ | Vận tốc y |
| 6 | ȧ | Tốc độ thay đổi aspect ratio |
| 7 | ḣ | Tốc độ thay đổi chiều cao |

**Mô hình vận tốc tuyến tính hằng số (Constant Velocity Model):**
```
x(t+1) = x(t) + Δt · ẋ(t)    với Δt = 1 frame
```

### 6.2 Ma trận hệ thống

**Ma trận chuyển trạng thái F (8×8):**

```
F = | I₄  Δt·I₄ |  =  | 1 0 0 0 1 0 0 0 |
    | 0₄    I₄  |     | 0 1 0 0 0 1 0 0 |
                       | 0 0 1 0 0 0 1 0 |
                       | 0 0 0 1 0 0 0 1 |
                       | 0 0 0 0 1 0 0 0 |
                       | 0 0 0 0 0 1 0 0 |
                       | 0 0 0 0 0 0 1 0 |
                       | 0 0 0 0 0 0 0 1 |
```

**Ma trận đo lường H (4×8):**

```
H = [I₄ | 0₄ₓ₄]  =  | 1 0 0 0  0 0 0 0 |
                      | 0 1 0 0  0 0 0 0 |
                      | 0 0 1 0  0 0 0 0 |
                      | 0 0 0 1  0 0 0 0 |
```

Chỉ quan sát được 4 thành phần vị trí `[x, y, a, h]`, không quan sát trực tiếp vận tốc.

**Ma trận nhiễu quá trình Q (8×8, diagonal):**

```
Q = diag(1, 1, 1, 1, 4, 4, 4, 4)
     ↑position noise  ↑velocity noise (lớn hơn vì không chắc chắn)
```

**Ma trận nhiễu đo lường R (4×4, diagonal):**

```
R = diag(10.0, 10.0, 0.05, 15.0)
        ↑x,y       ↑a    ↑h
```

- `R[a] = 0.05` nhỏ: aspect ratio ít biến động (xe không đổi dạng nhanh)
- `R[h] = 15.0` lớn: chiều cao thay đổi nhiều theo góc nhìn và khoảng cách

**Hiệp phương sai khởi tạo P₀ (8×8, diagonal) — cho track mới:**

```
P₀ = diag(20, 20, 1, 30, 50, 50, 5, 50)
      ↑vị trí      ↑vận tốc (không chắc chắn cao khi mới tạo)
```

### 6.3 Chu kỳ Kalman Filter

**Bước Predict** (mỗi frame, trước khi nhận detection):

```
x̂(t|t-1) = F · x(t-1|t-1)                 (state prediction)
P(t|t-1)  = F · P(t-1|t-1) · Fᵀ + Q        (covariance prediction)
```

**Bước Update** (khi track được khớp với detection):

```
ŷ    = z - H · x̂(t|t-1)                   (innovation/residual)
S    = H · P(t|t-1) · Hᵀ + R              (innovation covariance)
K    = P(t|t-1) · Hᵀ · S⁻¹               (Kalman gain)
x(t|t)  = x̂(t|t-1) + K · ŷ              (updated state)
P(t|t)  = (I - K · H) · P(t|t-1)         (updated covariance)
```

**Không được khớp** (missed detection): chỉ predict, không update; tăng `missed` counter.

### 6.4 Vòng đời Track

```
                    ┌─────────────────┐
  New detection ───►│  Tentative      │ hits=1,2
                    │  (chưa confirm) │
                    └────────┬────────┘
                 hits≥min_hits=3 ▼
                    ┌─────────────────┐
                    │   Confirmed     │◄── được đếm
                    │   (active)      │
                    └────────┬────────┘
             missed>max_missed=35 ▼
                    ┌─────────────────┐
                    │    Deleted      │── recently_removed (flush đếm)
                    └─────────────────┘
```

### 6.5 Cơ chế liên kết dữ liệu ba bước (Three-Stage Hungarian)

Mỗi frame, thuật toán chạy 3 vòng Hungarian matching tuần tự. Đầu vào mỗi vòng là tập track và detection **chưa được khớp** từ vòng trước.

**Ràng buộc chung:** Track và detection chỉ được xét khớp nếu **cùng class** (car↔car, truck↔truck).

---

#### Stage 1 — IoU Matching

**Ma trận chi phí:**
```
cost_IoU(i,j) = 1 - IoU(bbox_track_i, bbox_det_j)
```

**IoU công thức:**
```
IoU(A, B) = |A ∩ B| / |A ∪ B|
```

**Ngưỡng:** Chỉ chấp nhận cặp nếu `IoU ≥ 0.1`, tức `cost ≤ 0.9`.

**Lý do dùng IoU trước:** Phần lớn xe di chuyển đủ chậm để bbox liên tiếp chồng lấn. IoU nhanh và hiệu quả cho trường hợp phổ biến này.

---

#### Stage 2 — Mahalanobis Distance Matching

**Áp dụng cho:** Track và detection không khớp được bằng IoU (xe di chuyển nhanh, bbox không overlap).

**Công thức:**
```
d²(i,j) = (z_j - Hμ_i)ᵀ · S_i⁻¹ · (z_j - Hμ_i)
```

Trong đó:
- `z_j = [x, y, a, h]ᵀ` của detection j (bbox_to_xyah conversion)
- `Hμ_i` = projected mean của track i từ Kalman state
- `S_i = H·P_i·Hᵀ + R` = innovation covariance (đo lường uncertainty)

**Ngưỡng:** `d² ≤ 16.0` (tương đương ~4σ, chi-squared với 4 bậc tự do ở mức 99.97%).

**Ưu điểm:** Khoảng cách Mahalanobis có trọng số theo uncertainty của Kalman, nên track đang uncertain (P lớn) sẽ chấp nhận detection xa hơn track đang ổn định.

---

#### Stage 3 — Visual Histogram Matching

**Áp dụng cho:** Track và detection vẫn chưa khớp sau 2 stage.

**Điều kiện kép (AND):**
```
Điều kiện A: ‖center_track - center_det‖₂ ≤ 50 pixel
Điều kiện B: ‖H(d_i) - H(μ_j)‖₁ ≤ 0.45
```

**Công thức L₁ histogram:**
```
cost_hist(i,j) = ‖H(d_i) - H(μ_j)‖₁ = mean(|H_d - H_t|)  (48-dim vector)
```

Điều kiện A được kiểm tra trước để tránh tính histogram tốn kém cho cặp ở xa. Điều kiện B đo độ tương đồng màu sắc ngoại hình (appearance similarity).

---

**Bảng tổng hợp tham số tracker:**

| Tham số | Giá trị |
|---|---|
| `iou_threshold` | 0.1 |
| `mahalanobis_threshold` (d²) | 16.0 |
| `hist_dist_threshold` (L₁) | 0.45 |
| `center_dist_threshold` | 50 px |
| `min_hits` | 3 |
| `max_missed` | 35 |
| Q — vị trí (x,y,a,h) | 1.0 |
| Q — vận tốc (ẋ,ẏ,ȧ,ḣ) | 4.0 |
| R — tọa độ (x,y) | 10.0 |
| R — aspect ratio (a) | 0.05 |
| R — chiều cao (h) | 15.0 |

---

## 7. Module Đếm (Counting)

### 7.1 Các loại vùng quan sát

**eROI (Extended ROI):** Đa giác mở rộng từ ROI gốc để giảm miss khi xe ở vùng biên. Dùng làm vùng filter detection đầu vào.

**ROI (Region of Interest):** Đa giác gốc định nghĩa vùng đếm thực sự. Phương tiện được ghi nhận khi rời khỏi vùng này.

**iROI (Illegal ROI):** Đa giác đánh dấu vùng di chuyển không hợp lệ (ví dụ: làn ngược chiều). Tracklet đi qua iROI bị đánh dấu `is_illegal=True` và bị loại khỏi kết quả.

### 7.2 Tiêu chí kiểm tra phương tiện trong ROI

Ba điều kiện — thỏa mãn **ít nhất 1**:

```python
# Điều kiện 1: tâm bbox trong polygon
if point_in_polygon(center, polygon): return True

# Điều kiện 2: ít nhất 1 góc trong polygon
corners = [(x1,y1), (x1,y2), (x2,y1), (x2,y2)]
if any(point_in_polygon(p, polygon) for p in corners): return True

# Điều kiện 3: grid sampling 5×5 → overlap ≥ 15%
if bbox_roi_overlap_ratio(bbox, polygon, grid=5) >= 0.15: return True
```

### 7.3 Gán MOI (Movement-of-Interest Assignment)

Mỗi MOI `m` được định nghĩa bằng cặp điểm `(S_m, E_m)` biểu diễn hướng di chuyển.

**Trích xuất vector quỹ đạo:**

```
Ŝ = history[⌊0.1N⌋]   (vị trí tại 10% đầu quỹ đạo)
Ê = history[⌊0.9N⌋]   (vị trí tại 90% đầu quỹ đạo)
tv = Ê - Ŝ             (vector hướng di chuyển)
```

**Hàm tính điểm khớp MOI:**

```
score(m) = 100 · θ_m + ‖Ŝ - S_m‖₂ + ‖Ê - E_m‖₂
```

Trong đó `θ_m ∈ [0, π]` (radian) là góc giữa `tv` và vector MOI `m`:

```
θ_m = arccos( (tv · mv_m) / (‖tv‖ · ‖mv_m‖) )
```

**Gán nhãn:** MOI có `score(m)` nhỏ nhất được chọn. Không có ngưỡng cắt bỏ — mọi tracklet hợp lệ đều nhận đúng một MOI.

*Hệ số 100 đặt trọng số góc lớn hơn khoảng cách, ưu tiên hướng đúng trước khi xét vị trí bắt đầu/kết thúc.*

**Fallback (khi không có MOI vectors):** Chia đều các góc `[0, 2π]` theo số MOI:

```python
angle = atan2(end.y - start.y, end.x - start.x)
movement_id = int(((angle + π) / (2π)) * movement_count) + 1
```

### 7.4 Cơ chế kích hoạt đếm

Hàm `count_track()` kiểm tra đủ 3 điều kiện trước khi ghi nhận:

```
1. tr.hits ≥ min_hits=3     (track đã được xác nhận)
2. tr.ever_inside_roi=True  (track từng vào ROI)
3. tr.is_illegal=False      (không đi qua iROI)
```

**Cơ chế chính — Kalman-exit trigger:**

```python
inside_roi = bbox_effectively_in_roi(tr.bbox, tr.center, roi_polygon)
if tr.hits >= min_hits and tr.ever_inside_roi and (not inside_roi):
    count_track(tr, ...)  # ghi nhận ngay khi Kalman predict ra ngoài ROI
```

Kể cả khi YOLO không detect được xe ở vùng biên, Kalman vẫn dự đoán vị trí và kích hoạt đếm.

**Cơ chế dự phòng 1 — recently_removed:**

```python
for tr in tracker.recently_removed:
    count_track(tr, ...)  # đếm track vừa bị xóa sau max_missed=35 frame
```

**Cơ chế dự phòng 2 — end-of-video flush:**

```python
for tr in tracker.tracks.values():
    count_track(tr, ...)  # flush tất cả track còn sống khi video kết thúc
```

Đảm bảo xe không bị bỏ sót khi video kết thúc đột ngột.

---

## 8. Cải tiến: Grounded SAM Bootstrap

Pipeline tự động sinh ROI/MOI chạy **offline một lần** trước pipeline DTC. Không ảnh hưởng đến latency real-time.

### 8.1 Nền tảng lý thuyết

**Grounding DINO [7]:** Mô hình zero-shot open-set object detection, kết hợp DINO visual encoder với BERT text encoder. Nhận text prompt và trả về bounding boxes của các vùng phù hợp trong ảnh.

**SAM — Segment Anything Model [8]:** Foundation model segmentation được train trên >1 tỷ mask. Nhận bounding box prompt → xuất pixel-level mask chính xác.

### 8.2 Pipeline 5 bước

```
Frame đầu video
       │
       ▼ Bước 1
┌─────────────────────────────────────┐
│  Grounding DINO inference           │
│  Prompt: "road surface . traffic    │
│           lane . intersection"      │
│  Threshold: 0.35 (cascade fallback) │
│  → Danh sách bounding boxes         │
└──────────────┬──────────────────────┘
               │ boxes
               ▼ Bước 2
┌─────────────────────────────────────┐
│  SAM Box Prompt segmentation        │
│  SAM(frame, bboxes=boxes)          │
│  → Pixel-level masks               │
└──────────────┬──────────────────────┘
               │ masks
               ▼ Bước 3
┌─────────────────────────────────────┐
│  Mask → eROI Polygon               │
│  • Union masks (MAX operator)      │
│  • Morphological closing 25×25     │
│  • Dilation 15×15                   │
│  • Lọc thành phần nhỏ              │
│  • Douglas-Peucker simplification  │
│  • Expand polygon 1.12x            │
└──────────────┬──────────────────────┘
               │ eROI polygon
               ▼ Bước 4
┌─────────────────────────────────────┐
│  PCA → MOI Vectors                 │
│  Với mỗi mask làn đường:           │
│  • Thu thập pixel coordinates       │
│  • SVD → principal direction        │
│  • p1, p2 = cực trị chiếu           │
│  K-Means++ phân cụm → k vectors     │
└──────────────┬──────────────────────┘
               │ MOI vectors
               ▼ Bước 5
┌─────────────────────────────────────┐
│  Xuất JSON + Overlay visualization  │
│  {"roi": [...], "moi_vectors": {...}}│
└─────────────────────────────────────┘
```

### 8.3 Chi tiết kỹ thuật từng bước

**Bước 1 — Fallback cascade khi Grounding DINO không tìm thấy:**

```python
candidate_prompts = [
    "road surface . traffic lane . intersection",  # prompt chính
    "road . traffic lane . intersection",           # fallback 1
    "road surface . lane markings . street",        # fallback 2
]
candidate_thresholds = [0.35, 0.25, 0.15, 0.05]   # cascade threshold
# Nếu vẫn thất bại: full-frame ROI + 0 MOI vectors
```

**Bước 3 — Hậu xử lý morphological:**

```python
# 1. Union tất cả masks
combined = MAX(all_masks)

# 2. Morphological closing: lấp khoảng trống trong vùng đường
cv2.morphologyEx(combined, MORPH_CLOSE, kernel_25x25)

# 3. Dilation: mở rộng nhẹ vùng phủ
cv2.dilate(combined, kernel_15x15)

# 4. Lọc thành phần nhỏ
min_area = max(600, H×W // 1800)  # adaptive threshold
# Giữ tối đa 6 thành phần lớn nhất

# 5. Contour → Polygon
cv2.findContours() → largest contour
cv2.approxPolyDP(epsilon=0.003 * arcLength)  # Douglas-Peucker

# 6. Expand 12% từ centroid
expand_polygon(poly, scale=1.12)
```

**Bước 4 — PCA bằng SVD:**

```python
pts = pixel_coordinates  # shape (N, 2)
mean = pts.mean(axis=0)
centered = pts - mean
U, S, Vt = np.linalg.svd(centered, full_matrices=False)
direction = Vt[0]          # trục chính (principal axis)
proj = centered @ direction
p1 = mean + direction * proj.min()  # điểm bắt đầu
p2 = mean + direction * proj.max()  # điểm kết thúc
```

**K-Means++ phân cụm:**

```python
k = min(moi_count, len(vectors))  # clamp theo số mask thực tế
cv2.kmeans(data, k, criteria, 8, KMEANS_PP_CENTERS)
# Sort theo góc phương vị để đánh số MOI ổn định
```

### 8.4 So sánh với cách tiếp cận thủ công

| Tiêu chí | Thủ công | Grounded SAM |
|---|---|---|
| Thời gian cấu hình | 30–60 phút/camera | < 2 phút/camera |
| Yêu cầu chuyên gia | Có (biết annotation tool) | Không |
| Cập nhật khi scene thay đổi | Vẽ lại thủ công | Chạy lại script |
| Phụ thuộc heuristic | Cao (diện tích, centroid) | Thấp (text prompt) |
| Độ chính xác MOI | Cao (do người hiểu scene) | Trung bình (phụ thuộc camera angle) |
| Phù hợp quy mô lớn | Khó mở rộng | Dễ batch processing |

### 8.5 Hạn chế đã biết

- **Single-frame limitation:** Grounded SAM chỉ xử lý frame đầu → K-Means bị clamp khi DINO tìm được ít region (ví dụ: cam_5 fisheye → 1 mask → 1 MOI). Giải pháp: dùng `sam_bootstrap.py` (multi-frame) làm fallback.
- **Góc camera thấp:** DINO khó phân biệt đường và vỉa hè khi camera gần mặt đất.
- **Điều kiện ánh sáng xấu:** Framem đầu có thể tối (bình minh) → DINO confidence thấp → kích hoạt cascade.

---

## 9. Bốn Baseline so sánh

### Baseline 1 — Manual ROI + MOI vectors chuẩn

**Phương pháp:** Dùng ROI thủ công của AI City và file `MOI_vectors/cam_5.txt` digitize từ ảnh `screen_shot_with_roi_and_movement/cam_5.jpg`. Nếu không truyền `--moi-vectors`, hệ thống mới rơi về angle fallback:

```python
angle = atan2(end.y - start.y, end.x - start.x)
movement_id = int(((angle + π) / (2π)) * movement_count) + 1
```

`movement_count` được đọc từ file `movement_description/cam_x.txt`.

**Ưu điểm:** Có cùng ngữ nghĩa `movement_id` với GT khi file MOI được digitize đúng.
**Nhược điểm:** Cần kiểm tra overlay thủ công vì cam fisheye làm các mũi tên cong khó biểu diễn bằng một vector thẳng.

### Baseline 2 — MOI Mining từ Trajectories

**Phương pháp:** Chạy YOLO + tracker trên N frames, lấy quỹ đạo xe trong ROI, cluster bằng K-Means++ → MOI vectors. Khi có `--moi-vectors` tham chiếu, script sẽ align MOI tự sinh về ID chính thức trước khi evaluate.

```
YOLO + CentroidTracker trên video
    │
    ▼ trajectories inside ROI
roi_entry_exit_vector() → vector [sx, sy, ex, ey]
    │
    ▼ K-Means++ (k = movement_count)
cluster_vectors() với median refinement
    │
    ▼ MOI vectors file (.txt)
```

**K-Means++ với Median Refinement:**

```python
criteria = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 200, 0.05)
_, labels, centers = cv2.kmeans(data, k, None, criteria, 12, KMEANS_PP_CENTERS)

# Refine: thay mean bằng median (robust to outliers)
for ci in range(k):
    members = data[labels == ci]
    if len(members) >= 3:
        centers[ci] = np.median(members, axis=0)
```

**Tham số lọc trajectory hợp lệ:**

```python
_MIN_MOVING_DISP  = 80.0   # net displacement tối thiểu (tránh xe đứng yên)
_MIN_PATH_LENGTH  = 130.0  # cumulative path length tối thiểu
_MIN_STRAIGHTNESS = 0.40   # displacement/path_length (lọc xe xoay vòng)
_TOP_MARGIN_FRAC  = 0.20   # bỏ qua 20% top frame (background xa)
```

### Baseline 3 — SAM Automatic (Không dùng Detector)

**Phương pháp:** Chạy nguyên bản Segment Anything Model (SAM) mà không cần text prompt hay detector nào để tự tìm đường.
- Ưu điểm: Khởi chạy cực nhanh, độc lập.
- Nhược điểm: Sai số lớn với các ngã tư phức tạp. Nếu SAM không tìm được road mask hoặc sinh ROI gần full-frame, output được đánh dấu `quality.status = low_confidence` và không nên đưa vào bảng kết quả chính.

### Baseline 4 — Grounding DINO + SAM Bootstrap

Xem chi tiết ở [Mục 8](#8-cải-tiến-grounded-sam-bootstrap). Chế độ này dùng Grounding DINO kết hợp SAM để gợi ý ROI/MOI. Nếu model Grounding DINO không có trong cache hoặc output low-confidence, dùng `--fallback-to-trajectory-sam` để chuyển sang trajectory-guided SAM và nhãn kết quả phải ghi rõ là fallback.

### Kết Quả Đánh Giá Tổng Thể (Cam_5_1min)

Kết quả local sau khi sửa evaluator, align MOI tự sinh, dùng `--class-conf car=0.25,truck=0.75`, và chạy `run_full_comparison.py --skip-sam-auto --skip-sam-yolo` trên `counting_example_cam_5_1min.mp4`:

| Phương Pháp (Baseline) | Tổng Số Xe Thực Tế (GT) | Số Đếm Hệ Thống | nwRMSE ↓ | S1 Overall ↑ | Count Accuracy | MAE (Sai số/nhóm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1:** Manual ROI + MOI chuẩn | 96 | 95 | 0.2879 | 0.4985* | 98.96% | 3.32 |
| **B2:** Manual ROI + Track-Mined MOI đã align | 96 | 95 | 0.4350 | 0.3955* | 98.96% | 5.35 |
| **B3:** SAM Automatic | 96 | N/A | N/A | N/A | N/A | N/A |
| **B4:** Grounding DINO + SAM | 96 | N/A | N/A | N/A | N/A | N/A |

*S1 hiện là local approximation vì efficiency chính thức cần Efficiency Base/script của AI City. B3/B4 không được điền bằng số cũ: SAM Automatic cũ rơi về full-frame ROI, còn B4 cũ là trajectory-guided SAM chứ không phải Grounding DINO + SAM.*

---

## 10. Độ đo đánh giá

### 10.1 Điểm tổng thể AI City S₁

```
S₁ = 0.3 × S₁_Efficiency + 0.7 × S₁_Effectiveness
```

### 10.2 S₁_Effectiveness — nwRMSE

Đo độ chính xác đếm qua **normalized weighted Root Mean Square Error**:

1. Chia mỗi video thành các đoạn thời gian
2. Tính số đếm tích lũy có trọng số từ đầu đến cuối mỗi đoạn
3. So sánh với ground truth
4. Normalize để các camera có số xe khác nhau có thể so sánh được

### 10.3 S₁_Efficiency — Điểm tốc độ

Đánh giá tốc độ thực thi điều chỉnh bởi Base Factor (năng lực phần cứng):

```
S₁_Eff = f(tổng_thời_gian_xử_lý / độ_dài_video, hardware_factor)
```

Điểm cao khi *processing time ≤ video duration* (real-time capable).

### 10.4 Metrics bổ sung

| Metric | Công thức | Áp dụng |
|---|---|---|
| **mAP@0.5** | Mean AP tại IoU=0.5 | Đánh giá YOLO detector |
| **mAP@0.5:0.95** | Mean AP các IoU threshold | Đánh giá YOLO detector |
| **MOTA** | 1 − (FP+FN+IDSW)/GT | Đánh giá tracker tổng thể |
| **ID Switches** | Số lần đổi ID của cùng 1 xe | Đánh giá consistency tracker |

---

## 11. Web Demo (Django)

### 11.1 Kiến trúc

```
web_demo/
├── traffic_demo/          ← Django project settings
│   ├── settings.py        ← MEDIA_ROOT, ALLOWED_HOSTS, etc.
│   └── urls.py            ← URL routing chính
└── counter/               ← Django app
    ├── views.py           ← Business logic, _execute_run() background thread
    ├── forms.py           ← ManualForm, AutoForm
    ├── urls.py            ← /manual/ /auto/ /status/ /result/ /history/
    └── templates/counter/ ← HTML templates
        ├── index.html     ← Dashboard
        ├── manual.html    ← Canvas ROI/MOI drawing + upload
        ├── auto.html      ← Auto SAM mode
        └── history.html   ← Run history
```

### 11.2 Luồng xử lý

```
User submit form
      │
      ▼
_launch_run()
  ├── Validate form
  ├── Save uploaded files to MEDIA_ROOT/<timestamp>/
  ├── Compute moi_count_hint từ canvas JSON hoặc uploaded file
  └── Spawn background thread → _execute_run()
      │
      ▼ (background thread)
_execute_run()
  ├── [Auto mode] Grounded SAM bootstrap → roi.txt, moi.txt
  ├── Build command: python run_dtc_counting.py ...
  ├── subprocess.run() với stdout/stderr capture
  ├── Parse logs → _reg_log(), _reg_pct(), _reg_step()
  └── Save result_ctx → _reg_done()
      │
      ▼
User polls /status/<run_id>/  (AJAX every 2s)
      ↓ done
User views /result/<run_id>/  ← bảng đếm, video, CSV download
```

### 11.3 Hai chế độ nhập liệu

**Manual Mode (`/manual/`):**
- Upload video + weights
- Vẽ **ROI polygon** trực tiếp trên canvas HTML5 (click → đỉnh đa giác)
- Vẽ **MOI bezier arrows** với control point (drag → mũi tên hướng di chuyển)
- Hoặc upload file ROI/MOI .txt sẵn có
- Canvas state được lưu vào `sessionStorage` để không mất khi submit form

**Auto Mode (`/auto/`):**
- Upload video + weights
- Tùy chọn upload movement description để gợi ý số MOI
- Cung cấp checkbox **Sử dụng Grounding-DINO**:
  - **Bật:** Hệ thống tự động chạy Grounded SAM → sinh ROI/MOI.
  - **Tắt:** Chạy chế độ SAM Automatic cực nhanh (chỉ dùng SAM).
- Fallback tự động khi các phương pháp nâng cao thất bại.

### 11.4 MOI count logic (3-branch)

```python
if moi_count_hint > 0:         # User vẽ N mũi tên trên canvas
    moi_count = moi_count_hint
elif movement_path:            # User upload movement description
    moi_count = _infer_movement_count_from_file(movement_path)
else:                          # Auto-detect
    moi_count = 0              # bootstrap tự xác định
```

### 11.5 Persistence & History

- Mỗi run được lưu metadata vào `<media_dir>/<run_id>/run_meta.json`
- Trang `/history/` scan toàn bộ MEDIA_ROOT để hiện danh sách các run
- Sau khi restart server, `_rebuild_result_from_fs()` đọc lại kết quả từ disk

---

## 12. Cài đặt môi trường

### 12.1 Yêu cầu hệ thống

| Thành phần | Yêu cầu tối thiểu | Khuyến nghị |
|---|---|---|
| Python | 3.9+ | 3.10 / 3.11 |
| RAM | 8 GB | 16 GB |
| GPU | Không bắt buộc | NVIDIA CUDA 11.8+ |
| VRAM | — | ≥ 6 GB (cho SAM) |
| Disk | 5 GB (không có data) | 50 GB (với AIC21 dataset) |

### 12.2 Cài đặt dependencies

```bash
# Clone repository
git clone https://github.com/wsunicorn/VEHICLE-COUNTING-SYSTEM-USING-DETECTION-TRACKING-COUNTING.git
cd VEHICLE-COUNTING-SYSTEM-USING-DETECTION-TRACKING-COUNTING

# Tạo virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Cài thư viện cốt lõi
pip install ultralytics opencv-python numpy scipy

# Cài cho Grounded SAM bootstrap
pip install transformers torch torchvision

# Cài web demo
pip install django

# (Optional) GPU support — xem pytorch.org để chọn đúng phiên bản CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**File `requirements.txt` đầy đủ:**

```
ultralytics>=8.0.0
opencv-python>=4.7.0
numpy>=1.24.0
scipy>=1.10.0
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
django>=4.2.0
```

### 12.3 SAM weights

```bash
cd dtc_counting
# SAM-Base checkpoint (~375MB) — tải tự động qua Ultralytics lần đầu chạy
# Hoặc tải thủ công:
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O sam_b.pt
```

### 12.4 Grounding DINO weights

Model `IDEA-Research/grounding-dino-base` được tải tự động từ HuggingFace Hub lần đầu chạy:

```python
# Tải lần đầu (cần kết nối internet):
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")
# Cache sẽ được lưu ở ~/.cache/huggingface/
```

---

## 13. Hướng dẫn sử dụng

### 13.1 Chạy pipeline DTC chính (CLI)

```bash
cd dtc_counting

python run_dtc_counting.py \
  --video data/AIC21_Track1_Vehicle_Counting_Bosung/counting_gt_sample/counting_example_cam_5_1min.mp4 \
  --weights ../best2.pt \
  --roi-file data/AIC21_Track1_Vehicle_Counting_Bosung/ROIs/cam_5.txt \
  --eroi-file data/AIC21_Track1_Vehicle_Counting_Bosung/ROIs/cam_5.txt \
  --movement-description data/AIC21_Track1_Vehicle_Counting_Bosung/movement_description/cam_5.txt \
  --video-clip-id 10 \
  --output-csv outputs/cam5_pred.csv \
  --save-video outputs/cam5_vis.mp4 \
  --imgsz 1280 \
  --conf 0.25
```

**Các tùy chọn quan trọng:**

| Tham số | Mô tả |
|---|---|
| `--moi-vectors` | File MOI vectors (id,x1,y1,x2,y2 per line) |
| `--iroi-file` | File iROI cho vùng di chuyển không hợp lệ |
| `--frame-stride 2` | Xử lý 1 trong 2 frame (tăng tốc 2x) |
| `--max-frames 1800` | Giới hạn số frame (debug/preview nhanh) |
| `--show` | Hiển thị visualization real-time |
| `--save-video` | Lưu video đã vẽ bounding box và counter |

### 13.2 Chạy Grounded SAM Bootstrap (CLI)

```bash
python grounded_sam_bootstrap.py \
  --video data/.../cam_5.mp4 \
  --output-json outputs/cam5_grounded.json \
  --save-overlay outputs/cam5_overlay.jpg \
  --moi-count 12 \
  --box-threshold 0.35 \
  --sam-model sam_b.pt
```

### 13.3 Chạy 4 Baseline Đánh Giá

```bash
cd dtc_counting

# Chạy đầy đủ 4 baselines và xuất báo cáo
python run_full_comparison.py \
  --video data/AIC21_Track1_Vehicle_Counting/counting_gt_sample/counting_example_cam_5_1min.mp4 \
  --roi-file data/AIC21_Track1_Vehicle_Counting/ROIs/cam_5.txt \
  --movement-description data/AIC21_Track1_Vehicle_Counting/movement_description/cam_5.txt \
  --gt-csv data/AIC21_Track1_Vehicle_Counting/counting_gt_sample/counting_example_cam_5_1min.csv \
  --moi-vectors data/AIC21_Track1_Vehicle_Counting/MOI_vectors/cam_5.txt
```

**Output trong `outputs/comparison/`:**

```
comparison_summary.json           ← so sánh số đếm 4 baseline vs ground truth
baseline1_angle_fallback.csv    ← kết quả baseline 1
baseline2_moi_from_tracks.txt   ← MOI vectors do baseline 2 sinh ra
baseline2_track_moi.csv         ← kết quả đếm baseline 2
baseline3_grounded_sam_moi.csv  ← kết quả đếm baseline 3
```

### 13.4 Chạy Web Demo

```bash
cd dtc_counting/web_demo

# Khởi tạo database (chạy 1 lần)
python manage.py migrate

# Khởi động server
python manage.py runserver

# Mở trình duyệt: http://127.0.0.1:8000/
```

**Các URL chính:**

| URL | Chức năng |
|---|---|
| `/` | Dashboard |
| `/manual/` | Nhập thủ công + canvas vẽ ROI/MOI |
| `/auto/` | Tự động SAM bootstrap |
| `/history/` | Lịch sử các lần chạy |
| `/status/<run_id>/` | Polling status (JSON API) |
| `/result/<run_id>/` | Xem kết quả đếm + download CSV |

### 13.5 Format file ROI và MOI

**File ROI (.txt):** Mỗi dòng là một đỉnh đa giác `x,y`

```
920,540
1200,540
1350,720
1100,900
750,900
600,720
```

**File MOI vectors (.txt):** Mỗi dòng là một vector `id,x1,y1,x2,y2`

```
# MOI vectors cho cam_5 (12 movements)
1,850,300,1100,700
2,950,280,950,750
3,900,310,700,680
...
12,400,650,900,400
```

---

## 14. Cấu trúc dự án

```
Project/
├── README.md                          ← Tài liệu này
├── .gitignore
├── best.pt                            ← YOLO weights (model đầu)
├── best2.pt                           ← YOLO weights (model chính)
│
└── dtc_counting/
    ├── README.md                      ← Hướng dẫn chạy nhanh
    ├── run_dtc_counting.py            ← Pipeline DTC chính
    │                                     MultiStepTracker: Kalman + Hungarian
    │                                     collect_detections, draw_frame, count_track
    ├── run_three_baselines.py         ← Chạy và so sánh 3 baseline
    ├── build_moi_from_tracks.py       ← Baseline 2: mine MOI từ trajectory
    ├── grounded_sam_bootstrap.py      ← Baseline 3: Grounded DINO + SAM
    ├── sam_bootstrap.py               ← Fallback bootstrap (multi-frame SAM)
    ├── sam_b.pt                       ← SAM-Base checkpoint
    │
    ├── data/                          ← [gitignored] AIC21 dataset
    │   ├── AIC21_Track1_Vehicle_Counting/
    │   │   ├── ROIs/                  ← cam_1.txt ... cam_20.txt
    │   │   └── movement_description/  ← cam_1.txt ... cam_20.txt
    │   └── AIC21_Track1_Vehicle_Counting_Bosung/
    │       ├── ROIs/
    │       ├── movement_description/
    │       └── counting_gt_sample/
    │
    ├── outputs/                       ← [gitignored] kết quả sinh ra
    │   ├── baselines/
    │   └── web_demo/
    │
    └── web_demo/                      ← Django application
        ├── manage.py
        ├── db.sqlite3                 ← [gitignored]
        ├── traffic_demo/              ← Django project
        │   ├── settings.py
        │   ├── urls.py
        │   ├── wsgi.py
        │   └── asgi.py
        ├── counter/                   ← Django app
        │   ├── views.py               ← _execute_run(), _launch_run()
        │   ├── forms.py               ← ManualForm, AutoForm
        │   ├── urls.py
        │   ├── apps.py
        │   └── templates/counter/
        │       ├── index.html         ← Dashboard
        │       ├── manual.html        ← Canvas ROI/MOI
        │       ├── auto.html          ← Auto SAM mode
        │       └── history.html       ← Run history
        └── media/                     ← [gitignored] uploaded files & outputs
```

---

## 15. Tài liệu tham khảo

```
[1] S. V. Ha, N. M. Chung, T. C. Nguyen, and H. N. Phan,
    "Tiny-PIRATE: A Tiny model with Parallelized Intelligence for Real-time
    Analysis as a Traffic countEr," in Proc. IEEE/CVF CVPRW, 2021.

[2] A. Bewley, Z. Ge, L. Ott, F. Ramos, and B. Upcroft,
    "Simple online and realtime tracking," in Proc. IEEE ICIP, 2016.

[3] A. Bochkovskiy, C. Y. Wang, and H. Y. M. Liao,
    "YOLOv4: Optimal speed and accuracy of object detection,"
    arXiv:2004.10934, 2020.

[4] R. E. Kalman,
    "A New Approach to Linear Filtering and Prediction Problems,"
    J. Basic Engineering, vol. 82, no. 1, pp. 35–45, 1960.

[5] M. Naphade et al.,
    "The 5th AI City Challenge," in Proc. IEEE/CVF CVPRW, 2021.

[6] N. Wojke, A. Bewley, and D. Paulus,
    "Simple online and realtime tracking with a deep association metric,"
    in Proc. IEEE ICIP, 2017.

[7] S. Liu et al.,
    "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set
    Object Detection," in Proc. ECCV, 2024.

[8] A. Kirillov et al.,
    "Segment Anything," in Proc. IEEE/CVF ICCV, 2023.

[9] Y. Zhang et al.,
    "ByteTrack: Multi-Object Tracking by Associating Every Detection Box,"
    in Proc. ECCV, 2022.

[10] A. Ravi et al.,
     "SAM 2: Segment Anything in Images and Videos,"
     arXiv:2408.00714, 2024.
```

---

<div align="center">

**Nhóm 19 — Khoa CNTT, Trường Đại học Công nghiệp TP.HCM**

[GitHub Repository](https://github.com/wsunicorn/VEHICLE-COUNTING-SYSTEM-USING-DETECTION-TRACKING-COUNTING)

</div>
