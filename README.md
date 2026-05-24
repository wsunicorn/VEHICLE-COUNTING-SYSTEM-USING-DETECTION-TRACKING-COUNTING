# Vehicle Counting System Using Detection, Tracking, ROI and MOI

Đây là hệ thống đếm phương tiện giao thông theo từng hướng di chuyển trong video camera. Hệ thống đi theo kiến trúc **Detection - Tracking - Counting (DTC)**:

1. Phát hiện xe trong từng frame bằng YOLO.
2. Theo dõi xe qua thời gian bằng Kalman Filter và Hungarian Matching.
3. Dùng ROI/MOI để xác định xe có đi vào vùng cần đếm không và thuộc hướng chuyển động nào.
4. Xuất CSV, video overlay, bảng thống kê và điểm đánh giá.

Project cũng có web demo Django để người dùng vẽ ROI/MOI thủ công hoặc khởi tạo tự động bằng SAM / Grounding DINO + SAM.

---

## Mục Lục

1. [Bài toán cần giải quyết](#1-bài-toán-cần-giải-quyết)
2. [Tổng quan pipeline](#2-tổng-quan-pipeline)
3. [Các khái niệm cốt lõi](#3-các-khái-niệm-cốt-lõi)
4. [Công nghệ được sử dụng](#4-công-nghệ-được-sử-dụng)
5. [Module 1 - Phát hiện phương tiện](#5-module-1---phát-hiện-phương-tiện)
6. [Module 2 - Theo dõi đa đối tượng](#6-module-2---theo-dõi-đa-đối-tượng)
7. [Module 3 - Gán MOI và đếm xe](#7-module-3---gán-moi-và-đếm-xe)
8. [Bootstrap ROI/MOI bằng SAM](#8-bootstrap-roimoi-bằng-sam)
9. [Bốn baseline B1-B4](#9-bốn-baseline-b1-b4)
10. [Đánh giá và công thức](#10-đánh-giá-và-công-thức)
11. [Kết quả hiện tại](#11-kết-quả-hiện-tại)
12. [Web demo Django](#12-web-demo-django)
13. [Cách cài đặt và chạy](#13-cách-cài-đặt-và-chạy)
14. [Cấu trúc thư mục](#14-cấu-trúc-thư-mục)
15. [Cách trình bày hệ thống cho người khác](#15-cách-trình-bày-hệ-thống-cho-người-khác)

---

## 1. Bài Toán Cần Giải Quyết

Input của hệ thống là video giao thông từ camera cố định. Output cần có là danh sách xe được đếm theo:

- `video_clip_id`: ID video.
- `frame_id`: frame tại đó xe được ghi nhận.
- `movement_id`: hướng di chuyển/MOI.
- `vehicle_class_id`: loại xe, với `1 = car`, `2 = truck`.

Ví dụ output CSV:

```csv
video_clip_id,frame_id,movement_id,vehicle_class_id
10,145,3,1
10,302,7,2
```

Có thể hiểu dòng đầu tiên là: ở video 10, tại frame 145, hệ thống đếm được một xe hơi đi theo movement 3.

Mục tiêu không chỉ là đếm tổng số xe. Hệ thống phải đếm đúng theo **loại xe** và **hướng di chuyển**. Vì vậy bài toán khó hơn object detection thông thường: cần phát hiện, theo dõi quỹ đạo, gán hướng, rồi mới đếm.

---

## 2. Tổng Quan Pipeline

Pipeline tổng quát:

```text
Video + ROI/MOI config
        |
        v
YOLO detector
        |
        v
Detections: bbox, class, confidence
        |
        v
Kalman + Hungarian tracker
        |
        v
Tracklets / trajectories
        |
        v
ROI gate + MOI assignment
        |
        v
Counting CSV + overlay video + metrics
```

Trong project, pipeline này nằm chủ yếu ở:

- `dtc_counting/run_dtc_counting.py`: script đếm xe chính.
- `dtc_counting/run_full_comparison.py`: chạy và đánh giá B1-B4.
- `dtc_counting/evaluate_counting.py`: tính metric.
- `dtc_counting/web_demo/`: giao diện demo bằng Django.

Ảnh sơ đồ tổng quan nằm tại:

```text
docs/workflow_diagram.png
```

---

## 3. Các Khái Niệm Cốt Lõi

### 3.1 ROI - Region of Interest

ROI là đa giác định nghĩa vùng đường cần quan sát. Xe nằm ngoài ROI sẽ không được đếm.

File ROI có dạng:

```text
x1,y1
x2,y2
x3,y3
...
```

Ví dụ:

```text
116,25
116,303
816,338
816,220
```

Trong code:

- `--roi-file`: vùng đếm chính.
- `--eroi-file`: extended ROI, vùng chấp nhận detection. Nếu không truyền thì dùng ROI.
- `--iroi-file`: illegal ROI, vùng hoặc đường đi không hợp lệ để loại track sai.

### 3.2 MOI - Movement of Interest

MOI là vector mô tả một hướng di chuyển cần đếm. Mỗi MOI có ID riêng.

File MOI có dạng:

```text
movement_id,x1,y1,x2,y2
```

Ví dụ:

```text
1,850,300,1100,700
2,950,280,950,750
```

Có thể hiểu MOI 1 là một mũi tên từ `(850,300)` đến `(1100,700)`.

### 3.3 Tracklet

Tracklet là quỹ đạo của một xe qua nhiều frame. Mỗi track gồm:

- ID track.
- Bounding box hiện tại.
- Class xe.
- Lịch sử tâm bbox qua các frame.
- Trạng thái đã đếm hay chưa.
- Cờ đánh dấu track từng nằm trong ROI hay chưa.

Mục tiêu của tracker là giữ ID ổn định cho cùng một xe, kể cả khi detector bị mất xe trong vài frame.

---

## 4. Công Nghệ Được Sử Dụng

| Thành phần | Công nghệ | Vai trò |
|---|---|---|
| Detector | YOLO / Ultralytics | Phát hiện car/truck trong từng frame |
| Xử lý ảnh | OpenCV | Đọc video, vẽ overlay, xử lý polygon/histogram |
| Tính toán | NumPy | Vector, ma trận, Kalman Filter |
| Matching | SciPy Hungarian | Tối ưu ghép detection với track |
| Tracker | Kalman Filter + Hungarian | Nối detection thành quỹ đạo |
| ROI/MOI tự động | SAM | Phân đoạn mặt đường/ROI |
| Prompt grounding | Grounding DINO | Gợi ý vùng road/lane/intersection bằng ngôn ngữ |
| Web demo | Django | Upload video, vẽ ROI/MOI, chạy pipeline và xem kết quả |

---

## 5. Module 1 - Phát Hiện Phương Tiện

Script chính:

```text
dtc_counting/run_dtc_counting.py
```

Hàm quan trọng:

```text
collect_detections()
```

### 5.1 YOLO detector

YOLO nhận một frame `I_t` và trả về danh sách bounding box:

```text
d_i = (x1, y1, x2, y2, class, confidence)
```

Hệ thống chỉ giữ lại hai lớp:

| YOLO label | AI City class ID | Ý nghĩa |
|---|---:|---|
| car | 1 | Xe hơi |
| truck | 2 | Xe tải |

### 5.2 Lọc theo confidence

Hệ thống hỗ trợ ngưỡng riêng theo từng lớp:

```powershell
--class-conf car=0.25,truck=0.75
```

Lý do: truck dễ bị nhầm hơn car trong một số camera. Đặt ngưỡng truck cao hơn giúp giảm false positive.

### 5.3 Lọc theo ROI

Sau khi YOLO phát hiện, hệ thống tính tâm bbox:

```text
cx = (x1 + x2) / 2
cy = (y1 + y2) / 2
```

Detection chỉ được giữ lại nếu bbox hoặc tâm bbox nằm trong eROI/ROI. Bước này giúp bỏ bớt xe ở vùng nền, bãi đỗ xe, vỉa hè hoặc lane không cần đếm.

---

## 6. Module 2 - Theo Dõi Đa Đối Tượng

Script chính:

```text
MultiStepTracker trong run_dtc_counting.py
```

Tracker gồm hai ý tưởng:

1. **Kalman Filter** để dự đoán vị trí xe ở frame tiếp theo.
2. **Hungarian Matching** để ghép detection mới với track cũ.

### 6.1 Trạng thái Kalman 8 chiều

Mỗi track có vector trạng thái:

```text
x = [cx, cy, a, h, vcx, vcy, va, vh]^T
```

Trong đó:

- `cx, cy`: tâm bbox.
- `a`: tỉ lệ rộng/cao, `a = w / h`.
- `h`: chiều cao bbox.
- `vcx, vcy, va, vh`: vận tốc của từng thành phần.

Mô hình chuyển động là constant velocity:

```text
x'_t = F x_{t-1}
P'_t = F P_{t-1} F^T + Q
```

Khi có detection mới `z_t`, Kalman update:

```text
y_t = z_t - H x'_t
S_t = H P'_t H^T + R
K_t = P'_t H^T S_t^-1
x_t = x'_t + K_t y_t
P_t = (I - K_t H) P'_t
```

Ý nghĩa:

- Prediction giúp track vẫn tiếp tục sống khi mất detection ngắn hạn.
- Update giúp track bám lại detection thật khi YOLO nhìn thấy xe.

### 6.2 Ba tầng matching

Hệ thống không ghép track-detection bằng một điều kiện duy nhất. Nó dùng ba tầng:

#### Tầng 1: IoU matching

IoU đo độ chồng lắp giữa bbox track và bbox detection:

```text
IoU(A, B) = area(A intersect B) / area(A union B)
```

Cost:

```text
cost = 1 - IoU
```

Track và detection được ghép nếu:

```text
IoU >= 0.1
```

Tầng này tốt khi xe di chuyển chậm và bbox liên tiếp còn chồng lắp.

#### Tầng 2: Mahalanobis matching

Khi bbox không còn chồng lắp, hệ thống dùng khoảng cách Mahalanobis:

```text
d^2 = (z - Hx)^T S^-1 (z - Hx)
```

Trong đó:

- `z`: detection mới.
- `Hx`: vị trí tracker dự đoán.
- `S`: covariance của dự đoán.

Ngưỡng trong code:

```text
d^2 <= 16.0
```

Tầng này tốt khi xe đi nhanh, bbox bị lệch nhưng vẫn nằm trong vùng dự đoán của Kalman.

#### Tầng 3: Histogram matching

Nếu hai tầng trên chưa ghép được, hệ thống so sánh histogram màu của crop bbox.

Khoảng cách histogram:

```text
D_hist = mean(|hist_track - hist_detection|)
```

Điều kiện:

```text
D_hist <= 0.45
center_distance <= 50 px
```

Tầng này giúp cứu một số trường hợp bbox lệch nhưng màu xe còn gần nhau.

### 6.3 Hungarian Matching

Ở mỗi tầng, ta có ma trận cost giữa track và detection. Hungarian tìm cách ghép sao cho tổng cost nhỏ nhất:

```text
min sum cost(track_i, detection_j)
```

Sau matching:

- Detection đã ghép -> update track.
- Detection chưa ghép -> tạo track mới.
- Track chưa ghép -> tăng `missed`.
- Track `missed > max_missed` -> xóa track và đưa sang bước đếm nếu đủ điều kiện.

---

## 7. Module 3 - Gán MOI Và Đếm Xe

Hàm chính:

```text
count_track()
assign_moi_by_vector()
```

### 7.1 Điều kiện để một track được đếm

Một track chỉ được đếm khi:

- Chưa từng được đếm.
- Không bị đánh dấu illegal.
- Có ít nhất `hits >= 3`.
- Từng đi vào ROI.
- Lịch sử quỹ đạo đủ dài: `--min-count-history`.
- Độ dịch chuyển đủ lớn: `--min-count-displacement`.

Các điều kiện này tránh đếm:

- Xe đứng yên.
- Track quá ngắn.
- Detection nhầm.
- Xe ở ngoài vùng đếm.

### 7.2 Gán movement bằng vector MOI

Với track có điểm đầu `S` và điểm cuối `E`:

```text
v_track = E - S
```

Với MOI có điểm đầu `M1` và điểm cuối `M2`:

```text
v_moi = M2 - M1
```

Góc lệch:

```text
angle = arccos( dot(v_track, v_moi) / (||v_track|| ||v_moi||) )
```

Sai lệch vị trí đầu/cuối:

```text
dist = ||S - M1|| + ||E - M2||
```

Điểm gán MOI:

```text
score = angle_weight * angle + distance_weight * dist
```

Trong code mặc định:

```text
angle_weight = 300.0
distance_weight = 0.35
```

MOI có `score` nhỏ nhất được chọn làm `movement_id`.

### 7.3 Fallback khi không có MOI vector

Nếu không có file MOI, hệ thống có thể gán movement bằng góc quỹ đạo:

```text
angle = atan2(E_y - S_y, E_x - S_x)
movement_id = bucket(angle, movement_count)
```

Tuy nhiên cách này kém chính xác hơn MOI vector, vì nó không biết hình học giao lộ.

---

## 8. Bootstrap ROI/MOI Bằng SAM

ROI/MOI có thể được nhập bằng tay, đọc từ file, hoặc khởi tạo tự động.

### 8.1 SAM Automatic

Script:

```text
dtc_counting/sam_auto_bootstrap.py
```

Quy trình:

1. Lấy một frame đại diện.
2. Chạy SAM automatic segmentation để sinh nhiều mask.
3. Chấm điểm mask nào giống mặt đường.
4. Loại mask có khả năng là cây cỏ/vegetation.
5. Hợp các mask tốt thành ROI polygon.
6. Lấy hướng chính của mask bằng PCA/KMeans để đề xuất MOI.

Trong thực tế, SAM từ một frame tĩnh có thể sinh quá ít MOI. Vì vậy pipeline có **quality gate**:

- ROI gần full-frame -> không tin cậy.
- Không có MOI -> không tin cậy.
- Quá ít MOI -> dùng fallback.
- Mask bị nghi là vegetation -> dùng color/trajectory fallback.

### 8.2 Grounding DINO + SAM

Script:

```text
dtc_counting/grounded_sam_bootstrap.py
```

Grounding DINO dùng prompt ngôn ngữ để tìm vùng đường:

```text
road surface . traffic lane . intersection
```

Sau đó SAM phân đoạn chính xác hơn quanh box của Grounding DINO.

Ý nghĩa:

- Grounding DINO giúp SAM biết cần tìm "road/lane/intersection".
- SAM biến box thành mask/ROI chi tiết hơn.

### 8.3 Track-mined MOI fallback

Script:

```text
dtc_counting/build_moi_from_tracks.py
```

Nếu SAM tạo ROI được nhưng MOI quá ít, hệ thống dùng quỹ đạo xe để tạo MOI:

1. Chạy YOLO + tracker trong một số frame.
2. Lấy các track có dịch chuyển đủ lớn.
3. Cắt phần quỹ đạo nằm trong ROI.
4. Lấy vector đầu-cuối của track.
5. Gom cụm vector bằng KMeans.
6. Align ID về MOI reference nếu có file MOI chuẩn.

Đây là lý do B3/B4 có thể được đưa vào bảng định lượng: ROI đến từ SAM/Grounding-SAM, còn MOI được hoàn thiện bằng trajectory.

---

## 9. Bốn Baseline B1-B4

Script:

```text
dtc_counting/run_full_comparison.py
```

| Baseline | Cấu hình | Ý nghĩa |
|---|---|---|
| B1 | Manual ROI + Official MOI | Mốc tham chiếu đáng tin cậy nhất |
| B2 | Manual ROI + Track-Mined MOI | Kiểm tra khả năng tự suy MOI từ trajectory |
| B3 | SAM Automatic ROI + Track-Mined MOI | Kiểm tra ROI từ SAM và MOI fallback từ trajectory |
| B4 | Grounding DINO + SAM ROI + Track-Mined MOI | Kiểm tra ROI có prompt ngôn ngữ và MOI fallback |

Tư duy so sánh:

- B1 trả lời: nếu ROI/MOI đúng thì detector + tracker + counter tốt đến đâu?
- B2 trả lời: có thể giảm công vẽ MOI bằng track mining không?
- B3 trả lời: SAM Automatic có khởi tạo ROI đủ dùng không?
- B4 trả lời: thêm prompt ngôn ngữ có giúp bootstrap ROI tốt hơn không?

---

## 10. Đánh Giá Và Công Thức

Script:

```text
dtc_counting/evaluate_counting.py
```

### 10.1 Count Accuracy

Đo sai lệch tổng số xe:

```text
CountAccuracy = max(0, 1 - |PredTotal - GTTotal| / max(1, GTTotal))
```

Chỉ số này dễ hiểu nhưng chưa đủ, vì tổng số đúng vẫn có thể sai hướng movement.

### 10.2 MAE theo movement/class

Với mỗi cặp `(movement_id, class_id)`:

```text
abs_error_k = |pred_count_k - gt_count_k|
MAE = mean(abs_error_k)
```

MAE thấp nghĩa là phân bổ theo hướng và loại xe ổn định hơn.

### 10.3 Weighted cumulative nwRMSE

Hệ thống chia video thành các segment thời gian. Ở mỗi segment, tính count tích lũy đến thời điểm đó.

Với một key `k = (video, movement, class)`:

```text
error_s = pred_cumulative_s - gt_cumulative_s
WRMSE_k = sqrt( sum(w_s * error_s^2) / sum(w_s) )
```

Trong code, segment về sau có trọng số lớn hơn:

```text
w_s = s + 1
```

Normalize theo tổng số xe thật của key:

```text
score_k = max(0, 1 - WRMSE_k / max(1, gt_count_k))
```

S1 Effectiveness là trung bình có trọng số theo số xe thật:

```text
S1_Effectiveness = sum(score_k * gt_count_k) / sum(gt_count_k)
```

Trong JSON output, `nwRMSE` được báo cáo như phần lỗi:

```text
nwRMSE = 1 - S1_Effectiveness
```

### 10.4 S1 Overall

Theo AI City, điểm tổng quát kết hợp effectiveness và efficiency:

```text
S1 = 0.7 * S1_Effectiveness + 0.3 * S1_Efficiency
```

Trong project hiện tại, `S1_Efficiency` là xấp xỉ local vì chưa dùng official efficiency base của leaderboard. Khi trình bày, nên nhấn mạnh:

- `nwRMSE`
- `S1_Effectiveness`
- `MAE`
- `Count Accuracy`

---

## 11. Kết Quả Hiện Tại

Kết quả chạy lại trên mẫu local có ground truth:

```text
dtc_counting/outputs/final_cam5_b1_b4_20260524_v4/comparison_summary.json
```

| Baseline | Pred/GT | nwRMSE | S1 Eff. | S1 Overall | Accuracy | MAE |
|---|---:|---:|---:|---:|---:|---:|
| B1 Manual ROI + Official MOI | 95/96 | 0.2879 | 0.7121 | 0.4985 | 98.96% | 3.32 |
| B2 Manual ROI + Track-Mined MOI | 95/96 | 0.4350 | 0.5650 | 0.3955 | 98.96% | 5.35 |
| B3 SAM Automatic ROI + Track-Mined MOI | 118/96 | 0.3906 | 0.6094 | 0.4266 | 77.08% | 4.67 |
| B4 Grounding DINO + SAM ROI + Track-Mined MOI | 125/96 | 0.4071 | 0.5929 | 0.4150 | 69.79% | 5.11 |

Cách đọc bảng:

- B1 tốt nhất về tổng count và là baseline đáng tin cậy.
- B2 cho thấy track-mined MOI khá đúng, nhưng phân bổ movement kém hơn B1.
- B3 đã có kết quả định lượng nhờ SAM ROI + MOI fallback từ trajectory.
- B4 có prompt Grounding DINO, nhưng vẫn cần hậu xử lý MOI để ổn định.

---

## 12. Web Demo Django

Thư mục:

```text
dtc_counting/web_demo/
```

Chạy server:

```powershell
cd dtc_counting/web_demo
python manage.py migrate
python manage.py runserver
```

Mở:

```text
http://127.0.0.1:8000/
```

### 12.1 Manual mode

Dùng khi demo ổn định:

- Upload video/weights hoặc dùng bộ demo có sẵn.
- Vẽ ROI trên canvas.
- Vẽ MOI bằng các mũi tên.
- Hoặc upload file ROI/MOI.
- Chạy `run_dtc_counting.py`.
- Xem CSV, video overlay, bảng thống kê.

### 12.2 Auto mode

Dùng khi muốn trình bày tính tự động:

- Upload video.
- Chọn SAM Automatic hoặc Grounding DINO + SAM.
- Hệ thống bootstrap ROI/MOI.
- Quality gate kiểm tra chất lượng.
- Nếu MOI quá ít, hệ thống có cảnh báo/fallback.

### 12.3 Lưu kết quả

Mỗi lần chạy web được lưu ở:

```text
dtc_counting/web_demo/media/<timestamp>/
```

Thư mục này có thể có:

- `counting_result.csv`
- `counting_vis.mp4`
- `bootstrap_grounded_sam.json`
- `bootstrap_overlay.jpg`
- `run_meta.json`

---

## 13. Cách Cài Đặt Và Chạy

### 13.1 Cài dependencies

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install ultralytics opencv-python numpy scipy django transformers torch torchvision
```

Nếu dùng GPU, cài PyTorch theo đúng bản CUDA từ trang PyTorch.

### 13.2 Chuẩn bị weights

YOLO weights nên đặt tại:

```text
weights/best2.pt
```

SAM checkpoint:

```text
dtc_counting/sam_b.pt
```

File `sam_b.pt` lớn nên không đưa lên GitHub. Có thể tải từ Segment Anything official checkpoint nếu cần.

### 13.3 Chạy pipeline chính

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

### 13.4 Chạy so sánh B1-B4

```powershell
cd dtc_counting

python run_full_comparison.py `
  --video data/AIC21_Track1_Vehicle_Counting/counting_gt_sample/counting_example_cam_5_1min.mp4 `
  --weights ..\weights\best2.pt `
  --roi-file data/AIC21_Track1_Vehicle_Counting/ROIs/cam_5.txt `
  --movement-description data/AIC21_Track1_Vehicle_Counting/movement_description/cam_5.txt `
  --moi-vectors data/AIC21_Track1_Vehicle_Counting/MOI_vectors/cam_5.txt `
  --gt-csv data/AIC21_Track1_Vehicle_Counting/counting_gt_sample/counting_example_cam_5_1min.csv `
  --class-conf car=0.25,truck=0.75 `
  --output-dir outputs/comparison
```

Output quan trọng:

```text
outputs/comparison/comparison_summary.json
outputs/comparison/b1_eval.json
outputs/comparison/b2_eval.json
outputs/comparison/b3_eval.json
outputs/comparison/b4_eval.json
```

---

## 14. Cấu Trúc Thư Mục

```text
Project/
+-- README.md
+-- docs/
|   +-- Nhom19_paper_ComputerVision.docx
|   +-- workflow_diagram.png
+-- weights/
|   +-- best.pt
|   +-- best2.pt
+-- dtc_counting/
    +-- run_dtc_counting.py
    +-- run_full_comparison.py
    +-- evaluate_counting.py
    +-- build_moi_from_tracks.py
    +-- moi_utils.py
    +-- sam_auto_bootstrap.py
    +-- grounded_sam_bootstrap.py
    +-- sam_bootstrap.py
    +-- data/
    |   +-- AIC21_Track1_Vehicle_Counting/
    |       +-- ROIs/
    |       +-- MOI_vectors/
    |       +-- movement_description/
    |       +-- counting_gt_sample/
    +-- outputs/
    +-- web_demo/
        +-- manage.py
        +-- traffic_demo/
        +-- counter/
```

Ý nghĩa các file quan trọng:

| File | Vai trò |
|---|---|
| `run_dtc_counting.py` | Pipeline đếm xe chính |
| `run_full_comparison.py` | Chạy B1-B4 và gom summary |
| `evaluate_counting.py` | Tính nwRMSE, S1, accuracy, MAE |
| `build_moi_from_tracks.py` | Tạo MOI từ trajectory |
| `moi_utils.py` | Load/write/align MOI vectors |
| `sam_auto_bootstrap.py` | ROI bootstrap bằng SAM Automatic |
| `grounded_sam_bootstrap.py` | ROI bootstrap bằng Grounding DINO + SAM |
| `sam_bootstrap.py` | Trajectory-guided SAM fallback |
| `web_demo/counter/views.py` | Điều phối web demo và background run |

---

## 15. Cách Trình Bày Hệ Thống Cho Người Khác

Khi thuyết trình, nên đi theo mạch sau:

### 15.1 Mở đầu bài toán

"Bài toán của nhóm là đếm xe theo từng hướng di chuyển trong video giao thông. Không chỉ cần nhận ra xe, hệ thống phải theo dõi xe qua thời gian và xác định xe đi theo movement nào."

### 15.2 Giải thích pipeline

"Hệ thống gồm ba tầng chính: YOLO để phát hiện xe, Kalman-Hungarian tracker để nối detection thành quỹ đạo, và ROI/MOI counter để gán hướng và đếm."

### 15.3 Giải thích vì sao cần ROI/MOI

"ROI giúp hệ thống biết vùng nào cần quan sát. MOI là các vector hướng di chuyển, giúp biến quỹ đạo xe thành movement ID. Nếu không có MOI, tổng số xe có thể đúng nhưng hướng di chuyển sẽ dễ sai."

### 15.4 Giải thích tracker

"Kalman Filter dự đoán vị trí tiếp theo của xe. Hungarian Matching giải bài toán ghép detection mới với track cũ sao cho tổng cost nhỏ nhất. Hệ thống ghép qua ba tầng: IoU, Mahalanobis và histogram màu."

### 15.5 Giải thích SAM

"SAM không dùng để đếm xe trực tiếp. SAM được dùng để khởi tạo ROI/MOI. Nếu SAM sinh ROI được nhưng MOI quá ít, hệ thống dùng trajectory của xe để bổ sung MOI. Như vậy SAM là công cụ giảm công cấu hình, còn detector/tracker/counter vẫn là lõi chính."

### 15.6 Giải thích kết quả

"B1 là mốc tham chiếu với ROI/MOI chuẩn. B2 kiểm tra MOI từ trajectory. B3 và B4 kiểm tra khả năng tự động hóa ROI bằng SAM/Grounding-SAM. Kết quả cho thấy pipeline đếm chính hoạt động tốt nhất khi ROI/MOI được kiểm chứng, và các nhánh SAM có thể đưa vào định lượng khi có quality gate và fallback."

---

## Ghi Chú Về Git

Các file lớn được ignore:

- `dtc_counting/outputs/`
- `dtc_counting/web_demo/media/`
- video `.mp4`, `.avi`, ...
- `dtc_counting/sam_b.pt`
- database local Django

Repo chỉ nên commit code, README, paper, workflow diagram và các file cấu hình/CSV mẫu nhỏ.
