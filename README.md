# Vehicle Counting System Using Detection, Tracking, ROI and MOI

Đây là tài liệu giải thích đầy đủ project đếm phương tiện giao thông của nhóm. Mục tiêu của README này không chỉ là hướng dẫn chạy code, mà còn giúp người đọc hiểu:

- **What**: hệ thống này là gì?
- **Why**: vì sao phải thiết kế như vậy?
- **How**: từng module hoạt động như thế nào?
- **Data**: dữ liệu được lấy, xử lý và gán nhãn ra sao?
- **Training**: YOLO được train/fine-tune như thế nào?
- **Metrics**: hệ thống được đánh giá bằng công thức nào?
- **Demo**: chạy CLI và web demo như thế nào?
- **Presentation**: nên trình bày project này với người khác ra sao?

---

## 1. Tóm Tắt Nhanh

Project xây dựng một hệ thống **Detection - Tracking - Counting (DTC)** để đếm xe theo từng hướng di chuyển trong video giao thông.

Pipeline cốt lõi:

```text
Video giao thông
      |
      v
YOLO vehicle detector
      |
      v
Kalman + Hungarian multi-object tracker
      |
      v
ROI/MOI movement assignment
      |
      v
Counting CSV + overlay video + metrics + web demo
```

Các công nghệ chính:

| Thành phần | Công nghệ | Vai trò |
|---|---|---|
| Object Detection | YOLO / Ultralytics | Phát hiện xe hơi và xe tải |
| Multi-object Tracking | Kalman Filter + Hungarian Matching | Nối detection thành quỹ đạo |
| Counting | ROI/MOI vector matching | Gán hướng di chuyển và đếm |
| Auto ROI/MOI | SAM, Grounding DINO + SAM | Hỗ trợ khởi tạo vùng đếm tự động |
| Evaluation | nwRMSE, S1 Effectiveness, MAE, Count Accuracy | Đánh giá kết quả |
| Web Demo | Django | Giao diện demo và chạy pipeline |

---

## 2. What - Hệ Thống Này Là Gì?

Hệ thống nhận đầu vào là một video giao thông từ camera cố định, sau đó xuất ra danh sách các xe được đếm theo:

- xe thuộc video nào,
- xe được ghi nhận tại frame nào,
- xe thuộc loại nào,
- xe đi theo hướng di chuyển nào.

Output chính là file CSV:

```csv
video_clip_id,frame_id,movement_id,vehicle_class_id
10,145,3,1
10,302,7,2
```

Ý nghĩa:

- `video_clip_id = 10`: ID video.
- `frame_id = 145`: xe được ghi nhận tại frame 145.
- `movement_id = 3`: xe đi theo hướng số 3.
- `vehicle_class_id = 1`: xe hơi. Trong project: `1 = car`, `2 = truck`.

Điểm quan trọng: hệ thống không chỉ đếm tổng số xe, mà đếm theo **loại xe** và **hướng di chuyển**. Đây là yêu cầu quan trọng trong bài toán traffic flow analysis và AI City Challenge Track 1.

---

## 3. Why - Vì Sao Cần Detection, Tracking, ROI Và MOI?

Nếu chỉ dùng object detection, ta chỉ biết trong từng frame có bao nhiêu xe. Nhưng một xe xuất hiện trong nhiều frame, nên nếu cộng detection theo frame thì sẽ đếm trùng rất nhiều lần.

Vì vậy cần thêm tracking:

```text
Detection từng frame -> Tracklet của từng xe -> Đếm mỗi xe một lần
```

Tuy nhiên tracking vẫn chưa đủ. Ta còn cần biết xe đi theo hướng nào trong giao lộ. Vì vậy cần:

- **ROI** để biết vùng nào là vùng cần quan sát.
- **MOI** để biết các hướng di chuyển hợp lệ trong giao lộ.

Nói ngắn gọn:

| Vấn đề | Thành phần xử lý |
|---|---|
| Xe nằm ở đâu trong frame? | YOLO detection |
| Xe này có phải cùng xe ở frame trước không? | Tracker |
| Xe có đi vào vùng cần đếm không? | ROI |
| Xe đi theo hướng nào? | MOI |
| Xe đã đủ điều kiện để ghi nhận chưa? | Counting logic |

---

## 4. How - Pipeline Tổng Thể

Luồng xử lý end-to-end:

```text
1. Input
   - Video giao thông
   - YOLO weights
   - ROI/MOI file hoặc bootstrap bằng SAM
   - Ground truth CSV nếu cần đánh giá

2. Detection
   - YOLO phát hiện car/truck
   - Lọc theo confidence
   - Lọc theo ROI/eROI

3. Tracking
   - Kalman dự đoán vị trí tiếp theo
   - Hungarian ghép detection với track
   - Dùng IoU, Mahalanobis, histogram để tăng độ ổn định

4. Counting
   - Kiểm tra track có từng vào ROI không
   - Loại track quá ngắn hoặc đứng yên
   - Gán movement bằng vector MOI
   - Ghi event vào CSV

5. Evaluation
   - So sánh CSV dự đoán với ground truth
   - Tính nwRMSE, S1 Effectiveness, Count Accuracy, MAE

6. Demo
   - CLI script
   - Django web demo
   - Video overlay và bảng kết quả
```

Sơ đồ tổng quan:

```text
docs/workflow_diagram.png
```

---

## 5. Dữ Liệu Và Gán Nhãn

### 5.1 Nguồn dữ liệu

Project dùng dữ liệu giao thông theo hướng bài toán AI City Challenge Track 1. Trong repo chỉ giữ các file nhỏ để demo và tái lập kết quả:

```text
dtc_counting/data/AIC21_Track1_Vehicle_Counting/
├── ROIs/cam_5.txt
├── MOI_vectors/cam_5.txt
├── movement_description/cam_5.txt
└── counting_gt_sample/counting_example_cam_5_1min.csv
```

Các video lớn, output, media web demo và model nặng được ignore để repo không vượt giới hạn GitHub.

### 5.2 Dữ liệu train YOLO

Để train detector, nhóm cần tạo dataset có nhãn bounding box cho các lớp:

- `car`
- `truck`

Quy trình xử lý dữ liệu:

```text
Video AI City
    |
    v
Trích xuất frame
    |
    v
Upload frame lên Roboflow
    |
    v
Gán nhãn car/truck bằng bounding box
    |
    v
Review nhãn
    |
    v
Train/validation split
    |
    v
Export YOLO format
    |
    v
Train YOLO
```

### 5.3 Trích xuất frame từ video

Có thể dùng OpenCV hoặc ffmpeg để trích frame. Ví dụ với ffmpeg:

```bash
ffmpeg -i input_video.mp4 -vf "select=not(mod(n\,5))" -vsync vfr frames/frame_%06d.jpg
```

Ý nghĩa:

- `mod(n,5)`: lấy 1 frame sau mỗi 5 frame.
- Giảm trùng lặp giữa các frame liền kề.
- Giúp dataset đa dạng hơn mà không quá lớn.

### 5.4 Gán nhãn bằng Roboflow

Roboflow được dùng để quản lý ảnh, gán nhãn và export dataset YOLO.

Quy trình gán nhãn:

1. Tạo project detection trên Roboflow.
2. Tạo hai class: `car`, `truck`.
3. Upload các frame đã trích xuất.
4. Vẽ bounding box quanh từng xe.
5. Review lại nhãn để tránh box lệch, thiếu xe hoặc sai class.
6. Chia train/validation.
7. Export dataset ở định dạng YOLO.

Quy tắc gán nhãn nên dùng:

| Trường hợp | Cách xử lý |
|---|---|
| Xe nhìn rõ | Gán box sát thân xe |
| Xe bị che một phần | Gán phần nhìn thấy nếu vẫn nhận diện được |
| Xe quá nhỏ hoặc mờ | Có thể bỏ qua nếu không đủ thông tin |
| Xe ngoài vùng giao thông chính | Tùy mục tiêu detector; nếu gây nhiễu khi đếm thì nên hạn chế |
| Car/truck khó phân biệt | Review lại bằng nhiều người |

Lưu ý quan trọng: YOLO học từ nhãn detector, còn ROI/MOI xử lý ở bước counting. Vì vậy khi train detector, ta tập trung vào việc nhận diện xe chính xác; khi đếm, ta mới dùng ROI/MOI để quyết định xe nào được ghi nhận.

### 5.5 Format YOLO sau khi export

Dataset YOLO thường có cấu trúc:

```text
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

Mỗi file label `.txt` có dạng:

```text
class_id center_x center_y width height
```

Các tọa độ được normalize về `[0, 1]`.

Ví dụ:

```text
0 0.5123 0.4211 0.0832 0.0715
1 0.7220 0.5330 0.1200 0.1000
```

Trong Roboflow/YOLO:

| Class | YOLO class index | AI City output ID |
|---|---:|---:|
| car | 0 | 1 |
| truck | 1 | 2 |

Trong code, hàm `class_to_aicity_id()` chuyển nhãn YOLO sang ID output của AI City.

---

## 6. Huấn Luyện YOLO Detector

### 6.1 Vì sao dùng YOLO?

YOLO phù hợp với bài toán này vì:

- inference nhanh,
- dễ fine-tune,
- có API Ultralytics thuận tiện,
- đủ tốt cho bài toán car/truck trong video giao thông,
- dễ tích hợp vào pipeline Python/OpenCV.

### 6.2 Transfer learning

Thay vì train từ đầu, ta dùng pretrained YOLO đã học đặc trưng thị giác chung, sau đó fine-tune trên dataset car/truck.

```text
COCO pretrained YOLO
        |
        v
Fine-tune trên frame giao thông đã gán nhãn
        |
        v
best.pt / best2.pt
```

### 6.3 Lệnh train tham khảo

Ví dụ train bằng Ultralytics:

```bash
yolo train \
  model=yolov8m.pt \
  data=data.yaml \
  epochs=100 \
  imgsz=1280 \
  batch=16 \
  device=0 \
  project=runs/train \
  name=vehicle_counter
```

Nếu máy yếu hơn:

```bash
yolo train model=yolov8n.pt data=data.yaml epochs=80 imgsz=960 batch=8
```

### 6.4 Augmentation

Các augmentation hữu ích:

| Augmentation | Ý nghĩa |
|---|---|
| Mosaic | Tăng đa dạng scale và background |
| HSV jitter | Mô phỏng thay đổi ánh sáng/thời tiết |
| Horizontal flip | Tăng dữ liệu cho các hướng giao thông đối xứng |
| Scale/translate | Giúp model chịu được thay đổi góc nhìn |

Không nên dùng rotation quá mạnh vì camera giao thông thường cố định, xe không xoay tùy ý như ảnh tự nhiên.

### 6.5 Metric khi train detector

Các metric detector:

| Metric | Công thức/ý nghĩa |
|---|---|
| Precision | `TP / (TP + FP)` |
| Recall | `TP / (TP + FN)` |
| F1 | `2PR / (P + R)` |
| mAP@0.5 | AP trung bình tại IoU threshold 0.5 |
| mAP@0.5:0.95 | AP trung bình qua nhiều threshold IoU |

Detector tốt chưa đảm bảo counter tốt. Một detector có mAP cao nhưng tracking/MOI sai thì kết quả đếm vẫn sai. Vì vậy project đánh giá cả pipeline bằng counting metrics.

### 6.6 Weights trong project

Weights chính:

```text
weights/best2.pt
```

Khi chạy web demo hoặc CLI, truyền:

```powershell
--weights ..\weights\best2.pt
```

---

## 7. Module Detection Trong Pipeline

Script chính:

```text
dtc_counting/run_dtc_counting.py
```

Hàm quan trọng:

```text
collect_detections()
```

YOLO nhận frame `I_t` và trả về:

```text
d_i = (x1, y1, x2, y2, class, confidence)
```

Sau đó hệ thống:

1. Chỉ giữ `car` và `truck`.
2. Chuyển class về AI City ID: `car -> 1`, `truck -> 2`.
3. Lọc confidence chung hoặc confidence riêng theo class.
4. Tính tâm bbox.
5. Lọc theo eROI/ROI.
6. Tính histogram màu để hỗ trợ tracking.

Ngưỡng thường dùng:

```powershell
--class-conf car=0.25,truck=0.75
```

---

## 8. Module Tracking

Tracker nằm trong class:

```text
MultiStepTracker
```

### 8.1 Trạng thái Kalman 8 chiều

Mỗi track có trạng thái:

```text
x = [cx, cy, a, h, vcx, vcy, va, vh]^T
```

Trong đó:

- `cx, cy`: tâm bbox.
- `a`: aspect ratio, `a = width / height`.
- `h`: chiều cao bbox.
- `vcx, vcy, va, vh`: vận tốc tương ứng.

Prediction:

```text
x'_t = F x_{t-1}
P'_t = F P_{t-1} F^T + Q
```

Update:

```text
y_t = z_t - Hx'_t
S_t = H P'_t H^T + R
K_t = P'_t H^T S_t^-1
x_t = x'_t + K_t y_t
P_t = (I - K_t H) P'_t
```

### 8.2 Hungarian Matching

Hungarian giải bài toán gán tối ưu:

```text
min sum cost(track_i, detection_j)
```

Project dùng ba tầng cost:

#### Tầng 1 - IoU

```text
IoU(A, B) = area(A ∩ B) / area(A ∪ B)
cost = 1 - IoU
```

Điều kiện:

```text
IoU >= 0.1
```

#### Tầng 2 - Mahalanobis

```text
d^2 = (z - Hx)^T S^-1 (z - Hx)
```

Điều kiện:

```text
d^2 <= 16.0
```

#### Tầng 3 - Histogram

```text
D_hist = mean(|hist_track - hist_detection|)
```

Điều kiện:

```text
D_hist <= 0.45
center_distance <= 50 px
```

### 8.3 Vòng đời track

Sau mỗi frame:

- Track matched -> Kalman update, thêm điểm vào history.
- Detection unmatched -> tạo track mới.
- Track unmatched -> tăng `missed`.
- Track `missed > max_missed` -> xóa và xét đếm nếu đủ điều kiện.

---

## 9. Module Counting: ROI/MOI Và Gán Hướng

### 9.1 Điều kiện để đếm một track

Track chỉ được đếm khi:

- chưa từng được đếm,
- không bị đánh dấu illegal,
- có `hits >= 3`,
- từng đi vào ROI,
- quỹ đạo đủ dài,
- độ dịch chuyển đủ lớn.

Các điều kiện này giúp tránh đếm xe đứng yên, track quá ngắn hoặc detection nhiễu.

### 9.2 Gán MOI bằng vector

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

Score:

```text
score = angle_weight * angle + distance_weight * dist
```

Mặc định:

```text
angle_weight = 300.0
distance_weight = 0.35
```

MOI có score nhỏ nhất được chọn làm `movement_id`.

### 9.3 Nếu không có MOI

Nếu không có file MOI, hệ thống fallback bằng góc quỹ đạo:

```text
angle = atan2(E_y - S_y, E_x - S_x)
movement_id = bucket(angle, movement_count)
```

Cách này kém chính xác hơn vì không biết hình học thực tế của giao lộ.

---

## 10. Bootstrap ROI/MOI Bằng SAM

ROI/MOI có thể:

- vẽ thủ công trên web,
- đọc từ file `.txt`,
- khởi tạo tự động bằng SAM/Grounding-SAM.

### 10.1 SAM Automatic

Script:

```text
dtc_counting/sam_auto_bootstrap.py
```

Quy trình:

1. Lấy frame đại diện.
2. Chạy SAM automatic segmentation.
3. Lọc mask giống mặt đường.
4. Loại mask có đặc trưng cây cỏ.
5. Hợp mask thành ROI polygon.
6. Suy MOI sơ bộ bằng PCA/KMeans.

### 10.2 Grounding DINO + SAM

Script:

```text
dtc_counting/grounded_sam_bootstrap.py
```

Prompt dùng để tìm vùng đường:

```text
road surface . traffic lane . intersection
```

Grounding DINO tìm box liên quan đến prompt. SAM dùng box đó để phân đoạn ROI.

### 10.3 Quality gate

Tự động bootstrap có thể sai. Vì vậy project dùng quality gate:

| Kiểm tra | Ý nghĩa |
|---|---|
| ROI gần full-frame | Có thể SAM lấy cả màn hình |
| ROI quá nhỏ/hẹp | Không đủ vùng đếm |
| MOI quá ít | Không đủ hướng movement |
| Mask giống vegetation | Có thể lấy nhầm cỏ/cây |

Nếu quality gate chưa đạt, hệ thống có thể dùng fallback.

### 10.4 Track-mined MOI fallback

Script:

```text
dtc_counting/build_moi_from_tracks.py
```

Ý tưởng: nếu SAM tạo ROI được nhưng MOI quá ít, dùng chính quỹ đạo xe để suy MOI.

Quy trình:

1. Chạy YOLO + tracker trong một số frame.
2. Lấy track có dịch chuyển đủ lớn.
3. Lấy đoạn quỹ đạo nằm trong ROI.
4. Biến mỗi track thành vector hướng.
5. Gom cụm vector bằng KMeans.
6. Align ID về MOI chuẩn nếu có file reference.

Nhờ vậy, B3/B4 có thể đưa vào bảng định lượng:

- B3: SAM Automatic ROI + Track-Mined MOI.
- B4: Grounding DINO + SAM ROI + Track-Mined MOI.

---

## 11. Bốn Baseline B1-B4

Script:

```text
dtc_counting/run_full_comparison.py
```

| Baseline | Cấu hình | Câu hỏi cần trả lời |
|---|---|---|
| B1 | Manual ROI + Official MOI | Nếu ROI/MOI đúng thì pipeline DTC tốt đến đâu? |
| B2 | Manual ROI + Track-Mined MOI | Có thể tự sinh MOI từ trajectory không? |
| B3 | SAM Automatic ROI + Track-Mined MOI | SAM Automatic có giúp khởi tạo ROI dùng được không? |
| B4 | Grounding DINO + SAM ROI + Track-Mined MOI | Prompt ngôn ngữ có hỗ trợ bootstrap ROI tốt hơn không? |

So sánh này giúp tách riêng ảnh hưởng của:

- chất lượng detector/tracker,
- chất lượng ROI,
- chất lượng MOI,
- mức độ tự động hóa.

---

## 12. Metrics Và Công Thức Đánh Giá

Script:

```text
dtc_counting/evaluate_counting.py
```

### 12.1 Count Accuracy

Đo sai lệch tổng số xe:

```text
CountAccuracy = max(0, 1 - |PredTotal - GTTotal| / max(1, GTTotal))
```

Chỉ số này dễ hiểu nhưng chưa đủ, vì tổng số đúng vẫn có thể sai movement.

### 12.2 MAE theo movement/class

Với mỗi key `k = (movement_id, class_id)`:

```text
abs_error_k = |pred_count_k - gt_count_k|
MAE = mean(abs_error_k)
```

MAE thấp nghĩa là hệ thống phân bổ xe theo hướng và loại xe tốt hơn.

### 12.3 Weighted cumulative nwRMSE

Video được chia thành các segment thời gian. Ở mỗi segment, hệ thống so sánh count tích lũy.

```text
error_s = pred_cumulative_s - gt_cumulative_s
WRMSE_k = sqrt( sum(w_s * error_s^2) / sum(w_s) )
```

Trong code:

```text
w_s = s + 1
```

Segment về sau có trọng số lớn hơn vì nó phản ánh sai số tích lũy cuối video.

Normalize:

```text
score_k = max(0, 1 - WRMSE_k / max(1, gt_count_k))
```

S1 Effectiveness:

```text
S1_Effectiveness = sum(score_k * gt_count_k) / sum(gt_count_k)
```

Trong JSON output:

```text
nwRMSE = 1 - S1_Effectiveness
```

### 12.4 S1 Overall

Theo hướng AI City:

```text
S1 = 0.7 * S1_Effectiveness + 0.3 * S1_Efficiency
```

Trong project hiện tại, `S1_Efficiency` là xấp xỉ local do chưa dùng official hardware base/script của leaderboard. Khi trình bày, nên xem các chỉ số sau là trọng tâm:

- `nwRMSE`
- `S1_Effectiveness`
- `MAE`
- `Count Accuracy`

### 12.5 Metric detector khác metric counting

Khi train YOLO, ta quan tâm:

- Precision
- Recall
- F1
- mAP@0.5
- mAP@0.5:0.95

Khi đánh giá toàn hệ thống, ta quan tâm:

- xe có được đếm đúng một lần không,
- xe có đúng class không,
- xe có đúng movement không,
- count theo thời gian có khớp ground truth không.

Đây là điểm nên nhấn mạnh khi trình bày: **detector tốt là điều kiện cần, nhưng chưa đủ để counting tốt**.

---

## 13. Kết Quả Hiện Tại

Kết quả chạy trên mẫu local có ground truth:

```text
dtc_counting/outputs/final_cam5_b1_b4_20260524_v4/comparison_summary.json
```

| Baseline | Pred/GT | nwRMSE | S1 Eff. | S1 Overall | Accuracy | MAE |
|---|---:|---:|---:|---:|---:|---:|
| B1 Manual ROI + Official MOI | 95/96 | 0.2879 | 0.7121 | 0.4985 | 98.96% | 3.32 |
| B2 Manual ROI + Track-Mined MOI | 95/96 | 0.4350 | 0.5650 | 0.3955 | 98.96% | 5.35 |
| B3 SAM Automatic ROI + Track-Mined MOI | 118/96 | 0.3906 | 0.6094 | 0.4266 | 77.08% | 4.67 |
| B4 Grounding DINO + SAM ROI + Track-Mined MOI | 125/96 | 0.4071 | 0.5929 | 0.4150 | 69.79% | 5.11 |

Diễn giải:

- B1 là mốc tham chiếu tốt nhất vì dùng ROI/MOI đã kiểm chứng.
- B2 chứng minh MOI có thể khai thác từ trajectory, nhưng phân bổ movement chưa bằng MOI chuẩn.
- B3 cho thấy SAM Automatic có thể tham gia baseline định lượng khi kết hợp quality gate và MOI fallback.
- B4 cho thấy Grounding DINO + SAM có thể bootstrap ROI bằng prompt, nhưng vẫn cần hậu xử lý MOI.

---

## 14. Web Demo Django

Thư mục:

```text
dtc_counting/web_demo/
```

Chạy:

```powershell
cd dtc_counting/web_demo
python manage.py migrate
python manage.py runserver
```

Mở:

```text
http://127.0.0.1:8000/
```

### 14.1 Manual mode

Phù hợp để demo ổn định:

- upload video/weights hoặc dùng bộ demo có sẵn,
- vẽ ROI trên canvas,
- vẽ MOI bằng mũi tên,
- hoặc upload file ROI/MOI,
- chạy pipeline,
- xem CSV, video overlay, bảng thống kê.

### 14.2 Auto mode

Phù hợp để trình bày phần tự động hóa:

- upload video,
- chọn SAM Automatic hoặc Grounding DINO + SAM,
- hệ thống bootstrap ROI/MOI,
- quality gate kiểm tra chất lượng,
- fallback khi MOI quá ít hoặc ROI chưa tốt.

### 14.3 Kết quả web

Mỗi lần chạy được lưu tại:

```text
dtc_counting/web_demo/media/<timestamp>/
```

Có thể có:

- `counting_result.csv`
- `counting_vis.mp4`
- `bootstrap_grounded_sam.json`
- `bootstrap_overlay.jpg`
- `run_meta.json`

---

## 15. Cách Cài Đặt Và Chạy

### 15.1 Cài dependencies

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install ultralytics opencv-python numpy scipy django transformers torch torchvision
```

Nếu dùng GPU, cài PyTorch theo đúng bản CUDA từ trang PyTorch.

### 15.2 Chuẩn bị weights

YOLO weights:

```text
weights/best2.pt
```

SAM checkpoint:

```text
dtc_counting/sam_b.pt
```

`sam_b.pt` lớn nên không đưa lên GitHub.

### 15.3 Chạy pipeline chính

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

### 15.4 Chạy so sánh B1-B4

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

---

## 16. Cấu Trúc Thư Mục

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

File quan trọng:

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

## 17. Cách Trình Bày Hệ Thống Cho Người Khác

### 17.1 Mở đầu

"Hệ thống của nhóm giải quyết bài toán đếm xe theo từng hướng di chuyển trong video giao thông. Đây không chỉ là bài toán phát hiện xe, mà là bài toán kết hợp detection, tracking và movement counting."

### 17.2 Nói về dữ liệu

"Nhóm sử dụng video giao thông theo hướng AI City Challenge, trích xuất frame, gán nhãn car/truck bằng Roboflow, export định dạng YOLO và fine-tune detector. Detector sau đó được tích hợp vào pipeline đếm."

### 17.3 Nói về kiến trúc

"Pipeline gồm ba tầng chính: YOLO để phát hiện xe, Kalman-Hungarian tracker để tạo quỹ đạo, và ROI/MOI counter để gán hướng di chuyển."

### 17.4 Nói về ROI/MOI

"ROI định nghĩa vùng cần quan sát. MOI định nghĩa các hướng movement. Nhờ MOI, hệ thống không chỉ biết có bao nhiêu xe mà còn biết xe đi theo hướng nào."

### 17.5 Nói về SAM

"SAM không thay thế detector. SAM được dùng để hỗ trợ khởi tạo ROI/MOI. Khi SAM không sinh đủ MOI, hệ thống dùng trajectory của xe để bổ sung. Đây là cơ chế giúp giảm công cấu hình thủ công nhưng vẫn giữ được khả năng đánh giá định lượng."

### 17.6 Nói về metric

"Detector được đánh giá bằng mAP/Precision/Recall, còn toàn hệ thống được đánh giá bằng nwRMSE, S1 Effectiveness, Count Accuracy và MAE theo movement/class. Điều này quan trọng vì detector tốt chưa chắc counting tốt nếu tracking hoặc MOI assignment sai."

### 17.7 Kết luận khi trình bày

"Kết quả cho thấy baseline dùng ROI/MOI chuẩn vẫn là mốc tốt nhất. Các nhánh tự động bằng SAM/Grounding-SAM giúp giảm công cấu hình và có thể đưa vào đánh giá khi kết hợp quality gate và MOI fallback."

---

## 18. Ghi Chú Về Git

Các file lớn được ignore:

- `dtc_counting/outputs/`
- `dtc_counting/web_demo/media/`
- video `.mp4`, `.avi`, `.mov`, ...
- `dtc_counting/sam_b.pt`
- database local Django

Repo chỉ nên commit code, README, paper, workflow diagram và các file cấu hình/CSV mẫu nhỏ.
