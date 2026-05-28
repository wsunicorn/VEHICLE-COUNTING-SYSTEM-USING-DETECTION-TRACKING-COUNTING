# Vehicle Counting System Using Detection, Tracking, ROI and MOI

Đây là README chính của project đếm phương tiện giao thông theo hướng di chuyển. Nội dung tập trung vào chính hệ thống trong repo: dữ liệu, mô hình, pipeline xử lý, web demo, cách chạy và cách kiểm chứng kết quả đánh giá.

README này giúp người đọc hiểu:

- **What**: hệ thống này là gì?
- **Why**: vì sao phải thiết kế như vậy?
- **How**: từng module hoạt động như thế nào?
- **Data**: dữ liệu được lấy, xử lý và gán nhãn ra sao?
- **Training**: YOLO được train/fine-tune như thế nào?
- **Metrics**: hệ thống được đánh giá bằng công thức nào?
- **Demo**: chạy CLI và web demo như thế nào?
- **Verification**: thầy/cô hoặc người chấm có thể kiểm chứng kết quả đánh giá ở đâu?

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
| Auto ROI + Track-Mined MOI | SAM, Grounding DINO + SAM, trajectory mining | Bootstrap ROI và sinh MOI từ quỹ đạo xe |
| Evaluation | nwRMSE, S1 Effectiveness, MAE, Count Accuracy | Đánh giá kết quả |
| Web Demo | Django | Giao diện demo và chạy pipeline |

Hiểu nhanh các thuật ngữ trong pipeline:

- **Detection** là bước tìm xe trong từng frame. Output là `bounding box + class + confidence`.
- **Tracking** là bước nối các detection qua nhiều frame để biết đâu là cùng một xe.
- **Counting** là bước quyết định track nào được ghi nhận, ghi nhận tại frame nào và thuộc hướng nào.
- **ROI** trả lời câu hỏi "đếm ở vùng nào?".
- **MOI** trả lời câu hỏi "xe đi theo hướng nào?".
- **Track-Mined MOI** là MOI được sinh từ trajectory của xe, sau đó align về ID MOI chính thức bằng matching vector.
- **Metric** là công thức đánh giá kết quả, ví dụ đếm đúng tổng xe, đúng hướng, đúng loại xe và đúng theo thời gian.

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

Các trường này cần được hiểu như sau:

| Trường | Định nghĩa | Vì sao quan trọng? |
|---|---|---|
| `video_clip_id` | ID của video/camera đang xử lý | Giúp gộp hoặc tách kết quả theo từng camera |
| `frame_id` | thứ tự frame tại thời điểm xe được đếm | Dùng để đánh giá sai số theo thời gian |
| `movement_id` | ID hướng di chuyển | Cho biết xe đi thẳng, rẽ trái, rẽ phải hoặc một luồng cụ thể |
| `vehicle_class_id` | ID loại phương tiện | Cho phép đếm riêng car/truck |

Điểm quan trọng: hệ thống không chỉ đếm tổng số xe, mà đếm theo **loại xe** và **hướng di chuyển**. Đây là yêu cầu quan trọng trong bài toán traffic flow analysis và AI City Challenge Track 1.

---

## 3. Why - Vì Sao Cần Detection, Tracking, ROI Và MOI?

Nếu chỉ dùng object detection, ta chỉ biết trong từng frame có bao nhiêu xe. Nhưng một xe xuất hiện trong nhiều frame, nên nếu cộng detection theo frame thì sẽ đếm trùng rất nhiều lần.

**Object detection** là bài toán tìm vị trí và lớp đối tượng trong một ảnh. Với project này, detector trả lời câu hỏi: "frame hiện tại có xe hơi hoặc xe tải nào không, và chúng nằm ở đâu?". Detector không có trí nhớ theo thời gian, nên nó không biết xe ở frame 100 có phải cùng xe ở frame 101 hay không.

Vì vậy cần thêm tracking:

```text
Detection từng frame -> Tracklet của từng xe -> Đếm mỗi xe một lần
```

**Tracklet** là chuỗi vị trí của cùng một xe qua nhiều frame. Nếu tracklet ổn định, hệ thống có thể đếm một xe đúng một lần thay vì đếm lại ở mỗi frame.

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

Trong project có hai nhóm dữ liệu khác nhau:

| Nhóm dữ liệu | Dùng để làm gì? | Ví dụ |
|---|---|---|
| Dữ liệu train detector | Dạy YOLO nhận ra `car` và `truck` | frame ảnh đã gán bounding box |
| Dữ liệu đánh giá counting | Kiểm tra pipeline đếm xe theo hướng | video, ROI, MOI, ground truth CSV |

Điểm cần nhớ: YOLO học từ ảnh đã gán nhãn, còn kết quả cuối của hệ thống được đánh giá trên video counting. Detector tốt là điều kiện cần, nhưng tracking và ROI/MOI mới quyết định xe có được đếm đúng hướng hay không.

### 5.2 Dữ liệu train YOLO

Để train detector, nhóm cần tạo dataset có nhãn bounding box cho các lớp:

- `car`
- `truck`

Các khái niệm ở bước này:

- **Frame** là một ảnh đơn trong video. Video thực chất là chuỗi nhiều frame liên tiếp.
- **Annotation** là nhãn do người gán cho ảnh.
- **Bounding box** là hình chữ nhật bao quanh xe, thường được lưu bằng tọa độ tâm, chiều rộng và chiều cao.
- **Class label** là nhãn loại xe, trong project là `car` hoặc `truck`.

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

Nói đơn giản, Roboflow là công cụ giúp biến frame thô thành dataset có cấu trúc. Người dùng upload ảnh, vẽ box quanh xe, gắn nhãn class, kiểm tra lại nhãn và export sang định dạng mà YOLO có thể train.

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

Một số lỗi gán nhãn ảnh hưởng trực tiếp đến hệ thống:

| Lỗi nhãn | Hậu quả khi chạy pipeline |
|---|---|
| Thiếu box xe | YOLO dễ bỏ sót xe, làm đếm thiếu |
| Box lệch quá nhiều | Tracker nhận vị trí sai, dễ mất track |
| Sai class car/truck | CSV output sai `vehicle_class_id` |
| Nhãn không nhất quán | Model học không ổn định giữa các cảnh |

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

Các file sau khi export thường được chia thành `train` và `val`:

- **Train set** dùng để cập nhật trọng số model.
- **Validation set** dùng để kiểm tra model trong lúc train.
- **Test set**, nếu có, chỉ nên dùng để đánh giá cuối.

Nếu train và validation quá giống nhau, kết quả có thể nhìn tốt nhưng model không tổng quát sang camera khác. Đây gọi là **overfitting**: model nhớ dữ liệu cũ nhiều hơn là học đặc trưng thật của xe.

---

## 6. Huấn Luyện YOLO Detector

### 6.1 Vì sao dùng YOLO?

**YOLO** là họ mô hình object detection một giai đoạn. "Một giai đoạn" nghĩa là model nhìn ảnh một lần và dự đoán trực tiếp các bounding box, class và confidence. Điều này khác với các hướng hai giai đoạn, nơi model phải đề xuất vùng trước rồi mới phân loại.

YOLO phù hợp với bài toán này vì:

- inference nhanh,
- dễ fine-tune,
- có API Ultralytics thuận tiện,
- đủ tốt cho bài toán car/truck trong video giao thông,
- dễ tích hợp vào pipeline Python/OpenCV.

Trong hệ thống này, YOLO không đếm xe trực tiếp. YOLO chỉ tạo danh sách detection cho từng frame; các bước tracking, ROI/MOI và counting mới biến detection đó thành kết quả đếm.

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

**Transfer learning** giúp tiết kiệm dữ liệu và thời gian train. Model pretrained đã biết các đặc trưng thị giác cơ bản như cạnh, bánh xe, thân xe, hình khối; fine-tune chỉ điều chỉnh model cho camera giao thông và hai class của project.

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

Ý nghĩa các tham số train thường gặp:

| Tham số | Ý nghĩa |
|---|---|
| `model` | checkpoint YOLO ban đầu, ví dụ `yolov8m.pt` |
| `data` | file `data.yaml` mô tả train/val path và class |
| `epochs` | số vòng model học qua dataset |
| `imgsz` | kích thước ảnh đưa vào model |
| `batch` | số ảnh xử lý trong một bước cập nhật |
| `device` | CPU/GPU dùng để train |

Nếu máy yếu hơn:

```bash
yolo train model=yolov8n.pt data=data.yaml epochs=80 imgsz=960 batch=8
```

### 6.4 Augmentation

**Augmentation** là tạo biến thể ảnh trong lúc train để model gặp nhiều tình huống hơn. Mục tiêu là làm detector không quá phụ thuộc vào đúng một camera, một ánh sáng hoặc một background.

Các augmentation hữu ích:

| Augmentation | Ý nghĩa |
|---|---|
| Mosaic | Tăng đa dạng scale và background |
| HSV jitter | Mô phỏng thay đổi ánh sáng/thời tiết |
| Horizontal flip | Tăng dữ liệu cho các hướng giao thông đối xứng |
| Scale/translate | Giúp model chịu được thay đổi góc nhìn |

Không nên dùng rotation quá mạnh vì camera giao thông thường cố định, xe không xoay tùy ý như ảnh tự nhiên.

### 6.5 Metric khi train detector

Các metric ở mục này đánh giá riêng detector, chưa phải đánh giá toàn bộ hệ thống counting.

Các metric detector:

| Metric | Công thức/ý nghĩa |
|---|---|
| Precision | `TP / (TP + FP)` |
| Recall | `TP / (TP + FN)` |
| F1 | `2PR / (P + R)` |
| mAP@0.5 | AP trung bình tại IoU threshold 0.5 |
| mAP@0.5:0.95 | AP trung bình qua nhiều threshold IoU |

Trong đó:

- **TP** là detection đúng xe.
- **FP** là model báo có xe nhưng thực tế không có.
- **FN** là xe thật nhưng model bỏ sót.
- **IoU** là độ chồng lắp giữa box dự đoán và box ground truth.
- **mAP** là metric tổng hợp độ chính xác detection trên nhiều ngưỡng.

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

**Detection** trong pipeline là bước đọc một frame và trả về các xe được phát hiện. Mỗi detection gồm:

- bounding box: vị trí xe,
- class: `car` hoặc `truck`,
- confidence: độ tin cậy của model.

YOLO nhận frame `I_t` và trả về:

```text
d_i = (x1, y1, x2, y2, class, confidence)
```

Trong đó:

- `(x1, y1)` là góc trên trái của bbox.
- `(x2, y2)` là góc dưới phải của bbox.
- `class` là nhãn xe.
- `confidence` càng cao thì model càng tin detection đó là thật.

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

Vì truck dễ bị nhầm với các xe lớn hoặc vùng ảnh nhiễu, project thường đặt ngưỡng confidence cho truck cao hơn car.

YOLO/Ultralytics cũng thường xử lý sẵn các bước nền như:

- **letterbox resize**: resize ảnh nhưng giữ tỉ lệ, thêm padding nếu cần để tránh làm méo xe,
- **NMS**: loại bỏ nhiều box trùng nhau trên cùng một xe,
- decode output tensor thành bbox/class/confidence dễ dùng.

OpenCV hỗ trợ phần còn lại: đọc frame từ video, vẽ bbox, crop vùng xe và ghi video overlay.

---

## 8. Module Tracking

Tracker nằm trong class:

```text
MultiStepTracker
```

**Multi-object tracking** là bài toán theo dõi nhiều đối tượng cùng lúc trong video. Input của tracker là các detection ở từng frame; output là các `track_id` ổn định cho từng xe.

Nếu tracking tốt, cùng một xe sẽ giữ cùng ID từ lúc xuất hiện đến lúc rời khỏi vùng quan sát. Nếu tracking kém, hệ thống có thể gặp:

- **ID switch**: cùng một xe bị đổi ID giữa chừng,
- **fragmentation**: một xe bị tách thành nhiều track ngắn,
- **merge**: hai xe gần nhau bị nhập nhầm thành một track.

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

**Kalman Filter** là bộ lọc dự đoán-cập nhật. Nó không chỉ lưu vị trí hiện tại mà còn ước lượng vận tốc và độ không chắc chắn. Nhờ vậy, nếu YOLO bỏ sót xe trong vài frame, tracker vẫn có thể dự đoán xe đang ở đâu.

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

Ý nghĩa các ký hiệu:

| Ký hiệu | Cách hiểu |
|---|---|
| `x` | trạng thái track |
| `P` | covariance, tức độ không chắc chắn của trạng thái |
| `F` | ma trận chuyển trạng thái từ frame trước sang frame sau |
| `Q` | process noise, độ nhiễu của mô hình chuyển động |
| `z` | measurement, tức detection mới |
| `H` | ma trận ánh xạ trạng thái sang không gian đo |
| `R` | measurement noise, độ nhiễu của detection |
| `K` | Kalman gain, quyết định tin dự đoán hay detection nhiều hơn |

### 8.2 Hungarian Matching

Hungarian giải bài toán gán tối ưu:

```text
min sum cost(track_i, detection_j)
```

Ở mỗi frame, có thể có nhiều track cũ và nhiều detection mới. Tracker cần quyết định detection nào thuộc track nào. Hungarian dùng **cost matrix** để tìm cách ghép có tổng chi phí nhỏ nhất.

Ví dụ cost nhỏ nghĩa là khả năng cùng xe cao:

```text
              det1   det2   det3
track1        0.1    0.9    0.7
track2        0.8    0.2    0.6
track3        0.9    0.4    0.1
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

Nói theo luồng hệ thống:

```text
detection mới -> ghép với track cũ -> cập nhật track -> lưu lịch sử tâm xe -> khi track kết thúc thì xét đếm
```

Lịch sử tâm xe chính là trajectory, tức quỹ đạo dùng ở bước ROI/MOI và counting.

---

## 9. Module Counting: ROI/MOI Và Gán Hướng

Đây là phần biến trajectory thành counting event. Detection trả lời "có xe ở đâu?", tracking trả lời "xe đó đi như thế nào?", còn counting trả lời "xe này có được ghi vào kết quả không, và ghi vào movement nào?".

Các khái niệm hình học dùng trong mục này:

- **Point** là một điểm ảnh `(x, y)`.
- **Polygon** là đa giác gồm nhiều point.
- **Trajectory** là chuỗi tâm xe qua thời gian.
- **Displacement** là độ dời từ điểm đầu đến điểm cuối của trajectory.

### 9.1 Điều kiện để đếm một track

Track chỉ được đếm khi:

- chưa từng được đếm,
- không bị đánh dấu illegal,
- có `hits >= 3`,
- từng đi vào ROI,
- quỹ đạo đủ dài,
- độ dịch chuyển đủ lớn.

Các điều kiện này giúp tránh đếm xe đứng yên, track quá ngắn hoặc detection nhiễu.

**ROI - Region of Interest** là vùng cần quan sát. Trong project, ROI thường là một polygon vẽ quanh mặt đường/giao lộ cần đếm. Nếu tâm xe hoặc track chưa từng đi vào ROI, hệ thống không nên ghi event.

Ngoài ROI chính, project còn có thể dùng:

- **eROI - extended ROI**: vùng mở rộng để bắt xe sớm hơn, giảm mất track ở biên.
- **iROI - illegal ROI**: vùng/đường đi không hợp lệ, giúp loại track đi sai vùng hoặc gây nhiễu.

Một xe chỉ nên được đếm một lần, nên mỗi track có trạng thái đã đếm hay chưa. Đây là nguyên tắc **count once**.

### 9.2 Gán MOI bằng vector

**MOI - Movement of Interest** là vector biểu diễn một hướng di chuyển hợp lệ trong camera. Nếu ROI là "đếm ở đâu", thì MOI là "đếm theo hướng nào".

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

Cách hiểu công thức:

- phần `angle` ưu tiên MOI có hướng giống trajectory,
- phần `dist` ưu tiên MOI có vị trí đầu/cuối gần trajectory,
- `angle_weight` và `distance_weight` quyết định hướng hay vị trí quan trọng hơn.

Nhờ vậy, nếu hai movement có hướng gần giống nhau nhưng nằm ở hai lane khác nhau, thành phần khoảng cách vẫn giúp phân biệt.

### 9.3 Nếu không có MOI

Nếu không có file MOI, hệ thống fallback bằng góc quỹ đạo:

```text
angle = atan2(E_y - S_y, E_x - S_x)
movement_id = bucket(angle, movement_count)
```

Cách này kém chính xác hơn vì không biết hình học thực tế của giao lộ.

---

## 10. Bootstrap ROI Và Sinh MOI Tự Động

ROI/MOI có thể:

- vẽ thủ công trên web,
- đọc từ file `.txt`,
- khởi tạo ROI tự động bằng SAM/Grounding-SAM,
- sinh MOI từ trajectory của xe rồi align về ID MOI chính thức nếu có reference.

**Bootstrap** nghĩa là khởi tạo cấu hình ban đầu một cách tự động hoặc bán tự động. Trong project này, SAM/Grounding-SAM chủ yếu dùng để tìm ROI. MOI dùng cho B2/B3/B4 và web auto path được sinh từ trajectory bằng `build_moi_from_tracks.py`, sau đó align về ID chính thức bằng `moi_utils.align_to_reference()` khi có file MOI reference.

**Segmentation** là bài toán phân đoạn ảnh theo vùng pixel. Khác với detection chỉ trả về bbox, segmentation trả về **mask**, tức vùng pixel thuộc đối tượng hoặc vùng quan tâm.

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

**SAM - Segment Anything Model** là model segmentation tổng quát. Ở chế độ automatic, SAM tự sinh nhiều mask trên ảnh mà không cần prompt. Ưu điểm là dễ chạy, nhưng nhược điểm là nó nhìn ảnh tĩnh nên có thể lấy nhầm cỏ, vỉa hè, toàn khung hình hoặc vùng không phải mặt đường.

Vì vậy phần SAM Automatic của project có thêm các bước lọc mask:

- lọc theo diện tích,
- lọc theo vị trí,
- lọc vùng nghi là vegetation/cỏ,
- chuyển mask còn lại thành polygon ROI.

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

**Grounding DINO** là model open-vocabulary detection: thay vì chỉ nhận một tập class cố định, nó có thể nhận prompt ngôn ngữ như `road surface` hoặc `traffic lane`. Output của Grounding DINO là box vùng ảnh có khả năng khớp prompt.

**Prompt** là cụm từ hướng dẫn model tìm gì. Prompt càng sát cảnh giao thông thì khả năng tìm đúng mặt đường càng tốt. Sau khi có box từ Grounding DINO, SAM phân đoạn chi tiết vùng đó thành mask.

### 10.3 Quality gate

Tự động bootstrap có thể sai. Vì vậy project dùng quality gate:

| Kiểm tra | Ý nghĩa |
|---|---|
| ROI gần full-frame | Có thể SAM lấy cả màn hình |
| ROI quá nhỏ/hẹp | Không đủ vùng đếm |
| MOI quá ít | Không đủ hướng movement |
| Mask giống vegetation | Có thể lấy nhầm cỏ/cây |

Nếu quality gate chưa đạt, hệ thống có thể dùng fallback.

**Quality gate** là lớp kiểm tra chất lượng trước khi cho cấu hình tự động đi tiếp vào counting. Nó không làm SAM/Grounding DINO chính xác tuyệt đối, nhưng giúp tránh trường hợp một ROI sai rõ ràng lại được dùng để đánh giá định lượng.

### 10.4 Track-Mined MOI Và Align ID

Script:

```text
dtc_counting/build_moi_from_tracks.py
```

Ý tưởng: dùng chính quỹ đạo xe để suy hướng di chuyển. Đây là cách đang được dùng cho B2/B3/B4 và web auto path để đồng nhất với phần chạy baseline.

Với B3/B4, SAM hoặc Grounding-SAM tạo ROI. Nếu bootstrap sinh MOI quá ít hoặc không ổn định, hệ thống không dùng MOI đó để đánh giá chính; thay vào đó, project chạy YOLO + tracker trên một đoạn video, lấy trajectory thật của xe rồi gom cụm để tạo MOI.

Quy trình:

1. Chạy YOLO + tracker trong một số frame.
2. Lấy track có dịch chuyển đủ lớn.
3. Lấy đoạn quỹ đạo nằm trong ROI.
4. Biến mỗi track thành vector hướng.
5. Gom cụm vector bằng KMeans.
6. Align ID về MOI chuẩn nếu có file reference.

Trong đó:

- **PCA** có thể dùng để tìm trục chính của vùng/mask hoặc cụm điểm.
- **KMeans** gom các vector trajectory thành các nhóm hướng. Mỗi cụm có thể xem như một MOI ứng viên.
- **Align ID** là bước sắp lại ID movement cho gần với MOI/reference, tránh cùng một hướng nhưng bị đặt ID khác.
- File code thực hiện align: `dtc_counting/moi_utils.py`.

Nhờ vậy, B3/B4 có thể đưa vào bảng định lượng:

- B3: SAM Automatic ROI + Track-Mined MOI.
- B4: Grounding DINO + SAM ROI + Track-Mined MOI.

---

## 11. Bốn Baseline B1-B4

Script:

```text
dtc_counting/run_full_comparison.py
```

**Baseline** là cấu hình tham chiếu dùng để so sánh. Thay vì chỉ báo một kết quả cuối, project chạy nhiều cấu hình để biết phần nào đang đóng góp vào chất lượng hệ thống: ROI thủ công, MOI thủ công, MOI tự sinh, SAM Automatic hoặc Grounding DINO + SAM.

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

Ở phần đánh giá có ba khái niệm cần phân biệt:

- **Ground truth** là dữ liệu đúng dùng làm chuẩn so sánh.
- **Prediction** là CSV do hệ thống tạo ra.
- **Cumulative count** là số đếm tích lũy đến một segment/frame nhất định.

Evaluator quy các file về cùng logic `video_id/frame_id/movement_id/vehicle_class_id`, rồi so sánh prediction với ground truth theo tổng xe, theo movement/class và theo thời gian.

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

**RMSE** là căn trung bình bình phương sai số. **Weighted RMSE** thêm trọng số cho từng segment. **Normalized weighted RMSE** chuẩn hóa sai số theo số xe thật để các movement có lượng xe khác nhau vẫn so sánh hợp lý.

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

### 13.1 Cách Kiểm Chứng Kết Quả Đánh Giá

Các file minh chứng đã được đưa lên GitHub tại:

```text
docs/evaluation_evidence/cam5_b1_b4_20260524/
```

Thư mục này chứa các file sinh trực tiếp từ lần chạy `final_cam5_b1_b4_20260524_v4`, gồm:

- `comparison_summary.json` và `comparison_summary.csv`: bảng kết quả tổng hợp B1-B4.
- `b1_eval.json`, `b2_eval.json`, `b3_eval.json`, `b4_eval.json`: metric chi tiết từng baseline.
- `b1_manual_official_moi.csv`, `b2_manual_tracked_moi.csv`, `b3_sam_auto.csv`, `b4_grounded_sam.csv`: CSV prediction dùng để tính metric.
- `b2_moi_from_tracks.txt` và `b2_moi_from_tracks_aligned.txt`: MOI sinh từ trajectory và MOI sau khi align ID.
- `b3_bootstrap_decision.json`, `b4_bootstrap_decision.json`: minh chứng B3/B4 dùng track-mined MOI khi bootstrap sinh quá ít MOI hợp lệ.
- `run_stdout.log`: log console của lần chạy, có progress, số event và metric in ra.

README riêng trong thư mục evidence ghi rõ:

- file code nào sinh ra kết quả,
- lệnh chạy lại B1-B4 bằng `run_full_comparison.py`,
- lệnh đánh giá riêng từng baseline bằng `evaluate_counting.py`,
- file nào được commit và file nào không commit vì quá nặng.

Để kiểm chứng nhanh, mở:

```text
docs/evaluation_evidence/cam5_b1_b4_20260524/README.md
```

---

## 14. Web Demo Django

Thư mục:

```text
dtc_counting/web_demo/
```

**Django** là web framework Python. Trong project này, Django không làm thay thuật toán computer vision; nó là lớp giao diện để người dùng upload dữ liệu, chọn chế độ chạy, theo dõi log và xem output.

Các khái niệm Django xuất hiện trong web demo:

| Khái niệm | Cách hiểu trong project |
|---|---|
| Django project | phần cấu hình chung: settings, URL, media/static |
| Django app | module chức năng `counter` xử lý demo đếm xe |
| View | hàm nhận request, lưu file, gọi pipeline và trả response |
| Form | định nghĩa input người dùng được phép gửi |
| Template | HTML hiển thị giao diện |
| Media folder | nơi lưu upload và output theo từng run |

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
- hệ thống bootstrap ROI bằng SAM/Grounding-SAM,
- hệ thống sinh MOI từ trajectory và align về ID MOI chính thức khi có reference,
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

Web demo chạy pipeline ở background vì counting/SAM có thể mất thời gian. Giao diện dùng cơ chế hỏi trạng thái định kỳ để cập nhật progress/log. Cách này giúp trang không bị đứng khi model đang xử lý video.

---

## 15. Cách Cài Đặt Và Chạy

### 15.1 Cài dependencies

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install ultralytics opencv-python numpy scipy django transformers torch torchvision
```

Ý nghĩa các thư viện chính:

| Thư viện | Vai trò trong project |
|---|---|
| `ultralytics` | load và chạy YOLO detector |
| `opencv-python` | đọc/ghi video, xử lý frame, vẽ overlay |
| `numpy` | tính toán vector, bbox, mask, ma trận |
| `scipy` | hỗ trợ Hungarian matching qua `linear_sum_assignment` |
| `torch`, `torchvision` | nền tảng deep learning cho YOLO/SAM/Grounding DINO |
| `transformers` | hỗ trợ model Grounding DINO/processor liên quan |
| `django` | chạy web demo |

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
|   +-- evaluation_evidence/
|       +-- cam5_b1_b4_20260524/
+-- weights/
|   +-- best.pt
|   +-- best2.pt
|   +-- best4.pt
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
| `docs/evaluation_evidence/cam5_b1_b4_20260524/` | File minh chứng để kiểm chứng kết quả đánh giá |

Cách các script liên kết với nhau:

```text
run_dtc_counting.py
  -> chạy pipeline chính: video -> YOLO -> tracker -> ROI/MOI -> CSV/overlay

evaluate_counting.py
  -> so sánh CSV prediction với ground truth và tính metric

build_moi_from_tracks.py
  -> dùng trajectory thật để tạo MOI khi thiếu MOI thủ công

sam_auto_bootstrap.py / grounded_sam_bootstrap.py
  -> tạo ROI/MOI tự động hoặc bán tự động

run_full_comparison.py
  -> gọi các script trên để chạy B1-B4 và gom bảng kết quả

web_demo/counter/views.py
  -> nhận input từ web rồi gọi pipeline ở background
```

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

Repo hiện commit code, README, paper, workflow diagram, weights YOLO nhỏ cần cho demo, và các file evidence nhỏ trong `docs/evaluation_evidence/`. Các video/output/media lớn vẫn nên để ngoài Git.
